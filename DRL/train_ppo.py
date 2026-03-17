import argparse
import json
import os
import sys
import tempfile
import types
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

if "components" not in sys.modules:
    mod = types.ModuleType("components")

    class _Dummy:
        @staticmethod
        def load_from_yaml(path):
            return []

        @staticmethod
        def to_grid(items):
            return []

    mod.NOP = _Dummy
    mod.SOP = _Dummy
    sys.modules["components"] = mod

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from env.environment import Grid2AIEnvConfig, make_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on Grid2AIEnv.")
    parser.add_argument("--grid-yaml", default="config/IEEE33.yaml")
    parser.add_argument("--time-yaml", default="config/time.yaml")
    parser.add_argument("--event-yaml", default="config/event.yaml")
    parser.add_argument("--ev-yaml", default="config/ev.yaml")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000)
    parser.add_argument("--horizon-hours", type=int, default=168)
    parser.add_argument("--resolution-minutes", type=int, default=5)
    parser.add_argument("--ev-resolution-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--policy-net", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--log-dir", default="results/ppo")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def build_env(args: argparse.Namespace, seed: int, run_dir: Path):
    cfg = Grid2AIEnvConfig(
        horizon_hours=args.horizon_hours,
        resolution_minutes=args.resolution_minutes,
        ev_resolution_seconds=args.ev_resolution_seconds,
        time_yaml_path=args.time_yaml,
        queue_yaml_path=args.event_yaml,
        ev_yaml_path=args.ev_yaml,
    )
    return make_env(
        grid_yaml=args.grid_yaml,
        cfg=cfg,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"ppo_weekly_{ts}"
    run_dir = (ROOT / args.log_dir / run_name).resolve()
    ckpt_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_model"
    tb_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([lambda: build_env(args, args.seed, run_dir)])
    train_env = VecMonitor(train_env, filename=str(run_dir / "vec_monitor.csv"))
    print("train_env ready", flush=True)

    eval_env = DummyVecEnv([lambda: build_env(args, args.seed + 10_000, run_dir)])
    eval_env = VecMonitor(eval_env)
    print("eval_env ready", flush=True)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs={"net_arch": list(args.policy_net)},
        tensorboard_log=str(tb_dir),
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    print("ppo model ready", flush=True)

    meta = {
        "run_name": run_name,
        "grid_yaml": args.grid_yaml,
        "time_yaml": args.time_yaml,
        "event_yaml": args.event_yaml,
        "ev_yaml": args.ev_yaml,
        "total_timesteps": int(args.total_timesteps),
        "action_dim": int(train_env.action_space.shape[0]),
        "obs_dim": int(train_env.observation_space.shape[0]),
        "seed": int(args.seed),
        "ppo": {
            "learning_rate": float(args.learning_rate),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip_range": float(args.clip_range),
            "ent_coef": float(args.ent_coef),
            "vf_coef": float(args.vf_coef),
            "max_grad_norm": float(args.max_grad_norm),
            "policy_net": list(args.policy_net),
            "device": args.device,
        },
        "env": {
            "horizon_hours": int(args.horizon_hours),
            "resolution_minutes": int(args.resolution_minutes),
            "ev_resolution_seconds": int(args.ev_resolution_seconds),
        },
    }
    (run_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"run_dir={run_dir}")
    print(f"action_dim={meta['action_dim']}")
    print(f"obs_dim={meta['obs_dim']}")
    print("start learn", flush=True)

    if args.check_only:
        train_env.close()
        eval_env.close()
        return

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=max(1, int(args.checkpoint_freq)),
                save_path=str(ckpt_dir),
                name_prefix="ppo_model",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
            EvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(run_dir / "eval"),
                eval_freq=max(1, int(args.eval_freq)),
                n_eval_episodes=max(1, int(args.eval_episodes)),
                deterministic=True,
                render=False,
            ),
        ]
    )

    model.learn(
        total_timesteps=int(args.total_timesteps),
        callback=callbacks,
        progress_bar=bool(args.progress_bar),
        tb_log_name=run_name,
        log_interval=1,
    )
    model.save(str(run_dir / "final_model"))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
