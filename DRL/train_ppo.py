import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from env.environment import Grid2AIEnvConfig, make_env


class ScenarioSwitchingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        grid_yaml: str,
        cfg_kwargs: Dict[str, Any],
        scenario_specs: List[Dict[str, Any]],
        seed: int = 0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not scenario_specs:
            raise ValueError("scenario_specs must not be empty")
        self._grid_yaml = str(grid_yaml)
        self._cfg_kwargs = dict(cfg_kwargs)
        self._scenario_specs = [dict(spec) for spec in scenario_specs]
        self._rng = np.random.default_rng(seed)
        self._render_mode = render_mode
        self._current_env = self._build_inner_env(self._scenario_specs[0], seed=seed)
        self._current_spec = dict(self._scenario_specs[0])
        self.action_space = self._current_env.action_space
        self.observation_space = self._current_env.observation_space
        self.metadata = getattr(self._current_env, "metadata", self.metadata)

    def _build_inner_env(self, spec: Dict[str, Any], *, seed: int):
        cfg = Grid2AIEnvConfig(
            **self._cfg_kwargs,
            queue_yaml_path=str(spec["event_yaml"]),
            ev_yaml_path=str(spec["ev_yaml"]),
            baseline_exogenous_path=str(spec["baseline_exogenous_path"]),
        )
        return make_env(
            grid_yaml=self._grid_yaml,
            cfg=cfg,
            seed=seed,
            render_mode=self._render_mode,
        )

    def _sample_scenario_spec(self) -> Dict[str, Any]:
        if len(self._scenario_specs) == 1:
            return dict(self._scenario_specs[0])
        idx = int(self._rng.integers(0, len(self._scenario_specs)))
        return dict(self._scenario_specs[idx])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        spec = self._sample_scenario_spec()
        if self._current_env is not None:
            self._current_env.close()
        child_seed = int(self._rng.integers(0, 2**31 - 1))
        self._current_env = self._build_inner_env(spec, seed=child_seed)
        self._current_spec = dict(spec)
        obs, info = self._current_env.reset(seed=seed, options=options)
        info = dict(info or {})
        info["scenario_base"] = str(spec["scenario_base"])
        info["scenario"] = str(spec["scenario"])
        info["lookahead_hours"] = int(spec["lookahead_hours"])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._current_env.step(action)
        info = dict(info or {})
        info["scenario_base"] = str(self._current_spec["scenario_base"])
        info["scenario"] = str(self._current_spec["scenario"])
        info["lookahead_hours"] = int(self._current_spec["lookahead_hours"])
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._current_env is None:
            return None
        return self._current_env.render()

    def close(self):
        if self._current_env is not None:
            self._current_env.close()
            self._current_env = None


class RewardHistoryEvalCallback(EvalCallback):
    def __init__(self, *args, history_csv_path: Path, **kwargs) -> None:
        self.history_csv_path = Path(history_csv_path)
        self.history_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._eval_index = 0
        super().__init__(*args, **kwargs)

    def _append_history_row(self) -> None:
        ep_buffer = list(getattr(self.model, "ep_info_buffer", []) or [])
        train_rewards = [float(ep_info["r"]) for ep_info in ep_buffer if "r" in ep_info]
        train_lengths = [float(ep_info["l"]) for ep_info in ep_buffer if "l" in ep_info]
        eval_rewards = []
        eval_lengths = []
        if getattr(self, "evaluations_results", None):
            eval_rewards = [float(x) for x in self.evaluations_results[-1]]
        if getattr(self, "evaluations_length", None):
            eval_lengths = [float(x) for x in self.evaluations_length[-1]]
        row = {
            "epoch": int(self._eval_index),
            "timesteps": int(self.num_timesteps),
            "train_reward_mean": float(safe_mean(train_rewards)) if train_rewards else float("nan"),
            "train_reward_std": float(np.std(train_rewards)) if train_rewards else float("nan"),
            "train_ep_len_mean": float(safe_mean(train_lengths)) if train_lengths else float("nan"),
            "eval_reward_mean": float(self.last_mean_reward),
            "eval_reward_std": float(np.std(eval_rewards)) if eval_rewards else float("nan"),
            "eval_ep_len_mean": float(np.mean(eval_lengths)) if eval_lengths else float("nan"),
            "best_eval_reward": float(self.best_mean_reward),
        }
        write_header = not self.history_csv_path.exists()
        with self.history_csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._eval_index += 1
            self._append_history_row()
        return result


def parse_args(default_algo: Optional[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SB3 algorithms on Grid2AIEnv.")
    default_start_method = "fork" if os.name == "posix" else "spawn"
    parser.add_argument("--algo", choices=["ppo", "sac", "td3", "ddpg"], default=default_algo or "ppo")
    parser.add_argument("--grid-yaml", default="config/IEEE33.yaml")
    parser.add_argument("--time-yaml", default="config/time.yaml")
    parser.add_argument("--event-yaml", default="config/event.yaml")
    parser.add_argument("--ev-yaml", default="config/ev.yaml")
    parser.add_argument("--baseline-exogenous-path", default="results/baseline_1w_exogenous.yaml")
    parser.add_argument("--no-baseline-exogenous", action="store_true")
    parser.add_argument("--total-timesteps", type=int, default=4_000_000)
    parser.add_argument("--horizon-hours", type=int, default=168)
    parser.add_argument("--resolution-minutes", type=int, default=5)
    parser.add_argument("--ev-resolution-seconds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-rate-schedule", choices=["constant", "cosine"], default="constant")
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--action-noise-sigma", type=float, default=0.10)
    parser.add_argument("--sac-ent-coef", default="auto")
    parser.add_argument("--policy-net", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--vec-env", choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--start-method", choices=["fork", "forkserver", "spawn"], default=default_start_method)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--log-dir", default="results/ppo")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--scenario-batch-root", default=None)
    parser.add_argument("--train-scenarios", default=None, help="Comma-separated scenario bases, e.g. S1,S2,S3")
    parser.add_argument("--eval-scenario", default=None, help="Scenario base for evaluation, e.g. S0")
    parser.add_argument("--scenario-lookahead-hours", type=int, default=None, help="Pick scenario variant by lookahead hours")
    parser.add_argument("--scenario-variant-index", type=int, default=None, help="Pick scenario variant by suffix index")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def _cfg_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "horizon_hours": int(args.horizon_hours),
        "resolution_minutes": int(args.resolution_minutes),
        "ev_resolution_seconds": int(args.ev_resolution_seconds),
        "time_yaml_path": str(args.time_yaml),
        "use_baseline_exogenous": not bool(args.no_baseline_exogenous),
    }


def _parse_csv_items(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _resolve_scenario_variant_dir(
    batch_root: Path,
    scenario_base: str,
    *,
    lookahead_hours: Optional[int],
    variant_index: Optional[int],
) -> Path:
    scenario_dir = (batch_root / scenario_base).resolve()
    if not scenario_dir.exists():
        raise FileNotFoundError(f"scenario directory not found: {scenario_dir}")
    if variant_index is not None and lookahead_hours is not None:
        raise ValueError("use either --scenario-lookahead-hours or --scenario-variant-index, not both")
    if variant_index is not None:
        variant_dir = scenario_dir / f"{scenario_base}_{int(variant_index)}"
        if not variant_dir.exists():
            raise FileNotFoundError(f"scenario variant not found: {variant_dir}")
        return variant_dir
    target_lookahead = int(lookahead_hours) if lookahead_hours is not None else 24
    for variant_dir in sorted(scenario_dir.glob(f"{scenario_base}_*")):
        meta_path = variant_dir / "variant_meta.yaml"
        if not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        if int(meta.get("lookahead_hours", -1)) == target_lookahead:
            return variant_dir.resolve()
    raise FileNotFoundError(
        f"no scenario variant matched lookahead={target_lookahead} under {scenario_dir}"
    )


def _build_scenario_spec(
    batch_root: Path,
    scenario_base: str,
    *,
    lookahead_hours: Optional[int],
    variant_index: Optional[int],
) -> Dict[str, Any]:
    scenario_dir = (batch_root / scenario_base).resolve()
    variant_dir = _resolve_scenario_variant_dir(
        batch_root,
        scenario_base,
        lookahead_hours=lookahead_hours,
        variant_index=variant_index,
    )
    variant_meta = yaml.safe_load((variant_dir / "variant_meta.yaml").read_text(encoding="utf-8")) or {}
    event_yaml = scenario_dir / "event.yaml"
    ev_yaml = scenario_dir / "ev.yaml"
    exogenous_yaml = variant_dir / "exogenous.yaml"
    if not event_yaml.exists():
        raise FileNotFoundError(f"missing scenario event yaml: {event_yaml}")
    if not ev_yaml.exists():
        raise FileNotFoundError(f"missing scenario ev yaml: {ev_yaml}")
    if not exogenous_yaml.exists():
        raise FileNotFoundError(f"missing scenario exogenous yaml: {exogenous_yaml}")
    return {
        "scenario_base": str(scenario_base),
        "scenario": str(variant_dir.name),
        "scenario_dir": str(scenario_dir),
        "variant_dir": str(variant_dir),
        "lookahead_hours": int(variant_meta.get("lookahead_hours", lookahead_hours or 24)),
        "event_yaml": str(event_yaml),
        "ev_yaml": str(ev_yaml),
        "baseline_exogenous_path": str(exogenous_yaml),
        "summary_yaml": str(variant_dir / "summary.yaml"),
    }


def _resolve_multi_scenario_specs(args: argparse.Namespace) -> Dict[str, List[Dict[str, Any]]]:
    if not args.scenario_batch_root:
        return {"train": [], "eval": []}
    batch_root = (ROOT / str(args.scenario_batch_root)).resolve()
    train_names = _parse_csv_items(args.train_scenarios)
    eval_names = _parse_csv_items(args.eval_scenario)
    train_specs = [
        _build_scenario_spec(
            batch_root,
            name,
            lookahead_hours=args.scenario_lookahead_hours,
            variant_index=args.scenario_variant_index,
        )
        for name in train_names
    ]
    eval_specs = [
        _build_scenario_spec(
            batch_root,
            name,
            lookahead_hours=args.scenario_lookahead_hours,
            variant_index=args.scenario_variant_index,
        )
        for name in eval_names
    ]
    return {"train": train_specs, "eval": eval_specs}


def build_env(
    args: argparse.Namespace,
    seed: int,
    *,
    scenario_specs: Optional[List[Dict[str, Any]]] = None,
):
    cfg_kwargs = _cfg_kwargs_from_args(args)
    if scenario_specs:
        return ScenarioSwitchingEnv(
            grid_yaml=str(args.grid_yaml),
            cfg_kwargs=cfg_kwargs,
            scenario_specs=scenario_specs,
            seed=seed,
        )
    cfg = Grid2AIEnvConfig(
        **cfg_kwargs,
        queue_yaml_path=str(args.event_yaml),
        ev_yaml_path=str(args.ev_yaml),
        baseline_exogenous_path=str(args.baseline_exogenous_path),
    )
    return make_env(grid_yaml=str(args.grid_yaml), cfg=cfg, seed=seed)


def make_env_fn(args: argparse.Namespace, seed: int, scenario_specs: Optional[List[Dict[str, Any]]] = None):
    return partial(build_env, args, seed, scenario_specs=scenario_specs)


def build_vec_env(
    args: argparse.Namespace,
    num_envs: int,
    base_seed: int,
    monitor_path: Optional[Path],
    *,
    scenario_specs: Optional[List[Dict[str, Any]]] = None,
):
    env_fns = [make_env_fn(args, base_seed + i, scenario_specs=scenario_specs) for i in range(int(num_envs))]
    if str(args.vec_env) == "subproc" and int(num_envs) > 1:
        vec_env = SubprocVecEnv(env_fns, start_method=str(args.start_method))
    else:
        vec_env = DummyVecEnv(env_fns)
    return VecMonitor(vec_env, filename=str(monitor_path) if monitor_path is not None else None)


def callback_freq(target_steps: int, num_envs: int) -> int:
    return max(1, int(target_steps) // max(1, int(num_envs)))


def build_learning_rate(args: argparse.Namespace):
    initial_lr = float(args.learning_rate)
    if str(args.learning_rate_schedule) == "constant":
        return initial_lr

    def cosine_schedule(progress_remaining: float) -> float:
        elapsed = 1.0 - float(progress_remaining)
        return 0.5 * initial_lr * (1.0 + math.cos(math.pi * elapsed))

    return cosine_schedule


def is_offpolicy_algo(algo: str) -> bool:
    return str(algo).lower() in {"sac", "td3", "ddpg"}


def build_policy_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    algo = str(args.algo).lower()
    if algo == "ppo":
        return {"net_arch": list(args.policy_net)}
    return {"net_arch": {"pi": list(args.policy_net), "qf": list(args.policy_net)}}


def build_model(args: argparse.Namespace, train_env, tb_dir: Path):
    algo = str(args.algo).lower()
    common = dict(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=build_learning_rate(args),
        gamma=float(args.gamma),
        batch_size=int(args.batch_size),
        policy_kwargs=build_policy_kwargs(args),
        tensorboard_log=str(tb_dir),
        verbose=1,
        seed=int(args.seed),
        device=args.device,
    )
    if algo == "ppo":
        return PPO(
            n_steps=int(args.n_steps),
            n_epochs=int(args.n_epochs),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
            **common,
        )

    action_noise = None
    if algo in {"td3", "ddpg"}:
        n_actions = int(train_env.action_space.shape[0])
        sigma = float(args.action_noise_sigma)
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions, dtype=np.float64),
            sigma=np.full(n_actions, sigma, dtype=np.float64),
        )

    off_common = dict(
        buffer_size=int(args.buffer_size),
        learning_starts=int(args.learning_starts),
        tau=float(args.tau),
        train_freq=(int(args.train_freq), "step"),
        gradient_steps=int(args.gradient_steps),
        **common,
    )
    if algo == "sac":
        return SAC(ent_coef=args.sac_ent_coef, **off_common)
    if algo == "td3":
        return TD3(action_noise=action_noise, **off_common)
    if algo == "ddpg":
        return DDPG(action_noise=action_noise, **off_common)
    raise ValueError(f"unsupported algo: {algo}")


def main(default_algo: Optional[str] = None) -> None:
    args = parse_args(default_algo=default_algo)
    algo = str(args.algo).lower()
    if int(args.num_envs) < 1:
        raise ValueError("--num-envs must be >= 1")
    if int(args.num_eval_envs) < 1:
        raise ValueError("--num-eval-envs must be >= 1")
    if is_offpolicy_algo(algo) and int(args.num_envs) != 1:
        raise ValueError(f"--num-envs must be 1 for {algo.upper()} in this training script")
    if str(args.vec_env) == "subproc":
        try:
            mp.set_start_method(str(args.start_method), force=True)
        except RuntimeError:
            pass
    torch.set_num_threads(max(1, int(args.torch_num_threads)))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{algo}_weekly_{ts}"
    run_dir = (ROOT / args.log_dir / run_name).resolve()
    ckpt_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best_model"
    tb_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    scenario_specs = _resolve_multi_scenario_specs(args)
    train_specs = list(scenario_specs["train"])
    eval_specs = list(scenario_specs["eval"])
    if train_specs and not eval_specs:
        raise ValueError("multi-scenario training requires --eval-scenario")
    if eval_specs and len(eval_specs) != 1:
        raise ValueError("--eval-scenario must resolve to exactly one scenario")
    if train_specs:
        print(
            "train scenarios="
            + ",".join(str(spec["scenario"]) for spec in train_specs)
            + f" | eval scenario={eval_specs[0]['scenario']}",
            flush=True,
        )

    train_env = build_vec_env(
        args,
        num_envs=int(args.num_envs),
        base_seed=int(args.seed),
        monitor_path=run_dir / "vec_monitor.csv",
        scenario_specs=train_specs or None,
    )
    print("train_env ready", flush=True)

    eval_env = build_vec_env(
        args,
        num_envs=int(args.num_eval_envs),
        base_seed=int(args.seed) + 10_000,
        monitor_path=None,
        scenario_specs=eval_specs or train_specs[:1] or None,
    )
    print("eval_env ready", flush=True)

    model = build_model(args, train_env, tb_dir)
    print(f"{algo} model ready", flush=True)

    meta = {
        "algorithm": algo,
        "run_name": run_name,
        "reward_history_csv": str(run_dir / "reward_history.csv"),
        "grid_yaml": args.grid_yaml,
        "time_yaml": args.time_yaml,
        "event_yaml": args.event_yaml,
        "ev_yaml": args.ev_yaml,
        "baseline_exogenous_path": args.baseline_exogenous_path,
        "use_baseline_exogenous": not bool(args.no_baseline_exogenous),
        "scenario_batch_root": args.scenario_batch_root,
        "train_scenario_specs": train_specs,
        "eval_scenario_spec": eval_specs[0] if eval_specs else None,
        "total_timesteps": int(args.total_timesteps),
        "num_envs": int(args.num_envs),
        "num_eval_envs": int(args.num_eval_envs),
        "action_dim": int(train_env.action_space.shape[0]),
        "obs_dim": int(train_env.observation_space.shape[0]),
        "seed": int(args.seed),
        "sb3": {
            "learning_rate": float(args.learning_rate),
            "learning_rate_schedule": str(args.learning_rate_schedule),
            "batch_size": int(args.batch_size),
            "gamma": float(args.gamma),
            "policy_net": list(args.policy_net),
            "device": args.device,
            "vec_env": str(args.vec_env),
            "start_method": str(args.start_method),
            "torch_num_threads": int(args.torch_num_threads),
        },
        "algo_params": {
            "n_steps": int(args.n_steps),
            "n_epochs": int(args.n_epochs),
            "gae_lambda": float(args.gae_lambda),
            "clip_range": float(args.clip_range),
            "ent_coef": float(args.ent_coef),
            "vf_coef": float(args.vf_coef),
            "max_grad_norm": float(args.max_grad_norm),
            "buffer_size": int(args.buffer_size),
            "learning_starts": int(args.learning_starts),
            "tau": float(args.tau),
            "train_freq": int(args.train_freq),
            "gradient_steps": int(args.gradient_steps),
            "action_noise_sigma": float(args.action_noise_sigma),
            "sac_ent_coef": args.sac_ent_coef,
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

    reward_history_csv = run_dir / "reward_history.csv"
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=callback_freq(int(args.checkpoint_freq), int(args.num_envs)),
                save_path=str(ckpt_dir),
                name_prefix=f"{algo}_model",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
            RewardHistoryEvalCallback(
                eval_env=eval_env,
                best_model_save_path=str(best_dir),
                log_path=str(run_dir / "eval"),
                eval_freq=callback_freq(int(args.eval_freq), int(args.num_envs)),
                n_eval_episodes=max(1, int(args.eval_episodes)),
                deterministic=True,
                render=False,
                history_csv_path=reward_history_csv,
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
    main(default_algo="ppo")
