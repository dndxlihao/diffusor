import argparse
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))

from stable_baselines3 import DDPG, PPO, SAC, TD3

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.grid import Grid
from env.environment import Grid2AIEnvConfig, make_env
from utils.plot_style import use_times_new_roman
from utils.plot_weekly_opf import (
    _dump_result_yaml,
    _build_v2g_compare_series,
    _build_vehicle_soc_series,
    _lineplot,
    _plot_departure_soc_and_v2g,
    _plot_storage_weekly,
    _plot_v2g_compare,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy on Grid2AIEnv.")
    parser.add_argument("--run-dir", required=True, help="PPO run directory containing train_meta.json and best_model/")
    parser.add_argument("--model-path", default=None, help="Optional explicit model path. Defaults to run_dir/best_model/best_model.zip")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to run_dir/best_eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy instead of deterministic")
    return parser.parse_args()


def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "train_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing train meta: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _build_env_from_meta(meta: dict):
    eval_spec = meta.get("eval_scenario_spec") or {}
    queue_yaml = str(eval_spec.get("event_yaml", meta["event_yaml"]))
    ev_yaml = str(eval_spec.get("ev_yaml", meta["ev_yaml"]))
    baseline_exogenous_path = str(eval_spec.get("baseline_exogenous_path", meta["baseline_exogenous_path"]))
    cfg = Grid2AIEnvConfig(
        horizon_hours=int(meta["env"]["horizon_hours"]),
        resolution_minutes=int(meta["env"]["resolution_minutes"]),
        ev_resolution_seconds=int(meta["env"]["ev_resolution_seconds"]),
        time_yaml_path=str(meta["time_yaml"]),
        queue_yaml_path=queue_yaml,
        ev_yaml_path=ev_yaml,
        baseline_exogenous_path=baseline_exogenous_path,
        use_baseline_exogenous=bool(meta["use_baseline_exogenous"]),
    )
    env = make_env(
        grid_yaml=str(meta["grid_yaml"]),
        cfg=cfg,
        seed=int(meta.get("seed", 42)),
    )
    env.enable_trace()
    return env


def _build_replay(env, trace: dict):
    replay = types.SimpleNamespace()
    replay.T = list(range(int(env.T)))
    replay.TSOC = list(range(int(env.T) + 1))
    replay.KEV = list(range(int(env.K_ev)))
    replay.KEV_SOC = list(range(int(env.K_ev) + 1))
    replay.BUS = list(int(x) for x in env.bus_ids)
    replay.GEN = list(int(x) for x in env.gen_ids)
    replay.REN = list(int(x) for x in env.ren_ids)
    replay.STO = list(int(x) for x in env.sto_ids)
    replay.STN = list(int(x) for x in env.station_ids)
    replay.price = {tt: float(env.price[tt]) for tt in replay.T}
    replay.ev_price = {k: float(env.price_ev[k]) for k in replay.KEV}
    replay.Pg = dict(trace["Pg"])
    replay.Qg = dict(trace["Qg"])
    replay.Pr = dict(trace["Pr"])
    replay.curt = dict(trace["curt"])
    replay.Vu = dict(trace["Vu"])
    replay.Vl = dict(trace["Vl"])
    replay.P_buy = dict(trace["P_buy"])
    replay.P_sell = dict(trace["P_sell"])
    replay.P_ch = dict(trace["P_ch"])
    replay.P_dis = dict(trace["P_dis"])
    replay.soc = dict(trace["soc"])
    replay.st_pch = dict(trace["st_pch"])
    replay.st_pdis = dict(trace["st_pdis"])
    replay._dispatch = {
        "event_pch_kw": dict(trace["event_pch_kw"]),
        "event_pdis_kw": dict(trace["event_pdis_kw"]),
        "event_soc_init": dict(trace["event_soc_init"]),
        "final_soc": dict(trace["final_soc"]),
        "unmet_departure_kwh": dict(trace["unmet_departure_kwh"]),
    }
    replay._meta = {
        "baseMVA": float(env.baseMVA),
        "delta_t": float(env.delta_t),
        "T": int(env.T),
        "bus_ids": list(int(x) for x in env.bus_ids),
        "gen_ids": list(int(x) for x in env.gen_ids),
        "ren_ids": list(int(x) for x in env.ren_ids),
        "sto_ids": list(int(x) for x in env.sto_ids),
        "station_ids": list(int(x) for x in env.station_ids),
        "K_ev": int(env.K_ev),
        "ev_step_s": int(env.ev_step_s),
        "station_bus": {int(sid): int(env.station_bus[int(sid)]) for sid in env.station_ids},
        "events_cfg": [dict(ev) for ev in env.events_cfg],
        "rolling": True,
        "v_penalty": float(env.cfg.v_penalty),
        "ev_depart_penalty": float(env.cfg.ev_depart_penalty),
        "v2g_reward_coeff": float(env.cfg.v2g_reward_coeff),
        "ev_target_soc": float(env.cfg.ev_target_soc),
        "storage_terminal_value_coeff": float(env.cfg.storage_terminal_value_coeff),
        "storage_terminal_price": float(env.price[int(env.T - 1)]) if int(env.T) > 0 else 0.0,
        "time_yaml_path": str(env.cfg.time_yaml_path),
        "queue_yaml_path": str(env.cfg.queue_yaml_path),
    }
    return replay


def _build_station_netload_from_trace(env, trace: dict):
    time_d = np.arange(int(env.K_ev), dtype=float) * (float(env.ev_step_s) / 3600.0) / 24.0
    charge = {int(sid): np.zeros(int(env.K_ev), dtype=float) for sid in env.station_ids}
    discharge = {int(sid): np.zeros(int(env.K_ev), dtype=float) for sid in env.station_ids}
    for (eid, k), pch_kw in trace["event_pch_kw"].items():
        sid = int(env.event_by_id[int(eid)]["station_i"])
        charge[sid][int(k)] += float(pch_kw) / 1000.0
    for (eid, k), pdis_kw in trace["event_pdis_kw"].items():
        sid = int(env.event_by_id[int(eid)]["station_i"])
        discharge[sid][int(k)] += float(pdis_kw) / 1000.0
    series = {f"Station {int(sid)}": charge[int(sid)] - discharge[int(sid)] for sid in env.station_ids}
    return time_d, series


def _build_v2g_station_detail_from_trace(replay, grid: Grid):
    residential_ids = [
        int(s.station_i)
        for s in grid.stations
        if int(getattr(s, "status", 1)) == 1 and str(getattr(s, "type", "")).upper() == "RESIDENTIAL"
    ]
    k_count = int(replay._meta["K_ev"])
    charge = {sid: np.zeros(k_count, dtype=float) for sid in residential_ids}
    discharge = {sid: np.zeros(k_count, dtype=float) for sid in residential_ids}
    events_cfg = {int(ev["event_i"]): dict(ev) for ev in replay._meta["events_cfg"]}
    for (eid, k), pch_kw in replay._dispatch["event_pch_kw"].items():
        sid = int(events_cfg[int(eid)]["station_i"])
        if sid in charge:
            charge[sid][int(k)] += float(pch_kw) / 1000.0
    for (eid, k), pdis_kw in replay._dispatch["event_pdis_kw"].items():
        sid = int(events_cfg[int(eid)]["station_i"])
        if sid in discharge:
            discharge[sid][int(k)] += float(pdis_kw) / 1000.0
    best_sid = residential_ids[0] if residential_ids else int(replay.STN[0])
    best_discharge = -1.0
    for sid in residential_ids:
        total_dis = float(np.sum(discharge[sid]))
        if total_dis > best_discharge:
            best_discharge = total_dis
            best_sid = sid
    x_d = np.arange(k_count, dtype=float) * (float(replay._meta["ev_step_s"]) / 3600.0) / 24.0
    series = {
        "Charge": charge[int(best_sid)],
        "Discharge": discharge[int(best_sid)],
        "Net load": charge[int(best_sid)] - discharge[int(best_sid)],
    }
    return x_d, series, int(best_sid)


def _plot_cost(summary: dict, out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "Gen",
        "Grid",
        "Curt",
        "Volt",
        "Branch",
        "Grid cap",
        "-Storage terminal",
        "EV depart",
        "-V2G",
    ]
    values = [
        float(summary["gen_cost"]),
        float(summary["grid_cost"]),
        float(summary["curt_cost"]),
        float(summary["volt_pen"]),
        float(summary["branch_pen"]),
        float(summary["grid_pen"]),
        -float(summary["storage_terminal_value"]),
        float(summary["ev_depart_penalty"]),
        -float(summary["v2g_reward"]),
    ]
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#9d755d", "#bab0ab", "#d62728", "#2a9d8f"]
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    fig.patch.set_facecolor("#faf8f4")
    ax.set_facecolor("#fffdf8")
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.axhline(0.0, color="#666666", lw=0.9)
    for bar, val in zip(bars, values):
        y = val + (0.01 * max(1.0, max(abs(v) for v in values)))
        va = "bottom" if val >= 0.0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2.0, y, f"{val:.0f}", ha="center", va=va, fontsize=10)
    ax.set_title("RL Policy Cost Decomposition")
    ax.set_ylabel("Cost")
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_system(env, replay, summary: dict, out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    time_d = np.arange(int(env.T), dtype=float) * float(env.delta_t) / 24.0
    ev_time_d = np.arange(int(env.K_ev), dtype=float) * float(env.ev_step_s) / 3600.0 / 24.0

    gen_sum = np.asarray(
        [sum(float(replay.Pg.get((int(gid), tt), 0.0)) for gid in env.gen_ids) * float(env.baseMVA) for tt in range(env.T)],
        dtype=float,
    )
    pav = np.sum(env.Pav_pu, axis=1) * float(env.baseMVA)
    pr = np.asarray(
        [sum(float(replay.Pr.get((int(rid), tt), 0.0)) for rid in env.ren_ids) * float(env.baseMVA) for tt in range(env.T)],
        dtype=float,
    )
    load = np.sum(env.Pd_pu, axis=1) * float(env.baseMVA)
    p_buy = np.asarray([float(replay.P_buy.get(tt, 0.0)) * float(env.baseMVA) for tt in range(env.T)], dtype=float)
    p_sell = np.asarray([float(replay.P_sell.get(tt, 0.0)) * float(env.baseMVA) for tt in range(env.T)], dtype=float)
    sto_net = np.asarray(
        [
            sum(
                (float(replay.P_dis.get((int(sid), tt), 0.0)) - float(replay.P_ch.get((int(sid), tt), 0.0)))
                * float(env.baseMVA)
                for sid in env.sto_ids
            )
            for tt in range(env.T)
        ],
        dtype=float,
    )
    ev_charge = np.zeros(int(env.K_ev), dtype=float)
    ev_discharge = np.zeros(int(env.K_ev), dtype=float)
    for (_, k), pch_kw in replay._dispatch["event_pch_kw"].items():
        ev_charge[int(k)] += float(pch_kw) / 1000.0
    for (_, k), pdis_kw in replay._dispatch["event_pdis_kw"].items():
        ev_discharge[int(k)] += float(pdis_kw) / 1000.0

    fig, axes = plt.subplots(3, 1, figsize=(15.5, 9.0), sharex=False)
    fig.patch.set_facecolor("#faf8f4")

    ax = axes[0]
    ax.set_facecolor("#fffdf8")
    ax.plot(time_d, gen_sum, color="#4c78a8", lw=1.8, label="Dispatchable generation")
    ax.plot(time_d, pav, color="#2b83ba", lw=1.4, alpha=0.7, label="Renewable available")
    ax.plot(time_d, pr, color="#54a24b", lw=1.7, label="Renewable dispatched")
    ax.set_title("RL Weekly System Dispatch")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    ax = axes[1]
    ax.set_facecolor("#fffdf8")
    ax.plot(time_d, load, color="#222222", lw=1.8, label="Total load")
    ax.plot(time_d, p_buy, color="#cc5a3d", lw=1.6, label="Grid buy")
    ax.plot(time_d, p_sell, color="#54a24b", lw=1.6, label="Grid sell")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    ax = axes[2]
    ax.set_facecolor("#fffdf8")
    ax.plot(time_d, sto_net, color="#6f4e7c", lw=1.7, label="Storage net output")
    ax.plot(ev_time_d, ev_charge, color="#f58518", lw=1.4, label="EV charge")
    ax.plot(ev_time_d, ev_discharge, color="#2f6c8f", lw=1.4, label="EV discharge")
    ticks = np.arange(0.0, float(ev_time_d[-1]) + 1e-9, 1.0) if len(ev_time_d) else np.arange(0.0, 7.0 + 1e-9, 1.0)
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon+"][: len(ticks)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    fig.text(0.5, 0.94, f"objective={summary['objective']:.2f}", ha="center", va="center", fontsize=11, color="#3c3c3c")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _compute_summary(env, rewards: list[float], infos: list[dict], replay) -> dict:
    summary = {
        "objective": 0.0,
        "gen_cost": 0.0,
        "grid_buy_cost": 0.0,
        "grid_sell_rev": 0.0,
        "grid_cost": 0.0,
        "curt_cost": 0.0,
        "volt_pen": 0.0,
        "branch_pen": 0.0,
        "grid_pen": 0.0,
        "storage_terminal_value": 0.0,
        "ev_depart_penalty": 0.0,
        "ev_unmet_kwh": 0.0,
        "v2g_value": 0.0,
        "v2g_reward": 0.0,
        "episode_reward_scaled": float(sum(rewards)),
        "raw_reward_sum": 0.0,
        "step_cost_total_sum": 0.0,
        "episode_length": int(len(infos)),
        "voltage_penalty_coeff": float(env.cfg.v_penalty),
        "ev_depart_penalty_coeff": float(env.cfg.ev_depart_penalty),
        "v2g_reward_coeff": float(env.cfg.v2g_reward_coeff),
        "reward_scale": float(env.cfg.reward_scale),
    }
    for info in infos:
        summary["gen_cost"] += float(info.get("gen_cost", 0.0))
        summary["grid_buy_cost"] += float(info.get("grid_buy_cost", 0.0))
        summary["grid_sell_rev"] += float(info.get("grid_sell_rev", 0.0))
        summary["curt_cost"] += float(info.get("curt_cost", 0.0))
        summary["volt_pen"] += float(info.get("volt_pen", 0.0))
        summary["branch_pen"] += float(info.get("branch_pen", 0.0))
        summary["grid_pen"] += float(info.get("grid_pen", 0.0))
        summary["storage_terminal_value"] += float(info.get("storage_terminal_value", 0.0))
        summary["ev_depart_penalty"] += float(info.get("ev_depart_pen_step", 0.0))
        summary["ev_unmet_kwh"] += float(info.get("ev_depart_short_kwh_step", 0.0))
        summary["v2g_value"] += float(info.get("v2g_value", 0.0))
        summary["v2g_reward"] += float(info.get("v2g_reward", 0.0))
        summary["step_cost_total_sum"] += float(info.get("step_cost_total", 0.0))
        summary["raw_reward_sum"] += float(info.get("raw_reward", 0.0))
    summary["grid_cost"] = float(summary["grid_buy_cost"] - summary["grid_sell_rev"])
    summary["objective"] = (
        float(summary["gen_cost"])
        + float(summary["grid_cost"])
        + float(summary["curt_cost"])
        + float(summary["volt_pen"])
        + float(summary["branch_pen"])
        + float(summary["grid_pen"])
        + float(summary["ev_depart_penalty"])
        - float(summary["storage_terminal_value"])
        - float(summary["v2g_reward"])
    )
    summary["event_served"] = int(
        sum(1 for ev in replay._meta["events_cfg"] if int(ev.get("departure_t", 0)) <= int(env.horizon_s))
    )
    return summary


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    meta = _load_meta(run_dir)
    model_path = Path(args.model_path).resolve() if args.model_path else (run_dir / "best_model" / "best_model.zip").resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"missing model path: {model_path}")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "best_eval").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env_from_meta(meta)
    algo = str(meta.get("algorithm", "ppo")).lower()
    algo_cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}.get(algo)
    if algo_cls is None:
        raise ValueError(f"unsupported algorithm in train meta: {algo}")
    model = algo_cls.load(str(model_path), device=str(args.device))

    obs, _ = env.reset()
    rewards: list[float] = []
    infos: list[dict] = []
    done = False
    step_idx = 0
    while not done:
        action, _ = model.predict(obs, deterministic=not bool(args.stochastic))
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        infos.append(dict(info))
        step_idx += 1
        if step_idx % 200 == 0:
            print(f"eval step {step_idx}/{env.T}", flush=True)
        done = bool(terminated or truncated)

    trace = env.get_trace()
    replay = _build_replay(env, trace)
    grid = Grid.load_from_yaml(str(meta["grid_yaml"]))
    summary = _compute_summary(env, rewards, infos, replay)

    dep_stats = _plot_departure_soc_and_v2g(replay, str(meta["ev_yaml"]), out_dir / "best_eval_departure_soc_v2g.png")
    summary.update(dep_stats)

    time_d, station_series = _build_station_netload_from_trace(env, trace)
    _lineplot(time_d, station_series, out_dir / "best_eval_station_netload.png", "Weekly Charging Station Net Load", "Net load (MW)")

    ev_time_d, ev_soc_series = _build_vehicle_soc_series(replay, str(meta["ev_yaml"]))
    _lineplot(ev_time_d, ev_soc_series, out_dir / "best_eval_ev_soc_3.png", "Weekly SOC of 3 Representative EVs", "SOC")

    v2g_time_d, v2g_series = _build_v2g_compare_series(replay, grid)
    summary.update(_plot_v2g_compare(v2g_time_d, v2g_series, out_dir / "best_eval_v2g_compare.png"))

    _plot_storage_weekly(replay, out_dir / "best_eval_storage.png")
    _plot_cost(summary, out_dir / "best_eval_cost.png")
    _plot_system(env, replay, summary, out_dir / "best_eval_system.png")
    detail_out = out_dir / "best_eval_v2g_station_detail.png"
    if detail_out.exists():
        detail_out.unlink()

    _dump_result_yaml(out_dir / "best_eval_summary.yaml", summary)

    print(f"algo={algo}")
    print(f"model={model_path}")
    print(f"summary={out_dir / 'best_eval_summary.yaml'}")
    print(f"objective={summary['objective']:.4f}")
    print(f"ev_depart_penalty={summary['ev_depart_penalty']:.4f}")
    print(f"ev_unmet_kwh={summary['ev_unmet_kwh']:.4f}")
    print(f"volt_pen={summary['volt_pen']:.4f}")
    env.close()


if __name__ == "__main__":
    main()
