import argparse
import csv
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import yaml

os.environ.setdefault("LC_ALL", "C")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_diffusor")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DERs import ChargingEvent, EV, Station
from core.grid import Grid
from solvers.opf.linear_distflow import LinDistFlowConfig
from utils import OneLineDict
from utils.event_generation import generate_events
from utils.ev_generation import generate_evs
from utils.plot_ev_dataset import load_evs as load_ev_cfg, plot_dataset
from utils.plot_ev_events import plot_event_overview
from utils.plot_ev_queue import load_events as load_event_cfg, plot_queue_overview
from utils.plot_style import use_times_new_roman
from utils.plot_weekly_opf import (
    RollingNoPrimalSolutionError,
    _dump_result_yaml,
    _build_station_netload,
    _build_v2g_compare_series,
    _build_vehicle_soc_series,
    _compute_summary,
    _export_exogenous_sequences,
    _lineplot,
    _plot_departure_soc_and_v2g,
    _plot_storage_weekly,
    _plot_v2g_compare,
    solve_opf_rolling,
)
from utils.queue import assign_queue_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multiple random weekly EV scenarios and solve rolling baselines.")
    parser.add_argument("--grid-yaml", default="config/IEEE33.yaml")
    parser.add_argument("--time-yaml", default="config/time.yaml")
    parser.add_argument("--out-root", default="results/scenario_batch_1000ev_1w")
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--num-evs", type=int, default=1000)
    parser.add_argument("--w-v2g", type=float, default=0.60)
    parser.add_argument("--w-car", type=float, default=0.30)
    parser.add_argument("--w-bus", type=float, default=0.10)
    parser.add_argument("--v2g-opt", type=float, default=0.60)
    parser.add_argument("--rate-v2g", type=float, default=1.0)
    parser.add_argument("--rate-bus", type=float, default=1.0)
    parser.add_argument("--rate-car", type=float, default=3.0)
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--resolution-minutes", type=int, default=5)
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--mip-gap", type=float, default=5e-4)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-variants", default="")
    parser.add_argument("--retry-no-primal", action="store_true")
    parser.add_argument("--max-seed-attempts", type=int, default=20)
    parser.add_argument("--rolling-interval-minutes", type=int, default=360)
    parser.add_argument("--lookahead-hours", default="1,3,6,12")
    parser.add_argument("--ev-target-soc", type=float, default=0.9)
    parser.add_argument("--ev-depart-penalty", type=float, default=20.0)
    parser.add_argument("--v2g-reward-coeff", type=float, default=5.0)
    parser.add_argument("--storage-terminal-value-coeff", type=float, default=1.0)
    return parser.parse_args()


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True, width=1000)


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _backup_dir(path: Path, label: str) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.name}_{label}_{stamp}")
    path.rename(backup)
    return backup


def _next_unused_seed(seed_cursor: int, used_seeds: set[int]) -> int:
    seed = int(seed_cursor)
    while seed in used_seeds:
        seed += 1
    return seed


def _build_row(
    *,
    scenario: str,
    variant: str,
    seed: int,
    lookahead_h: int,
    effective_interval: int,
    ev_stats: dict,
    q_stats: dict,
    summary: dict,
) -> dict:
    return {
        "scenario_base": scenario,
        "scenario": variant,
        "seed": int(seed),
        "lookahead_hours": int(lookahead_h),
        "rolling_interval_minutes": int(effective_interval),
        **ev_stats,
        **q_stats,
        **summary,
    }


def _format_csv_value(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return f"{float(val):.4f}"
    if isinstance(val, float):
        return "0" if abs(val) < 1e-4 else f"{round(val, 4):.4f}"
    return val


def _queue_stats(events_yaml: Path) -> dict:
    events = load_event_cfg(events_yaml)
    waits = [int(ev.get("wait_t", -1)) for ev in events if int(ev.get("wait_t", -1)) > 0]
    return {
        "events_total": int(len(events)),
        "queued_events": int(sum(int(ev.get("wait_t", -1)) > 0 for ev in events)),
        "unserved_events": int(sum(int(ev.get("start_t", -1)) < 0 for ev in events)),
        "avg_wait_min": float(sum(waits) / len(waits) / 60.0) if waits else 0.0,
        "max_wait_min": float(max(waits) / 60.0) if waits else 0.0,
    }


def _ev_stats(ev_yaml: Path) -> dict:
    evs = load_ev_cfg(ev_yaml)
    counts = {"V2G": 0, "CAR": 0, "BUS": 0}
    for ev in evs:
        counts[str(ev["type"]).upper()] += 1
    return {
        "ev_total": int(len(evs)),
        "ev_v2g": int(counts["V2G"]),
        "ev_car": int(counts["CAR"]),
        "ev_bus": int(counts["BUS"]),
    }


def _plot_batch_objective(rows: list[dict], out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["scenario"] for row in rows]
    objective = [float(row["objective"]) for row in rows]
    ev_pen = [float(row["ev_depart_penalty"]) for row in rows]
    volt_pen = [float(row["volt_pen"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9.0), sharex=True)
    fig.patch.set_facecolor("#faf8f4")

    for ax, vals, title, color in zip(
        axes,
        [objective, ev_pen, volt_pen],
        ["Objective", "EV departure penalty", "Voltage penalty"],
        ["#4c78a8", "#e45756", "#b279a2"],
    ):
        ax.set_facecolor("#fffdf8")
        ax.bar(names, vals, color=color, alpha=0.88)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Scenario")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_batch_cost_breakdown(rows: list[dict], out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names = [row["scenario"] for row in rows]
    positives = [
        ("gen_cost", "#4c78a8"),
        ("grid_cost", "#f58518"),
        ("curt_cost", "#54a24b"),
        ("volt_pen", "#b279a2"),
        ("ev_depart_penalty", "#e45756"),
    ]
    negatives = [("v2g_reward", "#2a9d8f"), ("storage_terminal_value", "#bab0ab")]

    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    fig.patch.set_facecolor("#faf8f4")
    ax.set_facecolor("#fffdf8")

    bottom = [0.0] * len(rows)
    for key, color in positives:
        vals = [float(row.get(key, 0.0)) for row in rows]
        ax.bar(x, vals, bottom=bottom, color=color, width=0.72, label=key)
        bottom = [b + v for b, v in zip(bottom, vals)]

    neg_bottom = [0.0] * len(rows)
    for key, color in negatives:
        vals = [-float(row.get(key, 0.0)) for row in rows]
        ax.bar(x, vals, bottom=neg_bottom, color=color, width=0.72, label=f"-{key}")
        neg_bottom = [b + v for b, v in zip(neg_bottom, vals)]

    ax.set_xticks(list(x), names)
    ax.set_title("Scenario Cost Breakdown")
    ax.set_ylabel("Cost")
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _aggregate_summary(rows: list[dict]) -> dict:
    numeric_keys = [
        "objective",
        "gen_cost",
        "grid_cost",
        "curt_cost",
        "volt_pen",
        "storage_terminal_value",
        "ev_depart_penalty",
        "ev_unmet_kwh",
        "v2g_reward",
        "avg_depart_soc_v2g",
        "avg_depart_soc_non_v2g",
        "queued_events",
        "unserved_events",
        "avg_wait_min",
        "max_wait_min",
    ]
    agg = {}
    for key in numeric_keys:
        vals = [float(row.get(key, 0.0)) for row in rows]
        agg[key] = {
            "mean": float(mean(vals)) if vals else 0.0,
            "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)) if vals else 0.0,
            "max": float(max(vals)) if vals else 0.0,
        }
    return agg


def _aggregate_by_key(rows: list[dict], group_key: str) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row[group_key]), []).append(row)
    out = {}
    for key, group_rows in grouped.items():
        out[key] = _aggregate_summary(group_rows)
    return out


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    lookahead_list = [int(x.strip()) for x in str(args.lookahead_hours).split(",") if x.strip()]
    if not lookahead_list:
        raise ValueError("--lookahead-hours must contain at least one integer")
    skip_variants = {x.strip() for x in str(args.skip_variants).split(",") if x.strip()}

    grid = Grid.load_from_yaml(str(args.grid_yaml))
    stations = Station.load_from_yaml(str(args.grid_yaml))
    solver_options = {} if str(args.solver).lower() == "gurobi" else None
    if solver_options is not None and float(args.mip_gap) > 0.0:
        solver_options["MIPGap"] = float(args.mip_gap)
    if solver_options is not None and float(args.time_limit) > 0.0:
        solver_options["TimeLimit"] = float(args.time_limit)
    if solver_options == {}:
        solver_options = None

    rows: list[dict] = []
    used_seeds: set[int] = set()
    for meta_path in sorted(out_root.glob("S*/scenario_meta.yaml")):
        meta = _read_yaml(meta_path)
        if "seed" in meta:
            used_seeds.add(int(meta["seed"]))
    seed_cursor = max(int(args.start_seed), max(used_seeds) + 1 if used_seeds else int(args.start_seed))

    for idx in range(int(args.scenario_count)):
        scenario_idx = int(args.start_index) + int(idx)
        scenario = f"S{int(scenario_idx)}"
        scenario_dir = out_root / scenario
        expected_variants = [
            (lookahead_idx, int(lookahead_h), f"{scenario}_{lookahead_idx}")
            for lookahead_idx, lookahead_h in enumerate(lookahead_list)
            if f"{scenario}_{lookahead_idx}" not in skip_variants
        ]

        existing_rows: list[dict] = []
        completed_variants: set[str] = set()
        scenario_meta = None
        ev_yaml = scenario_dir / "ev.yaml"
        event_yaml = scenario_dir / "event.yaml"
        if bool(args.skip_existing) and scenario_dir.exists() and expected_variants:
            completed_variants = {
                variant for _, _, variant in expected_variants
                if (scenario_dir / variant / "summary.yaml").exists()
            }
            if completed_variants:
                scenario_meta_path = scenario_dir / "scenario_meta.yaml"
                if scenario_meta_path.exists() and ev_yaml.exists() and event_yaml.exists():
                    scenario_meta = _read_yaml(scenario_meta_path)
                    seed = int(scenario_meta.get("seed", 0))
                    used_seeds.add(seed)
                    ev_stats = _ev_stats(ev_yaml)
                    q_stats = _queue_stats(event_yaml)
                    for lookahead_idx, lookahead_h, variant in expected_variants:
                        if variant not in completed_variants:
                            continue
                        summary = _read_yaml(scenario_dir / variant / "summary.yaml")
                        effective_interval = int(min(int(args.rolling_interval_minutes), int(lookahead_h) * 60))
                        existing_rows.append(
                            _build_row(
                                scenario=scenario,
                                variant=variant,
                                seed=seed,
                                lookahead_h=int(lookahead_h),
                                effective_interval=int(effective_interval),
                                ev_stats=ev_stats,
                                q_stats=q_stats,
                                summary=summary,
                            )
                        )
                    if len(completed_variants) == len(expected_variants):
                        print(f"[{scenario}] reuse existing complete scenario", flush=True)
                        rows.extend(existing_rows)
                        continue
                    print(
                        f"[{scenario}] reuse {len(completed_variants)}/{len(expected_variants)} completed variants; "
                        "only missing variants will be solved",
                        flush=True,
                    )

        attempt = 0
        while True:
            if scenario_dir.exists() and not completed_variants:
                backup = _backup_dir(scenario_dir, "incomplete")
                if backup is not None:
                    print(f"[{scenario}] backup previous partial results -> {backup.name}", flush=True)

            if completed_variants and scenario_meta is not None:
                seed = int(scenario_meta.get("seed", 0))
                ev_stats = _ev_stats(ev_yaml)
                q_stats = _queue_stats(event_yaml)
                scenario_rows: list[dict] = list(existing_rows)
                completed_variants_local = set(completed_variants)
                completed_variants = set()
            else:
                seed = _next_unused_seed(seed_cursor, used_seeds)
                used_seeds.add(int(seed))
                seed_cursor = int(seed) + 1
                attempt += 1

                scenario_dir.mkdir(parents=True, exist_ok=True)
                print(f"[{scenario}] generate ev/event with seed={seed} (attempt {attempt})", flush=True)

                random.seed(seed)
                evs = generate_evs(
                    int(args.num_evs),
                    start_id=1,
                    type_weights={"V2G": float(args.w_v2g), "CAR": float(args.w_car), "BUS": float(args.w_bus)},
                    v2g_opt=float(args.v2g_opt),
                )
                ev_yaml = scenario_dir / "ev.yaml"
                _write_yaml(ev_yaml, {"evs": [OneLineDict(ev.to_config()) for ev in evs]})

                random.seed(seed)
                events = generate_events(
                    days=int(args.days),
                    evs=evs,
                    stations=stations,
                    events_per_ev_per_day={"V2G": float(args.rate_v2g), "BUS": float(args.rate_bus), "CAR": float(args.rate_car)},
                    start_id=1,
                )
                queued_events = assign_queue_fields(events=[e.to_config() for e in events], stations=stations)
                event_yaml = scenario_dir / "event.yaml"
                _write_yaml(event_yaml, {"events": [OneLineDict(ev) for ev in queued_events]})

                plot_dataset(load_ev_cfg(ev_yaml), scenario_dir / "ev_dataset.png")
                plot_queue_overview(event_yaml, ev_yaml, scenario_dir / "ev_queue_overview.png")
                plot_event_overview(event_yaml, ev_yaml, scenario_dir / "ev_events_overview.png")

                _write_yaml(scenario_dir / "scenario_meta.yaml", {
                    "scenario": scenario,
                    "seed": int(seed),
                    "grid_yaml": str(args.grid_yaml),
                    "time_yaml": str(args.time_yaml),
                    "num_evs": int(args.num_evs),
                    "weights": {"V2G": float(args.w_v2g), "CAR": float(args.w_car), "BUS": float(args.w_bus)},
                    "v2g_opt": float(args.v2g_opt),
                    "rates": {"V2G": float(args.rate_v2g), "BUS": float(args.rate_bus), "CAR": float(args.rate_car)},
                    "lookahead_hours_list": lookahead_list,
                    "base_rolling_interval_minutes": int(args.rolling_interval_minutes),
                })

                ev_stats = _ev_stats(ev_yaml)
                q_stats = _queue_stats(event_yaml)
                scenario_rows = []
                completed_variants_local = set()

            try:
                for lookahead_idx, lookahead_h, variant in expected_variants:
                    if variant in completed_variants_local:
                        continue
                    variant_dir = scenario_dir / variant
                    variant_dir.mkdir(parents=True, exist_ok=True)
                    effective_interval = int(min(int(args.rolling_interval_minutes), int(lookahead_h) * 60))
                    summary_path = variant_dir / "summary.yaml"

                    print(
                        f"[{variant}] solve rolling baseline, lookahead={lookahead_h}h, interval={effective_interval}min",
                        flush=True,
                    )

                    cfg = LinDistFlowConfig(
                        horizon_hours=int(args.hours),
                        resolution_minutes=int(args.resolution_minutes),
                        solver_name=str(args.solver),
                        tee=bool(args.tee),
                        time_yaml_path=str(args.time_yaml),
                        ev_yaml_path=str(ev_yaml),
                        queue_yaml_path=str(event_yaml),
                        ev_target_soc=float(args.ev_target_soc),
                        ev_depart_penalty=float(args.ev_depart_penalty),
                        v2g_reward_coeff=float(args.v2g_reward_coeff),
                        storage_terminal_value_coeff=float(args.storage_terminal_value_coeff),
                        solver_options=solver_options,
                    )
                    model = solve_opf_rolling(
                        grid,
                        cfg,
                        lookahead_hours=int(lookahead_h),
                        roll_interval_minutes=int(effective_interval),
                    )

                    summary = _compute_summary(model, grid, cfg)
                    summary.update(_plot_departure_soc_and_v2g(model, str(ev_yaml), variant_dir / "departure_soc_v2g.png"))

                    time_d, station_series = _build_station_netload(model, grid)
                    _lineplot(time_d, station_series, variant_dir / "station_netload.png", "Weekly Charging Station Net Load", "Net load (MW)")

                    ev_time_d, ev_soc_series = _build_vehicle_soc_series(model, str(ev_yaml))
                    _lineplot(ev_time_d, ev_soc_series, variant_dir / "ev_soc_3.png", "Weekly SOC of 3 Representative EVs", "SOC")

                    v2g_time_d, v2g_series = _build_v2g_compare_series(model, grid)
                    summary.update(_plot_v2g_compare(v2g_time_d, v2g_series, variant_dir / "v2g_compare.png"))

                    _plot_storage_weekly(model, variant_dir / "storage.png")
                    detail_out = variant_dir / "v2g_station_detail.png"
                    if detail_out.exists():
                        detail_out.unlink()
                    _export_exogenous_sequences(model, variant_dir / "exogenous.yaml")

                    scenario_rows.append(
                        _build_row(
                            scenario=scenario,
                            variant=variant,
                            seed=int(seed),
                            lookahead_h=int(lookahead_h),
                            effective_interval=int(effective_interval),
                            ev_stats=ev_stats,
                            q_stats=q_stats,
                            summary=summary,
                        )
                    )
                    _dump_result_yaml(summary_path, summary)
                    _write_yaml(variant_dir / "variant_meta.yaml", {
                        "scenario_base": scenario,
                        "scenario": variant,
                        "seed": int(seed),
                        "lookahead_hours": int(lookahead_h),
                        "rolling_interval_minutes": int(effective_interval),
                    })
                    print(
                        f"[{variant}] objective={summary['objective']:.2f} "
                        f"volt_pen={summary['volt_pen']:.2f} ev_depart_penalty={summary['ev_depart_penalty']:.2f}",
                        flush=True,
                    )
            except RollingNoPrimalSolutionError as exc:
                if not bool(args.retry_no_primal):
                    raise
                failed_backup = _backup_dir(scenario_dir, f"failed_seed{seed}")
                if failed_backup is not None:
                    print(
                        f"[{scenario}] seed={seed} failed at window {exc.window_idx}/{exc.total_windows} "
                        f"(t={exc.t0}); backed up to {failed_backup.name} and retrying with a new seed",
                        flush=True,
                    )
                if attempt >= int(args.max_seed_attempts):
                    raise RuntimeError(
                        f"[{scenario}] exceeded max seed attempts ({int(args.max_seed_attempts)}) "
                        "while retrying after no-primal windows"
                    ) from exc
                continue

            rows.extend(scenario_rows)
            break

    batch_summary = {
        "batch": {
            "scenario_count": int(args.scenario_count),
            "start_index": int(args.start_index),
            "start_seed": int(args.start_seed),
            "grid_yaml": str(args.grid_yaml),
            "time_yaml": str(args.time_yaml),
            "num_evs": int(args.num_evs),
            "days": int(args.days),
            "lookahead_hours_list": lookahead_list,
            "rolling_interval_minutes": int(args.rolling_interval_minutes),
            "mip_gap": float(args.mip_gap),
            "time_limit": float(args.time_limit),
            "retry_no_primal": bool(args.retry_no_primal),
            "max_seed_attempts": int(args.max_seed_attempts),
        },
        "scenarios": rows,
        "aggregate": _aggregate_summary(rows),
        "aggregate_by_lookahead_hours": _aggregate_by_key(rows, "lookahead_hours"),
    }
    _dump_result_yaml(out_root / "batch_summary.yaml", batch_summary)

    fieldnames = [
        "scenario_base",
        "scenario",
        "seed",
        "lookahead_hours",
        "rolling_interval_minutes",
        "ev_total",
        "ev_v2g",
        "ev_car",
        "ev_bus",
        "events_total",
        "queued_events",
        "unserved_events",
        "avg_wait_min",
        "max_wait_min",
        "objective",
        "gen_cost",
        "grid_cost",
        "curt_cost",
        "volt_pen",
        "storage_terminal_value",
        "ev_depart_penalty",
        "ev_unmet_kwh",
        "v2g_reward",
        "avg_depart_soc_v2g",
        "avg_depart_soc_non_v2g",
    ]
    with open(out_root / "batch_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_csv_value(row.get(k)) for k in fieldnames})

    print(f"batch summary: {out_root / 'batch_summary.yaml'}", flush=True)


if __name__ == "__main__":
    main()
