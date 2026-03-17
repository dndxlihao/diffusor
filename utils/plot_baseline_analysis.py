import argparse
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pyomo.environ import value

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

from core.grid import Grid
from solvers.opf.linear_distflow import LinDistFlowConfig, load_evs, solve_opf
from utils.plot_style import use_times_new_roman


def analyze_solution(model, cfg: LinDistFlowConfig) -> dict:
    base_mva = float(model._meta["baseMVA"])
    delta_t = float(model._meta["delta_t"])
    objective = float(value(model.Obj))

    grid_cost = 0.0
    curt_cost = 0.0
    storage_end_pen = 0.0
    ev_depart_pen = 0.0
    v2g_reward = 0.0
    gen_cost = 0.0

    for tt in model.T:
        price = float(value(model.price[tt]))
        p_buy = float(value(model.P_buy[tt])) * base_mva
        p_sell = float(value(model.P_sell[tt])) * base_mva
        grid_cost += (p_buy - p_sell) * price * delta_t * 1000.0
        for gid in model.GEN:
            g = model.Pg[gid, tt] * base_mva
            gen = next(item for item in model._meta["gens_cfg"] if int(item["gen_i"]) == int(gid))
            gen_cost += (
                float(gen["cost_c2"]) * float(value(g)) ** 2
                + float(gen["cost_c1"]) * float(value(g))
                + float(gen["cost_c0"])
            ) * delta_t
        for rid in model.REN:
            curt_cost += float(value(model.curt_cost[rid])) * (float(value(model.curt[rid, tt])) * base_mva) * delta_t

    for sid in model.STO:
        storage_end_pen += float(cfg.storage_end_penalty) * float(value(model.soc_end_short[sid]))

    for sid in model.STN:
        for k in model.KEV_SOC:
            if float(value(model.st_due_flag[sid, k])) > 0.5:
                ev_depart_pen += float(cfg.ev_depart_penalty) * float(value(model.st_short[sid, k]))
        for k in model.KEV:
            price = float(value(model.ev_price[k]))
            dt_h = float(model._meta["ev_step_s"]) / 3600.0
            v2g_reward += float(cfg.v2g_reward_coeff) * price * float(value(model.st_pdis[sid, k])) * dt_h

    return {
        "objective": objective,
        "gen_cost": gen_cost,
        "grid_cost": grid_cost,
        "curt_cost": curt_cost,
        "storage_end_penalty": storage_end_pen,
        "ev_depart_penalty": ev_depart_pen,
        "v2g_reward": v2g_reward,
        "ev_unmet_kwh": float(
            getattr(model, "_dispatch", {}).get("unmet_departure_kwh", {})
            and sum(model._dispatch["unmet_departure_kwh"].values())
            or 0.0
        ),
        "storage_short": {int(sid): float(value(model.soc_end_short[sid])) for sid in model.STO},
    }


def build_event_series(model, ev_yaml: str) -> tuple[list[int], list[float], np.ndarray, np.ndarray]:
    dispatch = getattr(model, "_dispatch", {})
    events = list(model._meta.get("events_cfg", []))
    ev_map = load_evs(ev_yaml)
    k_list = [int(k) for k in model.KEV]
    t_h = [float(k) * float(model._meta["ev_step_s"]) / 3600.0 for k in k_list]

    active_events = []
    for ev in events:
        eid = int(ev["event_i"])
        if any((eid, k) in dispatch.get("event_pch_kw", {}) or (eid, k) in dispatch.get("event_pdis_kw", {}) for k in k_list):
            active_events.append(ev)
    active_events.sort(key=lambda e: (int(e.get("start_t", e["arrival_t"])), int(e["station_i"]), int(e["event_i"])))

    p_mat = np.zeros((len(active_events), len(k_list)), dtype=float)
    soc_mat = np.full((len(active_events), len(k_list) + 1), np.nan, dtype=float)
    ev_ids: list[int] = []
    dt_h = float(model._meta["ev_step_s"]) / 3600.0

    for row, ev in enumerate(active_events):
        eid = int(ev["event_i"])
        ev_i = int(ev["ev_i"])
        obj = ev_map.get(ev_i)
        if obj is None:
            continue
        ev_ids.append(eid)
        ev_obj = ev_by_id.get(int(ev["ev_i"]))
        soc = float(ev.get("soc_init", getattr(ev_obj, "soc", 0.5)))
        soc_mat[row, 0] = soc
        cap = float(obj.capacity)
        for kk, k in enumerate(k_list):
            pch = float(dispatch.get("event_pch_kw", {}).get((eid, k), 0.0))
            pdis = float(dispatch.get("event_pdis_kw", {}).get((eid, k), 0.0))
            obj.soc = soc
            net = pch - pdis
            p_mat[row, kk] = net
            if pch > 1e-9:
                eta = max(1e-6, float(obj.charge_efficiency(pch)))
                soc = min(1.0, soc + (pch * eta * dt_h) / cap)
            elif pdis > 1e-9:
                soc = max(0.0, soc - (pdis / max(1e-6, float(obj.eta_dis)) * dt_h) / cap)
            soc_mat[row, kk + 1] = soc
    return ev_ids, t_h, p_mat, soc_mat


def plot_cost(summary: dict, out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fffdf9")
    labels = ["Gen", "Grid", "Curtail", "Storage end", "EV depart"]
    vals = [
        summary["gen_cost"],
        summary["grid_cost"],
        summary["curt_cost"],
        summary["storage_end_penalty"],
        summary["ev_depart_penalty"],
    ]
    colors = ["#4c78a8", "#54a24b", "#d1495b", "#6f4e7c", "#f58518"]
    ax.bar(labels, vals, color=colors, alpha=0.88)
    ax.set_title("Objective Cost Breakdown", fontsize=14)
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_system(model, summary: dict, out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.0), sharex=True)
    fig.patch.set_facecolor("#fbfaf7")
    time_h = [float(tt) * float(model._meta["delta_t"]) for tt in model.T]
    ev_h = [float(k) * float(model._meta["ev_step_s"]) / 3600.0 for k in model.KEV]
    base_mva = float(model._meta["baseMVA"])

    ax = axes[0]
    ax.set_facecolor("#fffdf9")
    gen_sum = [sum(float(value(model.Pg[g, tt])) for g in model.GEN) * base_mva for tt in model.T]
    pav = [sum(float(value(model.Pav[r, tt])) for r in model.REN) * base_mva for tt in model.T]
    pr = [sum(float(value(model.Pr[r, tt])) for r in model.REN) * base_mva for tt in model.T]
    curt = [sum(float(value(model.curt[r, tt])) for r in model.REN) * base_mva for tt in model.T]
    ax.plot(time_h, gen_sum, color="#4c78a8", lw=1.8, label="Dispatchable generation")
    ax.plot(time_h, pav, color="#2b83ba", lw=1.4, alpha=0.7, label="Renewable available")
    ax.plot(time_h, pr, color="#54a24b", lw=1.7, label="Renewable dispatched")
    ax.fill_between(time_h, curt, color="#d1495b", alpha=0.15, label="Curtailment")
    ax.set_title("Variable Generation", fontsize=13)
    ax.set_ylabel("MW")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")

    ax = axes[1]
    ax.set_facecolor("#fffdf9")
    p_buy = [float(value(model.P_buy[tt])) * base_mva for tt in model.T]
    p_sell = [float(value(model.P_sell[tt])) * base_mva for tt in model.T]
    bus_load = []
    for tt in model.T:
        t_sec = int(float(tt) * float(model._meta["delta_t"]) * 3600.0)
        total = 0.0
        for bid in model.BUS:
            bus = next(b for b in model._meta["buses_cfg"] if int(b["bus_i"]) == int(bid))
            total += float(bus["Pd"]) * float(model._meta["load_scale"][int(tt)]) / 1000.0
        bus_load.append(total)
    ax.plot(time_h, bus_load, color="#222222", lw=1.8, label="Total load")
    ax.plot(time_h, p_buy, color="#cc5a3d", lw=1.6, label="Grid buy")
    ax.plot(time_h, p_sell, color="#54a24b", lw=1.6, label="Grid sell")
    ax.set_title("Load and Grid Exchange", fontsize=13)
    ax.set_ylabel("MW")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[2]
    ax.set_facecolor("#fffdf9")
    sto_p = [
        sum((float(value(model.P_dis[s, tt])) - float(value(model.P_ch[s, tt]))) for s in model.STO) * base_mva
        for tt in model.T
    ]
    ev_charge = [sum(float(value(model.st_pch[sid, k])) for sid in model.STN) / 1000.0 for k in model.KEV]
    ev_dis = [sum(float(value(model.st_pdis[sid, k])) for sid in model.STN) / 1000.0 for k in model.KEV]
    ax.plot(time_h, sto_p, color="#6f4e7c", lw=1.7, label="Storage net output")
    ax.plot(ev_h, ev_charge, color="#f58518", lw=1.5, label="EV charge")
    ax.plot(ev_h, ev_dis, color="#2f6c8f", lw=1.5, label="EV discharge")
    ax.set_title("Flexible Loads and Storage", fontsize=13)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("MW")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Baseline Source-Load Dynamics (6 h, 5 min)", fontsize=17, y=0.98)
    fig.text(0.5, 0.94, f"objective={summary['objective']:.2f}", ha="center", va="center", fontsize=11, color="#3c3c3c")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ev_detail(model, ev_yaml: str, out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ev_ids, t_h, p_mat, soc_mat = build_event_series(model, ev_yaml)
    if not ev_ids:
        raise RuntimeError("no active EV events to plot")

    n_events = len(ev_ids)
    show_n = min(40, n_events)
    p_show = p_mat[:show_n, :]
    soc_show = soc_mat[:show_n, :]
    y_labels = [str(eid) for eid in ev_ids[:show_n]]

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 8.5), sharex=True)
    fig.patch.set_facecolor("#fbfaf7")

    ax = axes[0]
    ax.set_facecolor("#fffdf9")
    vmax = max(1.0, float(np.max(np.abs(p_show))))
    im = ax.imshow(
        p_show,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[t_h[0], t_h[-1] + (t_h[1] - t_h[0] if len(t_h) > 1 else 0.0833), show_n, 0],
    )
    ax.set_title("Event-Level Charging/Discharging Power (top active events)", fontsize=13)
    ax.set_ylabel("Event ID")
    ax.set_yticks(np.arange(show_n) + 0.5, y_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("Power (kW)")

    ax = axes[1]
    ax.set_facecolor("#fffdf9")
    im2 = ax.imshow(
        soc_show[:, :-1],
        aspect="auto",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        extent=[t_h[0], t_h[-1] + (t_h[1] - t_h[0] if len(t_h) > 1 else 0.0833), show_n, 0],
    )
    ax.set_title("Event-Level SOC Trajectories (top active events)", fontsize=13)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Event ID")
    ax.set_yticks(np.arange(show_n) + 0.5, y_labels)
    cbar2 = fig.colorbar(im2, ax=ax, fraction=0.02, pad=0.01)
    cbar2.set_label("SOC")

    fig.suptitle("EV Charging/Discharging and SOC Evolution (6 h, 5 min)", fontsize=17, y=0.98)
    fig.text(0.5, 0.94, f"showing top {show_n} active events out of {n_events}", ha="center", va="center", fontsize=11, color="#3c3c3c")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-yaml", type=str, default="config/IEEE33.yaml")
    parser.add_argument("--ev-yaml", type=str, default="config/ev.yaml")
    parser.add_argument("--event-yaml", type=str, default="config/event.yaml")
    parser.add_argument("--time-yaml", type=str, default="config/time.yaml")
    parser.add_argument("--hours", type=int, default=6)
    parser.add_argument("--resolution-minutes", type=int, default=5)
    parser.add_argument("--solver", type=str, default="gurobi")
    parser.add_argument("--ev-target-soc", type=float, default=0.8)
    parser.add_argument("--ev-depart-penalty", type=float, default=20.0)
    parser.add_argument("--cost-out", type=str, default="results/baseline_6h_costs.png")
    parser.add_argument("--system-out", type=str, default="results/baseline_6h_system.png")
    parser.add_argument("--ev-out", type=str, default="results/baseline_6h_ev.png")
    parser.add_argument("--summary-out", type=str, default="results/baseline_6h_summary.yaml")
    args = parser.parse_args()

    cfg = LinDistFlowConfig(
        horizon_hours=int(args.hours),
        resolution_minutes=int(args.resolution_minutes),
        solver_name=str(args.solver),
        tee=False,
        queue_yaml_path=str(args.event_yaml),
        ev_yaml_path=str(args.ev_yaml),
        time_yaml_path=str(args.time_yaml),
        v2g_reward_coeff=0.0,
        ev_target_soc=float(args.ev_target_soc),
        ev_depart_penalty=float(args.ev_depart_penalty),
    )
    grid = Grid.load_from_yaml(str(args.grid_yaml))
    model, _, report = solve_opf(grid, cfg)
    if report["objective"] is None:
        raise RuntimeError(f"solve failed: {report['termination']}")

    # Store simple metadata for plotting and later reading.
    with open(args.grid_yaml, "r", encoding="utf-8") as f:
        grid_cfg = yaml.safe_load(f) or {}
    model._meta["buses_cfg"] = list(grid_cfg.get("buses", []))
    model._meta["gens_cfg"] = list(grid_cfg.get("generators", []))
    with open(args.time_yaml, "r", encoding="utf-8") as f:
        time_cfg = yaml.safe_load(f) or {}
    load_series = time_cfg.get("load_scale", [])
    load_map = {int(item["time"]): float(item["value"]) for item in load_series if isinstance(item, dict)}
    model._meta["load_scale"] = [
        float(load_map.get(int(float(tt) * float(model._meta["delta_t"]) * 3600.0), 1.0)) for tt in model.T
    ]

    summary = analyze_solution(model, cfg)
    summary["ev_target_soc"] = float(cfg.ev_target_soc)
    summary["ev_depart_penalty_coeff"] = float(cfg.ev_depart_penalty)
    plot_cost(summary=summary, out_path=Path(args.cost_out))
    plot_system(model, summary, Path(args.system_out))
    plot_ev_detail(model, args.ev_yaml, Path(args.ev_out))
    with open(args.summary_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)
    print("summary", summary)
    print("cost_plot", args.cost_out)
    print("system_plot", args.system_out)
    print("ev_plot", args.ev_out)
    print("summary_yaml", args.summary_out)


if __name__ == "__main__":
    main()
