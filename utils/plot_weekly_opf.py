import argparse
import sys
import types
from collections import Counter
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

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon+"]


def _day_ticks(max_days: float):
    ticks = np.arange(0.0, max_days + 1e-9, 1.0)
    return ticks, DAY_LABELS[: len(ticks)]


def _lineplot(x_d: np.ndarray, series: dict[str, np.ndarray], out_path: Path, title: str, ylabel: str) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    fig.patch.set_facecolor("#faf8f4")
    ax.set_facecolor("#fffdf8")
    colors = [
        "#284b63",
        "#3c6e71",
        "#d9a441",
        "#bc4749",
        "#5b8e7d",
        "#7c6a98",
        "#6c584c",
        "#8f5a3c",
        "#457b9d",
        "#a44a3f",
        "#5f6f52",
        "#b08968",
    ]
    for idx, (name, y) in enumerate(series.items()):
        ax.plot(x_d, y, lw=1.4, color=colors[idx % len(colors)], label=name, alpha=0.96)
    ticks, labels = _day_ticks(float(x_d[-1]) if len(x_d) else 7.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(float(x_d[0]) if len(x_d) else 0.0, float(x_d[-1]) if len(x_d) else 7.0)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Day of week")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2 if len(series) <= 8 else 3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _compute_summary(model, grid: Grid, cfg: LinDistFlowConfig) -> dict:
    base_mva = float(model._meta["baseMVA"])
    delta_t = float(model._meta["delta_t"])
    summary = {
        "objective": float(value(model.Obj)),
        "gen_cost": 0.0,
        "grid_cost": 0.0,
        "curt_cost": 0.0,
        "storage_end_penalty": 0.0,
        "ev_depart_penalty": 0.0,
        "v2g_reward": 0.0,
    }
    gen_map = {int(g.gen_i): g for g in grid.generators}
    ren_map = {int(r.renewable_i): r for r in grid.renewables}

    for tt in model.T:
        price = float(value(model.price[tt]))
        p_buy = float(value(model.P_buy[tt])) * base_mva
        p_sell = float(value(model.P_sell[tt])) * base_mva
        summary["grid_cost"] += (p_buy - p_sell) * price * delta_t * 1000.0
        for gid in model.GEN:
            p_mw = float(value(model.Pg[gid, tt])) * base_mva
            g = gen_map[int(gid)]
            summary["gen_cost"] += (
                float(g.cost_c2) * p_mw * p_mw + float(g.cost_c1) * p_mw + float(g.cost_c0)
            ) * delta_t
        for rid in model.REN:
            summary["curt_cost"] += float(ren_map[int(rid)].curt_cost) * (
                float(value(model.curt[rid, tt])) * base_mva
            ) * delta_t

    for sid in model.STO:
        summary["storage_end_penalty"] += float(cfg.storage_end_penalty) * float(value(model.soc_end_short[sid]))

    dt_h_ev = float(model._meta["ev_step_s"]) / 3600.0
    for sid in model.STN:
        for k in model.KEV_SOC:
            if float(value(model.st_due_flag[sid, k])) > 0.5:
                summary["ev_depart_penalty"] += float(cfg.ev_depart_penalty) * float(value(model.st_short[sid, k]))
        for k in model.KEV:
            summary["v2g_reward"] += (
                float(cfg.v2g_reward_coeff)
                * float(value(model.ev_price[k]))
                * float(value(model.st_pdis[sid, k]))
                * dt_h_ev
            )

    dispatch = getattr(model, "_dispatch", {})
    summary["ev_unmet_kwh"] = float(sum(dispatch.get("unmet_departure_kwh", {}).values()))
    summary["event_served"] = int(len(dispatch.get("final_soc", {})))
    return summary


def _build_station_netload(model, grid: Grid) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time_d = np.arange(int(model._meta["K_ev"]), dtype=float) * (float(model._meta["ev_step_s"]) / 3600.0) / 24.0
    series = {}
    for st in grid.stations:
        if int(getattr(st, "status", 1)) != 1:
            continue
        sid = int(st.station_i)
        vals = np.asarray(
            [
                (float(value(model.st_pch[sid, k])) - float(value(model.st_pdis[sid, k]))) / 1000.0
                for k in model.KEV
            ],
            dtype=float,
        )
        series[f"Station {sid}"] = vals
    return time_d, series


def _estimate_trip_soc_drop(ev_obj, gap_s: int) -> float:
    gap_h = max(0.0, float(gap_s) / 3600.0)
    ev_type = str(getattr(ev_obj, "type", "V2G")).upper()
    if ev_type == "V2G":
        return min(0.30, 0.08 + 0.015 * gap_h)
    if ev_type == "BUS":
        return min(0.42, 0.14 + 0.02 * gap_h)
    if ev_type == "CAR":
        return min(0.22, 0.05 + 0.012 * gap_h)
    return min(0.25, 0.08 + 0.012 * gap_h)


def _build_vehicle_soc_series(model, ev_yaml: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dispatch = getattr(model, "_dispatch", {})
    ev_map = load_evs(ev_yaml)
    event_pch = dispatch.get("event_pch_kw", {})
    event_pdis = dispatch.get("event_pdis_kw", {})
    events_cfg = list(model._meta.get("events_cfg", []))
    ev_counts = Counter(int(ev["ev_i"]) for ev in events_cfg)
    chosen: list[int] = []
    for ev_type in ["V2G", "CAR", "BUS"]:
        candidates = [
            (ev_i, cnt)
            for ev_i, cnt in ev_counts.items()
            if str(getattr(ev_map.get(int(ev_i)), "type", "")).upper() == ev_type
        ]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        if candidates:
            chosen.append(int(candidates[0][0]))

    k_count = int(model._meta["K_ev"])
    dt_h = float(model._meta["ev_step_s"]) / 3600.0
    t_d = np.arange(k_count + 1, dtype=float) * dt_h / 24.0
    series: dict[str, np.ndarray] = {}

    for ev_i in chosen:
        ev_obj = ev_map.get(int(ev_i))
        if ev_obj is None:
            continue
        soc = float(ev_obj.soc)
        vals = np.zeros(k_count + 1, dtype=float)
        vals[0] = soc
        rel = sorted(
            [ev for ev in events_cfg if int(ev["ev_i"]) == int(ev_i)],
            key=lambda e: (int(e["arrival_t"]), int(e["event_i"])),
        )
        rel_ids = [int(ev["event_i"]) for ev in rel]
        gap_drop_at_k: dict[int, float] = {}
        for idx in range(1, len(rel)):
            gap_s = max(0, int(rel[idx]["arrival_t"]) - int(rel[idx - 1]["departure_t"]))
            arr_k = min(k_count, max(0, int(rel[idx]["arrival_t"] // int(model._meta["ev_step_s"]))))
            gap_drop_at_k[arr_k] = gap_drop_at_k.get(arr_k, 0.0) + _estimate_trip_soc_drop(ev_obj, gap_s)
        for k in range(k_count):
            if k in gap_drop_at_k:
                soc = max(0.0, soc - float(gap_drop_at_k[k]))
            pch = sum(float(event_pch.get((eid, k), 0.0)) for eid in rel_ids)
            pdis = sum(float(event_pdis.get((eid, k), 0.0)) for eid in rel_ids)
            ev_obj.soc = soc
            if pch > 1e-9:
                eta = max(1e-6, float(ev_obj.charge_efficiency(pch)))
                soc = min(1.0, soc + (pch * eta * dt_h) / float(ev_obj.capacity))
            elif pdis > 1e-9:
                soc = max(0.0, soc - (pdis / max(1e-6, float(ev_obj.eta_dis)) * dt_h) / float(ev_obj.capacity))
            vals[k + 1] = soc
        series[f"{str(ev_obj.type).upper()} EV {int(ev_i)}"] = vals
    return t_d, series


def _plot_departure_soc_and_v2g(model, ev_yaml: str, out_path: Path) -> dict:
    use_times_new_roman()
    dispatch = getattr(model, "_dispatch", {})
    final_soc = dispatch.get("final_soc", {})
    event_pdis = dispatch.get("event_pdis_kw", {})
    ev_map = load_evs(ev_yaml)
    events = list(model._meta.get("events_cfg", []))

    dep_soc_v2g = []
    dep_soc_non = []
    v2g_capable = set()
    v2g_active = set()
    v2g_active_events = 0
    v2g_energy_kwh = 0.0
    active_type = Counter()
    capable_type = Counter()
    dt_h = float(model._meta["ev_step_s"]) / 3600.0

    for ev in events:
        eid = int(ev["event_i"])
        ev_i = int(ev["ev_i"])
        ev_obj = ev_map.get(ev_i)
        if ev_obj is None:
            continue
        soc = float(final_soc.get(eid, getattr(ev_obj, "soc", 0.5)))
        if bool(getattr(ev_obj, "v2g_cap", False)):
            dep_soc_v2g.append(soc)
            v2g_capable.add(ev_i)
            capable_type[str(ev_obj.type).upper()] += 1
            e_kwh = sum(float(event_pdis.get((eid, k), 0.0)) * dt_h for k in range(int(model._meta["K_ev"])))
            if e_kwh > 1e-9:
                v2g_active.add(ev_i)
                v2g_active_events += 1
                v2g_energy_kwh += e_kwh
                active_type[str(ev_obj.type).upper()] += 1
        else:
            dep_soc_non.append(soc)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
    fig.patch.set_facecolor("#faf8f4")

    ax = axes[0]
    ax.set_facecolor("#fffdf8")
    bins = np.linspace(0.0, 1.0, 22)
    ax.hist(dep_soc_non, bins=bins, alpha=0.7, color="#d9a441", edgecolor="white", label="Non-V2G events")
    ax.hist(dep_soc_v2g, bins=bins, alpha=0.7, color="#2a9d8f", edgecolor="white", label="V2G-capable events")
    ax.set_title("Departure SOC Distribution")
    ax.set_xlabel("Departure SOC")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.set_facecolor("#fffdf8")
    labels = ["V2G-capable EVs", "Active V2G EVs", "Active V2G events"]
    vals = [len(v2g_capable), len(v2g_active), v2g_active_events]
    colors = ["#52796f", "#2a9d8f", "#e76f51"]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0, f"{val}", ha="center", va="bottom", fontsize=11)
    ax.set_title("V2G Participation Statistics")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.98,
        0.92,
        f"V2G energy = {v2g_energy_kwh:.1f} kWh",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="#3a3a3a",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "v2g_capable_evs": len(v2g_capable),
        "v2g_active_evs": len(v2g_active),
        "v2g_active_events": v2g_active_events,
        "v2g_energy_kwh": v2g_energy_kwh,
        "avg_depart_soc_v2g": float(np.mean(dep_soc_v2g)) if dep_soc_v2g else 0.0,
        "avg_depart_soc_non_v2g": float(np.mean(dep_soc_non)) if dep_soc_non else 0.0,
        "v2g_capable_events_by_type": dict(capable_type),
        "v2g_active_events_by_type": dict(active_type),
    }


def _build_v2g_compare_series(model, grid: Grid) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    station_map = {int(s.station_i): s for s in grid.stations if int(getattr(s, "status", 1)) == 1}
    k_count = int(model._meta["K_ev"])
    t_d = np.arange(k_count, dtype=float) * (float(model._meta["ev_step_s"]) / 3600.0) / 24.0
    charge_only = np.zeros(k_count, dtype=float)
    with_v2g = np.zeros(k_count, dtype=float)
    v2g_dis = np.zeros(k_count, dtype=float)
    for sid in model.STN:
        st = station_map.get(int(sid))
        if st is None:
            continue
        if str(getattr(st, "type", "")).upper() not in {"RESIDENTIAL", "DEPOT"}:
            continue
        for k in model.KEV:
            pch = float(value(model.st_pch[sid, k])) / 1000.0
            pdis = float(value(model.st_pdis[sid, k])) / 1000.0
            charge_only[int(k)] += pch
            with_v2g[int(k)] += pch - pdis
            v2g_dis[int(k)] += pdis
    return t_d, {
        "Charge only baseline": charge_only,
        "Net load with V2G": with_v2g,
        "V2G discharge": v2g_dis,
    }


def _plot_v2g_compare(x_d: np.ndarray, series: dict[str, np.ndarray], out_path: Path) -> dict:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    fig.patch.set_facecolor("#faf8f4")
    ax.set_facecolor("#fffdf8")
    colors = {
        "Charge only baseline": "#d4a373",
        "Net load with V2G": "#2a9d8f",
        "V2G discharge": "#264653",
    }
    ax.plot(x_d, series["Charge only baseline"], lw=1.8, color=colors["Charge only baseline"], label="Charge only baseline")
    ax.plot(x_d, series["Net load with V2G"], lw=1.8, color=colors["Net load with V2G"], label="Net load with V2G")
    ax.fill_between(
        x_d,
        series["Net load with V2G"],
        series["Charge only baseline"],
        where=series["Charge only baseline"] >= series["Net load with V2G"],
        color="#2a9d8f",
        alpha=0.18,
    )
    ax.plot(x_d, series["V2G discharge"], lw=1.15, color=colors["V2G discharge"], alpha=0.9, label="V2G discharge")
    ticks, labels = _day_ticks(float(x_d[-1]) if len(x_d) else 7.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(float(x_d[0]) if len(x_d) else 0.0, float(x_d[-1]) if len(x_d) else 7.0)
    ax.set_title("V2G Station Net Load: With vs Without V2G")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    delta = series["Charge only baseline"] - series["Net load with V2G"]
    precharge_idx = np.where((series["Charge only baseline"] > np.quantile(series["Charge only baseline"], 0.90)) & (series["V2G discharge"] > 1e-4))[0]
    return {
        "max_v2g_relief_mw": float(np.max(delta)) if len(delta) else 0.0,
        "avg_v2g_discharge_mw": float(np.mean(series["V2G discharge"])),
        "precharge_like_slots": int(len(precharge_idx)),
    }


def _build_v2g_station_detail(model, grid: Grid) -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    residential_ids = [
        int(s.station_i)
        for s in grid.stations
        if int(getattr(s, "status", 1)) == 1 and str(getattr(s, "type", "")).upper() == "RESIDENTIAL"
    ]
    best_sid = None
    best_discharge = -1.0
    for sid in residential_ids:
        total_dis = sum(float(value(model.st_pdis[sid, k])) for k in model.KEV) / 1000.0
        if total_dis > best_discharge:
            best_discharge = total_dis
            best_sid = sid
    if best_sid is None:
        best_sid = residential_ids[0] if residential_ids else int(next(iter(model.STN)))
    t_d = np.arange(int(model._meta["K_ev"]), dtype=float) * (float(model._meta["ev_step_s"]) / 3600.0) / 24.0
    charge = np.asarray([float(value(model.st_pch[best_sid, k])) / 1000.0 for k in model.KEV], dtype=float)
    discharge = np.asarray([float(value(model.st_pdis[best_sid, k])) / 1000.0 for k in model.KEV], dtype=float)
    net = charge - discharge
    return t_d, {
        "Charge": charge,
        "Discharge": discharge,
        "Net load": net,
    }, int(best_sid)


def _plot_v2g_station_detail(x_d: np.ndarray, series: dict[str, np.ndarray], station_i: int, out_path: Path) -> dict:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.5, 5.8))
    fig.patch.set_facecolor("#faf8f4")
    ax.set_facecolor("#fffdf8")
    colors = {"Charge": "#d4a373", "Discharge": "#264653", "Net load": "#2a9d8f"}
    for name in ["Charge", "Discharge", "Net load"]:
        ax.plot(x_d, series[name], lw=1.6, color=colors[name], label=name)
    ticks, labels = _day_ticks(float(x_d[-1]) if len(x_d) else 7.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(float(x_d[0]) if len(x_d) else 0.0, float(x_d[-1]) if len(x_d) else 7.0)
    ax.set_title(f"Most Active Residential V2G Station {station_i}")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    charge = series["Charge"]
    discharge = series["Discharge"]
    local_pre = np.where((charge > np.quantile(charge, 0.9)) & (discharge > 1e-4))[0]
    return {
        "v2g_detail_station": int(station_i),
        "v2g_detail_max_charge_mw": float(np.max(charge)) if len(charge) else 0.0,
        "v2g_detail_max_discharge_mw": float(np.max(discharge)) if len(discharge) else 0.0,
        "v2g_detail_precharge_slots": int(len(local_pre)),
    }


def _plot_storage_weekly(model, out_path: Path) -> None:
    use_times_new_roman()
    time_d = np.arange(len(list(model.T)), dtype=float) * float(model._meta["delta_t"]) / 24.0
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 8.2), sharex=True)
    fig.patch.set_facecolor("#faf8f4")
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#6d597a", "#8ab17d"]

    ax = axes[0]
    ax.set_facecolor("#fffdf8")
    for idx, sid in enumerate(model.STO):
        soc = np.asarray([float(value(model.soc[sid, t])) for t in model.TSOC], dtype=float)
        t_soc = np.arange(len(soc), dtype=float) * float(model._meta["delta_t"]) / 24.0
        ax.plot(t_soc, soc, lw=1.5, color=colors[idx % len(colors)], label=f"Storage {int(sid)}")
    ax.set_title("Weekly Storage SOC")
    ax.set_ylabel("SOC")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper right")

    ax = axes[1]
    ax.set_facecolor("#fffdf8")
    base_mva = float(model._meta["baseMVA"])
    for idx, sid in enumerate(model.STO):
        net = np.asarray(
            [(float(value(model.P_dis[sid, t])) - float(value(model.P_ch[sid, t]))) * base_mva for t in model.T],
            dtype=float,
        )
        ax.plot(time_d, net, lw=1.4, color=colors[idx % len(colors)], label=f"Storage {int(sid)}")
    ticks, labels = _day_ticks(float(time_d[-1]) if len(time_d) else 7.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_title("Weekly Storage Net Output")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-yaml", default="config/IEEE33.yaml")
    parser.add_argument("--ev-yaml", default="config/ev.yaml")
    parser.add_argument("--event-yaml", default="config/event.yaml")
    parser.add_argument("--time-yaml", default="config/time.yaml")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--resolution-minutes", type=int, default=5)
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument("--mip-gap", type=float, default=5e-4)
    parser.add_argument("--ev-target-soc", type=float, default=0.9)
    parser.add_argument("--ev-depart-penalty", type=float, default=20.0)
    parser.add_argument("--v2g-reward-coeff", type=float, default=5.0)
    parser.add_argument("--summary-out", default="results/baseline_1w_summary.yaml")
    parser.add_argument("--station-out", default="results/opf_weekly_station_netload.png")
    parser.add_argument("--ev-soc-out", default="results/opf_weekly_ev_soc_3.png")
    parser.add_argument("--dep-soc-out", default="results/opf_weekly_departure_soc_v2g.png")
    parser.add_argument("--v2g-compare-out", default="results/opf_weekly_v2g_compare.png")
    parser.add_argument("--v2g-station-detail-out", default="results/opf_weekly_v2g_station_detail.png")
    parser.add_argument("--storage-out", default="results/opf_weekly_storage.png")
    args = parser.parse_args()

    grid = Grid.load_from_yaml(str(args.grid_yaml))
    cfg = LinDistFlowConfig(
        horizon_hours=int(args.hours),
        resolution_minutes=int(args.resolution_minutes),
        solver_name=str(args.solver),
        time_yaml_path=str(args.time_yaml),
        ev_yaml_path=str(args.ev_yaml),
        queue_yaml_path=str(args.event_yaml),
        ev_target_soc=float(args.ev_target_soc),
        ev_depart_penalty=float(args.ev_depart_penalty),
        v2g_reward_coeff=float(args.v2g_reward_coeff),
        solver_options={"MIPGap": float(args.mip_gap)} if str(args.solver).lower() == "gurobi" else None,
        tee=True,
    )
    model, _, _ = solve_opf(grid, cfg)

    summary = _compute_summary(model, grid, cfg)
    dep_stats = _plot_departure_soc_and_v2g(model, str(args.ev_yaml), Path(args.dep_soc_out))
    summary.update(dep_stats)

    time_d, station_series = _build_station_netload(model, grid)
    _lineplot(time_d, station_series, Path(args.station_out), "Weekly Charging Station Net Load", "Net load (MW)")

    ev_time_d, ev_soc_series = _build_vehicle_soc_series(model, str(args.ev_yaml))
    _lineplot(ev_time_d, ev_soc_series, Path(args.ev_soc_out), "Weekly SOC of 3 Representative EVs", "SOC")

    v2g_time_d, v2g_series = _build_v2g_compare_series(model, grid)
    summary.update(_plot_v2g_compare(v2g_time_d, v2g_series, Path(args.v2g_compare_out)))

    st_time_d, st_detail, st_id = _build_v2g_station_detail(model, grid)
    summary.update(_plot_v2g_station_detail(st_time_d, st_detail, st_id, Path(args.v2g_station_detail_out)))

    _plot_storage_weekly(model, Path(args.storage_out))

    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
