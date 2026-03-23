import argparse
import sys
import types
from collections import Counter, defaultdict
from dataclasses import replace
from numbers import Integral, Real
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
from solvers.opf.linear_distflow import (
    LinDistFlowConfig,
    load_evs,
    load_queue_yaml,
    renewable_q_coeff,
    solve_opf,
    storage_q_coeff,
)
from utils import get_series_value
from utils.plot_style import use_times_new_roman

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon+"]


class _ResultDumper(yaml.SafeDumper):
    pass


def _represent_result_float(dumper: yaml.SafeDumper, value: float):
    if abs(float(value)) < 1e-4:
        value = 0.0
    else:
        value = round(float(value), 4)
    text = f"{float(value):.4f}"
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


_ResultDumper.add_representer(float, _represent_result_float)


def _sanitize_result_tree(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_result_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_result_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_result_tree(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize_result_tree(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Integral):
        return float(int(obj))
    if isinstance(obj, Real):
        val = float(obj)
        return 0.0 if abs(val) < 1e-4 else round(val, 4)
    return obj


def _sanitize_float_tree_preserve_ints(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_float_tree_preserve_ints(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_float_tree_preserve_ints(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_float_tree_preserve_ints(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_sanitize_float_tree_preserve_ints(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Integral):
        return int(obj)
    if isinstance(obj, Real):
        val = float(obj)
        return 0.0 if abs(val) < 1e-4 else round(val, 4)
    return obj


def _dump_result_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            _sanitize_result_tree(payload),
            f,
            Dumper=_ResultDumper,
            sort_keys=False,
            allow_unicode=True,
        )


def _dump_float_yaml_preserve_ints(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            _sanitize_float_tree_preserve_ints(payload),
            f,
            Dumper=_ResultDumper,
            sort_keys=False,
            allow_unicode=True,
        )


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


def _shade_mask_runs(ax, x: np.ndarray, mask: np.ndarray, *, color: str, alpha: float, label: str | None = None) -> None:
    if len(x) == 0 or len(mask) == 0:
        return
    step = float(x[1] - x[0]) if len(x) > 1 else 1.0 / 24.0
    start = None
    used_label = False
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        if start is not None and (idx == len(mask) - 1 or not mask[idx + 1]):
            end = idx
            ax.axvspan(
                float(x[start]),
                float(x[end] + step),
                color=color,
                alpha=alpha,
                label=(label if (label is not None and not used_label) else None),
            )
            used_label = True
            start = None


def _compute_summary(model, grid: Grid, cfg: LinDistFlowConfig) -> dict:
    base_mva = float(model._meta["baseMVA"])
    delta_t = float(model._meta["delta_t"])
    horizon_s = int(model._meta["K_ev"]) * int(model._meta["ev_step_s"])
    objective = None
    if hasattr(model, "Obj"):
        try:
            objective = float(value(model.Obj))
        except Exception:
            objective = None
    summary = {
        "objective": objective,
        "gen_cost": 0.0,
        "grid_cost": 0.0,
        "curt_cost": 0.0,
        "volt_pen": 0.0,
        "storage_terminal_value": 0.0,
        "ev_depart_penalty": 0.0,
        "v2g_reward": 0.0,
        "voltage_penalty_coeff": float(cfg.v_penalty),
        "storage_terminal_value_coeff": float(cfg.storage_terminal_value_coeff),
    }
    gen_map = {int(g.gen_i): g for g in grid.generators}
    ren_map = {int(r.renewable_i): r for r in grid.renewables}
    sto_map = {int(s.storage_i): s for s in grid.storages}

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
        if float(cfg.v_penalty) != 0.0:
            for bid in model.BUS:
                summary["volt_pen"] += float(cfg.v_penalty) * (
                    float(value(model.Vu[bid, tt])) + float(value(model.Vl[bid, tt]))
                )

    terminal_price = float(model._meta.get("storage_terminal_price", 0.0))
    final_t = int(model._meta["T"])
    for sid in model.STO:
        emax_mwh = float(getattr(sto_map[int(sid)], "Emax", 0.0))
        end_soc = float(value(model.soc[sid, final_t]))
        summary["storage_terminal_value"] += (
            float(cfg.storage_terminal_value_coeff) * terminal_price * emax_mwh * end_soc * 1000.0
        )

    dispatch = getattr(model, "_dispatch", {})
    dt_h_ev = float(model._meta["ev_step_s"]) / 3600.0
    if bool(model._meta.get("rolling", False)):
        for (eid, k), pdis_kw in dispatch.get("event_pdis_kw", {}).items():
            summary["v2g_reward"] += (
                float(cfg.v2g_reward_coeff)
                * float(value(model.ev_price[int(k)]))
                * float(pdis_kw)
                * dt_h_ev
            )
    else:
        for sid in model.STN:
            for k in model.KEV:
                summary["v2g_reward"] += (
                    float(cfg.v2g_reward_coeff)
                    * float(value(model.ev_price[k]))
                    * float(value(model.st_pdis[sid, k]))
                    * dt_h_ev
                )

    summary["ev_unmet_kwh"] = float(sum(dispatch.get("unmet_departure_kwh", {}).values()))
    if bool(model._meta.get("rolling", False)):
        summary["ev_depart_penalty"] = float(cfg.ev_depart_penalty) * float(summary["ev_unmet_kwh"])
    else:
        for sid in model.STN:
            for k in model.KEV_SOC:
                if float(value(model.st_due_flag[sid, k])) > 0.5:
                    summary["ev_depart_penalty"] += float(cfg.ev_depart_penalty) * float(value(model.st_short[sid, k]))
    summary["event_served"] = int(
        sum(1 for ev in model._meta.get("events_cfg", []) if int(ev.get("departure_t", 0)) <= horizon_s)
    )
    if summary["objective"] is None:
        summary["objective"] = (
            float(summary["gen_cost"])
            + float(summary["grid_cost"])
            + float(summary["curt_cost"])
            + float(summary["volt_pen"])
            + float(summary["ev_depart_penalty"])
            - float(summary["storage_terminal_value"])
            - float(summary["v2g_reward"])
        )
    return summary


def _read_solution_2d(comp, i: int, t: int) -> float:
    if isinstance(comp, dict):
        return float(comp.get((int(i), int(t)), 0.0))
    return float(value(comp[i, t]))


def _build_station_netload_matrix_pu(model, grid: Grid) -> tuple[list[int], dict[int, int], np.ndarray]:
    meta = getattr(model, "_meta", {})
    base_mva = float(meta["baseMVA"])
    delta_t = float(meta["delta_t"])
    total_t = int(meta["T"])
    k_ev = int(meta.get("K_ev", 0))
    ev_step_s = int(meta.get("ev_step_s", int(round(delta_t * 3600.0))))
    station_ids = [int(x) for x in meta.get("station_ids", [])]
    station_bus = {int(k): int(v) for k, v in meta.get("station_bus", {}).items()}
    out = np.zeros((total_t, len(station_ids)), dtype=float)
    if not station_ids:
        return station_ids, station_bus, out

    opf_step_s = int(round(delta_t * 3600.0))
    if total_t == k_ev and opf_step_s == ev_step_s:
        for tt in range(total_t):
            for jj, sid in enumerate(station_ids):
                pch_kw = _read_solution_2d(model.st_pch, sid, tt)
                pdis_kw = _read_solution_2d(model.st_pdis, sid, tt)
                out[tt, jj] = ((pch_kw - pdis_kw) / 1000.0) / base_mva
        return station_ids, station_bus, out

    def _sec_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
        lo = max(int(a0), int(b0))
        hi = min(int(a1), int(b1))
        return max(0, hi - lo)

    for tt in range(total_t):
        a0 = int(tt * opf_step_s)
        a1 = int((tt + 1) * opf_step_s)
        k0 = max(0, a0 // ev_step_s)
        k1 = min(k_ev - 1, (a1 - 1) // ev_step_s) if k_ev > 0 else -1
        for kk in range(k0, k1 + 1):
            b0 = int(kk * ev_step_s)
            b1 = int((kk + 1) * ev_step_s)
            overlap = _sec_overlap(a0, a1, b0, b1)
            if overlap <= 0:
                continue
            w = float(overlap) / float(opf_step_s)
            for jj, sid in enumerate(station_ids):
                pch_kw = _read_solution_2d(model.st_pch, sid, kk)
                pdis_kw = _read_solution_2d(model.st_pdis, sid, kk)
                out[tt, jj] += w * ((pch_kw - pdis_kw) / 1000.0) / base_mva
    return station_ids, station_bus, out


def _build_bus_netload_matrices(model, grid: Grid, station_net_pu: np.ndarray, station_ids: list[int], station_bus: dict[int, int]) -> dict[str, object]:
    meta = getattr(model, "_meta", {})
    T = int(meta["T"])
    delta_t = float(meta["delta_t"])
    step_s = int(round(delta_t * 3600.0))
    base_mva = float(meta["baseMVA"])
    gen_ids = [int(x) for x in meta.get("gen_ids", [])]
    ren_ids = [int(x) for x in meta.get("ren_ids", [])]
    sto_ids = [int(x) for x in meta.get("sto_ids", [])]
    bus_ids = [int(b.bus_i) for b in grid.buses]
    bus_idx = {bid: jj for jj, bid in enumerate(bus_ids)}

    base_p = np.zeros((T, len(bus_ids)), dtype=float)
    base_q = np.zeros((T, len(bus_ids)), dtype=float)
    gen_p = np.zeros((T, len(bus_ids)), dtype=float)
    gen_q = np.zeros((T, len(bus_ids)), dtype=float)
    ren_p = np.zeros((T, len(bus_ids)), dtype=float)
    ren_q = np.zeros((T, len(bus_ids)), dtype=float)
    sto_p = np.zeros((T, len(bus_ids)), dtype=float)
    sto_q = np.zeros((T, len(bus_ids)), dtype=float)
    stn_p = np.zeros((T, len(bus_ids)), dtype=float)
    stn_q = np.zeros((T, len(bus_ids)), dtype=float)

    gen_bus = {int(g.gen_i): int(g.bus_i) for g in grid.generators}
    ren_bus = {int(r.renewable_i): int(r.bus_i) for r in grid.renewables}
    ren_qc = {int(r.renewable_i): float(renewable_q_coeff(r)) for r in grid.renewables}
    sto_bus = {int(s.storage_i): int(s.bus_i) for s in grid.storages}
    sto_qc = {int(s.storage_i): float(storage_q_coeff(s)) for s in grid.storages}

    for tt in range(T):
        t_sec = int(tt * step_s)
        for bus in grid.buses:
            bid = int(bus.bus_i)
            jj = bus_idx[bid]
            Pd_pu, Qd_pu = grid.bus_PQ_pu(bus, t=t_sec)
            base_p[tt, jj] = float(Pd_pu)
            base_q[tt, jj] = float(Qd_pu)

        for gid in gen_ids:
            bid = gen_bus[int(gid)]
            jj = bus_idx[bid]
            gen_p[tt, jj] += _read_solution_2d(model.Pg, gid, tt)
            gen_q[tt, jj] += _read_solution_2d(model.Qg, gid, tt)

        for rid in ren_ids:
            bid = ren_bus[int(rid)]
            jj = bus_idx[bid]
            p_pu = _read_solution_2d(model.Pr, rid, tt)
            ren_p[tt, jj] += p_pu
            ren_q[tt, jj] += float(ren_qc[int(rid)]) * p_pu

        for sid in sto_ids:
            bid = sto_bus[int(sid)]
            jj = bus_idx[bid]
            p_inj_pu = _read_solution_2d(model.P_dis, sid, tt) - _read_solution_2d(model.P_ch, sid, tt)
            sto_p[tt, jj] += p_inj_pu
            sto_q[tt, jj] += float(sto_qc[int(sid)]) * p_inj_pu

        for jj, sid in enumerate(station_ids):
            bid = station_bus[int(sid)]
            kk = bus_idx[bid]
            stn_p[tt, kk] += float(station_net_pu[tt, jj])

    net_load_p = base_p + stn_p - gen_p - ren_p - sto_p
    net_load_q = base_q + stn_q - gen_q - ren_q - sto_q

    return {
        "bus_ids": bus_ids,
        "station_ids": [int(x) for x in station_ids],
        "station_bus": {int(k): int(v) for k, v in station_bus.items()},
        "bus_base_load_P_pu": base_p.tolist(),
        "bus_base_load_Q_pu": base_q.tolist(),
        "bus_gen_P_pu": gen_p.tolist(),
        "bus_gen_Q_pu": gen_q.tolist(),
        "bus_ren_P_pu": ren_p.tolist(),
        "bus_ren_Q_pu": ren_q.tolist(),
        "bus_sto_P_pu": sto_p.tolist(),
        "bus_sto_Q_pu": sto_q.tolist(),
        "bus_station_P_pu": stn_p.tolist(),
        "bus_station_Q_pu": stn_q.tolist(),
        "bus_net_load_P_pu": net_load_p.tolist(),
        "bus_net_load_Q_pu": net_load_q.tolist(),
        "station_net_P_pu": station_net_pu.tolist(),
        "station_net_Q_pu": np.zeros_like(station_net_pu).tolist(),
        "sign_convention": {
            "bus_net_load_P_pu": "positive means net demand, negative means net injection",
            "bus_net_load_Q_pu": "positive means net reactive demand, negative means net reactive injection",
            "component_PQ_arrays": "generator/renewable/storage arrays are injections; base_load/station arrays are demands",
        },
        "baseMVA": base_mva,
    }


def _export_exogenous_sequences(model, grid: Grid, out_path: Path) -> None:
    meta = getattr(model, "_meta", {})
    T = int(meta["T"])
    delta_t = float(meta["delta_t"])
    gen_ids = [int(x) for x in meta.get("gen_ids", [])]
    ren_ids = [int(x) for x in meta.get("ren_ids", [])]
    sto_ids = [int(x) for x in meta.get("sto_ids", [])]

    gen_P_pu = [[_read_solution_2d(model.Pg, gid, tt) for gid in gen_ids] for tt in range(T)]
    gen_Q_pu = [[_read_solution_2d(model.Qg, gid, tt) for gid in gen_ids] for tt in range(T)]
    ren_Pr_pu = [[_read_solution_2d(model.Pr, rid, tt) for rid in ren_ids] for tt in range(T)]

    sto_pnet_pu: list[list[float]] = []
    sto_soc: list[list[float]] = []
    if sto_ids:
        for tt in range(T):
            sto_pnet_pu.append([
                _read_solution_2d(model.P_ch, sid, tt) - _read_solution_2d(model.P_dis, sid, tt) for sid in sto_ids
            ])
            if isinstance(model.soc, dict):
                sto_soc.append([float(model.soc.get((int(sid), int(tt + 1)), 0.0)) for sid in sto_ids])
            else:
                sto_soc.append([float(value(model.soc[sid, tt + 1])) for sid in sto_ids])

    payload = {
        "horizon_hours": float(T * delta_t),
        "resolution_minutes": float(delta_t * 60.0),
        "gen_ids": gen_ids,
        "ren_ids": ren_ids,
        "sto_ids": sto_ids,
        "gen_P_pu": gen_P_pu,
        "gen_Q_pu": gen_Q_pu,
        "ren_Pr_pu": ren_Pr_pu,
        "sto_pnet_pu": sto_pnet_pu,
        "sto_soc": sto_soc,
    }
    _dump_float_yaml_preserve_ints(out_path, payload)


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
    event_soc_init = dispatch.get("event_soc_init", {})
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
        arrival_soc_at_k: dict[int, float] = {}
        gap_drop_at_k: dict[int, float] = {}
        if bool(model._meta.get("rolling", False)) and event_soc_init:
            for ev in rel:
                eid = int(ev["event_i"])
                if eid not in event_soc_init:
                    continue
                arr_k = min(k_count, max(0, int(ev["arrival_t"] // int(model._meta["ev_step_s"]))))
                arrival_soc_at_k[arr_k] = float(event_soc_init[eid])
        else:
            for idx in range(1, len(rel)):
                gap_s = max(0, int(rel[idx]["arrival_t"]) - int(rel[idx - 1]["departure_t"]))
                arr_k = min(k_count, max(0, int(rel[idx]["arrival_t"] // int(model._meta["ev_step_s"]))))
                gap_drop_at_k[arr_k] = gap_drop_at_k.get(arr_k, 0.0) + _estimate_trip_soc_drop(ev_obj, gap_s)
        for k in range(k_count):
            if k in arrival_soc_at_k:
                soc = float(arrival_soc_at_k[k])
            elif k in gap_drop_at_k:
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
    dep_soc_v2g_active = []
    dep_soc_non = []
    v2g_capable = set()
    v2g_active = set()
    v2g_capable_events = 0
    v2g_active_events = 0
    v2g_energy_kwh = 0.0
    active_type = Counter()
    capable_type = Counter()
    dt_h = float(model._meta["ev_step_s"]) / 3600.0
    horizon_s = int(model._meta["K_ev"]) * int(model._meta["ev_step_s"])

    for ev in events:
        if int(ev.get("departure_t", 0)) > horizon_s:
            continue
        eid = int(ev["event_i"])
        ev_i = int(ev["ev_i"])
        ev_obj = ev_map.get(ev_i)
        if ev_obj is None:
            continue
        soc = float(final_soc.get(eid, getattr(ev_obj, "soc", 0.5)))
        if bool(getattr(ev_obj, "v2g_cap", False)):
            dep_soc_v2g.append(soc)
            v2g_capable.add(ev_i)
            v2g_capable_events += 1
            capable_type[str(ev_obj.type).upper()] += 1
            e_kwh = sum(float(event_pdis.get((eid, k), 0.0)) * dt_h for k in range(int(model._meta["K_ev"])))
            if e_kwh > 1e-9:
                v2g_active.add(ev_i)
                v2g_active_events += 1
                v2g_energy_kwh += e_kwh
                dep_soc_v2g_active.append(soc)
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
    ax.hist(
        dep_soc_v2g_active,
        bins=bins,
        alpha=0.55,
        color="#264653",
        edgecolor="white",
        label="Actual V2G events",
    )
    ax.set_title("Departure SOC Distribution by Event Group")
    ax.set_xlabel("Departure SOC")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.set_facecolor("#fffdf8")
    labels = ["Capable EVs", "Active EVs", "Capable events", "Actual V2G events"]
    vals = [len(v2g_capable), len(v2g_active), v2g_capable_events, v2g_active_events]
    colors = ["#52796f", "#2a9d8f", "#7c9885", "#e76f51"]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0, f"{val}", ha="center", va="bottom", fontsize=11)
    ax.set_title("V2G Participation: EVs vs Events")
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
        "v2g_capable_events": v2g_capable_events,
        "v2g_active_events": v2g_active_events,
        "v2g_energy_kwh": v2g_energy_kwh,
        "avg_depart_soc_v2g": float(np.mean(dep_soc_v2g)) if dep_soc_v2g else 0.0,
        "avg_depart_soc_v2g_active": float(np.mean(dep_soc_v2g_active)) if dep_soc_v2g_active else 0.0,
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
    if bool(model._meta.get("rolling", False)):
        events_cfg = {int(ev["event_i"]): dict(ev) for ev in model._meta.get("events_cfg", [])}
        dispatch = getattr(model, "_dispatch", {})
        for (eid, k), pch_kw in dispatch.get("event_pch_kw", {}).items():
            ev = events_cfg.get(int(eid))
            if ev is None:
                continue
            st = station_map.get(int(ev["station_i"]))
            if st is None or str(getattr(st, "type", "")).upper() not in {"RESIDENTIAL", "DEPOT"}:
                continue
            charge_only[int(k)] += float(pch_kw) / 1000.0
            with_v2g[int(k)] += float(pch_kw) / 1000.0
        for (eid, k), pdis_kw in dispatch.get("event_pdis_kw", {}).items():
            ev = events_cfg.get(int(eid))
            if ev is None:
                continue
            st = station_map.get(int(ev["station_i"]))
            if st is None or str(getattr(st, "type", "")).upper() not in {"RESIDENTIAL", "DEPOT"}:
                continue
            with_v2g[int(k)] -= float(pdis_kw) / 1000.0
            v2g_dis[int(k)] += float(pdis_kw) / 1000.0
    else:
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
    fig.subplots_adjust(bottom=0.16, hspace=0.18)
    colors = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51", "#6d597a", "#8ab17d", "#457b9d", "#c1666b"]

    ax = axes[0]
    ax.set_facecolor("#fffdf8")
    for idx, sid in enumerate(model.STO):
        soc = np.asarray([float(value(model.soc[sid, t])) for t in model.TSOC], dtype=float)
        t_soc = np.arange(len(soc), dtype=float) * float(model._meta["delta_t"]) / 24.0
        ax.plot(t_soc, soc, lw=1.7, color=colors[idx % len(colors)], label=f"Storage {int(sid)}")
    price = np.asarray([float(value(model.price[t])) for t in model.T], dtype=float)
    price_ax = ax.twinx()
    low_thr = float(np.quantile(price, 0.25)) if len(price) else 0.0
    high_thr = float(np.quantile(price, 0.75)) if len(price) else 0.0
    _shade_mask_runs(ax, time_d, price <= low_thr, color="#b7d7f0", alpha=0.10, label="Low-price periods")
    _shade_mask_runs(ax, time_d, price >= high_thr, color="#f6c2c2", alpha=0.10, label="High-price periods")
    price_ax.plot(time_d, price, lw=1.8, color="#a44a3f", linestyle="--", alpha=0.92, label="Price")
    price_ax.set_ylabel("Price")
    price_ax.spines["top"].set_visible(False)
    price_ax.spines["left"].set_visible(False)
    price_ax.tick_params(axis="y", colors="#111111")
    ax.set_title("Weekly Storage SOC with Price")
    ax.set_ylabel("SOC")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    p_handles, p_labels = price_ax.get_legend_handles_labels()
    ax.legend(handles + p_handles, labels + p_labels, frameon=False, ncol=3, loc="upper right")

    ax = axes[1]
    ax.set_facecolor("#fffdf8")
    base_mva = float(model._meta["baseMVA"])
    for idx, sid in enumerate(model.STO):
        net = np.asarray(
            [(float(value(model.P_dis[sid, t])) - float(value(model.P_ch[sid, t]))) * base_mva for t in model.T],
            dtype=float,
        )
        ax.plot(time_d, net, lw=1.5, color=colors[idx % len(colors)], label=f"Storage {int(sid)}")
    ax.axhline(0.0, color="#666666", lw=0.9)
    ticks, labels = _day_ticks(float(time_d[-1]) if len(time_d) else 7.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_title("Weekly Storage Net Output")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Power (MW)")
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=max(1, len(list(model.STO))), loc="upper center", bbox_to_anchor=(0.5, -0.20))

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _ceil_div(a: int, b: int) -> int:
    return int((int(a) + int(b) - 1) // int(b))


def _event_soc_after_slot(ev_obj, soc: float, pch_kw: float, pdis_kw: float, dt_h: float) -> float:
    cap = max(1e-6, float(ev_obj.capacity))
    soc_next = float(soc)
    if float(pch_kw) > 1e-9:
        ev_obj.soc = soc_next
        eta = max(1e-6, float(ev_obj.charge_efficiency(float(pch_kw))))
        soc_next = min(1.0, soc_next + (float(pch_kw) * eta * dt_h) / cap)
    if float(pdis_kw) > 1e-9:
        soc_next = max(
            float(ev_obj.v2g_minsoc()),
            soc_next - (float(pdis_kw) / max(1e-6, float(ev_obj.eta_dis)) * dt_h) / cap,
        )
    return float(soc_next)


def _build_station_req_cum(events: list[dict], ev_map: dict[int, object], station_ids: list[int], target_soc: float, ev_step_s: int, k_ev: int) -> dict[tuple[int, int], float]:
    req = {(int(sid), int(k)): 0.0 for sid in station_ids for k in range(int(k_ev) + 1)}
    target_soc_eff = min(1.0, float(target_soc))
    for ev in events:
        sid = int(ev["station_i"])
        dep = int(ev["departure_t"])
        if sid not in station_ids or dep <= 0:
            continue
        ev_obj = ev_map.get(int(ev["ev_i"]))
        if ev_obj is None:
            continue
        soc0 = float(ev.get("soc_init", getattr(ev_obj, "soc", 0.5)))
        req_kwh = max(0.0, (target_soc_eff - soc0) * float(ev_obj.capacity))
        dep_k = min(int(k_ev), max(0, _ceil_div(dep, ev_step_s)))
        req[(sid, dep_k)] += req_kwh
    for sid in station_ids:
        cum = 0.0
        for k in range(int(k_ev) + 1):
            cum += float(req[(int(sid), int(k))])
            req[(int(sid), int(k))] = cum
    return req


def _build_rolling_events(
    full_events: list[dict],
    ev_map: dict[int, object],
    now_s: int,
    horizon_s: int,
    current_event_soc: dict[int, float],
    event_soc_init_override: dict[int, float],
) -> list[dict]:
    sub_events: list[dict] = []
    end_s = int(now_s + horizon_s)
    for ev in full_events:
        eid = int(ev["event_i"])
        ev_i = int(ev["ev_i"])
        arr_abs = int(ev["arrival_t"])
        dep_abs = int(ev["departure_t"])
        start_abs = int(ev.get("start_t", arr_abs))
        if dep_abs <= now_s or arr_abs >= end_s:
            continue
        ev_obj = ev_map.get(ev_i)
        if ev_obj is None:
            continue
        new_ev = dict(ev)
        new_ev["arrival_t"] = max(0, arr_abs - now_s)
        new_ev["start_t"] = max(0, start_abs - now_s)
        new_ev["departure_t"] = dep_abs - now_s
        if arr_abs <= now_s:
            new_ev["soc_init"] = float(
                current_event_soc.get(
                    eid,
                    event_soc_init_override.get(eid, ev.get("soc_init", getattr(ev_obj, "soc", 0.5))),
                )
            )
        else:
            new_ev["soc_init"] = float(event_soc_init_override.get(eid, ev.get("soc_init", getattr(ev_obj, "soc", 0.5))))
        sub_events.append(new_ev)
    return sub_events


class RollingNoPrimalSolutionError(RuntimeError):
    def __init__(self, *, window_idx: int, total_windows: int, t0: int) -> None:
        self.window_idx = int(window_idx)
        self.total_windows = int(total_windows)
        self.t0 = int(t0)
        super().__init__(
            f"rolling window {self.window_idx}/{self.total_windows} at t={self.t0} "
            "has no primal solution; consider increasing time limit or reducing lookahead"
        )


def solve_opf_rolling(grid: Grid, cfg: LinDistFlowConfig, *, lookahead_hours: int, roll_interval_minutes: int):
    stations, full_events = load_queue_yaml(cfg.queue_yaml_path)
    if not stations:
        stations = [dict(st.to_config()) for st in grid.stations if int(getattr(st, "status", 1)) == 1]
    ev_map = load_evs(cfg.ev_yaml_path)

    base_mva = float(grid.baseMVA)
    bus_ids = [int(b.bus_i) for b in grid.buses]
    gen_ids = [int(g.gen_i) for g in grid.generators if int(getattr(g, "status", 1)) == 1]
    ren_ids = [int(r.renewable_i) for r in grid.renewables if int(getattr(r, "status", 1)) == 1]
    sto_ids = [int(s.storage_i) for s in grid.storages if int(getattr(s, "status", 1)) == 1]
    station_ids = sorted(int(s["station_i"]) for s in stations)
    station_bus = {int(s["station_i"]): int(s["bus_i"]) for s in stations}
    station_event_ids = {int(sid): [] for sid in station_ids}
    for ev in full_events:
        sid = int(ev["station_i"])
        if sid in station_event_ids:
            station_event_ids[sid].append(int(ev["event_i"]))

    total_t = int(cfg.horizon_steps_override) if cfg.horizon_steps_override is not None else int(cfg.horizon_hours * 60 / cfg.resolution_minutes)
    delta_t = float(cfg.resolution_minutes) / 60.0
    step_s = int(round(delta_t * 3600.0))
    ev_step_s = int(cfg.ev_resolution_seconds)
    if step_s % ev_step_s != 0:
        raise ValueError("rolling baseline requires resolution_minutes to be an integer multiple of ev_resolution_seconds")
    ksub = int(step_s // ev_step_s)
    total_k_ev = int(total_t * ksub)

    lookahead_steps = max(1, int(round(float(lookahead_hours) * 60.0 / float(cfg.resolution_minutes))))
    roll_steps = max(1, int(round(float(roll_interval_minutes) / float(cfg.resolution_minutes))))

    current_storage_soc = {int(s.storage_i): float(getattr(s, "soc", 0.5)) for s in grid.storages if int(getattr(s, "status", 1)) == 1}
    current_event_soc: dict[int, float] = {}
    event_soc_init_override: dict[int, float] = {}
    event_soc_init: dict[int, float] = {}

    Pg = defaultdict(float)
    Qg = defaultdict(float)
    Pr = defaultdict(float)
    curt = defaultdict(float)
    Vu = defaultdict(float)
    Vl = defaultdict(float)
    P_buy = defaultdict(float)
    P_sell = defaultdict(float)
    P_ch = defaultdict(float)
    P_dis = defaultdict(float)
    soc = defaultdict(float)
    st_pch = defaultdict(float)
    st_pdis = defaultdict(float)
    st_short = defaultdict(float)
    st_due_flag = defaultdict(int)
    ev_price = defaultdict(float)
    dispatch_ch = {}
    dispatch_dis = {}
    final_soc = {}
    unmet_departure_kwh = {}

    for sid, soc0 in current_storage_soc.items():
        soc[(int(sid), 0)] = float(soc0)

    full_event_by_id = {int(ev["event_i"]): dict(ev) for ev in full_events}
    vehicle_events: dict[int, list[dict]] = defaultdict(list)
    next_event_by_id: dict[int, int] = {}
    for ev in full_events:
        vehicle_events[int(ev["ev_i"])].append(dict(ev))
    for ev_i, evs in vehicle_events.items():
        evs.sort(key=lambda item: (int(item["arrival_t"]), int(item["event_i"])))
        first_ev = evs[0]
        first_eid = int(first_ev["event_i"])
        first_ev_obj = ev_map.get(int(ev_i))
        if first_ev_obj is not None:
            event_soc_init_override[first_eid] = float(first_ev.get("soc_init", getattr(first_ev_obj, "soc", 0.5)))
        for idx in range(len(evs) - 1):
            next_event_by_id[int(evs[idx]["event_i"])] = int(evs[idx + 1]["event_i"])
    target_soc_eff = min(1.0, float(cfg.ev_target_soc))

    total_windows = int((total_t + roll_steps - 1) // roll_steps)
    for window_idx, t0 in enumerate(range(0, total_t, roll_steps), start=1):
        now_s = int(t0 * step_s)
        solve_steps = min(lookahead_steps, total_t - t0)
        exec_steps = min(roll_steps, total_t - t0)
        print(
            f"[rolling] window {window_idx}/{total_windows} start "
            f"(t0={t0}, solve_steps={solve_steps}, exec_steps={exec_steps})",
            flush=True,
        )
        sub_events = _build_rolling_events(
            full_events,
            ev_map,
            now_s,
            int(solve_steps * step_s),
            current_event_soc,
            event_soc_init_override,
        )
        sub_cfg = replace(
            cfg,
            tee=bool(cfg.tee),
            time_offset_seconds=now_s,
            horizon_steps_override=int(solve_steps),
            events_override=sub_events,
            storage_soc0_override=dict(current_storage_soc),
        )
        sub_model, _, sub_report = solve_opf(grid, sub_cfg)
        if sub_report.get("objective") is None:
            raise RollingNoPrimalSolutionError(
                window_idx=window_idx,
                total_windows=total_windows,
                t0=t0,
            )
        print(
            f"[rolling] window {window_idx}/{total_windows} done "
            f"(objective={float(sub_report['objective']):.4f})",
            flush=True,
        )
        sub_dispatch = getattr(sub_model, "_dispatch", {})
        sub_pch = sub_dispatch.get("event_pch_kw", {})
        sub_pdis = sub_dispatch.get("event_pdis_kw", {})
        exec_k = int(exec_steps * ksub)

        for u in range(exec_steps):
            gt = int(t0 + u)
            for gid in sub_model.GEN:
                Pg[(int(gid), gt)] = float(value(sub_model.Pg[gid, u]))
                Qg[(int(gid), gt)] = float(value(sub_model.Qg[gid, u]))
            for rid in sub_model.REN:
                Pr[(int(rid), gt)] = float(value(sub_model.Pr[rid, u]))
                curt[(int(rid), gt)] = float(value(sub_model.curt[rid, u]))
            for bid in sub_model.BUS:
                Vu[(int(bid), gt)] = float(value(sub_model.Vu[bid, u]))
                Vl[(int(bid), gt)] = float(value(sub_model.Vl[bid, u]))
            P_buy[gt] = float(value(sub_model.P_buy[u]))
            P_sell[gt] = float(value(sub_model.P_sell[u]))
            for sid in sub_model.STO:
                P_ch[(int(sid), gt)] = float(value(sub_model.P_ch[sid, u]))
                P_dis[(int(sid), gt)] = float(value(sub_model.P_dis[sid, u]))
                soc[(int(sid), gt)] = float(value(sub_model.soc[sid, u]))
                soc[(int(sid), gt + 1)] = float(value(sub_model.soc[sid, u + 1]))
            for ksub_i in range(ksub):
                gk = int(gt * ksub + ksub_i)
                lk = int(u * ksub + ksub_i)
                for sid in sub_model.STN:
                    st_pch[(int(sid), gk)] = float(value(sub_model.st_pch[sid, lk]))
                    st_pdis[(int(sid), gk)] = float(value(sub_model.st_pdis[sid, lk]))
                ev_price[gk] = float(value(sub_model.ev_price[lk]))

        sub_state = {}
        for ev in sub_events:
            eid = int(ev["event_i"])
            ev_i = int(ev["ev_i"])
            ev_obj = ev_map.get(ev_i)
            if ev_obj is None:
                continue
            sub_state[eid] = float(ev.get("soc_init", getattr(ev_obj, "soc", 0.5)))

        finalized_in_window = set()
        for lk in range(exec_k):
            gk = int(t0 * ksub + lk)
            slot_start_abs = int(now_s + lk * ev_step_s)
            slot_end_abs = int(now_s + (lk + 1) * ev_step_s)
            for eid, soc_now in list(sub_state.items()):
                ev = full_event_by_id[eid]
                ev_i = int(ev["ev_i"])
                ev_obj = ev_map.get(ev_i)
                if ev_obj is None:
                    continue
                arr_abs = int(ev["arrival_t"])
                dep_abs = int(ev["departure_t"])
                if eid not in event_soc_init and arr_abs < slot_end_abs and dep_abs > slot_start_abs:
                    event_soc_init[eid] = float(soc_now)
                pch_kw = float(sub_pch.get((eid, lk), 0.0))
                pdis_kw = float(sub_pdis.get((eid, lk), 0.0))
                if pch_kw > 1e-9:
                    dispatch_ch[(int(eid), gk)] = pch_kw
                if pdis_kw > 1e-9:
                    dispatch_dis[(int(eid), gk)] = pdis_kw
                sub_state[eid] = _event_soc_after_slot(ev_obj, soc_now, pch_kw, pdis_kw, float(ev_step_s) / 3600.0)

                if eid not in finalized_in_window and dep_abs <= slot_end_abs:
                    final_soc[eid] = float(sub_state[eid])
                    unmet = max(0.0, (target_soc_eff - float(sub_state[eid])) * float(ev_obj.capacity))
                    unmet_departure_kwh[eid] = float(unmet)
                    dep_k = min(total_k_ev, max(0, _ceil_div(dep_abs, ev_step_s)))
                    sid = int(ev["station_i"])
                    st_due_flag[(sid, dep_k)] = 1
                    st_short[(sid, dep_k)] += float(unmet)
                    next_eid = next_event_by_id.get(eid)
                    if next_eid is not None:
                        next_ev = full_event_by_id[int(next_eid)]
                        gap_s = max(0, int(next_ev["arrival_t"]) - dep_abs)
                        next_soc = max(0.0, float(sub_state[eid]) - _estimate_trip_soc_drop(ev_obj, gap_s))
                        event_soc_init_override[int(next_eid)] = float(next_soc)
                    finalized_in_window.add(eid)

        interval_end_abs = int(now_s + exec_steps * step_s)
        next_event_soc = {}
        for eid, soc_now in sub_state.items():
            ev = full_event_by_id[eid]
            arr_abs = int(ev["arrival_t"])
            dep_abs = int(ev["departure_t"])
            if eid in finalized_in_window:
                continue
            if arr_abs <= interval_end_abs and dep_abs > interval_end_abs:
                next_event_soc[eid] = float(soc_now)
        current_event_soc = next_event_soc
        current_storage_soc = {int(sid): float(value(sub_model.soc[sid, exec_steps])) for sid in sub_model.STO}

    for ev in full_events:
        eid = int(ev["event_i"])
        if eid in final_soc:
            continue
        ev_obj = ev_map.get(int(ev["ev_i"]))
        if ev_obj is None:
            continue
        final_soc[eid] = float(current_event_soc.get(eid, ev.get("soc_init", getattr(ev_obj, "soc", 0.5))))

    station_req_cum_kwh = _build_station_req_cum(full_events, ev_map, station_ids, cfg.ev_target_soc, ev_step_s, total_k_ev)

    replay = types.SimpleNamespace()
    replay.T = list(range(int(total_t)))
    replay.TSOC = list(range(int(total_t) + 1))
    replay.KEV = list(range(int(total_k_ev)))
    replay.KEV_SOC = list(range(int(total_k_ev) + 1))
    replay.BUS = list(bus_ids)
    replay.GEN = list(gen_ids)
    replay.REN = list(ren_ids)
    replay.STO = list(sto_ids)
    replay.STN = list(station_ids)
    replay.price = {tt: float(get_series_value(cfg.time_yaml_path, "price", t=int(tt * step_s))) for tt in replay.T}
    replay.ev_price = {k: float(get_series_value(cfg.time_yaml_path, "price", t=int(k * ev_step_s))) for k in replay.KEV}
    replay.Pg = Pg
    replay.Qg = Qg
    replay.Pr = Pr
    replay.curt = curt
    replay.Vu = Vu
    replay.Vl = Vl
    replay.P_buy = P_buy
    replay.P_sell = P_sell
    replay.P_ch = P_ch
    replay.P_dis = P_dis
    replay.soc = soc
    replay.st_pch = st_pch
    replay.st_pdis = st_pdis
    replay.st_short = st_short
    replay.st_due_flag = st_due_flag
    replay._dispatch = {
        "event_pch_kw": dispatch_ch,
        "event_pdis_kw": dispatch_dis,
        "event_soc_init": event_soc_init,
        "final_soc": final_soc,
        "unmet_departure_kwh": unmet_departure_kwh,
    }
    replay._meta = {
        "baseMVA": base_mva,
        "delta_t": float(delta_t),
        "T": int(total_t),
        "bus_ids": list(bus_ids),
        "gen_ids": list(gen_ids),
        "ren_ids": list(ren_ids),
        "sto_ids": list(sto_ids),
        "station_ids": list(station_ids),
        "K_ev": int(total_k_ev),
        "ev_step_s": int(ev_step_s),
        "station_bus": {int(sid): int(station_bus[sid]) for sid in station_ids},
        "events_cfg": [dict(ev) for ev in full_events],
        "station_event_ids": {int(sid): list(station_event_ids.get(int(sid), [])) for sid in station_ids},
        "station_req_cum_kwh": station_req_cum_kwh,
        "v_penalty": float(cfg.v_penalty),
        "ev_depart_penalty": float(cfg.ev_depart_penalty),
        "v2g_reward_coeff": float(cfg.v2g_reward_coeff),
        "ev_target_soc": float(cfg.ev_target_soc),
        "storage_terminal_value_coeff": float(cfg.storage_terminal_value_coeff),
        "storage_terminal_price": float(replay.price[int(total_t - 1)]) if int(total_t) > 0 else 0.0,
        "time_yaml_path": str(cfg.time_yaml_path),
        "queue_yaml_path": str(cfg.queue_yaml_path),
        "rolling": True,
        "rolling_lookahead_hours": int(lookahead_hours),
        "rolling_interval_minutes": int(roll_interval_minutes),
    }
    return replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-yaml", default="config/IEEE33.yaml")
    parser.add_argument("--ev-yaml", default="config/ev.yaml")
    parser.add_argument("--event-yaml", default="config/event.yaml")
    parser.add_argument("--time-yaml", default="config/time.yaml")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--resolution-minutes", type=int, default=5)
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument("--tee", action="store_true")
    parser.add_argument("--mip-gap", type=float, default=5e-4)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--rolling-lookahead-hours", type=int, default=24)
    parser.add_argument("--rolling-interval-minutes", type=int, default=60)
    parser.add_argument("--ev-target-soc", type=float, default=0.9)
    parser.add_argument("--ev-depart-penalty", type=float, default=20.0)
    parser.add_argument("--v2g-reward-coeff", type=float, default=5.0)
    parser.add_argument("--storage-terminal-value-coeff", type=float, default=1.0)
    parser.add_argument("--summary-out", default="results/baseline_1w_summary.yaml")
    parser.add_argument("--exogenous-out", default=None)
    parser.add_argument("--station-out", default="results/opf_weekly_station_netload.png")
    parser.add_argument("--ev-soc-out", default="results/opf_weekly_ev_soc_3.png")
    parser.add_argument("--dep-soc-out", default="results/opf_weekly_departure_soc_v2g.png")
    parser.add_argument("--v2g-compare-out", default="results/opf_weekly_v2g_compare.png")
    parser.add_argument("--v2g-station-detail-out", default="results/opf_weekly_v2g_station_detail.png")
    parser.add_argument("--storage-out", default="results/opf_weekly_storage.png")
    args = parser.parse_args()

    grid = Grid.load_from_yaml(str(args.grid_yaml))
    solver_options = {} if str(args.solver).lower() == "gurobi" else None
    if solver_options is not None and float(args.mip_gap) > 0.0:
        solver_options["MIPGap"] = float(args.mip_gap)
    if solver_options is not None and float(args.time_limit) > 0.0:
        solver_options["TimeLimit"] = float(args.time_limit)
    if solver_options == {}:
        solver_options = None
    cfg = LinDistFlowConfig(
        horizon_hours=int(args.hours),
        resolution_minutes=int(args.resolution_minutes),
        solver_name=str(args.solver),
        tee=bool(args.tee),
        time_yaml_path=str(args.time_yaml),
        ev_yaml_path=str(args.ev_yaml),
        queue_yaml_path=str(args.event_yaml),
        ev_target_soc=float(args.ev_target_soc),
        ev_depart_penalty=float(args.ev_depart_penalty),
        v2g_reward_coeff=float(args.v2g_reward_coeff),
        storage_terminal_value_coeff=float(args.storage_terminal_value_coeff),
        solver_options=solver_options,
    )
    if bool(args.rolling):
        model = solve_opf_rolling(
            grid,
            cfg,
            lookahead_hours=int(args.rolling_lookahead_hours),
            roll_interval_minutes=int(args.rolling_interval_minutes),
        )
    else:
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

    _plot_storage_weekly(model, Path(args.storage_out))
    detail_out = Path(args.v2g_station_detail_out)
    if detail_out.exists():
        detail_out.unlink()
    _dump_result_yaml(Path(args.summary_out), summary)
    if args.exogenous_out:
        _export_exogenous_sequences(model, grid, Path(args.exogenous_out))


if __name__ == "__main__":
    main()
