import argparse
import math
from numbers import Integral, Real
from pathlib import Path

import opendssdirect as dss
import yaml

from core.grid import Grid
from utils import get_series_value


class _ResultDumper(yaml.SafeDumper):
    pass


def _represent_result_float(dumper: yaml.SafeDumper, value: float):
    if abs(float(value)) < 1e-4:
        value = 0.0
    else:
        value = round(float(value), 4)
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{float(value):.4f}")


_ResultDumper.add_representer(float, _represent_result_float)


def _sanitize_tree(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_tree(v) for v in obj]
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, Integral):
        return int(obj)
    if isinstance(obj, Real):
        val = float(obj)
        return 0.0 if abs(val) < 1e-4 else round(val, 4)
    return obj


def _dump_yaml(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(_sanitize_tree(payload), f, Dumper=_ResultDumper, sort_keys=False, allow_unicode=True)


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_circuit(grid: Grid) -> tuple[int, dict[str, float]]:
    dss.Basic.ClearAll()
    slack = next(b for b in grid.buses if str(getattr(b, "type", "")).upper() == "SLACK")
    slack_bus = int(slack.bus_i)
    slack_kv = float(slack.baseKV)
    dss.Text.Command(f"New Circuit.stage2 basekV={slack_kv} pu=1.0 phases=3 bus1={slack_bus}")
    dss.Text.Command(f"Edit Vsource.Source bus1={slack_bus}.1.2.3 phases=3 basekV={slack_kv} pu=1.0")

    line_rating_mva: dict[str, float] = {}
    for br in grid.branches:
        if int(getattr(br, "status", 1)) != 1:
            continue
        name = f"LN{int(br.branch_i)}"
        dss.Text.Command(
            " ".join(
                [
                    f"New Line.{name}",
                    "phases=3",
                    f"bus1={int(br.f_bus)}.1.2.3",
                    f"bus2={int(br.t_bus)}.1.2.3",
                    f"r1={float(br.r)}",
                    f"x1={float(br.x)}",
                    f"r0={float(br.r)}",
                    f"x0={float(br.x)}",
                    "c1=0",
                    "c0=0",
                    "length=1",
                    "units=none",
                ]
            )
        )
        line_rating_mva[name.upper()] = float(getattr(br, "rateA", 0.0))

    for bus in grid.buses:
        bid = int(bus.bus_i)
        if bid == slack_bus:
            continue
        dss.Text.Command(
            " ".join(
                [
                    f"New Load.LD{bid}",
                    "phases=3",
                    f"bus1={bid}.1.2.3",
                    "conn=wye",
                    f"kV={float(bus.baseKV)}",
                    "model=1",
                    "kW=0",
                    "kvar=0",
                ]
            )
        )
    dss.Solution.Mode(0)
    return slack_bus, line_rating_mva


def _set_bus_netload(bus_id: int, p_mw: float, q_mvar: float) -> None:
    dss.Loads.Name(f"LD{int(bus_id)}")
    dss.Loads.kW(float(p_mw) * 1000.0)
    dss.Loads.kvar(float(q_mvar) * 1000.0)


def _line_loading_snapshot(line_rating_mva: dict[str, float]) -> tuple[float, str]:
    max_pct = 0.0
    max_name = ""
    for name in dss.Lines.AllNames():
        if not name:
            continue
        dss.Lines.Name(name)
        rating = float(line_rating_mva.get(str(name).upper(), 0.0))
        if rating <= 1e-9:
            continue
        powers = dss.CktElement.Powers()
        if len(powers) < 4:
            continue
        half = len(powers) // 2
        p1 = sum(float(powers[ii]) for ii in range(0, half, 2))
        q1 = sum(float(powers[ii + 1]) for ii in range(0, half, 2))
        p2 = sum(float(powers[ii]) for ii in range(half, len(powers), 2))
        q2 = sum(float(powers[ii + 1]) for ii in range(half, len(powers), 2))
        s1 = math.hypot(p1, q1) / 1000.0
        s2 = math.hypot(p2, q2) / 1000.0
        pct = max(s1, s2) / rating * 100.0
        if pct > max_pct:
            max_pct = pct
            max_name = str(name)
    return max_pct, max_name


def run_stage2(grid_yaml: Path, time_yaml: Path, replay_yaml: Path, summary_yaml: Path) -> dict:
    grid = Grid.load_from_yaml(str(grid_yaml))
    replay = _load_yaml(replay_yaml)
    summary = _load_yaml(summary_yaml)

    base_mva = float(replay.get("baseMVA", getattr(grid, "baseMVA", 1.0)))
    delta_h = float(replay["resolution_minutes"]) / 60.0
    bus_ids = [int(x) for x in replay["bus_ids"]]
    net_p = replay["bus_net_load_P_pu"]
    net_q = replay["bus_net_load_Q_pu"]
    total_steps = len(net_p)

    slack_bus, line_rating_mva = _build_circuit(grid)
    bus_base_v_ln = {
        int(b.bus_i): float(b.baseKV) * 1000.0 / math.sqrt(3.0) for b in grid.buses
    }

    corrected_grid_cost = 0.0
    loss_mwh = 0.0
    loss_cost = 0.0
    source_import_mwh = 0.0
    source_export_mwh = 0.0
    source_peak_import_mw = 0.0
    source_peak_export_mw = 0.0
    converged_steps = 0
    vmin_pu = float("inf")
    vmax_pu = 0.0
    max_line_loading_pct = 0.0
    max_loaded_line = ""

    for tt in range(total_steps):
        for jj, bid in enumerate(bus_ids):
            if int(bid) == int(slack_bus):
                continue
            _set_bus_netload(
                int(bid),
                float(net_p[tt][jj]) * base_mva,
                float(net_q[tt][jj]) * base_mva,
            )

        dss.Solution.Solve()
        if not bool(dss.Solution.Converged()):
            continue
        converged_steps += 1

        source_kw, source_kvar = dss.Circuit.TotalPower()
        source_import_mw = max(0.0, -float(source_kw) / 1000.0)
        source_export_mw = max(0.0, float(source_kw) / 1000.0)
        source_import_mwh += source_import_mw * delta_h
        source_export_mwh += source_export_mw * delta_h
        source_peak_import_mw = max(source_peak_import_mw, source_import_mw)
        source_peak_export_mw = max(source_peak_export_mw, source_export_mw)

        price = float(get_series_value(str(time_yaml), "price", t=int(tt * delta_h * 3600.0)))
        corrected_grid_cost += (source_import_mw - source_export_mw) * price * delta_h * 1000.0

        loss_w, _ = dss.Circuit.Losses()
        loss_step_mwh = float(loss_w) / 1e6 * delta_h
        loss_mwh += loss_step_mwh
        loss_cost += loss_step_mwh * price * 1000.0

        for bid in bus_ids:
            dss.Circuit.SetActiveBus(str(int(bid)))
            volts = dss.Bus.VMagAngle()
            base_v = float(bus_base_v_ln.get(int(bid), 1.0))
            mags_pu = [float(volts[ii]) / base_v for ii in range(0, len(volts), 2)] if base_v > 1e-9 else []
            if mags_pu:
                vmin_pu = min(vmin_pu, min(mags_pu))
                vmax_pu = max(vmax_pu, max(mags_pu))

        step_max_pct, step_max_line = _line_loading_snapshot(line_rating_mva)
        if step_max_pct > max_line_loading_pct:
            max_line_loading_pct = step_max_pct
            max_loaded_line = step_max_line

    opf_grid_cost = float(summary.get("grid_cost_stage1_opf", summary.get("grid_cost", 0.0)))
    opf_objective = float(summary.get("objective_stage1_opf", summary.get("objective", 0.0)))
    corrected_objective = (
        float(summary.get("gen_cost", 0.0))
        + corrected_grid_cost
        + float(summary.get("curt_cost", 0.0))
        + float(summary.get("volt_pen", 0.0))
        + float(summary.get("ev_depart_penalty", 0.0))
        - float(summary.get("storage_terminal_value", 0.0))
        - float(summary.get("v2g_reward", 0.0))
    )

    summary["grid_cost_stage1_opf"] = opf_grid_cost
    summary["objective_stage1_opf"] = opf_objective
    summary["grid_cost"] = corrected_grid_cost
    summary["objective"] = corrected_objective
    summary["stage2_scope"] = "OpenDSS replay from node-level net P/Q; source power left free"
    summary["stage2_converged_steps"] = int(converged_steps)
    summary["stage2_total_steps"] = int(total_steps)
    summary["stage2_network_loss_mwh"] = float(loss_mwh)
    summary["stage2_network_loss_cost"] = float(loss_cost)
    summary["stage2_source_import_mwh"] = float(source_import_mwh)
    summary["stage2_source_export_mwh"] = float(source_export_mwh)
    summary["stage2_source_peak_import_mw"] = float(source_peak_import_mw)
    summary["stage2_source_peak_export_mw"] = float(source_peak_export_mw)
    summary["stage2_vmin_pu"] = float(vmin_pu if vmin_pu != float("inf") else 0.0)
    summary["stage2_vmax_pu"] = float(vmax_pu)
    summary["stage2_max_line_loading_pct"] = float(max_line_loading_pct)
    summary["stage2_max_loaded_line"] = str(max_loaded_line)

    _dump_yaml(summary_yaml, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-yaml", default="config/IEEE33.yaml")
    parser.add_argument("--time-yaml", default="config/time.yaml")
    parser.add_argument("--replay-yaml", required=True)
    parser.add_argument("--summary-yaml", required=True)
    args = parser.parse_args()

    summary = run_stage2(
        grid_yaml=Path(args.grid_yaml),
        time_yaml=Path(args.time_yaml),
        replay_yaml=Path(args.replay_yaml),
        summary_yaml=Path(args.summary_yaml),
    )
    print(f"grid_cost={float(summary['grid_cost']):.4f}")
    print(f"objective={float(summary['objective']):.4f}")
    print(f"stage2_network_loss_mwh={float(summary['stage2_network_loss_mwh']):.4f}")


if __name__ == "__main__":
    main()
