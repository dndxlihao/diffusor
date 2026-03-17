import argparse
import sys
import types
from pathlib import Path

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
from solvers.opf.linear_distflow import LinDistFlowConfig, solve_opf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed non-EV baseline trajectories.")
    parser.add_argument("--grid-yaml", default="config/IEEE33.yaml")
    parser.add_argument("--time-yaml", default="config/time.yaml")
    parser.add_argument("--event-yaml", default="config/event.yaml")
    parser.add_argument("--ev-yaml", default="config/ev.yaml")
    parser.add_argument("--out", default="results/baseline_1w_exogenous.yaml")
    parser.add_argument("--horizon-hours", type=int, default=168)
    parser.add_argument("--resolution-minutes", type=int, default=5)
    parser.add_argument("--ev-resolution-seconds", type=int, default=300)
    parser.add_argument("--solver", default="gurobi")
    parser.add_argument("--mip-gap", type=float, default=5e-4)
    parser.add_argument("--tee", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grid = Grid.load_from_yaml(args.grid_yaml)
    cfg = LinDistFlowConfig(
        horizon_hours=args.horizon_hours,
        resolution_minutes=args.resolution_minutes,
        solver_name=args.solver,
        tee=args.tee,
        time_yaml_path=args.time_yaml,
        ev_yaml_path=args.ev_yaml,
        queue_yaml_path=args.event_yaml,
        ev_resolution_seconds=args.ev_resolution_seconds,
        solver_options={"MIPGap": float(args.mip_gap)},
    )
    model, _, report = solve_opf(grid, cfg)
    if report.get("objective") is None:
        raise RuntimeError(f"solve failed: {report.get('termination')}")

    base_mva = float(model._meta["baseMVA"])
    out = {
        "horizon_hours": int(args.horizon_hours),
        "resolution_minutes": int(args.resolution_minutes),
        "gen_ids": [int(g) for g in model.GEN],
        "ren_ids": [int(r) for r in model.REN],
        "sto_ids": [int(s) for s in model.STO],
        "gen_P_pu": [[float(value(model.Pg[g, t])) for g in model.GEN] for t in model.T],
        "gen_Q_pu": [[float(value(model.Qg[g, t])) for g in model.GEN] for t in model.T],
        "ren_Pr_pu": [[float(value(model.Pr[r, t])) for r in model.REN] for t in model.T],
        "sto_pnet_pu": [[float(value(model.P_dis[s, t]) - value(model.P_ch[s, t])) for s in model.STO] for t in model.T],
        "sto_soc": [[float(value(model.soc[s, t])) for s in model.STO] for t in model.T],
        "objective": float(report["objective"]),
        "grid_buy_mw": [float(value(model.P_buy[t])) * base_mva for t in model.T],
        "grid_sell_mw": [float(value(model.P_sell[t])) * base_mva for t in model.T],
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True, width=1000)
    print(out_path)


if __name__ == "__main__":
    main()
