import argparse
import random
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import OneLineDict
from DERs import EV

def rand_weighted(items: List[str], weights: List[float]) -> str:
    return random.choices(items, weights=weights, k=1)[0]

PARAM_RANGES: Dict[str, Dict[str, tuple[float, float]]] = {
    "V2G": {
        "capacity": (60.0, 75.0),
        "p_ch": (6.0, 9.0),
        "p_dis": (8.0, 12.0),
        "eta_ch": (0.93, 0.96),
        "eta_dis": (0.89, 0.93),
        "soc": (0.40, 0.65),
        "soc0": (0.78, 0.84),
        "k": (12.0, 16.0),
        "min_soc": (0.25, 0.35),
        "threshold": (0.0, 0.0),
    },
    "CAR": {
        "capacity": (58.0, 75.0),
        "p_ch": (15.0, 25.0),
        "eta_ch": (0.93, 0.96),
        "eta_dis": (0.90, 0.94),
        "soc": (0.15, 0.55),
        "soc0": (0.73, 0.80),
        "k": (11.0, 15.0),
    },
    "BUS": {
        "capacity": (280.0, 380.0),
        "p_ch": (50.0, 90.0),
        "p_dis": (20.0, 35.0),
        "eta_ch": (0.93, 0.96),
        "eta_dis": (0.89, 0.93),
        "soc": (0.45, 0.70),
        "soc0": (0.68, 0.75),
        "k": (9.0, 12.0),
        "min_soc": (0.35, 0.45),
    },
}


def rand_range(bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return round(random.uniform(float(lo), float(hi)), 2)


def sample_soc(ev_type: str) -> float:
    t = str(ev_type).upper()
    bounds = PARAM_RANGES.get(t, {}).get("soc", (0.20, 0.60))
    return rand_range(bounds)


def sample_ev_params(ev_type: str, *, v2g_opt: float) -> Dict[str, Any]:
    t = str(ev_type).upper()
    prof = EV.type_profile(t)
    prm = PARAM_RANGES.get(t, {})

    out: Dict[str, Any] = {
        "capacity": rand_range(prm["capacity"]) if "capacity" in prm else float(prof["capacity"]),
        "p_ch": rand_range(prm["p_ch"]) if "p_ch" in prm else float(prof["p_ch"]),
        "p_dis": rand_range(prm["p_dis"]) if "p_dis" in prm else float(prof["p_dis"]),
        "eta_ch": rand_range(prm["eta_ch"]) if "eta_ch" in prm else float(prof["eta_ch"]),
        "eta_dis": rand_range(prm["eta_dis"]) if "eta_dis" in prm else float(prof["eta_dis"]),
        "soc": sample_soc(t),
    }

    meta = EV._deepcopy_meta(prof.get("meta"))
    charge_curve = meta.get("charge_curve")
    if isinstance(charge_curve, dict):
        if "soc0" in prm:
            charge_curve["soc0"] = rand_range(prm["soc0"])
        if "k" in prm:
            charge_curve["k"] = rand_range(prm["k"])

    if t == "CAR":
        out["p_dis"] = 0.0
    elif t in {"V2G", "BUS"}:
        v2g_meta = meta.get("v2g")
        if isinstance(v2g_meta, dict):
            if "min_soc" in prm:
                v2g_meta["min_soc"] = rand_range(prm["min_soc"])
            if t == "V2G":
                v2g_meta["opt_in"] = bool(random.random() < float(v2g_opt))
                if v2g_meta["opt_in"]:
                    v2g_meta["threshold"] = rand_range(prm["threshold"])
                else:
                    v2g_meta.pop("threshold", None)

    out["meta"] = meta
    return out

def generate_evs(n: int, *, start_id: int = 1, type_weights: Dict[str, float], v2g_opt: float = 0.60) -> List[EV]:
    types = list(type_weights.keys())
    w = [float(type_weights[k]) for k in types]
    evs: List[EV] = []

    for i in range(n):
        ev_type = rand_weighted(types, w)
        params = sample_ev_params(ev_type, v2g_opt=float(v2g_opt))
        evs.append(
            EV(
                ev_i=int(start_id + i),
                type=str(ev_type).upper(),
                capacity=float(params["capacity"]),
                p_ch=float(params["p_ch"]),
                p_dis=float(params["p_dis"]),
                eta_ch=float(params["eta_ch"]),
                eta_dis=float(params["eta_dis"]),
                soc=float(params["soc"]),
                meta=params["meta"],
            )
        )

    return evs

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of EVs")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=str, default="./config/ev.yaml", help="output yaml file")
    parser.add_argument("--start_id", type=int, default=1, help="starting EV id")
    parser.add_argument("--w_v2g", type=float, default=0.60, help="share of V2G-capable passenger cars")
    parser.add_argument("--w_car", type=float, default=0.30, help="share of fast-charging non-V2G cars")
    parser.add_argument("--w_bus", type=float, default=0.10, help="share of V2G-capable buses")
    parser.add_argument("--v2g_opt", type=float, default=0.60, help="probability of user opting in V2G for type=V2G")
    args = parser.parse_args()
    random.seed(int(args.seed))

    type_weights = {
        "V2G": float(args.w_v2g),
        "CAR": float(args.w_car),
        "BUS": float(args.w_bus),
    }

    evs = generate_evs(
        int(args.n),
        start_id=int(args.start_id),
        type_weights=type_weights,
        v2g_opt=float(args.v2g_opt),
    )

    doc = {"evs": [OneLineDict(e.to_config()) for e in evs]}
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=1000)

if __name__ == "__main__":
    main()
