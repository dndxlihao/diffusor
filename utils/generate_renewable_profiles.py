import argparse
import math
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import OneLineDict

DAY_S = 86400
WEEK_S = 7 * DAY_S


def gauss(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def smooth_noise(n: int, *, scale: float, rng: random.Random) -> list[float]:
    raw = [rng.gauss(0.0, scale) for _ in range(n + 6)]
    out: list[float] = []
    for i in range(n):
        out.append(
            0.08 * raw[i]
            + 0.12 * raw[i + 1]
            + 0.18 * raw[i + 2]
            + 0.24 * raw[i + 3]
            + 0.18 * raw[i + 4]
            + 0.12 * raw[i + 5]
            + 0.08 * raw[i + 6]
        )
    return out


def pv_profile(hour: float, day: int, *, peak: float, shift: float, cloud: float, noise: float) -> float:
    weekend = day >= 5
    daylight = (
        0.95 * gauss(hour, 13.0 + shift, 2.9)
        + 0.42 * gauss(hour, 10.3 + 0.5 * shift, 1.9)
        + 0.36 * gauss(hour, 15.6 + 0.4 * shift, 2.0)
    )
    sunrise_cut = 1.0 / (1.0 + math.exp(-(hour - 6.1) * 2.2))
    sunset_cut = 1.0 / (1.0 + math.exp((hour - 19.2) * 2.0))
    base = daylight * sunrise_cut * sunset_cut
    day_bias = [0.00, 0.03, -0.02, 0.04, -0.01, -0.05, -0.02][day]
    weekend_soft = -0.02 if weekend else 0.0
    cloud_belt = cloud * (
        0.18 * gauss(hour, 11.8, 1.1)
        + 0.25 * gauss(hour, 14.3, 1.0)
    )
    # Add stronger daylight-only fluctuations to emulate passing clouds.
    cloud_event_1 = (0.06 + 0.06 * cloud) * gauss(hour, 10.6 + 0.35 * day + 0.8 * shift, 0.45)
    cloud_event_2 = (0.08 + 0.10 * cloud) * gauss(hour, 13.8 - 0.25 * day + 0.6 * shift, 0.55)
    cloud_event_3 = (0.05 + 0.05 * cloud) * gauss(hour, 16.0 + 0.15 * day - 0.4 * shift, 0.40)
    daylight_weight = max(0.0, min(1.0, base / 0.9))
    ripple = daylight_weight * (
        0.035 * math.sin(2.0 * math.pi * (hour - 6.0) / 12.0 + 0.5 * day)
        + 0.022 * math.sin(2.0 * math.pi * hour / 2.2 + 0.8 * day + 3.0 * shift)
    )
    val = peak * base * (
        1.0
        + day_bias
        + weekend_soft
        - cloud_belt
        - cloud_event_1
        - cloud_event_2
        - cloud_event_3
        + ripple
        + 1.35 * noise
    )
    return max(0.0, min(1.05, val))


def wind_profile(hour: float, day: int, *, base: float, amp: float, phase: float, noise: float) -> float:
    weekend = day >= 5
    daily = (
        base
        + amp * math.sin(2.0 * math.pi * hour / 24.0 + phase)
        + 0.10 * math.sin(2.0 * math.pi * hour / 8.0 + 0.4 * day)
        + 0.06 * math.cos(2.0 * math.pi * hour / 5.0 + phase)
    )
    overnight = 0.14 * gauss(hour, 2.5, 2.0) + 0.10 * gauss(hour, 22.8, 1.8)
    midday_dip = 0.08 * gauss(hour, 13.8, 2.5)
    day_bias = [0.03, -0.04, 0.02, 0.05, -0.01, -0.03, 0.01][day]
    weekend_bias = -0.02 if weekend else 0.0
    val = daily + overnight - midday_dip + day_bias + weekend_bias + noise
    return max(0.12, min(0.98, val))


def build_profiles(*, step_s: int, seed: int) -> dict:
    rng = random.Random(int(seed))
    n = WEEK_S // int(step_s)
    specs = {
        1: {"type": "PV", "peak": 0.96, "shift": -0.25, "cloud": 0.88, "noise": 0.028},
        2: {"type": "PV", "peak": 0.90, "shift": 0.35, "cloud": 0.62, "noise": 0.023},
        3: {"type": "WIND", "base": 0.50, "amp": 0.15, "phase": 0.30, "noise": 0.045},
        4: {"type": "WIND", "base": 0.58, "amp": 0.11, "phase": 1.10, "noise": 0.040},
    }

    out = {}
    for rid, spec in specs.items():
        noise = smooth_noise(n, scale=float(spec["noise"]), rng=rng)
        series = []
        for idx in range(n):
            t_sec = idx * int(step_s)
            day = t_sec // DAY_S
            hour = (t_sec % DAY_S) / 3600.0
            if spec["type"] == "PV":
                value = pv_profile(
                    hour,
                    int(day),
                    peak=float(spec["peak"]),
                    shift=float(spec["shift"]),
                    cloud=float(spec["cloud"]),
                    noise=noise[idx],
                )
            else:
                value = wind_profile(
                    hour,
                    int(day),
                    base=float(spec["base"]),
                    amp=float(spec["amp"]),
                    phase=float(spec["phase"]),
                    noise=noise[idx],
                )
            series.append(OneLineDict({"time": int(t_sec), "value": round(value, 4)}))
        out[int(rid)] = OneLineDict(
            {
                "resolution_s": int(step_s),
                "horizon_s": WEEK_S,
                "type": str(spec["type"]),
                "series": series,
            }
        )

    return {"renewable_profiles": out}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./config/renewable_profiles.yaml")
    parser.add_argument("--step-s", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    doc = build_profiles(step_s=int(args.step_s), seed=int(args.seed))
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=1000)


if __name__ == "__main__":
    main()
