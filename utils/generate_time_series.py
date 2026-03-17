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
    raw = [rng.gauss(0.0, scale) for _ in range(n + 4)]
    out: list[float] = []
    for i in range(n):
        out.append(
            0.10 * raw[i]
            + 0.20 * raw[i + 1]
            + 0.40 * raw[i + 2]
            + 0.20 * raw[i + 3]
            + 0.10 * raw[i + 4]
        )
    return out


def load_profile(hour: float, day: int, noise: float) -> float:
    weekend = day >= 5
    if not weekend:
        daily_bias = [0.000, 0.012, 0.020, 0.015, 0.006][day]
        val = (
            0.84
            + 0.10 * gauss(hour, 8.2, 1.5)
            + 0.07 * gauss(hour, 13.2, 2.4)
            + 0.23 * gauss(hour, 19.5, 2.0)
            + 0.03 * gauss(hour, 22.8, 1.2)
            - 0.03 * gauss(hour, 3.8, 2.0)
            + daily_bias
        )
    else:
        daily_bias = [0.015, 0.000][day - 5]
        val = (
            0.86
            + 0.04 * gauss(hour, 9.5, 2.0)
            + 0.15 * gauss(hour, 14.3, 2.8)
            + 0.18 * gauss(hour, 20.0, 2.4)
            - 0.025 * gauss(hour, 4.2, 2.2)
            + daily_bias
        )
    ripple = 0.010 * math.sin(2.0 * math.pi * hour / 24.0 * 3.0 + 0.7 * day)
    val = val + ripple + noise
    return max(0.78, min(1.22, val))


def price_profile(hour: float, day: int, noise: float) -> float:
    weekend = day >= 5
    if not weekend:
        daily_bias = [0.00, 0.02, 0.05, 0.03, 0.01][day]
        val = (
            0.52
            + 0.24 * gauss(hour, 7.8, 1.8)
            + 0.46 * gauss(hour, 11.8, 2.6)
            + 1.28 * gauss(hour, 18.9, 2.4)
            + 0.26 * gauss(hour, 21.8, 1.5)
            - 0.08 * gauss(hour, 3.2, 2.2)
            + daily_bias
        )
    else:
        daily_bias = [-0.03, -0.05][day - 5]
        val = (
            0.50
            + 0.14 * gauss(hour, 9.6, 2.0)
            + 0.34 * gauss(hour, 14.2, 2.8)
            + 0.88 * gauss(hour, 19.2, 2.7)
            + 0.14 * gauss(hour, 22.0, 1.6)
            - 0.06 * gauss(hour, 3.5, 2.0)
            + daily_bias
        )
    ripple = 0.025 * math.sin(2.0 * math.pi * hour / 24.0 * 2.0 + 0.4 * day)
    val = val + ripple + noise
    return max(0.42, min(2.35, val))


def build_weekly_series(*, step_s: int, seed: int) -> dict:
    rng = random.Random(int(seed))
    n = WEEK_S // int(step_s)
    load_noise = smooth_noise(n, scale=0.008, rng=rng)
    price_noise = smooth_noise(n, scale=0.020, rng=rng)

    load_scale = []
    price = []
    for idx in range(n):
        t_sec = idx * int(step_s)
        day = t_sec // DAY_S
        hour = (t_sec % DAY_S) / 3600.0
        load_scale.append(
            OneLineDict(
                {
                    "time": int(t_sec),
                    "value": round(load_profile(hour, int(day), load_noise[idx]), 5),
                }
            )
        )
        price.append(
            OneLineDict(
                {
                    "time": int(t_sec),
                    "value": round(price_profile(hour, int(day), price_noise[idx]), 4),
                }
            )
        )

    return {
        "meta": OneLineDict(
            {
                "resolution_s": int(step_s),
                "horizon_s": WEEK_S,
                "days": 7,
                "seed": int(seed),
            }
        ),
        "load_scale": load_scale,
        "price": price,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./config/time.yaml")
    parser.add_argument("--step-s", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    doc = build_weekly_series(step_s=int(args.step_s), seed=int(args.seed))
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=1000)


if __name__ == "__main__":
    main()
