import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_style import use_times_new_roman


COLORS = {
    "V2G": "#1f77b4",
    "CAR": "#ff7f0e",
    "BUS": "#2ca02c",
}


def load_evs(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return list(doc.get("evs", []))


def plot_dataset(evs: list[dict], out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter(str(ev["type"]).upper() for ev in evs)
    by_type = {
        t: [ev for ev in evs if str(ev["type"]).upper() == t]
        for t in ("V2G", "CAR", "BUS")
    }

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))

    ax = axes[0, 0]
    xs = ["V2G", "CAR", "BUS"]
    ys = [counts.get(x, 0) for x in xs]
    bars = ax.bar(xs, ys, color=[COLORS[x] for x in xs], alpha=0.9)
    ax.set_title("EV Type Counts")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2.0, y + max(2, 0.01 * max(ys, default=1)), str(y), ha="center", va="bottom", fontsize=8)

    ax = axes[0, 1]
    for t in xs:
        vals = [float(ev["capacity"]) for ev in by_type[t]]
        if vals:
            ax.hist(vals, bins=16, alpha=0.6, label=t, color=COLORS[t])
    ax.set_title("Battery Capacity by Type")
    ax.set_xlabel("Capacity (kWh)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    for t in xs:
        vals = [float(ev["p_ch"]) for ev in by_type[t]]
        if vals:
            ax.hist(vals, bins=16, alpha=0.6, label=t, color=COLORS[t])
    ax.set_title("Charging Power by Type")
    ax.set_xlabel("p_ch (kW)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    vals = by_type["V2G"]
    ax.scatter([float(ev["p_ch"]) for ev in vals], [float(ev["p_dis"]) for ev in vals], s=10, alpha=0.75, color=COLORS["V2G"], edgecolors="none")
    ax.set_title("Power Map: V2G")
    ax.set_xlabel("p_ch (kW)")
    ax.set_ylabel("p_dis (kW)")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    vals = by_type["CAR"]
    ax.scatter([float(ev["p_ch"]) for ev in vals], [float(ev["p_dis"]) for ev in vals], s=10, alpha=0.75, color=COLORS["CAR"], edgecolors="none")
    ax.set_title("Power Map: CAR")
    ax.set_xlabel("p_ch (kW)")
    ax.set_ylabel("p_dis (kW)")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    vals = by_type["BUS"]
    ax.scatter([float(ev["p_ch"]) for ev in vals], [float(ev["p_dis"]) for ev in vals], s=12, alpha=0.75, color=COLORS["BUS"], edgecolors="none")
    ax.set_title("Power Map: BUS")
    ax.set_xlabel("p_ch (kW)")
    ax.set_ylabel("p_dis (kW)")
    ax.grid(alpha=0.25)

    fig.suptitle("Generated EV Dataset Overview", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ev-yaml", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    plot_dataset(load_evs(Path(args.ev_yaml)), Path(args.out))


if __name__ == "__main__":
    main()
