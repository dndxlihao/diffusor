import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_style import use_times_new_roman

DAY_S = 86400
WEEK_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def series_xy(items: list[dict]) -> tuple[list[float], list[float]]:
    xs = [float(it["time"]) / DAY_S for it in items]
    ys = [float(it["value"]) for it in items]
    return xs, ys


def shade_weekend(ax) -> None:
    ax.axvspan(5.0, 6.0, color="#efe6cf", alpha=0.35, lw=0)
    ax.axvspan(6.0, 7.0, color="#efe6cf", alpha=0.35, lw=0)


def style_axis(ax, ylabel: str) -> None:
    ax.set_xlim(0, 7)
    ax.set_xticks(range(7), WEEK_LABELS)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_time_series(time_yaml: Path, out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_yaml(time_yaml)

    load_x, load_y = series_xy(cfg.get("load_scale", []))
    price_x, price_y = series_xy(cfg.get("price", []))

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 7.2), sharex=True)
    fig.patch.set_facecolor("#fbfaf7")

    ax = axes[0]
    ax.set_facecolor("#fffdf9")
    shade_weekend(ax)
    ax.fill_between(load_x, load_y, color="#4c78a8", alpha=0.16)
    ax.plot(load_x, load_y, color="#4c78a8", lw=1.6)
    ax.set_title("Weekly Load Scale (5-Minute Resolution)", fontsize=13)
    style_axis(ax, "Load scale")

    ax = axes[1]
    ax.set_facecolor("#fffdf9")
    shade_weekend(ax)
    ax.fill_between(price_x, price_y, color="#cc5a3d", alpha=0.18)
    ax.plot(price_x, price_y, color="#b6492d", lw=1.6)
    ax.set_title("Weekly Electricity Price (5-Minute Resolution)", fontsize=13)
    style_axis(ax, "Price")
    ax.set_xlabel("Day of week")

    fig.suptitle("Weekly Load and Price Profiles", fontsize=17, y=0.98)
    fig.text(
        0.5,
        0.94,
        "Weekdays and weekends use different base shapes, with smooth intra-day perturbations added.",
        ha="center",
        va="center",
        fontsize=11,
        color="#3c3c3c",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-yaml", type=str, default="./config/time.yaml")
    parser.add_argument("--out", type=str, default="./results/time_weekly_profiles.png")
    args = parser.parse_args()
    plot_time_series(Path(args.time_yaml), Path(args.out))


if __name__ == "__main__":
    main()
