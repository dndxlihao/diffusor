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
COLORS = {1: "#d29c2c", 2: "#d96c31", 3: "#2b83ba", 4: "#5aa469"}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def shade_weekend(ax) -> None:
    ax.axvspan(5.0, 6.0, color="#efe6cf", alpha=0.35, lw=0)
    ax.axvspan(6.0, 7.0, color="#efe6cf", alpha=0.35, lw=0)


def extract_series(profile: dict) -> tuple[list[float], list[float]]:
    items = profile.get("series", [])
    xs = [float(item["time"]) / DAY_S for item in items]
    ys = [float(item["value"]) for item in items]
    return xs, ys


def style_axis(ax, ylabel: str) -> None:
    ax.set_xlim(0, 7)
    ax.set_xticks(range(7), WEEK_LABELS)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_profiles(profile_yaml: Path, out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(profile_yaml)
    profiles = cfg.get("renewable_profiles", {})

    fig, axes = plt.subplots(2, 1, figsize=(14.5, 7.2), sharex=True)
    fig.patch.set_facecolor("#fbfaf7")

    pv_ax = axes[0]
    pv_ax.set_facecolor("#fffdf9")
    shade_weekend(pv_ax)
    for rid in (1, 2):
        prof = profiles.get(rid) or profiles.get(str(rid))
        if not isinstance(prof, dict):
            continue
        xs, ys = extract_series(prof)
        pv_ax.plot(xs, ys, lw=1.4, color=COLORS[rid], label=f"PV {rid}")
    pv_ax.set_title("PV Available Power Profiles (per-unit of Pmax)", fontsize=13)
    style_axis(pv_ax, "Pav / Pmax")
    pv_ax.legend(frameon=False, ncol=2, loc="upper right")

    wd_ax = axes[1]
    wd_ax.set_facecolor("#fffdf9")
    shade_weekend(wd_ax)
    for rid in (3, 4):
        prof = profiles.get(rid) or profiles.get(str(rid))
        if not isinstance(prof, dict):
            continue
        xs, ys = extract_series(prof)
        wd_ax.plot(xs, ys, lw=1.4, color=COLORS[rid], label=f"WIND {rid}")
    wd_ax.set_title("Wind Available Power Profiles (per-unit of Pmax)", fontsize=13)
    style_axis(wd_ax, "Pav / Pmax")
    wd_ax.set_xlabel("Day of week")
    wd_ax.legend(frameon=False, ncol=2, loc="upper right")

    fig.suptitle("Weekly Renewable Available Power Profiles", fontsize=17, y=0.98)
    fig.text(
        0.5,
        0.94,
        "Each unit uses a distinct 5-minute profile with weekday/weekend variation and smooth stochastic perturbations.",
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
    parser.add_argument("--profile-yaml", type=str, default="./config/renewable_profiles.yaml")
    parser.add_argument("--out", type=str, default="./results/renewable_weekly_profiles.png")
    args = parser.parse_args()
    plot_profiles(Path(args.profile_yaml), Path(args.out))


if __name__ == "__main__":
    main()
