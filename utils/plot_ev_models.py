import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DERs.ev import EV
from utils.plot_style import use_times_new_roman


def build_models() -> dict[str, EV]:
    return {
        "V2G": EV.from_type_template(ev_i=1, ev_type="V2G"),
        "CAR": EV.from_type_template(ev_i=2, ev_type="CAR"),
        "BUS": EV.from_type_template(ev_i=3, ev_type="BUS"),
    }


def plot_models(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_times_new_roman()
    models = build_models()
    colors = {"V2G": "#1f77b4", "CAR": "#ff7f0e", "BUS": "#2ca02c"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    for name, ev in models.items():
        curve = ev.charge_limit_curve(points=201)
        xs, ys = zip(*curve)
        y_norm = [y / max(1e-9, ev.p_ch) for y in ys]
        ax.plot(xs, y_norm, lw=2.2, color=colors[name], label=name)
    ax.set_title("Normalized Charge Limit vs SOC")
    ax.set_xlabel("SOC")
    ax.set_ylabel("P_limit / P_rated")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    for name, ev in models.items():
        curve = ev.charge_limit_curve(points=201)
        xs, ys = zip(*curve)
        ax.plot(xs, ys, lw=2.2, color=colors[name], label=name)
    ax.set_title("Actual Charge Power Limit vs SOC")
    ax.set_xlabel("SOC")
    ax.set_ylabel("Power (kW)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "ev_battery_models.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()
    plot_models(Path(args.out_dir))


if __name__ == "__main__":
    main()
