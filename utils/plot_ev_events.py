import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_style import use_times_new_roman

DAY_S = 86400
WEEK_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLORS = {"V2G": "#4c78a8", "CAR": "#f58518", "BUS": "#54a24b"}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_ev_type_map(ev_yaml: Path) -> dict[int, str]:
    evs = load_yaml(ev_yaml).get("evs", [])
    return {int(ev["ev_i"]): str(ev["type"]).upper() for ev in evs}


def load_events(events_yaml: Path) -> list[dict]:
    events = load_yaml(events_yaml).get("events", [])
    out = []
    for ev in events:
        item = dict(ev)
        item["event_i"] = int(item["event_i"])
        item["ev_i"] = int(item["ev_i"])
        item["station_i"] = int(item["station_i"])
        item["arrival_t"] = int(item["arrival_t"])
        item["departure_t"] = int(item["departure_t"])
        out.append(item)
    return out


def concurrent_series(events: list[dict], step_s: int = 900) -> tuple[list[float], list[int]]:
    if not events:
        return [], []
    horizon_s = max(int(ev["departure_t"]) for ev in events)
    xs = list(range(0, horizon_s + step_s, step_s))
    deltas: dict[int, int] = defaultdict(int)
    for ev in events:
        a = int(ev["arrival_t"]) // step_s
        d = (int(ev["departure_t"]) + step_s - 1) // step_s
        deltas[a] += 1
        deltas[d] -= 1
    cur = 0
    ys: list[int] = []
    for idx in range(len(xs)):
        cur += deltas.get(idx, 0)
        ys.append(cur)
    return [x / DAY_S for x in xs], ys


def plot_event_overview(events_yaml: Path, ev_yaml: Path, out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    events = load_events(events_yaml)
    ev_type = build_ev_type_map(ev_yaml)
    by_type: dict[str, list[dict]] = {"V2G": [], "CAR": [], "BUS": []}
    for ev in events:
        t = ev_type.get(int(ev["ev_i"]), "V2G")
        ev["type"] = t
        by_type.setdefault(t, []).append(ev)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    arrivals_day = [ev["arrival_t"] / DAY_S for ev in events]
    ax.hist(arrivals_day, bins=7 * 24, color=COLORS["V2G"], alpha=0.85, edgecolor="white", linewidth=0.2)
    ax.set_title("Arrival Time Distribution (Week)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 7)
    ax.set_xticks([i + 0.5 for i in range(7)], WEEK_LABELS)
    ax.grid(alpha=0.25, linestyle="--")

    ax = axes[0, 1]
    for t in ("V2G", "CAR", "BUS"):
        vals = [(ev["departure_t"] - ev["arrival_t"]) / 3600.0 for ev in by_type.get(t, [])]
        if vals:
            ax.hist(vals, bins=20, alpha=0.45, label=t, color=COLORS[t], histtype="bar", edgecolor="white", linewidth=0.35)
    ax.set_title("Session Duration by Type")
    ax.set_xlabel("Duration (h)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()

    ax = axes[1, 0]
    for t in ("V2G", "CAR", "BUS"):
        xs = [ev["arrival_t"] / DAY_S for ev in by_type.get(t, [])]
        ys = [(ev["departure_t"] - ev["arrival_t"]) / 3600.0 for ev in by_type.get(t, [])]
        if xs:
            ax.scatter(xs, ys, s=18, alpha=0.65, label=t, color=COLORS[t], edgecolors="white", linewidths=0.25)
    ax.set_title("Arrival Time vs Session Duration")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Duration (h)")
    ax.set_xlim(0, 7)
    ax.set_xticks(range(7), WEEK_LABELS)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend()

    ax = axes[1, 1]
    xs, ys = concurrent_series(events, step_s=900)
    ax.plot(xs, ys, color="#6f4e7c", lw=1.8)
    ax.set_title("Concurrent Charging Sessions")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Active events")
    ax.set_xlim(0, 7)
    ax.set_xticks(range(7), WEEK_LABELS)
    ax.grid(alpha=0.25, linestyle="--")

    fig.suptitle("Charging Event Overview", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-yaml", type=str, required=True)
    parser.add_argument("--ev-yaml", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    plot_event_overview(Path(args.events_yaml), Path(args.ev_yaml), Path(args.out))


if __name__ == "__main__":
    main()
