import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.plot_style import use_times_new_roman

DAY_S = 86400
WEEK_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
COLORS = {"V2G": "#4c78a8", "CAR": "#f58518", "BUS": "#54a24b"}
ACTIVE_COLOR = "#5b4b8a"
QUEUE_COLOR = "#d1495b"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_ev_type_map(ev_yaml: Path) -> dict[int, str]:
    evs = load_yaml(ev_yaml).get("evs", [])
    return {int(ev["ev_i"]): str(ev["type"]).upper() for ev in evs}


def load_events(path: Path) -> list[dict]:
    events = load_yaml(path).get("events", [])
    out: list[dict] = []
    for ev in events:
        item = dict(ev)
        item["event_i"] = int(item["event_i"])
        item["ev_i"] = int(item["ev_i"])
        item["station_i"] = int(item["station_i"])
        item["arrival_t"] = int(item["arrival_t"])
        item["departure_t"] = int(item["departure_t"])
        item["start_t"] = int(item.get("start_t", -1))
        item["plot_i"] = int(item.get("plot_i", -1))
        item["wait_t"] = int(item.get("wait_t", -1))
        out.append(item)
    return out


def queue_depth_series(events: list[dict], step_s: int = 900) -> tuple[list[float], list[int]]:
    if not events:
        return [], []
    horizon_s = max(int(ev["departure_t"]) for ev in events)
    deltas: dict[int, int] = defaultdict(int)
    for ev in events:
        a = int(ev["arrival_t"]) // step_s
        deltas[a] += 1
        if int(ev["start_t"]) >= 0:
            end_wait = int(ev["start_t"]) // step_s
        else:
            end_wait = int(ev["departure_t"]) // step_s
        deltas[end_wait] -= 1

    cur = 0
    xs: list[float] = []
    ys: list[int] = []
    for idx in range(0, (horizon_s // step_s) + 2):
        cur += deltas.get(idx, 0)
        xs.append((idx * step_s) / DAY_S)
        ys.append(max(cur, 0))
    return xs, ys


def active_sessions_series(events: list[dict], step_s: int = 900) -> tuple[list[float], list[int]]:
    if not events:
        return [], []
    horizon_s = max(int(ev["departure_t"]) for ev in events)
    deltas: dict[int, int] = defaultdict(int)
    for ev in events:
        start_t = int(ev.get("start_t", -1))
        if start_t < 0:
            continue
        a = start_t // step_s
        d = (int(ev["departure_t"]) + step_s - 1) // step_s
        deltas[a] += 1
        deltas[d] -= 1

    cur = 0
    xs: list[float] = []
    ys: list[int] = []
    for idx in range(0, (horizon_s // step_s) + 2):
        cur += deltas.get(idx, 0)
        xs.append((idx * step_s) / DAY_S)
        ys.append(max(cur, 0))
    return xs, ys


def plot_queue_overview(events_yaml: Path, ev_yaml: Path, out_path: Path) -> None:
    use_times_new_roman()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    events = load_events(events_yaml)
    ev_type = build_ev_type_map(ev_yaml)
    for ev in events:
        ev["type"] = ev_type.get(int(ev["ev_i"]), "V2G")

    total_events = len(events)
    unserved = sum(int(ev.get("start_t", -1)) < 0 for ev in events)
    unserved_ratio = (100.0 * unserved / total_events) if total_events else 0.0
    queued = [int(ev["wait_t"]) for ev in events if int(ev.get("wait_t", -1)) > 0]
    avg_wait = (sum(queued) / len(queued) / 60.0) if queued else 0.0
    max_wait = (max(queued) / 60.0) if queued else 0.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.6),
        gridspec_kw={"width_ratios": [1.0, 1.12]},
    )
    fig.patch.set_facecolor("#fbfaf7")

    ax = axes[0]
    ax.set_facecolor("#fffdf9")
    bins = np.arange(0, 361, 8)
    for t in ("V2G", "CAR", "BUS"):
        vals = [int(ev["wait_t"]) / 60.0 for ev in events if ev["type"] == t and int(ev["wait_t"]) > 0]
        if vals:
            ax.hist(
                vals,
                bins=bins,
                alpha=0.35,
                label=t,
                color=COLORS[t],
                edgecolor=COLORS[t],
                linewidth=0.6,
            )
    ax.set_title("Waiting Time Distribution", fontsize=13)
    ax.set_xlabel("Waiting time (min)")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 360)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ax.set_facecolor("#fffdf9")
    xs_active, ys_active = active_sessions_series(events, step_s=900)
    xs_queue, ys_queue = queue_depth_series(events, step_s=900)
    ax.fill_between(xs_active, ys_active, color=ACTIVE_COLOR, alpha=0.14)
    ax.plot(xs_active, ys_active, color=ACTIVE_COLOR, lw=2.0, label="Active sessions")
    ax.fill_between(xs_queue, ys_queue, color=QUEUE_COLOR, alpha=0.18)
    ax.plot(xs_queue, ys_queue, color=QUEUE_COLOR, lw=1.55, label="Queue length")
    ax.set_title("Queue Dynamics Over Week", fontsize=13)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 7)
    ax.set_xticks(range(7), WEEK_LABELS)
    ax.grid(alpha=0.18, linestyle="--", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")

    fig.suptitle("Queueing Overview", fontsize=17, y=0.98)
    fig.text(
        0.5,
        0.93,
        f"events={total_events}   unscheduled={unserved} ({unserved_ratio:.2f}%)   avg wait={avg_wait:.1f} min   max wait={max_wait:.1f} min",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#3c3c3c",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-yaml", type=str, required=True)
    parser.add_argument("--ev-yaml", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    plot_queue_overview(Path(args.events_yaml), Path(args.ev_yaml), Path(args.out))


if __name__ == "__main__":
    main()
