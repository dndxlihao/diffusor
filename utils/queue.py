import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DERs import Station
from utils import OneLineDict


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def _normalize_events(events_cfg: Any) -> List[Dict[str, Any]]:
    if not isinstance(events_cfg, list):
        raise ValueError("YAML field 'events' must be a list.")

    events: List[Dict[str, Any]] = []
    for e in events_cfg:
        if not isinstance(e, dict):
            continue
        item = dict(e)
        item["event_i"] = int(item["event_i"])
        item["ev_i"] = int(item["ev_i"])
        item["station_i"] = int(item["station_i"])
        item["arrival_t"] = int(item["arrival_t"])
        item["departure_t"] = int(item["departure_t"])
        if item.get("soc_init") is not None:
            item["soc_init"] = float(item["soc_init"])
        item["start_t"] = -1
        item["plot_i"] = -1
        item["wait_t"] = -1
        events.append(item)

    events.sort(key=lambda x: (int(x["arrival_t"]), int(x["event_i"])))
    return events


def _index_arrivals(events: List[Dict[str, Any]]) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
    by_station: Dict[int, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for e in events:
        by_station[int(e["station_i"])][int(e["arrival_t"])].append(e)

    for si in by_station:
        for t in by_station[si]:
            by_station[si][t].sort(key=lambda x: int(x["event_i"]))
    return by_station


def _key_times_for_station(events: List[Dict[str, Any]], station_i: int) -> List[int]:
    ts = set()
    for e in events:
        if int(e["station_i"]) != int(station_i):
            continue
        ts.add(int(e["arrival_t"]))
        ts.add(int(e["departure_t"]))
    return sorted(ts)


def assign_queue_fields(
    *,
    events: List[Dict[str, Any]],
    stations: List[Station],
) -> List[Dict[str, Any]]:
    normalized = _normalize_events(events)
    events_by_id = {int(e["event_i"]): e for e in normalized}
    arrivals_map = _index_arrivals(normalized)

    for st_src in stations:
        st = Station.from_config(st_src.to_config())
        si = int(st.station_i)
        times = _key_times_for_station(normalized, si)
        for tt in times:
            arrivals_at_t = arrivals_map.get(si, {}).get(int(tt), [])
            st.step(t=int(tt), arrivals=arrivals_at_t)
            for sess in st.at_plots.values():
                if int(sess.get("start_t", -1)) != int(tt):
                    continue
                eid = int(sess["event_i"])
                ev = events_by_id.get(eid)
                if ev is None:
                    continue
                if int(ev["plot_i"]) == -1:
                    ev["plot_i"] = int(sess["plot_i"])
                if int(ev["start_t"]) == -1:
                    ev["start_t"] = int(tt)

    for ev in normalized:
        if int(ev["start_t"]) >= 0:
            ev["wait_t"] = max(0, int(ev["start_t"]) - int(ev["arrival_t"]))
        else:
            ev["wait_t"] = -1

    normalized.sort(key=lambda x: (int(x["arrival_t"]), int(x["event_i"])))
    return normalized


def fill_start_and_plot(*, events_yaml: str, grid_yaml: str, out_yaml: str) -> None:
    cfg_events = _load_yaml(events_yaml)
    cfg_grid = _load_yaml(grid_yaml)
    stations_cfg = cfg_grid.get("stations", [])
    stations = [Station.from_config(s) for s in stations_cfg]
    events = assign_queue_fields(events=cfg_events.get("events", []), stations=stations)

    out_cfg: Dict[str, Any] = {"events": [OneLineDict(e) for e in events]}
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False, allow_unicode=True, width=1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=str, default="./config/event.yaml", help="Input events yaml")
    ap.add_argument("--station", type=str, default="./config/IEEE33.yaml", help="Grid yaml with stations")
    ap.add_argument("--out", type=str, default="./config/event.yaml", help="Output yaml with start_t/plot_i/wait_t")
    args = ap.parse_args()

    fill_start_and_plot(
        events_yaml=args.events,
        grid_yaml=args.station,
        out_yaml=args.out,
    )


if __name__ == "__main__":
    main()
