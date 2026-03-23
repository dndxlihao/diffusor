import argparse
import random
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import OneLineDict
from utils.queue import assign_queue_fields
from DERs import EV, Station, ChargingEvent

DAY_S = 86400  # seconds per day


def is_weekend(day: int) -> bool:
    return int(day) % 7 in (5, 6)

def sample_time(mu_s: int, sigma_s: int) -> int:
    x = int(random.gauss(float(mu_s), float(sigma_s)))
    while x < 0:
        x += DAY_S
    while x >= DAY_S:
        x -= DAY_S
    return int(x)

# 1. Residential(V2G): weekday evening dominant, weekend broader and later
def sample_residential(day: int) -> Tuple[int, int]:
    if is_weekend(day):
        arr_sec = sample_time(mu_s=int(17.0 * 3600), sigma_s=int(3.0 * 3600))
        dur_s = random.randint(int(3.0 * 3600), int(8.0 * 3600))
    else:
        arr_sec = sample_time(mu_s=int(19.2 * 3600), sigma_s=int(1.2 * 3600))
        dur_s = random.randint(int(3.0 * 3600), int(8.0 * 3600))
    arrival_t = int(day) * DAY_S + int(arr_sec)
    departure_t = int(arrival_t) + int(dur_s)
    return int(arrival_t), int(departure_t)

# 2. Bus depot(V2G): weekday more regular, weekend lower and slightly broader
def sample_depot(day: int) -> Tuple[int, int]:
    if is_weekend(day):
        arr_sec = sample_time(mu_s=int(20.5 * 3600), sigma_s=int(2.0 * 3600))
        dur_s = random.randint(int(4.0 * 3600), int(8.0 * 3600))
    else:
        arr_sec = sample_time(mu_s=int(22.0 * 3600), sigma_s=int(1.0 * 3600))
        dur_s = random.randint(int(4.0 * 3600), int(8.0 * 3600))
    arrival_t = int(day) * DAY_S + int(arr_sec)
    departure_t = int(arrival_t) + int(dur_s)
    return int(arrival_t), int(departure_t)

# 3. Highway(no V2G): weekday commute/business peaks, weekend leisure peaks
def sample_highway(day: int) -> Tuple[int, int]:
    if is_weekend(day):
        if random.random() < 0.55:
            arr_sec = sample_time(mu_s=int(14.0 * 3600), sigma_s=int(3.0 * 3600))
        else:
            arr_sec = sample_time(mu_s=int(19.0 * 3600), sigma_s=int(2.5 * 3600))
        dur_s = random.randint(int(1.0 * 3600), int(3.4 * 3600))
    else:
        r = random.random()
        if r < 0.35:
            arr_sec = sample_time(mu_s=int(8.5 * 3600), sigma_s=int(1.5 * 3600))
        elif r < 0.80:
            arr_sec = sample_time(mu_s=int(13.0 * 3600), sigma_s=int(2.2 * 3600))
        else:
            arr_sec = sample_time(mu_s=int(18.5 * 3600), sigma_s=int(1.8 * 3600))
        dur_s = random.randint(int(1.0 * 3600), int(3.5 * 3600))
    arrival_t = int(day) * DAY_S + int(arr_sec)
    departure_t = int(arrival_t) + int(dur_s)

    return int(arrival_t), int(departure_t)


def day_rate_multiplier(ev_type: str, day: int) -> float:
    weekend = is_weekend(day)
    t = str(ev_type).upper()
    if t == "V2G":
        return 0.80 if weekend else 1.00
    if t == "CAR":
        return 1.20 if weekend else 1.00
    if t == "BUS":
        return 0.65 if weekend else 1.00
    return 1.00

def evs_by_type(evs: List[EV]) -> Dict[str, List[EV]]:
    out: Dict[str, List[EV]] = {}
    for e in evs:
        k = str(getattr(e, "type", "V2G")).upper()
        out.setdefault(k, []).append(e)
    return out

def stations_by_type(stations: List[Station]) -> Dict[str, List[Station]]:
    out: Dict[str, List[Station]] = {}
    for s in stations:
        k = str(getattr(s, "type", "RESIDENTIAL")).upper()
        out.setdefault(k, []).append(s)
    return out

def choose_station(ev_type: str, stations_type: Dict[str, List[Station]]) -> int:
    t = str(ev_type).upper()
    if t == "V2G":
        candidates = stations_type.get("RESIDENTIAL", [])
    elif t == "CAR":
        candidates = stations_type.get("HIGHWAY", [])
    elif t == "BUS":
        candidates = stations_type.get("DEPOT", [])
    else:
        candidates = stations_type.get("RESIDENTIAL", [])

    if not candidates:
        raise RuntimeError(f"No station candidates for {t}")
    return int(random.choice(candidates).station_i)


def _sample_event_window(ev_type: str, day: int) -> Tuple[int, int]:
    if ev_type == "V2G":
        return sample_residential(day)
    if ev_type == "BUS":
        return sample_depot(day)
    if ev_type == "CAR":
        return sample_highway(day)
    return sample_highway(day)


def _nonoverlap_schedule(
    sampled: List[Tuple[int, int, int]],
    *,
    next_free_t: int,
) -> List[Tuple[int, int, int]]:
    scheduled: List[Tuple[int, int, int]] = []
    cursor_t = int(next_free_t)
    for arrival_t, departure_t, station_i in sorted(sampled, key=lambda x: (int(x[0]), int(x[1]), int(x[2]))):
        dur_s = max(0, int(departure_t) - int(arrival_t))
        start_t = max(int(arrival_t), int(cursor_t))
        end_t = int(start_t) + int(dur_s)
        scheduled.append((int(start_t), int(end_t), int(station_i)))
        cursor_t = int(end_t)
    return scheduled

def generate_events(
    *,
    days: int,
    evs: List[EV],
    stations: List[Station],
    events_per_ev_per_day: Dict[str, float],
    start_id: int = 1,
) -> List[ChargingEvent]:
    ev_by_type = evs_by_type(evs)
    station_by_type = stations_by_type(stations)

    events: List[ChargingEvent] = []
    eid = int(start_id)
    ev_next_free_t: Dict[int, int] = {}

    for day in range(int(days)):
        for t, ev_list in ev_by_type.items():
            rate = float(events_per_ev_per_day.get(t, 0.0)) * float(day_rate_multiplier(t, day))
            if rate <= 0.0:
                continue

            for ev in ev_list:
                if rate <= 1.0:
                    n = 1 if random.random() < rate else 0
                else:
                    n = int(rate)
                    if random.random() < (rate - n):
                        n += 1

                ev_type = str(getattr(ev, "type", "V2G")).upper()
                sampled_windows: List[Tuple[int, int, int]] = []
                for _ in range(int(n)):
                    arrival_t, departure_t = _sample_event_window(ev_type, day)
                    station_i = choose_station(ev_type, station_by_type)
                    sampled_windows.append((int(arrival_t), int(departure_t), int(station_i)))

                next_free_t = int(ev_next_free_t.get(int(ev.ev_i), 0))
                scheduled_windows = _nonoverlap_schedule(sampled_windows, next_free_t=next_free_t)

                for arrival_t, departure_t, station_i in scheduled_windows:
                    events.append(
                        ChargingEvent(
                            event_i=int(eid),
                            ev_i=int(ev.ev_i),
                            station_i=int(station_i),
                            arrival_t=int(arrival_t),
                            departure_t=int(departure_t),
                        )
                    )
                    eid += 1
                    ev_next_free_t[int(ev.ev_i)] = int(departure_t)

    events.sort(key=lambda e: (int(e.arrival_t), int(e.event_i)))
    return events

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ev", type=str, default="./config/ev.yaml", help="EV yaml path")
    parser.add_argument("--station", type=str, default="./config/IEEE33.yaml", help="station yaml path")
    parser.add_argument("--out", type=str, default="./config/event.yaml", help="output events yaml path")
    parser.add_argument("--days", type=int, default=1, help="number of days to simulate")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--start_id", type=int, default=1)
    parser.add_argument("--rate_v2g", type=float, default=1.0, help="events per V2G per day (residential)")
    parser.add_argument("--rate_bus", type=float, default=1.0, help="events per BUS per day (depot)")
    parser.add_argument("--rate_car", type=float, default=3.0, help="events per CAR per day (highway fast charge)")
    args = parser.parse_args()
    random.seed(int(args.seed))

    evs = EV.load_from_yaml(args.ev)
    stations = Station.load_from_yaml(args.station)

    rates = {
        "V2G": float(args.rate_v2g),
        "BUS": float(args.rate_bus),
        "CAR": float(args.rate_car),
    }

    events = generate_events(
        days=int(args.days),
        evs=evs,
        stations=stations,
        events_per_ev_per_day=rates,
        start_id=int(args.start_id),
    )

    queued_events = assign_queue_fields(
        events=[e.to_config() for e in events],
        stations=stations,
    )
    doc = {"events": [OneLineDict(e) for e in queued_events]}
    with open(args.out, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=1000)

if __name__ == "__main__":
    main()
