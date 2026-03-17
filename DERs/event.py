import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from utils import OneLineDict

@dataclass
class ChargingEvent:

    event_i: int           # Charging Event ID
    ev_i: int              # EV ID
    station_i: int         # Station ID
    arrival_t: int         # arrival time in seconds
    departure_t: int       # departure time in seconds
    soc_init: Optional[float] = None  # optional event-level initial soc
    plot_i: int = -1       # Parking Plot ID
    start_t: int = -1      # start time in seconds
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "ChargingEvent":
        return cls(
            event_i=int(cfg["event_i"]),
            ev_i=int(cfg["ev_i"]),
            station_i=int(cfg["station_i"]),
            arrival_t=int(cfg["arrival_t"]),
            departure_t=int(cfg["departure_t"]),
            soc_init=(float(cfg["soc_init"]) if cfg.get("soc_init") is not None else None),
            plot_i=int(cfg.get("plot_i", -1)),
            start_t=int(cfg.get("start_t", -1)),
            meta=(dict(cfg.get("meta")) if cfg.get("meta") is not None else None),
        )

    def to_config(self) -> Dict[str, Any]:
        csg: Dict[str, Any] = {
            "event_i": self.event_i,
            "ev_i": self.ev_i,
            "station_i": self.station_i,
            "arrival_t": self.arrival_t,
            "departure_t": self.departure_t,
        }
        if self.soc_init is not None:
            csg["soc_init"] = float(self.soc_init)
        if int(self.plot_i) > 0:
            csg["plot_i"] = int(self.plot_i)
        if int(self.start_t) >= 0:
            csg["start_t"] = int(self.start_t)
            csg["wait_t"] = int(self.waiting_time())
        if self.meta is not None:
            csg["meta"] = self.meta
        return csg

    @staticmethod
    def load_from_yaml(yaml_path: str) -> List["ChargingEvent"]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        events: List[Dict[str, Any]] = cfg.get("events", [])
        return [ChargingEvent.from_config(ev) for ev in events]

    @staticmethod
    def to_grid(events: List["ChargingEvent"]) -> List["OneLineDict"]:
        return [OneLineDict(ev.to_config()) for ev in events]

    def duration(self) -> int:
        return int(max(0, self.departure_t - self.arrival_t))

    def at_station(self, t: int) -> bool:
        tt = int(t)
        return int(self.arrival_t) <= tt < int(self.departure_t)
    
    def at_plot(self) -> bool:
        return int(self.start_t) >= 0 and int(self.plot_i) > 0
    
    def charging_time(self) -> int:
        if not self.at_plot():
            return 0
        return int(max(0, self.departure_t - self.start_t))

    def waiting_time(self) -> int:
        if int(self.start_t) < 0:
            return 0
        return int(max(0, self.start_t - self.arrival_t))
    
    def get_soc(self, evs: Dict[int, Any]) -> float:
        return float(evs[self.ev_i]["soc"])
