import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from utils import OneLineDict

@dataclass
class Station:
    station_i: int                   # Station ID
    bus_i: int                       # Connected bus ID
    status: int = 1                  # 1-in service, 0-out of service
    type: str = "RESIDENTIAL"        # "RESIDENTIAL" / "HIGHWAY" / "DEPOT"
    n_plots: int = 25                # Number of plots
    meta: Optional[Dict[str, Any]] = None
    # EV sessions currently occupying plots
    at_plots: Dict[int, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    # FIFO queue of arrival events waiting for a plot
    in_queue: List[Dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Station":
        return cls(
            station_i=int(cfg["station_i"]),
            bus_i=int(cfg["bus_i"]),
            status=int(cfg.get("status", 1)),
            type=str(cfg.get("type", "RESIDENTIAL")).upper(),
            n_plots=int(cfg.get("n_plots", 50)),
            meta=(dict(cfg.get("meta")) if cfg.get("meta") is not None else None),
        )

    def to_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "station_i": int(self.station_i),
            "bus_i": int(self.bus_i),
            "status": self.status,
            "type": str(self.type),
            "n_plots": int(self.n_plots),
        }
        if self.meta is not None:
            cfg["meta"] = self.meta
        return cfg

    @staticmethod
    def load_from_yaml(yaml_path: str) -> List["Station"]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        stations: List[Dict[str, Any]] = cfg.get("stations", [])
        return [Station.from_config(st) for st in stations]

    @staticmethod
    def to_grid(stations: List["Station"]) -> List["OneLineDict"]:
        return [OneLineDict(st.to_config()) for st in stations]
    
    # find free plots
    def free_plots(self) -> List[int]:
        used = set(self.at_plots.keys())
        return [i for i in range(1, int(self.n_plots) + 1) if i not in used]
    # arrival but no free plot -> enqueue
    def enqueue_event(self, event: Dict[str, Any]) -> None:
        self.in_queue.append(event)
    # dequeue for potential scheduling
    def pop_queue(self) -> Optional[Dict[str, Any]]:
        if not self.in_queue:
            return None
        return self.in_queue.pop(0)
    # get active sessions
    def active_sessions(self) -> List[Dict[str, Any]]:
        return list(self.at_plots.values())

    def start_session(self, *, plot_i: int, event: Dict[str, Any], t: int) -> None:
        pi = int(plot_i)
        if pi < 1 or pi > int(self.n_plots):
            raise ValueError(f"plot_{pi} out of range.")
        if pi in self.at_plots:
            raise ValueError(f"plot {pi} already occupied.")

        tt = int(t)
        arr = int(event["arrival_t"])
        dep = int(event["departure_t"])
        if dep <= arr:
            raise ValueError("invalid event time.")

        self.at_plots[pi] = {
            "plot_i": pi,
            "event_i": int(event["event_i"]),
            "ev_i": int(event["ev_i"]),
            "arrival_t": arr,
            "start_t": tt,
            "departure_t": dep,
        }

    def end_session(self, plot_i: int) -> None:
        pi = int(plot_i)
        if pi in self.at_plots:
            del self.at_plots[pi]

    def release_departed(self, t: int) -> List[int]:
        tt = int(t)
        freed: List[int] = []
        for pi, sess in list(self.at_plots.items()):
            if int(sess["departure_t"]) <= tt:
                freed.append(int(pi))
                del self.at_plots[pi]
        return freed

    def step(self, *, t: int, arrivals: List[Dict[str, Any]]) -> Dict[str, int]:
        tt = int(t)
        started = 0
        queued = 0
        self.release_departed(tt)
        for evnt in arrivals:
            if int(evnt.get("station_i", self.station_i)) != int(self.station_i):
                continue
            free = self.free_plots()
            if free:
                self.start_session(plot_i=free[0], event=evnt, t=tt)
                started += 1
            else:
                self.enqueue_event(evnt)
                queued += 1

        while True:
            free = self.free_plots()
            if not free:
                break
            evnt = self.pop_queue()
            if evnt is None:
                break
            if int(evnt["departure_t"]) <= tt:
                continue
            if int(evnt["arrival_t"]) > tt:
                self.in_queue.insert(0, evnt)
                break
            self.start_session(plot_i=free[0], event=evnt, t=tt)
            started += 1

        return {"started": started, "queued": queued, "active": len(self.at_plots), "queue": len(self.in_queue)}

    @staticmethod
    def event_power(power: Dict[Tuple[int, int], float], event_i: int, t: int) -> float:
        ei = int(event_i)
        tt = int(t)
        v = power.get((ei, tt), 0.0)
        try:
            return float(v)
        except Exception:
            return 0.0

    def aggregate_power(self, *, t: int, p_event_t: Dict[Tuple[int, int], float]) -> Dict[str, float]:
        tt = int(t)
        p_ch = 0.0
        p_dis = 0.0

        for sess in self.at_plots.values():
            st = int(sess["start_t"])
            dep = int(sess["departure_t"])
            if tt < st or tt >= dep:
                continue
            ei = int(sess["event_i"])
            p = self.event_power(p_event_t, ei, tt)
            if p >= 0.0:
                p_ch += p
            else:
                p_dis += (-p)

        return {
            "p_charge": float(p_ch),
            "p_discharge": float(p_dis),
            "p_net": float(p_ch - p_dis),
        }
