import yaml
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import List, Dict, Optional, Tuple, Any
from core import Bus, Branch, Generator
from DERs import Renewable, Storage, EV, Station, ChargingEvent
from utils import get_series_value

@dataclass
class Grid:

    baseMVA: float = 1.0
    buses: List[Bus] = field(default_factory=list)
    branches: List[Branch] = field(default_factory=list)
    generators: List[Generator] = field(default_factory=list)
    renewables: List[Renewable] = field(default_factory=list)
    storages: List[Storage] = field(default_factory=list)
    stations: List[Station] = field(default_factory=list)
    evs: List[EV] = field(default_factory=list)
    events: List[ChargingEvent] = field(default_factory=list)
    scale_path: str = "config/time.yaml"
    renewable_profiles: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    _ybus_cache: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def invalidate_cache(self) -> None:
        self._ybus_cache = None
 
    # 1. ID generation methods
    @staticmethod
    def _next_id(objs: List[object], attr: str, *, start: int = 1) -> int:
        mx = 0
        for o in objs:
            try:
                v = int(getattr(o, attr))
            except Exception:
                continue
            if v > mx:
                mx = v
        return max(int(start), mx + 1)

    def next_bus_i(self) -> int:
        return self._next_id(self.buses, "bus_i", start=1)

    def next_branch_i(self) -> int:
        return self._next_id(self.branches, "branch_i", start=1)

    def next_gen_i(self) -> int:
        return self._next_id(self.generators, "gen_i", start=1)

    def next_renewable_i(self) -> int:
        return self._next_id(self.renewables, "renewable_i", start=1)

    def next_storage_i(self) -> int:
        return self._next_id(self.storages, "storage_i", start=1)

    def next_station_i(self) -> int:
        return self._next_id(self.stations, "station_i", start=1)

    def next_ev_i(self) -> int:
        return self._next_id(self.evs, "ev_i", start=1)

    def next_event_i(self) -> int:
        return self._next_id(self.events, "event_i", start=1)

    # 2. ID validation methods
    def _validate_unique_bus_ids(self) -> None:
        ids = [b.bus_i for b in self.buses]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate bus_i found in Grid.buses")

    def _validate_unique_gen_ids(self) -> None:
        ids = [g.gen_i for g in self.generators]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate gen_i found in Grid.generators")

    def _validate_unique_branch_ids(self) -> None:
        ids = [br.branch_i for br in self.branches]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate branch_i found in Grid.branches")
    
    def _validate_unique_renewable_ids(self) -> None:
        ids = [r.renewable_i for r in self.renewables]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate renewable_i found in Grid.renewables")
        
    def _validate_unique_storage_ids(self) -> None:
        ids = [s.storage_i for s in self.storages]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate storage_i found in Grid.storages")
    
    def _validate_unique_station_ids(self) -> None:
        ids = [st.station_i for st in self.stations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate station_i found in Grid.stations")
    
    def _validate_unique_ev_ids(self) -> None:
        ids = [e.ev_i for e in self.evs]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate ev_i found in Grid.evs")
    
    def _validate_unique_event_ids(self) -> None:
        ids = [ev.event_i for ev in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate event_i found in Grid.events")

    def validate_references(self) -> None:
        bus_set = {b.bus_i for b in self.buses}
        for br in self.branches:
            if br.f_bus not in bus_set or br.t_bus not in bus_set:
                raise ValueError(f"Branch {br.branch_i} refers to missing bus: from={br.f_bus}, to={br.t_bus}")
        for g in self.generators:
            if g.bus_i not in bus_set:
                raise ValueError(f"Generator {g.gen_i} refers to missing bus: bus_i={g.bus_i}")
        for r in self.renewables:
            if r.bus_i not in bus_set:
                raise ValueError(f"Renewable {r.renewable_i} refers to missing bus: bus_i={r.bus_i}")
        for s in self.storages:
            if s.bus_i not in bus_set:
                raise ValueError(f"Storage {s.storage_i} refers to missing bus: bus_i={s.bus_i}")
        for st in self.stations:
            if st.bus_i not in bus_set:
                raise ValueError(f"Station {st.station_i} refers to missing bus: bus_i={st.bus_i}")
        station_set = {st.station_i for st in self.stations}
        ev_set = {e.ev_i for e in self.evs}
        for ev in self.events:
            if ev.ev_i not in ev_set:
                raise ValueError(f"Event {ev.event_i} refers to missing EV: ev_i={ev.ev_i}")
            if ev.station_i not in station_set:
                raise ValueError(f"Event {ev.event_i} refers to missing Station: station_i={ev.station_i}")

    # 3. YAML load/save
    @staticmethod
    def load_from_yaml(path: str) -> "Grid":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        base_mva = float(cfg.get("baseMVA", 1.0))
        buses = Bus.load_from_yaml(path)
        branches = Branch.load_from_yaml(path)
        generators = Generator.load_from_yaml(path)
        renewables = Renewable.load_from_yaml(path)
        storages = Storage.load_from_yaml(path)
        stations = Station.load_from_yaml(path)
        evs = EV.load_from_yaml(path)
        events = ChargingEvent.load_from_yaml(path)

        grid = Grid(baseMVA=base_mva, buses=buses, branches=branches, generators=generators, renewables=renewables, storages=storages,
                    stations=stations, evs=evs, events=events)
        grid._validate_unique_bus_ids()
        grid._validate_unique_branch_ids()
        grid._validate_unique_gen_ids()
        grid._validate_unique_renewable_ids()
        grid._validate_unique_storage_ids()
        grid._validate_unique_station_ids()
        grid._validate_unique_ev_ids()
        grid._validate_unique_event_ids()
        grid.validate_references()
        return grid

    def save_to_yaml(self, path: str) -> None:
        cfg = {
            "baseMVA": self.baseMVA,
            "buses": Bus.to_grid(self.buses),
            "generators": Generator.to_grid(self.generators),
            "branches": Branch.to_grid(self.branches),
            "renewables": Renewable.to_grid(self.renewables),
            "storages": Storage.to_grid(self.storages),
            "stations": Station.to_grid(self.stations),
            "evs": EV.to_grid(self.evs),
            "events": ChargingEvent.to_grid(self.events),
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False, width=1000)

    # 4. get tools
    @property
    def nb(self) -> int:
        return len(self.buses)

    @property
    def bus_index(self) -> Dict[int, int]:
        return {bus.bus_i: idx for idx, bus in enumerate(self.buses)}

    def get_bus(self, bus_i: int) -> Optional[Bus]:
        return next((b for b in self.buses if b.bus_i == bus_i), None)
    
    def get_branch(self, branch_i: int) -> Optional[Branch]:
        return next((br for br in self.branches if br.branch_i == branch_i), None)
    
    def get_generator(self, gen_i: int) -> Optional[Generator]:
        return next((g for g in self.generators if g.gen_i == gen_i), None)
    
    def get_renewable(self, renewable_i: int) -> Optional[Renewable]:
        return next((r for r in self.renewables if r.renewable_i == renewable_i), None)
    
    def get_storage(self, storage_i: int) -> Optional[Storage]:
        return next((s for s in self.storages if s.storage_i == storage_i), None)
    
    def get_station(self, station_i: int) -> Optional[Station]:
        return next((st for st in self.stations if st.station_i == station_i), None)
    
    def get_ev(self, ev_i: int) -> Optional[EV]:
        return next((e for e in self.evs if e.ev_i == ev_i), None)

    def get_event(self, event_i: int) -> Optional[ChargingEvent]:
        return next((ev for ev in self.events if ev.event_i == event_i), None)

    def get_generators(self, bus_i: int) -> List[Generator]:
        return [g for g in self.generators if g.bus_i == bus_i and g.status == 1]
    
    def get_renewables(self, bus_i: int) -> List[Renewable]:
        return [r for r in self.renewables if r.bus_i == bus_i and r.status == 1]
    
    def get_storages(self, bus_i: int) -> List[Storage]:
        return [s for s in self.storages if s.bus_i == bus_i and s.status == 1]
    
    def get_stations(self, bus_i: int) -> List[Station]:
        return [st for st in self.stations if st.bus_i == bus_i and st.status == 1]

    def gens_by_bus_index(self) -> Dict[int, List[Generator]]:
        idx = self.bus_index
        out: Dict[int, List[Generator]] = {}
        for g in self.generators:
            if g.status != 1:
                continue
            bi = idx.get(g.bus_i)
            if bi is None:
                continue
            out.setdefault(bi, []).append(g)
        return out

    def renewables_by_bus_index(self) -> Dict[int, List[Renewable]]:
        idx = self.bus_index
        out: Dict[int, List[Renewable]] = {}
        for r in self.renewables:
            if r.status != 1:
                continue
            bi = idx.get(r.bus_i)
            if bi is None:
                continue
            out.setdefault(bi, []).append(r)
        return out

    def storages_by_bus_index(self) -> Dict[int, List[Storage]]:
        idx = self.bus_index
        out: Dict[int, List[Storage]] = {}
        for s in self.storages:
            if s.status != 1:
                continue
            bi = idx.get(s.bus_i)
            if bi is None:
                continue
            out.setdefault(bi, []).append(s)
        return out

    def stations_by_bus_index(self) -> Dict[int, List[Station]]:
        idx = self.bus_index
        out: Dict[int, List[Station]] = {}
        for st in self.stations:
            if st.status != 1:
                continue
            bi = idx.get(st.bus_i)
            if bi is None:
                continue
            out.setdefault(bi, []).append(st)
        return out

    def events_by_station_id(self) -> Dict[int, List[ChargingEvent]]:
        out: Dict[int, List[ChargingEvent]] = {}
        for ev in self.events:
            out.setdefault(ev.station_i, []).append(ev)
        return out

    def events_by_ev_id(self) -> Dict[int, List[ChargingEvent]]:
        out: Dict[int, List[ChargingEvent]] = {}
        for ev in self.events:
            out.setdefault(ev.ev_i, []).append(ev)
        return out

    def bus_type_indices(self, *, require_single_slack: bool = True) -> Tuple[int, List[int], List[int]]:
        slack: List[int] = []
        pv: List[int] = []
        pq: List[int] = []

        for i, b in enumerate(self.buses):
            t = (b.type or "").upper()
            if t == "SLACK":
                slack.append(i)
            elif t == "PV":
                pv.append(i)
            else:
                pq.append(i)

        if require_single_slack and len(slack) != 1:
            raise ValueError(f"Power flow requires exactly 1 slack bus; got {len(slack)}.")

        return slack[0], pv, pq
    
    def get_item(self, kind: str, item_id: int):
        k = (kind or "").strip().lower()
        if k in {"bus", "buses"}:
            return self.get_bus(item_id)
        if k in {"branch", "branches"}:
            return self.get_branch(item_id)
        if k in {"gen", "generator", "generators"}:
            return self.get_generator(item_id)
        if k in {"renewable", "renewables"}:
            return self.get_renewable(item_id)
        if k in {"storage", "storages"}:
            return self.get_storage(item_id)
        if k in {"station", "stations"}:
            return self.get_station(item_id)
        if k in {"ev", "evs"}:
            return self.get_ev(item_id)
        if k in {"event", "events", "chargingevent", "chargingevents"}:
            return self.get_event(item_id)
        raise KeyError(f"Unknown kind: {kind!r}")

    def remove_item(self, kind: str, item_id: int, **kwargs) -> None:
        k = (kind or "").strip().lower()
        if k in {"bus", "buses"}:
            return self.remove_bus(item_id, **kwargs)
        if k in {"branch", "branches"}:
            return self.remove_branch(item_id)
        if k in {"gen", "generator", "generators"}:
            return self.remove_generator(item_id)
        if k in {"renewable", "renewables"}:
            return self.remove_renewable(item_id)
        if k in {"storage", "storages"}:
            return self.remove_storage(item_id)
        if k in {"station", "stations"}:
            return self.remove_station(item_id, **kwargs)
        if k in {"ev", "evs"}:
            return self.remove_ev(item_id, **kwargs)
        if k in {"event", "events", "chargingevent", "chargingevents"}:
            return self.remove_event(item_id)
        raise KeyError(f"Unknown kind: {kind!r}")

    @staticmethod
    def item_to_string(obj: object) -> str:
        return "<None>" if obj is None else repr(obj)
    def get_item_string(self, kind: str, item_id: int) -> str:
        return self.item_to_string(self.get_item(kind, item_id))
    
    # 5. modification methods
    def add_bus(self, bus: Bus, *, overwrite: bool = False) -> None:
        existing = self.get_bus(bus.bus_i)
        if existing is not None and not overwrite:
            raise ValueError(f"Bus {bus.bus_i} already exists.")
        if existing is not None and overwrite:
            self.buses = [b for b in self.buses if b.bus_i != bus.bus_i]
        self.buses.append(bus)
        self._validate_unique_bus_ids()
        self.invalidate_cache()

    def update_bus(self, bus_i: int, **fields) -> None:
        bus = self.get_bus(bus_i)
        if bus is None:
            raise KeyError(f"Bus {bus_i} not found")
        for k, v in fields.items():
            if not hasattr(bus, k):
                raise AttributeError(f"Bus has no attribute '{k}'")
            setattr(bus, k, v)
        self.invalidate_cache()

    def add_generator(self, gen: Generator, *, overwrite: bool = False) -> None:
        if self.get_bus(gen.bus_i) is None:
            raise ValueError(f"Generator {gen.gen_i} refers to missing bus {gen.bus_i}")
        existing = self.get_generator(gen.gen_i)
        if existing is not None and not overwrite:
            raise ValueError(f"Generator {gen.gen_i} already exists.")
        if existing is not None and overwrite:
            self.generators = [g for g in self.generators if g.gen_i != gen.gen_i]
        self.generators.append(gen)
        self._validate_unique_gen_ids()
        self.invalidate_cache()

    def update_generator(self, gen_i: int, **fields) -> None:
        gen = self.get_generator(gen_i)
        if gen is None:
            raise KeyError(f"Generator {gen_i} not found")
        for k, v in fields.items():
            if not hasattr(gen, k):
                raise AttributeError(f"Generator has no attribute '{k}'")
            setattr(gen, k, v)
        if hasattr(gen, "bus_i") and self.get_bus(int(gen.bus_i)) is None:
            raise ValueError(f"Generator {gen_i} refers to missing bus {gen.bus_i}")
        self.invalidate_cache()

    def remove_generator(self, gen_i: int) -> None:
        if not any(g.gen_i == gen_i for g in self.generators):
            raise KeyError(f"Generator {gen_i} not found")
        self.generators = [g for g in self.generators if g.gen_i != gen_i]
        self.invalidate_cache()

    def add_branch(self, br: Branch, *, overwrite: bool = False) -> None:
        bus_set = {b.bus_i for b in self.buses}
        if br.f_bus not in bus_set or br.t_bus not in bus_set:
            raise ValueError(f"Branch {br.branch_i} refers to missing bus")
        existing = self.get_branch(br.branch_i)
        if existing is not None and not overwrite:
            raise ValueError(f"Branch {br.branch_i} already exists.")
        if existing is not None and overwrite:
            self.branches = [x for x in self.branches if x.branch_i != br.branch_i]
        self.branches.append(br)
        self._validate_unique_branch_ids()
        self.invalidate_cache()

    def update_branch(self, branch_i: int, **fields) -> None:
        br = self.get_branch(branch_i)
        if br is None:
            raise KeyError(f"Branch {branch_i} not found")
        for k, v in fields.items():
            if not hasattr(br, k):
                raise AttributeError(f"Branch has no attribute '{k}'")
            setattr(br, k, v)
        if hasattr(br, "f_bus") and self.get_bus(int(br.f_bus)) is None:
            raise ValueError(f"Branch {branch_i} refers to missing bus: f_bus={br.f_bus}")
        if hasattr(br, "t_bus") and self.get_bus(int(br.t_bus)) is None:
            raise ValueError(f"Branch {branch_i} refers to missing bus: t_bus={br.t_bus}")
        self.invalidate_cache()

    def remove_branch(self, branch_i: int) -> None:
        if not any(br.branch_i == branch_i for br in self.branches):
            raise KeyError(f"Branch {branch_i} not found")
        self.branches = [br for br in self.branches if br.branch_i != branch_i]
        self.invalidate_cache()

    def add_renewable(self, ren: Renewable, *, overwrite: bool = False) -> None:
        if self.get_bus(ren.bus_i) is None:
            raise ValueError(f"Renewable {ren.renewable_i} refers to missing bus {ren.bus_i}")
        existing = self.get_renewable(ren.renewable_i)
        if existing is not None and not overwrite:
            raise ValueError(f"Renewable {ren.renewable_i} already exists.")
        if existing is not None and overwrite:
            self.renewables = [r for r in self.renewables if r.renewable_i != ren.renewable_i]
        self.renewables.append(ren)
        self._validate_unique_renewable_ids()
        self.invalidate_cache()

    def update_renewable(self, renewable_i: int, **fields) -> None:
        ren = self.get_renewable(renewable_i)
        if ren is None:
            raise KeyError(f"Renewable {renewable_i} not found")
        for k, v in fields.items():
            if not hasattr(ren, k):
                raise AttributeError(f"Renewable has no attribute '{k}'")
            setattr(ren, k, v)
        if hasattr(ren, "bus_i") and self.get_bus(int(ren.bus_i)) is None:
            raise ValueError(f"Renewable {renewable_i} refers to missing bus {ren.bus_i}")
        self.invalidate_cache()

    def remove_renewable(self, renewable_i: int) -> None:
        if not any(r.renewable_i == renewable_i for r in self.renewables):
            raise KeyError(f"Renewable {renewable_i} not found")
        self.renewables = [r for r in self.renewables if r.renewable_i != renewable_i]
        self.invalidate_cache()

    def add_storage(self, sto: Storage, *, overwrite: bool = False) -> None:
        if self.get_bus(sto.bus_i) is None:
            raise ValueError(f"Storage {sto.storage_i} refers to missing bus {sto.bus_i}")
        existing = self.get_storage(sto.storage_i)
        if existing is not None and not overwrite:
            raise ValueError(f"Storage {sto.storage_i} already exists.")
        if existing is not None and overwrite:
            self.storages = [s for s in self.storages if s.storage_i != sto.storage_i]
        self.storages.append(sto)
        self._validate_unique_storage_ids()
        self.invalidate_cache()

    def update_storage(self, storage_i: int, **fields) -> None:
        sto = self.get_storage(storage_i)
        if sto is None:
            raise KeyError(f"Storage {storage_i} not found")
        for k, v in fields.items():
            if not hasattr(sto, k):
                raise AttributeError(f"Storage has no attribute '{k}'")
            setattr(sto, k, v)
        if hasattr(sto, "bus_i") and self.get_bus(int(sto.bus_i)) is None:
            raise ValueError(f"Storage {storage_i} refers to missing bus {sto.bus_i}")
        self.invalidate_cache()

    def remove_storage(self, storage_i: int) -> None:
        if not any(s.storage_i == storage_i for s in self.storages):
            raise KeyError(f"Storage {storage_i} not found")
        self.storages = [s for s in self.storages if s.storage_i != storage_i]
        self.invalidate_cache()

    def add_station(self, st: Station, *, overwrite: bool = False) -> None:
        if self.get_bus(st.bus_i) is None:
            raise ValueError(f"Station {st.station_i} refers to missing bus {st.bus_i}")
        existing = self.get_station(st.station_i)
        if existing is not None and not overwrite:
            raise ValueError(f"Station {st.station_i} already exists.")
        if existing is not None and overwrite:
            self.stations = [x for x in self.stations if x.station_i != st.station_i]
            self.events = [ev for ev in self.events if ev.station_i != st.station_i]
        self.stations.append(st)
        self._validate_unique_station_ids()
        self.invalidate_cache()

    def update_station(self, station_i: int, **fields) -> None:
        st = self.get_station(station_i)
        if st is None:
            raise KeyError(f"Station {station_i} not found")
        for k, v in fields.items():
            if not hasattr(st, k):
                raise AttributeError(f"Station has no attribute '{k}'")
            setattr(st, k, v)
        if hasattr(st, "bus_i") and self.get_bus(int(st.bus_i)) is None:
            raise ValueError(f"Station {station_i} refers to missing bus {st.bus_i}")
        self.invalidate_cache()

    def remove_station(self, station_i: int, *, remove_attached: bool = True) -> None:
        if not any(st.station_i == station_i for st in self.stations):
            raise KeyError(f"Station {station_i} not found")
        if remove_attached:
            self.events = [ev for ev in self.events if ev.station_i != station_i]
        else:
            for ev in self.events:
                if ev.station_i == station_i:
                    raise ValueError(f"Cannot remove station {station_i}: event {ev.event_i} still attached")
        self.stations = [st for st in self.stations if st.station_i != station_i]
        self.invalidate_cache()

    def add_ev(self, ev: EV, *, overwrite: bool = False) -> None:
        existing = self.get_ev(ev.ev_i)
        if existing is not None and not overwrite:
            raise ValueError(f"EV {ev.ev_i} already exists.")
        if existing is not None and overwrite:
            self.events = [c for c in self.events if c.ev_i != ev.ev_i]
            self.evs = [e for e in self.evs if e.ev_i != ev.ev_i]
        self.evs.append(ev)
        self._validate_unique_ev_ids()

    def update_ev(self, ev_i: int, **fields) -> None:
        e = self.get_ev(ev_i)
        if e is None:
            raise KeyError(f"EV {ev_i} not found")
        for k, v in fields.items():
            if not hasattr(e, k):
                raise AttributeError(f"EV has no attribute '{k}'")
            setattr(e, k, v)

    def remove_ev(self, ev_i: int, *, remove_attached: bool = True) -> None:
        if not any(e.ev_i == ev_i for e in self.evs):
            raise KeyError(f"EV {ev_i} not found")
        if remove_attached:
            self.events = [c for c in self.events if c.ev_i != ev_i]
        else:
            for c in self.events:
                if c.ev_i == ev_i:
                    raise ValueError(f"Cannot remove EV {ev_i}: event {c.event_i} still attached")
        self.evs = [e for e in self.evs if e.ev_i != ev_i]

    def add_event(self, ev: ChargingEvent, *, overwrite: bool = False) -> None:
        if self.get_ev(ev.ev_i) is None:
            raise ValueError(f"Event {ev.event_i} refers to missing EV: ev_i={ev.ev_i}")
        if self.get_station(ev.station_i) is None:
            raise ValueError(f"Event {ev.event_i} refers to missing Station: station_i={ev.station_i}")
        existing = self.get_event(ev.event_i)
        if existing is not None and not overwrite:
            raise ValueError(f"Event {ev.event_i} already exists.")
        if existing is not None and overwrite:
            self.events = [c for c in self.events if c.event_i != ev.event_i]
        self.events.append(ev)
        self._validate_unique_event_ids()

    def update_event(self, event_i: int, **fields) -> None:
        c = self.get_event(event_i)
        if c is None:
            raise KeyError(f"Event {event_i} not found")
        for k, v in fields.items():
            if not hasattr(c, k):
                raise AttributeError(f"ChargingEvent has no attribute '{k}'")
            setattr(c, k, v)
        if hasattr(c, "ev_i") and self.get_ev(int(c.ev_i)) is None:
            raise ValueError(f"Event {event_i} refers to missing EV: ev_i={c.ev_i}")
        if hasattr(c, "station_i") and self.get_station(int(c.station_i)) is None:
            raise ValueError(f"Event {event_i} refers to missing Station: station_i={c.station_i}")

    def remove_event(self, event_i: int) -> None:
        if not any(c.event_i == event_i for c in self.events):
            raise KeyError(f"Event {event_i} not found")
        self.events = [c for c in self.events if c.event_i != event_i]

    def remove_bus(self, bus_i: int, *, remove_attached: bool = True) -> None:
        if self.get_bus(bus_i) is None:
            raise KeyError(f"Bus {bus_i} not found")

        if remove_attached:
            station_ids = {st.station_i for st in self.stations if st.bus_i == bus_i}
            if station_ids:
                self.events = [ev for ev in self.events if ev.station_i not in station_ids]
            self.stations = [st for st in self.stations if st.bus_i != bus_i]
            self.generators = [g for g in self.generators if g.bus_i != bus_i]
            self.renewables = [r for r in self.renewables if r.bus_i != bus_i]
            self.storages = [s for s in self.storages if s.bus_i != bus_i]
            self.branches = [br for br in self.branches if br.f_bus != bus_i and br.t_bus != bus_i]
        else:
            for st in self.stations:
                if st.bus_i == bus_i:
                    raise ValueError(f"Cannot remove bus {bus_i}: station {st.station_i} still attached")
            for g in self.generators:
                if g.bus_i == bus_i:
                    raise ValueError(f"Cannot remove bus {bus_i}: generator {g.gen_i} still attached")
            for r in self.renewables:
                if r.bus_i == bus_i:
                    raise ValueError(f"Cannot remove bus {bus_i}: renewable {r.renewable_i} still attached")
            for s in self.storages:
                if s.bus_i == bus_i:
                    raise ValueError(f"Cannot remove bus {bus_i}: storage {s.storage_i} still attached")
            for br in self.branches:
                if br.f_bus == bus_i or br.t_bus == bus_i:
                    raise ValueError(f"Cannot remove bus {bus_i}: branch {br.branch_i} still attached")
        self.buses = [b for b in self.buses if b.bus_i != bus_i]
        self.invalidate_cache()

    # 6. per-unit conversion methods
    @staticmethod
    def zbase(baseKV: float, baseMVA: float) -> float:
        return (float(baseKV) ** 2) / float(baseMVA)

    def bus_PQ_pu(self, bus: Bus, *, t: int) -> Tuple[float, float]:
        scale = get_series_value(self.scale_path, "load_scale", t=t)
        Pd_MW = float(bus.Pd) * scale / 1000.0
        Qd_MVar = float(bus.Qd) * scale / 1000.0
        return Pd_MW / self.baseMVA, Qd_MVar / self.baseMVA

    def gen_PQ_pu(self, gen: Generator) -> Tuple[float, float]:
        return float(gen.Pg) / self.baseMVA, float(gen.Qg) / self.baseMVA
    
    def renewable_PQ_pu(self, r: Renewable) -> Tuple[float, float]:
        return float(r.Pg) / self.baseMVA, float(r.Qg) / self.baseMVA
    
    def renewable_pav_mw(self, renewable_i: int, *, t: int) -> float:
        r = self.get_renewable(int(renewable_i))
        if r is None or int(getattr(r, "status", 1)) != 1:
            return 0.0
        profiles = getattr(self, "renewable_profiles", None)
        if not isinstance(profiles, dict):
            return float(r.Pmax)
        prof = profiles.get(int(renewable_i))
        if not isinstance(prof, dict):
            return float(r.Pmax)
        daily = prof.get("daily_pu")
        if not isinstance(daily, list) or len(daily) != 96:
            return float(r.Pmax)
        t_sec = max(0, int(t))
        idx = (t_sec % 86400) // 900  
        pu = float(daily[int(idx)])
        return float(r.Pmax) * pu

    def renewable_pav_pu(self, renewable_i: int, *, t: int) -> float:
        pav_mw = self.renewable_pav_mw(renewable_i, t=t)
        return pav_mw / self.baseMVA
    
    def storage_PQ_pu(self, s: Storage) -> Tuple[float, float]:
        Pg_MW = float(s.Pg) / 1000.0
        Qg_MVar = float(s.Qg) / 1000.0
        return Pg_MW / self.baseMVA, Qg_MVar / self.baseMVA
    
    def station_P_pu(self, st: Station, *, t: int, p_event_t: Dict[Tuple[int, int], float]) -> float:
        agg = st.aggregate_power(t=int(t), p_event_t=p_event_t)
        p_net = float(agg.get("p_net", 0.0)) 
        p_inj = -p_net
        p_inj_mw = p_inj / 1000.0 
        return p_inj_mw / float(self.baseMVA)

    def branch_z_pu(self, br: Branch) -> complex:
        fb = self.get_bus(br.f_bus)
        if fb is None:
            raise KeyError(f"Branch {br.branch_i} refers to unknown f_bus {br.f_bus}")
        zbase = self.zbase(fb.baseKV, self.baseMVA)
        return complex(float(br.r) / zbase, float(br.x) / zbase)

    def branch_y_pu(self, br: Branch) -> complex:
        z = self.branch_z_pu(br)
        if z == 0:
            return 0.0 + 0.0j
        return 1.0 / z

    # 7. Ybus calculation
    def ybus(self, *, rebuild: bool = False) -> np.ndarray:
        if rebuild or self._ybus_cache is None:
            self._ybus_cache = self.build_ybus()
        return self._ybus_cache

    def build_ybus(self) -> np.ndarray:
        nb = len(self.buses)
        if nb == 0:
            raise ValueError("Grid has no buses.")
        idx = self.bus_index
        Y = np.zeros((nb, nb), dtype=complex)
        for br in self.branches:
            if br.status != 1:
                continue
            y = self.branch_y_pu(br)
            i = idx[br.f_bus]
            k = idx[br.t_bus]
            Y[i, i] += y
            Y[k, k] += y
            Y[i, k] -= y
            Y[k, i] -= y

        return Y

    def build_radial_tree(self) -> Tuple[Dict[int, int], Dict[int, List[int]], List[Tuple[int, int, int]]]:
        slack_bus_i = int(self.buses[0].bus_i)
        bus_ids = [int(b.bus_i) for b in self.buses]
        adj: Dict[int, List[Tuple[int, int]]] = {i: [] for i in bus_ids}
        for br in self.branches:
            if int(getattr(br, "status", 1)) != 1:
                continue
            f = int(br.f_bus)
            t = int(br.t_bus)
            br_i = int(getattr(br, "branch_i", getattr(br, "br_i", -1)))
            adj.setdefault(f, []).append((t, br_i))
            adj.setdefault(t, []).append((f, br_i))
        parent: Dict[int, int] = {}
        children: Dict[int, List[int]] = {i: [] for i in bus_ids}
        edges: List[Tuple[int, int, int]] = []
        visited = set([slack_bus_i])
        q = deque([slack_bus_i])
        while q:
            u = q.popleft()
            for v, br_i in adj.get(u, []):
                if v in visited:
                    continue
                visited.add(v)
                parent[v] = u
                children[u].append(v)
                edges.append((u, v, br_i))
                q.append(v)
        if len(visited) != len(bus_ids):
            missing = sorted(set(bus_ids) - visited)
            raise ValueError(
                f"Grid is not connected. Unreachable buses from slack {slack_bus_i}: {missing}"
            )
        if len(edges) != len(bus_ids) - 1:
            raise ValueError(
                f"Grid is not radial. "
                f"Edges={len(edges)}, but nb-1={len(bus_ids)-1}."
        )
        return parent, children, edges
