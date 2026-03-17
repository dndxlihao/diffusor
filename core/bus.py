import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from utils import OneLineDict

@dataclass
class Bus:
    
    bus_i: int                      # Bus ID
    Pd: float                       # active load (kW) 
    Qd: float                       # reactive load (kVar) 
    type: str = "PQ"                # "PQ" / "PV" / "SLACK"
    Vm: float = 1.0                 # voltage magnitude (p.u.)
    Va: float = 0.0                 # voltage angle (degrees)
    baseKV: float = 12.66           # base voltage (kV)
    Vmax: float = 1.1               # max voltage (p.u.)
    Vmin: float = 0.9               # min voltage (p.u.)
    lat: Optional[float] = None     # latitude
    lon: Optional[float] = None     # longitude

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Bus":
        return cls(
            bus_i=int(cfg["bus_i"]),
            type=str(cfg["type"]).upper(),
            Pd=float(cfg.get("Pd", 0.0)),
            Qd=float(cfg.get("Qd", 0.0)),
            Vm=float(cfg.get("Vm", 1.0)),
            Va=float(cfg.get("Va", 0.0)),
            baseKV=float(cfg.get("baseKV", 12.66)),
            Vmax=float(cfg.get("Vmax", 1.1)),
            Vmin=float(cfg.get("Vmin", 0.9)),
            lat=float(cfg["lat"]) if "lat" in cfg else None,
            lon=float(cfg["lon"]) if "lon" in cfg else None,
        )

    def to_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "bus_i": self.bus_i,
            "type": self.type,
            "Pd": self.Pd,
            "Qd": self.Qd,
            "Vm": self.Vm,
            "Va": self.Va,
            "baseKV": self.baseKV,
            "Vmax": self.Vmax,
            "Vmin": self.Vmin,
        }
        if self.lat is not None:
            cfg["lat"] = self.lat
        if self.lon is not None:
            cfg["lon"] = self.lon
        return cfg

    @staticmethod
    def load_from_yaml(yaml_path: str) -> List["Bus"]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        buses: List[Dict[str, Any]] = cfg.get("buses", [])
        return [Bus.from_config(b) for b in buses]

    @staticmethod
    def to_grid(buses: List["Bus"]) -> List["OneLineDict"]:
        return [OneLineDict(b.to_config()) for b in buses]
