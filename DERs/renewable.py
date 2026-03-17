import math
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from utils import OneLineDict

@dataclass
class Renewable:

    renewable_i: int                 # Renewable ID
    bus_i: int                       # Connected bus ID
    type: str                        # "PV" / "WIND"
    status: int = 1                  # 1-in service, 0-out of service
    Pg: float = 2.0                  # Active power injection (MW)
    Qg: float = 1.9                  # Reactive power injection (MVAr)
    Pmax: Optional[float] = None     # Maximum active power (MW)
    Pav: Optional[float] = None      # Available active power (MW)
    pf: float = 0.95                 # Power factor (0~1)
    pf_mode: str = "lag"             # "lag" / "lead"
    curt_cost: float = 100000.0      # Curtailment cost (¥/MW)
    meta: Optional[Dict[str, Any]] = None

    def calculate_Qg(self, *, Pg: Optional[float] = None) -> float:
       
        p = float(self.Pg if Pg is None else Pg)
        pf = self.pf
        if abs(p) <= 1e-12 or abs(pf - 1.0) <= 1e-12:
            q = 0.0
        else:
            phi = math.acos(pf)
            q_mag = abs(p) * math.tan(phi)
            mode = self.pf_mode
            if mode not in ("lag", "lead"):
                mode = "lag"
            q = q_mag if mode == "lag" else -q_mag
        self.Qg = float(q)
    
        return self.Qg

    def curtailed_power(self, *, Pg: Optional[float] = None, Pav: Optional[float] = None) -> float:
        pg = float(self.Pg if Pg is None else Pg)
        pav = float(self.Pav if Pav is None else Pav)
        return float(max(0.0, pav - pg))

    def cost(self, *, Pg: Optional[float] = None, Pav: Optional[float] = None) -> float:
        return float(self.curt_cost) * self.curtailed_power(Pg=Pg, Pav=Pav)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Renewable":
        obj = cls(
            renewable_i=int(cfg["renewable_i"]),
            bus_i=int(cfg["bus_i"]),
            type=str(cfg.get("type", "PV")).upper(),
            status=int(cfg.get("status", 1)),
            Pg=float(cfg.get("Pg", 0.0)),
            Pav=float(cfg.get("Pav", 0.0)),
            Pmax=float(cfg.get("Pmax", 2.0)),
            pf=float(cfg.get("pf", 0.95)),
            pf_mode=str(cfg.get("pf_mode", "lag")).lower(),
            curt_cost=float(cfg.get("curt_cost", 100000.0)),
            meta=(dict(cfg.get("meta")) if cfg.get("meta") is not None else None),
        )
        obj.Qg=float(obj.calculate_Qg())
        return obj

    def to_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "renewable_i": self.renewable_i,
            "bus_i": self.bus_i,
            "type": self.type,
            "status": self.status,
            "Pg": self.Pg,
            "Qg": self.Qg,
            "Pav": self.Pav,
            "Pmax": self.Pmax,
            "pf": self.pf,
            "pf_mode": self.pf_mode,
            "curt_cost": self.curt_cost,
        }
        if self.meta is not None:
            cfg["meta"] = self.meta
        return cfg

    @staticmethod
    def load_from_yaml(yaml_path: str) -> List["Renewable"]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) 
        renewables: List[Dict[str, Any]] = cfg.get("renewables", [])
        return [Renewable.from_config(r) for r in renewables]

    @staticmethod
    def to_grid(renewables: List["Renewable"]) -> List["OneLineDict"]:
        return [OneLineDict(r.to_config()) for r in renewables]
