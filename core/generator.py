import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from utils import OneLineDict

@dataclass
class Generator:

    gen_i: int                # Generator ID
    bus_i: int                # Connected bus ID
    status: int = 1           # 1-in service, 0-out of service
    Pg: float = 5.0           # Active power output (MW)
    Qg: float = 5.0           # Reactive power output (MVar)
    Pmax: float = 10.0        # Maximum active power output (MW)
    Pmin: float = 0.0         # Minimum active power output (MW)
    Qmax: float = 10.0        # Maximum reactive power output (MVar)
    Qmin: float = -10.0       # Minimum reactive power output (MVar)
    cost_c2: float = 10.0     # Quadratic cost coefficient
    cost_c1: float = 500.0    # Linear cost coefficient
    cost_c0: float = 10000.0  # Constant cost coefficient

    def cost(self, Pg: Optional[float] = None) -> float:
        p = float(self.Pg if Pg is None else Pg)
        return self.cost_c2 * p * p + self.cost_c1 * p + self.cost_c0
    
    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Generator":
        return cls(
            gen_i=int(cfg["gen_i"]),
            bus_i=int(cfg["bus_i"]),
            status=int(cfg.get("status", 1)),
            Pg=float(cfg.get("Pg", 5.0)),
            Qg=float(cfg.get("Qg", 5.0)),
            Pmax=float(cfg.get("Pmax", 10.0)),
            Pmin=float(cfg.get("Pmin", 0.0)),
            Qmax=float(cfg.get("Qmax", 10.0)),
            Qmin=float(cfg.get("Qmin", -10.0)),
            cost_c2=float(cfg.get("cost_c2", 10.0)),
            cost_c1=float(cfg.get("cost_c1", 500.0)),
            cost_c0=float(cfg.get("cost_c0", 10000.0)),
    )
    def to_config(self) -> Dict[str, Any]:
        return {
            "gen_i": self.gen_i,
            "bus_i": self.bus_i,
            "status": self.status,
            "Pg": self.Pg,
            "Qg": self.Qg,
            "Pmax": self.Pmax,
            "Pmin": self.Pmin,
            "Qmax": self.Qmax,
            "Qmin": self.Qmin,
            "cost_c2": self.cost_c2,
            "cost_c1": self.cost_c1,
            "cost_c0": self.cost_c0,
        }

    @staticmethod
    def load_from_yaml(yaml_path: str) -> List["Generator"]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        gens: List[Dict[str, Any]] = cfg.get("generators", [])
        return [Generator.from_config(g) for g in gens]

    @staticmethod
    def to_grid(gens: List["Generator"]) -> List["OneLineDict"]:
        return [OneLineDict(g.to_config()) for g in gens]
