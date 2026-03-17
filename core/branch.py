import yaml
from dataclasses import dataclass
from typing import Dict, Any, List
from utils import OneLineDict

@dataclass
class Branch:

    branch_i: int            # Branch ID
    f_bus: int               # From Bus ID
    t_bus: int               # To Bus ID
    r: float                 # resistance (Ohm) 
    x: float                 # reactance (Ohm)  
    rateA: float = 5.0       # capacity (MVA)
    status: int = 1          # 1-in service, 0-out of service

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Branch":
        return cls(
            branch_i=int(cfg["branch_i"]),
            f_bus=int(cfg["f_bus"]),
            t_bus=int(cfg["t_bus"]),
            r=float(cfg["r"]),
            x=float(cfg["x"]),
            rateA=float(cfg.get("rateA", 5.0)),
            status=int(cfg.get("status", 1)),
        )

    def to_config(self) -> Dict[str, Any]:
        return {
            "branch_i": self.branch_i,
            "f_bus": self.f_bus,
            "t_bus": self.t_bus,
            "r": self.r,
            "x": self.x,
            "rateA": self.rateA,
            "status": self.status,
        }

    @staticmethod
    def load_from_yaml(yaml_path: str) -> List["Branch"]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        branches: List[Dict[str, Any]] = cfg.get("branches", [])
        return [Branch.from_config(br) for br in branches]

    @staticmethod
    def to_grid(branches: List["Branch"]) -> List["OneLineDict"]:
        return [OneLineDict(br.to_config()) for br in branches]
