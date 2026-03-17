import math
import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from utils import OneLineDict

@dataclass
class Storage:
    storage_i: int              # Storage ID
    bus_i: int                  # Connected bus ID
    status: int = 1             # 1 in-service, 0 out-of-service
    Emax: float = 5.0           # Energy capacity (MWh)
    soc: float = 0.5            # State of charge 
    P_ch: float = 100.0         # Max charging power (kW) 
    P_dis: float = -80.0        # Max discharging power (kW) 
    eta_ch: float = 0.95        # Charging efficiency
    eta_dis: float = 0.90       # Discharging efficiency
    pf: float = 0.95            # Power factor
    pf_mode: str = "lag"        # "lag" / "lead"
    Pg: float = 0.0             # Active power (kW)
    Qg: float = 0.0             # Reactive power (kVar)
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

    def step(self, *, delta_t: int, Pg: float) -> None:
       
        if not int(self.status):
            self.Pg = 0.0
            self.Qg = 0.0
            return
        P_cmd = float(Pg)
        Emax  = float(self.Emax)
        soc0  = float(self.soc)
        E0 = soc0 * Emax
        delta_h = float(delta_t) / 3600.0 
        if P_cmd > 0.0:
            E1 = E0 + P_cmd * float(self.eta_ch) * delta_h
        elif P_cmd < 0.0:
            E1 = E0 - (-P_cmd) / float(self.eta_dis) * delta_h
        else:
            E1 = E0

        self.soc = float(E1 / Emax)
        self.Pg = float(P_cmd)
        self.Qg = float(self.calculate_Qg(Pg=self.Pg))

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "Storage":
        obj = cls(
            storage_i=int(cfg["storage_i"]),
            bus_i=int(cfg["bus_i"]),
            status=int(cfg.get("status", 1)),
            Emax=float(cfg.get("Emax", 5.0)),
            soc=float(cfg.get("soc", 0.5)),
            P_ch=float(cfg.get("P_ch", 100.0)),
            P_dis=float(cfg.get("P_dis", -80.0)),
            eta_ch=float(cfg.get("eta_ch", 0.95)),
            eta_dis=float(cfg.get("eta_dis", 0.90)),
            pf=float(cfg.get("pf", 0.95)),
            pf_mode=str(cfg.get("pf_mode", "lag")).lower(),
            meta=(dict(cfg.get("meta")) if cfg.get("meta") is not None else None),
        )
        obj.Pg = float(cfg.get("Pg", 0.0))
        obj.Qg = obj.calculate_Qg(Pg=obj.Pg)
        return obj

    def to_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "storage_i": self.storage_i,
            "bus_i": self.bus_i,
            "status": self.status,
            "Emax": self.Emax,
            "soc": self.soc,
            "P_ch": self.P_ch,
            "P_dis": self.P_dis,
            "eta_ch": self.eta_ch,
            "eta_dis": self.eta_dis,
            "pf": self.pf,
            "pf_mode": self.pf_mode,
            "Pg": self.Pg,
            "Qg": self.Qg,
        }
        if self.meta is not None:
            cfg["meta"] = self.meta
        return cfg

    @staticmethod
    def load_from_yaml(yaml_path: str) -> List["Storage"]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        storages: List[Dict[str, Any]] = cfg.get("storages", [])
        return [Storage.from_config(s) for s in storages]

    @staticmethod
    def to_grid(storages: List["Storage"]) -> List["OneLineDict"]:
        return [OneLineDict(s.to_config()) for s in storages]
