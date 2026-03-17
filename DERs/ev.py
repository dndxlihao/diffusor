import yaml
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from utils import OneLineDict

EV_TYPE_PROFILES: Dict[str, Dict[str, Any]] = {
    "V2G": {
        "capacity": 60.0,
        "p_ch": 8.0,
        "p_dis": 10.0,
        "eta_ch": 0.95,
        "eta_dis": 0.9,
        "soc": 0.55,
        "meta": {
            "charge_curve": {"soc0": 0.80, "k": 14.0},
            "v2g": {"opt_in": True, "min_soc": 0.3, "threshold": 0.0},
            "efficiency_curve": {
                "charge": {"eta_max": 0.95, "eta_min": 0.8, "alpha": 0.16},
            },
        },
    },
    "CAR": {
        "capacity": 60.0,
        "p_ch": 20.0,
        "p_dis": 0.0,
        "eta_ch": 0.95,
        "eta_dis": 0.9,
        "soc": 0.35,
        "meta": {
            "charge_curve": {"soc0": 0.76, "k": 13.0},
            "efficiency_curve": {
                "charge": {"eta_max": 0.95, "eta_min": 0.8, "alpha": 0.15},
            },
        },
    },
    "BUS": {
        "capacity": 300.0,
        "p_ch": 60.0,
        "p_dis": 30.0,
        "eta_ch": 0.95,
        "eta_dis": 0.9,
        "soc": 0.6,
        "meta": {
            "charge_curve": {"soc0": 0.72, "k": 11.0},
            "v2g": {"opt_in": True, "min_soc": 0.4, "threshold": 0.0},
            "efficiency_curve": {
                "charge": {"eta_max": 0.95, "eta_min": 0.8, "alpha": 0.14},
            },
        },
    },
}

@dataclass
class EV:
    
    ev_i: int                     # EV ID
    type: str                     # "CAR" / "BUS" / "V2G"
    capacity: float = 60.0        # battery capacity (kWh)
    p_ch: float = 8.0             # max charging power (kW) 
    p_dis: float = 0.0            # max discharging power (kW)
    eta_ch: float = 0.95          # charging efficiency 
    eta_dis: float = 0.9          # discharging efficiency 
    soc: float = 0.5              # state of charge 
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.type = str(self.type).upper()
        if self.meta is None:
            self.meta = {}
        self._apply_missing_type_defaults()

    @staticmethod
    def _deepcopy_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(meta, dict):
            return {}
        out: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, dict):
                out[k] = EV._deepcopy_meta(v)
            else:
                out[k] = v
        return out

    @classmethod
    def type_profile(cls, ev_type: str) -> Dict[str, Any]:
        t = str(ev_type).upper()
        if t not in EV_TYPE_PROFILES:
            raise ValueError(f"unsupported EV type: {ev_type}")
        prof = dict(EV_TYPE_PROFILES[t])
        prof["meta"] = cls._deepcopy_meta(EV_TYPE_PROFILES[t].get("meta"))
        return prof

    def _merge_meta_defaults(self, defaults: Dict[str, Any]) -> None:
        if self.meta is None:
            self.meta = {}
        for k, v in defaults.items():
            if isinstance(v, dict):
                cur = self.meta.get(k)
                if not isinstance(cur, dict):
                    self.meta[k] = self._deepcopy_meta(v)
                else:
                    for kk, vv in v.items():
                        if kk not in cur or cur[kk] is None:
                            cur[kk] = vv
            elif k not in self.meta or self.meta[k] is None:
                self.meta[k] = v

    def _apply_missing_type_defaults(self) -> None:
        try:
            prof = self.type_profile(self.type)
        except ValueError:
            return
        for field in ("capacity", "p_ch", "p_dis", "eta_ch", "eta_dis", "soc"):
            if getattr(self, field) is None:
                setattr(self, field, prof[field])
        self._merge_meta_defaults(prof["meta"])

    @classmethod
    def from_type_template(
        cls,
        *,
        ev_i: int,
        ev_type: str,
        soc: Optional[float] = None,
        meta_overrides: Optional[Dict[str, Any]] = None,
    ) -> "EV":
        prof = cls.type_profile(ev_type)
        meta = cls._deepcopy_meta(prof["meta"])
        if isinstance(meta_overrides, dict):
            for k, v in meta_overrides.items():
                meta[k] = cls._deepcopy_meta(v) if isinstance(v, dict) else v
        return cls(
            ev_i=int(ev_i),
            type=str(ev_type).upper(),
            capacity=float(prof["capacity"]),
            p_ch=float(prof["p_ch"]),
            p_dis=float(prof["p_dis"]),
            eta_ch=float(prof["eta_ch"]),
            eta_dis=float(prof["eta_dis"]),
            soc=float(prof["soc"] if soc is None else soc),
            meta=meta,
        )

    def apply_type_template(self, *, keep_soc: bool = True) -> None:
        prof = self.type_profile(self.type)
        soc = self.soc if keep_soc else float(prof["soc"])
        self.capacity = float(prof["capacity"])
        self.p_ch = float(prof["p_ch"])
        self.p_dis = float(prof["p_dis"])
        self.eta_ch = float(prof["eta_ch"])
        self.eta_dis = float(prof["eta_dis"])
        self.soc = float(soc)
        self.meta = self._deepcopy_meta(prof["meta"])

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "EV":
        return cls(
            ev_i=int(cfg["ev_i"]),
            type=str(cfg["type"]).upper(),
            capacity=float(cfg["capacity"]),
            p_ch=float(cfg["p_ch"]),
            p_dis=float(cfg.get("p_dis", 0.0)),
            eta_ch=float(cfg.get("eta_ch", 0.95)),
            eta_dis=float(cfg.get("eta_dis", 0.9)),
            soc=float(cfg.get("soc", 0.5)),
            meta=(dict(cfg.get("meta")) if cfg.get("meta") is not None else None),
        )

    def to_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "ev_i": self.ev_i,
            "type": self.type,
            "capacity": self.capacity,
            "p_ch": self.p_ch,
            "p_dis": self.p_dis,
            "eta_ch": self.eta_ch,
            "eta_dis": self.eta_dis,
            "soc": self.soc,
        }
        if self.meta is not None:
            cfg["meta"] = self.meta
        return cfg

    @staticmethod
    def load_from_yaml(yaml_path: str) -> List["EV"]:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        evs: List[Dict[str, Any]] = cfg.get("evs", []) 
        return [EV.from_config(e) for e in evs]

    @staticmethod
    def to_grid(evs: List["EV"]) -> List["OneLineDict"]:
        return [OneLineDict(e.to_config()) for e in evs]

    @property
    def energy(self) -> float:
        return float(self.soc) * float(self.capacity)
    
    @property
    def v2g_cap(self) -> bool:
        return float(self.p_dis) > 0.0

    def efficiency_curve_params(self) -> Dict[str, float]:
        base = {
            "eta_max": float(self.eta_ch),
            "eta_min": max(0.50, float(self.eta_ch) - 0.06),
            "alpha": 0.15,
        }
        if isinstance(self.meta, dict):
            eff = self.meta.get("efficiency_curve")
            if isinstance(eff, dict):
                cur = eff.get("charge")
                if isinstance(cur, dict):
                    for kk in ("eta_max", "eta_min", "alpha"):
                        if kk in cur and cur[kk] is not None:
                            base[kk] = float(cur[kk])
        base["eta_min"] = min(base["eta_max"], base["eta_min"])
        return base

    def charge_efficiency(self, power_kw: float) -> float:
        p = max(0.0, float(power_kw))
        p_max = float(self.p_ch)
        if p_max <= 1e-9 or p <= 1e-9:
            return 0.0
        r = min(1.0, p / p_max)
        prm = self.efficiency_curve_params()
        eta = prm["eta_max"] - prm["alpha"] * (r ** 2)
        return max(prm["eta_min"], min(prm["eta_max"], eta))

    def battery_gain_kw(self, power_kw: float) -> float:
        p = max(0.0, float(power_kw))
        return p * self.charge_efficiency(p)

    def battery_loss_kw(self, power_kw: float) -> float:
        p = max(0.0, float(power_kw))
        eta = max(1e-6, float(self.eta_dis))
        return p / eta

    def charge_efficiency_samples(self, *, points: int = 32) -> List[Tuple[float, float]]:
        p_max = float(self.p_ch)
        if p_max <= 1e-9:
            return [(0.0, 0.0)]
        n = max(2, int(points))
        out: List[Tuple[float, float]] = []
        for i in range(n + 1):
            p = p_max * float(i) / float(n)
            out.append((p, self.charge_efficiency(p)))
        return out

    # 1. charge curve parameters
    def charge_curve(self) -> Dict[str, float]:
        params = {"soc0": 0.8, "k": 12.0}
        if isinstance(self.meta, dict):
            cc = self.meta.get("charge_curve", None)
            if isinstance(cc, dict):
                if "soc0" in cc and cc["soc0"] is not None:
                    params["soc0"] = float(cc["soc0"])
                if "k" in cc and cc["k"] is not None:
                    params["k"] = float(cc["k"])
        return params

    # 2. compute charge limit 
    def charge_limit(self) -> float:
        p_ch = float(self.p_ch)
        s = float(self.soc)
        prm = self.charge_curve()
        soc0 = float(prm["soc0"])
        k = float(prm["k"])
        steep = max(1e-6, k)

        # Normalized sigmoid: smooth for the whole SOC range, ~1 at low SOC and 0 at full SOC.
        def _sigmoid(x: float) -> float:
            z = max(-60.0, min(60.0, steep * (x - soc0)))
            return 1.0 / (1.0 + math.exp(z))

        y0 = _sigmoid(0.0)
        y1 = _sigmoid(1.0)
        ys = _sigmoid(min(1.0, max(0.0, s)))
        denom = max(1e-9, y0 - y1)
        factor = (ys - y1) / denom
        factor = min(1.0, max(0.0, factor))
        return p_ch * factor

    def charge_limit_curve(self, *, points: int = 101) -> List[Tuple[float, float]]:
        soc_old = float(self.soc)
        out: List[Tuple[float, float]] = []
        n = max(2, int(points))
        for i in range(n):
            self.soc = float(i) / float(n - 1)
            out.append((self.soc, self.charge_limit()))
        self.soc = soc_old
        return out

    # 3. step power
    def step(self, p: float, dt: int) -> None:
        dt_h = float(dt) / 3600.0
        cap = float(self.capacity)
        p_cmd = float(p)
        if p_cmd >= 0.0:
            p_allow = self.charge_limit()
            p = min(p_cmd, float(p_allow))
        else:
            p = max(-self.p_dis, p_cmd) 
        e = self.energy
        if p >= 0.0:
            e_new = e + self.battery_gain_kw(p) * dt_h
        else:
            e_new = e - self.battery_loss_kw(-p) * dt_h
        self.soc = min(1.0, max(0.0, e_new / cap))

    # 4. v2g preference
    def v2g_opt(self) -> bool:
        if not isinstance(self.meta, dict):
            return False
        v2g = self.meta.get("v2g", None)
        if not isinstance(v2g, dict):
            return False
        val = v2g.get("opt_in", False)
        return bool(val)

    def v2g_threshold(self) -> float:
        # The current baseline disables the price threshold so V2G access
        # depends on hardware, user opt-in and SOC only.
        return 0.0
    
    def v2g_minsoc(self) -> float:
        if not isinstance(self.meta, dict):
            return 1.0
        v2g = self.meta.get("v2g", None)
        if not isinstance(v2g, dict):
            return 1.0
        if "min_soc" not in v2g or v2g["min_soc"] is None:
            return 1.0
        return float(v2g["min_soc"])

    def is_v2g(self, *, sell_price: Optional[float] = None) -> bool:
        # 1) Hardware capability
        if not self.v2g_cap:
            return False
        # 2) User opt-in
        if not self.v2g_opt():
            return False
        # 3) SOC threshold 
        if float(self.soc) <= self.v2g_minsoc():
            return False
        # 4) Price threshold 
        if sell_price is None:
            return True
        return float(sell_price) >= self.v2g_threshold()
