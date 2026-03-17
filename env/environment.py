import gymnasium as gym
import numpy as np
import math
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from gymnasium.envs.registration import register
from utils import get_series_value
from DERs.ev import EV
from core.grid import Grid
from solvers.opf.linear_distflow import _event_priority, _v2g_priority

@dataclass(frozen=True)
class Grid2AIEnvConfig:
    horizon_hours: int = 24
    resolution_minutes: int = 5
    ev_resolution_seconds: int = 300
    time_yaml_path: str = "config/time.yaml"
    queue_yaml_path: str = "config/event.yaml"
    ev_yaml_path: str = "config/ev.yaml"
    baseline_exogenous_path: str = "results/baseline_1w_exogenous.yaml"
    use_baseline_exogenous: bool = True
    v_penalty: float = 10000
    ev_depart_penalty: float = 20.0
    v2g_reward_coeff: float = 5.0
    ev_target_soc: float = 0.9
    storage_end_soc: float = 0.5
    storage_end_penalty: float = 200.0
    branch_overload_penalty: float = 10000
    soc_violation_penalty: float = 10000
    grid_pmax_mw: float = 100.0
    grid_qmax_mvar: float = 100.0
    grid_limit_penalty: float = 1000.0
    obs_vm_min: float = 0.90
    obs_vm_max: float = 1.10
    reward_scale: float = 1e-6

def time_steps(horizon_hours: int, resolution_minutes: int) -> Tuple[int, float]:
    delta_t = float(resolution_minutes) / 60.0
    T = int(horizon_hours / delta_t)
    return T, delta_t

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

def _mag01(x: Any):
    if isinstance(x, np.ndarray):
        return 0.5 * (x.astype(np.float32) + 1.0)
    if isinstance(x, (list, tuple)):
        arr = np.asarray(x, dtype=np.float32)
        return 0.5 * (arr + 1.0)
    return 0.5 * (float(x) + 1.0)

def _decode_sign_mode(x: float) -> int:
    return -1 if float(x) < 0.0 else +1

def _sec_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    return max(0, hi - lo)

def _ceil_div(a: int, b: int) -> int:
    return int((int(a) + int(b) - 1) // int(b))

def load_queue_yaml(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    stations_cfg = cfg.get("stations", [])
    events_cfg = cfg.get("events", [])
    stations: List[Dict[str, Any]] = [dict(st) for st in stations_cfg]
    events: List[Dict[str, Any]] = [dict(ev) for ev in events_cfg]
    return stations, events

def load_evs(path: str) -> Dict[int, EV]:
    ev_list = EV.load_from_yaml(path)
    return {int(ev.ev_i): ev for ev in ev_list}

def load_baseline_exogenous(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

@dataclass
class ActionLayout:
    stn_mode: slice
    stn_mag: slice
    n_stn: int
    Ksub: int

class Grid2AIEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        grid_yaml: str = "config/IEEE33.yaml",
        cfg: Union[Grid2AIEnvConfig, Dict[str, Any], None] = None,
        *,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.grid = Grid.load_from_yaml(str(Path(grid_yaml)))
        if cfg is None:
            self.cfg = Grid2AIEnvConfig()
        elif isinstance(cfg, dict):
            self.cfg = Grid2AIEnvConfig(**cfg)
        else:
            self.cfg = cfg
        self._rng = np.random.default_rng(seed)

        self.T, self.delta_t = time_steps(self.cfg.horizon_hours, self.cfg.resolution_minutes)
        self.opf_step_s = int(round(self.delta_t * 3600.0))
        self.horizon_s = int(self.cfg.horizon_hours) * 3600

        # get buses
        self.baseMVA = float(self.grid.baseMVA)
        self.buses = list(self.grid.buses)
        self.bus_ids = [int(b.bus_i) for b in self.buses]
        self.nb = len(self.bus_ids)
        self.bus_to_idx = {bid: i for i, bid in enumerate(self.bus_ids)}
        self.slack = int(self.bus_ids[0])
        self.slack_idx = int(self.bus_to_idx[self.slack])

        # get branches
        self.parent, self.children, edges3 = self.grid.build_radial_tree()
        self.edges: List[Tuple[int, int]] = []
        self.edge_branch_id: List[int] = []
        self.r_pu: np.ndarray = np.zeros((len(edges3),), dtype=np.float64)
        self.x_pu: np.ndarray = np.zeros((len(edges3),), dtype=np.float64)
        self.smax_pu: np.ndarray = np.zeros((len(edges3),), dtype=np.float64)
        br_dict = {int(getattr(br, "branch_i")): br for br in (self.grid.branches or [])}
        for k, (i, j, br_i) in enumerate(edges3):
            i = int(i)
            j = int(j)
            br_i = int(br_i)
            self.edges.append((i, j))
            self.edge_branch_id.append(br_i)
            br = br_dict.get(br_i, None)
            z = self.grid.branch_z_pu(br)
            self.r_pu[k] = float(z.real)
            self.x_pu[k] = float(z.imag)
            rateA = float(getattr(br, "rateA", 50.0))
            self.smax_pu[k] = rateA / self.baseMVA
        self.ne = len(self.edges)
        self._edge_index: Dict[Tuple[int, int], int] = {e: k for k, e in enumerate(self.edges)}
        self._fwd_order, self._post_order = self._precompute_tree_orders()

        # get generators
        self.gens = [g for g in (self.grid.generators or []) if int(getattr(g, "status", 1)) == 1]
        self.gen_ids = [int(getattr(g, "gen_i")) for g in self.gens]
        self.n_gen = len(self.gen_ids)

        # get renewables
        self.rens = [r for r in (self.grid.renewables or []) if int(getattr(r, "status", 1)) == 1]
        self.ren_ids = [int(getattr(r, "renewable_i")) for r in self.rens]
        self.n_ren = len(self.ren_ids)

        # get storages
        self.stos = [s for s in (self.grid.storages or []) if int(getattr(s, "status", 1)) == 1]
        self.sto_ids = [int(getattr(s, "storage_i")) for s in self.stos]
        self.n_sto = len(self.sto_ids)

        # get EV events
        self.stations_cfg, self.events_cfg = load_queue_yaml(self.cfg.queue_yaml_path)
        if not self.stations_cfg:
            self.stations_cfg = [dict(st.to_config()) for st in (self.grid.stations or []) if int(getattr(st, "status", 1)) == 1]
        self.station_by_id = {int(s["station_i"]): dict(s) for s in self.stations_cfg}
        self.station_bus = {sid: int(self.station_by_id[sid]["bus_i"]) for sid in self.station_by_id}
        self.station_ids = sorted(int(sid) for sid in self.station_by_id)
        self.station_to_idx = {int(sid): i for i, sid in enumerate(self.station_ids)}
        self.n_stn = len(self.station_ids)
        self.station_bus_idx = np.asarray(
            [int(self.bus_to_idx.get(int(self.station_bus[sid]), -1)) for sid in self.station_ids],
            dtype=np.int64,
        )
        self.event_ids = [int(e["event_i"]) for e in self.events_cfg]
        self.event_by_id = {int(e["event_i"]): dict(e) for e in self.events_cfg}
        self.n_ev = len(self.event_ids)
        self.ev_by_id = load_evs(self.cfg.ev_yaml_path)

        self.ev_step_s = int(self.cfg.ev_resolution_seconds)
        self.K_ev = int(_ceil_div(self.horizon_s, self.ev_step_s))
        self.Ksub = int(self.opf_step_s // self.ev_step_s)

        self.price = np.zeros((self.T,), dtype=np.float64)
        for tt in range(self.T):
            self.price[tt] = float(get_series_value(self.cfg.time_yaml_path, "price", t=int(tt * self.opf_step_s)))

        self.price_ev = np.zeros((self.K_ev,), dtype=np.float64)
        for k in range(self.K_ev):
            self.price_ev[k] = float(get_series_value(self.cfg.time_yaml_path, "price", t=int(k * self.ev_step_s)))

        # bus params
        self.Pd_pu = np.zeros((self.T, self.nb), dtype=np.float64)
        self.Qd_pu = np.zeros((self.T, self.nb), dtype=np.float64)
        for tt in range(self.T):
            t_sec = int(tt * self.opf_step_s)
            for bi, b in enumerate(self.bus_ids):
                bus = self.grid.get_bus(int(b))
                Pd, Qd = self.grid.bus_PQ_pu(bus, t=t_sec)
                self.Pd_pu[tt, bi] = float(Pd)
                self.Qd_pu[tt, bi] = float(Qd)

        # renewable params
        self.Pav_pu = np.zeros((self.T, self.n_ren), dtype=np.float64)
        for tt in range(self.T):
            t_sec = int(tt * self.opf_step_s)
            for ri, rid in enumerate(self.ren_ids):
                pav_mw = float(self.grid.renewable_pav_mw(int(rid), t=t_sec))
                self.Pav_pu[tt, ri] = pav_mw / self.baseMVA

        # generator params
        self.gen_bus_idx = np.zeros((self.n_gen,), dtype=np.int64)
        self.Pmin_pu = np.zeros((self.n_gen,), dtype=np.float64)
        self.Pmax_pu = np.zeros((self.n_gen,), dtype=np.float64)
        self.Qmin_pu = np.zeros((self.n_gen,), dtype=np.float64)
        self.Qmax_pu = np.zeros((self.n_gen,), dtype=np.float64)
        self.cost_c2 = np.zeros((self.n_gen,), dtype=np.float64)
        self.cost_c1 = np.zeros((self.n_gen,), dtype=np.float64)
        self.cost_c0 = np.zeros((self.n_gen,), dtype=np.float64)
        for i, g in enumerate(self.gens):
            self.gen_bus_idx[i] = int(self.bus_to_idx[int(getattr(g, "bus_i"))])
            self.Pmin_pu[i] = float(getattr(g, "Pmin")) / self.baseMVA
            self.Pmax_pu[i] = float(getattr(g, "Pmax")) / self.baseMVA
            self.Qmin_pu[i] = float(getattr(g, "Qmin")) / self.baseMVA
            self.Qmax_pu[i] = float(getattr(g, "Qmax")) / self.baseMVA
            self.cost_c2[i] = float(getattr(g, "cost_c2", 10.0))
            self.cost_c1[i] = float(getattr(g, "cost_c1", 100.0))
            self.cost_c0[i] = float(getattr(g, "cost_c0", 2500.0))

        # storage params
        self.sto_bus_idx = np.zeros((self.n_sto,), dtype=np.int64)
        self.sto_Emax = np.zeros((self.n_sto,), dtype=np.float64)
        self.sto_soc0 = np.zeros((self.n_sto,), dtype=np.float64)
        self.sto_Pch_max = np.zeros((self.n_sto,), dtype=np.float64)
        self.sto_Pdis_max = np.zeros((self.n_sto,), dtype=np.float64)
        self.sto_eta_ch = np.zeros((self.n_sto,), dtype=np.float64)
        self.sto_eta_dis = np.zeros((self.n_sto,), dtype=np.float64)
        self.sto_q_coeff = np.zeros((self.n_sto,), dtype=np.float64)
        for i, s in enumerate(self.stos):
            self.sto_bus_idx[i] = int(self.bus_to_idx[int(getattr(s, "bus_i"))])
            self.sto_Emax[i] = float(getattr(s, "Emax"))
            self.sto_soc0[i] = float(getattr(s, "soc", 0.5))
            self.sto_Pch_max[i] = float(getattr(s, "P_ch")) / 1000.0
            self.sto_Pdis_max[i] = abs(float(getattr(s, "P_dis"))) / 1000.0
            self.sto_eta_ch[i] = float(getattr(s, "eta_ch", 0.95))
            self.sto_eta_dis[i] = float(getattr(s, "eta_dis", 0.95))
            self.sto_q_coeff[i] = float(s.calculate_Qg(Pg=1.0))

        # renewable params
        self.ren_bus_idx = np.zeros((self.n_ren,), dtype=np.int64)
        self.ren_curt_cost = np.zeros((self.n_ren,), dtype=np.float64)
        self.ren_q_coeff = np.zeros((self.n_ren,), dtype=np.float64)
        for i, r in enumerate(self.rens):
            self.ren_bus_idx[i] = int(self.bus_to_idx[int(getattr(r, "bus_i"))])
            self.ren_curt_cost[i] = float(getattr(r, "curt_cost", 100000.0))
            self.ren_q_coeff[i] = float(r.calculate_Qg(Pg=1.0))

        # EV per-event params
        self.ev_bus_idx = np.full((self.n_ev,), -1, dtype=np.int64)
        self.ev_station_idx = np.full((self.n_ev,), -1, dtype=np.int64)
        self.ev_vehicle_idx = np.full((self.n_ev,), -1, dtype=np.int64)
        self.ev_ev_i = np.full((self.n_ev,), -1, dtype=np.int64)
        self.ev_cap_kwh = np.zeros((self.n_ev,), dtype=np.float64)
        self.ev_pch_max_kw = np.zeros((self.n_ev,), dtype=np.float64)
        self.ev_pdis_max_kw = np.zeros((self.n_ev,), dtype=np.float64)
        self.ev_eta_ch = np.zeros((self.n_ev,), dtype=np.float64)
        self.ev_eta_dis = np.zeros((self.n_ev,), dtype=np.float64)
        self.ev_dis_ok = np.ones((self.n_ev,), dtype=np.int64)
        self.ev_min_soc = np.zeros((self.n_ev,), dtype=np.float64)
        self.ev_arr_k = np.zeros((self.n_ev,), dtype=np.int64)
        self.ev_start_k = np.zeros((self.n_ev,), dtype=np.int64)
        self.ev_dep_k = np.zeros((self.n_ev,), dtype=np.int64)
        self.ev_active_slot = np.zeros((self.K_ev, self.n_ev), dtype=np.int64)
        self.ev_dt_slot_s = np.zeros((self.K_ev, self.n_ev), dtype=np.float64)
        self.vehicle_ids = np.asarray(sorted(int(eid) for eid in self.ev_by_id.keys()), dtype=np.int64)
        self.n_vehicle = int(len(self.vehicle_ids))
        self.vehicle_to_idx = {int(eid): i for i, eid in enumerate(self.vehicle_ids)}
        self.vehicle_soc0 = np.zeros((self.n_vehicle,), dtype=np.float64)
        self.vehicle_soc = np.zeros((self.n_vehicle,), dtype=np.float64)
        for vi, ev_i in enumerate(self.vehicle_ids):
            self.vehicle_soc0[vi] = _clamp(float(getattr(self.ev_by_id[int(ev_i)], "soc", 0.5)), 0.0, 1.0)
        self.station_energy_kwh = np.zeros((self.n_stn,), dtype=np.float64)
        self.station_req_cum_kwh = np.zeros((self.n_stn, self.K_ev + 1), dtype=np.float64)
        self.station_due_flag = np.zeros((self.n_stn, self.K_ev + 1), dtype=np.int64)
        self._ev_obj_per_event: List[Optional[EV]] = [None] * self.n_ev
        self._ev_v2g_minsoc: np.ndarray = np.zeros((self.n_ev,), dtype=np.float64)
        self._init_ev_static_and_activity()
        self._init_station_due_demand()
        self._init_exogenous_sequences()

        self._active_evs_by_k: List[np.ndarray] = []
        for k in range(self.K_ev):
            idxs = np.nonzero(self.ev_active_slot[k, :])[0].astype(np.int64)
            self._active_evs_by_k.append(idxs)

        # action/obs spaces
        self._layout = self._build_action_layout()
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._action_dim(),), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim(),), dtype=np.float32)

        # state variables
        self._t = 0
        self.sto_soc = np.zeros((self.n_sto,), dtype=np.float64)
        self.v2 = np.ones((self.nb,), dtype=np.float64)
        self.Pij = np.zeros((self.ne,), dtype=np.float64)
        self.Qij = np.zeros((self.ne,), dtype=np.float64)

        # last actions decoded (for obs)
        self.last_Pg_pu = np.zeros((self.n_gen,), dtype=np.float64)
        self.last_Qg_pu = np.zeros((self.n_gen,), dtype=np.float64)
        self.last_ren_frac = np.zeros((self.n_ren,), dtype=np.float64)
        self.last_sto_pnet_pu = np.zeros((self.n_sto,), dtype=np.float64)
        self.last_ev_pnet_pu = np.zeros((self.n_ev,), dtype=np.float64)
        self.last_stn_pnet_mw = np.zeros((self.n_stn,), dtype=np.float64)

        # last grid buy/sell
        self.last_P_buy_pu = 0.0
        self.last_P_sell_pu = 0.0
        self.last_Q_grid_pu = 0.0
        self.last_grid_P_excess_pu = 0.0
        self.last_grid_Q_excess_pu = 0.0

    def _action_dim(self) -> int:
        return int(self._layout.stn_mag.stop)

    def _obs_dim(self) -> int:
        return (
            3
            + 3 * self.nb
            + 2 * self.n_gen
            + 2 * self.n_ren
            + 2 * self.n_sto
            + 8 * self.n_stn
        )

    def _build_action_layout(self) -> ActionLayout:
        idx = 0
        n_stnslot = self.n_stn * self.Ksub
        stn_mode = slice(idx, idx + n_stnslot)
        idx += n_stnslot
        stn_mag = slice(idx, idx + n_stnslot)
        idx += n_stnslot

        return ActionLayout(
            stn_mode=stn_mode,
            stn_mag=stn_mag,
            n_stn=self.n_stn,
            Ksub=self.Ksub,
        )

    def _precompute_tree_orders(self) -> Tuple[List[int], List[int]]:
        depth: Dict[int, int] = {self.slack: 0}
        stack = [self.slack]
        order: List[int] = []
        while stack:
            u = int(stack.pop())
            order.append(u)
            for c in self.children.get(u, []):
                c = int(c)
                depth[c] = depth[u] + 1
                stack.append(c)

        fwd = sorted(order, key=lambda x: depth.get(int(x), 0))
        post = sorted(order, key=lambda x: depth.get(int(x), 0), reverse=True)
        return fwd, post

    def _init_ev_static_and_activity(self) -> None:
        for ei, eid in enumerate(self.event_ids):
            e = self.event_by_id[eid]
            ev_i = int(e["ev_i"])
            station_i = int(e["station_i"])
            bus_i = int(self.station_bus.get(station_i, 0))

            self.ev_ev_i[ei] = int(ev_i)
            self.ev_bus_idx[ei] = int(self.bus_to_idx.get(bus_i, -1))
            self.ev_station_idx[ei] = int(self.station_to_idx.get(int(station_i), -1))
            self.ev_vehicle_idx[ei] = int(self.vehicle_to_idx.get(ev_i, -1))

            arr = int(e.get("arrival_t", 0))
            st = int(e.get("start_t", arr))
            dep = int(e.get("departure_t", st))
            if dep < st or st < 0:
                dep = st
            self.ev_arr_k[ei] = int(arr // self.ev_step_s)
            self.ev_start_k[ei] = int(st // self.ev_step_s)
            self.ev_dep_k[ei] = int(_ceil_div(dep, self.ev_step_s))

            ev_obj = self.ev_by_id.get(ev_i, None)
            self._ev_obj_per_event[ei] = ev_obj

            # default-safe reads
            self.ev_cap_kwh[ei] = float(getattr(ev_obj, "capacity", 0.0))
            self.ev_pch_max_kw[ei] = float(getattr(ev_obj, "p_ch", 0.0))
            self.ev_pdis_max_kw[ei] = float(getattr(ev_obj, "p_dis", 0.0))
            self.ev_eta_ch[ei] = float(getattr(ev_obj, "eta_ch", 1.0))
            self.ev_eta_dis[ei] = float(getattr(ev_obj, "eta_dis", 1.0))

            minsoc = 0.0
            if ev_obj is not None and hasattr(ev_obj, "v2g_minsoc"):
                try:
                    minsoc = float(ev_obj.v2g_minsoc())
                except Exception:
                    minsoc = 0.0
            self._ev_v2g_minsoc[ei] = minsoc

            self.ev_dis_ok[ei] = 1 if bool(getattr(ev_obj, "v2g_cap", False)) else 0
            self.ev_min_soc[ei] = minsoc

            for k in range(self.K_ev):
                t0 = k * self.ev_step_s
                t1 = min(self.horizon_s, (k + 1) * self.ev_step_s)
                ov = _sec_overlap(t0, t1, st, dep)
                if ov > 0:
                    self.ev_active_slot[k, ei] = 1
                    self.ev_dt_slot_s[k, ei] = float(ov)
                else:
                    self.ev_active_slot[k, ei] = 0
                    self.ev_dt_slot_s[k, ei] = 0.0

    def _init_station_due_demand(self) -> None:
        self.station_req_cum_kwh[:] = 0.0
        self.station_due_flag[:] = 0
        for ei, eid in enumerate(self.event_ids):
            si = int(self.ev_station_idx[ei])
            if si < 0:
                continue
            ev_i = int(self.ev_ev_i[ei])
            ev_obj = self.ev_by_id.get(ev_i)
            if ev_obj is None:
                continue
            e = self.event_by_id[eid]
            soc0 = float(e.get("soc_init", getattr(ev_obj, "soc", 0.5)))
            req_kwh = max(0.0, (float(self.cfg.ev_target_soc) - soc0) * float(ev_obj.capacity))
            dep_k = int(min(max(0, self.ev_dep_k[ei]), self.K_ev))
            self.station_req_cum_kwh[si, dep_k] += float(req_kwh)
            self.station_due_flag[si, dep_k] = 1
        self.station_req_cum_kwh = np.cumsum(self.station_req_cum_kwh, axis=1)

    def _init_exogenous_sequences(self) -> None:
        self.exo_gen_P_pu = np.zeros((self.T, self.n_gen), dtype=np.float64)
        self.exo_gen_Q_pu = np.zeros((self.T, self.n_gen), dtype=np.float64)
        self.exo_ren_Pr_pu = np.zeros((self.T, self.n_ren), dtype=np.float64)
        self.exo_sto_pnet_pu = np.zeros((self.T, self.n_sto), dtype=np.float64)
        self.exo_sto_soc = np.tile(self.sto_soc0.reshape(1, -1), (self.T, 1)) if self.n_sto > 0 else np.zeros((self.T, 0), dtype=np.float64)

        if not bool(self.cfg.use_baseline_exogenous):
            return

        doc = load_baseline_exogenous(self.cfg.baseline_exogenous_path)
        if not doc:
            raise FileNotFoundError(f"baseline exogenous file not found or empty: {self.cfg.baseline_exogenous_path}")

        def _load_matrix(key: str, rows: int, cols: int) -> np.ndarray:
            arr = np.asarray(doc.get(key, []), dtype=np.float64)
            if arr.shape != (rows, cols):
                raise ValueError(f"{key} shape mismatch: expected {(rows, cols)}, got {arr.shape}")
            return arr

        self.exo_gen_P_pu = _load_matrix("gen_P_pu", self.T, self.n_gen)
        self.exo_gen_Q_pu = _load_matrix("gen_Q_pu", self.T, self.n_gen)
        self.exo_ren_Pr_pu = _load_matrix("ren_Pr_pu", self.T, self.n_ren)
        self.exo_sto_pnet_pu = _load_matrix("sto_pnet_pu", self.T, self.n_sto)
        self.exo_sto_soc = _load_matrix("sto_soc", self.T, self.n_sto)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._t = 0
        self.v2[:] = 1.0
        self.Pij[:] = 0.0
        self.Qij[:] = 0.0
        self.sto_soc[:] = self.sto_soc0
        self.vehicle_soc[:] = self.vehicle_soc0
        self.station_energy_kwh[:] = 0.0

        self._simulate_step(np.zeros((self._action_dim(),), dtype=np.float32), update_states=True)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self._action_dim():
            raise ValueError(f"Action dim mismatch: got {action.shape[0]}, expected {self._action_dim()}")
        if not self.action_space.contains(action):
            raise ValueError("Action out of bounds [-1,1].")

        step_cost, info = self._simulate_step(action, update_states=True)

        self._t += 1
        terminated = self._t >= self.T
        truncated = False

        if terminated:
            if float(self.cfg.storage_end_penalty) > 0.0 and self.n_sto > 0:
                short = np.maximum(0.0, float(self.cfg.storage_end_soc) - self.sto_soc)
                term_pen = float(self.cfg.storage_end_penalty) * float(np.sum(short))
                step_cost += term_pen
                info["storage_terminal_penalty"] = float(term_pen)

        raw_reward = -float(step_cost)
        reward = raw_reward * float(self.cfg.reward_scale)
        obs = self._get_obs()

        info["step_cost_total"] = float(step_cost)
        info["raw_reward"] = float(raw_reward)
        info["scaled_reward"] = float(reward)

        return obs, reward, terminated, truncated, info

    def _simulate_step(self, action: np.ndarray, *, update_states: bool) -> Tuple[float, Dict[str, Any]]:
        tt = int(self._t)

        # fixed non-EV trajectories from baseline
        Pg_pu = self.exo_gen_P_pu[tt, :].copy()
        Qg_pu = self.exo_gen_Q_pu[tt, :].copy()
        self.last_Pg_pu[:] = Pg_pu
        self.last_Qg_pu[:] = Qg_pu

        P_inj = np.zeros((self.nb,), dtype=np.float64)
        Q_inj = np.zeros((self.nb,), dtype=np.float64)
        for gi in range(self.n_gen):
            bi = int(self.gen_bus_idx[gi])
            P_inj[bi] += float(Pg_pu[gi])
            Q_inj[bi] += float(Qg_pu[gi])

        Pav = self.Pav_pu[tt, :]
        Pr = np.minimum(np.maximum(self.exo_ren_Pr_pu[tt, :], 0.0), Pav)
        frac = np.divide(Pr, np.maximum(Pav, 1e-9))
        self.last_ren_frac[:] = frac
        curt = np.maximum(0.0, Pav - Pr)
        for ri in range(self.n_ren):
            bi = int(self.ren_bus_idx[ri])
            P_inj[bi] += float(Pr[ri])
            Q_inj[bi] += float(self.ren_q_coeff[ri] * Pr[ri])

        pnet_sto_pu = np.zeros((self.n_sto,), dtype=np.float64)
        soc_pen = 0.0
        for si in range(self.n_sto):
            pnet_sto_pu[si] = float(self.exo_sto_pnet_pu[tt, si])
            soc_next = _clamp(float(self.exo_sto_soc[tt, si]), 0.0, 1.0)
            if update_states:
                self.sto_soc[si] = soc_next

            bi = int(self.sto_bus_idx[si])
            P_inj[bi] += float(pnet_sto_pu[si])
            Q_inj[bi] += float(self.sto_q_coeff[si] * pnet_sto_pu[si])

        self.last_sto_pnet_pu[:] = pnet_sto_pu

        # EV actions: station-level action, vehicle-level greedy allocation
        ev_pnet_pu = np.zeros((self.n_ev,), dtype=np.float64)
        stn_net_mw = np.zeros((self.n_stn,), dtype=np.float64)
        v2g_value = 0.0
        ev_soc_pen = 0.0
        ev_depart_pen_step = 0.0
        stn_energy_delta_kwh = np.zeros((self.n_stn,), dtype=np.float64)

        if self.n_ev > 0 and self.Ksub > 0 and self.n_stn > 0:
            stn_mode_raw = action[self._layout.stn_mode]
            stn_mag = np.clip(_mag01(action[self._layout.stn_mag]), 0.0, 0.0 + 1.0)
            net_kw_avg = np.zeros((self.n_ev,), dtype=np.float64)
            opf_step_s_f = float(self.opf_step_s)
            ev_step_s_i = int(self.ev_step_s)

            for sub in range(self.Ksub):
                k_global = tt * self.Ksub + sub
                if k_global < 0 or k_global >= self.K_ev:
                    continue
                price_k = float(self.price_ev[k_global])
                active_eis = self._active_evs_by_k[k_global]
                if active_eis.size == 0:
                    continue

                by_station: Dict[int, List[int]] = {si: [] for si in range(self.n_stn)}
                for ei in active_eis:
                    si = int(self.ev_station_idx[ei])
                    if si >= 0:
                        by_station[si].append(int(ei))

                base_idx = sub * self.n_stn
                for si in range(self.n_stn):
                    station_eis = by_station.get(si, [])
                    if not station_eis:
                        continue
                    mode = _decode_sign_mode(float(stn_mode_raw[base_idx + si]))
                    mag = float(stn_mag[base_idx + si])
                    pch_target = 0.0
                    pdis_target = 0.0
                    ranked_ch: List[Tuple[float, int, float]] = []
                    ranked_dis: List[Tuple[float, int, float]] = []

                    for ei in station_eis:
                        dt_eff_s = float(self.ev_dt_slot_s[k_global, ei])
                        if dt_eff_s <= 0.0:
                            continue
                        vi = int(self.ev_vehicle_idx[ei])
                        if vi < 0:
                            continue
                        ev_obj = self._ev_obj_per_event[int(ei)]
                        if ev_obj is None:
                            continue
                        soc = float(self.vehicle_soc[vi])
                        cap = float(self.ev_cap_kwh[ei])
                        ev_obj.soc = soc
                        remain_s = max(0, int(self.ev_dep_k[ei]) * ev_step_s_i - int(k_global * ev_step_s_i))
                        wait_s = max(0, int(self.event_by_id[self.event_ids[ei]].get("start_t", 0)) - int(self.event_by_id[self.event_ids[ei]].get("arrival_t", 0)))

                        pch_max = max(0.0, min(float(self.ev_pch_max_kw[ei]), float(ev_obj.charge_limit())))
                        if pch_max > 1e-9 and soc < min(1.0, float(self.cfg.ev_target_soc)) - 1e-9:
                            score = _event_priority(ev_obj, soc, wait_s, remain_s)
                            ranked_ch.append((score, int(ei), pch_max))
                            pch_target += pch_max

                        dis_ok = bool(self.ev_dis_ok[ei]) and bool(ev_obj.is_v2g(sell_price=price_k))
                        min_soc_dyn = float(self._ev_v2g_minsoc[ei]) if dis_ok else 0.0
                        if dis_ok and soc > min_soc_dyn + 1e-9:
                            energy_margin = max(0.0, (soc - min_soc_dyn) * cap)
                            dt_h_eff = dt_eff_s / 3600.0
                            pdis_cap = min(float(self.ev_pdis_max_kw[ei]), energy_margin * float(ev_obj.eta_dis) / max(1e-9, dt_h_eff))
                            if pdis_cap > 1e-9:
                                score = _v2g_priority(ev_obj, soc, remain_s)
                                ranked_dis.append((score, int(ei), pdis_cap))
                                pdis_target += pdis_cap

                    if mode < 0:
                        pch_target *= mag
                        pdis_target = 0.0
                    else:
                        pdis_target *= mag
                        pch_target = 0.0

                    ranked_ch.sort(key=lambda x: (-x[0], x[1]))
                    ranked_dis.sort(key=lambda x: (-x[0], x[1]))

                    rem = float(pch_target)
                    for _, ei, pmax in ranked_ch:
                        if rem <= 1e-9:
                            break
                        vi = int(self.ev_vehicle_idx[ei])
                        if vi < 0:
                            continue
                        dt_eff_s = float(self.ev_dt_slot_s[k_global, ei])
                        dt_h_eff = dt_eff_s / 3600.0
                        soc = float(self.vehicle_soc[vi])
                        cap = float(self.ev_cap_kwh[ei])
                        ev_obj = self._ev_obj_per_event[int(ei)]
                        if ev_obj is None:
                            continue
                        ev_obj.soc = soc
                        eta = max(1e-6, float(ev_obj.charge_efficiency(min(rem, pmax))))
                        soc_room_kwh = max(0.0, (min(1.0, float(self.cfg.ev_target_soc)) - soc) * cap)
                        p_soc = soc_room_kwh / max(1e-9, eta * dt_h_eff)
                        p = min(rem, pmax, p_soc)
                        if p <= 1e-9:
                            continue
                        soc_next = _clamp(soc + (p * eta * dt_h_eff) / max(1e-6, cap), 0.0, 1.0)
                        if update_states:
                            self.vehicle_soc[vi] = soc_next
                        net_kw_avg[ei] += (dt_eff_s / max(1.0, opf_step_s_f)) * p
                        stn_net_mw[si] += (dt_eff_s / max(1.0, opf_step_s_f)) * (p / 1000.0)
                        stn_energy_delta_kwh[si] += p * eta * dt_h_eff
                        rem -= p

                    rem = float(pdis_target)
                    for _, ei, pmax in ranked_dis:
                        if rem <= 1e-9:
                            break
                        vi = int(self.ev_vehicle_idx[ei])
                        if vi < 0:
                            continue
                        dt_eff_s = float(self.ev_dt_slot_s[k_global, ei])
                        dt_h_eff = dt_eff_s / 3600.0
                        soc = float(self.vehicle_soc[vi])
                        cap = float(self.ev_cap_kwh[ei])
                        ev_obj = self._ev_obj_per_event[int(ei)]
                        if ev_obj is None:
                            continue
                        min_soc_dyn = float(self._ev_v2g_minsoc[ei])
                        energy_margin = max(0.0, (soc - min_soc_dyn) * cap)
                        p_soc = energy_margin * float(ev_obj.eta_dis) / max(1e-9, dt_h_eff)
                        p = min(rem, pmax, p_soc)
                        if p <= 1e-9:
                            continue
                        soc_next = _clamp(soc - (p / max(1e-6, float(ev_obj.eta_dis)) * dt_h_eff) / max(1e-6, cap), 0.0, 1.0)
                        if update_states:
                            self.vehicle_soc[vi] = soc_next
                        net_kw_avg[ei] -= (dt_eff_s / max(1.0, opf_step_s_f)) * p
                        stn_net_mw[si] -= (dt_eff_s / max(1.0, opf_step_s_f)) * (p / 1000.0)
                        stn_energy_delta_kwh[si] -= (p / max(1e-6, float(ev_obj.eta_dis))) * dt_h_eff
                        v2g_value += float(price_k * p * dt_h_eff)
                        rem -= p

            if update_states:
                self.station_energy_kwh[:] = np.maximum(0.0, self.station_energy_kwh + stn_energy_delta_kwh)

            for ei in range(self.n_ev):
                bi = int(self.ev_bus_idx[ei])
                if bi < 0:
                    continue
                ev_pnet_pu[ei] = (float(net_kw_avg[ei]) / 1000.0) / self.baseMVA
                P_inj[bi] -= float(ev_pnet_pu[ei])

            now_s_end = (tt + 1) * self.opf_step_s
            k_end = int(min(self.K_ev, max(0, (tt + 1) * self.Ksub)))
            for si in range(self.n_stn):
                due_now = bool(np.any(self.station_due_flag[si, max(0, k_end - self.Ksub + 1):k_end + 1]))
                if due_now:
                    due_req_cum = float(self.station_req_cum_kwh[si, k_end])
                    short_kwh = max(0.0, due_req_cum - float(self.station_energy_kwh[si]))
                    ev_depart_pen_step += float(self.cfg.ev_depart_penalty) * short_kwh

        self.last_ev_pnet_pu[:] = ev_pnet_pu
        self.last_stn_pnet_mw[:] = stn_net_mw
        soc_pen += ev_soc_pen

        # Slack exchange
        Pd = self.Pd_pu[tt, :]
        Qd = self.Qd_pu[tt, :]
        netP = Pd - P_inj
        netQ = Qd - Q_inj
        P_req = float(np.sum(netP))
        Q_req = float(np.sum(netQ))

        P_buy_pu = float(max(P_req, 0.0))
        P_sell_pu = float(max(-P_req, 0.0))
        Q_grid_pu = float(Q_req)

        self.last_P_buy_pu = P_buy_pu
        self.last_P_sell_pu = P_sell_pu
        self.last_Q_grid_pu = Q_grid_pu

        Pcap_pu = float(self.cfg.grid_pmax_mw) / max(1e-12, self.baseMVA)
        Qcap_pu = float(self.cfg.grid_qmax_mvar) / max(1e-12, self.baseMVA)
        P_excess = float(max(0.0, P_buy_pu - Pcap_pu) + max(0.0, P_sell_pu - Pcap_pu))
        Q_excess = float(max(0.0, abs(Q_grid_pu) - Qcap_pu))

        self.last_grid_P_excess_pu = P_excess
        self.last_grid_Q_excess_pu = Q_excess

        grid_pen = 0.0
        if (P_excess > 0.0) or (Q_excess > 0.0):
            grid_pen = float(self.cfg.grid_limit_penalty) * float(P_excess * P_excess + Q_excess * Q_excess)

        # linear distflow
        subtreeP = netP.copy()
        subtreeQ = netQ.copy()
        Pij = np.zeros((self.ne,), dtype=np.float64)
        Qij = np.zeros((self.ne,), dtype=np.float64)
        for j in self._post_order:
            if int(j) == self.slack:
                continue
            i = int(self.parent[int(j)])
            eidx = self._edge_index.get((i, int(j)), None)
            if eidx is None:
                continue
            Pij[eidx] = subtreeP[self.bus_to_idx[int(j)]]
            Qij[eidx] = subtreeQ[self.bus_to_idx[int(j)]]
            subtreeP[self.bus_to_idx[i]] += subtreeP[self.bus_to_idx[int(j)]]
            subtreeQ[self.bus_to_idx[i]] += subtreeQ[self.bus_to_idx[int(j)]]

        v2 = np.ones((self.nb,), dtype=np.float64)
        v2[self.slack_idx] = 1.0
        for i in self._fwd_order:
            for c in self.children.get(int(i), []):
                j = int(c)
                eidx = self._edge_index.get((int(i), j), None)
                if eidx is None:
                    continue
                ii = self.bus_to_idx[int(i)]
                jj = self.bus_to_idx[j]
                v2[jj] = v2[ii] - 2.0 * (self.r_pu[eidx] * Pij[eidx] + self.x_pu[eidx] * Qij[eidx])

        if update_states:
            self.v2[:] = v2
            self.Pij[:] = Pij
            self.Qij[:] = Qij

        # penalties
        volt_pen = 0.0
        if float(self.cfg.v_penalty) > 0:
            for bi, bid in enumerate(self.bus_ids):
                bus = self.grid.get_bus(int(bid))
                vmax2 = float(getattr(bus, "Vmax")) ** 2
                vmin2 = float(getattr(bus, "Vmin")) ** 2
                Vu = max(0.0, v2[bi] - vmax2)
                Vl = max(0.0, vmin2 - v2[bi])
                volt_pen += float(self.cfg.v_penalty) * (Vu + Vl)

        br_pen = 0.0
        for k in range(self.ne):
            S = math.sqrt(float(Pij[k]) ** 2 + float(Qij[k]) ** 2)
            lim = float(self.smax_pu[k])
            if S > lim + 1e-12:
                br_pen += float(self.cfg.branch_overload_penalty) * ((S - lim) ** 2)

        # objective components
        Pg_MW = Pg_pu * self.baseMVA
        gen_cost = float(np.sum(self.cost_c2 * Pg_MW * Pg_MW + self.cost_c1 * Pg_MW + self.cost_c0)) * float(self.delta_t)

        price = float(self.price[tt])
        Pbuy_MW = float(P_buy_pu * self.baseMVA)
        Psell_MW = float(P_sell_pu * self.baseMVA)
        grid_buy_cost = Pbuy_MW * price * float(self.delta_t) * 1000.0
        grid_sell_rev = Psell_MW * price * float(self.delta_t) * 1000.0

        curt_MW = curt * self.baseMVA
        curt_cost = float(np.sum(self.ren_curt_cost * curt_MW)) * float(self.delta_t)

        v2g_reward = float(self.cfg.v2g_reward_coeff) * float(v2g_value)

        step_cost = (
            gen_cost
            + grid_buy_cost
            - grid_sell_rev
            + curt_cost
            + volt_pen
            - v2g_reward
            + br_pen
            + soc_pen
            + grid_pen
            + ev_depart_pen_step
        )

        info = {
            "t": tt,
            "price": price,
            "P_buy_MW": float(Pbuy_MW),
            "P_sell_MW": float(Psell_MW),
            "Q_grid_MVar": float(Q_grid_pu * self.baseMVA),
            "grid_cap_P_MW": float(self.cfg.grid_pmax_mw),
            "grid_cap_Q_MVar": float(self.cfg.grid_qmax_mvar),
            "grid_P_req_pu": float(P_req),
            "grid_Q_req_pu": float(Q_req),
            "grid_P_excess_pu": float(P_excess),
            "grid_Q_excess_pu": float(Q_excess),
            "grid_pen": float(grid_pen),
            "gen_cost": float(gen_cost),
            "grid_buy_cost": float(grid_buy_cost),
            "grid_sell_rev": float(grid_sell_rev),
            "curt_cost": float(curt_cost),
            "volt_pen": float(volt_pen),
            "branch_pen": float(br_pen),
            "soc_pen": float(soc_pen),
            "ev_depart_pen_step": float(ev_depart_pen_step),
            "v2g_value": float(v2g_value),
            "v2g_reward": float(v2g_reward),
            "step_cost": float(step_cost),
        }
        return float(step_cost), info

    def _norm(self, x: float, lo: float, hi: float) -> float:
        if abs(hi - lo) < 1e-12:
            return 0.0
        y = 2.0 * (float(x) - float(lo)) / (float(hi) - float(lo)) - 1.0
        return _clamp(y, -1.0, 1.0)

    def _station_obs_features(self, k_global: int) -> np.ndarray:
        feats = np.zeros((self.n_stn, 8), dtype=np.float64)
        if self.n_stn <= 0 or self.n_ev <= 0 or k_global < 0 or k_global >= self.K_ev:
            return feats

        for si in range(self.n_stn):
            station_eis = np.nonzero((self.ev_station_idx == si) & (self.ev_active_slot[k_global, :] == 1))[0]
            active_n = int(len(station_eis))
            avg_soc = 0.0
            pch_cap_mw = 0.0
            pdis_cap_mw = 0.0
            min_rem_h = float(self.horizon_s) / 3600.0
            due_req_cum_kwh = 0.0
            short_kwh = 0.0

            if active_n > 0:
                socs = []
                for ei in station_eis:
                    vi = int(self.ev_vehicle_idx[ei])
                    if vi < 0:
                        continue
                    soc = float(self.vehicle_soc[vi])
                    socs.append(soc)
                    ev_obj = self._ev_obj_per_event[int(ei)]
                    if ev_obj is not None:
                        ev_obj.soc = soc
                        pch_cap_mw += max(0.0, min(float(self.ev_pch_max_kw[ei]), float(ev_obj.charge_limit()))) / 1000.0
                        if bool(self.ev_dis_ok[ei]) and soc > float(self._ev_v2g_minsoc[ei]) + 1e-9:
                            pdis_cap_mw += max(0.0, float(self.ev_pdis_max_kw[ei])) / 1000.0
                    dep_s = int(self.ev_dep_k[ei]) * int(self.ev_step_s)
                    rem_h = max(0.0, dep_s - int(k_global * self.ev_step_s)) / 3600.0
                    min_rem_h = min(min_rem_h, rem_h)
                avg_soc = float(np.mean(socs))
            else:
                min_rem_h = 0.0

            k_obs = int(min(self.K_ev, max(0, k_global + 1)))
            due_req_cum_kwh = float(self.station_req_cum_kwh[si, k_obs])
            short_kwh = max(0.0, due_req_cum_kwh - float(self.station_energy_kwh[si]))

            feats[si, 0] = _clamp(float(active_n) / 50.0, 0.0, 1.0)
            feats[si, 1] = _clamp(avg_soc, 0.0, 1.0)
            feats[si, 2] = _clamp(pch_cap_mw / 1.0, 0.0, 1.0)
            feats[si, 3] = _clamp(pdis_cap_mw / 1.0, 0.0, 1.0)
            feats[si, 4] = _clamp(min_rem_h / 24.0, 0.0, 1.0)
            feats[si, 5] = _clamp((float(self.last_stn_pnet_mw[si]) + 1.0) / 2.0, 0.0, 1.0)
            feats[si, 6] = _clamp(due_req_cum_kwh / 2000.0, 0.0, 1.0)
            feats[si, 7] = _clamp(short_kwh / 2000.0, 0.0, 1.0)
        return feats

    def _get_obs(self) -> np.ndarray:
        tt = int(self._t)
        t_sec = tt * self.opf_step_s

        frac = float(t_sec) / float(max(1, self.horizon_s))
        time_sin = math.sin(2.0 * math.pi * frac)
        time_cos = math.cos(2.0 * math.pi * frac)

        pmin = float(np.min(self.price)) if self.T > 0 else 0.0
        pmax = float(np.max(self.price)) if self.T > 0 else 1.0
        price_n = self._norm(float(self.price[min(tt, self.T - 1)]), pmin, pmax)

        obs: List[float] = [float(time_sin), float(time_cos), float(price_n)]

        vm_min = float(self.cfg.obs_vm_min)
        vm_max = float(self.cfg.obs_vm_max)

        Pd = self.Pd_pu[min(tt, self.T - 1), :]
        Qd = self.Qd_pu[min(tt, self.T - 1), :]

        for bi in range(self.nb):
            vm = math.sqrt(max(0.0, float(self.v2[bi])))
            obs.append(self._norm(vm, vm_min, vm_max))
            obs.append(_clamp(float(Pd[bi]) / 2.0, -1.0, 1.0))
            obs.append(_clamp(float(Qd[bi]) / 2.0, -1.0, 1.0))

        for gi in range(self.n_gen):
            obs.append(self._norm(float(self.last_Pg_pu[gi]), float(self.Pmin_pu[gi]), float(self.Pmax_pu[gi])))
            obs.append(self._norm(float(self.last_Qg_pu[gi]), float(self.Qmin_pu[gi]), float(self.Qmax_pu[gi])))

        for ri in range(self.n_ren):
            obs.append(_clamp(2.0 * float(self.last_ren_frac[ri]) - 1.0, -1.0, 1.0))
            obs.append(_clamp(float(self.Pav_pu[min(tt, self.T - 1), ri]) / 1.0, -1.0, 1.0))

        for si in range(self.n_sto):
            pcap = max(1e-12, max(float(self.sto_Pch_max[si]), float(self.sto_Pdis_max[si])) / self.baseMVA)
            obs.append(_clamp(float(self.last_sto_pnet_pu[si]) / pcap, -1.0, 1.0))
            obs.append(_clamp(2.0 * float(self.sto_soc[si]) - 1.0, -1.0, 1.0))

        k_global = min(max(tt * self.Ksub, 0), max(0, self.K_ev - 1))
        st_feats = self._station_obs_features(k_global)
        for si in range(self.n_stn):
            obs.append(_clamp(2.0 * float(st_feats[si, 0]) - 1.0, -1.0, 1.0))  # active EV count
            obs.append(_clamp(2.0 * float(st_feats[si, 1]) - 1.0, -1.0, 1.0))  # avg SOC
            obs.append(_clamp(2.0 * float(st_feats[si, 2]) - 1.0, -1.0, 1.0))  # charge headroom
            obs.append(_clamp(2.0 * float(st_feats[si, 3]) - 1.0, -1.0, 1.0))  # discharge headroom
            obs.append(_clamp(2.0 * float(st_feats[si, 4]) - 1.0, -1.0, 1.0))  # min remaining time
            obs.append(_clamp(2.0 * float(st_feats[si, 5]) - 1.0, -1.0, 1.0))  # last station net load
            obs.append(_clamp(2.0 * float(st_feats[si, 6]) - 1.0, -1.0, 1.0))  # cumulative due demand
            obs.append(_clamp(2.0 * float(st_feats[si, 7]) - 1.0, -1.0, 1.0))  # current shortfall

        return np.asarray(obs, dtype=np.float32)

    def render(self):
        tt = int(self._t)
        vms = np.sqrt(np.maximum(0.0, self.v2))
        print(
            f"[t={tt}/{self.T}] Vm(min/mean/max)={float(np.min(vms)):.4f}/{float(np.mean(vms)):.4f}/{float(np.max(vms)):.4f} "
            f"| P_buy={self.last_P_buy_pu*self.baseMVA:.3f}MW P_sell={self.last_P_sell_pu*self.baseMVA:.3f}MW "
            f"| cap_excess(P,Q)=({self.last_grid_P_excess_pu*self.baseMVA:.3f}MW, {self.last_grid_Q_excess_pu*self.baseMVA:.3f}MVar)"
        )

    def close(self):
        return

def register_envs() -> None:
    register(id="Grid2AI-v0", entry_point="env.environment:Grid2AIEnv")

def make_env(
    grid_yaml: str = "config/IEEE33.yaml",
    *,
    cfg: Optional[Grid2AIEnvConfig] = None,
    seed: int = 0,
    render_mode: Optional[str] = None,
) -> Grid2AIEnv:
    if cfg is None:
        cfg = Grid2AIEnvConfig()
    return Grid2AIEnv(grid_yaml=grid_yaml, cfg=cfg, seed=seed, render_mode=render_mode)
