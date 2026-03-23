from __future__ import annotations
import yaml
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pyomo.environ import *
from utils import get_series_value
from DERs.ev import EV

@dataclass(frozen=True)
class LinDistFlowConfig:
    horizon_hours: int = 24
    resolution_minutes: int = 15
    solver_name: str = "gurobi"
    tee: bool = True
    time_yaml_path: str = "config/time.yaml"
    ev_yaml_path: str = "config/ev.yaml"
    queue_yaml_path: str = "config/event.yaml"
    storage_terminal_value_coeff: float = 1.0
    ev_resolution_seconds: int = 300
    v_penalty: float = 10000
    ev_depart_penalty: float = 20
    v2g_reward_coeff: float = 5.0
    ev_target_soc: float = 0.9
    time_offset_seconds: int = 0
    horizon_steps_override: Optional[int] = None
    events_override: Optional[List[Dict[str, Any]]] = None
    storage_soc0_override: Optional[Dict[int, float]] = None
    solver_options: Optional[Dict[str, Any]] = None

def _value(x: Any, default: float = 0.0) -> float:
    y = value(x)  
    if y is None:
        return float(default)
    return float(y)

def time_steps(horizon_hours: int, resolution_minutes: int) -> Tuple[int, float]:
    delta_t = float(resolution_minutes) / 60.0
    T = int(horizon_hours / delta_t)
    return T, delta_t

def build_price_series(cfg: LinDistFlowConfig, T: int, delta_t: float) -> List[float]:
    step_s = int(float(delta_t) * 3600.0)
    offset_s = int(getattr(cfg, "time_offset_seconds", 0))
    prices: List[float] = []
    for tt in range(int(T)):
        t_sec = offset_s + tt * step_s
        p = get_series_value(cfg.time_yaml_path, "price", t=t_sec)
        prices.append(float(p))
    return prices

def renewable_q_coeff(r: Any) -> float:
    return float(r.calculate_Qg(Pg=1.0))

def storage_q_coeff(s: Any) -> float:
    return float(s.calculate_Qg(Pg=1.0))

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


def _event_priority(ev_obj: EV, soc: float, wait_s: int, remain_s: int) -> float:
    urgency = 1.0 / max(900.0, float(remain_s))
    low_soc = max(0.0, 1.0 - float(soc))
    wait_h = float(wait_s) / 3600.0
    return 3.0 * urgency + 1.8 * low_soc + 0.15 * wait_h


def _v2g_priority(ev_obj: EV, soc: float, remain_s: int) -> float:
    slack_h = max(0.0, float(remain_s) / 3600.0)
    base = 2.0 * max(0.0, float(soc) - float(ev_obj.v2g_minsoc())) + 0.12 * slack_h
    ev_type = str(getattr(ev_obj, "type", "V2G")).upper()
    type_weight = 1.2 if ev_type == "V2G" else 0.9 if ev_type == "BUS" else 1.0
    return type_weight * base


def _weighted_power_split(candidates: List[Tuple[float, int, float]], target_kw: float) -> Dict[int, float]:
    alloc: Dict[int, float] = {}
    remaining = max(0.0, float(target_kw))
    if remaining <= 1e-9 or not candidates:
        return alloc

    caps: Dict[int, float] = {}
    weights: Dict[int, float] = {}
    active: List[int] = []
    for score, item_id, cap in candidates:
        cap_f = max(0.0, float(cap))
        if cap_f <= 1e-9:
            continue
        caps[int(item_id)] = cap_f
        weights[int(item_id)] = max(1e-6, float(score))
        active.append(int(item_id))

    while remaining > 1e-9 and active:
        weight_sum = sum(weights[item_id] for item_id in active)
        if weight_sum <= 1e-12:
            break
        next_active: List[int] = []
        delivered = 0.0
        for item_id in active:
            share = remaining * (weights[item_id] / weight_sum)
            give = min(caps[item_id], share)
            if give > 1e-9:
                alloc[item_id] = alloc.get(item_id, 0.0) + float(give)
                caps[item_id] -= float(give)
                delivered += float(give)
            if caps[item_id] > 1e-9:
                next_active.append(item_id)
        if delivered <= 1e-9:
            break
        remaining -= delivered
        active = next_active

    return alloc


def allocate_station_power(
    *,
    station_power_kw: Dict[Tuple[int, int], Dict[str, float]],
    events: List[Dict[str, Any]],
    ev_by_id: Dict[int, EV],
    station_ids: List[int],
    K_ev: int,
    ev_step_s: int,
    target_soc: float,
    ev_slot_price: Dict[int, float],
) -> Dict[str, Any]:
    event_by_id = {int(ev["event_i"]): dict(ev) for ev in events}
    state: Dict[int, Dict[str, float]] = {}
    station_events: Dict[int, List[int]] = {int(sid): [] for sid in station_ids}
    power_ch: Dict[Tuple[int, int], float] = {}
    power_dis: Dict[Tuple[int, int], float] = {}
    unmet_departure_kwh: Dict[int, float] = {}
    target_soc_eff = min(1.0, float(target_soc))

    for ev in events:
        eid = int(ev["event_i"])
        ev_i = int(ev["ev_i"])
        sid = int(ev["station_i"])
        arr = int(ev["arrival_t"])
        st = int(ev.get("start_t", arr))
        dep = int(ev["departure_t"])
        station_events.setdefault(sid, []).append(eid)
        ev_obj = ev_by_id.get(ev_i)
        if ev_obj is None:
            continue
        soc0 = float(ev.get("soc_init", getattr(ev_obj, "soc", 0.5)))
        state[eid] = {
            "soc": soc0,
            "arr": float(arr),
            "start": float(st),
            "dep": float(dep),
            "wait": max(0.0, float(st - arr)),
            "cap": float(ev_obj.capacity),
        }

    for sid in station_ids:
        eids = sorted(station_events.get(int(sid), []))
        for k in range(int(K_ev)):
            t0 = int(k * ev_step_s)
            t1 = int((k + 1) * ev_step_s)
            dt_h = float(ev_step_s) / 3600.0
            pch_target = float(station_power_kw.get((int(sid), int(k)), {}).get("pch", 0.0))
            pdis_target = float(station_power_kw.get((int(sid), int(k)), {}).get("pdis", 0.0))
            active = [eid for eid in eids if eid in state and int(state[eid]["start"]) < t1 and int(state[eid]["dep"]) > t0]
            price_k = float(ev_slot_price.get(int(k), 0.0))

            charge_info: Dict[int, Tuple[float, float]] = {}
            discharge_info: Dict[int, Tuple[float, float]] = {}
            charge_pool: List[int] = []
            discharge_pool: List[int] = []
            dual_pool: List[Tuple[float, float, int]] = []

            if active and (pch_target > 1e-9 or pdis_target > 1e-9):
                for eid in active:
                    ev = event_by_id[eid]
                    ev_obj = ev_by_id.get(int(ev["ev_i"]))
                    if ev_obj is None:
                        continue
                    soc = float(state[eid]["soc"])
                    cap = float(state[eid]["cap"])
                    remain_s = max(0, int(state[eid]["dep"]) - t0)
                    ev_obj.soc = soc

                    if pch_target > 1e-9:
                        pmax_ch = max(0.0, min(float(ev_obj.p_ch), float(ev_obj.charge_limit())))
                        if pmax_ch > 1e-9 and soc < target_soc_eff - 1e-9:
                            ch_score = _event_priority(ev_obj, soc, int(state[eid]["wait"]), remain_s)
                            eta_ch = max(1e-6, float(ev_obj.charge_efficiency(pmax_ch)))
                            soc_room_kwh = max(0.0, (target_soc_eff - soc) * cap)
                            p_soc_ch = soc_room_kwh / max(1e-9, eta_ch * dt_h)
                            charge_info[eid] = (ch_score, min(pmax_ch, p_soc_ch))

                    if pdis_target > 1e-9 and ev_obj.is_v2g(sell_price=price_k):
                        pmax_dis = max(0.0, float(ev_obj.p_dis))
                        if pmax_dis > 1e-9:
                            target_surplus = max(0.0, soc - target_soc_eff)
                            dis_score = _v2g_priority(ev_obj, soc, remain_s) + 10.0 * target_surplus
                            energy_margin = max(0.0, (soc - float(ev_obj.v2g_minsoc())) * cap)
                            p_soc_dis = energy_margin * float(ev_obj.eta_dis) / max(1e-9, dt_h)
                            pdis_cap = min(pmax_dis, p_soc_dis)
                            if pdis_cap > 1e-9:
                                discharge_info[eid] = (dis_score, pdis_cap)

                    has_charge = eid in charge_info
                    has_discharge = eid in discharge_info
                    if has_charge and not has_discharge:
                        charge_pool.append(eid)
                    elif has_discharge and not has_charge:
                        discharge_pool.append(eid)
                    elif has_charge and has_discharge:
                        dual_pool.append((soc - target_soc_eff, discharge_info[eid][0] - charge_info[eid][0], eid))

                if pch_target <= 1e-9:
                    discharge_pool.extend(eid for _, _, eid in dual_pool)
                elif pdis_target <= 1e-9:
                    charge_pool.extend(eid for _, _, eid in dual_pool)
                else:
                    dual_pool.sort(reverse=True)
                    discharge_cap = sum(discharge_info[eid][1] for eid in discharge_pool)
                    for _, _, eid in dual_pool:
                        if discharge_cap + 1e-9 < pdis_target:
                            discharge_pool.append(eid)
                            discharge_cap += discharge_info[eid][1]
                        else:
                            charge_pool.append(eid)

            if pch_target > 1e-9 and charge_pool:
                weighted_ch: List[Tuple[float, int, float]] = []
                for eid in charge_pool:
                    score, cap_kw = charge_info[eid]
                    weighted_ch.append((score, eid, cap_kw))
                charge_alloc = _weighted_power_split(weighted_ch, pch_target)
                for eid, p in charge_alloc.items():
                    ev = event_by_id[eid]
                    ev_obj = ev_by_id.get(int(ev["ev_i"]))
                    if ev_obj is None:
                        continue
                    soc = float(state[eid]["soc"])
                    cap = float(state[eid]["cap"])
                    ev_obj.soc = soc
                    eta = max(1e-6, float(ev_obj.charge_efficiency(p)))
                    if p <= 1e-9:
                        continue
                    power_ch[(eid, k)] = float(p)
                    state[eid]["soc"] = min(1.0, soc + (p * eta * dt_h) / cap)

            if pdis_target > 1e-9 and discharge_pool:
                weighted_dis: List[Tuple[float, int, float]] = []
                for eid in discharge_pool:
                    score, cap_kw = discharge_info[eid]
                    weighted_dis.append((score, eid, cap_kw))
                discharge_alloc = _weighted_power_split(weighted_dis, pdis_target)
                for eid, p in discharge_alloc.items():
                    ev = event_by_id[eid]
                    ev_obj = ev_by_id.get(int(ev["ev_i"]))
                    if ev_obj is None:
                        continue
                    soc = float(state[eid]["soc"])
                    cap = float(state[eid]["cap"])
                    min_soc = float(ev_obj.v2g_minsoc())
                    if p <= 1e-9:
                        continue
                    power_dis[(eid, k)] = float(p)
                    state[eid]["soc"] = max(min_soc, soc - (p / max(1e-6, float(ev_obj.eta_dis)) * dt_h) / cap)

        for eid in eids:
            ev = event_by_id[eid]
            ev_obj = ev_by_id.get(int(ev["ev_i"]))
            if ev_obj is None or eid not in state:
                continue
            soc = float(state[eid]["soc"])
            cap = float(state[eid]["cap"])
            unmet_departure_kwh[eid] = max(0.0, (min(1.0, float(target_soc)) - soc) * cap)

    return {
        "event_pch_kw": power_ch,
        "event_pdis_kw": power_dis,
        "final_soc": {eid: float(st["soc"]) for eid, st in state.items()},
        "unmet_departure_kwh": unmet_departure_kwh,
    }

# Build model
def build_model(grid: Any, cfg: LinDistFlowConfig, T: int, delta_t: float, prices: List[float]) -> ConcreteModel:  # noqa: F405
    # 1) get data from grid
    baseMVA = float(grid.baseMVA)
    buses = list(grid.buses)
    bus_ids = [int(b.bus_i) for b in buses]
    slack = int(bus_ids[0])
    branches = [br for br in grid.branches if int(br.status) == 1]
    parent, children, edges = grid.build_radial_tree()
    gens = [g for g in grid.generators if int(g.status) == 1]
    gen_ids = [int(g.gen_i) for g in gens]
    rens = [r for r in grid.renewables if int(r.status) == 1]
    ren_ids = [int(r.renewable_i) for r in rens]
    stos = [s for s in grid.storages if int(s.status) == 1]
    sto_ids = [int(s.storage_i) for s in stos]
    stations, events = load_queue_yaml(cfg.queue_yaml_path)
    if cfg.events_override is not None:
        events = [dict(ev) for ev in cfg.events_override]
    event_ids: List[int] = [int(ev["event_i"]) for ev in events]
    time_offset_s = int(getattr(cfg, "time_offset_seconds", 0))

    # 1.1 grouping by bus
    gen_by_bus: Dict[int, List[int]] = {int(b): [] for b in bus_ids}
    gen_dict: Dict[int, Any] = {}
    for g in gens:
        gid = int(getattr(g, "gen_i"))
        bus = int(getattr(g, "bus_i"))
        gen_by_bus.setdefault(bus, []).append(gid)
        gen_dict[gid] = g

    ren_by_bus: Dict[int, List[int]] = {int(b): [] for b in bus_ids}
    ren_dict: Dict[int, Any] = {}
    for r in rens:
        rid = int(getattr(r, "renewable_i"))
        bus = int(getattr(r, "bus_i"))
        ren_by_bus.setdefault(bus, []).append(rid)
        ren_dict[rid] = r

    sto_by_bus: Dict[int, List[int]] = {int(b): [] for b in bus_ids}
    sto_dict: Dict[int, Any] = {}
    for s in stos:
        sid = int(getattr(s, "storage_i"))
        bus = int(getattr(s, "bus_i"))
        sto_by_bus.setdefault(bus, []).append(sid)
        sto_dict[sid] = s

    # 1.2 branch data
    edge_set: List[Tuple[int, int]] = []
    r_pu: Dict[Tuple[int, int], float] = {}
    x_pu: Dict[Tuple[int, int], float] = {}
    smax_pu: Dict[Tuple[int, int], float] = {}
    edge_to_branch: Dict[Tuple[int, int], int] = {}

    br_dict: Dict[int, Any] = {int(br.branch_i): br for br in branches}
    for (i, j, br_i) in edges:
        br = br_dict[int(br_i)]
        edge = (int(i), int(j))
        edge_set.append(edge)
        edge_to_branch[edge] = int(br_i)
        z = grid.branch_z_pu(br)
        r_pu[edge] = float(z.real)
        x_pu[edge] = float(z.imag)
        rateA = float(getattr(br, "rateA", 20.0))
        smax_pu[edge] = rateA / baseMVA

    # 1.3 renewables
    pav_mw: Dict[Tuple[int, int], float] = {}
    q_coeff: Dict[int, float] = {}
    curt_cost: Dict[int, float] = {}

    for rid in ren_ids:
        rr = ren_dict[rid]
        q_coeff[rid] = renewable_q_coeff(rr)
        curt_cost[rid] = float(rr.curt_cost)
        for tt in range(T):
            t_sec = time_offset_s + int(tt * delta_t * 3600.0)
            pav_mw[(rid, tt)] = float(grid.renewable_pav_mw(rid, t=int(t_sec)))

    # 1.4 storages
    sto_Emax: Dict[int, float] = {}
    sto_soc0: Dict[int, float] = {}
    sto_pch_max_mw: Dict[int, float] = {}
    sto_pdis_max_mw: Dict[int, float] = {}
    sto_eta_ch: Dict[int, float] = {}
    sto_eta_dis: Dict[int, float] = {}
    sto_qc: Dict[int, float] = {}

    for sid in sto_ids:
        ss = sto_dict[sid]
        sto_Emax[sid] = float(ss.Emax)
        if cfg.storage_soc0_override is not None and sid in cfg.storage_soc0_override:
            sto_soc0[sid] = float(cfg.storage_soc0_override[sid])
        else:
            sto_soc0[sid] = float(ss.soc)
        sto_pch_max_mw[sid] = float(ss.P_ch) / 1000.0
        sto_pdis_max_mw[sid] = abs(float(ss.P_dis)) / 1000.0
        sto_eta_ch[sid] = float(ss.eta_ch)
        sto_eta_dis[sid] = float(ss.eta_dis)
        sto_qc[sid] = float(storage_q_coeff(ss))

    # 1.5 EV station-aggregated params
    if not stations:
        stations = [dict(st.to_config()) for st in grid.stations if int(getattr(st, "status", 1)) == 1]
    station_by_id = {int(s["station_i"]): s for s in stations}
    station_ids = sorted(station_by_id.keys())
    station_bus = {sid: int(station_by_id[sid]["bus_i"]) for sid in station_by_id}

    ev_step_s = int(cfg.ev_resolution_seconds)
    horizon_s = int(round(float(T) * float(delta_t) * 3600.0))
    K_ev = int(_ceil_div(horizon_s, ev_step_s))

    ev_by_id: Dict[int, EV] = load_evs(cfg.ev_yaml_path)
    evnt_by_id: Dict[int, Dict[str, Any]] = {int(ev["event_i"]): ev for ev in events}
    station_event_ids: Dict[int, List[int]] = {sid: [] for sid in station_ids}
    st_pch_cap_kw: Dict[Tuple[int, int], float] = {}
    st_pdis_cap_kw: Dict[Tuple[int, int], float] = {}
    st_active_any: Dict[Tuple[int, int], int] = {}
    st_req_cum_kwh: Dict[Tuple[int, int], float] = {}
    st_due_flag: Dict[Tuple[int, int], int] = {}
    st_eta_ch_num: Dict[Tuple[int, int], float] = {}
    st_eta_ch_den: Dict[Tuple[int, int], float] = {}
    st_eta_dis_num: Dict[Tuple[int, int], float] = {}
    st_eta_dis_den: Dict[Tuple[int, int], float] = {}
    st_eta_ch_slot: Dict[Tuple[int, int], float] = {}
    st_eta_dis_slot: Dict[Tuple[int, int], float] = {}
    ev_slot_price: Dict[int, float] = {
        k: float(get_series_value(cfg.time_yaml_path, "price", t=int(k * ev_step_s))) for k in range(int(K_ev))
    }

    for eid in event_ids:
        e = evnt_by_id[eid]
        ev_i = int(e["ev_i"])
        sid = int(e["station_i"])
        if sid not in station_by_id:
            continue
        arr = int(e["arrival_t"])
        st = int(e.get("start_t", arr))
        dep = int(e["departure_t"])
        if dep <= 0 or st >= horizon_s:
            continue
        station_event_ids.setdefault(sid, []).append(eid)
        st = min(horizon_s, max(0, st))
        dep_active = min(horizon_s, max(st, dep))
        ev_obj = ev_by_id.get(ev_i, None)
        if ev_obj is None:
            continue
        soc0 = float(e.get("soc_init", getattr(ev_obj, "soc", 0.5)))
        cap_kwh = float(ev_obj.capacity)
        if dep <= horizon_s:
            dep_k = int(_ceil_div(dep, ev_step_s))
            req_kwh = max(0.0, (float(cfg.ev_target_soc) - soc0) * cap_kwh)
            st_req_cum_kwh[(sid, dep_k)] = float(st_req_cum_kwh.get((sid, dep_k), 0.0)) + req_kwh
            st_due_flag[(sid, dep_k)] = 1
        for k in range(int(K_ev)):
            t0 = int(k * ev_step_s)
            t1 = int(min(horizon_s, (k + 1) * ev_step_s))
            dt = _sec_overlap(t0, t1, st, dep_active)
            if dt <= 0:
                continue
            st_active_any[(sid, k)] = 1
            pch_cap = max(0.0, float(ev_obj.p_ch))
            pdis_cap = max(0.0, float(ev_obj.p_dis)) if ev_obj.is_v2g(sell_price=float(ev_slot_price.get(k, 0.0))) else 0.0
            st_pch_cap_kw[(sid, k)] = float(st_pch_cap_kw.get((sid, k), 0.0)) + pch_cap
            st_pdis_cap_kw[(sid, k)] = float(st_pdis_cap_kw.get((sid, k), 0.0)) + pdis_cap
            if pch_cap > 1e-9:
                st_eta_ch_num[(sid, k)] = float(st_eta_ch_num.get((sid, k), 0.0)) + pch_cap * max(1e-6, float(ev_obj.eta_ch))
                st_eta_ch_den[(sid, k)] = float(st_eta_ch_den.get((sid, k), 0.0)) + pch_cap
            if pdis_cap > 1e-9:
                st_eta_dis_num[(sid, k)] = float(st_eta_dis_num.get((sid, k), 0.0)) + pdis_cap * max(1e-6, float(ev_obj.eta_dis))
                st_eta_dis_den[(sid, k)] = float(st_eta_dis_den.get((sid, k), 0.0)) + pdis_cap
    for sid in station_ids:
        cum = 0.0
        for k in range(int(K_ev) + 1):
            cum += float(st_req_cum_kwh.get((sid, k), 0.0))
            st_req_cum_kwh[(sid, k)] = cum
        for k in range(int(K_ev)):
            ch_den = float(st_eta_ch_den.get((sid, k), 0.0))
            dis_den = float(st_eta_dis_den.get((sid, k), 0.0))
            st_eta_ch_slot[(sid, k)] = float(st_eta_ch_num.get((sid, k), 0.0)) / ch_den if ch_den > 1e-9 else 0.95
            st_eta_dis_slot[(sid, k)] = float(st_eta_dis_num.get((sid, k), 0.0)) / dis_den if dis_den > 1e-9 else 0.90

    # Overlap weights EV slots -> OPF steps (per station bus)
    opf_step_s = int(round(float(delta_t) * 3600.0))
    st_w: Dict[Tuple[int, int, int, int], float] = {}
    for tt in range(int(T)):
        a0 = int(tt * opf_step_s)
        a1 = int(min(horizon_s, (tt + 1) * opf_step_s))
        denom = max(1, a1 - a0)
        for sid in station_ids:
            b = int(station_bus.get(sid, 0))
            if b == 0:
                continue
            for k in range(int(K_ev)):
                if int(st_active_any.get((sid, k), 0)) == 0:
                    continue
                k0 = int(k * ev_step_s)
                k1 = int(min(horizon_s, (k + 1) * ev_step_s))
                ov = _sec_overlap(a0, a1, k0, k1)
                if ov <= 0:
                    continue
                st_w[(b, tt, sid, k)] = float(ov) / float(denom)

    # 2) build pyomo model
    m = ConcreteModel()  
    m.T = RangeSet(0, T - 1) 
    m.TSOC = RangeSet(0, T) 
    m.KEV = RangeSet(0, max(0, K_ev - 1))
    m.KEV_SOC = RangeSet(0, int(K_ev)) 
    m.BUS = Set(initialize=bus_ids) 
    m.GEN = Set(initialize=gen_ids) 
    m.REN = Set(initialize=ren_ids)
    m.STO = Set(initialize=sto_ids)  
    m.STN = Set(initialize=station_ids)
    m.EDGE = Set(dimen=2, initialize=edge_set) 

    # Params
    m.price = Param(m.T, initialize={t: float(prices[t]) for t in range(int(T))}, mutable=False)  
    m.br_id = Param(m.EDGE, initialize={e: int(edge_to_branch[e]) for e in edge_set}, mutable=False) 
    m.ev_price = Param(m.KEV, initialize={k: float(ev_slot_price.get(int(k), 0.0)) for k in range(int(K_ev))}, mutable=False) 
    terminal_price = float(prices[int(T) - 1]) if int(T) > 0 else 0.0

    # Renewables params
    m.Pav = Param(
        m.REN,
        m.T,
        initialize={(rid, tt): float(pav_mw[(int(rid), int(tt))]) / baseMVA for rid in ren_ids for tt in range(T)},
        mutable=False,
    )
    m.q_coeff = Param(m.REN, initialize={rid: float(q_coeff[int(rid)]) for rid in ren_ids}, mutable=False) 
    m.curt_cost = Param(m.REN, initialize={rid: float(curt_cost[int(rid)]) for rid in ren_ids}, mutable=False) 

    # Storage params
    m.Emax = Param(m.STO, initialize={sid: float(sto_Emax[int(sid)]) for sid in sto_ids}, mutable=False) 
    m.soc0 = Param(m.STO, initialize={sid: float(sto_soc0[int(sid)]) for sid in sto_ids}, mutable=False)  
    m.PchMax = Param(m.STO, initialize={sid: float(sto_pch_max_mw[int(sid)]) / baseMVA for sid in sto_ids}, mutable=False) 
    m.PdisMax = Param(m.STO, initialize={sid: float(sto_pdis_max_mw[int(sid)]) / baseMVA for sid in sto_ids}, mutable=False) 
    m.eta_ch = Param(m.STO, initialize={sid: float(sto_eta_ch[int(sid)]) for sid in sto_ids}, mutable=False) 
    m.eta_dis = Param(m.STO, initialize={sid: float(sto_eta_dis[int(sid)]) for sid in sto_ids}, mutable=False) 
    m.qc_sto = Param(m.STO, initialize={sid: float(sto_qc[int(sid)]) for sid in sto_ids}, mutable=False) 
    # Station-aggregated EV params
    m.st_active = Param(
        m.STN,
        m.KEV,
        initialize={(sid, k): int(st_active_any.get((int(sid), int(k)), 0)) for sid in station_ids for k in range(int(K_ev))},
        mutable=False,
    )
    m.st_pch_cap = Param(
        m.STN,
        m.KEV,
        initialize={(sid, k): float(st_pch_cap_kw.get((int(sid), int(k)), 0.0)) for sid in station_ids for k in range(int(K_ev))},
        mutable=False,
    )
    m.st_pdis_cap = Param(
        m.STN,
        m.KEV,
        initialize={(sid, k): float(st_pdis_cap_kw.get((int(sid), int(k)), 0.0)) for sid in station_ids for k in range(int(K_ev))},
        mutable=False,
    )
    m.st_eta_ch = Param(
        m.STN,
        m.KEV,
        initialize={(sid, k): float(st_eta_ch_slot.get((int(sid), int(k)), 0.95)) for sid in station_ids for k in range(int(K_ev))},
        mutable=False,
    )
    m.st_eta_dis = Param(
        m.STN,
        m.KEV,
        initialize={(sid, k): float(st_eta_dis_slot.get((int(sid), int(k)), 0.90)) for sid in station_ids for k in range(int(K_ev))},
        mutable=False,
    )
    m.st_req_cum = Param(
        m.STN,
        m.KEV_SOC,
        initialize={(sid, k): float(st_req_cum_kwh.get((int(sid), int(k)), 0.0)) for sid in station_ids for k in range(int(K_ev) + 1)},
        mutable=False,
    )
    m.st_due_flag = Param(
        m.STN,
        m.KEV_SOC,
        initialize={(sid, k): int(st_due_flag.get((int(sid), int(k)), 0)) for sid in station_ids for k in range(int(K_ev) + 1)},
        mutable=False,
    )

    # Variables
    m.v = Var(m.BUS, m.T, domain=Reals) 
    m.Vu = Var(m.BUS, m.T, domain=NonNegativeReals)  
    m.Vl = Var(m.BUS, m.T, domain=NonNegativeReals)  
    m.P = Var(m.EDGE, m.T, domain=Reals) 
    m.Q = Var(m.EDGE, m.T, domain=Reals) 
    m.Pg = Var(m.GEN, m.T, domain=Reals) 
    m.Qg = Var(m.GEN, m.T, domain=Reals) 

    # Renewable variables
    m.Pr = Var(m.REN, m.T, domain=NonNegativeReals) 
    m.curt = Var(m.REN, m.T, domain=NonNegativeReals) 

    # Storage variables
    m.P_ch = Var(m.STO, m.T, domain=NonNegativeReals) 
    m.P_dis = Var(m.STO, m.T, domain=NonNegativeReals)  
    m.u_sto_ch = Var(m.STO, m.T, domain=Binary) 
    m.u_sto_dis = Var(m.STO, m.T, domain=Binary) 
    m.soc = Var(m.STO, m.TSOC, domain=Reals) 

    # Station-aggregated EV variables
    m.st_pch = Var(m.STN, m.KEV, domain=NonNegativeReals)
    m.st_pdis = Var(m.STN, m.KEV, domain=NonNegativeReals)
    m.st_energy = Var(m.STN, m.KEV_SOC, domain=NonNegativeReals)
    m.st_short = Var(m.STN, m.KEV_SOC, domain=NonNegativeReals)
    for sid in station_ids:
        m.st_energy[sid, 0].fix(0.0)

    # Grid buy/sell
    m.P_buy = Var(m.T, domain=NonNegativeReals, bounds=(0.0, 100.0)) 
    m.P_sell = Var(m.T, domain=NonNegativeReals, bounds=(0.0, 100.0)) 
    m.u_grid_buy = Var(m.T, domain=Binary) 
    m.Q_grid = Var(m.T, domain=Reals, bounds=(-100.0, 100.0))  

    # slack constraints
    def _buy_bound(mm, tt):
        return mm.P_buy[tt] <= 100.0 * mm.u_grid_buy[tt]
    def _sell_bound(mm, tt):
        return mm.P_sell[tt] <= 100.0 * (1 - mm.u_grid_buy[tt])

    m.GridBuyBound = Constraint(m.T, rule=_buy_bound) 
    m.GridSellBound = Constraint(m.T, rule=_sell_bound) 

    for tt in range(int(T)):
        m.v[slack, tt].fix(1.0)

    # Voltage constraints
    def v_up(mm, i, tt):
        bus = grid.get_bus(int(i))
        return mm.v[i, tt] <= float(bus.Vmax) ** 2 + mm.Vu[i, tt]
    def v_low(mm, i, tt):
        bus = grid.get_bus(int(i))
        return mm.v[i, tt] >= float(bus.Vmin) ** 2 - mm.Vl[i, tt]

    m.VmaxLimit = Constraint(m.BUS, m.T, rule=v_up) 
    m.VminLimit = Constraint(m.BUS, m.T, rule=v_low)  

    # Generator limits
    def gen_p_lim(mm, gid, tt):
        g = grid.get_generator(int(gid))
        return (float(g.Pmin) / baseMVA, mm.Pg[gid, tt], float(g.Pmax) / baseMVA)
    def gen_q_lim(mm, gid, tt):
        g = grid.get_generator(int(gid))
        return (float(g.Qmin) / baseMVA, mm.Qg[gid, tt], float(g.Qmax) / baseMVA)

    m.GenPLimit = Constraint(m.GEN, m.T, rule=gen_p_lim)  
    m.GenQLimit = Constraint(m.GEN, m.T, rule=gen_q_lim)  

    # Renewable constraints
    def ren_p_le_pav(mm, rid, tt):
        return mm.Pr[rid, tt] <= mm.Pav[rid, tt]
    def curt_def(mm, rid, tt):
        return mm.curt[rid, tt] >= mm.Pav[rid, tt] - mm.Pr[rid, tt]

    m.RenPLePav = Constraint(m.REN, m.T, rule=ren_p_le_pav)  
    m.CurtDef = Constraint(m.REN, m.T, rule=curt_def) 

    # Storage constraints
    def sto_ch_bound(mm, sid, tt):
        return mm.P_ch[sid, tt] <= mm.PchMax[sid] * mm.u_sto_ch[sid, tt]
    def sto_dis_bound(mm, sid, tt):
        return mm.P_dis[sid, tt] <= mm.PdisMax[sid] * mm.u_sto_dis[sid, tt]
    def sto_no_simul(mm, sid, tt):
        return mm.u_sto_ch[sid, tt] + mm.u_sto_dis[sid, tt] <= 1

    m.StoChBound = Constraint(m.STO, m.T, rule=sto_ch_bound) 
    m.StoDisBound = Constraint(m.STO, m.T, rule=sto_dis_bound)  
    m.StoNoSimul = Constraint(m.STO, m.T, rule=sto_no_simul)

    def soc_bounds(mm, sid, tt):
        return (0.0, mm.soc[sid, tt], 1.0)

    m.SocBounds = Constraint(m.STO, m.TSOC, rule=soc_bounds) 
    for sid in sto_ids:
        m.soc[sid, 0].fix(float(sto_soc0[sid]))

    def soc_dyn(mm, sid, tt):
        Emax = float(mm.Emax[sid])
        Pch_mw = mm.P_ch[sid, tt] * baseMVA
        Pdis_mw = mm.P_dis[sid, tt] * baseMVA
        deltaE = (mm.eta_ch[sid] * Pch_mw - (1.0 / mm.eta_dis[sid]) * Pdis_mw) * float(delta_t)
        return mm.soc[sid, tt + 1] == mm.soc[sid, tt] + deltaE / Emax

    m.SocDyn = Constraint(m.STO, m.T, rule=soc_dyn) 

    # Station-aggregated EV constraints
    def st_pch_bound(mm, sid, k):
        return mm.st_pch[sid, k] <= mm.st_pch_cap[sid, k]

    def st_pdis_bound(mm, sid, k):
        return mm.st_pdis[sid, k] <= mm.st_pdis_cap[sid, k]

    def st_shared_bound(mm, sid, k):
        pch_cap = float(st_pch_cap_kw.get((int(sid), int(k)), 0.0))
        pdis_cap = float(st_pdis_cap_kw.get((int(sid), int(k)), 0.0))
        active = int(st_active_any.get((int(sid), int(k)), 0))
        if active <= 0 or (pch_cap <= 1e-9 and pdis_cap <= 1e-9):
            return mm.st_pch[sid, k] + mm.st_pdis[sid, k] == 0.0
        if pch_cap <= 1e-9:
            return mm.st_pdis[sid, k] <= pdis_cap
        if pdis_cap <= 1e-9:
            return mm.st_pch[sid, k] <= pch_cap
        return (mm.st_pch[sid, k] / pch_cap) + (mm.st_pdis[sid, k] / pdis_cap) <= 1.0

    def st_energy_dyn(mm, sid, k):
        dt_h = float(ev_step_s) / 3600.0
        dE = (mm.st_pch[sid, k] * mm.st_eta_ch[sid, k] - (mm.st_pdis[sid, k] / mm.st_eta_dis[sid, k])) * dt_h
        return mm.st_energy[sid, k + 1] == mm.st_energy[sid, k] + dE

    def st_shortfall(mm, sid, k):
        return mm.st_short[sid, k] >= mm.st_req_cum[sid, k] - mm.st_energy[sid, k]

    m.STPchBound = Constraint(m.STN, m.KEV, rule=st_pch_bound)
    m.STPdisBound = Constraint(m.STN, m.KEV, rule=st_pdis_bound)
    m.STSharedBound = Constraint(m.STN, m.KEV, rule=st_shared_bound)
    m.STEnergyDyn = Constraint(m.STN, m.KEV, rule=st_energy_dyn)
    m.STShortfall = Constraint(m.STN, m.KEV_SOC, rule=st_shortfall)

    # Branch constraints
    def branch_soc(mm, i, j, tt):
        lim = float(smax_pu[(int(i), int(j))])
        return mm.P[(i, j), tt] ** 2 + mm.Q[(i, j), tt] ** 2 <= lim**2

    m.BranchSOC = Constraint(m.EDGE, m.T, rule=branch_soc) 

    # linear distflow
    def v_drop(mm, i, j, tt):
        return mm.v[j, tt] == mm.v[i, tt] - 2.0 * (
            float(r_pu[(int(i), int(j))]) * mm.P[(i, j), tt] + float(x_pu[(int(i), int(j))]) * mm.Q[(i, j), tt]
        )

    m.VoltageDrop = Constraint(m.EDGE, m.T, rule=v_drop)  

    # Station -> BUS aggregation
    def st_bus_net_pu(mm, bus_i: int, tt: int):
        total_mw_expr = 0
        for sid in station_ids:
            if int(station_bus.get(int(sid), 0)) != int(bus_i):
                continue
            for k in range(int(K_ev)):
                w = st_w.get((int(bus_i), int(tt), int(sid), int(k)), 0.0)
                if w == 0.0:
                    continue
                pnet_kw = mm.st_pch[sid, k] - mm.st_pdis[sid, k]
                total_mw_expr += float(w) * (pnet_kw / 1000.0)
        return total_mw_expr / baseMVA

    # Power injections
    def P_inj(mm, i, tt):
        s = 0.0
        for gid in gen_by_bus[int(i)]:
            s += mm.Pg[gid, tt]
        for rid in ren_by_bus[int(i)]:
            s += mm.Pr[rid, tt]
        for sid in sto_by_bus.get(int(i), []):
            s += (mm.P_dis[sid, tt] - mm.P_ch[sid, tt])
        if station_ids:
            s -= st_bus_net_pu(mm, int(i), int(tt))
        return s

    def Q_inj(mm, i, tt):
        s = 0.0
        for gid in gen_by_bus[int(i)]:
            s += mm.Qg[gid, tt]
        for rid in ren_by_bus[int(i)]:
            s += mm.q_coeff[rid] * mm.Pr[rid, tt]
        for sid in sto_by_bus.get(int(i), []):
            Pnet = (mm.P_dis[sid, tt] - mm.P_ch[sid, tt])
            s += mm.qc_sto[sid] * Pnet
        return s

    # Nodal balances
    def p_balance(mm, i, tt):
        i_int = int(i)
        t_sec = time_offset_s + int(tt * delta_t * 3600.0)
        bus = grid.get_bus(i_int)
        Pd, _ = grid.bus_PQ_pu(bus, t=t_sec)
        incoming = 0.0 if i_int == slack else mm.P[(int(parent[i_int]), i_int), tt]
        outgoing = sum(mm.P[(i_int, int(c)), tt] for c in children.get(i_int, []))
        grid_term = (mm.P_buy[tt] - mm.P_sell[tt]) if i_int == slack else 0.0
        return incoming + P_inj(mm, i_int, tt) + grid_term - float(Pd) - outgoing == 0.0

    def q_balance(mm, i, tt):
        i_int = int(i)
        t_sec = time_offset_s + int(tt * delta_t * 3600.0)
        bus = grid.get_bus(i_int)
        _, Qd = grid.bus_PQ_pu(bus, t=t_sec)
        incoming = 0.0 if i_int == slack else mm.Q[(int(parent[i_int]), i_int), tt]
        outgoing = sum(mm.Q[(i_int, int(c)), tt] for c in children.get(i_int, []))
        grid_term = mm.Q_grid[tt] if i_int == slack else 0.0
        return incoming + Q_inj(mm, i_int, tt) + grid_term - float(Qd) - outgoing == 0.0

    m.PBalance = Constraint(m.BUS, m.T, rule=p_balance) 
    m.QBalance = Constraint(m.BUS, m.T, rule=q_balance) 

    # Objective
    ev_pen = float(cfg.ev_depart_penalty)
    v2g_rw = float(cfg.v2g_reward_coeff)
    sto_terminal_value_coeff = float(cfg.storage_terminal_value_coeff)

    def obj(mm):
        total = 0.0
        for tt in mm.T:
            # gen cost
            for g in gens:
                gi = int(g.gen_i)
                Pg_MW = mm.Pg[gi, tt] * baseMVA
                total += (float(g.cost_c2) * Pg_MW**2 + float(g.cost_c1) * Pg_MW + float(g.cost_c0)) * float(delta_t)
            # grid cost
            price = value(mm.price[tt])
            Pbuy_MW = mm.P_buy[tt] * baseMVA
            Psell_MW = mm.P_sell[tt] * baseMVA
            total += Pbuy_MW * price * float(delta_t) * 1000.0
            total -= Psell_MW * price * float(delta_t) * 1000.0
            # curtailment cost
            for rid in ren_ids:
                total += mm.curt_cost[rid] * (mm.curt[rid, tt] * baseMVA) * float(delta_t)
            # voltage soft penalty
            if cfg.v_penalty:
                for i in bus_ids:
                    total += float(cfg.v_penalty) * (mm.Vu[i, tt] + mm.Vl[i, tt])
        if sto_terminal_value_coeff != 0.0 and int(T) > 0:
            for sid in sto_ids:
                terminal_energy_mwh = mm.Emax[sid] * mm.soc[sid, T]
                total -= sto_terminal_value_coeff * terminal_price * terminal_energy_mwh * 1000.0

        # Station-level EV shortfall and V2G reward
        for sid in station_ids:
            for k in range(int(K_ev) + 1):
                if int(st_due_flag.get((int(sid), int(k)), 0)) == 0:
                    continue
                total += ev_pen * mm.st_short[sid, k]
        if v2g_rw != 0.0:
            dt_h = float(ev_step_s) / 3600.0
            for sid in station_ids:
                for k in range(int(K_ev)):
                    price_k = value(mm.ev_price[k])
                    e_kwh = mm.st_pdis[sid, k] * dt_h
                    total -= v2g_rw * price_k * e_kwh

        return total

    m.Obj = Objective(rule=obj, sense=minimize) 

    # Meta for printing
    m._meta = {
        "baseMVA": baseMVA,
        "delta_t": float(delta_t),
        "T": int(T),
        "slack": int(slack),
        "bus_ids": list(bus_ids),
        "gen_ids": list(gen_ids),
        "ren_ids": list(ren_ids),
        "sto_ids": list(sto_ids),
        "station_ids": list(station_ids),
        "K_ev": int(K_ev),
        "ev_step_s": int(ev_step_s),
        "station_bus": {int(sid): int(station_bus[sid]) for sid in station_ids},
        "events_cfg": [dict(ev) for ev in events],
        "station_event_ids": {int(sid): list(station_event_ids.get(int(sid), [])) for sid in station_ids},
        "station_req_cum_kwh": {(int(sid), int(k)): float(st_req_cum_kwh.get((int(sid), int(k)), 0.0)) for sid in station_ids for k in range(int(K_ev) + 1)},
        "v_penalty": float(cfg.v_penalty),
        "ev_depart_penalty": float(cfg.ev_depart_penalty),
        "v2g_reward_coeff": float(cfg.v2g_reward_coeff),
        "ev_target_soc": float(cfg.ev_target_soc),
        "storage_terminal_value_coeff": float(cfg.storage_terminal_value_coeff),
        "storage_terminal_price": float(terminal_price),
        "time_yaml_path": str(cfg.time_yaml_path),
        "queue_yaml_path": str(cfg.queue_yaml_path),
    }
    return m

# Solve
def solve_opf(grid: Any, cfg: LinDistFlowConfig):
    delta_t = float(cfg.resolution_minutes) / 60.0
    if cfg.horizon_steps_override is not None:
        T = int(cfg.horizon_steps_override)
    else:
        T = int(cfg.horizon_hours / delta_t)
    prices = build_price_series(cfg, T, delta_t)

    print("\n========== LinDistFlow OPF Time Configuration ==========")
    print(f"Horizon: {T * delta_t:.2f} hours")
    print(f"Resolution: {cfg.resolution_minutes} minutes")
    print(f"Delta t: {delta_t:.2f} hours")
    print(f"Time steps (T): {T}")

    model = build_model(grid, cfg, T, delta_t, prices)

    solver = SolverFactory(cfg.solver_name)  
    if cfg.solver_options:
        for k, v in cfg.solver_options.items():
            solver.options[k] = v
    results = solver.solve(model, tee=cfg.tee, load_solutions=False)
    if len(getattr(results, "solution", [])) > 0:
        try:
            model.solutions.load_from(results)
        except Exception:
            pass

    status = str(results.solver.status)
    term = str(results.solver.termination_condition)

    print("\n========== Solver Status ==========")
    print("Status:", status)
    print("Termination:", term)

    report = {
        "T": T,
        "delta_t": delta_t,
        "solver": str(cfg.solver_name),
        "status": status,
        "termination": term,
        "objective": None,
    }

    has_primal_solution = False
    try:
        _obj_probe = value(model.Obj)
        has_primal_solution = _obj_probe is not None
    except Exception:
        has_primal_solution = False

    if results.solver.termination_condition in (
        TerminationCondition.optimal, 
        TerminationCondition.locallyOptimal,  
        TerminationCondition.feasible, 
    ) or has_primal_solution:
        objv = round(value(model.Obj), 4) 
        report["objective"] = objv
        print("Objective:", objv)

        station_ids = list(model._meta.get("station_ids", []))
        station_power_kw: Dict[Tuple[int, int], Dict[str, float]] = {}
        for sid in station_ids:
            for k in range(int(model._meta.get("K_ev", 0))):
                station_power_kw[(int(sid), int(k))] = {
                    "pch": _value(model.st_pch[sid, k], 0.0),
                    "pdis": _value(model.st_pdis[sid, k], 0.0),
                }

        dispatch = allocate_station_power(
            station_power_kw=station_power_kw,
            events=list(model._meta.get("events_cfg", [])),
            ev_by_id=load_evs(cfg.ev_yaml_path),
            station_ids=station_ids,
            K_ev=int(model._meta.get("K_ev", 0)),
            ev_step_s=int(model._meta.get("ev_step_s", cfg.ev_resolution_seconds)),
            target_soc=float(model._meta.get("ev_target_soc", cfg.ev_target_soc)),
            ev_slot_price={k: float(get_series_value(cfg.time_yaml_path, "price", t=int(k * cfg.ev_resolution_seconds))) for k in range(int(model._meta.get("K_ev", 0)))},
        )
        model._dispatch = dispatch
        report["ev_unmet_kwh"] = round(sum(dispatch["unmet_departure_kwh"].values()), 4)
        report["event_served"] = len(dispatch["final_soc"])
    else:
        print("No primal solution available; objective not evaluated.")
    print("===================================\n")
    return model, results, report
