from bisect import bisect_right
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import yaml

DAY_S = 86400


@lru_cache(maxsize=8)
def _load_time_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _series_layout(arr: List[Dict[str, Any]]) -> Tuple[List[int], int]:
    times = [int(item.get("time", 0)) for item in arr]
    if len(times) >= 2:
        step = min(max(1, times[i + 1] - times[i]) for i in range(len(times) - 1))
    else:
        step = DAY_S
    horizon = int(times[-1]) + int(step)
    return times, max(step, horizon)


@lru_cache(maxsize=32)
def _series_cache(path: str, key: str) -> Tuple[List[int], List[float], int]:
    cfg = _load_time_yaml(path)
    arr = cfg.get(key)
    if not isinstance(arr, list) or not arr:
        return [0], [1.0], DAY_S
    times, horizon = _series_layout(arr)
    values = [float(item.get("value", 1.0)) for item in arr]
    return times, values, horizon


def get_series_value(path: str, key: str, *, t: int) -> float:
    times, values, horizon = _series_cache(path, key)
    tt = int(t) % int(horizon)
    idx = bisect_right(times, tt) - 1
    if idx < 0:
        idx = 0
    elif idx >= len(values):
        idx = len(values) - 1
    return float(values[idx])
