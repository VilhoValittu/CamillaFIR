from __future__ import annotations

from typing import Any

import numpy as np

_MISSING = object()
_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off", ""})


def _cfg_raw(cfg: Any, key: str, default: Any = _MISSING) -> Any:
    if isinstance(cfg, dict):
        if key in cfg:
            return cfg[key]
        return None if default is _MISSING else default
    try:
        if default is _MISSING:
            return getattr(cfg, key)
        return getattr(cfg, key, default)
    except (AttributeError, TypeError, ValueError):
        return None if default is _MISSING else default


def _coerce_float(value: Any, default: float, *, allow_zero: bool) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str) and value.strip() == "":
        return float(default)
    if not allow_zero and isinstance(value, (bool, int, float, np.integer, np.floating)) and float(value) == 0.0:
        return float(default)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not np.isfinite(parsed):
        return float(default)
    return float(parsed)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        return bool(default)
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(value)
        return bool(default)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return bool(default)
        if float(value) in (0.0, 1.0):
            return bool(int(value))
        return bool(default)
    return bool(default)


def coerce_range2(value: Any, default_min: float, default_max: float) -> list[float]:
    try:
        a = float(value[0])
        b = float(value[1])
        if np.isfinite(a) and np.isfinite(b) and b > a:
            return [a, b]
    except (TypeError, ValueError, IndexError):
        pass
    return [float(default_min), float(default_max)]


class CfgReader:
    def __init__(self, cfg: Any):
        self.cfg = cfg

    def raw(self, key: str, default: Any = None) -> Any:
        return _cfg_raw(self.cfg, key, default)

    def float(self, key: str, default: float) -> float:
        return _coerce_float(self.raw(key, default), default, allow_zero=False)

    def float_allow_zero(self, key: str, default: float) -> float:
        return _coerce_float(self.raw(key, default), default, allow_zero=True)

    def bool(self, key: str, default: bool) -> bool:
        return _coerce_bool(self.raw(key, default), default)

    def string(self, key: str, default: str) -> str:
        value = self.raw(key, default)
        if value is None:
            return str(default)
        if isinstance(value, str):
            text = value.strip()
            return text if text else str(default)
        try:
            return str(value)
        except (TypeError, ValueError):
            return str(default)

    def range2(self, key: str, default_min: float, default_max: float) -> list[float]:
        return coerce_range2(self.raw(key, [default_min, default_max]), default_min, default_max)

    def enum_string(self, key: str, default: str) -> str:
        return self.string(key, default)
