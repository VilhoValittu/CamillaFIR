"""Lightweight auto-mode helpers shared across entrypoint and orchestrators."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .shared import (
    AUTO_MODE_EVENT_PEN_BASE_PER_EVENT,
    AUTO_MODE_EVENT_PEN_DT_POWER,
    AUTO_MODE_EVENT_PEN_DT_REF_MS,
    AUTO_MODE_EVENT_PEN_DT_WEIGHT,
    _auto_safe_float,
)

_AUTO_MODE_MODE_MIN_HZ = 20.0
_AUTO_MODE_MODE_MAX_HZ = 250.0
_AUTO_MODE_MODE_MIN_GD_MS = 120.0


def coerce_orchestrator_runtime(runtime):
    if runtime is None:
        raise ValueError("Auto-mode orchestrator runtime must be provided explicitly.")
    if isinstance(runtime, dict):
        return SimpleNamespace(**runtime)
    return runtime


def _auto_collect_reflections(st: dict | None) -> list:
    st = st or {}
    refs = st.get("cmp_reflections", st.get("reflections", []))
    if isinstance(refs, list):
        return refs
    return []


def _auto_get_worst_mode_hz(result) -> float | None:
    def _pick_ch_worst(st: dict | None) -> dict | None:
        refs = _auto_collect_reflections(st)
        picks = []
        for r in refs:
            if not isinstance(r, dict):
                continue
            typ = str(r.get("type", "") or "").strip().lower()
            if typ != "resonance":
                continue
            freq = _auto_safe_float(r.get("freq", float("nan")), float("nan"))
            dt_ms = _auto_safe_float(
                r.get("gd_error", r.get("error_ms", float("nan"))),
                float("nan"),
            )
            if not (np.isfinite(freq) and np.isfinite(dt_ms)):
                continue
            if float(dt_ms) < float(_AUTO_MODE_MODE_MIN_GD_MS):
                continue
            if not (float(_AUTO_MODE_MODE_MIN_HZ) <= float(freq) <= float(_AUTO_MODE_MODE_MAX_HZ)):
                continue
            picks.append({"freq": float(freq), "dt_ms": float(dt_ms)})
        if not picks:
            return None
        return dict(max(picks, key=lambda x: float(x.get("dt_ms", 0.0))))

    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    lw = _pick_ch_worst(l_st)
    rw = _pick_ch_worst(r_st)
    if isinstance(lw, dict) and isinstance(rw, dict):
        f_l = float(_auto_safe_float(lw.get("freq", float("nan")), float("nan")))
        f_r = float(_auto_safe_float(rw.get("freq", float("nan")), float("nan")))
        dt_l = float(_auto_safe_float(lw.get("dt_ms", 0.0), 0.0))
        dt_r = float(_auto_safe_float(rw.get("dt_ms", 0.0), 0.0))
        if np.isfinite(f_l) and np.isfinite(f_r) and f_l > 0.0 and f_r > 0.0:
            if abs(float(np.log2(f_l / f_r))) <= 0.25:
                return float(0.5 * (f_l + f_r))
        return float(f_l if dt_l >= dt_r else f_r)
    if isinstance(lw, dict):
        return float(_auto_safe_float(lw.get("freq", float("nan")), float("nan")))
    if isinstance(rw, dict):
        return float(_auto_safe_float(rw.get("freq", float("nan")), float("nan")))
    return None


def _auto_get_top_modes_hz(result, *, top_n: int = 2) -> list[float]:
    n = int(max(1, top_n))

    def _collect(st: dict | None) -> list[dict]:
        refs = _auto_collect_reflections(st)
        out = []
        for r in refs:
            if not isinstance(r, dict):
                continue
            typ = str(r.get("type", "") or "").strip().lower()
            if typ != "resonance":
                continue
            freq = _auto_safe_float(r.get("freq", float("nan")), float("nan"))
            dt_ms = _auto_safe_float(
                r.get("gd_error", r.get("error_ms", float("nan"))),
                float("nan"),
            )
            if not (np.isfinite(freq) and np.isfinite(dt_ms)):
                continue
            if float(dt_ms) < float(_AUTO_MODE_MODE_MIN_GD_MS):
                continue
            if not (float(_AUTO_MODE_MODE_MIN_HZ) <= float(freq) <= float(_AUTO_MODE_MODE_MAX_HZ)):
                continue
            out.append({"freq": float(freq), "dt_ms": float(dt_ms)})
        return out

    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    picks = _collect(l_st) + _collect(r_st)
    if not picks:
        return []

    picks = sorted(picks, key=lambda x: float(x.get("dt_ms", 0.0)), reverse=True)
    out: list[dict] = []
    for p in picks:
        f = float(_auto_safe_float(p.get("freq", float("nan")), float("nan")))
        dt = float(_auto_safe_float(p.get("dt_ms", 0.0), 0.0))
        if not (np.isfinite(f) and f > 0.0):
            continue
        merged = False
        for q in out:
            fq = float(_auto_safe_float(q.get("freq", float("nan")), float("nan")))
            if not (np.isfinite(fq) and fq > 0.0):
                continue
            if abs(float(np.log2(f / fq))) <= 0.25:
                q["freq"] = float(0.5 * (fq + f))
                q["dt_ms"] = float(
                    max(float(_auto_safe_float(q.get("dt_ms", 0.0), 0.0)), dt)
                )
                merged = True
                break
        if not merged:
            out.append({"freq": float(f), "dt_ms": float(dt)})
        if len(out) >= n:
            break

    freqs = []
    for q in out:
        f = float(_auto_safe_float(q.get("freq", float("nan")), float("nan")))
        if np.isfinite(f) and f > 0.0:
            freqs.append(float(f))
    return freqs


def _auto_mode_band(
    f0_hz: float,
    base_data: dict | None = None,
) -> tuple[float, float] | None:
    f0 = _auto_safe_float(f0_hz, float("nan"))
    if not np.isfinite(f0) or f0 <= 0.0:
        return None
    lo = float(f0) / 1.35
    hi = float(f0) * 1.35
    lo = float(np.clip(lo, 20.0, 200.0))
    hi = float(np.clip(hi, 60.0, 280.0))
    if hi <= lo:
        lo = float(np.clip(min(float(lo), float(hi) - 5.0), 20.0, 200.0))
    if hi <= lo:
        hi = float(np.clip(float(lo) + 5.0, 60.0, 280.0))
    if hi <= lo:
        return None
    return float(lo), float(hi)


def _auto_event_severity(refs: list | None) -> float:
    refs = refs or []
    if not isinstance(refs, list) or not refs:
        return 0.0

    vals = []
    for r in refs:
        if not isinstance(r, dict):
            continue
        v = _auto_safe_float(r.get("gd_error", 0.0), 0.0)
        if np.isfinite(v):
            vals.append(abs(float(v)))
    if not vals:
        return 0.0

    vals = sorted(vals, reverse=True)[:5]
    weights = (1.00, 0.75, 0.55, 0.40, 0.30)
    sev = 0.0
    for i, v in enumerate(vals):
        sev += float(weights[i]) * max(0.0, float(v) - 2.0)
    return float(max(0.0, sev))


def _auto_event_penalty_weighted(
    events: list | None,
    *,
    base_per_event: float = AUTO_MODE_EVENT_PEN_BASE_PER_EVENT,
    dt_weight: float = AUTO_MODE_EVENT_PEN_DT_WEIGHT,
    power: float = AUTO_MODE_EVENT_PEN_DT_POWER,
    dt_ref_ms: float = AUTO_MODE_EVENT_PEN_DT_REF_MS,
) -> float:
    ev = events or []
    if not isinstance(ev, list) or not ev:
        return 0.0

    base_per_event_f = max(0.0, float(_auto_safe_float(base_per_event, 0.5)))
    dt_weight_f = max(0.0, float(_auto_safe_float(dt_weight, 0.02)))
    power_f = max(1.0, float(_auto_safe_float(power, 2.0)))
    dt_ref_ms_f = max(1e-6, float(_auto_safe_float(dt_ref_ms, 100.0)))

    dt_list = []
    for e in ev:
        dt = None
        if isinstance(e, dict):
            dt = e.get("dt_ms", e.get("gd_error", e.get("error_ms", None)))
        elif isinstance(e, (list, tuple)) and len(e) >= 3:
            dt = e[2]
        dt_f = _auto_safe_float(dt, float("nan"))
        if np.isfinite(dt_f):
            dt_list.append(max(0.0, abs(float(dt_f))))

    base = base_per_event_f * float(len(ev))
    if not dt_list:
        return float(max(0.0, base))
    sev = sum((float(dt) / dt_ref_ms_f) ** power_f for dt in dt_list)
    return float(max(0.0, base + dt_weight_f * float(sev)))


def _auto_pick_metric(
    st: dict | None,
    keys: tuple[str, ...],
    *,
    abs_value: bool = False,
    nonneg: bool = False,
):
    st = st or {}
    for k in keys:
        v = _auto_safe_float(st.get(k, None), default=float("nan"))
        if not np.isfinite(v):
            continue
        if abs_value:
            v = abs(v)
        if nonneg and v < 0.0:
            continue
        return float(v)
    return None


__all__ = [
    "_auto_collect_reflections",
    "_auto_event_penalty_weighted",
    "_auto_event_severity",
    "_auto_get_top_modes_hz",
    "_auto_get_worst_mode_hz",
    "_auto_mode_band",
    "_auto_pick_metric",
    "coerce_orchestrator_runtime",
]
