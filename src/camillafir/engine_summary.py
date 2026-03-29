from __future__ import annotations

import numpy as np

from .config.results import FilterResult
from .engine_build import _as_float


def summarize_run(result: FilterResult) -> dict:
    fs = int(result.fs) if result.fs else 0
    taps = int(result.taps) if result.taps else 0
    latency_ms = ((float(taps) / 2.0) / float(fs) * 1000.0) if fs > 0 and taps > 0 else 0.0
    resolution_hz = (float(fs) / float(taps)) if fs > 0 and taps > 0 else 0.0
    l_peak_idx = int(np.argmax(np.abs(np.asarray(result.l_ir)))) if np.asarray(result.l_ir).size else 0
    r_peak_idx = int(np.argmax(np.abs(np.asarray(result.r_ir)))) if np.asarray(result.r_ir).size else 0

    shift_samples = result.metrics.get("alignment_samples", l_peak_idx - r_peak_idx)
    try:
        shift_samples = int(shift_samples)
    except Exception:
        shift_samples = int(l_peak_idx - r_peak_idx)
    shift_ms = (float(shift_samples) / float(fs) * 1000.0) if fs > 0 else 0.0

    l_st = result.l_st or {}
    r_st = result.r_st or {}
    l_gd_enabled = bool(l_st.get("gd_limiter_enabled", l_st.get("gd_grad_limiter_enabled", False)))
    r_gd_enabled = bool(r_st.get("gd_limiter_enabled", r_st.get("gd_grad_limiter_enabled", False)))
    l_gd_limit = l_st.get("gd_limiter_limit_ms_per_oct", l_st.get("gd_grad_limit_ms_per_oct", None))
    r_gd_limit = r_st.get("gd_limiter_limit_ms_per_oct", r_st.get("gd_grad_limit_ms_per_oct", None))
    l_gd_reason = str(l_st.get("gd_limiter_reason", l_st.get("gd_grad_limiter_reason", "unknown")) or "unknown")
    r_gd_reason = str(r_st.get("gd_limiter_reason", r_st.get("gd_grad_limiter_reason", "unknown")) or "unknown")
    l_gd_grad_max = _as_float(
        l_st.get(
            "gd_limiter_max_grad_ms_per_oct",
            l_st.get(
                "gd_grad_limiter_max_grad_ms_per_oct",
                l_st.get(
                    "gd_limiter_max_grad_after_ms_per_oct",
                    l_st.get("gd_grad_limiter_max_grad_after_ms_per_oct", 0.0),
                ),
            ),
        ),
        0.0,
    )
    r_gd_grad_max = _as_float(
        r_st.get(
            "gd_limiter_max_grad_ms_per_oct",
            r_st.get(
                "gd_grad_limiter_max_grad_ms_per_oct",
                r_st.get(
                    "gd_limiter_max_grad_after_ms_per_oct",
                    r_st.get("gd_grad_limiter_max_grad_after_ms_per_oct", 0.0),
                ),
            ),
        ),
        0.0,
    )
    return {
        "fs": fs,
        "taps": taps,
        "latency_ms": float(latency_ms),
        "resolution_hz_per_bin": float(resolution_hz),
        "l_peak_idx": int(l_peak_idx),
        "r_peak_idx": int(r_peak_idx),
        "shift_samples": int(shift_samples),
        "shift_ms": float(shift_ms),
        "alignment_method": str(result.metrics.get("alignment_method", "peak")),
        "gd_limiter_enabled_l": bool(l_gd_enabled),
        "gd_limiter_enabled_r": bool(r_gd_enabled),
        "gd_limiter_reason_l": l_gd_reason,
        "gd_limiter_reason_r": r_gd_reason,
        "gd_limiter_limit_ms_per_oct_l": None if l_gd_limit is None else _as_float(l_gd_limit, 0.0),
        "gd_limiter_limit_ms_per_oct_r": None if r_gd_limit is None else _as_float(r_gd_limit, 0.0),
        "gd_limiter_max_grad_ms_per_oct_l": float(l_gd_grad_max),
        "gd_limiter_max_grad_ms_per_oct_r": float(r_gd_grad_max),
        "max_boost_db_effective_l": _as_float(l_st.get("max_boost_db_effective", l_st.get("max_boost_db", 0.0)), 0.0),
        "max_boost_db_effective_r": _as_float(r_st.get("max_boost_db_effective", r_st.get("max_boost_db", 0.0)), 0.0),
        "max_cut_db_l": _as_float(l_st.get("max_cut_db", 0.0), 0.0),
        "max_cut_db_r": _as_float(r_st.get("max_cut_db", 0.0), 0.0),
        "boost_peak_db_l": _as_float(l_st.get("boost_peak_db", 0.0), 0.0),
        "boost_peak_db_r": _as_float(r_st.get("boost_peak_db", 0.0), 0.0),
        "cut_peak_db_l": _as_float(l_st.get("cut_peak_db", 0.0), 0.0),
        "cut_peak_db_r": _as_float(r_st.get("cut_peak_db", 0.0), 0.0),
    }
