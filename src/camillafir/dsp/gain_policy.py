from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class GainPolicy:
    max_cut_db: float
    max_boost_db: float
    low_cut_enable: bool
    low_cut_hz: float
    low_cut_strength: float
    exc_prot: bool
    exc_freq: float
    exc_soft_hz: float


def resolve_gain_policy(
    cfg: Any,
    *,
    cfg_float_allow_zero_fn: Callable[[Any, str, float], float] | None = None,
) -> GainPolicy:
    try:
        max_cut_db = abs(float(getattr(cfg, "max_cut_db", 15.0) or 15.0))
    except (AttributeError, TypeError, ValueError):
        max_cut_db = 15.0
    try:
        max_boost_db = float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
    except (AttributeError, TypeError, ValueError):
        max_boost_db = 0.0

    try:
        low_cut_enable = bool(getattr(cfg, "low_bass_cut_enable", True))
    except (AttributeError, TypeError, ValueError):
        low_cut_enable = True
    try:
        if cfg_float_allow_zero_fn is not None:
            low_cut_hz = float(cfg_float_allow_zero_fn(cfg, "low_bass_cut_hz", 0.0) or 0.0)
        else:
            low_cut_hz = float(getattr(cfg, "low_bass_cut_hz", 0.0) or 0.0)
    except (AttributeError, TypeError, ValueError):
        low_cut_hz = 0.0
    try:
        low_cut_strength = float(getattr(cfg, "low_bass_cut_strength", 0.0) or 0.0)
    except (AttributeError, TypeError, ValueError):
        low_cut_strength = 0.0
    if not np.isfinite(low_cut_strength):
        low_cut_strength = 0.0
    low_cut_strength = float(np.clip(low_cut_strength, 0.0, 1.0))

    try:
        exc_prot = bool(getattr(cfg, "exc_prot", False))
    except (AttributeError, TypeError, ValueError):
        exc_prot = False
    try:
        exc_freq = float(getattr(cfg, "exc_freq", 0.0) or 0.0)
    except (AttributeError, TypeError, ValueError):
        exc_freq = 0.0
    if not np.isfinite(exc_freq) or exc_freq <= 0.0:
        exc_freq = 0.0
    exc_soft_hz = float(exc_freq * 1.41) if exc_freq > 0.0 else 0.0

    return GainPolicy(
        max_cut_db=float(max_cut_db),
        max_boost_db=float(max_boost_db),
        low_cut_enable=bool(low_cut_enable),
        low_cut_hz=float(low_cut_hz) if np.isfinite(low_cut_hz) else 0.0,
        low_cut_strength=float(low_cut_strength),
        exc_prot=bool(exc_prot),
        exc_freq=float(exc_freq),
        exc_soft_hz=float(exc_soft_hz),
    )


def build_low_frequency_guard_mask(
    freq_axis: np.ndarray,
    policy: GainPolicy,
    *,
    include_low_cut: bool = True,
    include_exc_soft: bool = True,
) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    guard_mask = np.zeros_like(f, dtype=bool)

    if include_low_cut and policy.low_cut_enable and np.isfinite(policy.low_cut_hz) and policy.low_cut_hz > 0.0:
        guard_mask |= (f > 0.0) & (f <= float(policy.low_cut_hz))
    if include_exc_soft and policy.exc_prot and np.isfinite(policy.exc_soft_hz) and policy.exc_soft_hz > 0.0:
        guard_mask |= (f > 0.0) & (f <= float(policy.exc_soft_hz))

    return guard_mask


def clamp_gain_curve(
    curve_db: np.ndarray,
    *,
    policy: GainPolicy,
    boost_cap_db: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    out = np.asarray(curve_db, dtype=float).copy()
    cap = None
    try:
        if boost_cap_db is not None:
            cap = np.asarray(boost_cap_db, dtype=float)
            if cap.shape != out.shape:
                cap = None
    except (TypeError, ValueError):
        cap = None

    if cap is None:
        out = np.minimum(out, float(policy.max_boost_db))
    else:
        m = np.ones_like(out, dtype=bool)
        try:
            if mask is not None:
                mm = np.asarray(mask, dtype=bool)
                if mm.shape == out.shape:
                    m = mm
        except (TypeError, ValueError):
            m = np.ones_like(out, dtype=bool)
        if np.any(m):
            out[m] = np.minimum(out[m], cap[m])
        if np.any(~m):
            out[~m] = np.minimum(out[~m], float(policy.max_boost_db))

    out = np.maximum(out, -float(policy.max_cut_db))
    return out


def apply_cuts_only_guard(
    curve_db: np.ndarray,
    *,
    mask: np.ndarray,
    guard_mask: np.ndarray,
    floor_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.asarray(curve_db, dtype=float).copy()
    m = np.asarray(mask, dtype=bool)
    g = np.asarray(guard_mask, dtype=bool)
    if out.shape != m.shape or out.shape != g.shape:
        return out, {
            "guard_bins": 0,
            "boost_clamped_bins": 0,
            "floor_reapplied_bins": 0,
        }

    apply_mask = m & g
    if not np.any(apply_mask):
        return out, {
            "guard_bins": 0,
            "boost_clamped_bins": 0,
            "floor_reapplied_bins": 0,
        }

    before = out.copy()
    out[apply_mask] = np.minimum(out[apply_mask], 0.0)

    floor_reapplied_bins = 0
    try:
        if isinstance(floor_ref, np.ndarray) and floor_ref.shape == out.shape:
            floor_vals = np.asarray(floor_ref[apply_mask], dtype=float)
            valid_floor = np.isfinite(floor_vals)
            if np.any(valid_floor):
                cur = np.asarray(out[apply_mask], dtype=float)
                cur_before = cur.copy()
                cur[valid_floor] = np.minimum(cur[valid_floor], floor_vals[valid_floor])
                out[apply_mask] = cur
                floor_reapplied_bins = int(
                    np.count_nonzero(cur_before[valid_floor] > (cur[valid_floor] + 1e-9))
                )
    except (TypeError, ValueError):
        floor_reapplied_bins = 0

    boost_clamped_bins = int(
        np.count_nonzero((before[apply_mask] > 1e-9) & (out[apply_mask] <= 1e-9))
    )
    return out, {
        "guard_bins": int(np.count_nonzero(apply_mask)),
        "boost_clamped_bins": int(boost_clamped_bins),
        "floor_reapplied_bins": int(floor_reapplied_bins),
    }
