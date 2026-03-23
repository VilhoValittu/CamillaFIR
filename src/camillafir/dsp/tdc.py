import logging

import numpy as np

from .limits import limit_slope_per_octave

logger = logging.getLogger(__name__)


def apply_smart_tdc(
    freq_axis,
    target_mags,
    reflections,
    rt60_info,
    base_strength=0.5,
    max_total_reduction_db: float = 9.0,
    max_slope_db_per_oct: float = 0.0,
):
    """Soveltaa tai paivittaa: apply smart tdc."""
    adjusted_target = np.copy(target_mags)
    tdc_reduction_db = np.zeros_like(adjusted_target)

    try:
        strength = float(base_strength)
    except (TypeError, ValueError, OverflowError):
        strength = 0.0
    if not np.isfinite(strength):
        strength = 0.0
    strength = float(np.clip(strength, 0.0, 3.0))

    try:
        max_red = float(max_total_reduction_db)
    except (TypeError, ValueError, OverflowError):
        max_red = 9.0
    if not np.isfinite(max_red):
        max_red = 9.0
    if strength <= 0.0 or max_red <= 0.0:
        return adjusted_target

    def rt60_at(freq_hz: float) -> float:
        default = 0.4
        try:
            if isinstance(rt60_info, (int, float)):
                v = float(rt60_info)
                return v if np.isfinite(v) and v > 0.1 else default
            if isinstance(rt60_info, dict) and rt60_info:
                c = np.array(sorted(rt60_info.keys()), dtype=float)
                r = np.array([rt60_info[k] for k in c], dtype=float)
                mask = np.isfinite(c) & np.isfinite(r) & (c > 0) & (r > 0.05) & (r < 5.0)
                if np.count_nonzero(mask) < 2:
                    vv = float(np.median(r[mask])) if np.count_nonzero(mask) else 0.0
                    return vv if vv > 0.1 else default
                c = c[mask]
                r = r[mask]
                x = np.log10(np.clip(freq_hz, c.min(), c.max()))
                return float(np.interp(x, np.log10(c), r))
        except (TypeError, ValueError, FloatingPointError, OverflowError):
            return default
        return default

    for rev in reflections or []:
        if not isinstance(rev, dict):
            continue
        try:
            f_res = float(rev.get("freq", np.nan))
            error_ms = float(rev.get("gd_error", rev.get("error_ms", np.nan)))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(f_res) and np.isfinite(error_ms)):
            continue
        if f_res <= 0.0 or error_ms <= 0.0:
            continue

        ref_rt60 = rt60_at(f_res)
        if not (np.isfinite(ref_rt60) and ref_rt60 > 0.0):
            ref_rt60 = 0.4
        excess_ratio = error_ms / (ref_rt60 * 1000.0 + 1e-12)
        if not np.isfinite(excess_ratio):
            continue

        if excess_ratio > 0.8:
            dynamic_mult = np.clip(excess_ratio * strength, 0.0, 3.0)

            bw = f_res / max(error_ms / 15.0, 1e-9)
            bw = float(np.clip(bw, 1.0, max(5.0, 2.0 * f_res)))
            if not np.isfinite(bw) or bw <= 0.0:
                continue
            dist = np.abs(freq_axis - f_res)
            kernel = np.exp(-0.5 * (dist / bw) ** 2)

            reduction_db = dynamic_mult * 4.0
            if not np.isfinite(reduction_db) or reduction_db <= 0.0:
                continue
            tdc_reduction_db += kernel * reduction_db

    tdc_reduction_db = np.minimum(tdc_reduction_db, max_red)

    try:
        if max_slope_db_per_oct and float(max_slope_db_per_oct) > 0:
            tdc_reduction_db = limit_slope_per_octave(
                freq_axis,
                tdc_reduction_db,
                max_db_per_oct=float(max_slope_db_per_oct),
            )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        logger.debug("TDC slope limiting failed; continuing without it.", exc_info=True)

    tdc_reduction_db = np.minimum(tdc_reduction_db, max_red)

    adjusted_target -= tdc_reduction_db
    return adjusted_target
