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
    """Temporal Decay Control (TDC)

    Idea: Instead of directly subtracting multiple overlapping kernels from the target
    (which can unintentionally stack into a deep, narrow notch), we accumulate a
    *reduction curve* and apply a safety brake:
      - hard cap max total reduction (dB)
      - optional slope limit (dB/oct) for smoothness
    """
    adjusted_target = np.copy(target_mags)
    tdc_reduction_db = np.zeros_like(adjusted_target)

    # rt60_info can be:
    #  - float (old usage)
    #  - dict: {center_hz: rt60_s, .} (new: per-band)
    def rt60_at(freq_hz: float) -> float:
        # fallback
        default = 0.4
        try:
            if isinstance(rt60_info, (int, float)):
                v = float(rt60_info)
                return v if np.isfinite(v) and v > 0.1 else default
            if isinstance(rt60_info, dict) and rt60_info:
                # interpoloidaan log-taajuudessa kaistakeskuksien yli
                c = np.array(sorted(rt60_info.keys()), dtype=float)
                r = np.array([rt60_info[k] for k in c], dtype=float)
                mask = np.isfinite(c) & np.isfinite(r) & (c > 0) & (r > 0.05) & (r < 5.0)
                if np.count_nonzero(mask) < 2:
                    # if not enough bands, try e.g. median
                    vv = float(np.median(r[mask])) if np.count_nonzero(mask) else 0.0
                    return vv if vv > 0.1 else default
                c = c[mask]
                r = r[mask]
                x = np.log10(np.clip(freq_hz, c.min(), c.max()))
                return float(np.interp(x, np.log10(c), r))
        except Exception:
            return default
        return default

    for rev in reflections or []:
        if not isinstance(rev, dict):
            continue
        try:
            f_res = float(rev.get("freq", np.nan))
            # Prefer gd_error, fallback to legacy error_ms if present.
            error_ms = float(rev.get("gd_error", rev.get("error_ms", np.nan)))
        except Exception:
            continue
        if not (np.isfinite(f_res) and np.isfinite(error_ms)):
            continue
        if f_res <= 0.0 or error_ms <= 0.0:
            continue

        ref_rt60 = rt60_at(f_res)
        if not (np.isfinite(ref_rt60) and ref_rt60 > 0.0):
            ref_rt60 = 0.4
        # SENSITIVE THRESHOLD: React at 80% of average RT60
        excess_ratio = error_ms / (ref_rt60 * 1000.0 + 1e-12)
        if not np.isfinite(excess_ratio):
            continue

        if excess_ratio > 0.8:
            # Dynaaminen kerroin
            dynamic_mult = np.clip(excess_ratio * base_strength, 0.2, 3.0)

            # Kapeampi ja kohdistetumpi kaistanleveys (BW)
            bw = f_res / max(error_ms / 15.0, 1e-9)
            # Keep kernel width in sane bounds to avoid numerical/pathological extremes.
            bw = float(np.clip(bw, 1.0, max(5.0, 2.0 * f_res)))
            if not np.isfinite(bw) or bw <= 0.0:
                continue
            dist = np.abs(freq_axis - f_res)
            kernel = np.exp(-0.5 * (dist / bw) ** 2)

            reduction_db = dynamic_mult * 4.0
            if not np.isfinite(reduction_db) or reduction_db <= 0.0:
                continue
            # Accumulate effect in separate curve (prevents "stacking surprise" notches)
            tdc_reduction_db += kernel * reduction_db

    # --- Safety brakes ---
    # 1) Hard cap total reduction (per bin)
    if max_total_reduction_db and max_total_reduction_db > 0:
        tdc_reduction_db = np.minimum(tdc_reduction_db, float(max_total_reduction_db))

    # 2) Optional slope limiting in dB/oct to keep the curve smooth/predictable
    try:
        if max_slope_db_per_oct and float(max_slope_db_per_oct) > 0:
            tdc_reduction_db = limit_slope_per_octave(
                freq_axis,
                tdc_reduction_db,
                max_db_per_oct=float(max_slope_db_per_oct),
            )
    except Exception:
        # Never let TDC fail the whole pipeline
        logger.debug("TDC slope limiting failed; continuing without it.", exc_info=True)

    # Re-apply hard cap after optional smoothing, so max reduction stays guaranteed.
    if max_total_reduction_db and max_total_reduction_db > 0:
        tdc_reduction_db = np.minimum(tdc_reduction_db, float(max_total_reduction_db))

    adjusted_target -= tdc_reduction_db
    return adjusted_target
