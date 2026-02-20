"""
CamillaFIR DSP - Leveling as a separate module.

Goal
- Single place for all leveling/SmartScan logic
- Robust fallbacks (empty masks, NaN/inf, out-of-range)
- Easy to unit test

Note
- This module does not do I/O and does not depend on other internal CamillaFIR functions.
- Return values are always defined and finite (especially calc_offset_db).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

__all__ = [
    "find_stable_level_window",
    "compute_leveling",
]


def _to_float(x, default: float) -> float:
    """Parse x as float; if it fails or is non-finite, return default."""
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)

def _tilt_aware_offset_db(
    freq_axis: np.ndarray,
    diff_db: np.ndarray,
    *,
    max_db_per_oct: float = 2.0,
) -> float:
    """
    Robust scalar offset from a windowed diff curve (m_anal - target),
    while compensating for a gentle broadband tilt (dB/oct) inside the window.

    Returns a SINGLE offset value (dB) to be used like:
      measured_aligned = m_anal - offset

    Notes
    - This does NOT apply any tilt correction to the response. It only avoids
      choosing a biased offset when the diff has slope.
    - Fit is done on log2(f) axis and centered at the window's log-midpoint,
      so returned offset is "at the center" of the window.
    - Slope is clamped to +/- max_db_per_oct for safety.
    """
    try:
        f = np.asarray(freq_axis, dtype=float)
        y = np.asarray(diff_db, dtype=float)
        if f.size != y.size or f.size < 20:
            return float(np.median(y)) if y.size else 0.0

        # Valid bins only
        mask = np.isfinite(f) & np.isfinite(y) & (f > 0.0)
        f = f[mask]
        y = y[mask]
        if f.size < 20:
            return float(np.median(y)) if y.size else 0.0

        # log2 frequency axis => slope in dB/oct
        x = np.log2(f)

        # Center at log-midpoint to make intercept meaningful for "window center"
        x0 = float(np.median(x))
        xc = x - x0

        # Light trimming against extreme outliers (comb notches etc.)
        # Keep middle 90% by absolute deviation from median.
        y_med = float(np.median(y))
        dev = np.abs(y - y_med)
        if dev.size >= 30:
            thr = float(np.quantile(dev, 0.90))
            keep = dev <= thr
            if np.count_nonzero(keep) >= 20:
                xc = xc[keep]
                y = y[keep]

        # Least squares slope (robust enough after trimming)
        denom = float(np.dot(xc, xc))
        if denom <= 1e-12:
            return float(np.median(y))

        slope = float(np.dot(xc, (y - float(np.median(y)))) / denom)
        # Clamp slope to sane range
        max_db_per_oct = float(max_db_per_oct)
        if not np.isfinite(max_db_per_oct) or max_db_per_oct <= 0:
            max_db_per_oct = 2.0
        slope = float(np.clip(slope, -max_db_per_oct, +max_db_per_oct))

        # Intercept at window center: median of detrended curve
        offset = float(np.median(y - slope * xc))
        if not np.isfinite(offset):
            offset = float(np.median(y)) if y.size else 0.0
        return float(offset)
    except Exception:
        try:
            return float(np.median(diff_db))
        except Exception:
            return 0.0
        


def _tilt_fit_offset_and_slope_db_per_oct(
    freq_axis: np.ndarray,
    diff_db: np.ndarray,
    *,
    max_db_per_oct: float = 2.0,
):
    """
    Like _tilt_aware_offset_db(), but also returns the fitted tilt slope (dB/oct)
    for reporting purposes only.

    Returns (offset_db, slope_db_per_oct).
    """
    try:
        f = np.asarray(freq_axis, dtype=float)
        y = np.asarray(diff_db, dtype=float)
        if f.size != y.size or f.size < 20:
            off = float(np.median(y)) if y.size else 0.0
            return off, 0.0

        mask = np.isfinite(f) & np.isfinite(y) & (f > 0.0)
        f = f[mask]
        y = y[mask]
        if f.size < 20:
            off = float(np.median(y)) if y.size else 0.0
            return off, 0.0

        x = np.log2(f)
        x0 = float(np.median(x))
        xc = x - x0

        # light trim against outliers
        y_med = float(np.median(y))
        dev = np.abs(y - y_med)
        if dev.size >= 30:
            thr = float(np.quantile(dev, 0.90))
            keep = dev <= thr
            if np.count_nonzero(keep) >= 20:
                xc = xc[keep]
                y = y[keep]

        denom = float(np.dot(xc, xc))
        if denom <= 1e-12:
            off = float(np.median(y))
            return off, 0.0

        slope = float(np.dot(xc, (y - y_med)) / denom)
        max_db_per_oct = float(max_db_per_oct)
        if not np.isfinite(max_db_per_oct) or max_db_per_oct <= 0:
            max_db_per_oct = 2.0
        slope = float(np.clip(slope, -max_db_per_oct, +max_db_per_oct))

        off = float(np.median(y - slope * xc))
        if not np.isfinite(off):
            off = float(np.median(y))

        return off, slope
    except Exception:
        try:
            return float(np.median(diff_db)), 0.0
        except Exception:
            return 0.0, 0.0


def find_stable_level_window(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray,
    f_min: float,
    f_max: float,
    window_size_octaves: float = 1.0,
    hpf_freq: float = 0.0,
) -> Tuple[float, float]:
    """
    Finds the most stable region from measurement data only.

    IMPORTANT:
    - `target_mags` is accepted for backward compatibility, but intentionally
      not used in the Smart Scan window search. This keeps the selected window
      independent from target shaping (e.g. TDC).

    Returns (s_min, s_max). Falls back to (f_min, f_max) if no valid window is found.
    """
    try:
        # Backward-compatible argument; intentionally unused.
        _ = target_mags

        f_min = _to_float(f_min, 0.0)
        f_max = _to_float(f_max, 0.0)
        hpf_freq = _to_float(hpf_freq, 0.0)
        window_size_octaves = _to_float(window_size_octaves, 1.0)

        if f_min <= 0 or f_max <= 0 or f_min >= f_max:
            return float(f_min), float(f_max)

        # Avoid HPF area if possible
        safe_f_min = max(float(f_min), float(hpf_freq) * 1.5)
        if safe_f_min >= float(f_max) * 0.8:
            safe_f_min = float(f_min)

        mask = (freq_axis >= safe_f_min) & (freq_axis <= float(f_max))
        f_search = freq_axis[mask]
        m_search = np.asarray(magnitudes, dtype=float)[mask]

        # Too few points => no reliable window
        if f_search.size < 50:
            return float(f_min), float(f_max)

        best_score = float("inf")
        res_min, res_max = float(safe_f_min), float(f_max)

        # Slide window on log scale
        current_f = float(safe_f_min)
        step = 2 ** (1 / 24.0)  # ~1/24 octave

        while current_f * (2 ** float(window_size_octaves)) <= float(f_max):
            w_start = current_f
            w_end = current_f * (2 ** float(window_size_octaves))
            w_mask = (f_search >= w_start) & (f_search <= w_end)
            n_w = int(np.count_nonzero(w_mask))
            if n_w >= 20:
                f_w = f_search[w_mask]
                m_w = m_search[w_mask]

                # Detrend by log-frequency tilt so score reflects local ripple
                # / stability instead of broad spectral slope.
                x = np.log2(np.clip(f_w, 1e-9, None))
                x0 = float(np.median(x))
                xc = x - x0
                y = np.asarray(m_w, dtype=float)
                y_med = float(np.median(y))
                denom = float(np.dot(xc, xc))
                if denom > 1e-12:
                    slope = float(np.dot(xc, (y - y_med)) / denom)
                    residual = y - (slope * xc)
                else:
                    residual = y

                std = float(np.std(residual))

                # Light weighting towards center area (prevents selecting only the lowest)
                # (small effect, but helps with strange data)
                weight = 1.0 + 0.05 * abs(np.log10(max(w_start, 1.0) / 1000.0))
                score = std * weight

                if score < best_score:
                    best_score = score
                    res_min, res_max = float(w_start), float(w_end)

            current_f *= step

        # if nothing reasonable was found
        if not np.isfinite(best_score):
            return float(f_min), float(f_max)

        return float(res_min), float(res_max)

    except Exception:
        return float(f_min), float(f_max)


def compute_leveling(cfg, freq_axis: np.ndarray, m_anal: np.ndarray, target_mags: np.ndarray):
    """
    Calculates leveling (offset) robustly.

    Returns:
      target_level_db, calc_offset_db, meas_level_db_window, target_level_db_window,
      offset_method, s_min, s_max

    Note:
      - calc_offset_db is always defined and finite
      - no mean/median operations are performed on empty masks
    """
    # basic values (always defined)
    target_level_db = 0.0
    calc_offset_db = 0.0
    meas_level_db_window = 0.0
    target_level_db_window = 0.0
    offset_method = "Unknown"

    manual_target_db = _to_float(getattr(cfg, "lvl_manual_db", 75.0), 75.0)

    # user-range (also used in return)
    s_min = _to_float(getattr(cfg, "lvl_min", 500.0), 500.0)
    s_max = _to_float(getattr(cfg, "lvl_max", 2000.0), 2000.0)

    # basic validation
    if s_min <= 0 or s_max <= 0 or s_min >= s_max:
        # last reasonable fallback
        s_min, s_max = 500.0, 2000.0

    mode = str(getattr(cfg, "lvl_mode", "Auto"))
    is_manual = ("Manual" in mode)
    # Optional: tilt-aware leveling for Auto/SmartScan/ForcedWindow (NOT ForcedOffset, NOT Manual)
    tilt_comp = bool(getattr(cfg, "lvl_tilt_comp", True))
    tilt_max_db_per_oct = _to_float(getattr(cfg, "lvl_tilt_max_db_per_oct", 2.0), 2.0)

    # Clear previous run's tilt (cfg may be reused)
    try:
        setattr(cfg, "_lvl_tilt_slope_db_per_oct", None)
    except Exception:
        pass

    # Stereo-link (DSP-owned):
    # - First channel computes window+offset normally and stores them on cfg.
    # - Subsequent calls reuse stored values automatically, even if caller re-runs compute_leveling()
    #   (e.g. after target alignment or comparison-mode recalcs).
    #
    # IMPORTANT:
    # To preserve L/R balance, stereo-link must also keep the *absolute* target level identical
    # between channels in Auto/SmartScan mode. Otherwise target-alignment (shift target to
    # target_level_db inside leveling window) can differ per channel and indirectly change the
    # effective correction headroom (e.g. one side hitting boost caps earlier).
    stereo_link = bool(getattr(cfg, "stereo_link", False))
    sl_win = getattr(cfg, "_stereo_link_window", None) if stereo_link else None
    sl_off = getattr(cfg, "_stereo_link_offset_db", None) if stereo_link else None
    sl_tgt = getattr(cfg, "_stereo_link_target_level_db", None) if stereo_link else None

    # ---------- Forced window / offset (Stereo-link support) ----------
    # If the caller provides a fixed window and/or offset, respect it.
    # This is used to ensure identical leveling between L/R channels.
    forced_window = getattr(cfg, "lvl_force_window", None)
    forced_offset = getattr(cfg, "lvl_force_offset_db", None)

    # If stereo-link is enabled and caller didn't force anything,
    # automatically reuse stored window/offset (if available).
    if stereo_link and (forced_window is None) and (forced_offset is None):
        if sl_win is not None:
            forced_window = sl_win
        if sl_off is not None:
            forced_offset = sl_off

    if forced_window is not None or forced_offset is not None:
        try:
            if forced_window is not None:
                fw0, fw1 = forced_window
                ss_min = _to_float(fw0, s_min)
                ss_max = _to_float(fw1, s_max)
                if (ss_min <= 0) or (ss_max <= 0) or (ss_min >= ss_max):
                    ss_min, ss_max = s_min, s_max
                ss_min = max(s_min, ss_min)
                ss_max = min(s_max, ss_max)
            else:
                ss_min, ss_max = s_min, s_max

            mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)
            if np.any(mask):
                meas_level_db_window = float(np.median(m_anal[mask]))
                target_level_db_window = float(np.median(target_mags[mask]))
            else:
                meas_level_db_window = 0.0
                target_level_db_window = 0.0

            if forced_offset is not None:
                calc_offset_db = _to_float(forced_offset, 0.0)
                offset_method = "ForcedOffset"

                # Reporting-only: still compute tilt slope in the chosen window (does NOT affect offset)
                try:
                    if tilt_comp and np.any(mask):
                        _off_tmp, _slope = _tilt_fit_offset_and_slope_db_per_oct(
                            freq_axis[mask],
                            (m_anal[mask] - target_mags[mask]),
                            max_db_per_oct=float(tilt_max_db_per_oct),
                        )
                        setattr(cfg, "_lvl_tilt_slope_db_per_oct", float(_slope))
                except Exception:
                    pass
            else:
                # If only window is forced, compute a deterministic offset from that.
                if np.any(mask):
                    diff = (m_anal[mask] - target_mags[mask])
                    if tilt_comp:
                        calc_offset_db, tilt_slope = _tilt_fit_offset_and_slope_db_per_oct(
                            freq_axis[mask],
                            diff,
                            max_db_per_oct=float(tilt_max_db_per_oct),
                        )
                        try:
                            setattr(cfg, "_lvl_tilt_slope_db_per_oct", float(tilt_slope))
                        except Exception:
                            pass
                        offset_method = "ForcedWindowTiltMedian"
                    else:
                        calc_offset_db = float(np.median(diff))
                        offset_method = "ForcedWindowMedian"
                else:
                    calc_offset_db = 0.0
                    offset_method = "ForcedWindowNoMask"

            # Manual -> respect user target level.
            # Auto/SmartScan -> follow measured SPL in the chosen window.
            # Stereo-link -> reuse FIRST channel's target_level_db so target alignment is identical.
            if is_manual:
                target_level_db = float(manual_target_db)
            else:
                if stereo_link and (sl_tgt is not None):
                    target_level_db = _to_float(sl_tgt, float(meas_level_db_window))
                else:
                    target_level_db = float(meas_level_db_window)

            if not np.isfinite(calc_offset_db):
                calc_offset_db = 0.0

            # Stereo-link: store first computed result for later calls/channels.
            if stereo_link and (sl_win is None) and (sl_off is None) and (sl_tgt is None):
                try:
                    setattr(cfg, "_stereo_link_window", (float(ss_min), float(ss_max)))
                    setattr(cfg, "_stereo_link_offset_db", float(calc_offset_db))
                    setattr(cfg, "_stereo_link_target_level_db", float(target_level_db))
                except Exception:
                    pass

            return (
                float(target_level_db),
                float(calc_offset_db),
                float(meas_level_db_window),
                float(target_level_db_window),
                str(offset_method),
                float(ss_min),
                float(ss_max),
            )
        except Exception:
            # If anything goes sideways, fall back to normal logic below.
            pass


    # ---------- Manual ----------
    if "Manual" in mode:
        mask = (freq_axis >= s_min) & (freq_axis <= s_max)

        if np.any(mask):
            meas_level_db_window = float(np.median(m_anal[mask]))
            target_level_db_window = float(np.median(target_mags[mask]))
            calc_offset_db = float(np.median(m_anal[mask] - target_mags[mask]))
            offset_method = "ManualMedian"
        else:
            calc_offset_db = 0.0
            offset_method = "ManualNoMask"

        # raportoinnin/plotin perusta
        target_level_db = float(manual_target_db)

        if not np.isfinite(calc_offset_db):
            calc_offset_db = 0.0
        # Stereo-link: store first computed result for later calls/channels.
        if stereo_link and (sl_win is None) and (sl_off is None) and (sl_tgt is None):
            try:
                setattr(cfg, "_stereo_link_window", (float(s_min), float(s_max)))
                setattr(cfg, "_stereo_link_offset_db", float(calc_offset_db))
                setattr(cfg, "_stereo_link_target_level_db", float(target_level_db))
            except Exception:
                pass

        return (
            float(target_level_db),
            float(calc_offset_db),
            float(meas_level_db_window),
            float(target_level_db_window),
            str(offset_method),
            float(s_min),
            float(s_max),
        )

    # ---------- Auto / SmartScan ----------
    hpf_freq = 0.0
    hpf_settings = getattr(cfg, "hpf_settings", None)
    if hpf_settings:
        try:
            hpf_freq = _to_float(hpf_settings.get("freq", 0.0), 0.0)
        except Exception:
            hpf_freq = 0.0

    ss_min, ss_max = find_stable_level_window(
        freq_axis,
        m_anal,
        target_mags,
        s_min,
        s_max,
        window_size_octaves=1.0,
        hpf_freq=float(hpf_freq),
    )

    # Validoi ja clampaa user-rangeen
    ss_min = _to_float(ss_min, s_min)
    ss_max = _to_float(ss_max, s_max)
    if (ss_min <= 0) or (ss_max <= 0) or (ss_min >= ss_max):
        ss_min, ss_max = s_min, s_max

    ss_min = max(s_min, ss_min)
    ss_max = min(s_max, ss_max)

    mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)

    # If window too small, try basic "hifi" fallback clamped to user-range
    if np.count_nonzero(mask) < 20:
        fb_min = max(s_min, 350.0)
        fb_max = min(s_max, 5000.0)
        if fb_min < fb_max:
            ss_min, ss_max = fb_min, fb_max
            mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)

    # Last fallback: entire user-range
    if np.count_nonzero(mask) < 20:
        ss_min, ss_max = s_min, s_max
        mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)

    if np.any(mask):
        meas_level_db_window = float(np.median(m_anal[mask]))
        target_level_db_window = float(np.median(target_mags[mask]))
        diff = (m_anal[mask] - target_mags[mask])
        if tilt_comp:
            calc_offset_db, tilt_slope = _tilt_fit_offset_and_slope_db_per_oct(
                freq_axis[mask],
                diff,
                max_db_per_oct=float(tilt_max_db_per_oct),
            )
            try:
                setattr(cfg, "_lvl_tilt_slope_db_per_oct", float(tilt_slope))
            except Exception:
                pass
            offset_method = "SmartScanTiltMedian"
        else:
            calc_offset_db = float(np.median(diff))
            offset_method = "SmartScanMedian"
    else:
        # IMPORTANT: don't do anything with empty arrays.
        calc_offset_db = 0.0
        meas_level_db_window = 0.0
        target_level_db_window = 0.0
        offset_method = "SmartScanNoMask"

    # Plot/raportti-basis:
    # - Manual: user-defined target SPL
    # - Auto/SmartScan: follow measured SPL in the chosen stable window
    if is_manual:
        target_level_db = float(manual_target_db)
    else:
        if stereo_link and (sl_tgt is not None):
            target_level_db = _to_float(sl_tgt, float(meas_level_db_window))
        else:
            target_level_db = float(meas_level_db_window)

    # Safety: force finitenness
    if not np.isfinite(calc_offset_db):
        calc_offset_db = 0.0
    # Stereo-link: store first computed result for later calls/channels.
    if stereo_link and (sl_win is None) and (sl_off is None) and (sl_tgt is None):
        try:
            setattr(cfg, "_stereo_link_window", (float(ss_min), float(ss_max)))
            setattr(cfg, "_stereo_link_offset_db", float(calc_offset_db))
            setattr(cfg, "_stereo_link_target_level_db", float(target_level_db))
        except Exception:
            pass


    return (
        float(target_level_db),
        float(calc_offset_db),
        float(meas_level_db_window),
        float(target_level_db_window),
        str(offset_method),
        float(ss_min),
        float(ss_max),
    )
