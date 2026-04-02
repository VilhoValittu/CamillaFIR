from __future__ import annotations

from typing import Any, Callable

import numpy as np


def compute_leveling_impl(
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    *,
    stereo_link_ctx: Any | None = None,
    to_float_fn: Callable[[object, float], float],
    to_bool_fn: Callable[[object, bool], bool],
    remember_leveling_error_fn: Callable[[Any, str, Exception | None], None],
    safe_setattr_fn: Callable[[Any, str, Any], None],
    log_median_fn: Callable[[np.ndarray, np.ndarray], float],
    tilt_fit_offset_and_slope_db_per_oct_fn: Callable[..., tuple[float, float]],
    perceptual_shape_score_fn: Callable[..., float],
    find_stable_level_window_fn: Callable[..., tuple[float, float]],
):
    target_level_db = 0.0
    calc_offset_db = 0.0
    meas_level_db_window = 0.0
    target_level_db_window = 0.0
    offset_method = "Unknown"
    perceptual_error_rms = None

    manual_target_db = to_float_fn(getattr(cfg, "lvl_manual_db", 0.0), 0.0)
    s_min = to_float_fn(getattr(cfg, "lvl_min", 500.0), 500.0)
    s_max = to_float_fn(getattr(cfg, "lvl_max", 2000.0), 2000.0)
    if s_min <= 0 or s_max <= 0 or s_min >= s_max:
        s_min, s_max = 500.0, 2000.0

    mode = str(getattr(cfg, "lvl_mode", "Auto"))
    is_manual = "Manual" in mode
    tilt_comp = bool(getattr(cfg, "lvl_tilt_comp", True))
    tilt_max_db_per_oct = to_float_fn(getattr(cfg, "lvl_tilt_max_db_per_oct", 2.0), 2.0)
    try:
        auto_goal = str(getattr(cfg, "auto_goal", "") or "").strip().lower()
    except (AttributeError, TypeError, ValueError):
        auto_goal = ""
    subwoofer_goal = auto_goal in ("subwoofer", "subwoofers", "subs")
    lvl_perceptual_weighting = to_bool_fn(getattr(cfg, "lvl_perceptual_weighting", False), False)
    lvl_perceptual_strength = max(0.0, to_float_fn(getattr(cfg, "lvl_perceptual_strength", 0.12), 0.12))
    lvl_perceptual_min_hz = to_float_fn(getattr(cfg, "lvl_perceptual_min_hz", 250.0), 250.0)
    lvl_perceptual_max_hz = to_float_fn(getattr(cfg, "lvl_perceptual_max_hz", 4000.0), 4000.0)
    lvl_perceptual_tie_only = to_bool_fn(getattr(cfg, "lvl_perceptual_tie_only", True), True)
    if lvl_perceptual_min_hz <= 0.0 or lvl_perceptual_max_hz <= lvl_perceptual_min_hz:
        lvl_perceptual_min_hz, lvl_perceptual_max_hz = 250.0, 4000.0

    safe_setattr_fn(cfg, "_lvl_last_error", None)
    safe_setattr_fn(cfg, "_lvl_tilt_slope_db_per_oct", None)
    safe_setattr_fn(cfg, "_lvl_perceptual_enabled", bool(lvl_perceptual_weighting))
    safe_setattr_fn(cfg, "_lvl_perceptual_strength", float(lvl_perceptual_strength))
    safe_setattr_fn(
        cfg,
        "_lvl_perceptual_band_hz",
        (float(lvl_perceptual_min_hz), float(lvl_perceptual_max_hz)),
    )
    safe_setattr_fn(cfg, "_lvl_perceptual_error_rms", None)
    safe_setattr_fn(cfg, "_lvl_window_debug", None)

    def _store_window_debug(ss_min_value, ss_max_value, offset_method_value, perceptual_value):
        safe_setattr_fn(cfg, "_lvl_perceptual_error_rms", perceptual_value)
        try:
            tilt_slope_value = getattr(cfg, "_lvl_tilt_slope_db_per_oct", None)
        except (AttributeError, TypeError, ValueError):
            tilt_slope_value = None
        safe_setattr_fn(
            cfg,
            "_lvl_window_debug",
            {
                "ss_min": float(ss_min_value),
                "ss_max": float(ss_max_value),
                "offset_method": str(offset_method_value),
                "tilt_slope_db_per_oct": tilt_slope_value,
                "perceptual_error_rms": perceptual_value,
                "perceptual_enabled": bool(lvl_perceptual_weighting),
            },
        )

    def _tilt_fit_for_window(freq_values, diff_values, *, window_hi_hz: float):
        prefer_lf_piecewise_tilt = bool(subwoofer_goal and np.isfinite(float(window_hi_hz)) and float(window_hi_hz) <= 220.0)
        return tilt_fit_offset_and_slope_db_per_oct_fn(
            freq_values,
            diff_values,
            max_db_per_oct=float(tilt_max_db_per_oct),
            prefer_lf_piecewise_tilt=bool(prefer_lf_piecewise_tilt),
        )

    try:
        target_arr = np.asarray(target_mags, dtype=float)
        if target_arr.shape != np.asarray(freq_axis).shape:
            target_arr = None
    except (TypeError, ValueError):
        target_arr = None

    forced_window = getattr(cfg, "lvl_force_window", None)
    forced_offset = getattr(cfg, "lvl_force_offset_db", None)
    shared_target_level_db = None
    if stereo_link_ctx is not None:
        try:
            if stereo_link_ctx.forced_window_hz is not None:
                forced_window = stereo_link_ctx.forced_window_hz
            if stereo_link_ctx.forced_offset_db is not None:
                forced_offset = stereo_link_ctx.forced_offset_db
            if stereo_link_ctx.shared_target_level_db is not None:
                shared_target_level_db = float(stereo_link_ctx.shared_target_level_db)
        except (AttributeError, TypeError, ValueError):
            pass
    if shared_target_level_db is not None and (not np.isfinite(float(shared_target_level_db))):
        shared_target_level_db = None

    if forced_window is not None or forced_offset is not None:
        try:
            if forced_window is not None:
                fw0, fw1 = forced_window
                ss_min = to_float_fn(fw0, s_min)
                ss_max = to_float_fn(fw1, s_max)
                if (ss_min <= 0) or (ss_max <= 0) or (ss_min >= ss_max):
                    ss_min, ss_max = s_min, s_max
                ss_min = max(s_min, ss_min)
                ss_max = min(s_max, ss_max)
            else:
                ss_min, ss_max = s_min, s_max

            mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)
            if np.any(mask):
                meas_level_db_window = log_median_fn(freq_axis[mask], m_anal[mask])
                target_level_db_window = log_median_fn(freq_axis[mask], target_mags[mask])
            else:
                meas_level_db_window = 0.0
                target_level_db_window = 0.0

            if forced_offset is not None:
                calc_offset_db = to_float_fn(forced_offset, 0.0)
                offset_method = "ForcedOffset"
                try:
                    if tilt_comp and np.any(mask):
                        _off_tmp, slope = _tilt_fit_for_window(
                            freq_axis[mask],
                            (m_anal[mask] - target_mags[mask]),
                            window_hi_hz=float(ss_max),
                        )
                        safe_setattr_fn(cfg, "_lvl_tilt_slope_db_per_oct", float(slope))
                except (TypeError, ValueError, FloatingPointError, IndexError):
                    pass
            else:
                if np.any(mask):
                    diff = m_anal[mask] - target_mags[mask]
                    if tilt_comp:
                        calc_offset_db, tilt_slope = _tilt_fit_for_window(
                            freq_axis[mask],
                            diff,
                            window_hi_hz=float(ss_max),
                        )
                        safe_setattr_fn(cfg, "_lvl_tilt_slope_db_per_oct", float(tilt_slope))
                        offset_method = "ForcedWindowTiltMedian"
                    else:
                        calc_offset_db = log_median_fn(freq_axis[mask], diff)
                        offset_method = "ForcedWindowMedian"
                else:
                    calc_offset_db = 0.0
                    offset_method = "ForcedWindowNoMask"

            if bool(lvl_perceptual_weighting) and np.count_nonzero(mask) >= 20 and target_arr is not None:
                perceptual_error_rms = perceptual_shape_score_fn(
                    freq_axis[mask],
                    m_anal[mask],
                    target_arr[mask],
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    min_hz=float(lvl_perceptual_min_hz),
                    max_hz=float(lvl_perceptual_max_hz),
                )
                if not np.isfinite(float(perceptual_error_rms)):
                    perceptual_error_rms = None
            else:
                perceptual_error_rms = None

            if is_manual:
                target_level_db = float(manual_target_db)
            elif shared_target_level_db is not None:
                target_level_db = to_float_fn(shared_target_level_db, float(meas_level_db_window))
            else:
                target_level_db = float(meas_level_db_window)

            if not np.isfinite(calc_offset_db):
                calc_offset_db = 0.0
            _store_window_debug(ss_min, ss_max, offset_method, perceptual_error_rms)
            return (
                float(target_level_db),
                float(calc_offset_db),
                float(meas_level_db_window),
                float(target_level_db_window),
                str(offset_method),
                float(ss_min),
                float(ss_max),
            )
        except (TypeError, ValueError, FloatingPointError, ZeroDivisionError) as exc:
            remember_leveling_error_fn(cfg, "forced_window", exc)

    if is_manual:
        mask = (freq_axis >= s_min) & (freq_axis <= s_max)
        if np.any(mask):
            meas_level_db_window = log_median_fn(freq_axis[mask], m_anal[mask])
            target_level_db_window = log_median_fn(freq_axis[mask], target_mags[mask])
            calc_offset_db = log_median_fn(freq_axis[mask], (m_anal[mask] - target_mags[mask]))
            offset_method = "ManualMedian"
        else:
            calc_offset_db = 0.0
            offset_method = "ManualNoMask"

        if bool(lvl_perceptual_weighting) and np.count_nonzero(mask) >= 20 and target_arr is not None:
            perceptual_error_rms = perceptual_shape_score_fn(
                freq_axis[mask],
                m_anal[mask],
                target_arr[mask],
                tilt_comp=bool(tilt_comp),
                tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                min_hz=float(lvl_perceptual_min_hz),
                max_hz=float(lvl_perceptual_max_hz),
            )
            if not np.isfinite(float(perceptual_error_rms)):
                perceptual_error_rms = None
        else:
            perceptual_error_rms = None

        target_level_db = float(manual_target_db)
        if not np.isfinite(calc_offset_db):
            calc_offset_db = 0.0
        _store_window_debug(s_min, s_max, offset_method, perceptual_error_rms)
        return (
            float(target_level_db),
            float(calc_offset_db),
            float(meas_level_db_window),
            float(target_level_db_window),
            str(offset_method),
            float(s_min),
            float(s_max),
        )

    hpf_freq = 0.0
    hpf_settings = getattr(cfg, "hpf_settings", None)
    if hpf_settings:
        try:
            hpf_freq = to_float_fn(hpf_settings.get("freq", 0.0), 0.0)
        except (AttributeError, TypeError, ValueError) as exc:
            remember_leveling_error_fn(cfg, "hpf_settings", exc)
            hpf_freq = 0.0

    ss_min, ss_max = find_stable_level_window_fn(
        freq_axis,
        m_anal,
        target_mags,
        s_min,
        s_max,
        window_size_octaves=1.0,
        hpf_freq=float(hpf_freq),
        tilt_comp=bool(tilt_comp),
        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
        perceptual_weighting=bool(lvl_perceptual_weighting),
        perceptual_strength=float(lvl_perceptual_strength),
        perceptual_min_hz=float(lvl_perceptual_min_hz),
        perceptual_max_hz=float(lvl_perceptual_max_hz),
        perceptual_tie_only=bool(lvl_perceptual_tie_only),
    )

    ss_min = to_float_fn(ss_min, s_min)
    ss_max = to_float_fn(ss_max, s_max)
    if (ss_min <= 0) or (ss_max <= 0) or (ss_min >= ss_max):
        ss_min, ss_max = s_min, s_max
    ss_min = max(s_min, ss_min)
    ss_max = min(s_max, ss_max)

    mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)
    if np.count_nonzero(mask) < 20:
        fb_min = max(s_min, 350.0)
        fb_max = min(s_max, 5000.0)
        if fb_min < fb_max:
            ss_min, ss_max = fb_min, fb_max
            mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)
    if np.count_nonzero(mask) < 20:
        ss_min, ss_max = s_min, s_max
        mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)

    if np.any(mask):
        meas_level_db_window = log_median_fn(freq_axis[mask], m_anal[mask])
        target_level_db_window = log_median_fn(freq_axis[mask], target_mags[mask])
        diff = m_anal[mask] - target_mags[mask]
        if tilt_comp:
            calc_offset_db, tilt_slope = _tilt_fit_for_window(
                freq_axis[mask],
                diff,
                window_hi_hz=float(ss_max),
            )
            safe_setattr_fn(cfg, "_lvl_tilt_slope_db_per_oct", float(tilt_slope))
            offset_method = "SmartScanTiltMedian"
        else:
            calc_offset_db = log_median_fn(freq_axis[mask], diff)
            offset_method = "SmartScanMedian"
    else:
        calc_offset_db = 0.0
        meas_level_db_window = 0.0
        target_level_db_window = 0.0
        offset_method = "SmartScanNoMask"

    if bool(lvl_perceptual_weighting) and np.count_nonzero(mask) >= 20 and target_arr is not None:
        perceptual_error_rms = perceptual_shape_score_fn(
            freq_axis[mask],
            m_anal[mask],
            target_arr[mask],
            tilt_comp=bool(tilt_comp),
            tilt_max_db_per_oct=float(tilt_max_db_per_oct),
            min_hz=float(lvl_perceptual_min_hz),
            max_hz=float(lvl_perceptual_max_hz),
        )
        if not np.isfinite(float(perceptual_error_rms)):
            perceptual_error_rms = None
    else:
        perceptual_error_rms = None

    if is_manual:
        target_level_db = float(manual_target_db)
    elif shared_target_level_db is not None:
        target_level_db = to_float_fn(shared_target_level_db, float(meas_level_db_window))
    else:
        target_level_db = float(meas_level_db_window)

    if not np.isfinite(calc_offset_db):
        calc_offset_db = 0.0
    _store_window_debug(ss_min, ss_max, offset_method, perceptual_error_rms)
    return (
        float(target_level_db),
        float(calc_offset_db),
        float(meas_level_db_window),
        float(target_level_db_window),
        str(offset_method),
        float(ss_min),
        float(ss_max),
    )
