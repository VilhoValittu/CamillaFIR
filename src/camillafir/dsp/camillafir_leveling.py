from __future__ import annotations
from typing import Tuple
import numpy as np

__all__ = [
    "find_stable_level_window",
    "compute_leveling",
]


def _to_float(x, default: float) -> float:
    """Sisainen apufunktio: to float."""
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)

def _remember_leveling_error(cfg, stage: str, exc: Exception | None = None) -> None:
    try:
        if exc is None:
            msg = str(stage)
        else:
            msg = f"{stage}:{type(exc).__name__}"
        setattr(cfg, "_lvl_last_error", msg)
    except Exception:
        return

def _tilt_aware_offset_db(
    freq_axis: np.ndarray,
    diff_db: np.ndarray,
    *,
    max_db_per_oct: float = 2.0,
) -> float:
    """Sisainen apufunktio: tilt aware offset db."""
    try:
        f = np.asarray(freq_axis, dtype=float)
        y = np.asarray(diff_db, dtype=float)
        if f.size != y.size or f.size < 20:
            return float(np.median(y)) if y.size else 0.0

        mask = np.isfinite(f) & np.isfinite(y) & (f > 0.0)
        f = f[mask]
        y = y[mask]
        if f.size < 20:
            return float(np.median(y)) if y.size else 0.0

        x = np.log2(f)

        x0 = float(np.median(x))
        xc = x - x0

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
            return float(np.median(y))

        slope = float(np.dot(xc, (y - float(np.median(y)))) / denom)
        max_db_per_oct = float(max_db_per_oct)
        if not np.isfinite(max_db_per_oct) or max_db_per_oct <= 0:
            max_db_per_oct = 2.0
        slope = float(np.clip(slope, -max_db_per_oct, +max_db_per_oct))

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
    """Sisainen apufunktio: tilt fit offset and slope db per oct."""
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
    """Hakee tai ratkaisee: find stable level window."""
    try:
        _ = target_mags

        f_min = _to_float(f_min, 0.0)
        f_max = _to_float(f_max, 0.0)
        hpf_freq = _to_float(hpf_freq, 0.0)
        window_size_octaves = _to_float(window_size_octaves, 1.0)

        if f_min <= 0 or f_max <= 0 or f_min >= f_max:
            return float(f_min), float(f_max)

        safe_f_min = max(float(f_min), float(hpf_freq) * 1.5)
        if safe_f_min >= float(f_max) * 0.8:
            safe_f_min = float(f_min)

        mask = (freq_axis >= safe_f_min) & (freq_axis <= float(f_max))
        f_search = freq_axis[mask]
        m_search = np.asarray(magnitudes, dtype=float)[mask]

        if f_search.size < 50:
            return float(f_min), float(f_max)

        best_score = float("inf")
        res_min, res_max = float(safe_f_min), float(f_max)

        current_f = float(safe_f_min)
        step = 2 ** (1 / 24.0)

        while current_f * (2 ** float(window_size_octaves)) <= float(f_max):
            w_start = current_f
            w_end = current_f * (2 ** float(window_size_octaves))
            w_mask = (f_search >= w_start) & (f_search <= w_end)
            n_w = int(np.count_nonzero(w_mask))
            if n_w >= 20:
                f_w = f_search[w_mask]
                m_w = m_search[w_mask]

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

                weight = 1.0 + 0.05 * abs(np.log10(max(w_start, 1.0) / 1000.0))
                score = std * weight

                if score < best_score:
                    best_score = score
                    res_min, res_max = float(w_start), float(w_end)

            current_f *= step

        if not np.isfinite(best_score):
            return float(f_min), float(f_max)

        return float(res_min), float(res_max)

    except Exception:
        return float(f_min), float(f_max)


def compute_leveling(cfg, freq_axis: np.ndarray, m_anal: np.ndarray, target_mags: np.ndarray):
    """Laskee: compute leveling."""
    target_level_db = 0.0
    calc_offset_db = 0.0
    meas_level_db_window = 0.0
    target_level_db_window = 0.0
    offset_method = "Unknown"

    manual_target_db = _to_float(getattr(cfg, "lvl_manual_db", 0.0), 0.0)

    s_min = _to_float(getattr(cfg, "lvl_min", 500.0), 500.0)
    s_max = _to_float(getattr(cfg, "lvl_max", 2000.0), 2000.0)

    if s_min <= 0 or s_max <= 0 or s_min >= s_max:
        s_min, s_max = 500.0, 2000.0

    mode = str(getattr(cfg, "lvl_mode", "Auto"))
    is_manual = ("Manual" in mode)
    tilt_comp = bool(getattr(cfg, "lvl_tilt_comp", True))
    tilt_max_db_per_oct = _to_float(getattr(cfg, "lvl_tilt_max_db_per_oct", 2.0), 2.0)

    try:
        setattr(cfg, "_lvl_last_error", None)
    except Exception:
        pass

    try:
        setattr(cfg, "_lvl_tilt_slope_db_per_oct", None)
    except Exception:
        pass

    stereo_link = bool(getattr(cfg, "stereo_link", False))
    sl_win = getattr(cfg, "_stereo_link_window", None) if stereo_link else None
    sl_off = getattr(cfg, "_stereo_link_offset_db", None) if stereo_link else None
    sl_tgt = getattr(cfg, "_stereo_link_target_level_db", None) if stereo_link else None

    forced_window = getattr(cfg, "lvl_force_window", None)
    forced_offset = getattr(cfg, "lvl_force_offset_db", None)

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

            if is_manual:
                target_level_db = float(manual_target_db)
            else:
                if stereo_link and (sl_tgt is not None):
                    target_level_db = _to_float(sl_tgt, float(meas_level_db_window))
                else:
                    target_level_db = float(meas_level_db_window)

            if not np.isfinite(calc_offset_db):
                calc_offset_db = 0.0

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
        except Exception as exc:
            _remember_leveling_error(cfg, "forced_window", exc)


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

        target_level_db = float(manual_target_db)

        if not np.isfinite(calc_offset_db):
            calc_offset_db = 0.0
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

    hpf_freq = 0.0
    hpf_settings = getattr(cfg, "hpf_settings", None)
    if hpf_settings:
        try:
            hpf_freq = _to_float(hpf_settings.get("freq", 0.0), 0.0)
        except Exception as exc:
            _remember_leveling_error(cfg, "hpf_settings", exc)
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

    ss_min = _to_float(ss_min, s_min)
    ss_max = _to_float(ss_max, s_max)
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
        calc_offset_db = 0.0
        meas_level_db_window = 0.0
        target_level_db_window = 0.0
        offset_method = "SmartScanNoMask"

    if is_manual:
        target_level_db = float(manual_target_db)
    else:
        if stereo_link and (sl_tgt is not None):
            target_level_db = _to_float(sl_tgt, float(meas_level_db_window))
        else:
            target_level_db = float(meas_level_db_window)

    if not np.isfinite(calc_offset_db):
        calc_offset_db = 0.0
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
