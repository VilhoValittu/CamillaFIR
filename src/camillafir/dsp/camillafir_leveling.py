from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

__all__ = [
    "StereoLinkContext",
    "find_stable_level_window",
    "compute_leveling",
]

@dataclass(frozen=True)
class StereoLinkContext:


    forced_window_hz: tuple[float, float] | None = None
    forced_offset_db: float | None = None
    shared_target_level_db: float | None = None
    shared_target_shift_db: float | None = None


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


def _resample_log_axis(
    freq_axis: np.ndarray,
    *series: np.ndarray,
    points_per_octave: float = 48.0,
    min_points: int = 32,
):
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        if f.size == 0:
            return f, tuple(np.asarray(s, dtype=float).reshape(-1)[:0] for s in series)

        mask = np.isfinite(f) & (f > 0.0)
        prepared = []
        for s in series:
            v = np.asarray(s, dtype=float).reshape(-1)
            if v.size != f.size:
                return f[:0], tuple(np.asarray(v[:0], dtype=float) for v in prepared)
            prepared.append(v)
            mask &= np.isfinite(v)

        f = f[mask]
        prepared = [v[mask] for v in prepared]
        if f.size == 0:
            return f, tuple(v for v in prepared)

        order = np.argsort(f, kind="mergesort")
        f = f[order]
        prepared = [v[order] for v in prepared]

        if f.size > 1:
            uniq = np.concatenate(([True], np.diff(f) > 0.0))
            f = f[uniq]
            prepared = [v[uniq] for v in prepared]

        if f.size < 2:
            return f, tuple(prepared)

        f_lo = float(f[0])
        f_hi = float(f[-1])
        if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)) or (f_hi <= f_lo):
            return f, tuple(prepared)

        octaves = float(np.log2(f_hi / f_lo))
        if not np.isfinite(octaves) or octaves <= 0.0:
            return f, tuple(prepared)

        ppo = float(points_per_octave)
        if (not np.isfinite(ppo)) or (ppo <= 0.0):
            ppo = 48.0
        n_points = max(int(np.ceil(octaves * ppo)) + 1, int(min_points))
        f_log = np.geomspace(f_lo, f_hi, n_points)
        prepared_log = [np.interp(f_log, f, v) for v in prepared]
        return f_log, tuple(prepared_log)
    except Exception:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        prepared = tuple(np.asarray(s, dtype=float).reshape(-1) for s in series)
        return f, prepared


def _log_median(freq_axis: np.ndarray, values: np.ndarray) -> float:
    try:
        _f_log, (v_log,) = _resample_log_axis(freq_axis, values)
        if v_log.size == 0:
            return 0.0
        return float(np.median(v_log))
    except Exception:
        try:
            return float(np.median(values))
        except Exception:
            return 0.0


def _lower_tail_robust_std_db(values: np.ndarray, *, clip_below_db: float = 6.0) -> float:
    try:
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return 0.0

        med = float(np.median(y))
        clip_below_db = _to_float(clip_below_db, 6.0)
        if clip_below_db > 0.0:
            keep = y >= (med - float(clip_below_db))
            if np.count_nonzero(keep) >= max(12, int(np.ceil(y.size * 0.4))):
                y = y[keep]

        center = float(np.median(y))
        mad = float(np.median(np.abs(y - center)))
        robust = 1.4826 * mad
        if np.isfinite(robust) and robust > 1e-9:
            return float(robust)
        return float(np.std(y))
    except Exception:
        try:
            return float(np.std(values))
        except Exception:
            return 0.0


def _centered_rms(values: np.ndarray) -> float:
    try:
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return float("inf")
        y = y - float(np.median(y))
        return float(np.sqrt(np.mean(y * y)))
    except Exception:
        return float("inf")


def _window_offset_consistency_score(
    freq_axis: np.ndarray,
    measured_db: np.ndarray,
    target_db: np.ndarray | None,
    *,
    tilt_comp: bool = True,
    tilt_max_db_per_oct: float = 2.0,
) -> tuple[float, float, float]:
    """
    Arvioi kuinka luotettava level-offset on ikkunan sisalla.

    Palauttaa kolmen metriikan tuplen:
    1) offset-spread (dB) eri alajaksojen valilla
    2) shape RMS targetiin nahden (offset/tilt poistettuna)
    3) absoluuttinen tilt (dB/okt)
    """
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        m = np.asarray(measured_db, dtype=float).reshape(-1)
        if target_db is None:
            t = np.zeros_like(m, dtype=float)
        else:
            t = np.asarray(target_db, dtype=float).reshape(-1)
        if f.size < 24 or m.size != f.size or t.size != f.size:
            return float("inf"), float("inf"), float("inf")

        valid = np.isfinite(f) & np.isfinite(m) & np.isfinite(t) & (f > 0.0)
        if int(np.count_nonzero(valid)) < 24:
            return float("inf"), float("inf"), float("inf")

        f = f[valid]
        diff = np.asarray(m[valid] - t[valid], dtype=float)
        if f.size < 24:
            return float("inf"), float("inf"), float("inf")

        if tilt_comp:
            off_full, slope_full = _tilt_fit_offset_and_slope_db_per_oct(
                f,
                diff,
                max_db_per_oct=float(tilt_max_db_per_oct),
            )
        else:
            off_full = _log_median(f, diff)
            slope_full = 0.0

        x = np.log2(np.clip(f, 1e-9, None))
        xc = x - float(np.median(x))
        shape_resid = diff - float(off_full) - (float(slope_full) * xc)
        shape_rms = _centered_rms(shape_resid)

        parts: list[tuple[float, float]] = []
        n = int(f.size)
        half = n // 2
        if half >= 12:
            parts.append((0.0, 0.5))
            parts.append((0.5, 1.0))

        third = n // 3
        if third >= 10:
            parts.append((0.0, 2.0 / 3.0))
            parts.append((1.0 / 3.0, 1.0))

        parts.append((0.2, 0.8))

        offsets = [float(off_full)]
        for start_frac, end_frac in parts:
            i0 = int(np.floor(float(start_frac) * n))
            i1 = int(np.ceil(float(end_frac) * n))
            i0 = max(0, min(i0, n - 1))
            i1 = max(i0 + 1, min(i1, n))
            if (i1 - i0) < 10:
                continue
            f_sub = f[i0:i1]
            d_sub = diff[i0:i1]
            if tilt_comp:
                off_sub, _ = _tilt_fit_offset_and_slope_db_per_oct(
                    f_sub,
                    d_sub,
                    max_db_per_oct=float(tilt_max_db_per_oct),
                )
            else:
                off_sub = _log_median(f_sub, d_sub)
            if np.isfinite(off_sub):
                offsets.append(float(off_sub))

        if len(offsets) >= 3:
            offset_spread = float(np.std(np.asarray(offsets, dtype=float)))
        else:
            offset_spread = 0.0

        return float(offset_spread), float(shape_rms), float(abs(slope_full))
    except Exception:
        return float("inf"), float("inf"), float("inf")


def _tilt_aware_offset_db(
    freq_axis: np.ndarray,
    diff_db: np.ndarray,
    *,
    max_db_per_oct: float = 2.0,
) -> float:

    try:
        f, (y,) = _resample_log_axis(freq_axis, diff_db)
        if f.size != y.size or f.size < 20:
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
        f, (y,) = _resample_log_axis(freq_axis, diff_db)
        if f.size != y.size or f.size < 20:
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
    tilt_comp: bool = True,
    tilt_max_db_per_oct: float = 2.0,
) -> Tuple[float, float]:

    try:
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
        try:
            t_arr = np.asarray(target_mags, dtype=float)
            t_search = t_arr[mask] if t_arr.shape == np.asarray(freq_axis).shape else None
        except Exception:
            t_search = None

        if f_search.size < 50:
            return float(f_min), float(f_max)

        best_score = float("inf")
        best_target_rms = float("inf")
        best_offset_spread = float("inf")
        best_tilt_abs = float("inf")
        res_min, res_max = float(safe_f_min), float(f_max)
        tie_eps_rel = 0.05

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
                if t_search is not None:
                    f_eval, (y, t_w) = _resample_log_axis(f_w, m_w, np.asarray(t_search[w_mask], dtype=float))
                else:
                    f_eval, (y,) = _resample_log_axis(f_w, m_w)
                    t_w = None
                if f_eval.size < 20 or y.size < 20:
                    current_f *= step
                    continue

                x = np.log2(np.clip(f_eval, 1e-9, None))
                x0 = float(np.median(x))
                xc = x - x0
                y_med = float(np.median(y))
                denom = float(np.dot(xc, xc))
                if denom > 1e-12:
                    slope = float(np.dot(xc, (y - y_med)) / denom)
                    residual = y - (slope * xc)
                else:
                    residual = y

                std = _lower_tail_robust_std_db(residual, clip_below_db=6.0)

                weight = 1.0 + 0.05 * abs(np.log10(max(w_start, 1.0) / 1000.0))
                score = std * weight

                target_rms = float("inf")
                offset_spread = float("inf")
                tilt_abs = float("inf")
                try:
                    if t_w is not None and t_w.size == y.size:
                        offset_spread, target_rms, tilt_abs = _window_offset_consistency_score(
                            f_eval,
                            y,
                            t_w,
                            tilt_comp=bool(tilt_comp),
                            tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                        )
                    else:
                        offset_spread, _shape_rms, tilt_abs = _window_offset_consistency_score(
                            f_eval,
                            y,
                            None,
                            tilt_comp=bool(tilt_comp),
                            tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                        )
                except Exception:
                    target_rms = float("inf")
                    offset_spread = float("inf")
                    tilt_abs = float("inf")

                if np.isfinite(offset_spread):
                    score += 0.85 * float(offset_spread)
                if np.isfinite(target_rms):
                    score += 0.20 * float(target_rms)
                if np.isfinite(tilt_abs):
                    score += 0.08 * float(tilt_abs)

                better_stability = score < (best_score * (1.0 - tie_eps_rel))
                near_tie = (score <= (best_score * (1.0 + tie_eps_rel)))
                better_tie_break = near_tie and (
                    (offset_spread < best_offset_spread)
                    or (
                        offset_spread <= (best_offset_spread + 1e-6)
                        and (
                            (target_rms < best_target_rms)
                            or (
                                target_rms <= (best_target_rms + 1e-6)
                                and tilt_abs < best_tilt_abs
                            )
                        )
                    )
                )

                if better_stability or better_tie_break:
                    best_score = score
                    best_target_rms = target_rms
                    best_offset_spread = offset_spread
                    best_tilt_abs = tilt_abs
                    res_min, res_max = float(w_start), float(w_end)

            current_f *= step

        if not np.isfinite(best_score):
            return float(f_min), float(f_max)

        return float(res_min), float(res_max)

    except Exception:
        return float(f_min), float(f_max)


def compute_leveling(
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    *,
    stereo_link_ctx: StereoLinkContext | None = None,
):
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
        except Exception:
            pass
    if shared_target_level_db is not None and (not np.isfinite(float(shared_target_level_db))):
        shared_target_level_db = None

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
                meas_level_db_window = _log_median(freq_axis[mask], m_anal[mask])
                target_level_db_window = _log_median(freq_axis[mask], target_mags[mask])
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
                        calc_offset_db = _log_median(freq_axis[mask], diff)
                        offset_method = "ForcedWindowMedian"
                else:
                    calc_offset_db = 0.0
                    offset_method = "ForcedWindowNoMask"

            if is_manual:
                target_level_db = float(manual_target_db)
            else:
                if shared_target_level_db is not None:
                    target_level_db = _to_float(shared_target_level_db, float(meas_level_db_window))
                else:
                    target_level_db = float(meas_level_db_window)

            if not np.isfinite(calc_offset_db):
                calc_offset_db = 0.0

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
            meas_level_db_window = _log_median(freq_axis[mask], m_anal[mask])
            target_level_db_window = _log_median(freq_axis[mask], target_mags[mask])
            calc_offset_db = _log_median(freq_axis[mask], (m_anal[mask] - target_mags[mask]))
            offset_method = "ManualMedian"
        else:
            calc_offset_db = 0.0
            offset_method = "ManualNoMask"

        target_level_db = float(manual_target_db)

        if not np.isfinite(calc_offset_db):
            calc_offset_db = 0.0

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
        tilt_comp=bool(tilt_comp),
        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
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
        meas_level_db_window = _log_median(freq_axis[mask], m_anal[mask])
        target_level_db_window = _log_median(freq_axis[mask], target_mags[mask])
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
            calc_offset_db = _log_median(freq_axis[mask], diff)
            offset_method = "SmartScanMedian"
    else:
        calc_offset_db = 0.0
        meas_level_db_window = 0.0
        target_level_db_window = 0.0
        offset_method = "SmartScanNoMask"

    if is_manual:
        target_level_db = float(manual_target_db)
    else:
        if shared_target_level_db is not None:
            target_level_db = _to_float(shared_target_level_db, float(meas_level_db_window))
        else:
            target_level_db = float(meas_level_db_window)

    if not np.isfinite(calc_offset_db):
        calc_offset_db = 0.0


    return (
        float(target_level_db),
        float(calc_offset_db),
        float(meas_level_db_window),
        float(target_level_db_window),
        str(offset_method),
        float(ss_min),
        float(ss_max),
    )
