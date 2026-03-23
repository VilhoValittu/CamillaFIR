from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

__all__ = [
    "StereoLinkContext",
    "find_stable_level_window",
    "find_shared_stereo_level_window",
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
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _to_bool(x, default: bool) -> bool:
    """Sisainen apufunktio: to bool."""
    try:
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if x is None:
            return bool(default)
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"1", "true", "yes", "on", "y"}:
                return True
            if s in {"0", "false", "no", "off", "n", ""}:
                return False
            return bool(default)
        if isinstance(x, (int, float, np.integer, np.floating)):
            v = float(x)
            if not np.isfinite(v):
                return bool(default)
            return bool(v != 0.0)
        return bool(x)
    except (TypeError, ValueError, OverflowError):
        return bool(default)


def _remember_leveling_error(cfg, stage: str, exc: Exception | None = None) -> None:
    try:
        if exc is None:
            msg = str(stage)
        else:
            msg = f"{stage}:{type(exc).__name__}"
        setattr(cfg, "_lvl_last_error", msg)
    except (AttributeError, TypeError, ValueError):
        return


def _safe_setattr(cfg, name: str, value) -> None:
    try:
        setattr(cfg, name, value)
    except (AttributeError, TypeError, ValueError):
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
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        prepared = tuple(np.asarray(s, dtype=float).reshape(-1) for s in series)
        return f, prepared


def _log_median(freq_axis: np.ndarray, values: np.ndarray) -> float:
    try:
        _f_log, (v_log,) = _resample_log_axis(freq_axis, values)
        if v_log.size == 0:
            return 0.0
        return float(np.median(v_log))
    except (TypeError, ValueError, FloatingPointError):
        try:
            return float(np.median(values))
        except (TypeError, ValueError, FloatingPointError):
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
    except (TypeError, ValueError, FloatingPointError):
        try:
            return float(np.std(values))
        except (TypeError, ValueError, FloatingPointError):
            return 0.0


def _centered_rms(values: np.ndarray) -> float:
    try:
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return float("inf")
        y = y - float(np.median(y))
        return float(np.sqrt(np.mean(y * y)))
    except (TypeError, ValueError, FloatingPointError):
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
    except (TypeError, ValueError, FloatingPointError, IndexError):
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
    except (TypeError, ValueError, FloatingPointError):
        try:
            return float(np.median(diff_db))
        except (TypeError, ValueError, FloatingPointError):
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
    except (TypeError, ValueError, FloatingPointError):
        try:
            return float(np.median(diff_db)), 0.0
        except (TypeError, ValueError, FloatingPointError):
            return 0.0, 0.0


def _hz_to_erb_number(freq_hz):
    try:
        f = np.asarray(freq_hz, dtype=float)
        f = np.clip(f, 0.0, None)
        erb = 21.4 * np.log10(1.0 + 4.37e-3 * f)
        if np.ndim(erb) == 0:
            return float(erb)
        return erb
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        try:
            f = max(float(freq_hz), 0.0)
            return float(21.4 * np.log10(1.0 + 4.37e-3 * f))
        except (TypeError, ValueError, FloatingPointError, OverflowError):
            return 0.0


def _perceptual_importance_weights(
    freq_axis,
    *,
    min_hz=250.0,
    max_hz=4000.0,
    min_weight=0.60,
    max_weight=1.35,
    bass_floor_weight=0.85,
):
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        if f.size == 0:
            return np.asarray([], dtype=float)

        min_hz = _to_float(min_hz, 250.0)
        max_hz = _to_float(max_hz, 4000.0)
        min_weight = _to_float(min_weight, 0.60)
        max_weight = _to_float(max_weight, 1.35)
        bass_floor_weight = _to_float(bass_floor_weight, 0.85)

        if min_hz <= 0.0:
            min_hz = 250.0
        if max_hz <= min_hz:
            min_hz, max_hz = 250.0, 4000.0
        if max_weight < min_weight:
            max_weight = min_weight

        weights = np.full(f.shape, float(min_weight), dtype=float)
        valid = np.isfinite(f) & (f > 0.0)
        if not np.any(valid):
            return weights

        erb = np.asarray(_hz_to_erb_number(f[valid]), dtype=float)
        erb_lo = float(_hz_to_erb_number(min_hz))
        erb_hi = float(_hz_to_erb_number(max_hz))
        erb_span = max(float(erb_hi - erb_lo), 1e-6)
        erb_mid = 0.5 * (erb_lo + erb_hi)

        shape_sigma = max(0.34 * erb_span, 1e-6)
        gate_sigma = max(0.10 * erb_span, 1e-6)

        mid_shape = np.exp(-0.5 * ((erb - erb_mid) / shape_sigma) ** 2)
        lo_gate = 1.0 / (1.0 + np.exp(-(erb - erb_lo) / gate_sigma))
        hi_gate = 1.0 / (1.0 + np.exp((erb - erb_hi) / gate_sigma))
        band_gate = lo_gate * hi_gate

        w_valid = min_weight + (max_weight - min_weight) * mid_shape * band_gate

        low_mask = f[valid] < min_hz
        if np.any(low_mask):
            w_valid[low_mask] = np.maximum(w_valid[low_mask], float(bass_floor_weight))

        w_lo = min(min_weight, max_weight, bass_floor_weight)
        w_hi = max(min_weight, max_weight, bass_floor_weight)
        weights[valid] = np.clip(w_valid, w_lo, w_hi)
        return weights
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        try:
            f = np.asarray(freq_axis, dtype=float).reshape(-1)
            return np.ones_like(f, dtype=float)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)


def _weighted_centered_rms(values, weights):
    try:
        y = np.asarray(values, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if y.size == 0 or w.size != y.size:
            return float("inf")

        valid = np.isfinite(y) & np.isfinite(w) & (w > 0.0)
        if not np.any(valid):
            return float("inf")

        y = y[valid]
        w = w[valid]
        w_sum = float(np.sum(w))
        if (not np.isfinite(w_sum)) or (w_sum <= 1e-12):
            return float("inf")

        y_mean = float(np.sum(w * y) / w_sum)
        resid = y - y_mean
        return float(np.sqrt(np.sum(w * resid * resid) / w_sum))
    except (TypeError, ValueError, FloatingPointError):
        return float("inf")


def _perceptual_shape_score(
    freq_axis,
    measured_db,
    target_db,
    *,
    tilt_comp=True,
    tilt_max_db_per_oct=2.0,
    min_hz=250.0,
    max_hz=4000.0,
):
    # This is a lightweight perceptual-weighting surrogate, not ANSI S3.4 loudness.
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        m = np.asarray(measured_db, dtype=float).reshape(-1)
        t = np.asarray(target_db, dtype=float).reshape(-1)
        if f.size < 12 or m.size != f.size or t.size != f.size:
            return float("inf")

        valid = np.isfinite(f) & np.isfinite(m) & np.isfinite(t) & (f > 0.0)
        if int(np.count_nonzero(valid)) < 12:
            return float("inf")

        f = f[valid]
        m = m[valid]
        t = t[valid]
        f, (m, t) = _resample_log_axis(f, m, t)
        if f.size < 12 or m.size != f.size or t.size != f.size:
            return float("inf")

        diff = np.asarray(m - t, dtype=float)
        if tilt_comp:
            off, slope = _tilt_fit_offset_and_slope_db_per_oct(
                f,
                diff,
                max_db_per_oct=float(tilt_max_db_per_oct),
            )
        else:
            off = _log_median(f, diff)
            slope = 0.0

        x = np.log2(np.clip(f, 1e-9, None))
        xc = x - float(np.median(x))
        resid = diff - float(off) - (float(slope) * xc)

        weights = _perceptual_importance_weights(
            f,
            min_hz=min_hz,
            max_hz=max_hz,
        )
        return _weighted_centered_rms(resid, weights)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float("inf")


def _prepare_level_window_search(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray | None,
    *,
    f_min: float,
    f_max: float,
    hpf_freq: float,
):
    try:
        freq_arr = np.asarray(freq_axis, dtype=float).reshape(-1)
        mag_arr = np.asarray(magnitudes, dtype=float).reshape(-1)
        if freq_arr.size != mag_arr.size or freq_arr.size == 0:
            return None

        safe_f_min = max(float(f_min), float(hpf_freq) * 1.5)
        if safe_f_min >= float(f_max) * 0.8:
            safe_f_min = float(f_min)

        mask = (
            np.isfinite(freq_arr)
            & np.isfinite(mag_arr)
            & (freq_arr >= float(safe_f_min))
            & (freq_arr <= float(f_max))
        )

        target_search = None
        if target_mags is not None:
            try:
                target_arr = np.asarray(target_mags, dtype=float).reshape(-1)
                if target_arr.size == freq_arr.size:
                    mask &= np.isfinite(target_arr)
                    target_search = target_arr
            except (TypeError, ValueError):
                target_search = None

        if int(np.count_nonzero(mask)) < 50:
            return None

        out = {
            "freq": freq_arr[mask],
            "magnitudes": mag_arr[mask],
            "safe_f_min": float(safe_f_min),
            "f_max": float(f_max),
        }
        if target_search is not None:
            out["target"] = np.asarray(target_search[mask], dtype=float)
        else:
            out["target"] = None
        return out
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return None


def _evaluate_level_window_candidate(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray | None,
    *,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    perceptual_weighting: bool,
    perceptual_strength: float,
    perceptual_min_hz: float,
    perceptual_max_hz: float,
    perceptual_tie_only: bool,
):
    try:
        f_w = np.asarray(freq_axis, dtype=float).reshape(-1)
        m_w = np.asarray(magnitudes, dtype=float).reshape(-1)
        if f_w.size < 20 or m_w.size != f_w.size:
            return None

        if target_mags is not None:
            t_w = np.asarray(target_mags, dtype=float).reshape(-1)
            if t_w.size != f_w.size:
                return None
            f_eval, (y, t_eval) = _resample_log_axis(f_w, m_w, t_w)
        else:
            f_eval, (y,) = _resample_log_axis(f_w, m_w)
            t_eval = None

        if f_eval.size < 20 or y.size < 20:
            return None

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
        weight = 1.0 + 0.05 * abs(np.log10(max(float(f_eval[0]), 1.0) / 1000.0))
        score = float(std * weight)

        target_rms = float("inf")
        offset_spread = float("inf")
        tilt_abs = float("inf")
        perceptual_rms = float("inf")
        try:
            if t_eval is not None and t_eval.size == y.size:
                offset_spread, target_rms, tilt_abs = _window_offset_consistency_score(
                    f_eval,
                    y,
                    t_eval,
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                )
                if perceptual_weighting:
                    perceptual_rms = _perceptual_shape_score(
                        f_eval,
                        y,
                        t_eval,
                        tilt_comp=bool(tilt_comp),
                        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                        min_hz=float(perceptual_min_hz),
                        max_hz=float(perceptual_max_hz),
                    )
            else:
                offset_spread, _shape_rms, tilt_abs = _window_offset_consistency_score(
                    f_eval,
                    y,
                    None,
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                )
        except (TypeError, ValueError, FloatingPointError, IndexError):
            target_rms = float("inf")
            offset_spread = float("inf")
            tilt_abs = float("inf")
            perceptual_rms = float("inf")

        if np.isfinite(offset_spread):
            score += 0.85 * float(offset_spread)
        if np.isfinite(target_rms):
            score += 0.20 * float(target_rms)
        if np.isfinite(tilt_abs):
            score += 0.08 * float(tilt_abs)
        if (not perceptual_tie_only) and np.isfinite(perceptual_rms):
            score += float(perceptual_strength) * float(perceptual_rms)

        return {
            "score": float(score),
            "target_rms": float(target_rms),
            "offset_spread": float(offset_spread),
            "tilt_abs": float(tilt_abs),
            "perceptual_rms": float(perceptual_rms),
        }
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return None


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
    perceptual_weighting: bool = False,
    perceptual_strength: float = 0.12,
    perceptual_min_hz: float = 250.0,
    perceptual_max_hz: float = 4000.0,
    perceptual_tie_only: bool = True,
) -> Tuple[float, float]:

    try:
        f_min = _to_float(f_min, 0.0)
        f_max = _to_float(f_max, 0.0)
        hpf_freq = _to_float(hpf_freq, 0.0)
        window_size_octaves = _to_float(window_size_octaves, 1.0)
        perceptual_weighting = _to_bool(perceptual_weighting, False)
        perceptual_strength = max(0.0, _to_float(perceptual_strength, 0.12))
        perceptual_min_hz = _to_float(perceptual_min_hz, 250.0)
        perceptual_max_hz = _to_float(perceptual_max_hz, 4000.0)
        perceptual_tie_only = _to_bool(perceptual_tie_only, True)

        if f_min <= 0 or f_max <= 0 or f_min >= f_max:
            return float(f_min), float(f_max)

        prepared = _prepare_level_window_search(
            freq_axis,
            magnitudes,
            target_mags,
            f_min=float(f_min),
            f_max=float(f_max),
            hpf_freq=float(hpf_freq),
        )
        if prepared is None:
            return float(f_min), float(f_max)
        f_search = np.asarray(prepared["freq"], dtype=float)
        m_search = np.asarray(prepared["magnitudes"], dtype=float)
        t_search = prepared.get("target", None)
        safe_f_min = float(prepared["safe_f_min"])

        best_score = float("inf")
        best_target_rms = float("inf")
        best_offset_spread = float("inf")
        best_tilt_abs = float("inf")
        best_perceptual_rms = float("inf")
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
                metrics = _evaluate_level_window_candidate(
                    f_search[w_mask],
                    m_search[w_mask],
                    None if t_search is None else np.asarray(t_search[w_mask], dtype=float),
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    perceptual_weighting=bool(perceptual_weighting),
                    perceptual_strength=float(perceptual_strength),
                    perceptual_min_hz=float(perceptual_min_hz),
                    perceptual_max_hz=float(perceptual_max_hz),
                    perceptual_tie_only=bool(perceptual_tie_only),
                )
                if metrics is None:
                    current_f *= step
                    continue

                score = float(metrics["score"])
                target_rms = float(metrics["target_rms"])
                offset_spread = float(metrics["offset_spread"])
                tilt_abs = float(metrics["tilt_abs"])
                perceptual_rms = float(metrics["perceptual_rms"])

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
                                and (
                                    (tilt_abs < best_tilt_abs)
                                    or (
                                        tilt_abs <= (best_tilt_abs + 1e-6)
                                        and perceptual_weighting
                                        and np.isfinite(perceptual_rms)
                                        and (perceptual_rms < best_perceptual_rms)
                                    )
                                )
                            )
                        )
                    )
                )

                if better_stability or better_tie_break:
                    best_score = score
                    best_target_rms = target_rms
                    best_offset_spread = offset_spread
                    best_tilt_abs = tilt_abs
                    best_perceptual_rms = perceptual_rms
                    res_min, res_max = float(w_start), float(w_end)

            current_f *= step

        if not np.isfinite(best_score):
            return float(f_min), float(f_max)

        return float(res_min), float(res_max)

    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float(f_min), float(f_max)


def find_shared_stereo_level_window(
    freq_axis_l: np.ndarray,
    magnitudes_l: np.ndarray,
    target_mags_l: np.ndarray,
    freq_axis_r: np.ndarray,
    magnitudes_r: np.ndarray,
    target_mags_r: np.ndarray,
    f_min: float,
    f_max: float,
    window_size_octaves: float = 1.0,
    hpf_freq: float = 0.0,
    tilt_comp: bool = True,
    tilt_max_db_per_oct: float = 2.0,
    perceptual_weighting: bool = False,
    perceptual_strength: float = 0.12,
    perceptual_min_hz: float = 250.0,
    perceptual_max_hz: float = 4000.0,
    perceptual_tie_only: bool = True,
) -> Tuple[float, float]:

    try:
        f_min = _to_float(f_min, 0.0)
        f_max = _to_float(f_max, 0.0)
        hpf_freq = _to_float(hpf_freq, 0.0)
        window_size_octaves = _to_float(window_size_octaves, 1.0)
        perceptual_weighting = _to_bool(perceptual_weighting, False)
        perceptual_strength = max(0.0, _to_float(perceptual_strength, 0.12))
        perceptual_min_hz = _to_float(perceptual_min_hz, 250.0)
        perceptual_max_hz = _to_float(perceptual_max_hz, 4000.0)
        perceptual_tie_only = _to_bool(perceptual_tie_only, True)

        if f_min <= 0 or f_max <= 0 or f_min >= f_max:
            return float(f_min), float(f_max)

        prep_l = _prepare_level_window_search(
            freq_axis_l,
            magnitudes_l,
            target_mags_l,
            f_min=float(f_min),
            f_max=float(f_max),
            hpf_freq=float(hpf_freq),
        )
        prep_r = _prepare_level_window_search(
            freq_axis_r,
            magnitudes_r,
            target_mags_r,
            f_min=float(f_min),
            f_max=float(f_max),
            hpf_freq=float(hpf_freq),
        )
        if prep_l is None or prep_r is None:
            return float(f_min), float(f_max)

        safe_f_min = max(float(prep_l["safe_f_min"]), float(prep_r["safe_f_min"]))
        safe_f_max = min(float(prep_l["f_max"]), float(prep_r["f_max"]))
        if safe_f_min <= 0.0 or safe_f_max <= safe_f_min:
            return float(f_min), float(f_max)

        f_l = np.asarray(prep_l["freq"], dtype=float)
        m_l = np.asarray(prep_l["magnitudes"], dtype=float)
        t_l = prep_l.get("target", None)
        f_r = np.asarray(prep_r["freq"], dtype=float)
        m_r = np.asarray(prep_r["magnitudes"], dtype=float)
        t_r = prep_r.get("target", None)

        best_primary = float("inf")
        best_secondary = float("inf")
        best_offset_spread = float("inf")
        best_target_rms = float("inf")
        best_tilt_abs = float("inf")
        best_perceptual_rms = float("inf")
        res_min, res_max = float(safe_f_min), float(safe_f_max)
        tie_eps_rel = 0.05

        current_f = float(safe_f_min)
        step = 2 ** (1 / 24.0)
        while current_f * (2 ** float(window_size_octaves)) <= float(safe_f_max):
            w_start = current_f
            w_end = current_f * (2 ** float(window_size_octaves))
            mask_l = (f_l >= w_start) & (f_l <= w_end)
            mask_r = (f_r >= w_start) & (f_r <= w_end)
            if int(np.count_nonzero(mask_l)) >= 20 and int(np.count_nonzero(mask_r)) >= 20:
                metrics_l = _evaluate_level_window_candidate(
                    f_l[mask_l],
                    m_l[mask_l],
                    None if t_l is None else np.asarray(t_l[mask_l], dtype=float),
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    perceptual_weighting=bool(perceptual_weighting),
                    perceptual_strength=float(perceptual_strength),
                    perceptual_min_hz=float(perceptual_min_hz),
                    perceptual_max_hz=float(perceptual_max_hz),
                    perceptual_tie_only=bool(perceptual_tie_only),
                )
                metrics_r = _evaluate_level_window_candidate(
                    f_r[mask_r],
                    m_r[mask_r],
                    None if t_r is None else np.asarray(t_r[mask_r], dtype=float),
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    perceptual_weighting=bool(perceptual_weighting),
                    perceptual_strength=float(perceptual_strength),
                    perceptual_min_hz=float(perceptual_min_hz),
                    perceptual_max_hz=float(perceptual_max_hz),
                    perceptual_tie_only=bool(perceptual_tie_only),
                )
                if metrics_l is not None and metrics_r is not None:
                    score_l = float(metrics_l["score"])
                    score_r = float(metrics_r["score"])
                    primary = max(score_l, score_r)
                    secondary = 0.5 * (score_l + score_r)
                    offset_spread = max(float(metrics_l["offset_spread"]), float(metrics_r["offset_spread"]))
                    target_rms = max(float(metrics_l["target_rms"]), float(metrics_r["target_rms"]))
                    tilt_abs = max(float(metrics_l["tilt_abs"]), float(metrics_r["tilt_abs"]))
                    perceptual_rms = max(float(metrics_l["perceptual_rms"]), float(metrics_r["perceptual_rms"]))

                    better_stability = primary < (best_primary * (1.0 - tie_eps_rel))
                    near_tie = primary <= (best_primary * (1.0 + tie_eps_rel))
                    better_tie_break = near_tie and (
                        (secondary < best_secondary)
                        or (
                            secondary <= (best_secondary + 1e-6)
                            and (
                                (offset_spread < best_offset_spread)
                                or (
                                    offset_spread <= (best_offset_spread + 1e-6)
                                    and (
                                        (target_rms < best_target_rms)
                                        or (
                                            target_rms <= (best_target_rms + 1e-6)
                                            and (
                                                (tilt_abs < best_tilt_abs)
                                                or (
                                                    tilt_abs <= (best_tilt_abs + 1e-6)
                                                    and perceptual_weighting
                                                    and np.isfinite(perceptual_rms)
                                                    and (perceptual_rms < best_perceptual_rms)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )

                    if better_stability or better_tie_break:
                        best_primary = float(primary)
                        best_secondary = float(secondary)
                        best_offset_spread = float(offset_spread)
                        best_target_rms = float(target_rms)
                        best_tilt_abs = float(tilt_abs)
                        best_perceptual_rms = float(perceptual_rms)
                        res_min, res_max = float(w_start), float(w_end)

            current_f *= step

        if not np.isfinite(best_primary):
            return float(f_min), float(f_max)

        return float(res_min), float(res_max)
    except (TypeError, ValueError, FloatingPointError, IndexError):
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
    perceptual_error_rms = None

    manual_target_db = _to_float(getattr(cfg, "lvl_manual_db", 0.0), 0.0)

    s_min = _to_float(getattr(cfg, "lvl_min", 500.0), 500.0)
    s_max = _to_float(getattr(cfg, "lvl_max", 2000.0), 2000.0)

    if s_min <= 0 or s_max <= 0 or s_min >= s_max:
        s_min, s_max = 500.0, 2000.0

    mode = str(getattr(cfg, "lvl_mode", "Auto"))
    is_manual = ("Manual" in mode)
    tilt_comp = bool(getattr(cfg, "lvl_tilt_comp", True))
    tilt_max_db_per_oct = _to_float(getattr(cfg, "lvl_tilt_max_db_per_oct", 2.0), 2.0)
    # Default behavior stays unchanged unless these opt-in perceptual flags are enabled.
    lvl_perceptual_weighting = _to_bool(getattr(cfg, "lvl_perceptual_weighting", False), False)
    lvl_perceptual_strength = max(
        0.0,
        _to_float(getattr(cfg, "lvl_perceptual_strength", 0.12), 0.12),
    )
    lvl_perceptual_min_hz = _to_float(getattr(cfg, "lvl_perceptual_min_hz", 250.0), 250.0)
    lvl_perceptual_max_hz = _to_float(getattr(cfg, "lvl_perceptual_max_hz", 4000.0), 4000.0)
    lvl_perceptual_tie_only = _to_bool(getattr(cfg, "lvl_perceptual_tie_only", True), True)
    if lvl_perceptual_min_hz <= 0.0 or lvl_perceptual_max_hz <= lvl_perceptual_min_hz:
        lvl_perceptual_min_hz, lvl_perceptual_max_hz = 250.0, 4000.0

    _safe_setattr(cfg, "_lvl_last_error", None)
    _safe_setattr(cfg, "_lvl_tilt_slope_db_per_oct", None)
    _safe_setattr(cfg, "_lvl_perceptual_enabled", bool(lvl_perceptual_weighting))
    _safe_setattr(cfg, "_lvl_perceptual_strength", float(lvl_perceptual_strength))
    _safe_setattr(
        cfg,
        "_lvl_perceptual_band_hz",
        (float(lvl_perceptual_min_hz), float(lvl_perceptual_max_hz)),
    )
    _safe_setattr(cfg, "_lvl_perceptual_error_rms", None)
    _safe_setattr(cfg, "_lvl_window_debug", None)

    def _store_window_debug(ss_min_value, ss_max_value, offset_method_value, perceptual_value):
        _safe_setattr(cfg, "_lvl_perceptual_error_rms", perceptual_value)
        try:
            tilt_slope_value = getattr(cfg, "_lvl_tilt_slope_db_per_oct", None)
        except (AttributeError, TypeError, ValueError):
            tilt_slope_value = None
        _safe_setattr(
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
                        _safe_setattr(cfg, "_lvl_tilt_slope_db_per_oct", float(_slope))
                except (TypeError, ValueError, FloatingPointError, IndexError):
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
                        _safe_setattr(cfg, "_lvl_tilt_slope_db_per_oct", float(tilt_slope))
                        offset_method = "ForcedWindowTiltMedian"
                    else:
                        calc_offset_db = _log_median(freq_axis[mask], diff)
                        offset_method = "ForcedWindowMedian"
                else:
                    calc_offset_db = 0.0
                    offset_method = "ForcedWindowNoMask"

            # Debug-only perceptual metric; main offset selection remains unchanged.
            if bool(lvl_perceptual_weighting) and np.count_nonzero(mask) >= 20 and target_arr is not None:
                perceptual_error_rms = _perceptual_shape_score(
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
            else:
                if shared_target_level_db is not None:
                    target_level_db = _to_float(shared_target_level_db, float(meas_level_db_window))
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

        if bool(lvl_perceptual_weighting) and np.count_nonzero(mask) >= 20 and target_arr is not None:
            perceptual_error_rms = _perceptual_shape_score(
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
            hpf_freq = _to_float(hpf_settings.get("freq", 0.0), 0.0)
        except (AttributeError, TypeError, ValueError) as exc:
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
        perceptual_weighting=bool(lvl_perceptual_weighting),
        perceptual_strength=float(lvl_perceptual_strength),
        perceptual_min_hz=float(lvl_perceptual_min_hz),
        perceptual_max_hz=float(lvl_perceptual_max_hz),
        perceptual_tie_only=bool(lvl_perceptual_tie_only),
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
            except (AttributeError, TypeError, ValueError):
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

    # Final perceptual debug metric uses only the selected window and never replaces offset math.
    if bool(lvl_perceptual_weighting) and np.count_nonzero(mask) >= 20 and target_arr is not None:
        perceptual_error_rms = _perceptual_shape_score(
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
    else:
        if shared_target_level_db is not None:
            target_level_db = _to_float(shared_target_level_db, float(meas_level_db_window))
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
