from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .leveling_compute import compute_leveling_impl
from .leveling_window import (
    find_shared_stereo_level_window_impl,
    find_stable_level_window_impl,
)

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
    return find_stable_level_window_impl(
        freq_axis,
        magnitudes,
        target_mags,
        f_min,
        f_max,
        window_size_octaves=window_size_octaves,
        hpf_freq=hpf_freq,
        tilt_comp=tilt_comp,
        tilt_max_db_per_oct=tilt_max_db_per_oct,
        perceptual_weighting=perceptual_weighting,
        perceptual_strength=perceptual_strength,
        perceptual_min_hz=perceptual_min_hz,
        perceptual_max_hz=perceptual_max_hz,
        perceptual_tie_only=perceptual_tie_only,
        to_float_fn=_to_float,
        to_bool_fn=_to_bool,
        resample_log_axis_fn=_resample_log_axis,
        lower_tail_robust_std_db_fn=_lower_tail_robust_std_db,
        window_offset_consistency_score_fn=_window_offset_consistency_score,
        perceptual_shape_score_fn=_perceptual_shape_score,
    )


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
    return find_shared_stereo_level_window_impl(
        freq_axis_l,
        magnitudes_l,
        target_mags_l,
        freq_axis_r,
        magnitudes_r,
        target_mags_r,
        f_min,
        f_max,
        window_size_octaves=window_size_octaves,
        hpf_freq=hpf_freq,
        tilt_comp=tilt_comp,
        tilt_max_db_per_oct=tilt_max_db_per_oct,
        perceptual_weighting=perceptual_weighting,
        perceptual_strength=perceptual_strength,
        perceptual_min_hz=perceptual_min_hz,
        perceptual_max_hz=perceptual_max_hz,
        perceptual_tie_only=perceptual_tie_only,
        to_float_fn=_to_float,
        to_bool_fn=_to_bool,
        resample_log_axis_fn=_resample_log_axis,
        lower_tail_robust_std_db_fn=_lower_tail_robust_std_db,
        window_offset_consistency_score_fn=_window_offset_consistency_score,
        perceptual_shape_score_fn=_perceptual_shape_score,
    )


def compute_leveling(
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    *,
    stereo_link_ctx: StereoLinkContext | None = None,
):
    return compute_leveling_impl(
        cfg,
        freq_axis,
        m_anal,
        target_mags,
        stereo_link_ctx=stereo_link_ctx,
        to_float_fn=_to_float,
        to_bool_fn=_to_bool,
        remember_leveling_error_fn=_remember_leveling_error,
        safe_setattr_fn=_safe_setattr,
        log_median_fn=_log_median,
        tilt_fit_offset_and_slope_db_per_oct_fn=_tilt_fit_offset_and_slope_db_per_oct,
        perceptual_shape_score_fn=_perceptual_shape_score,
        find_stable_level_window_fn=find_stable_level_window,
    )
