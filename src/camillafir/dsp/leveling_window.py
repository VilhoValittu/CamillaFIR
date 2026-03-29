from __future__ import annotations

from typing import Callable

import numpy as np


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

        return {
            "freq": freq_arr[mask],
            "magnitudes": mag_arr[mask],
            "safe_f_min": float(safe_f_min),
            "f_max": float(f_max),
            "target": None if target_search is None else np.asarray(target_search[mask], dtype=float),
        }
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
    resample_log_axis_fn: Callable[..., tuple[np.ndarray, tuple[np.ndarray, ...]]],
    lower_tail_robust_std_db_fn: Callable[..., float],
    window_offset_consistency_score_fn: Callable[..., tuple[float, float, float]],
    perceptual_shape_score_fn: Callable[..., float],
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
            f_eval, (y, t_eval) = resample_log_axis_fn(f_w, m_w, t_w)
        else:
            f_eval, (y,) = resample_log_axis_fn(f_w, m_w)
            t_eval = None

        if f_eval.size < 20 or y.size < 20:
            return None

        channel_offset = float(np.median(y - t_eval)) if (t_eval is not None and t_eval.size == y.size) else float(np.median(y))

        x = np.log2(np.clip(f_eval, 1e-9, None))
        x0 = float(np.median(x))
        xc = x - x0
        y_med = float(np.median(y))
        denom = float(np.dot(xc, xc))
        residual = y - (float(np.dot(xc, (y - y_med)) / denom) * xc) if denom > 1e-12 else y

        std = lower_tail_robust_std_db_fn(residual, clip_below_db=6.0)
        f_center = float(np.sqrt(max(float(f_eval[0]), 1e-9) * max(float(f_eval[-1]), 1e-9)))
        weight = 1.0 + 0.05 * abs(np.log10(max(f_center, 1.0) / 1000.0))
        score = float(std * weight)

        target_rms = float("inf")
        offset_spread = float("inf")
        tilt_abs = float("inf")
        perceptual_rms = float("inf")
        try:
            if t_eval is not None and t_eval.size == y.size:
                offset_spread, target_rms, tilt_abs = window_offset_consistency_score_fn(
                    f_eval,
                    y,
                    t_eval,
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                )
                if perceptual_weighting:
                    perceptual_rms = perceptual_shape_score_fn(
                        f_eval,
                        y,
                        t_eval,
                        tilt_comp=bool(tilt_comp),
                        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                        min_hz=float(perceptual_min_hz),
                        max_hz=float(perceptual_max_hz),
                    )
            else:
                offset_spread, _shape_rms, tilt_abs = window_offset_consistency_score_fn(
                    f_eval,
                    y,
                    None,
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                )
        except (TypeError, ValueError, FloatingPointError, IndexError):
            pass

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
            "channel_offset": float(channel_offset),
        }
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return None


def find_stable_level_window_impl(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray,
    f_min: float,
    f_max: float,
    *,
    window_size_octaves: float,
    hpf_freq: float,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    perceptual_weighting: bool,
    perceptual_strength: float,
    perceptual_min_hz: float,
    perceptual_max_hz: float,
    perceptual_tie_only: bool,
    to_float_fn: Callable[[object, float], float],
    to_bool_fn: Callable[[object, bool], bool],
    resample_log_axis_fn: Callable[..., tuple[np.ndarray, tuple[np.ndarray, ...]]],
    lower_tail_robust_std_db_fn: Callable[..., float],
    window_offset_consistency_score_fn: Callable[..., tuple[float, float, float]],
    perceptual_shape_score_fn: Callable[..., float],
):
    try:
        f_min = to_float_fn(f_min, 0.0)
        f_max = to_float_fn(f_max, 0.0)
        hpf_freq = to_float_fn(hpf_freq, 0.0)
        window_size_octaves = to_float_fn(window_size_octaves, 1.0)
        perceptual_weighting = to_bool_fn(perceptual_weighting, False)
        perceptual_strength = max(0.0, to_float_fn(perceptual_strength, 0.12))
        perceptual_min_hz = to_float_fn(perceptual_min_hz, 250.0)
        perceptual_max_hz = to_float_fn(perceptual_max_hz, 4000.0)
        perceptual_tie_only = to_bool_fn(perceptual_tie_only, True)
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
        step = 2 ** (1 / 48.0)

        while current_f * (2 ** float(window_size_octaves)) <= float(f_max):
            w_start = current_f
            w_end = current_f * (2 ** float(window_size_octaves))
            w_mask = (f_search >= w_start) & (f_search <= w_end)
            if int(np.count_nonzero(w_mask)) >= 20:
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
                    resample_log_axis_fn=resample_log_axis_fn,
                    lower_tail_robust_std_db_fn=lower_tail_robust_std_db_fn,
                    window_offset_consistency_score_fn=window_offset_consistency_score_fn,
                    perceptual_shape_score_fn=perceptual_shape_score_fn,
                )
                if metrics is not None:
                    score = float(metrics["score"])
                    target_rms = float(metrics["target_rms"])
                    offset_spread = float(metrics["offset_spread"])
                    tilt_abs = float(metrics["tilt_abs"])
                    perceptual_rms = float(metrics["perceptual_rms"])
                    better_stability = score < (best_score * (1.0 - tie_eps_rel))
                    near_tie = score <= (best_score * (1.0 + tie_eps_rel))
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

        return (float(f_min), float(f_max)) if not np.isfinite(best_score) else (float(res_min), float(res_max))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float(f_min), float(f_max)


def find_shared_stereo_level_window_impl(
    freq_axis_l: np.ndarray,
    magnitudes_l: np.ndarray,
    target_mags_l: np.ndarray,
    freq_axis_r: np.ndarray,
    magnitudes_r: np.ndarray,
    target_mags_r: np.ndarray,
    f_min: float,
    f_max: float,
    *,
    window_size_octaves: float,
    hpf_freq: float,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    perceptual_weighting: bool,
    perceptual_strength: float,
    perceptual_min_hz: float,
    perceptual_max_hz: float,
    perceptual_tie_only: bool,
    to_float_fn: Callable[[object, float], float],
    to_bool_fn: Callable[[object, bool], bool],
    resample_log_axis_fn: Callable[..., tuple[np.ndarray, tuple[np.ndarray, ...]]],
    lower_tail_robust_std_db_fn: Callable[..., float],
    window_offset_consistency_score_fn: Callable[..., tuple[float, float, float]],
    perceptual_shape_score_fn: Callable[..., float],
):
    try:
        f_min = to_float_fn(f_min, 0.0)
        f_max = to_float_fn(f_max, 0.0)
        hpf_freq = to_float_fn(hpf_freq, 0.0)
        window_size_octaves = to_float_fn(window_size_octaves, 1.0)
        perceptual_weighting = to_bool_fn(perceptual_weighting, False)
        perceptual_strength = max(0.0, to_float_fn(perceptual_strength, 0.12))
        perceptual_min_hz = to_float_fn(perceptual_min_hz, 250.0)
        perceptual_max_hz = to_float_fn(perceptual_max_hz, 4000.0)
        perceptual_tie_only = to_bool_fn(perceptual_tie_only, True)
        if f_min <= 0 or f_max <= 0 or f_min >= f_max:
            return float(f_min), float(f_max)

        prep_l = _prepare_level_window_search(freq_axis_l, magnitudes_l, target_mags_l, f_min=float(f_min), f_max=float(f_max), hpf_freq=float(hpf_freq))
        prep_r = _prepare_level_window_search(freq_axis_r, magnitudes_r, target_mags_r, f_min=float(f_min), f_max=float(f_max), hpf_freq=float(hpf_freq))
        if prep_l is None or prep_r is None:
            return float(f_min), float(f_max)

        safe_f_min = max(float(prep_l["safe_f_min"]), float(prep_r["safe_f_min"]))
        if safe_f_min <= 0.0 or safe_f_min >= float(f_max):
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
        res_min, res_max = float(safe_f_min), float(f_max)
        tie_eps_rel = 0.05
        current_f = float(safe_f_min)
        step = 2 ** (1 / 48.0)

        while current_f * (2 ** float(window_size_octaves)) <= float(f_max):
            w_start = current_f
            w_end = current_f * (2 ** float(window_size_octaves))
            mask_l = (f_l >= w_start) & (f_l <= w_end)
            mask_r = (f_r >= w_start) & (f_r <= w_end)
            if int(np.count_nonzero(mask_l)) >= 20 and int(np.count_nonzero(mask_r)) >= 20:
                metrics_l = _evaluate_level_window_candidate(
                    f_l[mask_l], m_l[mask_l], None if t_l is None else np.asarray(t_l[mask_l], dtype=float),
                    tilt_comp=bool(tilt_comp), tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    perceptual_weighting=bool(perceptual_weighting), perceptual_strength=float(perceptual_strength),
                    perceptual_min_hz=float(perceptual_min_hz), perceptual_max_hz=float(perceptual_max_hz),
                    perceptual_tie_only=bool(perceptual_tie_only), resample_log_axis_fn=resample_log_axis_fn,
                    lower_tail_robust_std_db_fn=lower_tail_robust_std_db_fn,
                    window_offset_consistency_score_fn=window_offset_consistency_score_fn,
                    perceptual_shape_score_fn=perceptual_shape_score_fn,
                )
                metrics_r = _evaluate_level_window_candidate(
                    f_r[mask_r], m_r[mask_r], None if t_r is None else np.asarray(t_r[mask_r], dtype=float),
                    tilt_comp=bool(tilt_comp), tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    perceptual_weighting=bool(perceptual_weighting), perceptual_strength=float(perceptual_strength),
                    perceptual_min_hz=float(perceptual_min_hz), perceptual_max_hz=float(perceptual_max_hz),
                    perceptual_tie_only=bool(perceptual_tie_only), resample_log_axis_fn=resample_log_axis_fn,
                    lower_tail_robust_std_db_fn=lower_tail_robust_std_db_fn,
                    window_offset_consistency_score_fn=window_offset_consistency_score_fn,
                    perceptual_shape_score_fn=perceptual_shape_score_fn,
                )
                if metrics_l is not None and metrics_r is not None:
                    score_l = float(metrics_l["score"])
                    score_r = float(metrics_r["score"])
                    offset_diff = abs(float(metrics_l["channel_offset"]) - float(metrics_r["channel_offset"]))
                    primary = max(score_l, score_r) + 0.25 * offset_diff
                    secondary = 0.5 * (score_l + score_r)
                    offset_spread = max(float(metrics_l["offset_spread"]), float(metrics_r["offset_spread"]))
                    target_rms = max(float(metrics_l["target_rms"]), float(metrics_r["target_rms"]))
                    tilt_abs = max(float(metrics_l["tilt_abs"]), float(metrics_r["tilt_abs"]))
                    perceptual_rms = max(float(metrics_l["perceptual_rms"]), float(metrics_r["perceptual_rms"]))
                    better_stability = primary < (best_primary * (1.0 - tie_eps_rel))
                    near_tie = primary <= (best_primary * (1.0 + tie_eps_rel))
                    better_tie_break = near_tie and (
                        (offset_spread < best_offset_spread)
                        or (
                            offset_spread <= (best_offset_spread + 1e-6)
                            and (
                                (secondary < best_secondary)
                                or (
                                    secondary <= (best_secondary + 1e-6)
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

        return (float(f_min), float(f_max)) if not np.isfinite(best_primary) else (float(res_min), float(res_max))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float(f_min), float(f_max)
