from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.ndimage

from . import bassfirst as bf
from .camillafir_analysis import _sigma_bins_from_hz
from .correction_types import (
    _MagAdaptiveStageOutputs,
    _MagCoreOutputs,
    _MagCorrectionContext,
    _MagPipelineInputs,
    _MagPostProcessInputs,
    _MagPostProcessOutputs,
    _MagRawStageOutputs,
)
from .dsp_config import CfgReader
from .dsp_telemetry import safe_put_many
from .gain_policy import apply_cuts_only_guard, build_low_frequency_guard_mask, resolve_gain_policy
from .mag_limits import (
    _apply_hard_boost_cut_clamp,
    _apply_max_boost_cut,
    _apply_slope_limits,
    _blend_masked_fractional_octave,
)
from .mag_post_limits import apply_post_limits_and_metrics as _apply_post_limits_and_metrics_impl
from .mag_postprocess import apply_bass_boost_post_restore, apply_confpull_post_slope
from .mag_shape import (
    _apply_confidence_logic,
    _apply_regularization,
    _compute_error_db,
    _error_to_correction_mag,
    _resolve_filter_smooth,
    _select_active_band,
)
from .mag_stage import (
    run_mag_bassfirst_afdw_conf_stage,
    run_mag_core_stage,
    run_mag_raw_stage,
)
from .mag_telemetry import (
    _log_stage_stats,
    _record_stage_probe,
    _summarize_correction_metrics,
)
from .phase_ir_utils import _cosine_fade_out_01
from .smoothing import (
    AFDW_BW_MAX_OCT,
    AFDW_BW_MIN_OCT,
    apply_adaptive_fdw,
    psycho_smooth_safe_gain,
    smooth_gain_fractional_octave,
)

def _apply_peak_priority_error_shaping(
    err_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    *,
    mask_c: np.ndarray,
    logger: Any,
) -> np.ndarray:
    """
   Peak-priority error formulation:
      - Keep CUT requests (negative error) intact.
      - Soft-limit BOOST requests (positive error) to ~max_boost_db (tanh saturation),
        so deep dips don't dominate the solution when max_boost is small anyway.
    """
    cfg_reader = CfgReader(cfg)
    enable = cfg_reader.bool("peak_priority_enable", True)
    if not enable:
        return err_db

    strength = cfg_reader.float_allow_zero("peak_priority_strength", 0.5) #vaikuttaa boostin voimakkuuteen, mitä isompi, sitä miedompi boost
    if not np.isfinite(strength):
        strength = 0.0
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return err_db

    max_boost = cfg_reader.float_allow_zero("max_boost_db", 0.0)
    if not np.isfinite(max_boost) or max_boost <= 0.0:
        return err_db

    e = np.asarray(err_db, dtype=float).copy()
    m = np.asarray(mask_c, dtype=bool)
    pos = m & (e > 0.0)
    if not np.any(pos):
        return e

    # Soft saturation for positive error: cap * tanh(err/cap)
    cap = float(max_boost)
    e_pos = e[pos]
    e_sat = cap * np.tanh(e_pos / (cap + 1e-12))
    e[pos] = (1.0 - strength) * e_pos + strength * e_sat

    # Optional (small) cut emphasis if you want even more peak-first behavior.
    cut_emph = cfg_reader.float_allow_zero("peak_priority_cut_emphasis", 0.0)
    if np.isfinite(cut_emph) and cut_emph > 0.0:
        cut_emph = float(np.clip(cut_emph, 0.0, 2.0))
        neg = m & (e < 0.0)
        if np.any(neg):
            e[neg] *= (1.0 + cut_emph * strength)

    # Stats (safe)
    try:
        if isinstance(st, dict):
            st["peak_priority_enable"] = True
            st["peak_priority_strength"] = float(strength)
            st["peak_priority_max_boost_db"] = float(max_boost)
            st["peak_priority_pos_err_peak_before_db"] = float(np.max(np.asarray(err_db, dtype=float)[pos]))
            st["peak_priority_pos_err_peak_after_db"] = float(np.max(e[pos]))
    except (TypeError, ValueError):
        pass

    try:
        logger.info(
            "PeakPriority: "
            f"enable=ON, strength={strength:.2f}, max_boost={max_boost:.2f} dB, "
            f"pos_err_peak: {float(np.max(np.asarray(err_db, dtype=float)[pos])):.2f} -> {float(np.max(e[pos])):.2f} dB"
        )
    except (AttributeError, TypeError, ValueError):
        pass

    return e


def _apply_smoothing(
    err_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    filter_smooth: float,
    df_mode: bool,
    conf_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Soveltaa nykyisen valinnan mukaista smoothingia (DF tai fractional octave)."""
    cfg_reader = CfgReader(cfg)
    if df_mode:
        base_sigma = 60 // (filter_smooth / 12 if filter_smooth > 0 else 1)
        df_ref = 44100.0 / 65536.0
        sigma_hz = float(base_sigma) * df_ref
        sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=sigma_hz, fallback_bins=max(2.0, float(base_sigma)))
        smoothed_base = scipy.ndimage.gaussian_filter1d(err_db, sigma=float(sigma_bins))
    else:
        smoothed_base = smooth_gain_fractional_octave(freq_axis, err_db, filter_smooth)

    if not cfg_reader.bool("bass_smooth_adaptive", False):
        try:
            if isinstance(st, dict):
                st["bass_adaptive_smoothing_enabled"] = False
                st["bass_adaptive_smoothing_avg_w_20_200"] = 0.0
        except (TypeError, ValueError):
            pass
        return smoothed_base

    try:
        conf_arr = np.asarray(conf_mask, dtype=float)
    except (TypeError, ValueError):
        conf_arr = None
    if conf_arr is None or conf_arr.shape != np.asarray(freq_axis, dtype=float).shape:
        try:
            if isinstance(st, dict):
                st["bass_adaptive_smoothing_enabled"] = False
                st["bass_adaptive_smoothing_avg_w_20_200"] = 0.0
        except (TypeError, ValueError):
            pass
        return smoothed_base

    f_bass = cfg_reader.float("bass_smooth_hz", 200.0)
    if not np.isfinite(f_bass) or f_bass <= 0.0:
        f_bass = 200.0

    conf_floor = cfg_reader.float("bass_smooth_conf_floor", 0.3)
    if not np.isfinite(conf_floor) or conf_floor <= 0.0:
        conf_floor = 0.3

    sigma_scale = cfg_reader.float("bass_smooth_sigma_scale", 1.4)
    if (not np.isfinite(sigma_scale)) or sigma_scale < 1.0:
        sigma_scale = 1.0
    w_gamma = cfg_reader.float("bass_smooth_w_gamma", 2.0)
    if (not np.isfinite(w_gamma)) or w_gamma <= 0.0:
        w_gamma = 2.0
    w_max = cfg_reader.float("bass_smooth_w_max", 0.55)
    if not np.isfinite(w_max):
        w_max = 0.55
    w_max = float(np.clip(w_max, 0.0, 1.0))
    try:
        if isinstance(st, dict):
            st["bass_adaptive_smoothing_sigma_scale"] = float(sigma_scale)
            st["bass_adaptive_smoothing_conf_floor"] = float(conf_floor)
            st["bass_adaptive_smoothing_w_gamma"] = float(w_gamma)
            st["bass_adaptive_smoothing_w_max"] = float(w_max)
    except (TypeError, ValueError):
        pass

    f = np.asarray(freq_axis, dtype=float)
    c = np.asarray(conf_arr, dtype=float)
    c = np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=0.0)
    c = np.clip(c, 0.0, 1.0)
    w = np.clip((conf_floor - c) / max(conf_floor, 1e-9), 0.0, 1.0)
    w = np.power(w, float(w_gamma))
    w = np.minimum(w, float(w_max))
    bass = (f > 0.0) & (f <= float(f_bass))
    if not np.any(bass):
        try:
            if isinstance(st, dict):
                st["bass_adaptive_smoothing_enabled"] = False
                st["bass_adaptive_smoothing_avg_w_20_200"] = 0.0
        except (TypeError, ValueError):
            pass
        return smoothed_base

    if df_mode:
        sigma_bins2 = float(max(1.0, float(sigma_bins) * float(sigma_scale)))
        smoothed_more = scipy.ndimage.gaussian_filter1d(err_db, sigma=sigma_bins2)
    else:
        smooth2 = float(max(1.0, float(filter_smooth) / float(sigma_scale)))
        smoothed_more = smooth_gain_fractional_octave(freq_axis, err_db, smooth2)

    out = np.asarray(smoothed_base, dtype=float).copy()
    # Explicitly touch bass-only bins; keep >bass range identical.
    w_eff = np.zeros_like(w, dtype=float)
    w_eff[bass] = np.asarray(w[bass], dtype=float)
    out[bass] = (out[bass] * (1.0 - w_eff[bass])) + (np.asarray(smoothed_more, dtype=float)[bass] * w_eff[bass])

    try:
        if isinstance(st, dict):
            st["bass_adaptive_smoothing_enabled"] = True
            b20 = (f >= 20.0) & (f <= 200.0)
            st["bass_adaptive_smoothing_avg_w_20_200"] = float(np.mean(w_eff[b20])) if np.any(b20) else 0.0
    except (TypeError, ValueError):
        pass
    return out


def _apply_confidence_adaptive_bass_smoothing(
    curve_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    conf_mask: np.ndarray | None,
    *,
    stage_tag: str = "core",
) -> np.ndarray:
    """Lisasmoottaa bassoa confidence-maskin mukaan myohaisessa vaiheessa."""
    base = np.asarray(curve_db, dtype=float)
    f = np.asarray(freq_axis, dtype=float)
    stage_key = str(stage_tag or "core").strip().lower()
    cfg_reader = CfgReader(cfg)

    def _mark_disabled() -> np.ndarray:
        try:
            if isinstance(st, dict):
                safe_put_many(
                    st,
                    {
                        "bass_adaptive_smoothing_enabled": False,
                        "bass_adaptive_smoothing_avg_w_20_200": 0.0,
                        "bass_adaptive_smoothing_delta_rms_db_20_200": 0.0,
                        "bass_adaptive_smoothing_delta_max_db_20_200": 0.0,
                        "bass_adaptive_smoothing_delta_max_hz_20_200": None,
                        "bass_adaptive_smoothing_delta_rms_db_20_200_stage_local": 0.0,
                        "bass_adaptive_smoothing_delta_max_db_20_200_stage_local": 0.0,
                        "bass_adaptive_smoothing_delta_max_hz_20_200_stage_local": None,
                        f"bass_adaptive_smoothing_{stage_key}_enabled": False,
                        f"bass_adaptive_smoothing_{stage_key}_avg_w_20_200": 0.0,
                        f"bass_adaptive_smoothing_{stage_key}_delta_rms_db_20_200": 0.0,
                        f"bass_adaptive_smoothing_{stage_key}_delta_max_db_20_200": 0.0,
                        f"bass_adaptive_smoothing_{stage_key}_delta_max_hz_20_200": None,
                    },
                )
        except (TypeError, ValueError):
            pass
        return base

    if not cfg_reader.bool("bass_smooth_adaptive", False):
        return _mark_disabled()

    try:
        c = np.asarray(conf_mask, dtype=float)
    except (TypeError, ValueError):
        c = None
    if c is None or c.shape != f.shape:
        return _mark_disabled()

    f_bass = cfg_reader.float("bass_smooth_hz", 200.0)
    if not np.isfinite(f_bass) or f_bass <= 0.0:
        f_bass = 200.0

    conf_floor = cfg_reader.float("bass_smooth_conf_floor", 0.3)
    if not np.isfinite(conf_floor) or conf_floor <= 0.0:
        conf_floor = 0.3

    sigma_scale = cfg_reader.float("bass_smooth_sigma_scale", 1.4)
    if (not np.isfinite(sigma_scale)) or sigma_scale < 1.0:
        sigma_scale = 1.0
    w_gamma = cfg_reader.float("bass_smooth_w_gamma", 2.0)
    if (not np.isfinite(w_gamma)) or w_gamma <= 0.0:
        w_gamma = 2.0
    w_max = cfg_reader.float("bass_smooth_w_max", 0.55)
    if not np.isfinite(w_max):
        w_max = 0.55
    w_max = float(np.clip(w_max, 0.0, 1.0))
    try:
        if isinstance(st, dict):
            st["bass_adaptive_smoothing_sigma_scale"] = float(sigma_scale)
            st["bass_adaptive_smoothing_conf_floor"] = float(conf_floor)
            st["bass_adaptive_smoothing_w_gamma"] = float(w_gamma)
            st["bass_adaptive_smoothing_w_max"] = float(w_max)
    except (TypeError, ValueError):
        pass

    c = np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=0.0)
    c = np.clip(c, 0.0, 1.0)
    w = np.clip((conf_floor - c) / max(conf_floor, 1e-9), 0.0, 1.0)
    w = np.power(w, float(w_gamma))
    w = np.minimum(w, float(w_max))
    w = np.clip(w, 0.0, 1.0)

    bass = (f > 0.0) & (f <= float(f_bass))
    if not np.any(bass):
        return _mark_disabled()

    sigma_hz = max(3.0, float(f_bass) / 24.0) * float(sigma_scale)
    sigma_bins = _sigma_bins_from_hz(
        f,
        sigma_hz=float(sigma_hz),
        fallback_bins=max(3.0, 6.0 * float(sigma_scale)),
    )
    sigma_bins = float(max(1.0, sigma_bins))
    sm_more = scipy.ndimage.gaussian_filter1d(base, sigma=sigma_bins)

    out = base.copy()
    # Explicit bass-only blend; keep out-of-band untouched.
    w_eff = np.zeros_like(w, dtype=float)
    w_eff[bass] = np.asarray(w[bass], dtype=float)
    out[bass] = (out[bass] * (1.0 - w_eff[bass])) + (np.asarray(sm_more, dtype=float)[bass] * w_eff[bass])

    b20 = (f >= 20.0) & (f <= 200.0)
    delta = out - base
    d_hz = None
    if np.any(b20):
        d_rms = float(np.sqrt(np.mean(delta[b20] * delta[b20])))
        ad = np.abs(delta[b20])
        d_max = float(np.max(ad))
        fb = np.asarray(f[b20], dtype=float)
        if fb.size == ad.size and ad.size:
            d_hz = float(fb[int(np.argmax(ad))])
        w_avg = float(np.mean(w_eff[b20]))
    else:
        d_rms, d_max, w_avg = 0.0, 0.0, 0.0

    # If effect is numerically tiny, force a stronger fallback pass.
    if d_rms < 5e-4 and w_avg > 0.10:
        sigma_bins2 = float(max(1.0, sigma_bins * 2.0))
        sm_more2 = scipy.ndimage.gaussian_filter1d(base, sigma=sigma_bins2)
        out2 = base.copy()
        out2[bass] = (out2[bass] * (1.0 - w_eff[bass])) + (np.asarray(sm_more2, dtype=float)[bass] * w_eff[bass])
        delta2 = out2 - base
        if np.any(b20):
            d_rms2 = float(np.sqrt(np.mean(delta2[b20] * delta2[b20])))
            ad2 = np.abs(delta2[b20])
            d_max2 = float(np.max(ad2))
            fb2 = np.asarray(f[b20], dtype=float)
            d_hz2 = float(fb2[int(np.argmax(ad2))]) if (fb2.size == ad2.size and ad2.size) else None
        else:
            d_rms2, d_max2 = 0.0, 0.0
            d_hz2 = None
        if d_rms2 > d_rms:
            out, d_rms, d_max, d_hz = out2, d_rms2, d_max2, d_hz2
            sigma_bins = sigma_bins2

    try:
        if isinstance(st, dict):
            safe_put_many(
                st,
                {
                    "bass_adaptive_smoothing_enabled": True,
                    "bass_adaptive_smoothing_avg_w_20_200": float(w_avg),
                    "bass_adaptive_smoothing_delta_rms_db_20_200": float(d_rms),
                    "bass_adaptive_smoothing_delta_max_db_20_200": float(d_max),
                    "bass_adaptive_smoothing_delta_max_hz_20_200": float(d_hz) if d_hz is not None else None,
                    "bass_adaptive_smoothing_delta_rms_db_20_200_stage_local": float(d_rms),
                    "bass_adaptive_smoothing_delta_max_db_20_200_stage_local": float(d_max),
                    "bass_adaptive_smoothing_delta_max_hz_20_200_stage_local": float(d_hz) if d_hz is not None else None,
                    "bass_adaptive_smoothing_sigma_bins": float(sigma_bins),
                    f"bass_adaptive_smoothing_{stage_key}_enabled": True,
                    f"bass_adaptive_smoothing_{stage_key}_avg_w_20_200": float(w_avg),
                    f"bass_adaptive_smoothing_{stage_key}_delta_rms_db_20_200": float(d_rms),
                    f"bass_adaptive_smoothing_{stage_key}_delta_max_db_20_200": float(d_max),
                    f"bass_adaptive_smoothing_{stage_key}_delta_max_hz_20_200": float(d_hz) if d_hz is not None else None,
                },
            )
    except (TypeError, ValueError):
        pass
    return out


def _select_bass_adaptive_conf_mask(
    conf_mask: np.ndarray | None,
    bf_conf_for_smoothing: np.ndarray | None,
    *,
    use_bassfirst: bool,
) -> tuple[np.ndarray | None, str]:
    """Valitsee bass-adaptive smoothingille conf-maskin ilman bassfirst-floor lockia."""
    try:
        c_raw = np.asarray(conf_mask, dtype=float) if conf_mask is not None else None
    except (TypeError, ValueError):
        c_raw = None
    try:
        c_bf = np.asarray(bf_conf_for_smoothing, dtype=float) if bf_conf_for_smoothing is not None else None
    except (TypeError, ValueError):
        c_bf = None

    if c_raw is not None:
        c_raw = np.clip(np.nan_to_num(c_raw, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return c_raw, "raw_conf"

    if bool(use_bassfirst) and c_bf is not None:
        c_bf = np.clip(np.nan_to_num(c_bf, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return c_bf, "bassfirst_fallback"

    if c_bf is not None:
        c_bf = np.clip(np.nan_to_num(c_bf, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return c_bf, "bassfirst_fallback"

    return None, "missing"


def _apply_mid_refit_pre_slope(
    gain_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    *,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    calc_offset_db: float,
    conf_mask: np.ndarray | None,
    cfg: Any,
    st: Any,
    logger: Any,
) -> np.ndarray:
    """Kevyt mid-band match-pass ennen slope-rajoja."""
    out = np.asarray(gain_db, dtype=float).copy()
    cfg_reader = CfgReader(cfg)

    def _write_stats(
        *,
        enabled: bool,
        k_used: float,
        before_rms: float | None,
        after_rms: float | None,
        delta_rms: float | None,
        conf_avg: float | None,
        reason: str,
    ) -> None:
        try:
            if isinstance(st, dict):
                before_val = float(before_rms) if before_rms is not None and np.isfinite(float(before_rms)) else None
                after_val = float(after_rms) if after_rms is not None and np.isfinite(float(after_rms)) else None
                delta_val = float(delta_rms) if delta_rms is not None and np.isfinite(float(delta_rms)) else None
                conf_val = float(conf_avg) if conf_avg is not None and np.isfinite(float(conf_avg)) else None
                safe_put_many(
                    st,
                    {
                        "mid_refit_enabled": bool(enabled),
                        "mid_refit_k": float(k_used),
                        "mid_refit_err_rms_before": before_val,
                        "mid_refit_err_rms_after": after_val,
                        "mid_refit_delta_rms": delta_val,
                        "mid_refit_err_rms_before_stage_local": before_val,
                        "mid_refit_err_rms_after_stage_local": after_val,
                        "mid_refit_delta_rms_stage_local": delta_val,
                        "mid_refit_conf_avg_200_2000": conf_val,
                        "mid_refit_reason": str(reason),
                    },
                )
        except (TypeError, ValueError):
            pass

    if not cfg_reader.bool("enable_mag_correction", False):
        _write_stats(
            enabled=False,
            k_used=0.0,
            before_rms=None,
            after_rms=None,
            delta_rms=None,
            conf_avg=None,
            reason="mag_off",
        )
        return out

    refit_on = cfg_reader.bool("mid_refit_enable", True)
    if not refit_on:
        _write_stats(
            enabled=False,
            k_used=0.0,
            before_rms=None,
            after_rms=None,
            delta_rms=None,
            conf_avg=None,
            reason="disabled",
        )
        return out

    mid_lo = cfg_reader.float("mid_refit_hz_lo", 200.0)
    mid_hi = cfg_reader.float("mid_refit_hz_hi", 2000.0)
    if (not np.isfinite(mid_lo)) or mid_lo < 1.0:
        mid_lo = 200.0
    if (not np.isfinite(mid_hi)) or mid_hi <= (mid_lo + 1.0):
        mid_hi = 2000.0
        if mid_hi <= (mid_lo + 1.0):
            mid_hi = mid_lo + 1.0

    k_refit = cfg_reader.float("mid_refit_k", 0.45)
    if not np.isfinite(k_refit):
        k_refit = 0.45
    k_refit = float(np.clip(k_refit, 0.0, 1.0))
    if k_refit <= 0.0:
        _write_stats(
            enabled=False,
            k_used=0.0,
            before_rms=None,
            after_rms=None,
            delta_rms=None,
            conf_avg=None,
            reason="k<=0",
        )
        return out

    smooth_oct = cfg_reader.float("mid_refit_smooth_oct", 0.6)
    if (not np.isfinite(smooth_oct)) or smooth_oct <= 0.0:
        smooth_oct = 0.6
    smooth_oct = float(np.clip(smooth_oct, 1.0 / 192.0, 1.0))
    smooth_value = float(np.clip(1.0 / smooth_oct, 1.0, 192.0))

    conf_min_avg = cfg_reader.float("mid_refit_conf_min_avg", 0.2)
    if not np.isfinite(conf_min_avg):
        conf_min_avg = 0.2
    conf_min_avg = float(np.clip(conf_min_avg, 0.0, 1.0))

    f = np.asarray(freq_axis, dtype=float).reshape(-1)
    g = np.asarray(out, dtype=float).reshape(-1)
    m = np.asarray(mask_c, dtype=bool).reshape(-1)
    meas = np.asarray(m_anal, dtype=float).reshape(-1)
    tgt = np.asarray(target_mags, dtype=float).reshape(-1)
    n = int(min(f.size, g.size, m.size, meas.size, tgt.size))
    if n < 32:
        _write_stats(
            enabled=False,
            k_used=0.0,
            before_rms=None,
            after_rms=None,
            delta_rms=None,
            conf_avg=None,
            reason="insufficient_data",
        )
        return out

    f = f[:n]
    g = g[:n]
    m = m[:n]
    meas = meas[:n]
    tgt = tgt[:n]
    valid = np.isfinite(f) & np.isfinite(g) & np.isfinite(m) & np.isfinite(meas) & np.isfinite(tgt)
    mid = valid & m & (f >= float(mid_lo)) & (f <= float(mid_hi))
    if int(np.count_nonzero(mid)) < 8:
        _write_stats(
            enabled=False,
            k_used=0.0,
            before_rms=None,
            after_rms=None,
            delta_rms=None,
            conf_avg=None,
            reason="no_mid_bins",
        )
        return out

    conf_avg = None
    try:
        if conf_mask is not None:
            c = np.asarray(conf_mask, dtype=float).reshape(-1)
            if c.size >= n:
                c = np.clip(np.nan_to_num(c[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
                conf_avg = float(np.mean(c[mid])) if np.any(mid) else None
    except (TypeError, ValueError, FloatingPointError, IndexError):
        conf_avg = None
    if conf_avg is not None and conf_avg < float(conf_min_avg):
        err_before = tgt - ((meas - float(calc_offset_db)) + g)
        rms_before = float(np.sqrt(np.mean(err_before[mid] * err_before[mid])))
        _write_stats(
            enabled=False,
            k_used=0.0,
            before_rms=rms_before,
            after_rms=rms_before,
            delta_rms=0.0,
            conf_avg=conf_avg,
            reason="low_mid_conf",
        )
        return out

    err_before = tgt - ((meas - float(calc_offset_db)) + g)
    rms_before = float(np.sqrt(np.mean(err_before[mid] * err_before[mid])))
    err_s = smooth_gain_fractional_octave(f, err_before, smooth_value)

    g_refit = g.copy()
    g_refit[mid] = g_refit[mid] + float(k_refit) * np.asarray(err_s[mid], dtype=float)
    err_after = tgt - ((meas - float(calc_offset_db)) + g_refit)
    rms_after = float(np.sqrt(np.mean(err_after[mid] * err_after[mid])))
    delta_rms = float(rms_before - rms_after)

    if delta_rms <= 0.0:
        _write_stats(
            enabled=False,
            k_used=0.0,
            before_rms=rms_before,
            after_rms=rms_before,
            delta_rms=0.0,
            conf_avg=conf_avg,
            reason="no_improvement",
        )
        return out

    out[:n] = g_refit
    _write_stats(
        enabled=True,
        k_used=float(k_refit),
        before_rms=rms_before,
        after_rms=rms_after,
        delta_rms=delta_rms,
        conf_avg=conf_avg,
        reason="applied",
    )
    try:
        conf_txt = "n/a" if conf_avg is None else f"{float(conf_avg):.3f}"
        logger.info(
            "MidRefit: "
            f"k={float(k_refit):.2f}, smooth_oct={float(smooth_oct):.2f}, "
            f"conf_avg={conf_txt}, "
            f"rms_before={float(rms_before):.3f} dB, rms_after={float(rms_after):.3f} dB, "
            f"delta={float(delta_rms):.3f} dB, band={float(mid_lo):.0f}-{float(mid_hi):.0f} Hz"
        )
    except (AttributeError, TypeError, ValueError):
        pass
    return out


def _apply_bass_boost_post_restore(
    gain_db: np.ndarray,
    target_db: np.ndarray,
    boost_cap_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    *,
    hz_lo: float = 20.0,
    hz_hi: float = 200.0,
    strength: float = 0.6,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    return apply_bass_boost_post_restore(
        gain_db,
        target_db,
        boost_cap_db,
        freq_axis,
        mask_c,
        hz_lo=hz_lo,
        hz_hi=hz_hi,
        strength=strength,
    )


def _apply_confpull_post_slope(
    gain_db_in: np.ndarray,
    mask_c_in: np.ndarray,
    measured_ref_db: np.ndarray | None,
    *,
    cfg: Any,
    st: Any,
    conf_mask: np.ndarray,
    freq_axis: np.ndarray,
    logger: Any,
    apply_confidence_weighted_target_pull: Callable[..., Any],
) -> np.ndarray:
    return apply_confpull_post_slope(
        gain_db_in,
        mask_c_in,
        measured_ref_db,
        cfg=cfg,
        st=st,
        conf_mask=conf_mask,
        freq_axis=freq_axis,
        logger=logger,
        apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
    )


def _apply_post_limits_and_metrics(inputs: _MagPostProcessInputs) -> _MagPostProcessOutputs:
    return _apply_post_limits_and_metrics_impl(
        inputs,
        apply_mid_refit_pre_slope=_apply_mid_refit_pre_slope,
    )


def _run_mag_raw_stage(inputs: _MagPipelineInputs) -> _MagRawStageOutputs:
    return run_mag_raw_stage(
        inputs,
        apply_peak_priority_error_shaping=_apply_peak_priority_error_shaping,
        apply_smoothing=_apply_smoothing,
    )


def _run_mag_bassfirst_afdw_conf_stage(
    inputs: _MagPipelineInputs,
    raw_stage: _MagRawStageOutputs,
) -> _MagAdaptiveStageOutputs:
    return run_mag_bassfirst_afdw_conf_stage(
        inputs,
        raw_stage,
        select_bass_adaptive_conf_mask=_select_bass_adaptive_conf_mask,
        apply_confidence_adaptive_bass_smoothing=_apply_confidence_adaptive_bass_smoothing,
    )


def _run_mag_core_stage(inputs: _MagPipelineInputs) -> _MagCoreOutputs:
    return run_mag_core_stage(
        inputs,
        run_mag_raw_stage_fn=_run_mag_raw_stage,
        run_mag_bassfirst_afdw_conf_stage_fn=_run_mag_bassfirst_afdw_conf_stage,
    )


def _run_mag_correction_pipeline(inputs: _MagPipelineInputs) -> _MagCorrectionContext:
    cfg = inputs.cfg
    freq_axis = inputs.freq_axis
    st = inputs.st
    conf_mask = inputs.conf_mask
    logger = inputs.logger
    _stage_probe = inputs.stage_probe
    _cfg_float_allow_zero = inputs.cfg_float_allow_zero
    apply_confidence_weighted_target_pull = inputs.apply_confidence_weighted_target_pull

    core = _run_mag_core_stage(inputs)

    afdw_on = core.afdw_on
    base_sigma = core.base_sigma
    _filter_smooth = core.filter_smooth
    df_mode = core.df_mode
    raw_g = core.raw_g
    final_g = core.final_g
    mask_c = core.mask_c
    stage_probes = core.stage_probes
    use_bassfirst = core.use_bassfirst
    bf_room_mode = core.bf_room_mode
    bf_rel = core.bf_rel
    bf_conf_for_smoothing = core.bf_conf_for_smoothing
    gain_db = core.gain_db

    boost_peak_db = 0.0
    cut_peak_db = 0.0
    n_boost = 0
    boost_cand_peak = 0.0
    boost_cand_min_hz = float("nan")
    n_boost_cand = 0
    n_boost_cand_low = 0
    n_boost_cand_exc = 0
    softclip_boost_bins = 0
    softclip_cut_bins = 0
    over_boost = 0.0
    over_cut = 0.0
    hardclamp_boost_bins = 0
    hardclamp_cut_bins = 0
    hard_over_boost = 0.0
    hard_over_cut = 0.0
    clamp_dominance_level = "NONE"

    if core.mag_enabled:
        post = _apply_post_limits_and_metrics(
            _MagPostProcessInputs(
                cfg=cfg,
                freq_axis=freq_axis,
                st=st,
                logger=logger,
                stage_probe=_stage_probe,
                cfg_float_allow_zero=_cfg_float_allow_zero,
                mask_c=mask_c,
                gain_db=gain_db,
                gain_apply=core.gain_apply,
                raw_g=raw_g,
                final_g=final_g,
                pre_bass_adapt_g=(
                    None
                    if core.pre_bass_adapt_g is None
                    else np.asarray(core.pre_bass_adapt_g, dtype=float)
                ),
                raw_safe_ref=core.raw_safe_ref,
                conf_mask=conf_mask,
                filter_smooth=_filter_smooth,
                debug_stage_stats=core.debug_stage_stats,
                stage_probes=stage_probes,
                apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
                m_anal=np.asarray(inputs.m_anal, dtype=float),
                target_mags=np.asarray(inputs.target_mags, dtype=float),
                calc_offset_db=float(inputs.calc_offset_db),
            )
        )
        gain_db = post.gain_db
        stage_probes = post.stage_probes
        boost_peak_db = post.boost_peak_db
        cut_peak_db = post.cut_peak_db
        n_boost = post.n_boost
        boost_cand_peak = post.boost_cand_peak
        boost_cand_min_hz = post.boost_cand_min_hz
        n_boost_cand = post.n_boost_cand
        n_boost_cand_low = post.n_boost_cand_low
        n_boost_cand_exc = post.n_boost_cand_exc
        softclip_boost_bins = post.softclip_boost_bins
        softclip_cut_bins = post.softclip_cut_bins
        over_boost = post.over_boost
        over_cut = post.over_cut
        hardclamp_boost_bins = post.hardclamp_boost_bins
        hardclamp_cut_bins = post.hardclamp_cut_bins
        hard_over_boost = post.hard_over_boost
        hard_over_cut = post.hard_over_cut
        clamp_dominance_level = post.clamp_dominance_level

    return _MagCorrectionContext(
        afdw_on=bool(afdw_on),
        base_sigma=base_sigma,
        filter_smooth=_filter_smooth,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        mask_c=np.asarray(mask_c, dtype=bool),
        stage_probes=dict(stage_probes),
        use_bassfirst=bool(use_bassfirst),
        bf_room_mode=bf_room_mode,
        bf_rel=bf_rel,
        bf_conf_for_smoothing=bf_conf_for_smoothing,
        boost_peak_db=float(boost_peak_db),
        cut_peak_db=float(cut_peak_db),
        n_boost=int(n_boost),
        boost_cand_peak=float(boost_cand_peak),
        boost_cand_min_hz=float(boost_cand_min_hz),
        n_boost_cand=int(n_boost_cand),
        n_boost_cand_low=int(n_boost_cand_low),
        n_boost_cand_exc=int(n_boost_cand_exc),
        softclip_boost_bins=int(softclip_boost_bins),
        softclip_cut_bins=int(softclip_cut_bins),
        over_boost=float(over_boost),
        over_cut=float(over_cut),
        hardclamp_boost_bins=int(hardclamp_boost_bins),
        hardclamp_cut_bins=int(hardclamp_cut_bins),
        hard_over_boost=float(hard_over_boost),
        hard_over_cut=float(hard_over_cut),
        clamp_dominance_level=str(clamp_dominance_level),
        gain_db=np.asarray(gain_db, dtype=float),
    )
