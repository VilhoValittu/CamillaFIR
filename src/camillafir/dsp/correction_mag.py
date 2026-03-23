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
from .mag_limits import (
    _apply_hard_boost_cut_clamp,
    _apply_max_boost_cut,
    _apply_slope_limits,
    _blend_masked_fractional_octave,
)
from .mag_shape import (
    _apply_confidence_logic,
    _apply_regularization,
    _compute_error_db,
    _error_to_correction_mag,
    _resolve_filter_smooth,
    _select_active_band,
)
from .mag_telemetry import (
    _band_delta_metrics,
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
    try:
        enable = bool(getattr(cfg, "peak_priority_enable", True))
    except Exception:
        enable = True
    if not enable:
        return err_db

    try:
        strength = float(getattr(cfg, "peak_priority_strength", 0.5) or 0.0) #vaikuttaa boostin voimakkuuteen, mitä isompi, sitä miedompi boost
    except Exception:
        strength = 0.5
    if not np.isfinite(strength):
        strength = 0.0
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return err_db

    try:
        max_boost = float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
    except Exception:
        max_boost = 0.0
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
    try:
        cut_emph = float(getattr(cfg, "peak_priority_cut_emphasis", 0.0) or 0.0)
    except Exception:
        cut_emph = 0.0
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
    except Exception:
        pass

    try:
        logger.info(
            "PeakPriority: "
            f"enable=ON, strength={strength:.2f}, max_boost={max_boost:.2f} dB, "
            f"pos_err_peak: {float(np.max(np.asarray(err_db, dtype=float)[pos])):.2f} -> {float(np.max(e[pos])):.2f} dB"
        )
    except Exception:
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
    if df_mode:
        base_sigma = 60 // (filter_smooth / 12 if filter_smooth > 0 else 1)
        df_ref = 44100.0 / 65536.0
        sigma_hz = float(base_sigma) * df_ref
        sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=sigma_hz, fallback_bins=max(2.0, float(base_sigma)))
        smoothed_base = scipy.ndimage.gaussian_filter1d(err_db, sigma=float(sigma_bins))
    else:
        smoothed_base = smooth_gain_fractional_octave(freq_axis, err_db, filter_smooth)

    if not bool(getattr(cfg, "bass_smooth_adaptive", False)):
        try:
            if isinstance(st, dict):
                st["bass_adaptive_smoothing_enabled"] = False
                st["bass_adaptive_smoothing_avg_w_20_200"] = 0.0
        except Exception:
            pass
        return smoothed_base

    try:
        conf_arr = np.asarray(conf_mask, dtype=float)
    except Exception:
        conf_arr = None
    if conf_arr is None or conf_arr.shape != np.asarray(freq_axis, dtype=float).shape:
        try:
            if isinstance(st, dict):
                st["bass_adaptive_smoothing_enabled"] = False
                st["bass_adaptive_smoothing_avg_w_20_200"] = 0.0
        except Exception:
            pass
        return smoothed_base

    try:
        f_bass = float(getattr(cfg, "bass_smooth_hz", 200.0) or 200.0)
    except Exception:
        f_bass = 200.0
    if not np.isfinite(f_bass) or f_bass <= 0.0:
        f_bass = 200.0

    try:
        conf_floor = float(getattr(cfg, "bass_smooth_conf_floor", 0.3) or 0.3)
    except Exception:
        conf_floor = 0.3
    if not np.isfinite(conf_floor) or conf_floor <= 0.0:
        conf_floor = 0.3

    try:
        sigma_scale = float(getattr(cfg, "bass_smooth_sigma_scale", 1.4) or 1.4)
    except Exception:
        sigma_scale = 1.4
    if (not np.isfinite(sigma_scale)) or sigma_scale < 1.0:
        sigma_scale = 1.0
    try:
        w_gamma = float(getattr(cfg, "bass_smooth_w_gamma", 2.0) or 2.0)
    except Exception:
        w_gamma = 2.0
    if (not np.isfinite(w_gamma)) or w_gamma <= 0.0:
        w_gamma = 2.0
    try:
        w_max = float(getattr(cfg, "bass_smooth_w_max", 0.55) or 0.55)
    except Exception:
        w_max = 0.55
    if not np.isfinite(w_max):
        w_max = 0.55
    w_max = float(np.clip(w_max, 0.0, 1.0))
    try:
        if isinstance(st, dict):
            st["bass_adaptive_smoothing_sigma_scale"] = float(sigma_scale)
            st["bass_adaptive_smoothing_conf_floor"] = float(conf_floor)
            st["bass_adaptive_smoothing_w_gamma"] = float(w_gamma)
            st["bass_adaptive_smoothing_w_max"] = float(w_max)
    except Exception:
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
        except Exception:
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
    except Exception:
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

    def _mark_disabled() -> np.ndarray:
        try:
            if isinstance(st, dict):
                st["bass_adaptive_smoothing_enabled"] = False
                st["bass_adaptive_smoothing_avg_w_20_200"] = 0.0
                st["bass_adaptive_smoothing_delta_rms_db_20_200"] = 0.0
                st["bass_adaptive_smoothing_delta_max_db_20_200"] = 0.0
                st["bass_adaptive_smoothing_delta_max_hz_20_200"] = None
                st[f"bass_adaptive_smoothing_{stage_key}_enabled"] = False
                st[f"bass_adaptive_smoothing_{stage_key}_avg_w_20_200"] = 0.0
                st[f"bass_adaptive_smoothing_{stage_key}_delta_rms_db_20_200"] = 0.0
                st[f"bass_adaptive_smoothing_{stage_key}_delta_max_db_20_200"] = 0.0
                st[f"bass_adaptive_smoothing_{stage_key}_delta_max_hz_20_200"] = None
        except Exception:
            pass
        return base

    if not bool(getattr(cfg, "bass_smooth_adaptive", False)):
        return _mark_disabled()

    try:
        c = np.asarray(conf_mask, dtype=float)
    except Exception:
        c = None
    if c is None or c.shape != f.shape:
        return _mark_disabled()

    try:
        f_bass = float(getattr(cfg, "bass_smooth_hz", 200.0) or 200.0)
    except Exception:
        f_bass = 200.0
    if not np.isfinite(f_bass) or f_bass <= 0.0:
        f_bass = 200.0

    try:
        conf_floor = float(getattr(cfg, "bass_smooth_conf_floor", 0.3) or 0.3)
    except Exception:
        conf_floor = 0.3
    if not np.isfinite(conf_floor) or conf_floor <= 0.0:
        conf_floor = 0.3

    try:
        sigma_scale = float(getattr(cfg, "bass_smooth_sigma_scale", 1.4) or 1.4)
    except Exception:
        sigma_scale = 1.4
    if (not np.isfinite(sigma_scale)) or sigma_scale < 1.0:
        sigma_scale = 1.0
    try:
        w_gamma = float(getattr(cfg, "bass_smooth_w_gamma", 2.0) or 2.0)
    except Exception:
        w_gamma = 2.0
    if (not np.isfinite(w_gamma)) or w_gamma <= 0.0:
        w_gamma = 2.0
    try:
        w_max = float(getattr(cfg, "bass_smooth_w_max", 0.55) or 0.55)
    except Exception:
        w_max = 0.55
    if not np.isfinite(w_max):
        w_max = 0.55
    w_max = float(np.clip(w_max, 0.0, 1.0))
    try:
        if isinstance(st, dict):
            st["bass_adaptive_smoothing_sigma_scale"] = float(sigma_scale)
            st["bass_adaptive_smoothing_conf_floor"] = float(conf_floor)
            st["bass_adaptive_smoothing_w_gamma"] = float(w_gamma)
            st["bass_adaptive_smoothing_w_max"] = float(w_max)
    except Exception:
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
            st["bass_adaptive_smoothing_enabled"] = True
            st["bass_adaptive_smoothing_avg_w_20_200"] = float(w_avg)
            st["bass_adaptive_smoothing_delta_rms_db_20_200"] = float(d_rms)
            st["bass_adaptive_smoothing_delta_max_db_20_200"] = float(d_max)
            st["bass_adaptive_smoothing_delta_max_hz_20_200"] = (float(d_hz) if d_hz is not None else None)
            st["bass_adaptive_smoothing_sigma_bins"] = float(sigma_bins)
            st[f"bass_adaptive_smoothing_{stage_key}_enabled"] = True
            st[f"bass_adaptive_smoothing_{stage_key}_avg_w_20_200"] = float(w_avg)
            st[f"bass_adaptive_smoothing_{stage_key}_delta_rms_db_20_200"] = float(d_rms)
            st[f"bass_adaptive_smoothing_{stage_key}_delta_max_db_20_200"] = float(d_max)
            st[f"bass_adaptive_smoothing_{stage_key}_delta_max_hz_20_200"] = (float(d_hz) if d_hz is not None else None)
    except Exception:
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
    except Exception:
        c_raw = None
    try:
        c_bf = np.asarray(bf_conf_for_smoothing, dtype=float) if bf_conf_for_smoothing is not None else None
    except Exception:
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
                st["mid_refit_enabled"] = bool(enabled)
                st["mid_refit_k"] = float(k_used)
                st["mid_refit_err_rms_before"] = (
                    float(before_rms) if before_rms is not None and np.isfinite(float(before_rms)) else None
                )
                st["mid_refit_err_rms_after"] = (
                    float(after_rms) if after_rms is not None and np.isfinite(float(after_rms)) else None
                )
                st["mid_refit_delta_rms"] = (
                    float(delta_rms) if delta_rms is not None and np.isfinite(float(delta_rms)) else None
                )
                st["mid_refit_conf_avg_200_2000"] = (
                    float(conf_avg) if conf_avg is not None and np.isfinite(float(conf_avg)) else None
                )
                st["mid_refit_reason"] = str(reason)
        except Exception:
            pass

    if not bool(getattr(cfg, "enable_mag_correction", False)):
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

    try:
        refit_on = bool(getattr(cfg, "mid_refit_enable", True))
    except Exception:
        refit_on = True
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

    try:
        mid_lo = float(getattr(cfg, "mid_refit_hz_lo", 200.0) or 200.0)
    except Exception:
        mid_lo = 200.0
    try:
        mid_hi = float(getattr(cfg, "mid_refit_hz_hi", 2000.0) or 2000.0)
    except Exception:
        mid_hi = 2000.0
    if (not np.isfinite(mid_lo)) or mid_lo < 1.0:
        mid_lo = 200.0
    if (not np.isfinite(mid_hi)) or mid_hi <= (mid_lo + 1.0):
        mid_hi = 2000.0
        if mid_hi <= (mid_lo + 1.0):
            mid_hi = mid_lo + 1.0

    try:
        k_refit = float(getattr(cfg, "mid_refit_k", 0.45) or 0.45)
    except Exception:
        k_refit = 0.45
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

    try:
        smooth_oct = float(getattr(cfg, "mid_refit_smooth_oct", 0.6) or 0.6)
    except Exception:
        smooth_oct = 0.6
    if (not np.isfinite(smooth_oct)) or smooth_oct <= 0.0:
        smooth_oct = 0.6
    smooth_oct = float(np.clip(smooth_oct, 1.0 / 192.0, 1.0))
    smooth_value = float(np.clip(1.0 / smooth_oct, 1.0, 192.0))

    try:
        conf_min_avg = float(getattr(cfg, "mid_refit_conf_min_avg", 0.2) or 0.2)
    except Exception:
        conf_min_avg = 0.2
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
    except Exception:
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
    except Exception:
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
    """Palauttaa osan bassoboostista takaisin target-suuntaan post-limit-vaiheessa.

    Muokkaa vain boost-bineja, joissa:
    - ollaan mask_c-alueella
    - taajuus on [hz_lo, hz_hi]
    - per-bin boost_cap sallii lisaboostia
    - target_db on nykykayraa korkeampi
    """
    out = np.asarray(gain_db, dtype=float).copy()
    tgt = np.asarray(target_db, dtype=float)
    cap = np.asarray(boost_cap_db, dtype=float)
    f = np.asarray(freq_axis, dtype=float)
    m = np.asarray(mask_c, dtype=bool)
    n = int(min(out.size, tgt.size, cap.size, f.size, m.size))
    meta: dict[str, float | int | bool] = {
        "enabled": False,
        "bins": 0,
        "delta_rms_20_200": 0.0,
        "delta_max_20_200": 0.0,
    }
    if n < 8:
        return out, meta
    out = out[:n].copy()
    tgt = tgt[:n]
    cap = cap[:n]
    f = f[:n]
    m = m[:n]
    s = float(np.clip(float(strength), 0.0, 1.0))
    if s <= 0.0:
        return out, meta
    valid = np.isfinite(out) & np.isfinite(tgt) & np.isfinite(cap) & np.isfinite(f)
    rmask = (
        valid
        & m
        & (f >= float(hz_lo))
        & (f <= float(hz_hi))
        & (cap > out + 1e-9)
        & (tgt > out + 1e-9)
    )
    if not np.any(rmask):
        return out, meta
    pre = out.copy()
    out[rmask] = out[rmask] + s * (tgt[rmask] - out[rmask])
    out[rmask] = np.minimum(out[rmask], cap[rmask])
    b20 = valid & (f >= 20.0) & (f <= 200.0)
    d = out - pre
    if np.any(b20):
        meta["delta_rms_20_200"] = float(np.sqrt(np.mean(d[b20] * d[b20])))
        meta["delta_max_20_200"] = float(np.max(np.abs(d[b20])))
    meta["enabled"] = True
    meta["bins"] = int(np.count_nonzero(rmask))
    return out, meta


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
    """Soveltaa confidence-pohjaisen target-pullin slope-rajoituksen jalkeen."""
    try:
        if gain_db_in is None or mask_c_in is None:
            return gain_db_in
        if not (isinstance(gain_db_in, np.ndarray) and isinstance(mask_c_in, np.ndarray)):
            return gain_db_in
        if gain_db_in.size < 16 or gain_db_in.shape != mask_c_in.shape:
            return gain_db_in
        if not np.any(mask_c_in):
            return gain_db_in
        _conf_floor = float(getattr(cfg, "conf_pull_floor", 0.05) or 0.05)
        _conf_ceil = float(getattr(cfg, "conf_pull_ceil", 0.95) or 0.95)
        _conf_max_hz = getattr(cfg, "conf_pull_max_hz", 200.0)
        _conf_max_hz = None if _conf_max_hz is None else float(_conf_max_hz)
        _gamma_cut = float(getattr(cfg, "conf_pull_gamma_cut", 0.55) or 0.55)
        _gamma_boost = float(getattr(cfg, "conf_pull_gamma_boost", 1.35) or 1.35)
        _conf_sigma = float(getattr(cfg, "conf_pull_conf_smooth_sigma", 2.0) or 2.0)
        _bass_floor_hz = float(getattr(cfg, "conf_pull_bass_floor_hz", 120.0) or 120.0)
        _bass_floor_min = float(getattr(cfg, "conf_pull_bass_floor_min", 0.25) or 0.25)
        _bass_boost_floor_hz = float(getattr(cfg, "conf_pull_bass_boost_floor_hz", 200.0) or 200.0)
        _bass_boost_floor_min = float(getattr(cfg, "conf_pull_bass_boost_floor_min", 0.45) or 0.45)
        _bass_boost_restore = float(getattr(cfg, "conf_pull_bass_boost_restore", 0.55) or 0.0)
        _bass_adaptive_isolation_mode = bool(getattr(cfg, "bass_adaptive_isolation_mode", False))
        if _bass_adaptive_isolation_mode:
            _bass_boost_floor_hz = 0.0
            _bass_boost_floor_min = 0.0
            _bass_boost_restore = 0.0
        if not np.isfinite(_conf_sigma) or _conf_sigma < 0.0:
            _conf_sigma = 0.0
        if not np.isfinite(_bass_floor_hz) or _bass_floor_hz < 0.0:
            _bass_floor_hz = 0.0
        if not np.isfinite(_bass_floor_min) or _bass_floor_min < 0.0:
            _bass_floor_min = 0.0
        if not np.isfinite(_bass_boost_floor_hz) or _bass_boost_floor_hz < 0.0:
            _bass_boost_floor_hz = 0.0
        if not np.isfinite(_bass_boost_floor_min) or _bass_boost_floor_min < 0.0:
            _bass_boost_floor_min = 0.0
        if not np.isfinite(_bass_boost_restore) or _bass_boost_restore < 0.0:
            _bass_boost_restore = 0.0
        _bass_floor_min = float(np.clip(_bass_floor_min, 0.0, 1.0))
        _bass_boost_floor_min = float(np.clip(_bass_boost_floor_min, 0.0, 1.0))
        _bass_boost_restore = float(np.clip(_bass_boost_restore, 0.0, 1.0))
        try:
            if isinstance(st, dict):
                st["bass_adaptive_isolation_mode"] = bool(_bass_adaptive_isolation_mode)
                st["conf_pull_post_bass_boost_floor_hz"] = float(_bass_boost_floor_hz)
                st["conf_pull_post_bass_boost_floor_min"] = float(_bass_boost_floor_min)
                st["conf_pull_post_bass_boost_restore"] = float(_bass_boost_restore)
        except Exception:
            pass
        conf_for_pull = conf_mask
        try:
            c0 = np.asarray(conf_mask, dtype=float)
            if c0.shape == gain_db_in.shape:
                if _conf_sigma > 0.0:
                    c0 = scipy.ndimage.gaussian_filter1d(c0, sigma=float(_conf_sigma))
                c0 = np.clip(c0, 0.0, 1.0)
                if _bass_floor_hz > 0.0 and _bass_floor_min > 0.0:
                    f0 = np.asarray(freq_axis, dtype=float)
                    bm = (f0 > 0.0) & (f0 <= float(_bass_floor_hz))
                    if np.any(bm):
                        c0[bm] = np.maximum(c0[bm], float(_bass_floor_min))
                if _bass_boost_floor_hz > 0.0 and _bass_boost_floor_min > 0.0:
                    f0 = np.asarray(freq_axis, dtype=float)
                    g0 = np.asarray(gain_db_in, dtype=float)
                    bb = (f0 > 0.0) & (f0 <= float(_bass_boost_floor_hz)) & (g0 > 0.0)
                    if np.any(bb):
                        c0[bb] = np.maximum(c0[bb], float(_bass_boost_floor_min))
                conf_for_pull = np.clip(c0, 0.0, 1.0)
        except Exception:
            conf_for_pull = conf_mask
        try:
            if measured_ref_db is not None:
                g_ref = np.asarray(measured_ref_db, dtype=float)
                if g_ref.shape != gain_db_in.shape:
                    g_ref = None
            else:
                g_ref = None
        except Exception:
            g_ref = None
        if g_ref is None:
            try:
                g_in = np.asarray(gain_db_in, dtype=float).copy()
                idx = np.where(mask_c_in)[0]
                i0, i1 = int(idx[0]), int(idx[-1])
                if i0 > 0:
                    g_in[:i0] = g_in[i0]
                if i1 < (g_in.size - 1):
                    g_in[i1 + 1:] = g_in[i1]
                g_ref = psycho_smooth_safe_gain(freq_axis, g_in)
            except Exception:
                g_ref = np.asarray(gain_db_in, dtype=float)
        g_ref = np.where(mask_c_in, np.asarray(g_ref, dtype=float), gain_db_in)
        try:
            # Conf-pull must not introduce new/increased boost.
            # If current curve is non-boost at a bin, pull reference cannot be > 0 dB.
            # If current curve is already boosting, pull reference cannot exceed that boost.
            g_cur = np.asarray(gain_db_in, dtype=float)
            g_ref = np.where(g_cur > 0.0, np.minimum(g_ref, g_cur), np.minimum(g_ref, 0.0))
        except Exception:
            pass
        out = apply_confidence_weighted_target_pull(
            target_db=gain_db_in,
            measured_db=g_ref,
            confidence_mask=conf_for_pull,
            conf_floor=_conf_floor,
            conf_ceil=_conf_ceil,
            freq_axis=freq_axis,
            freq_limit_hz=_conf_max_hz,
            gamma_cut=_gamma_cut,
            gamma_boost=_gamma_boost,
            return_telemetry=True,
        )
        if isinstance(out, tuple) and len(out) == 2:
            gain_out, _tel = out
        else:
            gain_out, _tel = out, None
        gain_out = np.where(mask_c_in, np.asarray(gain_out, dtype=float), gain_db_in)
        try:
            if _bass_boost_floor_hz > 0.0 and _bass_boost_restore > 0.0:
                f0 = np.asarray(freq_axis, dtype=float)
                gt = np.asarray(gain_db_in, dtype=float)
                go = np.asarray(gain_out, dtype=float)
                cp = np.asarray(conf_for_pull, dtype=float)
                bb = mask_c_in & (f0 > 0.0) & (f0 <= float(_bass_boost_floor_hz)) & (gt > 0.0)
                if np.any(bb):
                    w = np.clip(
                        (cp[bb] - float(_bass_boost_floor_min)) / max(1e-9, 1.0 - float(_bass_boost_floor_min)),
                        0.0,
                        1.0,
                    )
                    restore = np.clip(float(_bass_boost_restore) * w, 0.0, 1.0)
                    gb = go[bb]
                    go[bb] = gb + restore * (gt[bb] - gb)
                    gain_out = np.where(mask_c_in, go, gain_db_in)
                    if isinstance(st, dict):
                        st["conf_pull_post_bass_boost_restore"] = float(_bass_boost_restore)
                        st["conf_pull_post_bass_boost_restore_mean_eff"] = float(np.mean(restore))
                        st["conf_pull_post_bass_boost_restore_max_eff"] = float(np.max(restore))
                        st["conf_pull_post_bass_boost_restore_bins"] = int(np.count_nonzero(bb))
        except Exception:
            pass
        try:
            if isinstance(_tel, dict):
                _w_eff = _tel.get("w_eff", None)
                _pm = _tel.get("pull_mask", None)
                _ps = _tel.get("pull_strength", None)
            else:
                _w_eff = _pm = _ps = None
            if _w_eff is not None:
                _w_eff = np.asarray(_w_eff, dtype=float)
            if _ps is not None:
                _ps = np.asarray(_ps, dtype=float)
            if _pm is not None:
                _pm = np.asarray(_pm, dtype=bool)
            if (_pm is None) or (_pm.shape != mask_c_in.shape):
                _pm2 = mask_c_in
            else:
                _pm2 = (_pm & mask_c_in)
            if (_w_eff is not None) and (_w_eff.shape == _pm2.shape) and np.any(_pm2):
                wv = _w_eff[_pm2]
                if _ps is not None and (_ps.shape == _pm2.shape):
                    pv = _ps[_pm2]
                else:
                    pv = np.clip(1.0 - wv, 0.0, 1.0)
                act = pv > 0.05
                n_mask = int(np.count_nonzero(_pm2))
                n_act = int(np.count_nonzero(act))
                act_pct = 100.0 * n_act / max(1, n_mask)
                w_mean = float(np.mean(wv))
                w_min = float(np.min(wv))
                w_p10 = float(np.percentile(wv, 10))
                w_p50 = float(np.percentile(wv, 50))
                w_p90 = float(np.percentile(wv, 90))
                p_mean = float(np.mean(pv))
                p_max = float(np.max(pv))
                f_pull_max = None
                try:
                    idxs = np.where(_pm2)[0]
                    k = int(np.argmax(pv))
                    idxm = int(idxs[k])
                    f_pull_max = float(freq_axis[idxm])
                except Exception:
                    f_pull_max = None
                freq_txt = f", max@{f_pull_max:.1f}Hz" if f_pull_max is not None else ""
                logger.info(
                    "ConfPullPost: "
                    f"mask_bins={n_mask}, active_bins={n_act} ({act_pct:.1f}%), "
                    f"w_eff(mean={w_mean:.3f}, p10={w_p10:.3f}, p50={w_p50:.3f}, "
                    f"p90={w_p90:.3f}, min={w_min:.3f}), "
                    f"pull_strength(mean={p_mean:.3f}, max={p_max:.3f}{freq_txt}), "
                    f"floor={_conf_floor:.3f}, ceil={_conf_ceil:.3f}, "
                    f"max_hz={_conf_max_hz}, gamma_cut={_gamma_cut:.2f}, gamma_boost={_gamma_boost:.2f}"
                )
                if isinstance(st, dict):
                    st["conf_pull_post_floor"] = float(_conf_floor)
                    st["conf_pull_post_ceil"] = float(_conf_ceil)
                    st["conf_pull_post_max_hz"] = None if _conf_max_hz is None else float(_conf_max_hz)
                    st["conf_pull_post_gamma_cut"] = float(_gamma_cut)
                    st["conf_pull_post_gamma_boost"] = float(_gamma_boost)
                    st["conf_pull_post_active_pct"] = float(act_pct)
                    st["conf_pull_post_w_eff_mean"] = float(w_mean)
                    st["conf_pull_post_strength_mean"] = float(p_mean)
                    st["conf_pull_post_strength_max"] = float(p_max)
                    st["conf_pull_post_strength_max_hz"] = float(f_pull_max) if f_pull_max is not None else None
                    st["conf_pull_post_conf_smooth_sigma"] = float(_conf_sigma)
                    st["conf_pull_post_bass_floor_hz"] = float(_bass_floor_hz)
                    st["conf_pull_post_bass_floor_min"] = float(_bass_floor_min)
                    st["conf_pull_post_bass_boost_floor_hz"] = float(_bass_boost_floor_hz)
                    st["conf_pull_post_bass_boost_floor_min"] = float(_bass_boost_floor_min)
                    st["conf_pull_post_bass_boost_restore"] = float(_bass_boost_restore)
        except Exception:
            pass
        return gain_out
    except Exception:
        return gain_db_in


def _apply_post_limits_and_metrics(inputs: _MagPostProcessInputs) -> _MagPostProcessOutputs:
    """Ajaa mag-korjauksen loppuvaiheen: low-bass policy, clamping, slope/fade ja metriikat."""

    cfg = inputs.cfg
    freq_axis = inputs.freq_axis
    st = inputs.st
    logger = inputs.logger
    _stage_probe = inputs.stage_probe
    _cfg_float_allow_zero = inputs.cfg_float_allow_zero
    mask_c = inputs.mask_c
    gain_db = inputs.gain_db
    gain_apply = inputs.gain_apply
    raw_g = inputs.raw_g
    final_g = inputs.final_g
    raw_safe_ref = inputs.raw_safe_ref
    conf_mask = inputs.conf_mask
    _filter_smooth = inputs.filter_smooth
    debug_stage_stats = inputs.debug_stage_stats
    stage_probes = dict(inputs.stage_probes)
    apply_confidence_weighted_target_pull = inputs.apply_confidence_weighted_target_pull

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

    low_cut_enable = True
    try:
        low_cut_enable = bool(getattr(cfg, "low_bass_cut_enable", True))
    except Exception:
        low_cut_enable = True
    low_hz = _cfg_float_allow_zero(cfg, "low_bass_cut_hz", 0.0)
    try:
        low_cut_strength = float(getattr(cfg, "low_bass_cut_strength", 0.0) or 0.0)
    except Exception:
        low_cut_strength = 0.0
    if not np.isfinite(low_cut_strength):
        low_cut_strength = 0.0
    low_cut_strength = float(np.clip(low_cut_strength, 0.0, 1.0))
    low_cut_floor_ref = None
    low_mask = mask_c & (freq_axis > 0) & (freq_axis <= low_hz)
    if low_cut_enable and np.any(low_mask):
        low_cut = np.minimum(gain_apply[low_mask], 0.0)
        if low_cut_strength > 0.0:
            stronger_cut = np.minimum(final_g[low_mask], raw_g[low_mask])
            stronger_cut = np.minimum(stronger_cut, 0.0)
            low_cut = (1.0 - low_cut_strength) * low_cut + (low_cut_strength) * stronger_cut
        gain_apply[low_mask] = low_cut
        if low_cut_strength > 0.0:
            low_cut_floor_ref = np.full_like(gain_apply, np.nan, dtype=float)
            low_cut_floor_ref[low_mask] = np.asarray(low_cut, dtype=float)
    try:
        _tmp_after_low = np.zeros_like(gain_db, dtype=float)
        _tmp_after_low[mask_c] = gain_apply[mask_c]
        _record_stage_probe(stage_probes, "after_lowbass_policy", _stage_probe, freq_axis, _tmp_after_low, mask_c, cfg, logger)
        logger.info(
            f"CFG CHECK: conf_pull_floor={cfg.conf_pull_floor}, "
            f"gamma_cut={cfg.conf_pull_gamma_cut}, "
            f"low_bass_cut_strength={cfg.low_bass_cut_strength}"
        )
    except Exception:
        pass

    max_cut_db = abs(float(getattr(cfg, "max_cut_db", 15.0) or 15.0))
    max_boost_db_base = float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
    boost_cap_db = np.full_like(gain_db, float(max_boost_db_base), dtype=float)
    bass_boost_cap_mask = np.zeros_like(mask_c, dtype=bool)
    bass_boost_cap_enabled = False
    bass_boost_cap_extra_db = 0.0
    bass_boost_cap_hz = 200.0
    bass_boost_cap_conf_min = 0.55
    bass_boost_post_restore_enable = True
    bass_boost_post_restore_strength = 0.60
    bass_adaptive_isolation_mode = False
    try:
        bass_adaptive_isolation_mode = bool(getattr(cfg, "bass_adaptive_isolation_mode", False))
    except Exception:
        bass_adaptive_isolation_mode = False
    try:
        cap_enable = bool(getattr(cfg, "bass_boost_cap_enable", True))
    except Exception:
        cap_enable = True
    try:
        bass_boost_cap_extra_db = float(getattr(cfg, "bass_boost_cap_extra_db", 2.0) or 0.0)
    except Exception:
        bass_boost_cap_extra_db = 2.0
    try:
        bass_boost_cap_hz = float(getattr(cfg, "bass_boost_cap_hz", 200.0) or 200.0)
    except Exception:
        bass_boost_cap_hz = 200.0
    try:
        bass_boost_cap_conf_min = float(getattr(cfg, "bass_boost_cap_conf_min", 0.55) or 0.55)
    except Exception:
        bass_boost_cap_conf_min = 0.55
    try:
        bass_boost_post_restore_enable = bool(getattr(cfg, "bass_boost_post_restore_enable", True))
    except Exception:
        bass_boost_post_restore_enable = True
    if bass_adaptive_isolation_mode:
        cap_enable = False
        bass_boost_post_restore_enable = False
    try:
        bass_boost_post_restore_strength = float(getattr(cfg, "bass_boost_post_restore_strength", 0.60) or 0.0)
    except Exception:
        bass_boost_post_restore_strength = 0.60
    bass_boost_cap_extra_db = float(max(0.0, bass_boost_cap_extra_db))
    bass_boost_cap_hz = float(max(20.0, bass_boost_cap_hz))
    bass_boost_cap_conf_min = float(np.clip(bass_boost_cap_conf_min, 0.0, 0.99))
    bass_boost_post_restore_strength = float(np.clip(bass_boost_post_restore_strength, 0.0, 1.0))
    if (
        bool(cap_enable)
        and max_boost_db_base > 0.0
        and bass_boost_cap_extra_db > 0.0
        and isinstance(conf_mask, np.ndarray)
        and conf_mask.shape == gain_db.shape
    ):
        try:
            c = np.asarray(conf_mask, dtype=float)
            c = np.clip(np.nan_to_num(c, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
            w = np.clip(
                (c - float(bass_boost_cap_conf_min)) / max(1e-9, 1.0 - float(bass_boost_cap_conf_min)),
                0.0,
                1.0,
            )
            bass_boost_cap_mask = mask_c & (freq_axis >= 20.0) & (freq_axis <= float(bass_boost_cap_hz))
            if np.any(bass_boost_cap_mask):
                boost_cap_db[bass_boost_cap_mask] = (
                    float(max_boost_db_base)
                    + float(bass_boost_cap_extra_db) * np.asarray(w[bass_boost_cap_mask], dtype=float)
                )
                bass_boost_cap_enabled = bool(np.any(boost_cap_db[bass_boost_cap_mask] > (max_boost_db_base + 1e-6)))
        except Exception:
            bass_boost_cap_mask = np.zeros_like(mask_c, dtype=bool)
            bass_boost_cap_enabled = False
    try:
        if isinstance(st, dict):
            st["bass_adaptive_isolation_mode"] = bool(bass_adaptive_isolation_mode)
            st["bass_boost_cap_enabled"] = bool(bass_boost_cap_enabled)
            st["bass_boost_cap_hz"] = float(bass_boost_cap_hz)
            st["bass_boost_cap_conf_min"] = float(bass_boost_cap_conf_min)
            st["bass_boost_cap_extra_db"] = float(bass_boost_cap_extra_db)
            st["bass_boost_post_restore_enable"] = bool(bass_boost_post_restore_enable)
            st["bass_boost_post_restore_strength"] = float(bass_boost_post_restore_strength)
            b20 = mask_c & (freq_axis >= 20.0) & (freq_axis <= 200.0)
            if np.any(b20):
                extra = np.maximum(0.0, np.asarray(boost_cap_db[b20], dtype=float) - float(max_boost_db_base))
                st["bass_boost_cap_avg_extra_db_20_200"] = float(np.mean(extra))
                st["bass_boost_cap_max_extra_db_20_200"] = float(np.max(extra))
            else:
                st["bass_boost_cap_avg_extra_db_20_200"] = 0.0
                st["bass_boost_cap_max_extra_db_20_200"] = 0.0
    except Exception:
        pass
    try:
        logger.info(
            "Diagnostic: "
            f"max_boost_db={float(max_boost_db_base):.2f} dB, "
            f"bass_adaptive_isolation={'ON' if bool(bass_adaptive_isolation_mode) else 'OFF'}, "
            f"bass_boost_cap={'ON' if bass_boost_cap_enabled else 'OFF'} "
            f"(extra={float(bass_boost_cap_extra_db):.2f} dB, hz<={float(bass_boost_cap_hz):.1f}, conf_min={float(bass_boost_cap_conf_min):.2f}), "
            f"bass_boost_post_restore={'ON' if bool(bass_boost_post_restore_enable) else 'OFF'} "
            f"(strength={float(bass_boost_post_restore_strength):.2f}), "
            f"max_cut_db={float(max_cut_db):.2f} dB, "
            f"low_bass_cut_hz={float(low_hz):.1f} Hz, "
            f"exc_prot={'ON' if bool(getattr(cfg,'exc_prot',False)) else 'OFF'}, "
            f"exc_freq={float(getattr(cfg,'exc_freq',0.0) or 0.0):.1f} Hz, "
            f"do_normalize={'ON' if bool(getattr(cfg,'do_normalize',False)) else 'OFF'}, "
            f"global_gain_db={float(getattr(cfg,'global_gain_db',0.0) or 0.0):.2f} dB, "
            f"max_slope_db_per_oct={float(getattr(cfg,'max_slope_db_per_oct',0.0) or 0.0):.1f}"
        )
    except Exception:
        pass

    try:
        _cand = np.zeros_like(gain_db, dtype=float)
        _cand[mask_c] = gain_apply[mask_c]
        cand_boost_mask = ((_cand > 1e-6) & mask_c)
        boost_cand_peak = float(np.max(_cand[mask_c])) if np.any(mask_c) else 0.0
        cand_boost_pos_mask = cand_boost_mask & (freq_axis > 0.0)
        if np.any(cand_boost_pos_mask):
            boost_cand_min_hz = float(np.min(freq_axis[cand_boost_pos_mask]))
        else:
            boost_cand_min_hz = float("nan")
        cut_cand_peak = float(np.min(_cand[mask_c])) if np.any(mask_c) else 0.0
        n_boost_cand = int(np.sum(cand_boost_mask))
        n_boost_cand_low = int(np.sum((_cand > 1e-6) & mask_c & (freq_axis <= low_hz)))
        if bool(getattr(cfg, "exc_prot", False)):
            exc_f = float(getattr(cfg, "exc_freq", 0.0) or 0.0)
            n_boost_cand_exc = int(np.sum((_cand > 1e-6) & mask_c & (freq_axis < exc_f)))
        else:
            n_boost_cand_exc = 0
    except Exception:
        boost_cand_peak, cut_cand_peak = 0.0, 0.0
        boost_cand_min_hz = float("nan")
        n_boost_cand, n_boost_cand_low, n_boost_cand_exc = 0, 0, 0

    tmp = np.zeros_like(gain_db, dtype=float)
    tmp[mask_c] = gain_apply[mask_c]
    _record_stage_probe(stage_probes, "pre_softclip", _stage_probe, freq_axis, tmp, mask_c, cfg, logger)
    try:
        _pre_soft = tmp.copy()
        _max_boost = float(max_boost_db_base)
        _max_boost_local = np.asarray(boost_cap_db, dtype=float)
        _max_cut = float(max_cut_db)
        if np.any(mask_c):
            over_boost = (
                float(np.max(_pre_soft[mask_c] - _max_boost_local[mask_c]))
                if _max_boost > 0
                else float(np.max(_pre_soft[mask_c]))
            )
            over_boost = max(0.0, over_boost)
            over_cut = float(np.max((-_pre_soft[mask_c]) - _max_cut))
            over_cut = max(0.0, over_cut)
        else:
            over_boost, over_cut = 0.0, 0.0
    except Exception:
        _pre_soft = tmp
        over_boost, over_cut = 0.0, 0.0
    try:
        _mc = float(max_cut_db)
        _mb = np.asarray(boost_cap_db, dtype=float)
        if _mb.shape != tmp.shape:
            _mb = np.full_like(tmp, float(max_boost_db_base), dtype=float)
        _post_soft = np.asarray(tmp, dtype=float).copy()
        pos = mask_c & (_post_soft > 0.0)
        if np.any(pos):
            mbp = np.maximum(np.asarray(_mb[pos], dtype=float), 0.0)
            _post_soft[pos] = np.where(
                mbp > 0.0,
                mbp * np.tanh(_post_soft[pos] / (mbp + 1e-12)),
                0.0,
            )
        neg = mask_c & (~pos)
        if np.any(neg):
            _post_soft[neg] = (-_mc * np.tanh((-_post_soft[neg]) / (_mc + 1e-12))) if _mc > 0.0 else _post_soft[neg]
        tmp = _post_soft
    except Exception:
        tmp = _apply_max_boost_cut(tmp, cfg, max_cut_db)
    try:
        _post_soft = np.asarray(tmp, dtype=float)
        if np.any(mask_c):
            _mb = np.asarray(boost_cap_db, dtype=float)
            softclip_boost_bins = int(
                np.sum(
                    (_pre_soft[mask_c] > (_mb[mask_c] + 1e-9))
                    & (_post_soft[mask_c] <= (_mb[mask_c] + 1e-9))
                )
            )
            softclip_cut_bins = int(np.sum((_pre_soft[mask_c] < (-_max_cut - 1e-9)) & (_post_soft[mask_c] >= (-_max_cut - 1e-9))))
        else:
            softclip_boost_bins, softclip_cut_bins = 0, 0
        logger.info(
            "Clamp: soft_clip "
            f"(max_boost_base={_max_boost:.2f} dB, max_cut={_max_cut:.2f} dB) -> "
            f"boost_clipped_bins={softclip_boost_bins}, cut_clipped_bins={softclip_cut_bins}, "
            f"worst_over_boost={over_boost:.2f} dB, worst_over_cut={over_cut:.2f} dB"
        )
    except Exception:
        softclip_boost_bins, softclip_cut_bins = 0, 0
        over_boost, over_cut = 0.0, 0.0
    _record_stage_probe(stage_probes, "post_softclip", _stage_probe, freq_axis, tmp, mask_c, cfg, logger)

    gain_db[mask_c] = tmp[mask_c]
    try:
        _pre_mid_refit = np.asarray(gain_db, dtype=float).copy()
        gain_db = _apply_mid_refit_pre_slope(
            gain_db,
            freq_axis,
            mask_c,
            m_anal=inputs.m_anal,
            target_mags=inputs.target_mags,
            calc_offset_db=inputs.calc_offset_db,
            conf_mask=conf_mask,
            cfg=cfg,
            st=st,
            logger=logger,
        )
        _log_stage_stats(
            "gain_db_post_mid_refit",
            gain_db,
            mask_c,
            ref=_pre_mid_refit,
            logger=logger,
            enabled=debug_stage_stats,
        )
    except Exception:
        pass
    gain_db, slope_info = _apply_slope_limits(gain_db, freq_axis, cfg, st, mask_c)
    max_slope = float(slope_info["max_slope"])
    max_slope_boost = float(slope_info["max_slope_boost"])
    max_slope_cut = float(slope_info["max_slope_cut"])
    if max_slope > 0 or max_slope_boost > 0 or max_slope_cut > 0:
        _log_stage_stats("gain_db_post_slope", gain_db, mask_c, logger=logger, enabled=debug_stage_stats)
        try:
            _pre = gain_db.copy()
            gain_db = _apply_confpull_post_slope(
                gain_db,
                mask_c,
                raw_safe_ref,
                cfg=cfg,
                st=st,
                conf_mask=conf_mask,
                freq_axis=freq_axis,
                logger=logger,
                apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
            )
            _log_stage_stats("gain_db_post_confpull", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats)
        except Exception:
            pass
        try:
            logger.info(
                "Slope limit: "
                f"boost={float(max_slope_boost):.1f} dB/oct | "
                f"cut={float(max_slope_cut):.1f} dB/oct "
                f"(legacy max_slope_db_per_oct={float(max_slope):.1f})"
            )
        except Exception:
            pass
    try:
        if np.any(mask_c):
            mix = float(np.clip(float(getattr(cfg, "reg_strength", 30.0) or 30.0) / 100.0, 0.0, 1.0))
            if mix > 0.0:
                _pre = gain_db.copy()
                gain_db = _blend_masked_fractional_octave(
                    gain_db,
                    freq_axis,
                    mask_c,
                    smooth_value=_filter_smooth,
                    mix=mix,
                )
                _log_stage_stats("gain_db_post_filter_smooth", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats)
    except Exception:
        pass

    if cfg.exc_prot:
        f_start = cfg.exc_freq
        f_end = cfg.exc_freq * 1.41
        prot_mask = freq_axis < f_start
        gain_db[prot_mask] = np.minimum(gain_db[prot_mask], 0.0)
        trans_mask = (freq_axis >= f_start) & (freq_axis <= f_end)
        if np.any(trans_mask):
            fade = (freq_axis[trans_mask] - f_start) / (f_end - f_start)
            allowed_boost = fade * cfg.max_boost_db
            gain_db[trans_mask] = np.minimum(gain_db[trans_mask], allowed_boost)
        logger.info(f"Exc Prot: Full protection < {f_start}Hz, Soft fade up to {f_end:.1f}Hz.")

    try:
        if bool(getattr(cfg, "is_wav_source", False)) and np.any(mask_c):
            cmin = float(getattr(cfg, "mag_c_min", 0.0) or 0.0)
            cmax = float(getattr(cfg, "mag_c_max", 0.0) or 0.0)
            tw = float(getattr(cfg, "trans_width", 0.0) or 0.0)
            if np.isfinite(cmin) and np.isfinite(cmax) and np.isfinite(tw) and (cmax > cmin):
                if tw <= 0.0:
                    tw = max(50.0, 0.4 * cmax)
                f_lo = max(cmin, cmax - max(30.0, 0.35 * tw))
                f_hi = min(float(np.max(freq_axis)), cmax + max(45.0, 0.55 * tw))
                zone = (freq_axis >= f_lo) & (freq_axis <= f_hi)
                if int(np.count_nonzero(zone)) >= 8:
                    _pre = gain_db.copy()
                    g0 = np.asarray(gain_db, dtype=float).copy()
                    g_sm = smooth_gain_fractional_octave(freq_axis, g0, 24.0)
                    span = max(1e-9, float(f_hi - f_lo))
                    x = np.clip((freq_axis - f_lo) / span, 0.0, 1.0)
                    ramp = 0.5 - 0.5 * np.cos(np.pi * x)
                    sigma_hz = max(20.0, 0.20 * max(tw, 1.0))
                    focus = np.exp(-0.5 * ((freq_axis - cmax) / sigma_hz) ** 2)
                    w = np.zeros_like(g0, dtype=float)
                    w[zone] = ramp[zone] * focus[zone]
                    mix = 0.55
                    gain_db = g0 + (g_sm - g0) * (mix * w)
                    _log_stage_stats("gain_db_post_wav_transition_smooth", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats)
                    if isinstance(st, dict):
                        st["wav_transition_smoothing"] = True
                        st["wav_transition_smoothing_zone_hz"] = [float(f_lo), float(f_hi)]
    except Exception:
        pass
    try:
        if "after_slope" not in stage_probes:
            _record_stage_probe(stage_probes, "after_slope", _stage_probe, freq_axis, gain_db, mask_c, cfg, logger)
    except Exception:
        pass

    max_cut_db = float(getattr(cfg, "max_cut_db", 15.0))
    max_cut_db = abs(float(getattr(cfg, "max_cut_db", 15.0) or 15.0))
    _record_stage_probe(stage_probes, "pre_hardclamp", _stage_probe, freq_axis, gain_db, mask_c, cfg, logger)
    try:
        _pre_hard = gain_db.copy()
        _max_boost2 = float(max_boost_db_base)
        _max_boost2_local = np.asarray(boost_cap_db, dtype=float)
        _max_cut2 = float(max_cut_db)
    except Exception:
        _pre_hard = gain_db
        _max_boost2, _max_cut2 = 0.0, float(max_cut_db)
        _max_boost2_local = np.full_like(gain_db, _max_boost2, dtype=float)
    gain_db = _apply_hard_boost_cut_clamp(
        gain_db,
        cfg,
        max_cut_db,
        boost_cap_db=boost_cap_db,
        mask=mask_c,
    )
    _record_stage_probe(stage_probes, "post_hardclamp", _stage_probe, freq_axis, gain_db, mask_c, cfg, logger)
    try:
        if np.any(mask_c):
            hardclamp_boost_bins = int(
                np.sum(
                    (_pre_hard[mask_c] > (_max_boost2_local[mask_c] + 1e-9))
                    & (gain_db[mask_c] <= (_max_boost2_local[mask_c] + 1e-9))
                )
            )
            hardclamp_cut_bins = int(np.sum((_pre_hard[mask_c] < (-_max_cut2 - 1e-9)) & (gain_db[mask_c] >= (-_max_cut2 - 1e-9))))
            hard_over_boost = max(0.0, float(np.max(_pre_hard[mask_c] - _max_boost2_local[mask_c])))
            hard_over_cut = max(0.0, float(np.max((-_pre_hard[mask_c]) - _max_cut2)))
            _band_bins = int(np.sum(mask_c))
        else:
            hardclamp_boost_bins, hardclamp_cut_bins = 0, 0
            hard_over_boost, hard_over_cut = 0.0, 0.0
            _band_bins = 0
        logger.info(
            "Clamp: hard_clamp "
            f"(max_boost_base={_max_boost2:.2f} dB, max_cut={_max_cut2:.2f} dB) -> "
            f"boost_clipped_bins={hardclamp_boost_bins}, cut_clipped_bins={hardclamp_cut_bins}, "
            f"worst_over_boost={hard_over_boost:.2f} dB, worst_over_cut={hard_over_cut:.2f} dB"
        )
        clipped_total = int(hardclamp_boost_bins + hardclamp_cut_bins)
        clip_pct = (100.0 * clipped_total / float(max(1, _band_bins)))
        over_peak = float(max(hard_over_boost, hard_over_cut))
        if over_peak >= 12.0 or clip_pct >= 15.0:
            clamp_dominance_level = "HIGH"
        elif over_peak >= 6.0 or clip_pct >= 5.0:
            clamp_dominance_level = "MEDIUM"
        elif clipped_total > 0:
            clamp_dominance_level = "LOW"
        else:
            clamp_dominance_level = "NONE"
        logger.info(
            "Clamp dominance: "
            f"{clamp_dominance_level} | "
            f"clipped={clipped_total}/{int(_band_bins)} ({clip_pct:.2f}%), "
            f"over_boost={hard_over_boost:.2f} dB, over_cut={hard_over_cut:.2f} dB"
            + (" | smoothing impact may be masked" if clamp_dominance_level != "NONE" else "")
        )
        try:
            if isinstance(st, dict):
                st["clamp_dominance_level"] = str(clamp_dominance_level)
                st["clamp_dominance_clip_pct"] = float(clip_pct)
                st["clamp_dominance_clipped_bins"] = int(clipped_total)
                st["clamp_dominance_band_bins"] = int(_band_bins)
        except Exception:
            pass
    except Exception:
        hardclamp_boost_bins, hardclamp_cut_bins = 0, 0
        hard_over_boost, hard_over_cut = 0.0, 0.0
        clamp_dominance_level = "NONE"
    try:
        _clamp_active = bool((hardclamp_boost_bins > 0) or (hardclamp_cut_bins > 0))
    except Exception:
        _clamp_active = False
    try:
        if _clamp_active and np.any(mask_c):
            mix = float(np.clip(float(getattr(cfg, "reg_strength", 30.0) or 30.0) / 100.0, 0.0, 1.0))
            if mix > 0.0:
                _pre = gain_db.copy()
                gain_db = _blend_masked_fractional_octave(
                    gain_db,
                    freq_axis,
                    mask_c,
                    smooth_value=_filter_smooth,
                    mix=mix,
                )
                gain_db = _apply_hard_boost_cut_clamp(
                    gain_db,
                    cfg,
                    max_cut_db,
                    boost_cap_db=boost_cap_db,
                    mask=mask_c,
                )
                _log_stage_stats("gain_db_post_final_clamp_smooth", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats)
    except Exception:
        pass
    try:
        if bool(getattr(cfg, "is_wav_source", False)) and np.any(mask_c):
            cmin = float(getattr(cfg, "mag_c_min", 0.0) or 0.0)
            cmax = float(getattr(cfg, "mag_c_max", 0.0) or 0.0)
            tw = float(getattr(cfg, "trans_width", 0.0) or 0.0)
            if np.isfinite(cmin) and np.isfinite(cmax) and (cmax > cmin):
                if not np.isfinite(tw) or tw <= 0.0:
                    tw = max(50.0, 0.4 * cmax)
                f_lo = max(cmin, cmax - 0.95 * tw)
                f_hi = min(float(np.max(freq_axis)), cmax + 1.45 * tw)
                zone = (freq_axis >= f_lo) & (freq_axis <= f_hi)
                if int(np.count_nonzero(zone)) >= 8:
                    _pre = gain_db.copy()
                    g0 = np.asarray(gain_db, dtype=float).copy()
                    sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=8.0, fallback_bins=12.0)
                    g_sm = scipy.ndimage.gaussian_filter1d(g0, sigma=float(max(2.0, sigma_bins)))
                    x = np.zeros_like(g0, dtype=float)
                    span = max(1e-9, float(f_hi - f_lo))
                    x[zone] = np.clip((freq_axis[zone] - f_lo) / span, 0.0, 1.0)
                    w = np.zeros_like(g0, dtype=float)
                    w[zone] = 0.5 - 0.5 * np.cos(np.pi * x[zone])
                    mix = 0.95
                    gain_db = g0 + (g_sm - g0) * (mix * w)
                    gain_db = _apply_hard_boost_cut_clamp(
                        gain_db,
                        cfg,
                        max_cut_db,
                        boost_cap_db=boost_cap_db,
                        mask=mask_c,
                    )
                    _log_stage_stats("gain_db_post_wav_final_ripple_polish", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats)
                    if isinstance(st, dict):
                        st["wav_final_ripple_polish"] = True
                        st["wav_final_ripple_polish_zone_hz"] = [float(f_lo), float(f_hi)]
    except Exception:
        pass

    # Ensure bass boost changes can propagate to final filter (post-limits domain),
    # instead of being fully neutralized by conf-pull/slope interactions.
    try:
        if bool(bass_boost_post_restore_enable) and np.any(mask_c):
            restore_lo = float(max(20.0, float(low_hz) + 1e-6))
            restore_hi = float(max(restore_lo, bass_boost_cap_hz))
            tgt_restore = np.asarray(gain_apply, dtype=float).copy()
            tgt_restore = np.clip(tgt_restore, -float(max_cut_db), np.asarray(boost_cap_db, dtype=float))
            _pre_post_restore = np.asarray(gain_db, dtype=float).copy()
            gain_db, _restore_meta = _apply_bass_boost_post_restore(
                gain_db,
                tgt_restore,
                boost_cap_db,
                freq_axis,
                mask_c,
                hz_lo=restore_lo,
                hz_hi=restore_hi,
                strength=float(bass_boost_post_restore_strength),
            )
            gain_db = _apply_hard_boost_cut_clamp(
                gain_db,
                cfg,
                max_cut_db,
                boost_cap_db=boost_cap_db,
                mask=mask_c,
            )
            if isinstance(st, dict):
                st["bass_boost_post_restore_applied"] = bool(_restore_meta.get("enabled", False))
                st["bass_boost_post_restore_bins"] = int(_restore_meta.get("bins", 0) or 0)
                st["bass_boost_post_restore_delta_rms_20_200"] = float(_restore_meta.get("delta_rms_20_200", 0.0) or 0.0)
                st["bass_boost_post_restore_delta_max_20_200"] = float(_restore_meta.get("delta_max_20_200", 0.0) or 0.0)
            _log_stage_stats(
                "gain_db_post_bass_boost_restore",
                gain_db,
                mask_c,
                ref=_pre_post_restore,
                logger=logger,
                enabled=debug_stage_stats,
            )
    except Exception:
        pass

    # Final hard reapply: low-bass cuts-only policy must survive all smoothing/clamps.
    try:
        if bool(low_cut_enable) and np.isfinite(float(low_hz)) and float(low_hz) > 0.0:
            low_mask_final = mask_c & (freq_axis > 0.0) & (freq_axis <= float(low_hz))
            if np.any(low_mask_final):
                _pre_lf = np.asarray(gain_db, dtype=float).copy()
                gain_db[low_mask_final] = np.minimum(gain_db[low_mask_final], 0.0)
                lf_floor_reapplied_bins = 0
                try:
                    if (
                        low_cut_strength > 0.0
                        and isinstance(low_cut_floor_ref, np.ndarray)
                        and low_cut_floor_ref.shape == gain_db.shape
                    ):
                        floor_vals = np.asarray(low_cut_floor_ref[low_mask_final], dtype=float)
                        valid_floor = np.isfinite(floor_vals)
                        if np.any(valid_floor):
                            cur = np.asarray(gain_db[low_mask_final], dtype=float)
                            cur_before = cur.copy()
                            cur[valid_floor] = np.minimum(cur[valid_floor], floor_vals[valid_floor])
                            gain_db[low_mask_final] = cur
                            lf_floor_reapplied_bins = int(
                                np.count_nonzero(cur_before[valid_floor] > (cur[valid_floor] + 1e-9))
                            )
                except Exception:
                    lf_floor_reapplied_bins = 0
                lf_boost_clamped_bins = int(
                    np.count_nonzero((_pre_lf[low_mask_final] > 1e-9) & (gain_db[low_mask_final] <= 1e-9))
                )
                if lf_boost_clamped_bins > 0 or lf_floor_reapplied_bins > 0:
                    logger.info(
                        "Low-bass hard lock reapply: "
                        f"clamped {lf_boost_clamped_bins} boost bins, "
                        f"reapplied cut floor on {lf_floor_reapplied_bins} bins <= {float(low_hz):.1f} Hz"
                    )
                try:
                    if isinstance(st, dict):
                        st["low_bass_hard_reapply_hz"] = float(low_hz)
                        st["low_bass_hard_reapply_bins"] = int(np.count_nonzero(low_mask_final))
                        st["low_bass_hard_reapply_clamped_bins"] = int(lf_boost_clamped_bins)
                        st["low_bass_hard_reapply_floor_bins"] = int(lf_floor_reapplied_bins)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        mag_c_min_fade = float(getattr(cfg, "mag_c_min", 0.0))
    except Exception:
        mag_c_min_fade = 0.0
    try:
        mag_c_max_fade = float(getattr(cfg, "mag_c_max", 0.0))
    except Exception:
        mag_c_max_fade = 0.0
    tw_raw = getattr(cfg, "trans_width", 100.0)
    try:
        trans_width_fade = 100.0 if tw_raw is None else float(tw_raw)
    except Exception:
        trans_width_fade = 100.0
    if not np.isfinite(trans_width_fade):
        trans_width_fade = 100.0

    f_start = max(mag_c_max_fade - trans_width_fade, mag_c_min_fade)
    f_mask = (freq_axis > f_start) & (freq_axis <= mag_c_max_fade)
    fade_len = mag_c_max_fade - f_start
    if np.any(f_mask) and fade_len > 0:
        x = (freq_axis[f_mask] - f_start) / fade_len
        w = _cosine_fade_out_01(x)
        gain_db[f_mask] *= w
        try:
            if isinstance(st, dict):
                st["mag_transition_fade_applied"] = True
                try:
                    band = (freq_axis >= (mag_c_max_fade - trans_width_fade)) & (freq_axis <= mag_c_max_fade)
                    if np.count_nonzero(band) >= 8:
                        g = np.asarray(gain_db, dtype=float)
                        f = np.asarray(freq_axis, dtype=float)
                        x_band = np.log2(np.maximum(f[band], 1e-9))
                        dg = np.diff(g[band])
                        dx = np.diff(x_band)
                        slope = dg / (dx + 1e-30)
                        st["mag_transition_slope_abs_max_db_per_oct"] = float(np.max(np.abs(slope)))
                except Exception:
                    pass
        except Exception:
            pass
    _record_stage_probe(stage_probes, "after_fade", _stage_probe, freq_axis, gain_db, mask_c, cfg, logger)

    try:
        summary = _summarize_correction_metrics(
            gain_db,
            freq_axis,
            cfg,
            st,
            mask_c,
            logger,
            boost_cand_peak=boost_cand_peak,
            n_boost_cand=n_boost_cand,
            n_boost_cand_low=n_boost_cand_low,
            n_boost_cand_exc=n_boost_cand_exc,
        )
        boost_peak_db = float(summary["boost_peak_db"])
        cut_peak_db = float(summary["cut_peak_db"])
        n_boost = int(summary["n_boost"])
    except Exception:
        boost_peak_db, cut_peak_db, n_boost = 0.0, 0.0, 0

    return _MagPostProcessOutputs(
        gain_db=np.asarray(gain_db, dtype=float),
        stage_probes=dict(stage_probes),
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
    )


def _run_mag_raw_stage(inputs: _MagPipelineInputs) -> _MagRawStageOutputs:
    """Core-vaihe osa 1: error, smoothing ja regularisointi."""

    cfg = inputs.cfg
    freq_axis = inputs.freq_axis
    st = inputs.st
    m_anal = inputs.m_anal
    target_mags = inputs.target_mags
    calc_offset_db = inputs.calc_offset_db
    logger = inputs.logger
    gain_db = inputs.gain_db

    debug_stage_stats = bool(getattr(cfg, "debug_stage_stats", True))
    afdw_on = False
    afdw_base = float(getattr(cfg, "fdw_cycles", 15.0))
    afdw_min = max(3.0, afdw_base / 3.0)
    _filter_smooth = None
    base_sigma = None
    df_mode = None
    raw_g = None
    final_g = None
    stage_probes: dict[str, Any] = {}

    if not bool(getattr(cfg, "enable_mag_correction", False)):
        return _MagRawStageOutputs(
            mag_enabled=False,
            debug_stage_stats=bool(debug_stage_stats),
            afdw_on=bool(afdw_on),
            afdw_base=float(afdw_base),
            afdw_min=float(afdw_min),
            filter_smooth=_filter_smooth,
            base_sigma=base_sigma,
            df_mode=df_mode,
            raw_g=raw_g,
            final_g=final_g,
            gain_db=np.asarray(gain_db, dtype=float),
            stage_probes=dict(stage_probes),
        )

    _filter_smooth = _resolve_filter_smooth(cfg)
    afdw_on = bool(getattr(cfg, "enable_afdw", False))
    afdw_base = float(getattr(cfg, "fdw_cycles", 15.0))
    afdw_min = max(3.0, afdw_base / 3.0)

    manual_target_bias_db = 0.0
    try:
        lvl_mode_s = str(getattr(cfg, "lvl_mode", "Auto") or "Auto").strip().lower()
        if "manual" in lvl_mode_s:
            # In manual level mode, apply the actual baseline-derived target shift.
            # This keeps gain/headroom behavior tied to the same reference shift
            # that leveling computed for the selected level window.
            st_shift = None
            if isinstance(st, dict):
                st_shift = st.get("target_shift_db", None)
            if st_shift is None:
                st_shift = getattr(cfg, "lvl_manual_db", 0.0)
            manual_target_bias_db = float(st_shift or 0.0)
            if not np.isfinite(manual_target_bias_db):
                manual_target_bias_db = 0.0
            if isinstance(st, dict):
                st["manual_target_bias_db"] = float(manual_target_bias_db)
    except Exception:
        manual_target_bias_db = 0.0

    # Base error (target - measured_aligned)
    err_db = _compute_error_db(
        m_anal - calc_offset_db,
        target_mags,
    )

    # Peak-priority error shaping should happen BEFORE mapping error -> correction,
    # so deep dips don't dominate when max_boost is small anyway.
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        try:
            fmin = float(getattr(cfg, "mag_c_min", 20.0) or 20.0)
            fmax = float(getattr(cfg, "mag_c_max", 200.0) or 200.0)
        except Exception:
            fmin, fmax = 20.0, 200.0
        mask_c_raw = (f >= float(fmin)) & (f <= float(fmax))
    except Exception:
        mask_c_raw = np.zeros_like(np.asarray(freq_axis, dtype=float), dtype=bool)

    err_db = _apply_peak_priority_error_shaping(
        err_db,
        np.asarray(freq_axis, dtype=float),
        cfg,
        st,
        mask_c=mask_c_raw,
        logger=logger,
    )

    raw_g = _error_to_correction_mag(err_db + float(manual_target_bias_db))
    try:
        # Maski on tassa vaiheessa viela tyhja; logiikka pidetaan ennallaan.
        mm = np.zeros_like(freq_axis, dtype=bool)
        if mm is not None and np.any(mm):
            dv = raw_g[mm]
            logger.info(f"RAW_G(mask): max={float(np.max(dv)):.3f} min={float(np.min(dv)):.3f} rms={float(np.sqrt(np.mean(dv*dv))):.3f}")
    except Exception:
        pass
    _log_stage_stats("raw_g_pre_confpull", raw_g, np.zeros_like(freq_axis, dtype=bool), logger=logger, enabled=debug_stage_stats)

    base_sigma = 60 // (_filter_smooth / 12 if _filter_smooth > 0 else 1)
    df_mode = bool(getattr(cfg, "df_smoothing", False))
    sm_g = _apply_smoothing(
        raw_g,
        freq_axis,
        cfg,
        st,
        filter_smooth=_filter_smooth,
        df_mode=df_mode,
        conf_mask=inputs.conf_mask,
    )
    final_g = _apply_regularization(raw_g, freq_axis, cfg, st, sm_g)

    return _MagRawStageOutputs(
        mag_enabled=True,
        debug_stage_stats=bool(debug_stage_stats),
        afdw_on=bool(afdw_on),
        afdw_base=float(afdw_base),
        afdw_min=float(afdw_min),
        filter_smooth=_filter_smooth,
        base_sigma=base_sigma,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        gain_db=np.asarray(gain_db, dtype=float),
        stage_probes=dict(stage_probes),
    )


def _run_mag_bassfirst_afdw_conf_stage(
    inputs: _MagPipelineInputs,
    raw_stage: _MagRawStageOutputs,
) -> _MagAdaptiveStageOutputs:
    """Core-vaihe osa 2: bassfirst/afdw, confidence-logiikka ja after_gain_apply-probe."""

    cfg = inputs.cfg
    freq_axis = inputs.freq_axis
    st = inputs.st
    m_interp = inputs.m_interp
    conf_mask = inputs.conf_mask
    complex_meas = inputs.complex_meas
    logger = inputs.logger
    analysis_mode = inputs.analysis_mode
    cmp = inputs.cmp
    _stage_probe = inputs.stage_probe

    gain_db = np.asarray(raw_stage.gain_db, dtype=float).copy()
    final_g = np.asarray(raw_stage.final_g, dtype=float).copy()
    stage_probes = dict(raw_stage.stage_probes)

    use_bassfirst = bool(getattr(cfg, "bass_first_ai", False))
    bf_room_mode = None
    bf_rel = None
    bf_conf_for_smoothing = None
    if use_bassfirst:
        try:
            ph_u = np.unwrap(np.angle(complex_meas))
            df = np.gradient(freq_axis) + 1e-12
            gd_ms_local = (-np.gradient(ph_u) / (2*np.pi*df)) * 1000.0
            try:
                _gd_sigma_hz = float(getattr(cfg, "bass_first_gd_sigma_hz", 2.0) or 2.0)
            except Exception:
                _gd_sigma_hz = 2.0
            if not np.isfinite(_gd_sigma_hz) or _gd_sigma_hz <= 0.0:
                _gd_sigma_hz = 2.0
            sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=float(_gd_sigma_hz), fallback_bins=20.0)
            gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms_local, sigma=float(sigma_bins))
            gd_diff_local = np.abs(gd_ms_local - gd_smooth)
            _bf_mode_f2 = float(getattr(cfg, "bass_first_mode_max_hz", 200.0) or 200.0)
            _win_mode = "auto"
            _left_ms = 0.0
            try:
                _win_mode = str(getattr(cfg, "ir_export_window_mode", "auto") or "auto").strip().lower()
                _left_ms = float(getattr(cfg, "ir_window_left", getattr(cfg, "ir_window_ms_left", 0.0)) or 0.0)
                if _win_mode == "rew_asym" and _left_ms < 15.0:
                    _bf_mode_f2 = min(_bf_mode_f2, 80.0)
                    logger.info(f"REW Asym low-latency: left_ms={_left_ms:.1f} -> bass-first limited to {float(_bf_mode_f2):.0f} Hz")
            except Exception:
                pass
            bf_rel, bf_room_mode, _ = bf.build_bassfirst_masks(
                freq_axis=freq_axis,
                m_raw_db=m_interp,
                phase_rad_unwrapped=ph_u,
                gd_ms=gd_ms_local,
                gd_diff=gd_diff_local,
                is_wav_source=bool(getattr(cfg, "is_wav_source", False)),
                mode_f2=_bf_mode_f2,
                rew_asym=(_win_mode == "rew_asym"),
                left_ms=_left_ms,
            )
            bf_conf_for_smoothing = bf.fuse_conf_for_smoothing(
                freq_axis=freq_axis,
                reliability_mask=bf_rel,
                bass_floor_lo=float(getattr(cfg, "bass_first_smooth_floor_lo", 0.75) or 0.75),
                bass_floor_hi=float(getattr(cfg, "bass_first_smooth_floor_hi", 0.35) or 0.35),
            )
        except Exception:
            bf_rel = bf_room_mode = bf_conf_for_smoothing = None

    if raw_stage.afdw_on:
        try:
            conf_for_afdw = (bf_conf_for_smoothing if (use_bassfirst and bf_conf_for_smoothing is not None) else conf_mask)
            c = np.clip(conf_for_afdw, 0.0, 1.0)
            adaptive_cycles = float(raw_stage.afdw_min) + (c * (float(raw_stage.afdw_base) - float(raw_stage.afdw_min)))
            bw = 2.0 / np.maximum(adaptive_cycles, 1.0)
            bw = np.clip(bw, AFDW_BW_MIN_OCT, AFDW_BW_MAX_OCT)
            afdw_bw_oct = bw
            afdw_bw_min_oct = float(np.min(bw))
            afdw_bw_mean_oct = float(np.mean(bw))
            afdw_bw_max_oct = float(np.max(bw))
            bw_min_idx = np.where(bw == np.min(bw))[0]
            bw_max_idx = np.where(bw == np.max(bw))[0]
            afdw_bw_min_hz = float(freq_axis[int(bw_min_idx[len(bw_min_idx)//2])])
            afdw_bw_max_hz = float(freq_axis[int(bw_max_idx[len(bw_max_idx)//2])])
            # Kuvaajalle lasketaan erillinen BW raw acoustic confidence-maskista.
            bw_plot = np.asarray(afdw_bw_oct, dtype=float)
            try:
                c_plot = np.clip(np.asarray(conf_mask, dtype=float), 0.0, 1.0)
                if c_plot.shape == np.asarray(freq_axis, dtype=float).shape:
                    adaptive_cycles_plot = float(raw_stage.afdw_min) + (c_plot * (float(raw_stage.afdw_base) - float(raw_stage.afdw_min)))
                    bw_plot = 2.0 / np.maximum(adaptive_cycles_plot, 1.0)
                    bw_plot = np.clip(bw_plot, AFDW_BW_MIN_OCT, AFDW_BW_MAX_OCT)
            except Exception:
                bw_plot = np.asarray(afdw_bw_oct, dtype=float)
            # Prosessointi- ja kuvaajadata tallennetaan erikseen regressioiden estamiseksi.
            try:
                if isinstance(st, dict):
                    st["afdw_bw_oct"] = np.asarray(afdw_bw_oct, dtype=float).tolist()
                    st["afdw_bw_plot_oct"] = np.asarray(bw_plot, dtype=float).tolist()
                    try:
                        if str(analysis_mode).lower() == "comparison":
                            _fx_cmp = None
                            _cmp = cmp
                            if isinstance(_cmp, dict):
                                _fx_cmp = _cmp.get("cmp_freq_axis", None)
                            if _fx_cmp is None:
                                _fx_cmp = st.get("cmp_freq_axis", None)
                            if _fx_cmp is not None:
                                fx_cmp = np.asarray(_fx_cmp, dtype=float)
                                bw_cmp = np.interp(fx_cmp, freq_axis, np.asarray(afdw_bw_oct, dtype=float))
                                st["cmp_afdw_bw_oct"] = np.asarray(bw_cmp, dtype=float).tolist()
                                bw_cmp_plot = np.interp(fx_cmp, freq_axis, np.asarray(bw_plot, dtype=float))
                                st["cmp_afdw_bw_plot_oct"] = np.asarray(bw_cmp_plot, dtype=float).tolist()
                    except Exception:
                        pass
                    st["afdw_bw_min_oct"] = float(afdw_bw_min_oct)
                    st["afdw_bw_mean_oct"] = float(afdw_bw_mean_oct)
                    st["afdw_bw_max_oct"] = float(afdw_bw_max_oct)
                    st["afdw_bw_min_hz"] = float(afdw_bw_min_hz)
                    st["afdw_bw_max_hz"] = float(afdw_bw_max_hz)
            except Exception:
                pass
        except Exception:
            pass
        final_g = apply_adaptive_fdw(
            freq_axis,
            final_g,
            (bf_conf_for_smoothing if (use_bassfirst and bf_conf_for_smoothing is not None) else conf_mask),
            base_cycles=float(raw_stage.afdw_base),
            min_cycles=float(raw_stage.afdw_min)
        )
        try:
            if isinstance(st, dict):
                st["fdw_mode"] = "adaptive"
        except Exception:
            pass
    else:
        # A-FDW OFF => apply fixed FDW using configured cycles.
        # This keeps response predictable and avoids narrow spike/notch artifacts.
        fixed_cycles = float(max(1.0, raw_stage.afdw_base))
        final_g = apply_adaptive_fdw(
            freq_axis,
            final_g,
            np.ones_like(freq_axis, dtype=float),
            base_cycles=fixed_cycles,
            min_cycles=fixed_cycles,
        )
        try:
            bw_fixed_oct = float(np.clip(2.0 / max(fixed_cycles, 1.0), AFDW_BW_MIN_OCT, AFDW_BW_MAX_OCT))
            if isinstance(st, dict):
                st["fdw_mode"] = "fixed"
                st["fdw_fixed_cycles"] = float(fixed_cycles)
                st["fdw_fixed_bw_oct"] = float(bw_fixed_oct)
            logger.info(
                "FDW fixed smoothing (A-FDW OFF): "
                f"cycles={fixed_cycles:.2f}, bw={bw_fixed_oct:.4f} oct"
            )
        except Exception:
            pass

    mask_c, _active_band = _select_active_band(freq_axis, cfg)
    raw_safe_ref = None
    try:
        g0 = np.asarray(raw_stage.raw_g, dtype=float).copy()
        idx = np.where(mask_c)[0]
        if idx.size >= 2:
            i0, i1 = int(idx[0]), int(idx[-1])
            if i0 > 0:
                g0[:i0] = g0[i0]
            if i1 < (g0.size - 1):
                g0[i1+1:] = g0[i1]
        raw_safe_ref = psycho_smooth_safe_gain(freq_axis, g0)
        raw_safe_ref = np.where(mask_c, np.asarray(raw_safe_ref, dtype=float), 0.0)
    except Exception:
        raw_safe_ref = None

    if use_bassfirst and bf_room_mode is not None:
        try:
            final_g = bf.modulate_gain_bassfirst(
                final_g, bf_room_mode,
                k_mode_cut=float(getattr(cfg, "bass_first_k_mode_cut", 0.6) or 0.6),
                k_mode_boost=float(getattr(cfg, "bass_first_k_mode_boost", 0.9) or 0.9),
            )
        except Exception:
            pass

    _pre_bass_adapt_g = np.asarray(final_g, dtype=float).copy()
    try:
        _pre_bass_adapt = np.asarray(final_g, dtype=float).copy()
        conf_for_bass_adapt, conf_for_bass_adapt_src = _select_bass_adaptive_conf_mask(
            conf_mask=conf_mask,
            bf_conf_for_smoothing=bf_conf_for_smoothing,
            use_bassfirst=bool(use_bassfirst),
        )
        try:
            if isinstance(st, dict):
                st["bass_adaptive_conf_source"] = str(conf_for_bass_adapt_src)
        except Exception:
            pass
        final_g = _apply_confidence_adaptive_bass_smoothing(
            final_g,
            freq_axis,
            cfg,
            st,
            conf_for_bass_adapt,
            stage_tag="core",
        )
        _pre_bass_adapt_g = _pre_bass_adapt
        _log_stage_stats(
            "final_g_post_bass_adaptive",
            final_g,
            mask_c,
            ref=_pre_bass_adapt,
            logger=logger,
            enabled=raw_stage.debug_stage_stats,
        )
    except Exception:
        pass

    gain_apply, conf_debug = _apply_confidence_logic(final_g, freq_axis, cfg, st, conf_mask)
    try:
        if isinstance(st, dict):
            st["conf_logic_mode"] = str(conf_debug.get("mode", "unknown"))
    except Exception:
        pass
    _gain_apply_pre_limits = gain_apply.copy()
    _log_stage_stats("gain_apply_pre_limits", gain_apply, mask_c, logger=logger, enabled=raw_stage.debug_stage_stats)
    try:
        _win_mode = str(getattr(cfg, "ir_export_window_mode", "auto") or "auto").strip().lower()
        _left_ms  = float(getattr(cfg, "ir_window_left", getattr(cfg, "ir_window_ms_left", 0.0)) or 0.0)
        if _win_mode == "rew_asym" and _left_ms < 10.0:
            _hz = 120.0
            _m = mask_c & (freq_axis > 0.0) & (freq_axis <= _hz)
            if np.any(_m):
                gain_apply[_m] = np.minimum(gain_apply[_m], 0.0)
                logger.info(f"REW Asym safety: left_ms={_left_ms:.1f} -> no LF boost below {_hz:.0f} Hz")
    except Exception:
        pass
    try:
        _tmp_after_apply = np.zeros_like(gain_db, dtype=float)
        _tmp_after_apply[mask_c] = gain_apply[mask_c]
        _record_stage_probe(stage_probes, "after_gain_apply", _stage_probe, freq_axis, _tmp_after_apply, mask_c, cfg, logger)
    except Exception:
        pass

    return _MagAdaptiveStageOutputs(
        final_g=np.asarray(final_g, dtype=float),
        mask_c=np.asarray(mask_c, dtype=bool),
        stage_probes=dict(stage_probes),
        use_bassfirst=bool(use_bassfirst),
        bf_room_mode=bf_room_mode,
        bf_rel=bf_rel,
        bf_conf_for_smoothing=bf_conf_for_smoothing,
        pre_bass_adapt_g=np.asarray(_pre_bass_adapt_g, dtype=float),
        gain_db=np.asarray(gain_db, dtype=float),
        gain_apply=np.asarray(gain_apply, dtype=float),
        raw_safe_ref=raw_safe_ref,
    )


def _run_mag_core_stage(inputs: _MagPipelineInputs) -> _MagCoreOutputs:
    """Orkestroi core-vaiheen kahtena perakkaisena alipolkuna."""

    raw_stage = _run_mag_raw_stage(inputs)
    if not raw_stage.mag_enabled:
        return _MagCoreOutputs(
            mag_enabled=False,
            debug_stage_stats=bool(raw_stage.debug_stage_stats),
            afdw_on=bool(raw_stage.afdw_on),
            base_sigma=raw_stage.base_sigma,
            filter_smooth=raw_stage.filter_smooth,
            df_mode=raw_stage.df_mode,
            raw_g=raw_stage.raw_g,
            final_g=raw_stage.final_g,
            mask_c=np.zeros_like(inputs.freq_axis, dtype=bool),
            stage_probes=dict(raw_stage.stage_probes),
            use_bassfirst=False,
            bf_room_mode=None,
            bf_rel=None,
            bf_conf_for_smoothing=None,
            pre_bass_adapt_g=None,
            gain_db=np.asarray(raw_stage.gain_db, dtype=float),
            gain_apply=np.zeros_like(inputs.freq_axis, dtype=float),
            raw_safe_ref=None,
        )

    adaptive_stage = _run_mag_bassfirst_afdw_conf_stage(inputs, raw_stage)
    return _MagCoreOutputs(
        mag_enabled=True,
        debug_stage_stats=bool(raw_stage.debug_stage_stats),
        afdw_on=bool(raw_stage.afdw_on),
        base_sigma=raw_stage.base_sigma,
        filter_smooth=raw_stage.filter_smooth,
        df_mode=raw_stage.df_mode,
        raw_g=raw_stage.raw_g,
        final_g=adaptive_stage.final_g,
        mask_c=np.asarray(adaptive_stage.mask_c, dtype=bool),
        stage_probes=dict(adaptive_stage.stage_probes),
        use_bassfirst=bool(adaptive_stage.use_bassfirst),
        bf_room_mode=adaptive_stage.bf_room_mode,
        bf_rel=adaptive_stage.bf_rel,
        bf_conf_for_smoothing=adaptive_stage.bf_conf_for_smoothing,
        pre_bass_adapt_g=adaptive_stage.pre_bass_adapt_g,
        gain_db=np.asarray(adaptive_stage.gain_db, dtype=float),
        gain_apply=np.asarray(adaptive_stage.gain_apply, dtype=float),
        raw_safe_ref=adaptive_stage.raw_safe_ref,
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

        # Bass-adaptive delta metrics should reflect final, exported gain curve.
        # Current core-stage delta is kept as debug, but report keys are overwritten
        # with the effective post-limits/post-clamp impact.
        try:
            if isinstance(st, dict):
                d_rms_core = float(st.get("bass_adaptive_smoothing_delta_rms_db_20_200", 0.0) or 0.0)
                d_max_core = float(st.get("bass_adaptive_smoothing_delta_max_db_20_200", 0.0) or 0.0)
            else:
                d_rms_core = 0.0
                d_max_core = 0.0
        except Exception:
            d_rms_core = 0.0
            d_max_core = 0.0

        try:
            pre_bass = np.asarray(core.pre_bass_adapt_g, dtype=float) if core.pre_bass_adapt_g is not None else None
        except Exception:
            pre_bass = None

        try:
            bass_enabled = bool(st.get("bass_adaptive_smoothing_enabled", False)) if isinstance(st, dict) else False
        except Exception:
            bass_enabled = False

        if bass_enabled and pre_bass is not None and pre_bass.shape == np.asarray(final_g, dtype=float).shape:
            try:
                class _NullLogger:
                    def info(self, *args, **kwargs):
                        return None

                def _noop_stage_probe(*args, **kwargs):
                    return {}

                shadow_st: dict[str, Any] = {}
                gain_apply_shadow, _ = _apply_confidence_logic(pre_bass, freq_axis, cfg, shadow_st, conf_mask)
                shadow_post = _apply_post_limits_and_metrics(
                    _MagPostProcessInputs(
                        cfg=cfg,
                        freq_axis=freq_axis,
                        st=shadow_st,
                        logger=_NullLogger(),
                        stage_probe=_noop_stage_probe,
                        cfg_float_allow_zero=_cfg_float_allow_zero,
                        mask_c=mask_c,
                        gain_db=np.asarray(core.gain_db, dtype=float).copy(),
                        gain_apply=np.asarray(gain_apply_shadow, dtype=float),
                        raw_g=np.asarray(raw_g, dtype=float),
                        final_g=np.asarray(pre_bass, dtype=float),
                        raw_safe_ref=core.raw_safe_ref,
                        conf_mask=conf_mask,
                        filter_smooth=_filter_smooth,
                        debug_stage_stats=False,
                        stage_probes={},
                        apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
                        m_anal=np.asarray(inputs.m_anal, dtype=float),
                        target_mags=np.asarray(inputs.target_mags, dtype=float),
                        calc_offset_db=float(inputs.calc_offset_db),
                    )
                )
                d_rms_eff, d_max_eff, d_max_hz_eff = _band_delta_metrics(
                    np.asarray(post.gain_db, dtype=float),
                    np.asarray(shadow_post.gain_db, dtype=float),
                    np.asarray(freq_axis, dtype=float),
                    f_lo=20.0,
                    f_hi=200.0,
                )
                if isinstance(st, dict):
                    st["bass_adaptive_smoothing_delta_rms_db_20_200_core"] = float(d_rms_core)
                    st["bass_adaptive_smoothing_delta_max_db_20_200_core"] = float(d_max_core)
                    st["bass_adaptive_smoothing_delta_rms_db_20_200"] = float(d_rms_eff)
                    st["bass_adaptive_smoothing_delta_max_db_20_200"] = float(d_max_eff)
                    st["bass_adaptive_smoothing_delta_max_hz_20_200"] = (
                        float(d_max_hz_eff)
                        if (d_max_hz_eff is not None and np.isfinite(float(d_max_hz_eff)))
                        else None
                    )
                    st["bass_adaptive_smoothing_delta_basis"] = "mag_post_limits_pre_ir"
            except Exception:
                if isinstance(st, dict):
                    st["bass_adaptive_smoothing_delta_max_hz_20_200"] = None
                    st["bass_adaptive_smoothing_delta_basis"] = "core_stage_fallback"

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
