from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .camillafir_analysis import calculate_rt60, calculate_rt60_bands
from .camillafir_leveling import compute_leveling
from .correction_types import (
    BaselineComparisonTelemetry,
    BaselineNativeTelemetry,
    _BaselineContext,
)
from .limits import build_slope_limit_envelope
from .phase import get_min_phase_impulse
from .tdc import apply_smart_tdc

def apply_null_guard_target(
    freq_axis: np.ndarray,
    target_mags: np.ndarray,
    measured_mags: np.ndarray,
    *,
    mag_c_min: float,
    mag_c_max: float,
    enable: bool = True,
    depth_db: float = 12.0,          # min dip depth to start guarding
    max_blend: float = 0.85,         # max blend toward measured at dip center
    max_total_relax_db: float = 12.0,# cap how much target can be relaxed downward vs original
    smooth_oct: float = 0.18,        # gaussian smoothing width (in octaves) for the mask
) -> np.ndarray:
    """
    Makes target more realistic by relaxing deep, narrow cancellation nulls.
    We detect dips relative to a smoothed "envelope" of the measured response,
    then blend target toward measured at those frequencies.
    """
    if not enable:
        return target_mags
    f = np.asarray(freq_axis, dtype=float)
    t = np.asarray(target_mags, dtype=float)
    m = np.asarray(measured_mags, dtype=float)
    if f.size != t.size or f.size != m.size or f.size < 32:
        return t

    # active correction band
    try:
        fmin = float(mag_c_min or 0.0)
        fmax = float(mag_c_max or 0.0)
    except (TypeError, ValueError, OverflowError):
        fmin, fmax = 0.0, 0.0
    if not (np.isfinite(fmin) and np.isfinite(fmax) and fmax > fmin and fmax > 0):
        # if no band set, do nothing (safer default)
        return t

    band = (f >= fmin) & (f <= fmax)
    if np.count_nonzero(band) < 16:
        return t

    # log-frequency axis for smoothing
    lf = np.log2(np.clip(f, 1e-3, None))
    # estimate median log-step
    dlf = np.diff(lf[band])
    dlf = dlf[np.isfinite(dlf) & (dlf > 0)]
    if dlf.size < 8:
        return t
    step = float(np.median(dlf))

    # build a smooth "envelope" of measured response in-band
    # (gaussian smoothing in log-frequency domain)
    sigma_bins = max(1.0, float(smooth_oct) / max(step, 1e-6))
    env = m.copy()
    # simple gaussian smoothing without scipy: do it via convolution
    # kernel length ~ 6*sigma
    half = int(np.ceil(3.0 * sigma_bins))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= np.sum(k)
    env_band = np.convolve(env[band], k, mode="same")
    env[band] = env_band

    # dip depth relative to envelope
    dip = env - m  # positive where measured is below envelope
    dip = np.where(band, dip, 0.0)

    # soft mask based on depth
    try:
        d0 = float(depth_db)
    except (TypeError, ValueError, OverflowError):
        d0 = 12.0
    d0 = float(np.clip(d0, 3.0, 30.0))

    # weight grows from 0 at depth=d0 to 1 at ~2*d0
    w = np.clip((dip - d0) / max(d0, 1e-6), 0.0, 1.0)
    # sharpen slightly so only true nulls get high weight
    w = w**1.4

    # smooth the mask too (prevents zipper artifacts)
    w_band = np.convolve(w[band], k, mode="same")
    w2 = np.zeros_like(w)
    w2[band] = np.clip(w_band, 0.0, 1.0)

    try:
        mb = float(max_blend)
    except (TypeError, ValueError, OverflowError):
        mb = 0.85
    mb = float(np.clip(mb, 0.0, 1.0))
    w2 *= mb

    # blend target toward measured (only where mask > 0)
    t0 = t.copy()
    t_new = (1.0 - w2) * t + w2 * m

    # cap total downward relaxation vs original target (avoid over-darkening)
    try:
        cap = float(max_total_relax_db)
    except (TypeError, ValueError, OverflowError):
        cap = 12.0
    cap = float(np.clip(cap, 0.0, 30.0))
    # do not allow target to drop more than 'cap' dB below original target at any bin
    t_new = np.maximum(t_new, t0 - cap)

    return t_new
def _resample_or_interpolate_to_axis(
    meas_freq: np.ndarray,
    meas_values: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """Yhtenainen interpolointikaare, jotta akselisiirrot ovat yhdessa paikassa."""
    return np.interp(axis, meas_freq, meas_values)


def _build_target_curve(
    freq_axis: np.ndarray,
    cfg: Any,
    interpolate_response: Callable[..., np.ndarray],
) -> np.ndarray:
    """Rakentaa tavoitekayran (house curve) nykyisen ehdollisen logiikan mukaan."""

    if (
        cfg.house_freqs is not None
        and cfg.house_mags is not None
        and len(cfg.house_freqs) >= 2
        and len(cfg.house_mags) >= 2
    ):
        return interpolate_response(cfg.house_freqs, cfg.house_mags, freq_axis)
    return np.zeros_like(freq_axis, dtype=float)


def _apply_target_preview_adjustments(target_mag_db: np.ndarray, cfg: Any) -> np.ndarray:
    """Paikka UI-preview-saadolle; nyt identiteetti, jotta kaytos sailyy ennallaan."""

    return target_mag_db


def _prepare_correction_baseline(
    *,
    cfg,
    freq_axis,
    f_in,
    m_in,
    reflections,
    st,
    m_anal,
    m_plot_db,
    is_psy,
    cmp,
    analysis_mode,
    gain_db,
    logger,
    interpolate_response,
    _cfg_float_allow_zero,
    stereo_link_ctx=None,
) -> _BaselineContext:
    m_interp_for_rt = _resample_or_interpolate_to_axis(f_in, m_in, freq_axis)
    m_rt_lin = _resample_or_interpolate_to_axis(freq_axis, m_interp_for_rt, np.linspace(0, cfg.fs / 2, 65537))
    rt_ir = get_min_phase_impulse(m_rt_lin, 131072)
    current_rt60 = calculate_rt60(rt_ir, cfg.fs)
    rt60_bands = calculate_rt60_bands(rt_ir, cfg.fs, f_min=31.5, f_max=8000.0, order=4)
    band_avg = 0.0
    if rt60_bands:
        ks = np.array(sorted(rt60_bands.keys()), dtype=float)
        vs = np.array([rt60_bands[k] for k in ks], dtype=float)
        mid = (ks >= 125.0) & (ks <= 4000.0) & (vs > 0.05) & (vs < 5.0)
        if np.any(mid):
            band_avg = float(np.median(vs[mid]))
        else:
            band_avg = float(np.median(vs))

    # B) Target ja mahdollinen preview-saato (nyt no-op)
    target_mags = _build_target_curve(freq_axis, cfg, interpolate_response)
    target_mags = _apply_target_preview_adjustments(target_mags, cfg)

    if cfg.enable_tdc:
        rt60_for_tdc = rt60_bands if rt60_bands else current_rt60
        tdc_strength = _cfg_float_allow_zero(cfg, "tdc_strength", 50.0)
        tdc_max_red = _cfg_float_allow_zero(cfg, "tdc_max_reduction_db", 9.0)
        tdc_slope = _cfg_float_allow_zero(cfg, "tdc_slope_db_per_oct", 0.0)
        if tdc_strength < 0:
            tdc_strength = 0.0
        if tdc_strength > 100:
            tdc_strength = 100.0
        if tdc_max_red < 0:
            tdc_max_red = 0.0
        if tdc_max_red > 24:
            tdc_max_red = 24.0
        if tdc_slope < 0:
            tdc_slope = 0.0
        if tdc_slope > 24:
            tdc_slope = 24.0
        target_mags = apply_smart_tdc(
            freq_axis,
            target_mags,
            reflections,
            rt60_for_tdc,
            tdc_strength / 100.0,
            max_total_reduction_db=tdc_max_red,
            max_slope_db_per_oct=tdc_slope,
        )
    # ---- Null-guard: make target realistic for deep/narrow cancellations ----
    try:
        ng_enable = bool(getattr(cfg, "enable_null_guard", True))
    except (AttributeError, TypeError, ValueError):
        ng_enable = True

    if ng_enable:
        try:
            mag_c_min = float(getattr(cfg, "mag_c_min", 0.0) or 0.0)
            mag_c_max = float(getattr(cfg, "mag_c_max", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            mag_c_min, mag_c_max = 0.0, 0.0

        # use analysis magnitude (m_anal) already on freq_axis scale
        target_mags = apply_null_guard_target(
            np.asarray(freq_axis, dtype=float),
            np.asarray(target_mags, dtype=float),
            np.asarray(m_anal, dtype=float),
            mag_c_min=mag_c_min,
            mag_c_max=mag_c_max,
            enable=True,
            depth_db=float(getattr(cfg, "null_guard_depth_db", 12.0) or 12.0),
            max_blend=float(getattr(cfg, "null_guard_max_blend", 0.85) or 0.85),
            max_total_relax_db=float(getattr(cfg, "null_guard_max_relax_db", 12.0) or 12.0),
            smooth_oct=float(getattr(cfg, "null_guard_smooth_oct", 0.18) or 0.18),
        )
    hpf_f = 0.0
    hpf_order = 0
    if cfg.hpf_settings and cfg.hpf_settings.get("enabled"):
        hpf_f = float(cfg.hpf_settings.get("freq", 0.0) or 0.0)
        hpf_order = int(cfg.hpf_settings.get("order", 0) or 0)
    target_level_db = 0.0
    calc_offset_db = 0.0
    meas_level_db_window = 0.0
    target_level_db_window = 0.0
    offset_method = "Init"
    shared_target_shift_db = None
    if stereo_link_ctx is not None:
        try:
            v = getattr(stereo_link_ctx, "shared_target_shift_db", None)
            if v is not None:
                v = float(v)
                if np.isfinite(v):
                    shared_target_shift_db = float(v)
        except (AttributeError, TypeError, ValueError):
            shared_target_shift_db = None
    try:
        s_min = float(getattr(cfg, "lvl_min", 500.0) or 500.0)
        s_max = float(getattr(cfg, "lvl_max", 2000.0) or 2000.0)
    except (AttributeError, TypeError, ValueError):
        s_min, s_max = 500.0, 2000.0
    try:
        (
            target_level_db,
            calc_offset_db,
            meas_level_db_window,
            target_level_db_window,
            offset_method,
            s_min,
            s_max,
        ) = compute_leveling(
            cfg,
            np.asarray(freq_axis, dtype=float),
            np.asarray(m_anal, dtype=float),
            np.asarray(target_mags, dtype=float),
            stereo_link_ctx=stereo_link_ctx,
        )
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass
    target_shift_db = 0.0
    try:
        f = np.asarray(freq_axis, dtype=float)
        t = np.asarray(target_mags, dtype=float)
        if f.size == t.size and f.size > 16:
            if shared_target_shift_db is not None:
                target_shift_db = float(shared_target_shift_db)
                target_mags = t + target_shift_db
                (
                    target_level_db,
                    calc_offset_db,
                    meas_level_db_window,
                    target_level_db_window,
                    offset_method,
                    s_min,
                    s_max,
                ) = compute_leveling(cfg, f, m_anal, target_mags, stereo_link_ctx=stereo_link_ctx)
            elif np.isfinite(float(s_min)) and np.isfinite(float(s_max)):
                mask_lvl = (f >= float(s_min)) & (f <= float(s_max))
                if int(np.count_nonzero(mask_lvl)) > 10:
                    tgt_win_mean = float(np.mean(t[mask_lvl]))
                    if np.isfinite(tgt_win_mean) and np.isfinite(float(target_level_db)):
                        target_shift_db = float(target_level_db) - tgt_win_mean
                        target_mags = t + target_shift_db
                        (
                            target_level_db,
                            calc_offset_db,
                            meas_level_db_window,
                            target_level_db_window,
                            offset_method,
                            s_min,
                            s_max,
                        ) = compute_leveling(cfg, f, m_anal, target_mags, stereo_link_ctx=stereo_link_ctx)
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        target_shift_db = 0.0
    native_telemetry: BaselineNativeTelemetry | None = None
    try:
        m_src = np.asarray(m_anal, dtype=float)
        if is_psy and (m_plot_db is not None):
            mp = np.asarray(m_plot_db, dtype=float)
            if mp.size == m_src.size:
                m_src = mp
        measured_aligned = m_src - float(calc_offset_db)
        target_aligned = np.asarray(target_mags, dtype=float) - float(target_level_db)
        env_lo = env_hi = None
        env_pivot = None
        try:
            max_slope = float(getattr(cfg, "max_slope_db_per_oct", 0.0) or 0.0)
            max_slope_boost = float(getattr(cfg, "max_slope_boost_db_per_oct", 0.0) or 0.0)
            max_slope_cut = float(getattr(cfg, "max_slope_cut_db_per_oct", 0.0) or 0.0)
            if max_slope_boost <= 0.0:
                max_slope_boost = max_slope
            if max_slope_cut <= 0.0:
                max_slope_cut = max_slope
            env_lo, env_hi, env_pivot = build_slope_limit_envelope(
                np.asarray(freq_axis, dtype=float),
                target_aligned,
                mag_c_min=float(getattr(cfg, "mag_c_min", 0.0) or 0.0),
                mag_c_max=float(getattr(cfg, "mag_c_max", 0.0) or 0.0),
                max_slope_boost_db_per_oct=float(max_slope_boost),
                max_slope_cut_db_per_oct=float(max_slope_cut),
            )
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
            env_lo = env_hi = None
            env_pivot = None
        native_telemetry = BaselineNativeTelemetry(
            analysis_mode="native",
            freq_axis=np.asarray(freq_axis, dtype=float),
            measured_mags=np.asarray(measured_aligned, dtype=float),
            target_mags=np.asarray(target_aligned, dtype=float),
            target_env_lo=None if env_lo is None else np.asarray(env_lo, dtype=float),
            target_env_hi=None if env_hi is None else np.asarray(env_hi, dtype=float),
            target_env_pivot_hz=float(env_pivot) if env_pivot is not None else None,
            target_shift_db=float(target_shift_db),
            eff_target_db=float(target_level_db),
            target_level_db_window=float(target_level_db_window),
            meas_level_db_window=float(meas_level_db_window),
            offset_db=float(calc_offset_db),
            offset_method=str(offset_method),
            smart_scan_range=(float(s_min), float(s_max)),
        )
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        native_telemetry = None

    comparison_telemetry: BaselineComparisonTelemetry | None = None
    try:
        if isinstance(cmp, dict) and analysis_mode == "comparison":
            freq_cmp = np.asarray(cmp.get("cmp_freq_axis", []) or [], dtype=float)
            if freq_cmp.size > 8:
                m_cmp_raw = np.interp(freq_cmp, freq_axis, m_anal)
                target_cmp = np.interp(freq_cmp, freq_axis, target_mags)
                filt_cmp = np.interp(freq_cmp, freq_axis, gain_db)
                (
                    target_level_db_cmp,
                    calc_offset_db_cmp,
                    meas_level_db_window_cmp,
                    target_level_db_window_cmp,
                    offset_method_cmp,
                    s_min_cmp,
                    s_max_cmp,
                ) = compute_leveling(cfg, freq_cmp, m_cmp_raw, target_cmp, stereo_link_ctx=stereo_link_ctx)
                comparison_telemetry = BaselineComparisonTelemetry(
                    analysis_mode="comparison",
                    target_mags=np.asarray(target_cmp, dtype=float),
                    measured_mags=np.asarray(m_cmp_raw - calc_offset_db_cmp, dtype=float),
                    filter_mags=np.asarray(filt_cmp, dtype=float),
                    eff_target_db=float(target_level_db_cmp),
                    offset_db=float(calc_offset_db_cmp),
                    smart_scan_range=(float(s_min_cmp), float(s_max_cmp)),
                    meas_level_db_window=float(meas_level_db_window_cmp),
                    target_level_db_window=float(target_level_db_window_cmp),
                    offset_method=str(offset_method_cmp),
                    target_shift_db=float(target_shift_db),
                )
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        comparison_telemetry = None

    return _BaselineContext(
        m_interp=np.asarray(m_interp_for_rt, dtype=float),
        current_rt60=float(current_rt60),
        rt60_bands=dict(rt60_bands) if isinstance(rt60_bands, dict) else {},
        band_avg=float(band_avg),
        target_mags=np.asarray(target_mags, dtype=float),
        hpf_f=float(hpf_f),
        hpf_order=int(hpf_order),
        target_level_db=float(target_level_db),
        calc_offset_db=float(calc_offset_db),
        meas_level_db_window=float(meas_level_db_window),
        target_level_db_window=float(target_level_db_window),
        offset_method=str(offset_method),
        s_min=float(s_min),
        s_max=float(s_max),
        target_shift_db=float(target_shift_db),
        cmp=cmp,
        analysis_mode=str(analysis_mode),
        gain_db=np.asarray(gain_db, dtype=float),
        native_telemetry=native_telemetry,
        comparison_telemetry=comparison_telemetry,
    )
