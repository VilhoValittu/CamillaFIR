from __future__ import annotations

from typing import Any, Callable

import numpy as np
import scipy.ndimage

from . import bassfirst as bf
from .camillafir_analysis import _sigma_bins_from_hz
from .correction_types import (
    _MagAdaptiveStageOutputs,
    _MagCoreOutputs,
    _MagPipelineInputs,
    _MagRawStageOutputs,
)
from .mag_shape import (
    _apply_confidence_logic,
    _apply_regularization,
    _compute_error_db,
    _error_to_correction_mag,
    _resolve_filter_smooth,
    _select_active_band,
)
from .mag_telemetry import _log_stage_stats, _record_stage_probe
from .smoothing import (
    AFDW_BW_MAX_OCT,
    AFDW_BW_MIN_OCT,
    apply_adaptive_fdw,
    psycho_smooth_safe_gain,
)


def run_mag_raw_stage(
    inputs: _MagPipelineInputs,
    *,
    apply_peak_priority_error_shaping: Callable[..., np.ndarray],
    apply_smoothing: Callable[..., np.ndarray],
) -> _MagRawStageOutputs:
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
    filter_smooth = None
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
            filter_smooth=filter_smooth,
            base_sigma=base_sigma,
            df_mode=df_mode,
            raw_g=raw_g,
            final_g=final_g,
            gain_db=np.asarray(gain_db, dtype=float),
            stage_probes=dict(stage_probes),
        )

    filter_smooth = _resolve_filter_smooth(cfg)
    afdw_on = bool(getattr(cfg, "enable_afdw", False))
    afdw_base = float(getattr(cfg, "fdw_cycles", 15.0))
    afdw_min = max(3.0, afdw_base / 3.0)

    manual_target_bias_db = 0.0
    try:
        lvl_mode_s = str(getattr(cfg, "lvl_mode", "Auto") or "Auto").strip().lower()
        if "manual" in lvl_mode_s:
            st_shift = st.get("target_shift_db", None) if isinstance(st, dict) else None
            if st_shift is None:
                st_shift = getattr(cfg, "lvl_manual_db", 0.0)
            manual_target_bias_db = float(st_shift or 0.0)
            if not np.isfinite(manual_target_bias_db):
                manual_target_bias_db = 0.0
            if isinstance(st, dict):
                st["manual_target_bias_db"] = float(manual_target_bias_db)
    except (AttributeError, TypeError, ValueError):
        manual_target_bias_db = 0.0

    err_db = _compute_error_db(m_anal - calc_offset_db, target_mags)
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        try:
            fmin = float(getattr(cfg, "mag_c_min", 20.0) or 20.0)
            fmax = float(getattr(cfg, "mag_c_max", 200.0) or 200.0)
        except (AttributeError, TypeError, ValueError):
            fmin, fmax = 20.0, 200.0
        mask_c_raw = (f >= float(fmin)) & (f <= float(fmax))
    except (TypeError, ValueError):
        mask_c_raw = np.zeros_like(np.asarray(freq_axis, dtype=float), dtype=bool)

    err_db = apply_peak_priority_error_shaping(
        err_db,
        np.asarray(freq_axis, dtype=float),
        cfg,
        st,
        mask_c=mask_c_raw,
        logger=logger,
    )
    raw_g = _error_to_correction_mag(err_db + float(manual_target_bias_db))
    try:
        mm = np.zeros_like(freq_axis, dtype=bool)
        if np.any(mm):
            dv = raw_g[mm]
            logger.info(
                f"RAW_G(mask): max={float(np.max(dv)):.3f} min={float(np.min(dv)):.3f} rms={float(np.sqrt(np.mean(dv*dv))):.3f}"
            )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass
    _log_stage_stats("raw_g_pre_confpull", raw_g, np.zeros_like(freq_axis, dtype=bool), logger=logger, enabled=debug_stage_stats)

    base_sigma = 60 // (filter_smooth / 12 if filter_smooth > 0 else 1)
    df_mode = bool(getattr(cfg, "df_smoothing", False))
    sm_g = apply_smoothing(
        raw_g,
        freq_axis,
        cfg,
        st,
        filter_smooth=filter_smooth,
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
        filter_smooth=filter_smooth,
        base_sigma=base_sigma,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        gain_db=np.asarray(gain_db, dtype=float),
        stage_probes=dict(stage_probes),
    )


def run_mag_bassfirst_afdw_conf_stage(
    inputs: _MagPipelineInputs,
    raw_stage: _MagRawStageOutputs,
    *,
    select_bass_adaptive_conf_mask: Callable[..., tuple[np.ndarray, str]],
    apply_confidence_adaptive_bass_smoothing: Callable[..., np.ndarray],
) -> _MagAdaptiveStageOutputs:
    cfg = inputs.cfg
    freq_axis = inputs.freq_axis
    st = inputs.st
    m_interp = inputs.m_interp
    conf_mask = inputs.conf_mask
    complex_meas = inputs.complex_meas
    logger = inputs.logger
    analysis_mode = inputs.analysis_mode
    cmp = inputs.cmp
    stage_probe = inputs.stage_probe

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
            gd_ms_local = (-np.gradient(ph_u) / (2 * np.pi * df)) * 1000.0
            try:
                gd_sigma_hz = float(getattr(cfg, "bass_first_gd_sigma_hz", 2.0) or 2.0)
            except (AttributeError, TypeError, ValueError):
                gd_sigma_hz = 2.0
            if not np.isfinite(gd_sigma_hz) or gd_sigma_hz <= 0.0:
                gd_sigma_hz = 2.0
            sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=float(gd_sigma_hz), fallback_bins=20.0)
            gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms_local, sigma=float(sigma_bins))
            gd_diff_local = np.abs(gd_ms_local - gd_smooth)
            bf_mode_f2 = float(getattr(cfg, "bass_first_mode_max_hz", 200.0) or 200.0)
            win_mode = "auto"
            left_ms = 0.0
            try:
                win_mode = str(getattr(cfg, "ir_export_window_mode", "auto") or "auto").strip().lower()
                left_ms = float(getattr(cfg, "ir_window_left", getattr(cfg, "ir_window_ms_left", 0.0)) or 0.0)
                if win_mode == "rew_asym" and left_ms < 15.0:
                    bf_mode_f2 = min(bf_mode_f2, 80.0)
                    logger.info(f"REW Asym low-latency: left_ms={left_ms:.1f} -> bass-first limited to {float(bf_mode_f2):.0f} Hz")
            except (AttributeError, TypeError, ValueError):
                pass
            bf_rel, bf_room_mode, _ = bf.build_bassfirst_masks(
                freq_axis=freq_axis,
                m_raw_db=m_interp,
                phase_rad_unwrapped=ph_u,
                gd_ms=gd_ms_local,
                gd_diff=gd_diff_local,
                is_wav_source=bool(getattr(cfg, "is_wav_source", False)),
                mode_f2=bf_mode_f2,
                rew_asym=(win_mode == "rew_asym"),
                left_ms=left_ms,
            )
            bf_conf_for_smoothing = bf.fuse_conf_for_smoothing(
                freq_axis=freq_axis,
                reliability_mask=bf_rel,
                bass_floor_lo=float(getattr(cfg, "bass_first_smooth_floor_lo", 0.75) or 0.75),
                bass_floor_hi=float(getattr(cfg, "bass_first_smooth_floor_hi", 0.35) or 0.35),
            )
        except (TypeError, ValueError, FloatingPointError, IndexError):
            bf_rel = bf_room_mode = bf_conf_for_smoothing = None

    if raw_stage.afdw_on:
        try:
            conf_for_afdw = bf_conf_for_smoothing if (use_bassfirst and bf_conf_for_smoothing is not None) else conf_mask
            c = np.clip(conf_for_afdw, 0.0, 1.0)
            adaptive_cycles = float(raw_stage.afdw_min) + (c * (float(raw_stage.afdw_base) - float(raw_stage.afdw_min)))
            bw = np.clip(2.0 / np.maximum(adaptive_cycles, 1.0), AFDW_BW_MIN_OCT, AFDW_BW_MAX_OCT)
            afdw_bw_oct = bw
            afdw_bw_min_oct = float(np.min(bw))
            afdw_bw_mean_oct = float(np.mean(bw))
            afdw_bw_max_oct = float(np.max(bw))
            bw_min_idx = np.where(bw == np.min(bw))[0]
            bw_max_idx = np.where(bw == np.max(bw))[0]
            afdw_bw_min_hz = float(freq_axis[int(bw_min_idx[len(bw_min_idx) // 2])])
            afdw_bw_max_hz = float(freq_axis[int(bw_max_idx[len(bw_max_idx) // 2])])
            bw_plot = np.asarray(afdw_bw_oct, dtype=float)
            try:
                c_plot = np.clip(np.asarray(conf_mask, dtype=float), 0.0, 1.0)
                if c_plot.shape == np.asarray(freq_axis, dtype=float).shape:
                    adaptive_cycles_plot = float(raw_stage.afdw_min) + (c_plot * (float(raw_stage.afdw_base) - float(raw_stage.afdw_min)))
                    bw_plot = np.clip(2.0 / np.maximum(adaptive_cycles_plot, 1.0), AFDW_BW_MIN_OCT, AFDW_BW_MAX_OCT)
            except (TypeError, ValueError, FloatingPointError):
                bw_plot = np.asarray(afdw_bw_oct, dtype=float)
            try:
                if isinstance(st, dict):
                    st["afdw_bw_oct"] = np.asarray(afdw_bw_oct, dtype=float).tolist()
                    st["afdw_bw_plot_oct"] = np.asarray(bw_plot, dtype=float).tolist()
                    try:
                        if str(analysis_mode).lower() == "comparison":
                            fx_cmp = cmp.get("cmp_freq_axis", None) if isinstance(cmp, dict) else None
                            if fx_cmp is None:
                                fx_cmp = st.get("cmp_freq_axis", None)
                            if fx_cmp is not None:
                                fx_cmp = np.asarray(fx_cmp, dtype=float)
                                st["cmp_afdw_bw_oct"] = np.asarray(np.interp(fx_cmp, freq_axis, np.asarray(afdw_bw_oct, dtype=float)), dtype=float).tolist()
                                st["cmp_afdw_bw_plot_oct"] = np.asarray(np.interp(fx_cmp, freq_axis, np.asarray(bw_plot, dtype=float)), dtype=float).tolist()
                    except (TypeError, ValueError, FloatingPointError):
                        pass
                    st["afdw_bw_min_oct"] = float(afdw_bw_min_oct)
                    st["afdw_bw_mean_oct"] = float(afdw_bw_mean_oct)
                    st["afdw_bw_max_oct"] = float(afdw_bw_max_oct)
                    st["afdw_bw_min_hz"] = float(afdw_bw_min_hz)
                    st["afdw_bw_max_hz"] = float(afdw_bw_max_hz)
            except (TypeError, ValueError):
                pass
        except (TypeError, ValueError, FloatingPointError, IndexError):
            pass
        final_g = apply_adaptive_fdw(freq_axis, final_g, bf_conf_for_smoothing if (use_bassfirst and bf_conf_for_smoothing is not None) else conf_mask, base_cycles=float(raw_stage.afdw_base), min_cycles=float(raw_stage.afdw_min))
        try:
            if isinstance(st, dict):
                st["fdw_mode"] = "adaptive"
        except (TypeError, ValueError):
            pass
    else:
        fixed_cycles = float(max(1.0, raw_stage.afdw_base))
        final_g = apply_adaptive_fdw(freq_axis, final_g, np.ones_like(freq_axis, dtype=float), base_cycles=fixed_cycles, min_cycles=fixed_cycles)
        try:
            bw_fixed_oct = float(np.clip(2.0 / max(fixed_cycles, 1.0), AFDW_BW_MIN_OCT, AFDW_BW_MAX_OCT))
            if isinstance(st, dict):
                st["fdw_mode"] = "fixed"
                st["fdw_fixed_cycles"] = float(fixed_cycles)
                st["fdw_fixed_bw_oct"] = float(bw_fixed_oct)
            logger.info("FDW fixed smoothing (A-FDW OFF): " f"cycles={fixed_cycles:.2f}, bw={bw_fixed_oct:.4f} oct")
        except (TypeError, ValueError):
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
                g0[i1 + 1:] = g0[i1]
        raw_safe_ref = psycho_smooth_safe_gain(freq_axis, g0)
        raw_safe_ref = np.where(mask_c, np.asarray(raw_safe_ref, dtype=float), 0.0)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        raw_safe_ref = None

    if use_bassfirst and bf_room_mode is not None:
        try:
            final_g = bf.modulate_gain_bassfirst(
                final_g,
                bf_room_mode,
                k_mode_cut=float(getattr(cfg, "bass_first_k_mode_cut", 0.6) or 0.6),
                k_mode_boost=float(getattr(cfg, "bass_first_k_mode_boost", 0.9) or 0.9),
            )
        except (TypeError, ValueError, FloatingPointError):
            pass

    pre_bass_adapt_g = np.asarray(final_g, dtype=float).copy()
    try:
        pre_bass_adapt = np.asarray(final_g, dtype=float).copy()
        conf_for_bass_adapt, conf_for_bass_adapt_src = select_bass_adaptive_conf_mask(
            conf_mask=conf_mask,
            bf_conf_for_smoothing=bf_conf_for_smoothing,
            use_bassfirst=bool(use_bassfirst),
        )
        try:
            if isinstance(st, dict):
                st["bass_adaptive_conf_source"] = str(conf_for_bass_adapt_src)
        except (TypeError, ValueError):
            pass
        final_g = apply_confidence_adaptive_bass_smoothing(final_g, freq_axis, cfg, st, conf_for_bass_adapt, stage_tag="core")
        pre_bass_adapt_g = pre_bass_adapt
        _log_stage_stats("final_g_post_bass_adaptive", final_g, mask_c, ref=pre_bass_adapt, logger=logger, enabled=raw_stage.debug_stage_stats)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass

    gain_apply, conf_debug = _apply_confidence_logic(final_g, freq_axis, cfg, st, conf_mask)
    try:
        if isinstance(st, dict):
            st["conf_logic_mode"] = str(conf_debug.get("mode", "unknown"))
    except (AttributeError, TypeError, ValueError):
        pass
    _log_stage_stats("gain_apply_pre_limits", gain_apply, mask_c, logger=logger, enabled=raw_stage.debug_stage_stats)
    try:
        win_mode = str(getattr(cfg, "ir_export_window_mode", "auto") or "auto").strip().lower()
        left_ms = float(getattr(cfg, "ir_window_left", getattr(cfg, "ir_window_ms_left", 0.0)) or 0.0)
        if win_mode == "rew_asym" and left_ms < 10.0:
            hz = 120.0
            m = mask_c & (freq_axis > 0.0) & (freq_axis <= hz)
            if np.any(m):
                gain_apply[m] = np.minimum(gain_apply[m], 0.0)
                logger.info(f"REW Asym safety: left_ms={left_ms:.1f} -> no LF boost below {hz:.0f} Hz")
    except (AttributeError, TypeError, ValueError, FloatingPointError):
        pass
    try:
        tmp_after_apply = np.zeros_like(gain_db, dtype=float)
        tmp_after_apply[mask_c] = gain_apply[mask_c]
        _record_stage_probe(stage_probes, "after_gain_apply", stage_probe, freq_axis, tmp_after_apply, mask_c, cfg, logger)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass

    return _MagAdaptiveStageOutputs(
        final_g=np.asarray(final_g, dtype=float),
        mask_c=np.asarray(mask_c, dtype=bool),
        stage_probes=dict(stage_probes),
        use_bassfirst=bool(use_bassfirst),
        bf_room_mode=bf_room_mode,
        bf_rel=bf_rel,
        bf_conf_for_smoothing=bf_conf_for_smoothing,
        pre_bass_adapt_g=np.asarray(pre_bass_adapt_g, dtype=float),
        gain_db=np.asarray(gain_db, dtype=float),
        gain_apply=np.asarray(gain_apply, dtype=float),
        raw_safe_ref=raw_safe_ref,
    )


def run_mag_core_stage(
    inputs: _MagPipelineInputs,
    *,
    run_mag_raw_stage_fn: Callable[[_MagPipelineInputs], _MagRawStageOutputs],
    run_mag_bassfirst_afdw_conf_stage_fn: Callable[[_MagPipelineInputs, _MagRawStageOutputs], _MagAdaptiveStageOutputs],
) -> _MagCoreOutputs:
    raw_stage = run_mag_raw_stage_fn(inputs)
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

    adaptive_stage = run_mag_bassfirst_afdw_conf_stage_fn(inputs, raw_stage)
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
