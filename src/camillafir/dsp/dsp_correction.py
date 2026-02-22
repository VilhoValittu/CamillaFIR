from __future__ import annotations

import numpy as np
import scipy.ndimage

from . import bassfirst as bf
from .analysis import _sigma_bins_from_hz, calculate_rt60, calculate_rt60_bands
from .camillafir_leveling import compute_leveling
from .limits import (
    build_slope_limit_envelope,
    limit_slope_per_octave,
    limit_slope_per_octave_asym,
    soft_clip_gain,
)
from .phase import get_min_phase_impulse
from .smoothing import (
    AFDW_BW_MAX_OCT,
    AFDW_BW_MIN_OCT,
    apply_adaptive_fdw,
    psycho_smooth_safe_gain,
    smooth_gain_fractional_octave,
)
from .tdc import apply_smart_tdc


def run_correction_stage(
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
    conf_mask,
    complex_meas,
    logger,
    interpolate_response_fn,
    apply_confidence_weighted_target_pull_fn,
    stage_probe_fn,
    cfg_float_allow_zero_fn,
):
    interpolate_response = interpolate_response_fn
    apply_confidence_weighted_target_pull = apply_confidence_weighted_target_pull_fn
    _stage_probe = stage_probe_fn
    _cfg_float_allow_zero = cfg_float_allow_zero_fn

    # --- 1. RT60 & TARGET ---
    m_rt_lin = np.interp(np.linspace(0, cfg.fs/2, 65537), freq_axis, np.interp(freq_axis, f_in, m_in))
    rt_ir = get_min_phase_impulse(m_rt_lin, 131072)
    current_rt60 = calculate_rt60(rt_ir, cfg.fs)
    rt60_bands = calculate_rt60_bands(rt_ir, cfg.fs, f_min=31.5, f_max=8000.0, order=4)
    # "One number" of bands (good for scoring/reporting): median 125–4000 Hz if found
    band_avg = 0.0
    if rt60_bands:
        ks = np.array(sorted(rt60_bands.keys()), dtype=float)
        vs = np.array([rt60_bands[k] for k in ks], dtype=float)
        mid = (ks >= 125.0) & (ks <= 4000.0) & (vs > 0.05) & (vs < 5.0)
        if np.any(mid):
            band_avg = float(np.median(vs[mid]))
        else:
            band_avg = float(np.median(vs))
    
    if cfg.house_freqs is not None and cfg.house_mags is not None and len(cfg.house_freqs) >= 2 and len(cfg.house_mags) >= 2:
        target_mags = interpolate_response(cfg.house_freqs, cfg.house_mags, freq_axis)
    else:
        # fallback: flat 0 dB target
        target_mags = np.zeros_like(freq_axis, dtype=float)
    
    # NOTE (HPF):
    # Do NOT bake HPF into target curve. HPF is applied as a real magnitude filter later
    # (gain_db += hpf_db), which keeps magnitude+phase consistent and avoids double-HPF.
    
    if cfg.enable_tdc:
        # NEW: TDC receives frequency-dependent RT60 (dict), auto-fallback if empty
        rt60_for_tdc = rt60_bands if rt60_bands else current_rt60
        # Safety brakes: cap total TDC reduction and keep it smooth (avoid deep, stacked notches)
        # Configurable TDC safety brakes for easy A/B testing
        tdc_strength = _cfg_float_allow_zero(cfg, "tdc_strength", 50.0)
        tdc_max_red = _cfg_float_allow_zero(cfg, "tdc_max_reduction_db", 9.0)
        tdc_slope = _cfg_float_allow_zero(cfg, "tdc_slope_db_per_oct", 0.0)

        # Clamp to sane values (never explode)
        if tdc_strength < 0: tdc_strength = 0.0
        if tdc_strength > 100: tdc_strength = 100.0
        if tdc_max_red < 0: tdc_max_red = 0.0
        if tdc_max_red > 24: tdc_max_red = 24.0
        if tdc_slope < 0: tdc_slope = 0.0
        if tdc_slope > 24: tdc_slope = 24.0

        target_mags = apply_smart_tdc(
            freq_axis,
            target_mags,
            reflections,
            rt60_for_tdc,
            tdc_strength / 100.0,
            max_total_reduction_db=tdc_max_red,
            max_slope_db_per_oct=tdc_slope
        )

    # --- HPF params (always defined) ---
    hpf_f = 0.0
    hpf_order = 0
    if cfg.hpf_settings and cfg.hpf_settings.get('enabled'):
        hpf_f = float(cfg.hpf_settings.get('freq', 0.0) or 0.0)
        hpf_order = int(cfg.hpf_settings.get('order', 0) or 0)

    # --- 2. TASONSOVITUS ---
    # Huom: tasosovitus on erotettu omaan moduuliin testattavuuden ja edge-case -robustiuden takia.
    #
    # IMPORTANT:
    # target_level_db / calc_offset_db / s_min / s_max MUST be defined before any later use.
    # compute_leveling() guarantees finite outputs; stereo-link is handled inside compute_leveling().
    target_level_db = 0.0
    calc_offset_db = 0.0
    meas_level_db_window = 0.0
    target_level_db_window = 0.0
    offset_method = "Init"
    try:
        s_min = float(getattr(cfg, "lvl_min", 500.0) or 500.0)
        s_max = float(getattr(cfg, "lvl_max", 2000.0) or 2000.0)
    except Exception:
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
        ) = compute_leveling(cfg, np.asarray(freq_axis, dtype=float), np.asarray(m_anal, dtype=float), np.asarray(target_mags, dtype=float))
    except Exception:
        # Keep safe defaults; never crash later with UnboundLocalError.
        pass

    # --- 7A. Align target level to the SAME leveling window ---
    # Keep deterministic behavior for both Auto and Manual:
    # shift target, then recompute leveling so plotting/metadata stay consistent.
    target_shift_db = 0.0
    try:
        f = np.asarray(freq_axis, dtype=float)
        t = np.asarray(target_mags, dtype=float)
        if f.size == t.size and f.size > 16 and np.isfinite(float(s_min)) and np.isfinite(float(s_max)):
            mask_lvl = (f >= float(s_min)) & (f <= float(s_max))
            if int(np.count_nonzero(mask_lvl)) > 10:
                tgt_win_mean = float(np.mean(t[mask_lvl]))
                if np.isfinite(tgt_win_mean) and np.isfinite(float(target_level_db)):
                    # shift target so that its mean in the leveling window equals target_level_db
                    target_shift_db = float(target_level_db) - tgt_win_mean
                    target_mags = t + target_shift_db

                    # Recompute leveling with shifted target so calc_offset_db matches the new absolute target
                    (
                        target_level_db,
                        calc_offset_db,
                        meas_level_db_window,
                        target_level_db_window,
                        offset_method,
                        s_min,
                        s_max,
                    ) = compute_leveling(cfg, f, m_anal, target_mags)
    except Exception:
        target_shift_db = 0.0


    # Plotly uses st["target_mags"] / st["measured_mags"]. If we shift target in DSP but don't update st,
    # the "Target & Magnitude" curves can disappear or desync.
    # --- 2A2. Expose EFFECTIVE target & axis to native UI/plots ---
    try:
        if isinstance(st, dict):
            st["analysis_mode"] = "native"
            st["freq_axis"] = np.asarray(freq_axis, dtype=float).tolist()

            # --- PLOT REFERENCE FIX ---
            # Plot everything relative to Smart Scan / Manual leveling window
            # so target & measured share the SAME 0 dB reference

            m_src = np.asarray(m_anal, dtype=float)
            if is_psy and (m_plot_db is not None):
                # UI-only psychoacoustic magnitude view (REW-like)
                mp = np.asarray(m_plot_db, dtype=float)
                if mp.size == m_src.size:
                    m_src = mp

            measured_aligned = m_src - float(calc_offset_db)
            target_aligned   = np.asarray(target_mags, dtype=float) - float(target_level_db)

            st["measured_mags"] = measured_aligned.tolist()
            st["target_mags"]   = target_aligned.tolist()

            # --- VISUAL: slope-limit envelope around Target (dB/oct) ---
            # This is UI-only (does not change DSP).
            try:
                # Inherit legacy symmetric slope limit if boost/cut are unset
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
                if env_lo is not None and env_hi is not None:
                    st["target_env_lo"] = np.asarray(env_lo, dtype=float).tolist()
                    st["target_env_hi"] = np.asarray(env_hi, dtype=float).tolist()
                    st["target_env_pivot_hz"] = float(env_pivot) if env_pivot is not None else None
            except Exception:
                pass

            st["target_shift_db"] = float(target_shift_db)

            # keep alignment metadata consistent
            st["eff_target_db"] = float(target_level_db)
            st["target_level_db_window"] = float(target_level_db_window)
            st["meas_level_db_window"] = float(meas_level_db_window)
            st["offset_db"] = float(calc_offset_db)
            st["offset_method"] = str(offset_method)
            st["smart_scan_range"] = [float(s_min), float(s_max)]
    except Exception:
        pass


    # --- 2B. Keep comparison-mode target/leveling consistent after target shaping (HPF/TDC/etc.) ---
    # NOTE: comparison grid is built earlier for confidence stability, but target_mags is shaped later.
    # Without this, plots in comparison mode may show the *pre-shaping* target curve.
    try:
        if isinstance(cmp, dict) and analysis_mode == "comparison":
            freq_cmp = np.asarray(cmp.get("cmp_freq_axis", []) or [], dtype=float)
            if freq_cmp.size > 8:
                # reconstruct raw cmp measured (before offset) from stored arrays
                # reconstruct RAW measured on comparison grid
                m_cmp_raw = np.interp(freq_cmp, freq_axis, m_anal)

                # final effective target on comparison grid
                target_cmp = np.interp(freq_cmp, freq_axis, target_mags)

                # re-run leveling on comparison grid
                filt_cmp = np.interp(freq_cmp, freq_axis, gain_db)
                (
                    target_level_db_cmp,
                    calc_offset_db_cmp,
                    meas_level_db_window_cmp,
                    target_level_db_window_cmp,
                    offset_method_cmp,
                    s_min_cmp,
                    s_max_cmp,
                ) = compute_leveling(cfg, freq_cmp, m_cmp_raw, target_cmp)

                # final aligned measured + filter on comparison grid
                meas_cmp_final = m_cmp_raw - calc_offset_db_cmp
                filt_cmp = np.interp(freq_cmp, freq_axis, gain_db)

                cmp["analysis_mode"] = "comparison"
                cmp["cmp_target_mags"] = target_cmp.tolist()
                cmp["cmp_measured_mags"] = meas_cmp_final.tolist()
                cmp["cmp_filter_mags"] = filt_cmp.tolist()
                cmp["cmp_filter_mags"] = filt_cmp.tolist()
                cmp["cmp_eff_target_db"] = float(target_level_db_cmp)
                cmp["cmp_offset_db"] = float(calc_offset_db_cmp)
                cmp["cmp_measured_mags"] = (m_cmp_raw - calc_offset_db_cmp).tolist()
                cmp["cmp_smart_scan_range"] = [float(s_min_cmp), float(s_max_cmp)]
                cmp["cmp_meas_level_db_window"] = float(meas_level_db_window_cmp)
                cmp["cmp_target_level_db_window"] = float(target_level_db_window_cmp)
                cmp["cmp_offset_method"] = str(offset_method_cmp)
                cmp["cmp_target_shift_db"] = float(target_shift_db)
    except Exception:
        pass

    # --- 3. KORJAUS ---
    if cfg.enable_mag_correction:
        # Optional debug: stage stats (no UI; enable via config)
        debug_stage_stats = bool(getattr(cfg, "debug_stage_stats", True))

        def _log_stats(name: str, x: np.ndarray, mask: np.ndarray | None = None, ref: np.ndarray | None = None):
            """
            Lightweight stats: peak boost/cut, RMS, and delta vs ref.
            Only logs when debug_stage_stats is enabled.
            """
            if not debug_stage_stats:
                return
            try:
                a = np.asarray(x, dtype=float)
                if mask is not None:
                    m = np.asarray(mask, dtype=bool)
                    if m.shape == a.shape and np.any(m):
                        v = a[m]
                    else:
                        v = a
                else:
                    v = a
                v = v[np.isfinite(v)]
                if v.size < 4:
                    return
                v_max = float(np.max(v))
                v_min = float(np.min(v))
                v_rms = float(np.sqrt(np.mean(v * v)))
                msg = f"StageStats: {name}: max={v_max:.3f} dB, min={v_min:.3f} dB, rms={v_rms:.3f} dB"
                if ref is not None:
                    r = np.asarray(ref, dtype=float)
                    if mask is not None and m.shape == r.shape and np.any(m):
                        dv = (a - r)[m]
                    else:
                        dv = (a - r)
                    dv = dv[np.isfinite(dv)]
                    if dv.size >= 4:
                        d_abs_max = float(np.max(np.abs(dv)))
                        d_rms = float(np.sqrt(np.mean(dv * dv)))
                        msg += f" | Δmax={d_abs_max:.3f} dB, Δrms={d_rms:.3f} dB"
                logger.info(msg)
            except Exception:
                pass
        def _apply_confpull_post_slope(
            gain_db_in: np.ndarray,
            mask_c_in: np.ndarray,
            measured_ref_db: np.ndarray | None = None
        ):
            """
            Apply confidence-weighted target pull AFTER slope limiting.
            This makes ConfPull actually affect the final filter (previously slope limiter dominated).
            Pulls towards a STABLE reference (raw_safe from raw_g) when provided..
            """
            try:
                if gain_db_in is None or mask_c_in is None:
                    return gain_db_in
                if not (isinstance(gain_db_in, np.ndarray) and isinstance(mask_c_in, np.ndarray)):
                    return gain_db_in
                if gain_db_in.size < 16 or gain_db_in.shape != mask_c_in.shape:
                    return gain_db_in
                if not np.any(mask_c_in):
                    return gain_db_in

                # Tunables (cfg override; safe defaults if not present)
                _conf_floor = float(getattr(cfg, "conf_pull_floor", 0.05) or 0.05)
                _conf_ceil  = float(getattr(cfg, "conf_pull_ceil", 0.95) or 0.95)
                _conf_max_hz = getattr(cfg, "conf_pull_max_hz", 200.0)
                _conf_max_hz = None if _conf_max_hz is None else float(_conf_max_hz)
                _gamma_cut = float(getattr(cfg, "conf_pull_gamma_cut", 0.55) or 0.55)
                _gamma_boost = float(getattr(cfg, "conf_pull_gamma_boost", 1.35) or 1.35)

                # NEW: confidence smoothing + bass-floor (same behavior as earlier versions, but now applied post-slope)
                _conf_sigma = float(getattr(cfg, "conf_pull_conf_smooth_sigma", 2.0) or 2.0)
                _bass_floor_hz = float(getattr(cfg, "conf_pull_bass_floor_hz", 120.0) or 120.0)
                _bass_floor_min = float(getattr(cfg, "conf_pull_bass_floor_min", 0.25) or 0.25)

                if not np.isfinite(_conf_sigma) or _conf_sigma < 0.0:
                    _conf_sigma = 0.0
                if not np.isfinite(_bass_floor_hz) or _bass_floor_hz < 0.0:
                    _bass_floor_hz = 0.0
                if not np.isfinite(_bass_floor_min) or _bass_floor_min < 0.0:
                    _bass_floor_min = 0.0
                _bass_floor_min = float(np.clip(_bass_floor_min, 0.0, 1.0))

                # Prepare confidence for pull (shape must match gain_db)
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
                        conf_for_pull = np.clip(c0, 0.0, 1.0)
                except Exception:
                    conf_for_pull = conf_mask

                # Build measured reference:
                # Prefer a STABLE "raw_safe" reference from raw_g (measured_ref_db).
                # Fallback: old behavior using safe smoothing of gain_db_in.
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
                    # Fallback: safe smoothing of already-limited gain curve (old behavior)
                    try:
                        g_in = np.asarray(gain_db_in, dtype=float).copy()
                        idx = np.where(mask_c_in)[0]
                        i0, i1 = int(idx[0]), int(idx[-1])
                        if i0 > 0:
                            g_in[:i0] = g_in[i0]
                        if i1 < (g_in.size - 1):
                            g_in[i1+1:] = g_in[i1]
                        g_ref = psycho_smooth_safe_gain(freq_axis, g_in)
                    except Exception:
                        g_ref = np.asarray(gain_db_in, dtype=float)

                # Always: only use reference in correction band; outside keep target
                g_ref = np.where(mask_c_in, np.asarray(g_ref, dtype=float), gain_db_in)

                # Apply pull (request telemetry for logs/stats)
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

                # Enforce: only affect correction band
                gain_out = np.where(mask_c_in, np.asarray(gain_out, dtype=float), gain_db_in)

                # Telemetry log (Post-slope)
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

                    # Combine pull_mask with correction mask so stats match "where it matters"
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
                except Exception:
                    pass

                return gain_out
            except Exception:
                return gain_db_in

        # Filter smoothing (DSP only). Backward compatible with old config key.
        try:
            _filter_smooth = float(getattr(cfg, "filter_smooth", getattr(cfg, "smoothing_level", 12)) or 12)
        except Exception:
            _filter_smooth = 12.0
        if not np.isfinite(_filter_smooth) or _filter_smooth <= 0:
            _filter_smooth = 12.0

        afdw_on = bool(getattr(cfg, "enable_afdw", False))
        afdw_base = float(getattr(cfg, "fdw_cycles", 15.0))
        afdw_min = max(3.0, afdw_base / 3.0)
        
        # NOTE: ConfPull moved to POST-SLOPE (gain_db stage), because slope limiter dominated
        # and made early raw_g pulls mostly invisible in final filters.
        manual_target_bias_db = 0.0
        try:
            lvl_mode_s = str(getattr(cfg, "lvl_mode", "Auto") or "Auto").strip().lower()
            if "manual" in lvl_mode_s:
                # Reference point for manual mode:
                # 0 dB == neutral, lower/higher values bias correction down/up.
                manual_target_bias_db = float(getattr(cfg, "lvl_manual_db", 0.0) or 0.0)
                if isinstance(st, dict):
                    st["manual_target_bias_db"] = float(manual_target_bias_db)
        except Exception:
            manual_target_bias_db = 0.0

        raw_g = target_mags - (m_anal - calc_offset_db) + float(manual_target_bias_db)
        try:
            mm = mask_c if "mask_c" in locals() else None
            if mm is not None and np.any(mm):
                dv = raw_g[mm]
                logger.info(f"RAW_G(mask): max={float(np.max(dv)):.3f} min={float(np.min(dv)):.3f} rms={float(np.sqrt(np.mean(dv*dv))):.3f}")
        except Exception:
            pass
        _log_stats("raw_g_pre_confpull", raw_g, mask_c if "mask_c" in locals() else None)
        base_sigma = 60 // (_filter_smooth / 12 if _filter_smooth > 0 else 1)

        # Raw_g smoothing:
        # - df_smoothing ON: keep legacy Gaussian model with constant-Hz behavior across fs/taps
        # - df_smoothing OFF: true 1/N-octave smoothing (log-frequency), matches UI wording
        df_mode = bool(getattr(cfg, "df_smoothing", False))
        if df_mode:
            # Reference bin width ~ 44100/65536 Hz; match the old "base_sigma bins" at ref
            df_ref = 44100.0 / 65536.0
            sigma_hz = float(base_sigma) * df_ref
            # Convert Hz -> bins for current axis
            sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=sigma_hz, fallback_bins=max(2.0, float(base_sigma)))
            sm_g = scipy.ndimage.gaussian_filter1d(raw_g, sigma=float(sigma_bins))
        else:
            sm_g = smooth_gain_fractional_octave(
                freq_axis,
                raw_g,
                _filter_smooth,
            )

        final_g = raw_g - (raw_g - sm_g) * (cfg.reg_strength / 100.0)
        # Stage probes container (per channel/run)
        stage_probes = {}
        
        # --- 8B. A-FDW directly to correction curve ---
        # Debug/telemetry: effective BW (oct) per frequency (continuous)
        afdw_bw_oct = None
        afdw_bw_min_oct = afdw_bw_mean_oct = afdw_bw_max_oct = None
        afdw_bw_min_hz = afdw_bw_max_hz = None

        
        # Smooths final_g adaptively based on confidence mask:
        # - low confidence => more "cycles" => softer correction
        # - high confidence => less smoothing => sharper correction
        # --- 8B0. Bass-first AI masks (optional) ---
        use_bassfirst = bool(getattr(cfg, "bass_first_ai", False))
        bf_room_mode = None
        bf_rel = None
        bf_conf_for_smoothing = None

        if use_bassfirst:
            try:
                # Tarvitaan raakamitta akselille + vaihe (unwrap) + gd_diff
                # Note: at this point freq_axis exists and complex_meas was built earlier in analysis.
                # Use same base data as analyze_acoustic_confidence: phase unwrap + gd
                ph_u = np.unwrap(np.angle(complex_meas))
                df = np.gradient(freq_axis) + 1e-12
                gd_ms_local = (-np.gradient(ph_u) / (2*np.pi*df)) * 1000.0

                # GD-diff should be deterministic across different freq_axis resolutions (fs/taps).
                # Use Hz-based smoothing width (converted to bins) instead of fixed sigma-in-bins.
                try:
                    _gd_sigma_hz = float(getattr(cfg, "bass_first_gd_sigma_hz", 2.0) or 2.0)
                except Exception:
                    _gd_sigma_hz = 2.0
                if not np.isfinite(_gd_sigma_hz) or _gd_sigma_hz <= 0.0:
                    _gd_sigma_hz = 2.0
                sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=float(_gd_sigma_hz), fallback_bins=20.0)
                gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms_local, sigma=float(sigma_bins))
                gd_diff_local = np.abs(gd_ms_local - gd_smooth)
                # Limit bass-first effect at very low latency REW Asym to avoid unstable LF behavior.
                # NOTE: this only affects bass-first masks (A-FDW confidence shaping), not the main correction band.
                # Limit bass-first effect at very low latency REW Asym to avoid unstable LF behavior.
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

                # m_interp should be the "raw measured mags" interpolated to freq_axis
                # If at this point a different name is used, use it (usually m_interp / meas_interp etc.)
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

        
        if afdw_on:
            try:
                conf_for_afdw = (bf_conf_for_smoothing if (use_bassfirst and bf_conf_for_smoothing is not None) else conf_mask)
                c = np.clip(conf_for_afdw, 0.0, 1.0)
                adaptive_cycles = float(afdw_min) + (c * (float(afdw_base) - float(afdw_min)))
                bw = 2.0 / np.maximum(adaptive_cycles, 1.0)
                # clamp to same range used by the continuous blender
                bw = np.clip(bw, AFDW_BW_MIN_OCT, AFDW_BW_MAX_OCT)
                afdw_bw_oct = bw
                afdw_bw_min_oct = float(np.min(bw))
                afdw_bw_mean_oct = float(np.mean(bw))
                afdw_bw_max_oct = float(np.max(bw))
                bw_min_idx = np.where(bw == np.min(bw))[0]
                bw_max_idx = np.where(bw == np.max(bw))[0]
                afdw_bw_min_hz = float(freq_axis[int(bw_min_idx[len(bw_min_idx)//2])])
                afdw_bw_max_hz = float(freq_axis[int(bw_max_idx[len(bw_max_idx)//2])])

                # Feed dashboard: A-FDW Effective BW (oct)
                try:
                    if isinstance(st, dict):
                        st["afdw_bw_oct"] = np.asarray(afdw_bw_oct, dtype=float).tolist()
                        # Build cmp_afdw_bw_oct only if comparison grid exists; MUST match cmp_freq_axis length
                        try:
                            if str(locals().get("analysis_mode", "native")).lower() == "comparison":
                                _fx_cmp = None
                                # prefer cmp dict if present
                                _cmp = locals().get("cmp", None)
                                if isinstance(_cmp, dict):
                                    _fx_cmp = _cmp.get("cmp_freq_axis", None)
                                if _fx_cmp is None:
                                    _fx_cmp = st.get("cmp_freq_axis", None)
                                if _fx_cmp is None:
                                    _fx_cmp = stats.get("cmp_freq_axis", None) if isinstance(locals().get("stats", None), dict) else None

                                if _fx_cmp is not None:
                                    fx_cmp = np.asarray(_fx_cmp, dtype=float)
                                    bw_cmp = np.interp(fx_cmp, freq_axis, np.asarray(afdw_bw_oct, dtype=float))
                                    st["cmp_afdw_bw_oct"] = np.asarray(bw_cmp, dtype=float).tolist()
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
                base_cycles=afdw_base,
                min_cycles=afdw_min
            )
        mask_c = (freq_axis >= (0 if cfg.hpf_settings else cfg.mag_c_min)) & (freq_axis <= cfg.mag_c_max)

        # --- Build STABLE measured reference for ConfPullPost: raw_safe from raw_g ---
        # Use edge-hold inside correction band before smoothing to avoid pulling towards 0 dB outside band.
        raw_safe_ref = None
        try:
            g0 = np.asarray(raw_g, dtype=float).copy()
            idx = np.where(mask_c)[0]
            if idx.size >= 2:
                i0, i1 = int(idx[0]), int(idx[-1])
                if i0 > 0:
                    g0[:i0] = g0[i0]
                if i1 < (g0.size - 1):
                    g0[i1+1:] = g0[i1]
            raw_safe_ref = psycho_smooth_safe_gain(freq_axis, g0)
            # Keep reference only in correction band (outside is irrelevant / can confuse)
            raw_safe_ref = np.where(mask_c, np.asarray(raw_safe_ref, dtype=float), 0.0)
        except Exception:
            raw_safe_ref = None
        # When A-FDW is ON, don't multiply final_g by eff_conf,
        # because A-FDW already applies "caution" to shape (smoothing).
        # This avoids double caution (shape softens + amplitude attenuates).
        # --- 8B1. Bass-first gain modulation (room modes) ---
        if use_bassfirst and bf_room_mode is not None:
            try:
                final_g = bf.modulate_gain_bassfirst(
                    final_g, bf_room_mode,
                    k_mode_cut=float(getattr(cfg, "bass_first_k_mode_cut", 0.6) or 0.6),
                    k_mode_boost=float(getattr(cfg, "bass_first_k_mode_boost", 0.9) or 0.9),
                )
            except Exception:
                pass
        if bool(getattr(cfg, "enable_afdw", False)):
            gain_apply = final_g.copy()
        else:
            eff_conf = np.where(freq_axis < 100, np.maximum(conf_mask, 0.6), conf_mask)
            gain_apply = (final_g * eff_conf).copy()
        _gain_apply_pre_limits = gain_apply.copy()
        _log_stats("gain_apply_pre_limits", gain_apply, mask_c)
        # --- REW ASYM ultra-low latency safety (gain side) ---
        # If latency target is extremely small, prevent LF boosts (cuts still allowed)
        # to avoid unstable correction / ripple in bass.
        try:
            _win_mode = str(getattr(cfg, "ir_export_window_mode", "auto") or "auto").strip().lower()
            _left_ms  = float(getattr(cfg, "ir_window_left", getattr(cfg, "ir_window_ms_left", 0.0)) or 0.0)
            if _win_mode == "rew_asym" and _left_ms < 10.0:
                _hz = 120.0
                _m = mask_c & (freq_axis > 0.0) & (freq_axis <= _hz)
                if np.any(_m):
                    gain_apply[_m] = np.minimum(gain_apply[_m], 0.0)  # no boost, cuts OK
                    logger.info(f"REW Asym safety: left_ms={_left_ms:.1f} -> no LF boost below {_hz:.0f} Hz")
        except Exception:
            pass

        # --- CHECKPOINT 1: after_gain_apply (before slope/fade/exc/limits) ---
        try:
            _tmp_after_apply = np.zeros_like(gain_db, dtype=float)
            _tmp_after_apply[mask_c] = gain_apply[mask_c]
            stage_probes["after_gain_apply"] = _stage_probe(
                "after_gain_apply", freq_axis, _tmp_after_apply, mask_c,
                global_gain_db=float(getattr(cfg, "global_gain_db", 0.0) or 0.0),
                auto_headroom_db=0.0,
                logger_obj=logger
            )
        except Exception:
            pass

        # --- 8C. Low-bass CUT allowance ---
        # < low_hz: allow ONLY attenuation (no boost).
        # IMPORTANT: do NOT force "min(final_g, raw_g)" by default, because that can
        # collapse different settings into identical low-frequency filters.
        low_cut_enable = True
        try:
            # optional config flag (default ON if missing)
            low_cut_enable = bool(getattr(cfg, "low_bass_cut_enable", True))
        except Exception:
            low_cut_enable = True

        low_hz = _cfg_float_allow_zero(cfg, "low_bass_cut_hz", 0.0)
        # strength: 0.0 => just "no boost" (preserve differences)
        #           1.0 => old aggressive behavior (prefer stronger cuts)
        try:
            low_cut_strength = float(getattr(cfg, "low_bass_cut_strength", 0.0) or 0.0)
        except Exception:
            low_cut_strength = 0.0
        if not np.isfinite(low_cut_strength):
            low_cut_strength = 0.0
        low_cut_strength = float(np.clip(low_cut_strength, 0.0, 1.0))

        low_mask = mask_c & (freq_axis > 0) & (freq_axis <= low_hz)
        if low_cut_enable and np.any(low_mask):
            # Base policy: keep computed correction but never boost in very low bass
            low_cut = np.minimum(gain_apply[low_mask], 0.0)

            # Optional aggressiveness: blend toward "stronger of (final_g, raw_g)" cuts
            # (this approximates the old behavior but avoids always dominating)
            if low_cut_strength > 0.0:
                stronger_cut = np.minimum(final_g[low_mask], raw_g[low_mask])
                stronger_cut = np.minimum(stronger_cut, 0.0)
                low_cut = (1.0 - low_cut_strength) * low_cut + (low_cut_strength) * stronger_cut

            gain_apply[low_mask] = low_cut

        # --- CHECKPOINT 2: after_lowbass_policy ---
        try:
            _tmp_after_low = np.zeros_like(gain_db, dtype=float)
            _tmp_after_low[mask_c] = gain_apply[mask_c]
            stage_probes["after_lowbass_policy"] = _stage_probe(
                "after_lowbass_policy", freq_axis, _tmp_after_low, mask_c,
                global_gain_db=float(getattr(cfg, "global_gain_db", 0.0) or 0.0),
                auto_headroom_db=0.0,
                logger_obj=logger
            )
            logger.info(
                f"CFG CHECK: conf_pull_floor={cfg.conf_pull_floor}, "
                f"gamma_cut={cfg.conf_pull_gamma_cut}, "
                f"low_bass_cut_strength={cfg.low_bass_cut_strength}"
)

        except Exception:
            pass

        # --- SLOPE LIMIT (if present in your pipeline) ---
        # NOTE: This patch assumes you have a section that modifies gain_apply or tmp via slope limiting.
        # If your slope limiting modifies a different array, place the probe right after that modification.


        # --- 8D. Max cut + max boost (soft) ---
        max_cut_db = abs(float(getattr(cfg, "max_cut_db", 15.0) or 15.0))  # default: sallitaan kohtuullinen leikkaus

        # --- 8D-a. Diagnostic: config snapshot (once per run path) ---
        try:
            logger.info(
                "Diagnostic: "
                f"max_boost_db={float(getattr(cfg,'max_boost_db',0.0) or 0.0):.2f} dB, "
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

        # Pre-clamp diagnostics (what the algorithm "wanted" to do before soft clip)
        try:
            _cand = np.zeros_like(gain_db, dtype=float)
            _cand[mask_c] = gain_apply[mask_c]
            boost_cand_peak = float(np.max(_cand[mask_c])) if np.any(mask_c) else 0.0
            cut_cand_peak = float(np.min(_cand[mask_c])) if np.any(mask_c) else 0.0
            n_boost_cand = int(np.sum((_cand > 1e-6) & mask_c))
            n_boost_cand_low = int(np.sum((_cand > 1e-6) & mask_c & (freq_axis <= low_hz)))
            if bool(getattr(cfg, "exc_prot", False)):
                exc_f = float(getattr(cfg, "exc_freq", 0.0) or 0.0)
                n_boost_cand_exc = int(np.sum((_cand > 1e-6) & mask_c & (freq_axis < exc_f)))
            else:
                n_boost_cand_exc = 0
        except Exception:
            boost_cand_peak, cut_cand_peak = 0.0, 0.0
            n_boost_cand, n_boost_cand_low, n_boost_cand_exc = 0, 0, 0
        tmp = np.zeros_like(gain_db, dtype=float)
        tmp[mask_c] = gain_apply[mask_c]

        # --- CHECKPOINT 3: pre_softclip (this is the input to soft_clip_gain) ---
        try:
            stage_probes["pre_softclip"] = _stage_probe(
                "pre_softclip", freq_axis, tmp, mask_c,
                global_gain_db=float(getattr(cfg, "global_gain_db", 0.0) or 0.0),
                auto_headroom_db=0.0,
                logger_obj=logger
            )
        except Exception:
            pass


        # --- 8D-b. Clamp diagnostics: SOFT CLIP stage (what got limited here?) ---
        try:
            _pre_soft = tmp.copy()
            _max_boost = float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
            _max_cut = float(max_cut_db)

            # How much candidate exceeded limits before soft clip?
            if np.any(mask_c):
                over_boost = float(np.max(_pre_soft[mask_c] - _max_boost)) if _max_boost > 0 else float(np.max(_pre_soft[mask_c]))
                over_boost = max(0.0, over_boost)
                over_cut = float(np.max((-_pre_soft[mask_c]) - _max_cut))
                over_cut = max(0.0, over_cut)
            else:
                over_boost, over_cut = 0.0, 0.0
        except Exception:
            _pre_soft = tmp
            over_boost, over_cut = 0.0, 0.0

        tmp = soft_clip_gain(tmp, cfg.max_boost_db, max_cut_db)

        try:
            _post_soft = tmp
            if np.any(mask_c):
                softclip_boost_bins = int(np.sum((_pre_soft[mask_c] > (_max_boost + 1e-9)) & (_post_soft[mask_c] <= (_max_boost + 1e-9))))
                softclip_cut_bins   = int(np.sum((_pre_soft[mask_c] < (-_max_cut - 1e-9)) & (_post_soft[mask_c] >= (-_max_cut - 1e-9))))
            else:
                softclip_boost_bins, softclip_cut_bins = 0, 0
            logger.info(
                "Clamp: soft_clip "
                f"(max_boost={_max_boost:.2f} dB, max_cut={_max_cut:.2f} dB) -> "
                f"boost_clipped_bins={softclip_boost_bins}, cut_clipped_bins={softclip_cut_bins}, "
                f"worst_over_boost={over_boost:.2f} dB, worst_over_cut={over_cut:.2f} dB"
            )
        except Exception:
            softclip_boost_bins, softclip_cut_bins = 0, 0
            over_boost, over_cut = 0.0, 0.0


        # --- CHECKPOINT 4: post_softclip ---
        try:
            stage_probes["post_softclip"] = _stage_probe(
                "post_softclip", freq_axis, tmp, mask_c,
                global_gain_db=float(getattr(cfg, "global_gain_db", 0.0) or 0.0),
                auto_headroom_db=0.0,
                logger_obj=logger
            )
        except Exception:
            pass


        gain_db[mask_c] = tmp[mask_c]

        # --- 8E. Slope/octave limiter (gain curve steepness) ---
        #----------
        # Confidence pull is applied post-slope using a stable raw_safe reference, 
        # ensuring it affects the final filter rather than being overridden by slope limiting.”
        #---------
        # Note: done before exc_prot and run exc_prot again at the end,
        # so slope limiting cannot "leak" boost into protection zones.
        max_slope = float(getattr(cfg, "max_slope_db_per_oct", 24.0) or 0.0)  # legacy (symmetric)
        # NEW: separate limits for boost/cut; <=0 inherits legacy max_slope
        max_slope_boost = float(getattr(cfg, "max_slope_boost_db_per_oct", 0.0) or 0.0)
        max_slope_cut   = float(getattr(cfg, "max_slope_cut_db_per_oct", 0.0) or 0.0)
        if max_slope_boost <= 0.0:
            max_slope_boost = max_slope
        if max_slope_cut <= 0.0:
            max_slope_cut = max_slope

        # Only run if any slope limiting is enabled
        if max_slope > 0 or max_slope_boost > 0 or max_slope_cut > 0:
            # Limit only in correction area, keep outside untouched
            g2 = gain_db.copy()
            try:
                # If equal, keep old behavior (bit-for-bit close) using symmetric limiter
                if max_slope_boost == max_slope_cut and max_slope_boost > 0:
                    g2 = limit_slope_per_octave(freq_axis, g2, max_db_per_oct=float(max_slope_boost))
                else:
                    g2 = limit_slope_per_octave_asym(
                        freq_axis,
                        g2,
                        max_db_per_oct_boost=float(max_slope_boost),
                        max_db_per_oct_cut=float(max_slope_cut),
                    )
            except Exception:
                # Never break the pipeline due to slope limiting
                pass
            gain_db[mask_c] = g2[mask_c]
            _log_stats("gain_db_post_slope", gain_db, mask_c)

            # --- ConfPull POST-SLOPE (makes A/B settings visible in final filter) ---
            try:
                _pre = gain_db.copy()
                gain_db = _apply_confpull_post_slope(gain_db, mask_c, measured_ref_db=raw_safe_ref)
                _log_stats("gain_db_post_confpull", gain_db, mask_c, ref=_pre)
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

        # --- 8E1. Post-shape filter smoothing ---
        # Keep filter_smooth/reg_strength visible in the final filter, even when
        # A-FDW + slope/confpull rails have already reshaped the curve.
        try:
            if np.any(mask_c):
                g0 = np.asarray(gain_db, dtype=float).copy()
                idx = np.where(mask_c)[0]
                if idx.size >= 2:
                    i0, i1 = int(idx[0]), int(idx[-1])
                    # Edge-hold outside correction band to avoid pull toward 0 dB.
                    if i0 > 0:
                        g0[:i0] = g0[i0]
                    if i1 < (g0.size - 1):
                        g0[i1 + 1:] = g0[i1]

                g_sm = smooth_gain_fractional_octave(
                    freq_axis,
                    g0,
                    _filter_smooth,
                )
                mix = float(np.clip(float(getattr(cfg, "reg_strength", 30.0) or 30.0) / 100.0, 0.0, 1.0))
                if mix > 0.0:
                    _pre = gain_db.copy()
                    gain_db[mask_c] = gain_db[mask_c] + (g_sm[mask_c] - gain_db[mask_c]) * mix
                    _log_stats("gain_db_post_filter_smooth", gain_db, mask_c, ref=_pre)
        except Exception:
            pass
        
        f_start = max(cfg.mag_c_max - cfg.trans_width, cfg.mag_c_min)
        
        f_mask = (freq_axis > f_start) & (freq_axis <= cfg.mag_c_max)
        # Varmistetaan jakolasku (ettei jaeta nollalla, jos trans_width on 0)
        fade_len = cfg.mag_c_max - f_start
        if np.any(f_mask) and fade_len > 0: 
            gain_db[f_mask] *= (cfg.mag_c_max - freq_axis[f_mask]) / fade_len
        if cfg.exc_prot:
            # Define transition zone (about 1/2 octave, factor 1.41)
            f_start = cfg.exc_freq
            f_end = cfg.exc_freq * 1.41
            
            # 1. Full protection below f_start: Force boost to zero, allow cuts
            prot_mask = freq_axis < f_start
            gain_db[prot_mask] = np.minimum(gain_db[prot_mask], 0.0)
            
            # 2. Soft transition zone f_start -> f_end
            # In this zone allowed boost rises linearly 0 dB -> max_boost_db
            trans_mask = (freq_axis >= f_start) & (freq_axis <= f_end)
            if np.any(trans_mask):
                # Calculate fade factor (0.0 -> 1.0)
                fade = (freq_axis[trans_mask] - f_start) / (f_end - f_start)
                # Maximum allowed boost at this frequency
                allowed_boost = fade * cfg.max_boost_db
                # Limit boost, but keep all attenuations (cuts)
                gain_db[trans_mask] = np.minimum(gain_db[trans_mask], allowed_boost)
            
            logger.info(f"Exc Prot: Full protection < {f_start}Hz, Soft fade up to {f_end:.1f}Hz.")

        # --- 8E2. WAV-only ripple cleanup near correction upper edge ---
        # Keep TXT path untouched; only smooth a narrow zone around mag_c_max.
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

                        _log_stats("gain_db_post_wav_transition_smooth", gain_db, mask_c, ref=_pre)
                        if isinstance(st, dict):
                            st["wav_transition_smoothing"] = True
                            st["wav_transition_smoothing_zone_hz"] = [float(f_lo), float(f_hi)]
        except Exception:
            pass

        # NOTE (HPF):
        # Do NOT "zero correction" below HPF. That is not a high-pass filter.
        # Real HPF magnitude is applied later via gain_db += hpf_db.

        # --- CHECKPOINT 5: after_fade (place right AFTER your fade/transition operations) ---
        # If your fade is applied on gain_db, this is correct. If it is applied on another array, move accordingly.
        try:
            stage_probes["after_fade"] = _stage_probe(
                "after_fade", freq_axis, gain_db, mask_c,
                global_gain_db=float(getattr(cfg, "global_gain_db", 0.0) or 0.0),
                auto_headroom_db=0.0,
                logger_obj=logger
            )
        except Exception:
            pass

        # --- CHECKPOINT 6: after_slope (place right AFTER your slope-limit operation) ---
        # IMPORTANT: If slope-limit happens earlier than fade in your code, move this probe accordingly.
        # This stays here as a fallback; you should move it to the actual slope-limit section if different.
        try:
            if "after_slope" not in stage_probes:
                stage_probes["after_slope"] = _stage_probe(
                    "after_slope", freq_axis, gain_db, mask_c,
                    global_gain_db=float(getattr(cfg, "global_gain_db", 0.0) or 0.0),
                    auto_headroom_db=0.0,
                    logger_obj=logger
                )
        except Exception:
            pass

        
        # --- 8F. Final safety clamp (max boost / max cut) ---
        # Ensure no later operation (fade/slope/exc_prot) exceeds limits.
        max_cut_db = float(getattr(cfg, "max_cut_db", 15.0))
        max_cut_db = abs(float(getattr(cfg, "max_cut_db", 15.0) or 15.0))


        # --- CHECKPOINT 7: post_hardclamp ---
        try:
            stage_probes["post_hardclamp"] = _stage_probe(
                "post_hardclamp", freq_axis, gain_db, mask_c,
                global_gain_db=float(getattr(cfg, "global_gain_db", 0.0) or 0.0),
                auto_headroom_db=0.0,
                logger_obj=logger
            )
        except Exception:
            pass

        # --- 8F-b. Clamp diagnostics: FINAL HARD CLAMP stage ---
        try:
            _pre_hard = gain_db.copy()
            _max_boost2 = float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
            _max_cut2 = float(max_cut_db)
        except Exception:
            _pre_hard = gain_db
            _max_boost2, _max_cut2 = 0.0, float(max_cut_db)

        gain_db = np.minimum(gain_db, float(getattr(cfg, "max_boost_db", 0.0) or 0.0))
        gain_db = np.maximum(gain_db, -max_cut_db)

        try:
            if np.any(mask_c):
                hardclamp_boost_bins = int(np.sum((_pre_hard[mask_c] > (_max_boost2 + 1e-9)) & (gain_db[mask_c] <= (_max_boost2 + 1e-9))))
                hardclamp_cut_bins   = int(np.sum((_pre_hard[mask_c] < (-_max_cut2 - 1e-9)) & (gain_db[mask_c] >= (-_max_cut2 - 1e-9))))
                hard_over_boost = max(0.0, float(np.max(_pre_hard[mask_c] - _max_boost2)))
                hard_over_cut   = max(0.0, float(np.max((-_pre_hard[mask_c]) - _max_cut2)))
                _band_bins = int(np.sum(mask_c))
            else:
                hardclamp_boost_bins, hardclamp_cut_bins = 0, 0
                hard_over_boost, hard_over_cut = 0.0, 0.0
                _band_bins = 0
            logger.info(
                "Clamp: hard_clamp "
                f"(max_boost={_max_boost2:.2f} dB, max_cut={_max_cut2:.2f} dB) -> "
                f"boost_clipped_bins={hardclamp_boost_bins}, cut_clipped_bins={hardclamp_cut_bins}, "
                f"worst_over_boost={hard_over_boost:.2f} dB, worst_over_cut={hard_over_cut:.2f} dB"
            )

            # Clamp-dominance indicator: if high, many settings (e.g. smoothing) may look similar.
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

        # --- 8F1. Clamp-aware final smoothing ---
        # If hard clamp is active, smooth the already-limited curve and re-apply limits.
        # This preserves safety while making filter_smooth/reg_strength visible in final export.
        try:
            _clamp_active = bool((hardclamp_boost_bins > 0) or (hardclamp_cut_bins > 0))
        except Exception:
            _clamp_active = False
        try:
            if _clamp_active and np.any(mask_c):
                g0 = np.asarray(gain_db, dtype=float).copy()
                idx = np.where(mask_c)[0]
                if idx.size >= 2:
                    i0, i1 = int(idx[0]), int(idx[-1])
                    if i0 > 0:
                        g0[:i0] = g0[i0]
                    if i1 < (g0.size - 1):
                        g0[i1 + 1:] = g0[i1]

                g_sm = smooth_gain_fractional_octave(
                    freq_axis,
                    g0,
                    _filter_smooth,
                )
                mix = float(np.clip(float(getattr(cfg, "reg_strength", 30.0) or 30.0) / 100.0, 0.0, 1.0))
                if mix > 0.0:
                    _pre = gain_db.copy()
                    gain_db[mask_c] = gain_db[mask_c] + (g_sm[mask_c] - gain_db[mask_c]) * mix
                    # Re-apply hard safety limits after polish.
                    gain_db = np.minimum(gain_db, float(getattr(cfg, "max_boost_db", 0.0) or 0.0))
                    gain_db = np.maximum(gain_db, -max_cut_db)
                    _log_stats("gain_db_post_final_clamp_smooth", gain_db, mask_c, ref=_pre)
        except Exception:
            pass

        # --- 8F2. WAV-only final ripple polish near correction edge ---
        # Applied at the end so later clamp/smoothing stages won't reintroduce ripple.
        try:
            if bool(getattr(cfg, "is_wav_source", False)) and np.any(mask_c):
                cmin = float(getattr(cfg, "mag_c_min", 0.0) or 0.0)
                cmax = float(getattr(cfg, "mag_c_max", 0.0) or 0.0)
                tw = float(getattr(cfg, "trans_width", 0.0) or 0.0)
                if np.isfinite(cmin) and np.isfinite(cmax) and (cmax > cmin):
                    if not np.isfinite(tw) or tw <= 0.0:
                        tw = max(50.0, 0.4 * cmax)

                    # Wider than 8E2 to catch the visible 200-300 Hz comb/ripple region.
                    f_lo = max(cmin, cmax - 0.95 * tw)
                    f_hi = min(float(np.max(freq_axis)), cmax + 1.45 * tw)
                    zone = (freq_axis >= f_lo) & (freq_axis <= f_hi)
                    if int(np.count_nonzero(zone)) >= 8:
                        _pre = gain_db.copy()
                        g0 = np.asarray(gain_db, dtype=float).copy()
                        # Stronger and deterministic smoothing in Hz domain.
                        sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=8.0, fallback_bins=12.0)
                        g_sm = scipy.ndimage.gaussian_filter1d(g0, sigma=float(max(2.0, sigma_bins)))

                        x = np.zeros_like(g0, dtype=float)
                        span = max(1e-9, float(f_hi - f_lo))
                        x[zone] = np.clip((freq_axis[zone] - f_lo) / span, 0.0, 1.0)
                        w = np.zeros_like(g0, dtype=float)
                        w[zone] = 0.5 - 0.5 * np.cos(np.pi * x[zone])

                        # Apply mostly full blend inside the zone to ensure ripple attenuation.
                        mix = 0.95
                        gain_db = g0 + (g_sm - g0) * (mix * w)

                        # Keep safety limits intact.
                        gain_db = np.minimum(gain_db, float(getattr(cfg, "max_boost_db", 0.0) or 0.0))
                        gain_db = np.maximum(gain_db, -max_cut_db)

                        _log_stats("gain_db_post_wav_final_ripple_polish", gain_db, mask_c, ref=_pre)
                        if isinstance(st, dict):
                            st["wav_final_ripple_polish"] = True
                            st["wav_final_ripple_polish_zone_hz"] = [float(f_lo), float(f_hi)]
        except Exception:
            pass

        # --- 8F-a. Diagnostic: post-clamp boost/cut summary ---
        try:
            if np.any(mask_c):
                boost_peak_db = float(np.max(gain_db[mask_c]))
                cut_peak_db = float(np.min(gain_db[mask_c]))
                n_boost = int(np.sum((gain_db > 1e-6) & mask_c))
            else:
                boost_peak_db, cut_peak_db, n_boost = 0.0, 0.0, 0
            logger.info(
                "Diagnostic: "
                f"boost_peak={boost_peak_db:.2f} dB, cut_peak={cut_peak_db:.2f} dB, "
                f"boost_bins={n_boost}, "
                f"boost_candidate_peak={float(boost_cand_peak):.2f} dB, "
                f"boost_candidate_bins={int(n_boost_cand)}, "
                f"boost_candidate_bins_lowbass={int(n_boost_cand_low)}, "
                f"boost_candidate_bins_excprot={int(n_boost_cand_exc)}"
            )
        except Exception:
            boost_peak_db, cut_peak_db, n_boost = 0.0, 0.0, 0


    return {
        "current_rt60": locals().get("current_rt60", 0.0),
        "rt60_bands": locals().get("rt60_bands", {}),
        "band_avg": locals().get("band_avg", 0.0),
        "target_mags": locals().get("target_mags"),
        "hpf_f": locals().get("hpf_f", 0.0),
        "hpf_order": locals().get("hpf_order", 0),
        "target_level_db": locals().get("target_level_db", 0.0),
        "calc_offset_db": locals().get("calc_offset_db", 0.0),
        "meas_level_db_window": locals().get("meas_level_db_window", 0.0),
        "target_level_db_window": locals().get("target_level_db_window", 0.0),
        "offset_method": locals().get("offset_method", "Init"),
        "s_min": locals().get("s_min", float(getattr(cfg, "lvl_min", 500.0) or 500.0)),
        "s_max": locals().get("s_max", float(getattr(cfg, "lvl_max", 2000.0) or 2000.0)),
        "target_shift_db": locals().get("target_shift_db", 0.0),
        "cmp": locals().get("cmp", cmp),
        "analysis_mode": locals().get("analysis_mode", analysis_mode),
        "gain_db": locals().get("gain_db", gain_db),
        "afdw_on": locals().get("afdw_on", False),
        "base_sigma": locals().get("base_sigma"),
        "_filter_smooth": locals().get("_filter_smooth"),
        "df_mode": locals().get("df_mode"),
        "raw_g": locals().get("raw_g"),
        "final_g": locals().get("final_g"),
        "mask_c": locals().get("mask_c", np.zeros_like(freq_axis, dtype=bool)),
        "stage_probes": locals().get("stage_probes", {}),
        "use_bassfirst": locals().get("use_bassfirst", False),
        "bf_room_mode": locals().get("bf_room_mode"),
        "bf_rel": locals().get("bf_rel"),
        "bf_conf_for_smoothing": locals().get("bf_conf_for_smoothing"),
        "boost_peak_db": locals().get("boost_peak_db", 0.0),
        "cut_peak_db": locals().get("cut_peak_db", 0.0),
        "n_boost": locals().get("n_boost", 0),
        "boost_cand_peak": locals().get("boost_cand_peak", 0.0),
        "n_boost_cand": locals().get("n_boost_cand", 0),
        "n_boost_cand_low": locals().get("n_boost_cand_low", 0),
        "n_boost_cand_exc": locals().get("n_boost_cand_exc", 0),
        "softclip_boost_bins": locals().get("softclip_boost_bins", 0),
        "softclip_cut_bins": locals().get("softclip_cut_bins", 0),
        "over_boost": locals().get("over_boost", 0.0),
        "over_cut": locals().get("over_cut", 0.0),
        "hardclamp_boost_bins": locals().get("hardclamp_boost_bins", 0),
        "hardclamp_cut_bins": locals().get("hardclamp_cut_bins", 0),
        "hard_over_boost": locals().get("hard_over_boost", 0.0),
        "hard_over_cut": locals().get("hard_over_cut", 0.0),
        "clamp_dominance_level": locals().get("clamp_dominance_level", "NONE"),
    }
