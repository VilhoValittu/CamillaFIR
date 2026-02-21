from __future__ import annotations

import math
import numpy as np
import scipy.fft
import scipy.ndimage

from .analysis import _sigma_bins_from_hz
from .limits import limit_slope_per_octave, limit_slope_per_octave_asym, soft_clip_gain
from .phase import calculate_minimum_phase, calculate_theoretical_phase, combine_mixed_phase
from .smoothing import smooth_gain_fractional_octave


def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _mixed_excess_weight(freqs: np.ndarray, full_hz: float, none_hz: float) -> np.ndarray:
    f = np.asarray(freqs, dtype=float)
    w = np.zeros_like(f, dtype=float)
    valid = np.isfinite(f) & (f > 0.0)
    if not np.any(valid):
        return w

    f_full = float(max(20.0, full_hz))
    f_none = float(max(f_full + 1.0, none_hz))

    low = valid & (f <= f_full)
    w[low] = 1.0
    mid = valid & (f > f_full) & (f < f_none)
    if np.any(mid):
        x = (f[mid] - f_full) / (f_none - f_full)
        w[mid] = 0.5 * (1.0 + np.cos(np.pi * x))
    return np.clip(w, 0.0, 1.0)


def _max_abs_group_delay_ms(freqs: np.ndarray, phase_rad: np.ndarray, mask: np.ndarray | None = None) -> float:
    f = np.asarray(freqs, dtype=float)
    p = np.unwrap(np.asarray(phase_rad, dtype=float))
    if f.size < 4 or p.size != f.size:
        return 0.0
    w = 2.0 * np.pi * f
    dw = np.gradient(w) + 1e-30
    gd_ms = (-np.gradient(p) / dw) * 1000.0
    if mask is None:
        sel = np.isfinite(gd_ms)
    else:
        sel = np.asarray(mask, dtype=bool) & np.isfinite(gd_ms)
    if not np.any(sel):
        return 0.0
    try:
        return float(np.max(np.abs(gd_ms[sel])))
    except Exception:
        return 0.0


def _pre_ringing_db(ir: np.ndarray) -> float:
    x = np.asarray(ir, dtype=float)
    if x.size < 8:
        return float("-inf")
    peak = int(np.argmax(np.abs(x)))
    if peak <= 0:
        return float("-inf")
    pre_e = float(np.sum(np.square(x[:peak])))
    post_e = float(np.sum(np.square(x[peak + 1:])))
    return float(10.0 * np.log10((pre_e + 1e-30) / (post_e + 1e-30)))


def run_phase_ir_stage(
    *,
    cfg,
    freq_axis,
    n_fft,
    gain_db,
    p_rad_interp,
    conf_mask,
    m_anal,
    calc_offset_db,
    target_mags,
    st,
    mask_c,
    base_sigma,
    _filter_smooth,
    df_mode,
    raw_g,
    final_g,
    use_bassfirst,
    afdw_on,
    logger,
    apply_hpf_to_mags_fn,
    limit_gd_gradient_ms_per_oct_fn,
    cfg_float_allow_zero_fn,
):
    apply_hpf_to_mags = apply_hpf_to_mags_fn
    _limit_gd_gradient_ms_per_oct = limit_gd_gradient_ms_per_oct_fn
    _cfg_float_allow_zero = cfg_float_allow_zero_fn

    # --- 9. GENERATE PHASE ---
    # --- THEORETICAL PHASE (single source of truth) ---
    hpf_freq = None
    hpf_slope = None
    hs = cfg.hpf_settings

    if isinstance(hs, dict) and hs.get('enabled'):
        hpf_freq = float(hs.get('freq', 0.0) or 0.0)
        hpf_order = int(hs.get('order', 0) or 0)
        if hpf_freq > 0 and hpf_order > 0:
            hpf_slope = float(hpf_order * 6)  # dB/oct

    # --- DEBUG: what XO/HPF are we modeling? ---
    try:
        xo_list = getattr(cfg, "crossovers", None) or []
        if xo_list:
            xo_txt = ", ".join([
                f"{float(x.get('freq',0.0)):.1f}Hz/{int(x.get('slope', int(x.get('order',1))*6))}dB/oct"
                for x in xo_list if x.get("freq", None) is not None
            ])
        else:
            xo_txt = "off"
        if hpf_freq and hpf_slope and float(hpf_freq) > 0 and float(hpf_slope) > 0:
            hpf_txt = f"{float(hpf_freq):.1f}Hz/{int(round(float(hpf_slope)))}dB/oct"
        else:
            hpf_txt = "off"
        logger.info(f"Phase model: XO={xo_txt} | HPF={hpf_txt}")
    except Exception:
        xo_txt = "n/a"
        hpf_txt = "n/a"

    # RAW theoretical phase model (UNCLAMPED for debug indicators)
    theo_on_raw = calculate_theoretical_phase(
        freq_axis,
        cfg.crossovers,
        hpf_freq=hpf_freq,
        hpf_slope=hpf_slope,
        max_phase_deg=None
    )
    theo_off_raw = calculate_theoretical_phase(
        freq_axis,
        [],               # no XO
        hpf_freq=None,    # no HPF
        hpf_slope=None,
        max_phase_deg=None
    )

    # Main theoretical phase model used in correction.
    # Keep this UNCLAMPED so only excess-phase correction is safety-limited later.
    theo_xo = calculate_theoretical_phase(
        freq_axis,
        cfg.crossovers,
        hpf_freq=hpf_freq,
        hpf_slope=hpf_slope,
        max_phase_deg=None
    )
    # --- XO/HPF ON/OFF diff indicator (debug-friendly, relevant band) ---
    try:
        xo_list = getattr(cfg, "crossovers", None) or []
        hpf_on = (hpf_freq and hpf_slope and float(hpf_freq) > 0 and float(hpf_slope) > 0)

        if xo_list or hpf_on:
            f = np.asarray(freq_axis, float)

            # IMPORTANT: use UNCLAMPED theoretical phase for "raw" indicator
            theo_on_raw = calculate_theoretical_phase(
                f,
                xo_list,
                hpf_freq=hpf_freq,
                hpf_slope=hpf_slope,
                max_phase_deg=None
            )
            theo_off_raw = calculate_theoretical_phase(
                f,
                [],
                hpf_freq=None,
                hpf_slope=None,
                max_phase_deg=None
            )

            # RAW model delta (really no clamp)
            dphi_raw = np.unwrap(np.asarray(theo_on_raw, float) - np.asarray(theo_off_raw, float))
            dphi_raw_deg = np.rad2deg(dphi_raw)

            # Wrap to [-180, 180) so "max phase" doesn't get dominated by unwrap turns / band edges
            def _wrap_deg(x_deg: np.ndarray) -> np.ndarray:
                return (x_deg + 180.0) % 360.0 - 180.0

            dphi_wrapped_deg = _wrap_deg(dphi_raw_deg)

            # Build masks separately so HPF doesn't dominate XO
            xo_mask = np.zeros_like(f, dtype=bool)
            xo_band_masks = []  # list of (fc_hz, mask)
            for xo in xo_list:
                try:
                    fc = float(xo.get("freq", 0.0) or 0.0)
                except Exception:
                    continue
                if fc > 0:
                    m = (f >= fc / 4.0) & (f <= fc * 4.0)
                    xo_mask |= m
                    xo_band_masks.append((fc, m))

            hpf_mask = np.zeros_like(f, dtype=bool)
            if hpf_on:
                fc = float(hpf_freq)
                # HPF phase effect is most meaningful below/around fc -> focus a bit tighter
                hpf_mask |= (f >= max(fc / 8.0, 10.0)) & (f <= fc * 2.0)

            # Fallbacks
            wide_mask = np.isfinite(f) & (f >= 20.0) & (f <= float(np.nanmax(f)))
            if np.count_nonzero(xo_mask) < 8:
                xo_mask = wide_mask
            if np.count_nonzero(hpf_mask) < 8:
                hpf_mask = wide_mask if hpf_on else np.zeros_like(f, dtype=bool)

            def _max_abs_in_mask(arr, mask):
                a = np.asarray(arr, float)
                m = np.asarray(mask, bool)
                if np.count_nonzero(m) < 8:
                    return None, None
                aa = np.where(m, np.abs(a), -1.0)
                idx = int(np.argmax(aa))
                if aa[idx] < 0:
                    return None, None
                return float(np.abs(a[idx])), float(f[idx])

            # Phase maxima (XO only; also identify which XO band "won")
            xo_phi, xo_phi_hz = _max_abs_in_mask(dphi_wrapped_deg, xo_mask)
            if xo_phi is not None:
                st["xo_diff_raw_max_phase_deg"] = xo_phi
                st["xo_diff_raw_max_phase_hz"] = xo_phi_hz

                # Determine which XO band contains the winning frequency (closest fc if overlap)
                best_fc = None
                if xo_band_masks:
                    # Find all fc where the winning frequency is in that band mask
                    candidates = []
                    for fc, m in xo_band_masks:
                        try:
                            idx_win = int(np.argmin(np.abs(f - xo_phi_hz)))
                        except Exception:
                            continue
                        if bool(m[idx_win]):
                            candidates.append(fc)
                    if candidates:
                        # pick the closest fc to the winning frequency
                        best_fc = float(min(candidates, key=lambda c: abs(float(c) - float(xo_phi_hz))))
                    else:
                        # fallback: closest fc overall
                        best_fc = float(min([fc for fc, _ in xo_band_masks], key=lambda c: abs(float(c) - float(xo_phi_hz))))
                if best_fc is not None:
                    st["xo_diff_raw_max_phase_xo_fc_hz"] = float(best_fc)

            # Per-XO phase delta at fc (wrapped) — sanity check / debug gold
            for i, xo in enumerate(xo_list, start=1):
                try:
                    fc = float(xo.get("freq", 0.0) or 0.0)
                except Exception:
                    continue
                if fc <= 0:
                    continue
                idx_fc = int(np.argmin(np.abs(f - fc)))
                st[f"xo{i}_dphi_wrapped_deg@fc"] = float(dphi_wrapped_deg[idx_fc])

            # Per-XO GD delta at fc (RAW, ms) — correlates best with transient tightness
            for i, xo in enumerate(xo_list, start=1):
                try:
                    fc = float(xo.get("freq", 0.0) or 0.0)
                except Exception:
                    continue
                if fc <= 0:
                    continue
                idx_fc = int(np.argmin(np.abs(f - fc)))
                try:
                    st[f"xo{i}_dgd_ms@fc"] = float(dgd[idx_fc])
                except Exception:
                    pass

            hpf_phi, hpf_phi_hz = _max_abs_in_mask(dphi_raw_deg, hpf_mask)
            if hpf_on and hpf_phi is not None:
                st["hpf_diff_raw_max_phase_deg"] = hpf_phi
                st["hpf_diff_raw_max_phase_hz"] = hpf_phi_hz

            # GD delta (RAW) from RAW phases
            w = 2.0 * math.pi * f
            dw = np.gradient(w) + 1e-30
            ph_on = np.unwrap(np.asarray(theo_on_raw, float))
            ph_off = np.unwrap(np.asarray(theo_off_raw, float))
            gd_on = (-np.gradient(ph_on) / dw) * 1000.0
            gd_off = (-np.gradient(ph_off) / dw) * 1000.0
            dgd = gd_on - gd_off

            # Per-XO GD delta at fc (RAW, ms) — best proxy for "transient tightness"
            for i, xo in enumerate(xo_list, start=1):
                try:
                    fc = float(xo.get("freq", 0.0) or 0.0)
                except Exception:
                    continue
                if fc <= 0:
                    continue
                idx_fc = int(np.argmin(np.abs(f - fc)))
                try:
                    st[f"xo{i}_dgd_ms@fc"] = float(dgd[idx_fc])
                except Exception:
                    pass

            # GD maxima (separately)
            xo_gd, xo_gd_hz = _max_abs_in_mask(dgd, xo_mask)
            if xo_gd is not None:
                st["xo_diff_raw_max_gd_ms"] = xo_gd
                st["xo_diff_raw_max_gd_hz"] = xo_gd_hz
                # Optional: which XO band "won" for GD too (same idea)
                try:
                    best_fc_gd = None
                    if xo_band_masks:
                        idx_win = int(np.argmin(np.abs(f - xo_gd_hz)))
                        candidates = [fc for fc, m in xo_band_masks if bool(m[idx_win])]
                        if candidates:
                            best_fc_gd = float(min(candidates, key=lambda c: abs(float(c) - float(xo_gd_hz))))
                        else:
                            best_fc_gd = float(min([fc for fc, _ in xo_band_masks], key=lambda c: abs(float(c) - float(xo_gd_hz))))
                    if best_fc_gd is not None:
                        st["xo_diff_raw_max_gd_xo_fc_hz"] = float(best_fc_gd)
                except Exception:
                    pass

            hpf_gd, hpf_gd_hz = _max_abs_in_mask(dgd, hpf_mask)
            if hpf_on and hpf_gd is not None:
                st["hpf_diff_raw_max_gd_ms"] = hpf_gd
                st["hpf_diff_raw_max_gd_hz"] = hpf_gd_hz

    except Exception as e:
        logger.debug("XO raw diff indicator failed: %s", e)



    # Expose to stats/Summary (human readable)
    try:
        st["xo_summary"] = xo_txt
        st["hpf_summary"] = hpf_txt
        # small checkpoint samples (degrees) for debugging
        for fchk in (20.0, 80.0, 200.0, 1000.0, 5000.0):
            idx = int(np.argmin(np.abs(freq_axis - fchk)))
            st[f"theo_xo_deg@{int(fchk)}Hz"] = float(np.rad2deg(theo_xo[idx]))
    except Exception:
        pass

    # --- Phase correction limit (Hz) with smooth blend ---
    # Goal:
    #   - below phase_limit: keep measured/room phase (TOF removed) in p_rad_interp
    #   - above phase_limit*1.41 (~1/2 octave): use only theoretical XO/HPF phase (theo_xo)
    #   - between: smooth crossfade (prevents GD "bump" from hard cut)
    try:
        phase_lim = float(getattr(cfg, "phase_limit", 0.0) or 0.0)
        if phase_lim > 0.0 and freq_axis.size == theo_xo.size:
            f0 = float(phase_lim)
            f1 = float(phase_lim) * 1.41  # ~1/2 octave

            # Clamp transition to valid frequency span
            f_min = float(freq_axis[0]) if freq_axis.size else 0.0
            f_max = float(freq_axis[-1]) if freq_axis.size else 0.0
            f0 = max(f0, max(f_min, 1e-6))
            f1 = min(f1, f_max)

            if f1 <= f0:
                # Degenerate case: act like hard limit
                mask = freq_axis <= f0
                p_rad_interp = np.where(mask, p_rad_interp, theo_xo)
            else:
                # Crossfade weight w: 0 below f0, 1 above f1
                w = (freq_axis - f0) / (f1 - f0)
                w = np.clip(w, 0.0, 1.0)

                # Blend phase itself (radians). Both are already unwrapped-ish and TOF-removed upstream.
                p_rad_interp = (1.0 - w) * p_rad_interp + w * theo_xo
    except Exception:
        pass

    # --- LOGGING: HPF inclusion status ---
    if hpf_freq and hpf_slope:
        logger.info(
            f"Theoretical phase includes HPF: f={hpf_freq:.1f} Hz, "
            f"slope={hpf_slope:.0f} dB/oct (order={int(hpf_slope/6)})"
        )
    else:
        logger.info("Theoretical phase: HPF not included")
    
    if getattr(cfg, "phase_safe_2058", False):
        logger.info("Phase mode: 2058-safe (no room phase correction)")
    else:
        logger.info("Phase mode: modern (excess-phase + confidence + FDW)")

    # --- 9A. HPF magnitude into FIR (enabled-check) ---
    if isinstance(hs, dict) and hs.get('enabled'):
        hpf_f = float(hs.get('freq', 0.0) or 0.0)
        hpf_order = int(hs.get('order', 0) or 0)
        if hpf_f > 0 and hpf_order > 0:
            hpf_db = apply_hpf_to_mags(freq_axis, np.zeros_like(freq_axis), hpf_f, hpf_order)
            gain_db = gain_db + hpf_db
            try:
                logger.info(
                    "HPF magnitude applied to FIR: "
                    f"fc={hpf_f:.1f} Hz, "
                    f"order={hpf_order} "
                    f"({hpf_order * 6:.0f} dB/oct)"
                )
            except Exception:
                pass

    # --- 8G/9A2. Optional residual magnitude pass (2-pass target tracking) ---
    # Improves target tracking without making correction aggressive:
    #   pred0 = (m_anal - calc_offset_db) + gain_db
    #   residual = target_mags - pred0
    #   apply broad smoothing + confidence gating, add fraction back to gain_db
    # Then re-apply the same magnitude safety constraints (lowbass policy, soft/hard clamp, slope, fades).
    if bool(getattr(cfg, "enable_residual_pass", False)) and bool(getattr(cfg, "enable_mag_correction", True)):
        try:
            def _reapply_mag_constraints(_g):
                """Re-apply the same post-shaping constraints after residual tweaks (UI/DSP consistency)."""
                _g = np.asarray(_g, dtype=float).copy()

                # Keep low-bass 'no boost + stronger cut' policy intact
                try:
                    low_hz = _cfg_float_allow_zero(cfg, "low_bass_cut_hz", 40.0)
                    low_mask = mask_c & (freq_axis > 0) & (freq_axis <= low_hz)
                    if np.any(low_mask):
                        # no boost below low_hz
                        _g[low_mask] = np.minimum(_g[low_mask], 0.0)
                        # If raw_g/final_g exist (mag correction path), keep 'stronger cut' rule
                        if 'raw_g' in locals() and 'final_g' in locals():
                            low_cut = np.minimum(final_g[low_mask], raw_g[low_mask])  # stronger cut
                            low_cut = np.minimum(low_cut, 0.0)
                            _g[low_mask] = np.minimum(_g[low_mask], low_cut)
                except Exception:
                    pass

                # Soft clip (same as 8D)
                try:
                    max_cut_db = abs(float(getattr(cfg, "max_cut_db", 15.0) or 15.0))
                    _g = soft_clip_gain(
                        _g,
                        float(getattr(cfg, "max_boost_db", 0.0) or 0.0),
                        max_cut_db
                    )
                except Exception:
                    pass

                # Slope limiter (same as 8E)
                try:
                    max_slope = float(getattr(cfg, "max_slope_db_per_oct", 24.0) or 0.0)  # legacy symmetric
                    max_slope_boost = float(getattr(cfg, "max_slope_boost_db_per_oct", 0.0) or 0.0)
                    max_slope_cut   = float(getattr(cfg, "max_slope_cut_db_per_oct", 0.0) or 0.0)
                    if max_slope_boost <= 0.0:
                        max_slope_boost = max_slope
                    if max_slope_cut <= 0.0:
                        max_slope_cut = max_slope

                    if max_slope > 0 or max_slope_boost > 0 or max_slope_cut > 0:
                        if max_slope_boost == max_slope_cut and max_slope_boost > 0:
                            _g = limit_slope_per_octave(freq_axis, _g, max_db_per_oct=float(max_slope_boost))
                        else:
                            _g = limit_slope_per_octave_asym(
                                freq_axis,
                                _g,
                                max_db_per_oct_boost=float(max_slope_boost),
                                max_db_per_oct_cut=float(max_slope_cut),
                            )
                except Exception:
                    pass

                # Fade to zero near mag_c_max (same as post-8E fade)
                try:
                    mag_c_min = float(getattr(cfg, "mag_c_min", 0.0) or 0.0)
                    mag_c_max = float(getattr(cfg, "mag_c_max", 0.0) or 0.0)
                    trans_w   = float(getattr(cfg, "trans_width", 0.0) or 0.0)
                    if mag_c_max > 0 and trans_w > 0:
                        f_start = max(mag_c_max - trans_w, mag_c_min)
                        f_mask = (freq_axis > f_start) & (freq_axis <= mag_c_max)
                        fade_len = mag_c_max - f_start
                        if np.any(f_mask) and fade_len > 0:
                            _g[f_mask] *= (mag_c_max - freq_axis[f_mask]) / fade_len
                except Exception:
                    pass

                # Exc-protection (same as existing exc_prot re-application)
                try:
                    if bool(getattr(cfg, "exc_prot", False)):
                        f_start = float(getattr(cfg, "exc_freq", 0.0) or 0.0)
                        if f_start > 0:
                            f_end = f_start * 1.41
                            prot_mask = freq_axis < f_start
                            _g[prot_mask] = np.minimum(_g[prot_mask], 0.0)
                            trans_mask = (freq_axis >= f_start) & (freq_axis <= f_end)
                            if np.any(trans_mask):
                                fade = (freq_axis[trans_mask] - f_start) / (f_end - f_start)
                                allowed_boost = fade * float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
                                _g[trans_mask] = np.minimum(_g[trans_mask], allowed_boost)
                except Exception:
                    pass

                # NOTE (HPF):
                # No "HPF policy fade" here. Real HPF magnitude is applied once (gain_db += hpf_db),
                # so residual constraints should not zero/shape below fc.

                # Final hard clamp (same as 8F)
                try:
                    max_cut_db2 = abs(float(getattr(cfg, "max_cut_db", 15.0) or 15.0))
                    _g = np.minimum(_g, float(getattr(cfg, "max_boost_db", 0.0) or 0.0))
                    _g = np.maximum(_g, -max_cut_db2)
                except Exception:
                    pass

                return _g

            # Predicted response from current gain
            measured_aligned = (m_anal - calc_offset_db)
            pred0 = measured_aligned + gain_db
            resid0 = (target_mags - pred0)

            resid = np.zeros_like(gain_db, dtype=float)
            resid[mask_c] = resid0[mask_c]

            # Confidence gating (prefer reliable bins)
            k = float(getattr(cfg, "residual_conf_power", 2.0) or 2.0)
            try:
                if conf_mask is not None:
                    resid[mask_c] *= np.clip(conf_mask[mask_c], 0.0, 1.0) ** k
            except Exception:
                pass

            # Broad smoothing: avoid chasing narrow combing
            strength = float(getattr(cfg, "residual_strength", 0.6) or 0.6)
            strength = float(np.clip(strength, 0.0, 1.0))
            mult = float(getattr(cfg, "residual_smoothing_mult", 2.0) or 2.0)
            mult = max(1.0, float(mult))

            # Use same smoothing model as raw_g smoothing (df_smoothing aware)
            _base_sigma = locals().get('base_sigma', 60 // (_filter_smooth / 12 if _filter_smooth > 0 else 1))
            _df_mode = bool(locals().get('df_mode', bool(getattr(cfg, "df_smoothing", False))))
            if _df_mode:
                df_ref = 44100.0 / 65536.0
                sigma_hz = float(_base_sigma) * df_ref * mult
                sigma_bins = _sigma_bins_from_hz(
                    freq_axis,
                    sigma_hz=sigma_hz,
                    fallback_bins=max(2.0, float(_base_sigma) * mult)
                )
                resid_sm = scipy.ndimage.gaussian_filter1d(resid, sigma=float(sigma_bins))
            else:
                resid_sm = smooth_gain_fractional_octave(
                    freq_axis,
                    resid,
                    _filter_smooth,
                    mult=mult,
                )

            # Apply fraction of residual only in correction band
            gain_db[mask_c] = gain_db[mask_c] + (resid_sm[mask_c] * strength)

            # Re-apply constraints to keep all safety rules intact
            gain_db = _reapply_mag_constraints(gain_db)

            try:
                if isinstance(st, dict):
                    st["residual_pass_enabled"] = True
                    st["residual_strength"] = float(strength)
                    st["residual_smoothing_mult"] = float(mult)
                    st["residual_conf_power"] = float(k)
            except Exception:
                pass
        except Exception:
            pass

    # --- 9B. AUTO OUTPUT LEVEL (realized max boost + user margin) ---
    # `gain` UI value is treated as a non-negative safety margin in dB.
    # Auto level is then:
    #   auto_global_gain_db = -(realized_max_boost + gain_margin_db)
    # This keeps net filter boost <= -margin in the realized response.
    try:
        _g_all = np.asarray(gain_db, dtype=float)
        _g_all = _g_all[np.isfinite(_g_all)]
        current_peak_gain = float(np.max(_g_all)) if _g_all.size else 0.0
    except Exception:
        current_peak_gain = 0.0

    try:
        gain_margin_db = float(
            getattr(cfg, "auto_gain_margin_db", getattr(cfg, "global_gain_db", 0.0)) or 0.0
        )
    except Exception:
        gain_margin_db = 0.0
    if (not np.isfinite(gain_margin_db)) or (gain_margin_db < 0.0):
        gain_margin_db = 0.0

    auto_headroom_db = 0.0
    auto_global_gain_db = 0.0
    try:
        if bool(getattr(cfg, "debug_stage_stats", True)):
            _v = np.asarray(gain_db, dtype=float)
            _m = np.asarray(mask_c, dtype=bool)
            _vv = _v[_m] if (_m.shape == _v.shape and np.any(_m)) else _v
            _vv = _vv[np.isfinite(_vv)]
            if _vv.size >= 4:
                logger.info(
                    "StageStats: gain_db_pre_headroom: "
                    f"max={float(np.max(_vv)):.3f} dB, "
                    f"min={float(np.min(_vv)):.3f} dB, "
                    f"rms={float(np.sqrt(np.mean(_vv * _vv))):.3f} dB"
                )
    except Exception:
        pass

    # Stereo-link pass can force one shared auto gain for both channels.
    try:
        _override = getattr(cfg, "auto_gain_db_override", None)
        if _override is None:
            raise ValueError("no override")
        auto_global_gain_db = float(_override)
        if not np.isfinite(auto_global_gain_db):
            raise ValueError("non-finite override")
        logger.info(
            f"Auto Level: using shared override {auto_global_gain_db:.2f} dB "
            f"(peak={current_peak_gain:.2f} dB, margin={gain_margin_db:.2f} dB)"
        )
    except Exception:
        auto_global_gain_db = -max(0.0, float(current_peak_gain)) - float(gain_margin_db)
        logger.info(
            f"Auto Level: peak={current_peak_gain:.2f} dB + margin={gain_margin_db:.2f} dB "
            f"-> auto_global_gain={auto_global_gain_db:.2f} dB"
        )

    # Keep normalize as an optional final safety trim after auto level.
    if bool(getattr(cfg, "do_normalize", False)):
        peak_after_auto = float(current_peak_gain + auto_global_gain_db)
        if peak_after_auto > -0.1:
            auto_headroom_db = -peak_after_auto - 0.1
            logger.info(
                f"Clip Prevention (Normalize ON): post-auto peak={peak_after_auto:.2f} dB "
                f"-> extra headroom={auto_headroom_db:.2f} dB"
            )
        else:
            logger.info(
                f"Clip Prevention (Normalize ON): no extra headroom needed "
                f"(post-auto peak={peak_after_auto:.2f} dB)"
            )

    final_gain_total = gain_db + auto_global_gain_db + auto_headroom_db
    total_mag = 10**(final_gain_total / 20.0)
    # Keep minimum-phase UNCLAMPED.
    # Safety clamp is applied only to excess-phase correction component below.
    min_p = calculate_minimum_phase(total_mag, max_phase_deg=None)

    # --- 9C. PHASE LOGIC (single entry point) ---
    # IMPORTANT: do NOT recompute theo_xo here without HPF/crossover context.
    # `theo_xo` above is the single source of truth (and already includes HPF if enabled).

    is_mixed = ('Mixed' in cfg.filter_type_str)
    low_phase = None
    mixed_split_hz = float(np.clip(
        float(getattr(cfg, "mixed_split_freq", 300.0) or 300.0),
        20.0,
        float(cfg.fs) * 0.49,
    ))
    mixed_transition_hz = float(getattr(cfg, "trans_width", mixed_split_hz) or mixed_split_hz)
    if not np.isfinite(mixed_transition_hz) or mixed_transition_hz < 0.0:
        mixed_transition_hz = mixed_split_hz

    if bool(getattr(cfg, "phase_safe_2058", False)):
        # === 2058-SAFE PHASE MODE ===
        # No room phase correction (no excess-phase, FDW, confidence)

        if 'Min' in cfg.filter_type_str:
            final_phase = min_p

        elif is_mixed:
            low_phase = -theo_xo
            final_phase = low_phase

        else:
            # Linear / Asym
            final_phase = -theo_xo

    else:
        # Includes excess-phase, confidence, phase_limit blend, etc.
        # IMPORTANT: compute low_phase HERE before using it.

        # 1) Build/smooth confidence safely (avoid sawtooth weighting and None edge cases).
        try:
            conf_arr = np.asarray(conf_mask, dtype=float) if conf_mask is not None else np.ones_like(freq_axis, dtype=float)
            conf_s = scipy.ndimage.gaussian_filter1d(conf_arr, sigma=2)
            conf_s = np.clip(conf_s, 0.0, 1.0)
        except Exception:
            conf_s = np.clip(conf_arr if 'conf_arr' in locals() else np.ones_like(freq_axis, dtype=float), 0.0, 1.0)

        phase_lim_hz = float(getattr(cfg, "phase_limit", 1000.0))
        phase_mask = (freq_axis > 0) & (freq_axis <= phase_lim_hz)

        # Soft fade-out at the upper edge of phase_mask to reduce FIR ringing
        # from hard truncation (DSP-only safety, no user control).
        try:
            fade_oct = 1.0 / 1.0  # ~1/1 octave taper near phase_lim_hz
            f1 = float(phase_lim_hz)
            f0_fade = f1 / (2.0 ** fade_oct)
            if f0_fade < (f1 - 1.0):
                x = (freq_axis - f0_fade) / (f1 - f0_fade + 1e-12)
                x = np.clip(x, 0.0, 1.0)
                w_hi = 0.5 * (1.0 + np.cos(np.pi * x))  # 1 -> 0
                w_hi = np.where(freq_axis <= f0_fade, 1.0, w_hi)
                w_hi = np.where(freq_axis >= f1, 0.0, w_hi)
            else:
                w_hi = np.ones_like(freq_axis, dtype=float)
        except Exception:
            w_hi = np.ones_like(freq_axis, dtype=float)
        # Excess phase = measured - theoretical
        excess_phase = (p_rad_interp - theo_xo)

        phase_weight = np.zeros_like(freq_axis, dtype=float)
        if is_mixed:
            full_hz = float(getattr(cfg, "low_freq_full_correction_hz", mixed_split_hz) or mixed_split_hz)
            none_hz = float(getattr(cfg, "high_freq_no_correction_hz", phase_lim_hz) or phase_lim_hz)
            if phase_lim_hz > 0.0:
                none_hz = min(none_hz, phase_lim_hz)
            if none_hz <= (full_hz + 1.0):
                none_hz = full_hz + 1.0

            phase_weight = _mixed_excess_weight(freq_axis, full_hz, none_hz)
            phase_weight *= phase_mask.astype(float)

            strength = float(getattr(cfg, "excess_phase_strength", 0.9) or 0.0)
            strength = float(np.clip(strength, 0.0, 1.0))
            phase_weight *= strength

            try:
                if isinstance(st, dict):
                    st["mixed_phase_strength"] = float(strength)
                    st["mixed_phase_full_correction_hz"] = float(full_hz)
                    st["mixed_phase_no_correction_hz"] = float(none_hz)
            except Exception:
                pass
        else:
            # Legacy weighting for non-mixed modes.
            bass_f2 = float(np.clip(phase_lim_hz, 20.0, 400.0))
            f0, w0 = 20.0, 0.30
            f2, w2 = bass_f2, 0.00
            f1 = float(np.clip(0.5 * f2, 80.0, 140.0))
            w1 = float(np.clip(0.20 - 0.04 * ((f1 - 100.0) / 40.0), 0.14, 0.20))
            if f2 <= (f1 + 1.0):
                f2 = f1 + 1.0

            bass_band = phase_mask & (freq_axis >= f0) & (freq_axis <= f2)
            f = freq_axis[bass_band]
            w = np.empty_like(f, dtype=float)
            seg1 = f <= f1
            x1 = (f[seg1] - f0) / (f1 - f0)
            s1 = _smoothstep01(x1)
            w[seg1] = w0 + (w1 - w0) * s1
            seg2 = ~seg1
            x2 = (f[seg2] - f1) / (f2 - f1)
            s2 = _smoothstep01(x2)
            w[seg2] = w1 + (w2 - w1) * s2
            phase_weight[bass_band] = np.maximum(phase_weight[bass_band], w)

        # Apply the upper-edge taper so correction fades out smoothly near phase_limit
        # and does not create hard-edge ringing in time domain.
        try:
            phase_weight *= w_hi
        except Exception:
            pass

        # Confidence-aware phase weighting:
        # - low confidence reduces correction strength clearly
        # - high confidence allows near full correction strength
        try:
            conf_floor = 0.10
            conf_power = 1.25
            conf_gain = np.clip(conf_s, 0.0, 1.0) ** conf_power
            conf_gain = conf_floor + (1.0 - conf_floor) * conf_gain
            phase_weight *= conf_gain
        except Exception:
            pass

        extra_phase = -excess_phase * phase_weight

        # Clamp ONLY the excess-phase correction component.
        # Adaptive clamp is stricter:
        # - in low-confidence bins
        # - near the upper edge of correction band
        # This preserves useful LF correction while reducing risk of GD spikes/ringing.
        try:
            extra_phase_before = np.asarray(extra_phase, dtype=float).copy()

            if is_mixed:
                clamp_max_deg = float(getattr(cfg, "mixed_phase_budget_lf_deg", 45.0) or 45.0)
                clamp_min_deg = float(getattr(cfg, "mixed_phase_budget_hf_deg", 22.5) or 22.5)
            else:
                clamp_max_deg = 45.0
                clamp_min_deg = 15.0
            if clamp_max_deg < clamp_min_deg:
                clamp_max_deg, clamp_min_deg = clamp_min_deg, clamp_max_deg

            # Confidence contribution: low confidence -> closer to min clamp.
            conf_part = np.clip(conf_s, 0.0, 1.0) ** 0.85

            # Frequency contribution: lower frequencies can tolerate larger correction.
            if phase_lim_hz > 0.0:
                freq_rel = np.clip((phase_lim_hz - freq_axis) / max(phase_lim_hz, 1e-9), 0.0, 1.0)
            else:
                freq_rel = np.ones_like(freq_axis, dtype=float)
            freq_part = np.sqrt(freq_rel)

            # Blend confidence and frequency evidence into per-bin clamp limits.
            blend = 0.70 * conf_part + 0.30 * freq_part
            limit_deg_arr = clamp_min_deg + (clamp_max_deg - clamp_min_deg) * blend
            limit_deg_arr = np.clip(limit_deg_arr, clamp_min_deg, clamp_max_deg)
            limit_rad_arr = np.deg2rad(limit_deg_arr)

            _before_rad = float(np.max(np.abs(extra_phase)))
            extra_phase = np.clip(extra_phase, -limit_rad_arr, limit_rad_arr)
            _after_rad = float(np.max(np.abs(extra_phase)))

            # Always report in logs + stats.
            _before_deg = float(np.rad2deg(_before_rad))
            _after_deg = float(np.rad2deg(_after_rad))
            _clipped = bool(np.any(np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)))

            try:
                _clipped_bins = int(np.sum((np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)) & phase_mask))
            except Exception:
                _clipped_bins = int(_clipped)
            if _clipped:
                msg = (
                    "Phase Correction Clamp (adaptive): "
                    f"max={_before_deg:.1f} deg -> {_after_deg:.1f} deg "
                    f"(limit {clamp_min_deg:.1f}..{clamp_max_deg:.1f} deg, clipped_bins={_clipped_bins})"
                )
            else:
                msg = (
                    "Phase Correction Clamp (adaptive): "
                    f"max={_before_deg:.1f} deg (limit {clamp_min_deg:.1f}..{clamp_max_deg:.1f} deg)"
                )
            logger.info(msg)

            try:
                if isinstance(st, dict):
                    # Backward-compatible scalar + new adaptive diagnostics.
                    st["phase_corr_clamp_deg"] = float(clamp_max_deg)
                    st["phase_corr_clamp_min_deg"] = float(clamp_min_deg)
                    st["phase_corr_clamp_max_deg"] = float(clamp_max_deg)
                    st["phase_corr_clamp_mean_deg"] = float(np.mean(limit_deg_arr[phase_mask])) if np.any(phase_mask) else float(np.mean(limit_deg_arr))
                    st["phase_corr_max_before_deg"] = float(_before_deg)
                    st["phase_corr_max_after_deg"] = float(_after_deg)
                    st["phase_corr_clipped"] = bool(_clipped)
                    st["phase_corr_clipped_bins"] = int(_clipped_bins)
                    st["phase_corr_clamp_msg"] = str(msg)
            except Exception:
                pass
        except Exception:
            pass

        # --- GD-gradient limiter (DSP-only safety) ---
        # Limits abrupt changes of group delay induced by the correction phase.
        # Apply only inside the same correction band (phase_mask).
        try:
           # Conditional enable:
           # - With Bass-first + A-FDW, phase is already stabilized; the limiter can reduce "liveliness".
           # - Keep it only for risky setups (ultra-short REW Asym) or legacy mode (no BF, no A-FDW).
            _use_bassfirst = bool(locals().get("use_bassfirst", False))
            _afdw_on = bool(locals().get("afdw_on", False))

            # REW Asym tight-left detection (phase can get fragile)
            try:
                _win_mode = str(getattr(cfg, "ir_export_window_mode", "auto") or "auto").strip().lower()
            except Exception:
                _win_mode = "auto"
            try:
                _left_ms = float(getattr(cfg, "ir_window_left", getattr(cfg, "ir_window_ms_left", 0.0)) or 0.0)
            except Exception:
                _left_ms = 0.0
            if not np.isfinite(_left_ms):
                _left_ms = 0.0

            _rew_asym_tight = (_win_mode == "rew_asym" and _left_ms > 0.0 and _left_ms < 15.0)
            _legacy_mode = (not _use_bassfirst) and (not _afdw_on)

            _gd_grad_enable = bool(_rew_asym_tight or _legacy_mode)

            # Choose limiter strength (ms/oct)
            # - tight REW Asym: strict
            # - legacy: moderate (still safer than "always on 8")
            if _gd_grad_enable:
                _gd_grad_lim = 8.0 if _rew_asym_tight else 20.0

                extra_phase = _limit_gd_gradient_ms_per_oct(
                    freq_axis,
                    extra_phase,
                    mask=phase_mask,
                    max_grad_ms_per_oct=_gd_grad_lim,
                )

            try:
                if isinstance(st, dict):
                    st["gd_grad_limiter_enabled"] = bool(_gd_grad_enable)
                    st["gd_grad_limit_ms_per_oct"] = float(_gd_grad_lim) if _gd_grad_enable else None
                    st["gd_grad_limiter_reason"] = (
                        "rew_asym_tight_left" if _rew_asym_tight else
                        ("legacy_no_bassfirst_no_afdw" if _legacy_mode else
                         "skipped_bassfirst_or_afdw")
                    )
            except Exception:
                pass
        except Exception:
            pass

        # --- Mixed-only: absolute excess-delay guard ---
        if is_mixed:
            try:
                max_excess_delay_ms = float(getattr(cfg, "max_excess_delay_ms", 2.5) or 0.0)
            except Exception:
                max_excess_delay_ms = 0.0
            if np.isfinite(max_excess_delay_ms) and max_excess_delay_ms > 0.0:
                try:
                    max_gd_ms = _max_abs_group_delay_ms(freq_axis, extra_phase, phase_mask)
                    if np.isfinite(max_gd_ms) and max_gd_ms > max_excess_delay_ms:
                        gd_scale = float(np.clip(max_excess_delay_ms / max(max_gd_ms, 1e-9), 0.05, 1.0))
                        extra_phase *= gd_scale
                        logger.info(
                            "Mixed phase excess-delay guard: "
                            f"max|GD|={max_gd_ms:.2f} ms -> target<={max_excess_delay_ms:.2f} ms "
                            f"(scale={gd_scale:.3f})"
                        )
                        try:
                            if isinstance(st, dict):
                                st["mixed_max_excess_delay_ms"] = float(max_excess_delay_ms)
                                st["mixed_excess_delay_before_ms"] = float(max_gd_ms)
                                st["mixed_excess_delay_scale"] = float(gd_scale)
                        except Exception:
                            pass
                except Exception:
                    pass

        low_phase = (-theo_xo) + extra_phase

        # --- Mixed-only: pre-ringing guard (iteratively reduce excess correction) ---
        if is_mixed:
            try:
                max_pre_db = float(getattr(cfg, "max_pre_ringing_db", -35.0) or -35.0)
            except Exception:
                max_pre_db = -35.0
            if np.isfinite(max_pre_db):
                max_pre_db = float(min(max_pre_db, 0.0))
                extra_guard = np.asarray(extra_phase, dtype=float).copy()
                pre_before_db = None
                pre_after_db = None
                guard_scale_total = 1.0
                h_min = total_mag * np.exp(1j * min_p)
                ir_min = scipy.fft.irfft(h_min, n=n_fft)

                for i in range(3):
                    h_lin_guard = total_mag * np.exp(1j * ((-theo_xo) + extra_guard))
                    ir_lin_guard = scipy.fft.irfft(h_lin_guard, n=n_fft)
                    ir_mixed_guard = combine_mixed_phase(
                        ir_lin_guard,
                        ir_min,
                        fs=float(cfg.fs),
                        split_freq=mixed_split_hz,
                        transition_hz=mixed_transition_hz,
                    )
                    pre_now_db = _pre_ringing_db(ir_mixed_guard)
                    if i == 0:
                        pre_before_db = float(pre_now_db)
                    pre_after_db = float(pre_now_db)
                    if (not np.isfinite(pre_now_db)) or (pre_now_db <= max_pre_db):
                        break

                    # Energy-ratio based scale keeps iterations stable.
                    ratio_now = 10.0 ** (pre_now_db / 10.0)
                    ratio_target = 10.0 ** (max_pre_db / 10.0)
                    step_scale = float(np.clip(np.sqrt(ratio_target / max(ratio_now, 1e-30)), 0.20, 0.95))
                    extra_guard *= step_scale
                    guard_scale_total *= step_scale

                if guard_scale_total < 0.999:
                    extra_phase = extra_guard
                    low_phase = (-theo_xo) + extra_phase
                    logger.info(
                        "Mixed phase pre-ringing guard: "
                        f"{pre_before_db:.1f} dB -> {pre_after_db:.1f} dB "
                        f"(limit={max_pre_db:.1f} dB, scale={guard_scale_total:.3f})"
                    )
                try:
                    if isinstance(st, dict):
                        st["mixed_max_pre_ringing_db"] = float(max_pre_db)
                        st["mixed_pre_ringing_before_db"] = (
                            None if pre_before_db is None else float(pre_before_db)
                        )
                        st["mixed_pre_ringing_after_db"] = (
                            None if pre_after_db is None else float(pre_after_db)
                        )
                        st["mixed_pre_ringing_scale"] = float(guard_scale_total)
                except Exception:
                    pass

            # Recompute correction weight metric for stats if guards changed extra phase.
            try:
                corr_band = phase_mask & (np.abs(excess_phase) > 1e-12)
                if np.any(corr_band):
                    eff = np.abs(extra_phase[corr_band]) / np.maximum(np.abs(excess_phase[corr_band]), 1e-12)
                    if isinstance(st, dict):
                        st["mixed_phase_eff_strength_mean"] = float(np.mean(eff))
                        st["mixed_phase_eff_strength_max"] = float(np.max(eff))
            except Exception:
                pass

        if is_mixed:
            f_center = float(mixed_split_hz)
        else:
            f_center = float(getattr(cfg, "phase_limit", 1000.0) or 1000.0)

        f_center = float(np.clip(
            f_center, 20.0,
            float(freq_axis[-1] if freq_axis.size else 20000.0)
        ))

        safe_freqs = np.maximum(freq_axis, 1.0)
        octave_dist = np.log2(safe_freqs / f_center)
        mask = np.clip((octave_dist + 0.5), 0.0, 1.0)
        sm_mask = 3.0 * mask**2 - 2.0 * mask**3

        if 'Min' in cfg.filter_type_str:
            final_phase = min_p
        elif is_mixed:
            final_phase = low_phase
        else:
            final_phase = (1.0 - sm_mask) * low_phase + sm_mask * min_p

    # --- 10. IMPULSE GENERATION (common path) ---
    if is_mixed and low_phase is not None:
        try:
            if isinstance(st, dict):
                st["mixed_blend_split_hz"] = float(mixed_split_hz)
                st["mixed_blend_transition_hz"] = float(mixed_transition_hz)
        except Exception:
            pass

        h_lin = total_mag * np.exp(1j * low_phase)
        h_min = total_mag * np.exp(1j * min_p)
        ir_lin = scipy.fft.irfft(h_lin, n=n_fft)
        ir_min = scipy.fft.irfft(h_min, n=n_fft)
        raw_imp = combine_mixed_phase(
            ir_lin,
            ir_min,
            fs=float(cfg.fs),
            split_freq=mixed_split_hz,
            transition_hz=mixed_transition_hz,
        )
        final_phase = np.angle(scipy.fft.rfft(raw_imp))
    else:
        h_complex = total_mag * np.exp(1j * final_phase)
        raw_imp = scipy.fft.irfft(h_complex, n=n_fft)
    requested_win_mode = str(getattr(cfg, 'ir_export_window_mode', 'auto') or 'auto').strip().lower()
    is_min_filter = ('Min' in cfg.filter_type_str)
    # NOTE:
    # Strict "off" for minimum-phase can produce visible GD ripple in external tools
    # (FFT boundary/truncation sensitivity). Keep UI-selected mode by default.
    # Power users can enable strict off via cfg.min_strict_off = True.
    min_strict_off = bool(getattr(cfg, "min_strict_off", False))
    win_mode = 'off' if (is_min_filter and min_strict_off) else requested_win_mode
    if is_min_filter and min_strict_off and requested_win_mode != 'off':
        logger.info(
            f"IR export: forcing mode=off for Minimum filter (requested={requested_win_mode}, strict)"
        )

    def _ms(name_ms: str, name_alias: str, default: float = 0.0) -> float:
        """
        Read milliseconds value from config, preferring canonical *_ms fields,
        with fallback to legacy aliases for backwards compatibility.
        """
        v = getattr(cfg, name_ms, None)
        if v is None:
            v = getattr(cfg, name_alias, default)
        try:
            return float(v or 0.0)
        except Exception:
            return float(default)

    left_ms  = _ms('ir_window_ms_left',  'ir_window_left',  0.0)
    # Prefer 'ir_window_right' / 'ir_window_ms_right'; fall back to legacy 'ir_window' / 'ir_window_ms'
    right_ms = _ms('ir_window_ms_right', 'ir_window_right',
                   _ms('ir_window_ms', 'ir_window', 0.0))


    # --- 10a. INITIAL PLACEMENT (single-pass) ---
    if ('Asym' in cfg.filter_type_str):
        shift = min(
            int(left_ms * cfg.fs / 1000.0),
            int(n_fft * 0.4)
        )
        impulse = np.roll(raw_imp, shift)
    elif 'Min' in cfg.filter_type_str:
        impulse = raw_imp
    else:
        impulse = np.roll(raw_imp, n_fft // 2)

    def _shift_zeropad_local(x: np.ndarray, shift_samp: int) -> np.ndarray:
        """Integer shift with zero padding (no circular wrap)."""
        n_loc = int(len(x))
        if n_loc == 0 or int(shift_samp) == 0:
            return x
        if shift_samp > 0:
            if shift_samp >= n_loc:
                return np.zeros_like(x)
            return np.concatenate((np.zeros(shift_samp, dtype=x.dtype), x[:-shift_samp]))
        s = -int(shift_samp)
        if s >= n_loc:
            return np.zeros_like(x)
        return np.concatenate((x[s:], np.zeros(s, dtype=x.dtype)))

    # Minimum + OFF window mode:
    # enforce causal placement without wrap-around (peak at sample 0)
    # to avoid FFT boundary discontinuity that appears as GD sawtooth.
    if is_min_filter and win_mode == "off":
        try:
            peak_now = int(np.argmax(np.abs(impulse))) if impulse.size else 0
        except Exception:
            peak_now = 0
        shift_samp = -int(peak_now)
        if shift_samp != 0:
            impulse = _shift_zeropad_local(np.asarray(impulse), shift_samp)
            try:
                if isinstance(st, dict):
                    st["min_off_peak_shift_samples"] = int(shift_samp)
                    st["min_off_peak_before_samples"] = int(peak_now)
            except Exception:
                pass
            logger.info(
                f"Minimum OFF causal shift applied: peak {peak_now} -> 0 (shift={shift_samp})"
            )
    # --- 10b. IR WINDOWING (REW-style export; separate from filter type) ---
    win_shape = str(getattr(cfg, 'ir_export_window_shape', 'hann') or 'hann').strip().lower()
    try:
        tukey_alpha = float(getattr(cfg, 'ir_export_tukey_alpha', 0.25))
    except Exception:
        tukey_alpha = 0.25
    if not np.isfinite(tukey_alpha):
        tukey_alpha = 0.25
    tukey_alpha = float(np.clip(tukey_alpha, 0.0, 1.0))
    if win_shape not in ('hann', 'tukey'):
        win_shape = 'hann'

    n = len(impulse)
    peak_idx = int(np.argmax(np.abs(impulse)))

    s_left = int(left_ms * cfg.fs / 1000.0)
    s_right = int(right_ms * cfg.fs / 1000.0)

    logger.info(
        f"IR export: mode={win_mode}, peak={peak_idx}, user_s_left={s_left}, user_s_right={s_right}"
    )
    logger.info(f"IR export: shape={win_shape}, tukey_alpha={tukey_alpha:.2f}")

    if win_mode == 'off':
        window = np.ones(n, dtype=float)
        s_left = 0
        s_right = 0
    else:
        window = np.zeros(n, dtype=float)

        if win_mode in ('rew_sym', 'rew_asym'):
            radius = max(s_left, s_right)
            s_left = radius
            s_right = radius
            logger.info(f"IR export: mode={win_mode}, eff_radius={radius}")
        elif win_mode == 'auto':
            if not ('Asym' in cfg.filter_type_str or 'Min' in cfg.filter_type_str):
                radius = s_right
                s_left = radius
                s_right = radius
        # win_mode == 'rew_asym' -> use symmetric window, do peak shift AFTER window

        # Build window around peak
        def _edge_taper(L: int, *, alpha: float, side: str) -> np.ndarray:
            """
            Edge taper for IR export window.
            - hann: legacy sin^2/cos^2 (smoothest)
            - tukey: rectangular-ish plateau near peak + raised-cosine taper to 0 at edge (alpha controls taper amount)
            side: 'left' (toward peak) or 'right' (away from peak)
            Returns length L weights from 0..1.
            """
            L = int(L)
            if L <= 0:
                return np.zeros(0, dtype=float)
            if win_shape == 'hann' or alpha >= 0.999:
                # legacy behavior (keep sound identical by default)
                if side == 'left':
                    return (np.sin(np.linspace(0, np.pi / 2, L + 1))[:-1] ** 2).astype(float)
                else:
                    return (np.cos(np.linspace(0, np.pi / 2, L + 1))[1:] ** 2).astype(float)

            # Tukey-like: keep 1.0 near peak (plateau), then cosine down to 0 at edge.
            a = float(np.clip(alpha, 0.0, 1.0))
            taper_len = int(max(1, round(a * L)))
            plateau_len = int(max(0, L - taper_len))

            # k=0 near peak, increasing toward edge
            k = (np.arange(L)[::-1] if side == 'left' else np.arange(L)).astype(float)
            w = np.ones(L, dtype=float)
            if plateau_len > 0:
                w[k >= plateau_len] = 0.0  # will be overwritten by taper below
                w[k < plateau_len] = 1.0
            else:
                w[:] = 0.0

            # Taper region: k in [plateau_len .. plateau_len+taper_len]
            kk = np.clip(k - plateau_len, 0.0, float(taper_len))
            x = kk / float(max(taper_len, 1))  # 0..1
            taper = 0.5 * (1.0 + np.cos(np.pi * x))  # 1..0

            mask = (k >= plateau_len)
            w[mask] = taper[mask]
            return np.clip(w, 0.0, 1.0)

        if s_left > 0:
            win_rise = _edge_taper(s_left, alpha=tukey_alpha, side='left')
            start_idx = peak_idx - s_left
            if start_idx >= 0:
                window[start_idx:peak_idx] = win_rise
            else:
                window[0:peak_idx] = win_rise[-start_idx:]

        if s_right > 0:
            win_fall = _edge_taper(s_right, alpha=tukey_alpha, side='right')
            end_idx = peak_idx + 1 + s_right
            if end_idx <= n:
                window[peak_idx + 1:end_idx] = win_fall
            else:
                avail = n - (peak_idx + 1)
                if avail > 0:
                    window[peak_idx + 1:n] = win_fall[:avail]

        window[peak_idx] = 1.0

    # --- APPLY EXPORT WINDOW TO *THE* IMPULSE ---
    impulse_before = impulse.copy()
    impulse *= window

    # For REW asymmetric export:
    # - keep SAME windowing as rew_sym (magnitude identical)
    # - then shift peak earlier to reduce startup latency (phase/delay only)
    if win_mode == 'rew_asym' and n > 0:
        desired_peak = int(np.clip(int(left_ms * cfg.fs / 1000.0), 0, n - 1))
        peak_now = int(np.argmax(np.abs(impulse)))
        shift_samp = desired_peak - peak_now
        if shift_samp != 0:
            # IMPORTANT: do NOT use np.roll here (circular wrap).
            # Circular wrap leaves non-zero tail at the end -> comb/sawtooth ripple.
            if shift_samp > 0:
                # shift right (later): pad zeros at start
                if shift_samp >= n:
                    impulse[:] = 0.0
                else:
                    impulse = np.concatenate((np.zeros(shift_samp, dtype=impulse.dtype),
                                              impulse[:-shift_samp]))
            else:
                # shift left (earlier): pad zeros at end
                s = -shift_samp
                if s >= n:
                    impulse[:] = 0.0
                else:
                    impulse = np.concatenate((impulse[s:],
                                              np.zeros(s, dtype=impulse.dtype)))

    # Minimum + OFF window mode:
    # apply a very short end taper to suppress circular-boundary discontinuity
    # (common source of comb/sawtooth ripple in phase/group-delay views).
    if is_min_filter and win_mode == "off" and n > 16:
        try:
            taper_ms = float(getattr(cfg, "min_off_tail_taper_ms", 2.0) or 2.0)
        except Exception:
            taper_ms = 2.0
        if not np.isfinite(taper_ms):
            taper_ms = 2.0
        taper_ms = float(np.clip(taper_ms, 0.0, 20.0))
        taper_n = int(round((taper_ms / 1000.0) * float(cfg.fs)))
        taper_n = int(np.clip(taper_n, 8, max(8, n // 8)))
        if taper_n < n:
            t = (np.cos(np.linspace(0.0, np.pi / 2.0, taper_n, endpoint=True)) ** 2).astype(float)
            impulse[-taper_n:] *= t
            try:
                if isinstance(st, dict):
                    st["min_off_tail_taper_ms"] = float(taper_ms)
                    st["min_off_tail_taper_samples"] = int(taper_n)
            except Exception:
                pass
            logger.info(
                f"Minimum OFF anti-wrap taper applied: {taper_ms:.2f} ms ({taper_n} samples)"
            )

    # --- GUARD (non-fatal): Tukey should usually change IR, but allow edge-cases ---
    # Some impulses have tails that are exactly/near-zero in float, so a Tukey taper can be
    # bit-identical in practice. This is not an error worth crashing the run.
    if win_shape == "tukey" and tukey_alpha > 0.0 and win_mode != "off":
        try:
            d = float(np.max(np.abs(impulse - impulse_before)))
        except Exception:
            d = 0.0
        if (not math.isfinite(d)) or (d <= 0.0):
            logger.warning(
                "IR export window: Tukey had no measurable effect on impulse "
                f"(mode={win_mode}, alpha={tukey_alpha:.3f}). Continuing."
            )
    # DC removal (preserves zero outside the window).
    # Skip when windowing is OFF to keep "pure" export unchanged.
    if win_mode != "off":
        try:
            w_sum = float(np.sum(window))
            if w_sum > 0.0:
                dc = float(np.sum(impulse) / w_sum)
                impulse = impulse - (dc * window)
        except Exception:
            pass

    # Mixed filters: force fixed startup delay regardless of UI/window settings.
    # Do this AFTER window/DC steps so those operations stay aligned to their window.
    if is_mixed and n > 0:
        mixed_forced_peak_ms = 90.0
        desired_peak = int(np.clip(int(round(mixed_forced_peak_ms * cfg.fs / 1000.0)), 0, n - 1))
        peak_now = int(np.argmax(np.abs(impulse)))
        shift_samp = desired_peak - peak_now
        if shift_samp != 0:
            impulse = _shift_zeropad_local(np.asarray(impulse), int(shift_samp))
        try:
            if isinstance(st, dict):
                st["mixed_forced_peak_ms"] = float(mixed_forced_peak_ms)
                st["mixed_forced_peak_samples"] = int(desired_peak)
                st["mixed_forced_shift_samples"] = int(shift_samp)
        except Exception:
            pass
        logger.info(
            f"Mixed forced peak shift applied: peak {peak_now} -> {desired_peak} "
            f"({mixed_forced_peak_ms:.1f} ms, shift={shift_samp})"
        )




    return {
        "impulse": impulse,
        "gain_db": gain_db,
        "auto_global_gain_db": float(locals().get("auto_global_gain_db", 0.0)),
        "gain_margin_db": float(locals().get("gain_margin_db", 0.0)),
        "auto_headroom_db": float(locals().get("auto_headroom_db", 0.0)),
        "current_peak_gain": float(locals().get("current_peak_gain", 0.0)),
        "final_gain_total": np.asarray(locals().get("final_gain_total", gain_db), dtype=float),
    }

