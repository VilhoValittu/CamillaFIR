import numpy as np
import scipy.signal
import scipy.fft
import math
import scipy.ndimage
import copy
import logging
import hashlib
logger = logging.getLogger("CamillaFIR.dsp")
from . import bassfirst as bf
from camillafir.config.models import FilterConfig
from .camillafir_leveling import compute_leveling
from .analysis import (
    _sigma_bins_from_hz,
    analyze_acoustic_confidence,
    calculate_rt60,
    calculate_rt60_bands,
    _third_oct_centers,
    calculate_group_delay,
)
from .smoothing import (
    psychoacoustic_smoothing,
    apply_fdw_smoothing,
    apply_adaptive_fdw,
    apply_smoothing_std,
)
from .limits import (
    soft_clip_boost,
    soft_clip_gain,
    limit_slope_per_octave,
    limit_slope_per_octave_asym,
    build_slope_limit_envelope,
)
from .tdc import apply_smart_tdc
from .phase import (
    calculate_minimum_phase,
    calculate_theoretical_phase,
    combine_mixed_phase,
    remove_time_of_flight,
    get_min_phase_impulse,
)

#CamillaFIR DSP Engine v1.1.1

#1.0.2 Fix comma mistake at HPF
#1.03 Fix at phase calculation that caused "spikes"
#1.04 All features works at different configurations
#1.05 Multiplier changes
#1.06 added no phase correction mode (2058-safe)
#1.07 TDC improvements and bugfixes
#1.08 Bugfixes for bass-first and TDC
#1.09 Improved TDC stability and reliability
#1.1.0 More presice leveling tilt for magnitude calculation
#1.1.1 Added tukey windowing option

def _stage_probe(stage_name, freq_axis, arr_db, mask_c, global_gain_db=0.0, auto_headroom_db=0.0, logger_obj=None):
    """
    Lightweight stage checkpoint for debugging gain evolution.
    Records boost/cut peaks and bin counts inside correction mask.
    """
    try:
        out = {
            "stage": str(stage_name),
            "boost_peak_db": 0.0,
            "cut_peak_db": 0.0,
            "boost_bins": 0,
            "cut_bins": 0,
            "net_boost_peak_db": 0.0
        }
        if arr_db is None or mask_c is None or not np.any(mask_c):
            return out
        v = np.asarray(arr_db, dtype=float)
        m = np.asarray(mask_c, dtype=bool)
        vv = v[m]
        out["boost_peak_db"] = float(np.max(vv)) if vv.size else 0.0
        out["cut_peak_db"] = float(np.min(vv)) if vv.size else 0.0
        out["boost_bins"] = int(np.sum(vv > 1e-6))
        out["cut_bins"] = int(np.sum(vv < -1e-6))
        out["net_boost_peak_db"] = float(out["boost_peak_db"] + float(global_gain_db) + float(auto_headroom_db))
        if logger_obj is not None:
            logger_obj.info(
                f"StageProbe[{out['stage']}]: "
                f"boost_peak={out['boost_peak_db']:.2f} dB, cut_peak={out['cut_peak_db']:.2f} dB, "
                f"boost_bins={out['boost_bins']}, cut_bins={out['cut_bins']}, "
                f"net_boost_peak={out['net_boost_peak_db']:.2f} dB"
            )
        return out
    except Exception:
        return {
            "stage": str(stage_name),
            "boost_peak_db": 0.0,
            "cut_peak_db": 0.0,
            "boost_bins": 0,
            "cut_bins": 0,
            "net_boost_peak_db": 0.0
        }

def apply_hpf_to_mags(freqs, mags, cutoff, order):
    """Applies Butterworth high-pass filter to magnitude response (dB)."""
    if cutoff <= 0 or order <= 0:
        return mags
    f = np.asarray(freqs, dtype=float)
    # Avoid infinite attenuation at DC bin (0 Hz) in stats/plot
    if f.size > 1 and f[0] == 0.0:
        f = f.copy()
        f[0] = f[1] if f[1] > 0 else 1e-6
    # Butterworth vaste: 1 / sqrt(1 + (fc/f)^(2*order))
    # Muutetaan desibeleiksi: -10 * log10(1 + (fc/f)^(2*order))
    with np.errstate(divide='ignore'):
        attenuation = -10 * np.log10(1 + (cutoff / (f + 1e-12))**(2 * order))
    return mags + attenuation

def interpolate_response(input_freqs, input_values, target_freqs):
    """Interpolate response linearly to target frequencies."""
    return np.interp(target_freqs, input_freqs, input_values)


#----------- Filtteri ---------------------


import numpy as np
import scipy.signal
import scipy.fft
import scipy.ndimage
import logging
from camillafir.config.models import FilterConfig

logger = logging.getLogger("CamillaFIR.dsp")

def generate_filter(freqs, meas_mags, raw_phases, cfg: FilterConfig):
    # --- 1. DATA ALIGNMENT ---
    min_len = min(len(freqs), len(meas_mags), len(raw_phases))
    f_in, m_in, p_in = freqs[:min_len], meas_mags[:min_len], raw_phases[:min_len]

    # --- 2. AXES ---
    n_fft = cfg.num_taps if cfg.num_taps % 2 != 0 else cfg.num_taps + 1
    freq_axis = np.linspace(0, cfg.fs/2.0, n_fft // 2 + 1)

    # Must exist early:
    # - comparison-mode block interpolates gain_db before correction stage
    # - later blocks may reference gain_db even if correction is disabled
    gain_db = np.zeros_like(freq_axis, dtype=float)
    
    # --- 3. SMOOTHING (Scalable resolution) ---
    oct_frac = 1.0 / float(cfg.smoothing_level) if cfg.smoothing_level > 0 else 1/24.0
    is_psy = 'psy' in str(cfg.smoothing_type).lower()
    

    # --- IMPORTANT ---
    # Keep DSP math stable: analysis/correction uses STANDARD smoothing only.
    # Psychoacoustic mode affects only what we *show* in UI (plots/score),
    # not confidence/leveling/correction curve generation.
    m_smooth_std, _ = apply_smoothing_std(
        f_in, m_in, np.zeros_like(m_in), float(oct_frac)
    )
    
    # FIX: Dynamic phase smoothing based on sample rate
    # At high sample rates (>96kHz) interpolation creates "corners" in phase,
    # which are cleaned with heavier (1/12 oct) smoothing.
    p_smooth_oct = 1/12.0 if cfg.fs > 96000 else 1/24.0
    p_smooth, _ = apply_smoothing_std(f_in, p_in, np.zeros_like(p_in), p_smooth_oct)

    # --- 4. TOF & INTERPOLOINTI ---
    m_interp = np.interp(freq_axis, f_in, m_in)
    p_rad_raw = np.deg2rad(np.interp(freq_axis, f_in, p_in))
    p_rad_interp, delay_slope = remove_time_of_flight(freq_axis, p_rad_raw)

    # Psychoacoustic plot magnitude on the *analysis axis* (UI only)
    m_plot_db = None
    if is_psy:
        try:
            m_plot_db = psychoacoustic_smoothing(
                freq_axis,
                m_interp,
                heavy_bw=1/3.0,
                light_bw=1/48.0,
                f_lo=200.0,
                f_hi=2000.0,
            )
        except Exception:
            m_plot_db = None

    # Raw measurement as complex FR (TOF removed) for features that need phase/GD.
    # NOTE: This is intentionally "raw" (not smoothing-analysis), to match Bass-first intent.
    # (Magnitude is dB -> linear amplitude.)
    complex_meas = 10**(m_interp/20.0) * np.exp(1j * p_rad_interp)

    # --- 5. ANALYYSI (Skaalattu luottamusmaski) ---
    m_anal = np.interp(freq_axis, f_in, m_smooth_std)
    p_anal_rad = np.deg2rad(np.interp(freq_axis, f_in, p_smooth))
    p_anal_rad, _ = remove_time_of_flight(freq_axis, p_anal_rad)
    complex_anal = 10**(m_anal/20.0) * np.exp(1j * p_anal_rad)
    
    # Confidence mask ja heijastusanalyysi skaalautuvalla sigmalla
    conf_mask, reflections, _ = analyze_acoustic_confidence(freq_axis, complex_anal, cfg.fs)

    # --- 5C. COMPARISON MODE (locked 44.1k analysis grid for score/match/report) ---
    cmp = None
    analysis_mode = "native"
    try:
        if bool(getattr(cfg, "comparison_mode", False)):
            ref_fs = int(getattr(cfg, "comparison_ref_fs", 44100) or 44100)
            ref_taps = int(getattr(cfg, "comparison_ref_taps", 65536) or 65536)
            ref_nfft = ref_taps if (ref_taps % 2 != 0) else (ref_taps + 1)
            freq_cmp_full = np.linspace(0, ref_fs / 2.0, ref_nfft // 2 + 1)

            # clamp comparison grid to what we can represent with current freq_axis
            fmax = float(freq_axis[-1]) if freq_axis.size else 0.0
            if fmax > 0:
                freq_cmp = freq_cmp_full[freq_cmp_full <= fmax]
            else:
                freq_cmp = freq_cmp_full

            # resample analysis magnitude/phase to comparison grid
            m_cmp_raw = np.interp(freq_cmp, freq_axis, m_anal)
            p_cmp_rad = np.interp(freq_cmp, freq_axis, p_anal_rad)
            complex_cmp = 10 ** (m_cmp_raw / 20.0) * np.exp(1j * p_cmp_rad)

            # recompute confidence on reference fs (makes GD-based confidence stable across cfg.fs/taps)
            conf_cmp, refl_cmp, _ = analyze_acoustic_confidence(freq_cmp, complex_cmp, ref_fs)

            # resample target and compute leveling on comparison grid (for stable offset + match window)
            target_cmp = np.interp(freq_cmp, freq_axis, target_mags)
            (
                target_level_db_cmp,
                calc_offset_db_cmp,
                meas_level_db_window_cmp,
                target_level_db_window_cmp,
                offset_method_cmp,
                s_min_cmp,
                s_max_cmp,
            ) = compute_leveling(cfg, freq_cmp, m_cmp_raw, target_cmp)

            # resample filter correction curve to comparison grid
            filt_cmp = np.interp(freq_cmp, freq_axis, gain_db)

            cmp = {
                "cmp_ref_fs": float(ref_fs),
                "cmp_ref_taps": float(ref_taps),
                "cmp_freq_axis": freq_cmp.tolist(),
                "cmp_target_mags": target_cmp.tolist(),
                # IMPORTANT: calc_offset_db = median(meas - target)
                # => aligned measured = meas - calc_offset_db
                "cmp_measured_mags": (m_cmp_raw - calc_offset_db_cmp).tolist(),
                "cmp_filter_mags": filt_cmp.tolist(),
                "cmp_confidence_mask": conf_cmp.tolist(),
                "cmp_reflections": refl_cmp,
                "cmp_smart_scan_range": [float(s_min_cmp), float(s_max_cmp)],
                "cmp_eff_target_db": float(target_level_db_cmp),
                "cmp_offset_db": float(calc_offset_db_cmp),
                "cmp_meas_level_db_window": float(meas_level_db_window_cmp),
                "cmp_target_level_db_window": float(target_level_db_window_cmp),
                "cmp_offset_method": str(offset_method_cmp),
                "cmp_avg_confidence": float(np.mean(conf_cmp) * 100.0),
            }
            if (
                isinstance(cmp.get("cmp_freq_axis", None), list)
                and isinstance(cmp.get("cmp_measured_mags", None), list)
                and isinstance(cmp.get("cmp_target_mags", None), list)
                and isinstance(cmp.get("cmp_filter_mags", None), list)
                and isinstance(cmp.get("cmp_confidence_mask", None), list)
                and len(cmp["cmp_freq_axis"]) > 16
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_measured_mags"])
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_target_mags"])
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_filter_mags"])
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_confidence_mask"])
            ):
                analysis_mode = "comparison"
    except Exception:
        cmp = None
        analysis_mode = "native"

    # --- 5B. A-FDW: use confidence mask for magnitude "analysis version" smoothing
    # 
    # - conf_mask ~ 1.0 => sharper (more cycles) => get more "true" correction
    # - conf_mask ~ 0.0 => heavier smoothing => avoid aggressive correction in unreliable data
    #
    # This affects directly:
    # - leveling (m_anal)
    # - mag correction (raw_g)
    # but does NOT change reflection/gd detection (based mainly on phase).
    if getattr(cfg, "enable_afdw", False):
        # min_cycles kept reasonable (avoid "too sharp" even in poor confidence areas)
            base = float(getattr(cfg, "fdw_cycles", 15.0))
            min_c = max(3.0, base / 3.0)
            m_anal = apply_adaptive_fdw(
            freq_axis,
            m_anal,
            conf_mask,
            base_cycles=base,
            min_cycles=min_c
        )


    # --- 6. RT60 & TARGET ---
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
    if cfg.hpf_settings and cfg.hpf_settings.get('enabled'):
        target_mags = apply_hpf_to_mags(freq_axis, target_mags, cfg.hpf_settings['freq'], cfg.hpf_settings['order'])
    
    if cfg.enable_tdc:
        # NEW: TDC receives frequency-dependent RT60 (dict), auto-fallback if empty
        rt60_for_tdc = rt60_bands if rt60_bands else current_rt60
        # Safety brakes: cap total TDC reduction and keep it smooth (avoid deep, stacked notches)
        # Configurable TDC safety brakes for easy A/B testing
        tdc_max_red = float(getattr(cfg, "tdc_max_reduction_db", 9.0) or 9.0)
        tdc_slope = float(getattr(cfg, "tdc_slope_db_per_oct", 0.0) or 0.0)

        # Clamp to sane values (never explode)
        if tdc_max_red < 0: tdc_max_red = 0.0
        if tdc_max_red > 24: tdc_max_red = 24.0
        if tdc_slope < 0: tdc_slope = 0.0
        if tdc_slope > 24: tdc_slope = 24.0

        target_mags = apply_smart_tdc(
            freq_axis,
            target_mags,
            reflections,
            rt60_for_tdc,
            cfg.tdc_strength / 100.0,
            max_total_reduction_db=tdc_max_red,
            max_slope_db_per_oct=tdc_slope
        )

    # --- HPF params (always defined) ---
    hpf_f = 0.0
    hpf_order = 0
    if cfg.hpf_settings and cfg.hpf_settings.get('enabled'):
        hpf_f = float(cfg.hpf_settings.get('freq', 0.0) or 0.0)
        hpf_order = int(cfg.hpf_settings.get('order', 0) or 0)

    # --- 7. TASONSOVITUS ---
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

    # --- 7A. Align target level to the SAME leveling window (Smart Scan / Manual) ---
    # Goal: target curve must "follow" the chosen level window, not float at an arbitrary absolute level.
    # This makes leveling deterministic and consistent with REW/OCA-style workflows.
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
    # --- 7A2. Expose EFFECTIVE target & axis to native UI/plots ---
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


    # --- 7B. Keep comparison-mode target/leveling consistent after target shaping (HPF/TDC/etc.) ---
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

    # --- 8. KORJAUS ---
    if cfg.enable_mag_correction:
        afdw_on = bool(getattr(cfg, "enable_afdw", False))
        afdw_base = float(getattr(cfg, "fdw_cycles", 15.0))
        afdw_min = max(3.0, afdw_base / 3.0)
        
        raw_g = target_mags - (m_anal - calc_offset_db)

        base_sigma = 60 // (cfg.smoothing_level / 12 if cfg.smoothing_level > 0 else 1)

        # Raw_g smoothing:
        # - legacy: sigma in bins scales directly with fs (can over-smooth at high fs)
        # - df_smoothing: keep smoothing width constant in Hz (reference: 44.1k/65536 behavior)
        df_mode = bool(getattr(cfg, "df_smoothing", False))
        if df_mode:
            # Reference bin width ~ 44100/65536 Hz; match the old "base_sigma bins" at ref
            df_ref = 44100.0 / 65536.0
            sigma_hz = float(base_sigma) * df_ref
            # Convert Hz -> bins for current axis
            sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=sigma_hz, fallback_bins=max(2.0, float(base_sigma)))
            sm_g = scipy.ndimage.gaussian_filter1d(raw_g, sigma=float(sigma_bins))
        else:
            sigma_scaling = cfg.fs / 44100.0
            sigma = max(2, int(base_sigma * sigma_scaling))
            sm_g = scipy.ndimage.gaussian_filter1d(raw_g, sigma=sigma)

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

                # Same gd_diff idea as analyze_acoustic_confidence (sigma_bins/Hz not critical here)
                gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms_local, sigma=20)
                gd_diff_local = np.abs(gd_ms_local - gd_smooth)

                # m_interp should be the "raw measured mags" interpolated to freq_axis
                # If at this point a different name is used, use it (usually m_interp / meas_interp etc.)
                bf_rel, bf_room_mode, _ = bf.build_bassfirst_masks(
                    freq_axis=freq_axis,
                    m_raw_db=m_interp,
                    phase_rad_unwrapped=ph_u,
                    gd_ms=gd_ms_local,
                    gd_diff=gd_diff_local,
                    is_wav_source=bool(getattr(cfg, "is_wav_source", False)),
                    mode_f2=float(getattr(cfg, "bass_first_mode_max_hz", 200.0) or 200.0)
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
                c = np.clip(conf_mask, 0.0, 1.0)
                adaptive_cycles = float(afdw_min) + (c * (float(afdw_base) - float(afdw_min)))
                bw = 2.0 / np.maximum(adaptive_cycles, 1.0)
                # clamp to same range used by the continuous blender
                bw = np.clip(bw, 1.0/96.0, 1.0/3.0)
                afdw_bw_oct = bw
                afdw_bw_min_oct = float(np.min(bw))
                afdw_bw_mean_oct = float(np.mean(bw))
                afdw_bw_max_oct = float(np.max(bw))
                bw_min_idx = np.where(bw == np.min(bw))[0]
                bw_max_idx = np.where(bw == np.max(bw))[0]
                afdw_bw_min_hz = float(freq_axis[int(bw_min_idx[len(bw_min_idx)//2])])
                afdw_bw_max_hz = float(freq_axis[int(bw_max_idx[len(bw_max_idx)//2])])
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

        # --- 8C. Low-bass CUT allowance (targeted fix for 32 Hz-like modes) ---
        # < low_hz: allow ONLY attenuation (no boost),
        # and use stronger cuts if needed (min(final_g, raw_g)),
        # so regularization / confidence doesn't zero out clear room mode peaks.
        low_hz = float(getattr(cfg, "low_bass_cut_hz", 40.0))
        low_mask = mask_c & (freq_axis > 0) & (freq_axis <= low_hz)
        if np.any(low_mask):
            low_cut = np.minimum(final_g[low_mask], raw_g[low_mask])  # valitse negatiivisempi (vahvempi cut)
            low_cut = np.minimum(low_cut, 0.0)                       # ei koskaan boostia
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

            try:
                logger.info(
                    "Slope limit: "
                    f"boost={float(max_slope_boost):.1f} dB/oct | "
                    f"cut={float(max_slope_cut):.1f} dB/oct "
                    f"(legacy max_slope_db_per_oct={float(max_slope):.1f})"
                )
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

        # --- HPF params (always defined) ---
        hpf_f = 0.0
        if cfg.hpf_settings and cfg.hpf_settings.get('enabled'):
            hpf_f = float(cfg.hpf_settings.get('freq', 0.0) or 0.0)

        # --- HPF policy: full stop + smooth fade (asym-safe) ---
        if hpf_f > 0:
                hpf_end = hpf_f * 1.41  # ~1/2 octave

                # 1) Full stop below HPF
                below = freq_axis < hpf_f
                gain_db[below] = 0.0

                # 2) Smooth fade HPF -> HPF*1.41 (0..1)
                trans = (freq_axis >= hpf_f) & (freq_axis <= hpf_end)
                if np.any(trans):
                    fade = (freq_axis[trans] - hpf_f) / (hpf_end - hpf_f)
                    gain_db[trans] *= fade

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
            else:
                hardclamp_boost_bins, hardclamp_cut_bins = 0, 0
                hard_over_boost, hard_over_cut = 0.0, 0.0
            logger.info(
                "Clamp: hard_clamp "
                f"(max_boost={_max_boost2:.2f} dB, max_cut={_max_cut2:.2f} dB) -> "
                f"boost_clipped_bins={hardclamp_boost_bins}, cut_clipped_bins={hardclamp_cut_bins}, "
                f"worst_over_boost={hard_over_boost:.2f} dB, worst_over_cut={hard_over_cut:.2f} dB"
            )
        except Exception:
            hardclamp_boost_bins, hardclamp_cut_bins = 0, 0
            hard_over_boost, hard_over_cut = 0.0, 0.0

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

        theo_xo = calculate_theoretical_phase(
            freq_axis,
            cfg.crossovers,
            hpf_freq=hpf_freq,
            hpf_slope=hpf_slope
        )


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
                        low_hz = float(getattr(cfg, "low_bass_cut_hz", 40.0) or 40.0)
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

                    # HPF policy fade (same as earlier HPF policy)
                    try:
                        hs2 = getattr(cfg, "hpf_settings", None)
                        if isinstance(hs2, dict) and hs2.get('enabled'):
                            _hpf_f = float(hs2.get('freq', 0.0) or 0.0)
                            if _hpf_f > 0:
                                hpf_end2 = _hpf_f * 1.41
                                below2 = freq_axis < _hpf_f
                                _g[below2] = 0.0
                                trans2 = (freq_axis >= _hpf_f) & (freq_axis <= hpf_end2)
                                if np.any(trans2):
                                    fade2 = (freq_axis[trans2] - _hpf_f) / (hpf_end2 - _hpf_f)
                                    _g[trans2] *= fade2
                    except Exception:
                        pass

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
                _base_sigma = locals().get('base_sigma', 60 // (cfg.smoothing_level / 12 if cfg.smoothing_level > 0 else 1))
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
                    sigma_scaling = cfg.fs / 44100.0
                    sigma = max(2, int(float(_base_sigma) * sigma_scaling * mult))
                    resid_sm = scipy.ndimage.gaussian_filter1d(resid, sigma=sigma)

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

    # --- 9B. CLIP PREVENTION & HEADROOM ---
    # NOTE:
    # - If normalization is OFF, we do NOT auto-lower the whole response.
    #   (Otherwise users will observe "no boosts" + global -1..-2 dB drop.)
    # - If normalization is ON, we apply a small headroom so the FIR does not clip.
    current_peak_gain = float(np.max(gain_db + cfg.global_gain_db))
    auto_headroom_db = 0.0

    if bool(getattr(cfg, "do_normalize", False)) and current_peak_gain > 0.0:
        auto_headroom_db = -current_peak_gain - 0.1
        logger.info(
            f"Clip Prevention (Normalize ON): peak={current_peak_gain:.2f} dB -> headroom={auto_headroom_db:.2f} dB"
        )
    else:
        # Keep behavior transparent in logs.
        if current_peak_gain > 0.0:
            logger.info(
                f"Clip Prevention: OFF (Normalize OFF). peak would be {current_peak_gain:.2f} dB"
            )
        else:
            logger.info("Clip Prevention: no positive peak gain detected")

    final_gain_total = gain_db + cfg.global_gain_db + auto_headroom_db
    total_mag = 10**(final_gain_total / 20.0)
    min_p = calculate_minimum_phase(total_mag)

    # --- 9C. PHASE LOGIC (single entry point) ---
    # IMPORTANT: do NOT recompute theo_xo here without HPF/crossover context.
    # `theo_xo` above is the single source of truth (and already includes HPF if enabled).

    if bool(getattr(cfg, "phase_safe_2058", False)):
        # === 2058-SAFE PHASE MODE ===
        # No room phase correction (no excess-phase, FDW, confidence)

        if 'Min' in cfg.filter_type_str:
            final_phase = min_p

        elif 'Mixed' in cfg.filter_type_str:
            f_center = float(getattr(cfg, "mixed_split_freq", 300.0) or 300.0)
            f_center = float(np.clip(
                f_center, 20.0,
                float(freq_axis[-1] if freq_axis.size else 20000.0)
            ))

            safe_freqs = np.maximum(freq_axis, 1.0)
            octave_dist = np.log2(safe_freqs / f_center)
            mask = np.clip((octave_dist + 0.5), 0.0, 1.0)
            sm_mask = 3.0 * mask**2 - 2.0 * mask**3  # smoothstep

            low_phase = -theo_xo
            final_phase = (1.0 - sm_mask) * low_phase + sm_mask * min_p

        else:
            # Linear / Asym
            final_phase = -theo_xo

    else:
        # Includes excess-phase, confidence, phase_limit blend, etc.
        # IMPORTANT: compute low_phase HERE before using it.

        # 1) Smooth confidence slightly (avoid sawtooth weighting)
        try:
            conf_s = scipy.ndimage.gaussian_filter1d(conf_mask.astype(float), sigma=2)
            conf_s = np.clip(conf_s, 0.0, 1.0)
        except Exception:
            conf_s = np.clip(conf_mask.astype(float), 0.0, 1.0)

        phase_lim_hz = float(getattr(cfg, "phase_limit", 1000.0))
        phase_mask = (freq_axis > 0) & (freq_axis <= phase_lim_hz)
        bass_f2 = float(np.clip(phase_lim_hz, 20.0, 400.0))

        # Excess phase = measured - theoretical
        excess_phase = (p_rad_interp - theo_xo)

        # Weighting (your smoothstep bass weighting)
        phase_weight = np.zeros_like(freq_axis, dtype=float)
        f0, w0 = 20.0, 0.30
        f2, w2 = bass_f2, 0.00
        f1 = float(np.clip(0.5 * f2, 80.0, 140.0))
        w1 = float(np.clip(0.20 - 0.04 * ((f1 - 100.0) / 40.0), 0.14, 0.20))
        if f2 <= (f1 + 1.0):
            f2 = f1 + 1.0

        def smoothstep01(x):
            x = np.clip(x, 0.0, 1.0)
            return x*x*(3.0 - 2.0*x)

        bass_band = phase_mask & (freq_axis >= f0) & (freq_axis <= f2)
        f = freq_axis[bass_band]
        w = np.empty_like(f, dtype=float)
        seg1 = f <= f1
        x1 = (f[seg1] - f0) / (f1 - f0)
        s1 = smoothstep01(x1)
        w[seg1] = w0 + (w1 - w0) * s1
        seg2 = ~seg1
        x2 = (f[seg2] - f1) / (f2 - f1)
        s2 = smoothstep01(x2)
        w[seg2] = w1 + (w2 - w1) * s2
        phase_weight[bass_band] = np.maximum(phase_weight[bass_band], w)

        extra_phase = -excess_phase * phase_weight
        low_phase = (-theo_xo) + extra_phase

        if 'Mixed' in cfg.filter_type_str:
            f_center = float(getattr(cfg, "mixed_split_freq", 300.0) or 300.0)
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
        else:
            final_phase = (1.0 - sm_mask) * low_phase + sm_mask * min_p

    # --- 10. IMPULSE GENERATION (common path) ---
    
    h_complex = total_mag * np.exp(1j * final_phase)
    raw_imp = scipy.fft.irfft(h_complex, n=n_fft)
    win_mode = str(getattr(cfg, 'ir_export_window_mode', 'auto') or 'auto').strip().lower()

    # --- 10a. INITIAL PLACEMENT (single-pass) ---
    if ('Asym' in cfg.filter_type_str) or (win_mode == 'rew_asym'):
        shift = min(
            int(float(getattr(cfg, 'ir_window_left', 0.0) or 0.0) * cfg.fs / 1000.0),
            int(n_fft * 0.4)
        )
        impulse = np.roll(raw_imp, shift)
    elif 'Min' in cfg.filter_type_str:
        impulse = raw_imp
    else:
        impulse = np.roll(raw_imp, n_fft // 2)
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

    s_left = int(float(getattr(cfg, 'ir_window_left', 0.0) or 0.0) * cfg.fs / 1000.0)
    s_right = int(float(getattr(cfg, 'ir_window', 0.0) or 0.0) * cfg.fs / 1000.0)

#    if win_mode == 'rew_asym' and n > 0:
#        desired_peak = int(np.clip(s_left, 0, n - 1))
#       shift_samp = desired_peak - peak_idx
#        if shift_samp != 0:
#            impulse = np.roll(impulse, shift_samp)
#            peak_idx = desired_peak

    logger.info(f"IR export: mode={win_mode}, peak={peak_idx}, s_left={s_left}, s_right={s_right}")
    logger.info(f"IR export: shape={win_shape}, tukey_alpha={tukey_alpha:.2f}")


    # For REW asymmetric export, shift the peak earlier (reduces CamillaDSP startup latency)
    if win_mode == 'rew_asym' and n > 0:
        desired_peak = int(np.clip(s_left, 0, n - 1))
        shift_samp = desired_peak - peak_idx
        if shift_samp != 0:
            impulse = np.roll(impulse, shift_samp)
            peak_idx = desired_peak

    if win_mode == 'off':
        window = np.ones(n, dtype=float)
        s_left = 0
        s_right = 0
    else:
        window = np.zeros(n, dtype=float)

        if win_mode == 'rew_sym':
            radius = max(s_left, s_right)
            s_left = radius
            s_right = radius
        elif win_mode == 'auto':
            if not ('Asym' in cfg.filter_type_str or 'Min' in cfg.filter_type_str):
                radius = s_right
                s_left = radius
                s_right = radius
        # win_mode == 'rew_asym' -> literal L/R (already shifted)

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
    # DC removal (preserves zero outside the window)
    # Subtract only inside the active window to avoid breaking the windowed zeros.
    try:
        w_sum = float(np.sum(window))
        if w_sum > 0.0:
            dc = float(np.sum(impulse) / w_sum)
            impulse = impulse - (dc * window)
    except Exception:
        pass
    

    
    # --- 11. STATS & RETURN ---
    max_peak = np.max(np.abs(impulse))
    if cfg.do_normalize and max_peak > 0: impulse *= (0.89 / max_peak)

    stats = {
        'analysis_mode': analysis_mode,
        'freq_axis': freq_axis.tolist(),
        # User-visible correction band (for plots / diagnostics)
        'mag_c_min': float(getattr(cfg, 'mag_c_min', 0.0) or 0.0),
        'mag_c_max': float(getattr(cfg, 'mag_c_max', 0.0) or 0.0),
        'target_mags': target_mags.tolist(),
        'measured_mags': (m_anal - calc_offset_db).tolist(),
        'filter_mags': gain_db.tolist(),
        'confidence_mask': conf_mask.tolist(),
        'afdw_active': bool(afdw_on),
        'reflections': reflections,
        'smart_scan_range': [float(s_min), float(s_max)],
        'eff_target_db': float(target_level_db),
        'offset_db': float(calc_offset_db),
        'meas_level_db_window': float(meas_level_db_window),
        'target_level_db_window': float(target_level_db_window),
        'offset_method': str(offset_method),
            # Leveling tilt (reporting only). None if tilt-comp was not used or not computed.
            'tilt_slope_db_per_oct': (
                float(getattr(cfg, "_lvl_tilt_slope_db_per_oct"))
                if getattr(cfg, "_lvl_tilt_slope_db_per_oct", None) is not None
                else None
            ),
        'rt60_val': float(current_rt60),
        'rt60_band_avg': float(band_avg),
        'rt60_bands': rt60_bands,        
        'avg_confidence': float(np.mean(conf_mask)*100),
        'delay_samples': float((delay_slope * cfg.fs) / (2 * np.pi)) if 'delay_slope' in locals() else 0.0,
        'peak_before_norm': float(20*np.log10(max_peak + 1e-12)),
        'do_normalize': bool(getattr(cfg, 'do_normalize', False)),
        'auto_headroom_db': float(auto_headroom_db),
        'peak_gain_db': float(current_peak_gain),
        'final_max_db': float(np.max(final_gain_total)),

        # Diagnostics for user-visible Summary / troubleshooting
        # max_boost_db in stats is EFFECTIVE (may be safety-capped in UI before DSP)
        'max_boost_db': float(getattr(cfg, 'max_boost_db', 0.0) or 0.0),
        'max_boost_db_effective': float(getattr(cfg, 'max_boost_db', 0.0) or 0.0),
        'max_boost_db_user': float(getattr(cfg, 'max_boost_db_user', getattr(cfg, 'max_boost_db', 0.0)) or 0.0),
        'max_safe_boost_db': float(getattr(cfg, 'max_safe_boost_db', 0.0) or 0.0),
        'max_cut_db': float(abs(float(getattr(cfg, 'max_cut_db', 15.0) or 15.0))),
        'low_bass_cut_hz': float(getattr(cfg, 'low_bass_cut_hz', 40.0) or 40.0),
        'exc_prot': bool(getattr(cfg, 'exc_prot', False)),
        'exc_freq': float(getattr(cfg, 'exc_freq', 0.0) or 0.0),
        'max_slope_db_per_oct': float(getattr(cfg, 'max_slope_db_per_oct', 0.0) or 0.0),
        'max_slope_boost_db_per_oct': float(getattr(cfg, 'max_slope_boost_db_per_oct', 0.0) or 0.0),
        'max_slope_cut_db_per_oct': float(getattr(cfg, 'max_slope_cut_db_per_oct', 0.0) or 0.0),
        

        # Post-clamp peaks & counts in correction band
        'boost_peak_db': float(locals().get('boost_peak_db', 0.0)),
        'cut_peak_db': float(locals().get('cut_peak_db', 0.0)),
        'boost_bins': int(locals().get('n_boost', 0)),
        'boost_candidate_peak_db': float(locals().get('boost_cand_peak', 0.0)),
        'boost_candidate_bins': int(locals().get('n_boost_cand', 0)),
        'boost_candidate_bins_lowbass': int(locals().get('n_boost_cand_low', 0)),
        'boost_candidate_bins_excprot': int(locals().get('n_boost_cand_exc', 0)),

        # --- Bass-first AI markers (for Summary/debug) ---
        'bass_first_ai': bool(locals().get('use_bassfirst', False)),

        'bass_first_mode_peak_hz': (
            float(freq_axis[int(np.argmax(np.asarray(bf_room_mode)))])
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_room_mode', None) is not None)
                and len(np.asarray(bf_room_mode)) > 0
            )
            else None
        ),

        'bass_first_mode_peak_score': (
            float(np.max(np.asarray(bf_room_mode)))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_room_mode', None) is not None)
                and len(np.asarray(bf_room_mode)) > 0
            )
            else None
        ),

        # True if confidence floor (or any uplift) was applied vs raw reliability
        'bass_first_conf_floor_applied': (
            bool(
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_conf_for_smoothing', None) is not None)
                and (locals().get('bf_rel', None) is not None)
                and np.any(
                    np.asarray(bf_conf_for_smoothing) > (np.asarray(bf_rel) + 1e-6)
                )
            )
            if (
                locals().get('bf_conf_for_smoothing', None) is not None
                and locals().get('bf_rel', None) is not None
            )
            else False
        ),

        # --- BF debug stats (Summary-friendly, 20–200 Hz) ---
        # Raw reliability (0..1)
        'bass_first_rel_mean_20_200': (
            float(np.mean(np.asarray(bf_rel)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_rel', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_rel_min_20_200': (
            float(np.min(np.asarray(bf_rel)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_rel', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        # Effective confidence used by A-FDW smoothing (after floor fuse)
        'bass_first_conf_eff_mean_20_200': (
            float(np.mean(np.asarray(bf_conf_for_smoothing)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_conf_for_smoothing', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_conf_eff_min_20_200': (
            float(np.min(np.asarray(bf_conf_for_smoothing)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_conf_for_smoothing', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        # Room-mode mask strength (0..1)
        'bass_first_roommode_max_20_200': (
            float(np.max(np.asarray(bf_room_mode)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_room_mode', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

    }


    # Attach A-FDW effective BW data if available
    try:
        if bool(afdw_on) and (afdw_bw_oct is not None):
            stats['afdw_bw_oct'] = np.asarray(afdw_bw_oct, dtype=float).tolist()
            stats['afdw_bw_min_oct'] = float(afdw_bw_min_oct) if afdw_bw_min_oct is not None else None
            stats['afdw_bw_mean_oct'] = float(afdw_bw_mean_oct) if afdw_bw_mean_oct is not None else None
            stats['afdw_bw_max_oct'] = float(afdw_bw_max_oct) if afdw_bw_max_oct is not None else None
            stats['afdw_bw_min_hz'] = float(afdw_bw_min_hz) if afdw_bw_min_hz is not None else None
            stats['afdw_bw_max_hz'] = float(afdw_bw_max_hz) if afdw_bw_max_hz is not None else None
    except Exception:
        pass

    # Attach stage probes (for Summary + debugging)
    try:
        # ensure JSON-serializable plain dicts
        stats["stage_probes"] = {k: dict(v) for k, v in stage_probes.items()} if isinstance(stage_probes, dict) else {}
    except Exception:
        stats["stage_probes"] = {}

    # --- Clamp summary into stats (for Summary.txt) ---
    try:
        stats['softclip_boost_bins'] = int(locals().get('softclip_boost_bins', 0))
        stats['softclip_cut_bins']   = int(locals().get('softclip_cut_bins', 0))
        stats['softclip_worst_over_boost_db'] = float(locals().get('over_boost', 0.0))
        stats['softclip_worst_over_cut_db']   = float(locals().get('over_cut', 0.0))

        stats['hardclamp_boost_bins'] = int(locals().get('hardclamp_boost_bins', 0))
        stats['hardclamp_cut_bins']   = int(locals().get('hardclamp_cut_bins', 0))
        stats['hardclamp_worst_over_boost_db'] = float(locals().get('hard_over_boost', 0.0))
        stats['hardclamp_worst_over_cut_db']   = float(locals().get('hard_over_cut', 0.0))

        stats['clamp_summary'] = (
            f"soft_clip: boost={stats['softclip_boost_bins']} cut={stats['softclip_cut_bins']} "
            f"(worst_over_boost={stats['softclip_worst_over_boost_db']:.2f} dB, worst_over_cut={stats['softclip_worst_over_cut_db']:.2f} dB); "
            f"hard_clamp: boost={stats['hardclamp_boost_bins']} cut={stats['hardclamp_cut_bins']} "
            f"(worst_over_boost={stats['hardclamp_worst_over_boost_db']:.2f} dB, worst_over_cut={stats['hardclamp_worst_over_cut_db']:.2f} dB)"
        )
    except Exception:
        stats['clamp_summary'] = "n/a"


    # --- BOOST BLOCKED REASONS (human-readable) ---
    try:
        max_boost_db_cfg = float(getattr(cfg, 'max_boost_db', 0.0) or 0.0)
        low_hz_cfg = float(getattr(cfg, 'low_bass_cut_hz', 40.0) or 40.0)
        exc_on = bool(getattr(cfg, 'exc_prot', False))
        exc_f_cfg = float(getattr(cfg, 'exc_freq', 0.0) or 0.0)
        do_norm = bool(getattr(cfg, 'do_normalize', False))
        g_global = float(getattr(cfg, 'global_gain_db', 0.0) or 0.0)

        boost_bins_post = int(stats.get('boost_bins', 0) or 0)
        boost_bins_cand = int(stats.get('boost_candidate_bins', 0) or 0)
        boost_bins_cand_low = int(stats.get('boost_candidate_bins_lowbass', 0) or 0)
        boost_bins_cand_exc = int(stats.get('boost_candidate_bins_excprot', 0) or 0)

        boost_peak_post = float(stats.get('boost_peak_db', 0.0) or 0.0)
        net_boost_peak = boost_peak_post + g_global + float(stats.get('auto_headroom_db', 0.0) or 0.0)
        stats['net_boost_peak_db'] = float(net_boost_peak)

        reasons = []

        # 0) Absolute: max boost disabled
        if max_boost_db_cfg <= 0.0:
            reasons.append("max_boost_db <= 0 (boost disabled)")

        # 1) No boost candidates at all
        if boost_bins_cand == 0 and boost_bins_post == 0:
            reasons.append("no boost candidates (algorithm produced only cuts in correction band)")

        # 2) Candidates existed but none survived post-clamp
        if boost_bins_cand > 0 and boost_bins_post == 0:
            # All candidates in special restricted regions?
            if boost_bins_cand_low == boost_bins_cand and low_hz_cfg > 0:
                reasons.append(f"all boost candidates were <= low_bass_cut_hz ({low_hz_cfg:.1f} Hz) where cuts-only policy applies")
            if exc_on and exc_f_cfg > 0 and boost_bins_cand_exc == boost_bins_cand:
                reasons.append(f"all boost candidates were < exc_freq ({exc_f_cfg:.1f} Hz) while exc_prot is ON")
            if not reasons:
                reasons.append("boost candidates existed but were removed by limits/safety clamp (check max_boost_db, slope limits, exc/low-bass policies)")

        # 3) Some candidates survived but got reduced
        if boost_bins_cand > 0 and boost_bins_post > 0 and boost_bins_post < boost_bins_cand:
            if boost_bins_cand_low > 0:
                reasons.append(f"some boost candidates were in low-bass restricted region (<= {low_hz_cfg:.1f} Hz)")
            if exc_on and boost_bins_cand_exc > 0 and exc_f_cfg > 0:
                reasons.append(f"some boost candidates were in exc_prot region (< {exc_f_cfg:.1f} Hz)")
            reasons.append("some boost candidates were reduced by limits/safety clamp")

        # 4) Net peak is not positive (user-perceived 'no boosts' even if shape has boosts)
        # This is the classic case when global gain/headroom pushes everything <= 0 dB.
        if boost_bins_post > 0 and net_boost_peak <= 0.0:
            reasons.append(
                f"net boost peak <= 0.00 dB after global gain/headroom (net_peak={net_boost_peak:.2f} dB, normalize={'ON' if do_norm else 'OFF'})"
            )

        # Final text
        stats['boost_blocked_reason'] = "; ".join(dict.fromkeys(reasons)) if reasons else "no blocking detected"
    except Exception:
        stats['boost_blocked_reason'] = "diagnostic unavailable (exception)"


    # attach comparison-mode stats (if any)
    if isinstance(cmp, dict) and cmp:
        stats.update(cmp)
        if stats.get('analysis_mode') != "comparison":
            stats['analysis_mode'] = "native"

    return impulse, stats

def _safe_range(x, default_min=200.0, default_max=3000.0):
    try:
        a = float(x[0]); b = float(x[1])
        if np.isfinite(a) and np.isfinite(b) and b > a:
            return [a, b]
    except Exception:
        pass
    return [float(default_min), float(default_max)]


def generate_filter_pair(f_l, m_l, p_l, f_r, m_r, p_r, cfg: FilterConfig):
    """
    Stereo-link (shared offset) implementation:
    - Determine smart-scan/manual window and per-channel offsets first (no forcing).
    - Compute shared offset from BOTH channels in that same window.
    - Re-run both channels with forced window+shared offset.
    """
    if not bool(getattr(cfg, "stereo_link", False)):
        l_imp, l_st = generate_filter(f_l, m_l, p_l, cfg)
        r_imp, r_st = generate_filter(f_r, m_r, p_r, cfg)
        return l_imp, l_st, r_imp, r_st

    # --- Pass 1: independent leveling, just to discover window + offsets ---
    cfg1 = copy.deepcopy(cfg)
    try:
        cfg1.stereo_link = False
        cfg1.lvl_force_window = None
        cfg1.lvl_force_offset_db = None
    except Exception:
        pass

    l_imp1, l_st1 = generate_filter(f_l, m_l, p_l, cfg1)
    r_imp1, r_st1 = generate_filter(f_r, m_r, p_r, cfg1)

    # Determine the reference window:
    # - Manual mode: use user lvl_min/lvl_max
    # - Auto: use the smart-scan window (prefer left; fall back to right)
    mode = str(getattr(cfg1, "lvl_mode", "Auto") or "Auto")
    if "Manual" in mode:
        win = [float(getattr(cfg1, "lvl_min", 200.0) or 200.0), float(getattr(cfg1, "lvl_max", 3000.0) or 3000.0)]
    else:
        win = _safe_range((l_st1 or {}).get("smart_scan_range"), getattr(cfg1, "lvl_min", 200.0), getattr(cfg1, "lvl_max", 3000.0))
        if not (win[1] > win[0]):
            win = _safe_range((r_st1 or {}).get("smart_scan_range"), getattr(cfg1, "lvl_min", 200.0), getattr(cfg1, "lvl_max", 3000.0))

    # Read per-channel offsets from pass1
    off_l = float((l_st1 or {}).get("offset_db", 0.0) or 0.0)
    off_r = float((r_st1 or {}).get("offset_db", 0.0) or 0.0)
    off_shared = 0.5 * (off_l + off_r)

    # --- Pass 2: force common window + common offset ---
    cfg2 = copy.deepcopy(cfg)
    try:
        cfg2.stereo_link = False  # we force explicitly; do not let per-call stereo state interfere
        cfg2.lvl_force_window = (float(win[0]), float(win[1]))
        cfg2.lvl_force_offset_db = float(off_shared)
    except Exception:
        pass

    l_imp2, l_st2 = generate_filter(f_l, m_l, p_l, cfg2)
    r_imp2, r_st2 = generate_filter(f_r, m_r, p_r, cfg2)

    # Tag stats for visibility
    try:
        if isinstance(l_st2, dict):
            l_st2["offset_method"] = str(l_st2.get("offset_method", "")) + " (StereoLinkShared)"
            l_st2["stereo_link_shared_offset_db"] = float(off_shared)
            l_st2["stereo_link_shared_window"] = [float(win[0]), float(win[1])]
        if isinstance(r_st2, dict):
            r_st2["offset_method"] = str(r_st2.get("offset_method", "")) + " (StereoLinkShared)"
            r_st2["stereo_link_shared_offset_db"] = float(off_shared)
            r_st2["stereo_link_shared_window"] = [float(win[0]), float(win[1])]
    except Exception:
        pass

    return l_imp2, l_st2, r_imp2, r_st2