from __future__ import annotations

import numpy as np

from .camillafir_analysis import analyze_acoustic_confidence
from .camillafir_leveling import compute_leveling
from .phase import remove_time_of_flight
from .smoothing import apply_adaptive_fdw, apply_smoothing_std, psychoacoustic_smoothing
from .dsp_types import DspContext, PreprocessResult


def analysis_smoothing_lf_to_hf(
    freqs,
    mags,
    *,
    low_bw=1 / 48.0,
    high_bw=1 / 3.0,
    f_lo=230.0,
    f_hi=500.0,
):
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(mags, dtype=float)
    if f.size < 8 or m.size != f.size:
        return np.copy(m)

    dummy = np.zeros_like(m)
    try:
        m_low, _ = apply_smoothing_std(f, m, dummy, float(low_bw))
        m_high, _ = apply_smoothing_std(f, m, dummy, float(high_bw))
    except Exception:
        return np.copy(m)

    ff = np.maximum(f, 1.0)
    lo = float(max(f_lo, 1.0))
    hi = float(max(f_hi, lo * 1.01))
    w = (np.log10(ff) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    w = np.clip(w, 0.0, 1.0)

    return (1.0 - w) * m_low + w * m_high


def run_preprocess(freqs, meas_mags, raw_phases, cfg, *, stereo_link_ctx=None) -> PreprocessResult:
    min_len = min(len(freqs), len(meas_mags), len(raw_phases))
    f_in = np.asarray(freqs[:min_len], dtype=float)
    m_in = np.asarray(meas_mags[:min_len], dtype=float)
    p_in = np.asarray(raw_phases[:min_len], dtype=float)

    if f_in.size > 1:
        order = np.argsort(f_in, kind="mergesort")
        f_in = f_in[order]
        m_in = m_in[order]
        p_in = p_in[order]
        uniq_mask = np.concatenate(([True], np.diff(f_in) > 0.0))
        f_in = f_in[uniq_mask]
        m_in = m_in[uniq_mask]
        p_in = p_in[uniq_mask]

    n_fft = cfg.num_taps if cfg.num_taps % 2 != 0 else cfg.num_taps + 1
    freq_axis = np.linspace(0, cfg.fs / 2.0, n_fft // 2 + 1)

    gain_db = np.zeros_like(freq_axis, dtype=float)
    st: dict = {}
    target_mags = np.zeros_like(freq_axis, dtype=float)

    is_psy = "psy" in str(cfg.plot_smoothing_level).lower()

    m_smooth_std = analysis_smoothing_lf_to_hf(
        f_in, m_in, low_bw=1 / 48.0, high_bw=1 / 3.0, f_lo=230.0, f_hi=400.0
    )

    p_smooth_oct = 1 / 12.0 if cfg.fs > 96000 else 1 / 24.0
    p_smooth, _ = apply_smoothing_std(f_in, p_in, np.zeros_like(p_in), p_smooth_oct)

    m_interp = np.interp(freq_axis, f_in, m_in)
    p_rad_raw = np.deg2rad(np.interp(freq_axis, f_in, p_in))
    p_rad_interp, delay_slope = remove_time_of_flight(freq_axis, p_rad_raw)

    m_plot_db = None
    if is_psy:
        try:
            m_plot_db = psychoacoustic_smoothing(freq_axis, m_interp)
        except Exception:
            m_plot_db = None

    complex_meas = 10 ** (m_interp / 20.0) * np.exp(1j * p_rad_interp)

    m_anal = np.interp(freq_axis, f_in, m_smooth_std)
    p_anal_rad = np.deg2rad(np.interp(freq_axis, f_in, p_smooth))
    p_anal_rad, _ = remove_time_of_flight(freq_axis, p_anal_rad)
    complex_anal = 10 ** (m_anal / 20.0) * np.exp(1j * p_anal_rad)

    conf_mask, reflections, _ = analyze_acoustic_confidence(freq_axis, complex_anal, cfg.fs)

    cmp = None
    analysis_mode = "native"
    try:
        if bool(getattr(cfg, "comparison_mode", False)):
            ref_fs = int(getattr(cfg, "comparison_ref_fs", 44100) or 44100)
            ref_taps = int(getattr(cfg, "comparison_ref_taps", 65536) or 65536)
            ref_nfft = ref_taps if (ref_taps % 2 != 0) else (ref_taps + 1)
            freq_cmp_full = np.linspace(0, ref_fs / 2.0, ref_nfft // 2 + 1)

            fmax = float(freq_axis[-1]) if freq_axis.size else 0.0
            if fmax > 0:
                freq_cmp = freq_cmp_full[freq_cmp_full <= fmax]
            else:
                freq_cmp = freq_cmp_full

            m_cmp_raw = np.interp(freq_cmp, freq_axis, m_anal)
            p_cmp_rad = np.interp(freq_cmp, freq_axis, p_anal_rad)
            complex_cmp = 10 ** (m_cmp_raw / 20.0) * np.exp(1j * p_cmp_rad)

            conf_cmp, refl_cmp, _ = analyze_acoustic_confidence(freq_cmp, complex_cmp, ref_fs)

            target_cmp = np.interp(freq_cmp, freq_axis, target_mags)
            (
                target_level_db_cmp,
                calc_offset_db_cmp,
                meas_level_db_window_cmp,
                target_level_db_window_cmp,
                offset_method_cmp,
                s_min_cmp,
                s_max_cmp,
            ) = compute_leveling(cfg, freq_cmp, m_cmp_raw, target_cmp, stereo_link_ctx=stereo_link_ctx)

            filt_cmp = np.interp(freq_cmp, freq_axis, gain_db)

            cmp = {
                "cmp_ref_fs": float(ref_fs),
                "cmp_ref_taps": float(ref_taps),
                "cmp_freq_axis": freq_cmp.tolist(),
                "cmp_target_mags": target_cmp.tolist(),
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

    if getattr(cfg, "enable_afdw", False):
        base = float(getattr(cfg, "fdw_cycles", 15.0))
        min_c = max(3.0, base / 3.0)
        m_anal = apply_adaptive_fdw(
            freq_axis,
            m_anal,
            conf_mask,
            base_cycles=base,
            min_cycles=min_c,
        )

    ctx = DspContext(
        n_fft=n_fft,
        freq_axis=freq_axis,
        gain_db=gain_db,
        target_mags=target_mags,
        st=st,
    )

    return PreprocessResult(
        ctx=ctx,
        f_in=f_in,
        m_in=m_in,
        p_in=p_in,
        m_smooth_std=m_smooth_std,
        p_smooth=p_smooth,
        m_interp=m_interp,
        p_rad_raw=p_rad_raw,
        p_rad_interp=p_rad_interp,
        delay_slope=float(delay_slope),
        m_plot_db=None if m_plot_db is None else np.asarray(m_plot_db, dtype=float),
        complex_meas=complex_meas,
        m_anal=m_anal,
        p_anal_rad=p_anal_rad,
        complex_anal=complex_anal,
        conf_mask=conf_mask,
        reflections=reflections,
        cmp=cmp,
        analysis_mode=analysis_mode,
        is_psy=is_psy,
    )
