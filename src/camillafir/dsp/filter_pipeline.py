import logging

import numpy as np

from camillafir.config.models import FilterConfig

from ._pruning import get_pruning_hook as _get_pruning_hook
from .camillafir_leveling import StereoLinkContext
from .dsp_correction import run_correction_stage
from .dsp_ops import (
    _limit_gd_gradient_ms_per_oct as _limit_gd_gradient_ms_per_oct_impl,
    _stage_probe as _stage_probe_impl,
    apply_confidence_weighted_target_pull as _apply_confidence_weighted_target_pull_impl,
    apply_hpf_to_mags as _apply_hpf_to_mags_impl,
    interpolate_response as _interpolate_response_impl,
)
from .dsp_phase_ir import run_phase_ir_stage
from .dsp_preprocess import run_preprocess
from .dsp_stats import (
    apply_afdw_stats as _apply_afdw_stats_impl,
    apply_boost_blocked_reason as _apply_boost_blocked_reason_impl,
    apply_clamp_stats as _apply_clamp_stats_impl,
    apply_lf_guard_stats as _apply_lf_guard_stats_impl,
    apply_measured_mag_stats as _apply_measured_mag_stats_impl,
    arr_if_valid_for_stats as _arr_if_valid_for_stats_impl,
    safe_stage_probes as _safe_stage_probes_impl,
    safe_stats_update as _safe_stats_update_impl,
)
from .dsp_utils import cfg_float_allow_zero as _cfg_float_allow_zero

logger = logging.getLogger("CamillaFIR.dsp")


def _safe_stats_update(stats: dict, extra) -> None:
    return _safe_stats_update_impl(stats, extra)


def _arr_if_valid_for_stats(value, *, expected_size: int | None = None):
    return _arr_if_valid_for_stats_impl(value, expected_size=expected_size)


def _apply_measured_mag_stats(
    stats: dict,
    *,
    target_mags,
    freq_axis,
    m_anal,
    calc_offset_db: float,
) -> None:
    return _apply_measured_mag_stats_impl(
        stats,
        target_mags=target_mags,
        freq_axis=freq_axis,
        m_anal=m_anal,
        calc_offset_db=calc_offset_db,
    )


def _apply_afdw_stats(
    stats: dict,
    *,
    afdw_on: bool,
    afdw_bw_oct,
    afdw_bw_min_oct,
    afdw_bw_mean_oct,
    afdw_bw_max_oct,
    afdw_bw_min_hz,
    afdw_bw_max_hz,
) -> None:
    return _apply_afdw_stats_impl(
        stats,
        afdw_on=afdw_on,
        afdw_bw_oct=afdw_bw_oct,
        afdw_bw_min_oct=afdw_bw_min_oct,
        afdw_bw_mean_oct=afdw_bw_mean_oct,
        afdw_bw_max_oct=afdw_bw_max_oct,
        afdw_bw_min_hz=afdw_bw_min_hz,
        afdw_bw_max_hz=afdw_bw_max_hz,
    )


def _safe_stage_probes(stage_probes) -> dict:
    return _safe_stage_probes_impl(stage_probes)


def _apply_lf_guard_stats(stats: dict, *, cfg, freq_axis, gain_db) -> None:
    return _apply_lf_guard_stats_impl(stats, cfg=cfg, freq_axis=freq_axis, gain_db=gain_db)


def _apply_clamp_stats(
    stats: dict,
    *,
    softclip_boost_bins: int,
    softclip_cut_bins: int,
    over_boost: float,
    over_cut: float,
    hardclamp_boost_bins: int,
    hardclamp_cut_bins: int,
    hard_over_boost: float,
    hard_over_cut: float,
) -> None:
    return _apply_clamp_stats_impl(
        stats,
        softclip_boost_bins=softclip_boost_bins,
        softclip_cut_bins=softclip_cut_bins,
        over_boost=over_boost,
        over_cut=over_cut,
        hardclamp_boost_bins=hardclamp_boost_bins,
        hardclamp_cut_bins=hardclamp_cut_bins,
        hard_over_boost=hard_over_boost,
        hard_over_cut=hard_over_cut,
    )


def _apply_boost_blocked_reason(stats: dict, *, cfg) -> None:
    return _apply_boost_blocked_reason_impl(stats, cfg=cfg)


def apply_confidence_weighted_target_pull(
    target_db,
    measured_db,
    confidence_mask,
    *,
    conf_floor: float = 0.07,
    conf_ceil: float = 0.95,
    freq_axis=None,
    freq_limit_hz: float | None = 400.0,
    gamma_cut: float = 0.70,
    gamma_boost: float = 1.20,
    return_telemetry: bool = False,
):
    return _apply_confidence_weighted_target_pull_impl(
        target_db,
        measured_db,
        confidence_mask,
        conf_floor=conf_floor,
        conf_ceil=conf_ceil,
        freq_axis=freq_axis,
        freq_limit_hz=freq_limit_hz,
        gamma_cut=gamma_cut,
        gamma_boost=gamma_boost,
        return_telemetry=return_telemetry,
    )


def _limit_gd_gradient_ms_per_oct(
    freq_axis,
    phase_rad,
    *,
    mask=None,
    max_grad_ms_per_oct=30.0,
    f_min=20.0,
    f_max=250.0,
    grad_smooth_sigma=0.8,
    soft_limit=True,
):
    return _limit_gd_gradient_ms_per_oct_impl(
        freq_axis,
        phase_rad,
        mask=mask,
        max_grad_ms_per_oct=max_grad_ms_per_oct,
        f_min=f_min,
        f_max=f_max,
        grad_smooth_sigma=grad_smooth_sigma,
        soft_limit=soft_limit,
    )


def _stage_probe(stage_name, freq_axis, arr_db, mask_c, global_gain_db=0.0, auto_headroom_db=0.0, logger_obj=None):
    return _stage_probe_impl(
        stage_name,
        freq_axis,
        arr_db,
        mask_c,
        global_gain_db=global_gain_db,
        auto_headroom_db=auto_headroom_db,
        logger_obj=logger_obj,
    )


def apply_hpf_to_mags(freqs, mags, cutoff, order):
    return _apply_hpf_to_mags_impl(freqs, mags, cutoff, order)


def interpolate_response(input_freqs, input_values, target_freqs):
    return _interpolate_response_impl(input_freqs, input_values, target_freqs)


def _run_generate_filter_pipeline(
    freqs,
    meas_mags,
    raw_phases,
    cfg: FilterConfig,
    *,
    stereo_link_ctx: StereoLinkContext | None = None,
) -> dict:
    prep = run_preprocess(freqs, meas_mags, raw_phases, cfg, stereo_link_ctx=stereo_link_ctx)
    f_in = prep.f_in
    m_in = prep.m_in
    n_fft = prep.ctx.n_fft
    freq_axis = prep.ctx.freq_axis
    gain_db = prep.ctx.gain_db
    st = prep.ctx.st
    target_mags = prep.ctx.target_mags
    m_interp = prep.m_interp
    p_rad_interp = prep.p_rad_interp
    delay_slope = prep.delay_slope
    m_plot_db = prep.m_plot_db
    complex_meas = prep.complex_meas
    m_anal = prep.m_anal
    conf_mask = prep.conf_mask
    reflections = prep.reflections
    cmp = prep.cmp
    analysis_mode = prep.analysis_mode
    is_psy = prep.is_psy
    corr = run_correction_stage(
        cfg=cfg,
        freq_axis=freq_axis,
        f_in=f_in,
        m_in=m_in,
        reflections=reflections,
        st=st,
        m_anal=m_anal,
        m_plot_db=m_plot_db,
        is_psy=is_psy,
        cmp=cmp,
        analysis_mode=analysis_mode,
        gain_db=gain_db,
        conf_mask=conf_mask,
        complex_meas=complex_meas,
        logger=logger,
        interpolate_response_fn=interpolate_response,
        apply_confidence_weighted_target_pull_fn=apply_confidence_weighted_target_pull,
        stage_probe_fn=_stage_probe,
        cfg_float_allow_zero_fn=_cfg_float_allow_zero,
        stereo_link_ctx=stereo_link_ctx,
    )

    current_rt60 = corr.current_rt60
    rt60_bands = corr.rt60_bands
    band_avg = corr.band_avg
    target_mags = corr.target_mags
    hpf_f = corr.hpf_f
    hpf_order = corr.hpf_order
    target_level_db = corr.target_level_db
    calc_offset_db = corr.calc_offset_db
    meas_level_db_window = corr.meas_level_db_window
    target_level_db_window = corr.target_level_db_window
    offset_method = corr.offset_method
    s_min = corr.s_min
    s_max = corr.s_max
    target_shift_db = corr.target_shift_db
    cmp = corr.cmp
    analysis_mode = corr.analysis_mode
    gain_db = corr.gain_db
    afdw_on = corr.afdw_on
    base_sigma = corr.base_sigma
    _filter_smooth = corr.filter_smooth
    df_mode = corr.df_mode
    raw_g = corr.raw_g
    final_g = corr.final_g
    mask_c = corr.mask_c
    stage_probes = corr.stage_probes
    use_bassfirst = corr.use_bassfirst
    bf_room_mode = corr.bf_room_mode
    bf_rel = corr.bf_rel
    bf_conf_for_smoothing = corr.bf_conf_for_smoothing
    boost_peak_db = corr.boost_peak_db
    cut_peak_db = corr.cut_peak_db
    n_boost = corr.n_boost
    boost_cand_peak = corr.boost_cand_peak
    boost_cand_min_hz = corr.boost_cand_min_hz
    n_boost_cand = corr.n_boost_cand
    n_boost_cand_low = corr.n_boost_cand_low
    n_boost_cand_exc = corr.n_boost_cand_exc
    softclip_boost_bins = corr.softclip_boost_bins
    softclip_cut_bins = corr.softclip_cut_bins
    over_boost = corr.over_boost
    over_cut = corr.over_cut
    hardclamp_boost_bins = corr.hardclamp_boost_bins
    hardclamp_cut_bins = corr.hardclamp_cut_bins
    hard_over_boost = corr.hard_over_boost
    hard_over_cut = corr.hard_over_cut

    _pruning_hook = _get_pruning_hook()
    if callable(_pruning_hook):
        try:
            _g = np.asarray(getattr(corr, "final_g", []), dtype=float)
            _g_fin = _g[np.isfinite(_g)]
            _p90 = float(np.percentile(np.abs(_g_fin), 90)) if _g_fin.size > 0 else 0.0
            _clip_pen = (
                float(getattr(corr, "over_boost", 0.0) or 0.0) * 5.0
                + float(getattr(corr, "over_cut", 0.0) or 0.0) * 2.0
            )
            _pruning_hook(-(_p90 + _clip_pen))
        except Exception:
            pass

    phase_ir = run_phase_ir_stage(
        cfg=cfg,
        freq_axis=freq_axis,
        n_fft=n_fft,
        gain_db=gain_db,
        p_rad_interp=p_rad_interp,
        conf_mask=conf_mask,
        m_anal=m_anal,
        calc_offset_db=calc_offset_db,
        target_mags=target_mags,
        st=st,
        mask_c=mask_c,
        base_sigma=base_sigma,
        _filter_smooth=_filter_smooth,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        use_bassfirst=use_bassfirst,
        afdw_on=afdw_on,
        logger=logger,
        apply_hpf_to_mags_fn=apply_hpf_to_mags,
        limit_gd_gradient_ms_per_oct_fn=_limit_gd_gradient_ms_per_oct,
        cfg_float_allow_zero_fn=_cfg_float_allow_zero,
    )

    impulse = phase_ir.impulse
    gain_db = phase_ir.gain_db
    auto_global_gain_db = phase_ir.auto_global_gain_db
    gain_margin_db = phase_ir.gain_margin_db
    auto_headroom_db = phase_ir.auto_headroom_db
    current_peak_gain = phase_ir.current_peak_gain
    final_gain_total = phase_ir.final_gain_total

    return {
        "cfg": cfg,
        "freq_axis": freq_axis,
        "st": st,
        "reflections": reflections,
        "target_mags": target_mags,
        "m_anal": m_anal,
        "conf_mask": conf_mask,
        "cmp": cmp,
        "analysis_mode": analysis_mode,
        "delay_slope": delay_slope,
        "current_rt60": current_rt60,
        "rt60_bands": rt60_bands,
        "band_avg": band_avg,
        "target_level_db": target_level_db,
        "calc_offset_db": calc_offset_db,
        "meas_level_db_window": meas_level_db_window,
        "target_level_db_window": target_level_db_window,
        "offset_method": offset_method,
        "s_min": s_min,
        "s_max": s_max,
        "target_shift_db": target_shift_db,
        "gain_db": gain_db,
        "afdw_on": afdw_on,
        "mask_c": mask_c,
        "stage_probes": stage_probes,
        "use_bassfirst": use_bassfirst,
        "bf_room_mode": bf_room_mode,
        "bf_rel": bf_rel,
        "bf_conf_for_smoothing": bf_conf_for_smoothing,
        "boost_peak_db": boost_peak_db,
        "cut_peak_db": cut_peak_db,
        "n_boost": n_boost,
        "boost_cand_peak": boost_cand_peak,
        "boost_cand_min_hz": boost_cand_min_hz,
        "n_boost_cand": n_boost_cand,
        "n_boost_cand_low": n_boost_cand_low,
        "n_boost_cand_exc": n_boost_cand_exc,
        "softclip_boost_bins": softclip_boost_bins,
        "softclip_cut_bins": softclip_cut_bins,
        "over_boost": over_boost,
        "over_cut": over_cut,
        "hardclamp_boost_bins": hardclamp_boost_bins,
        "hardclamp_cut_bins": hardclamp_cut_bins,
        "hard_over_boost": hard_over_boost,
        "hard_over_cut": hard_over_cut,
        "impulse": impulse,
        "auto_global_gain_db": auto_global_gain_db,
        "gain_margin_db": gain_margin_db,
        "auto_headroom_db": auto_headroom_db,
        "current_peak_gain": current_peak_gain,
        "final_gain_total": final_gain_total,
    }
