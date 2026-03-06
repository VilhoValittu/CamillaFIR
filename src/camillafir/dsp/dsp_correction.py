from __future__ import annotations

import numpy as np

from .correction_baseline import _prepare_correction_baseline
from .correction_mag import _run_mag_correction_pipeline
from .correction_types import (
    CorrectionInputs,
    CorrectionOutputs,
    _MagPipelineInputs,
)


def _sanitize_freq_axis(freq_axis: np.ndarray) -> np.ndarray:
    """Normalisoi taajuusakselin 1D-float-taulukoksi ilman algoritmimuutosta."""
    axis = np.asarray(freq_axis, dtype=float)
    return axis.ravel() if axis.ndim != 1 else axis


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
    stereo_link_ctx=None,
):
    # Vaihe 1.1: julkinen rajapinta pidetaan ennallaan, mutta data paketoidaan.
    inputs = CorrectionInputs(
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
        stereo_link_ctx=stereo_link_ctx,
        logger=logger,
        interpolate_response_fn=interpolate_response_fn,
        apply_confidence_weighted_target_pull_fn=apply_confidence_weighted_target_pull_fn,
        stage_probe_fn=stage_probe_fn,
        cfg_float_allow_zero_fn=cfg_float_allow_zero_fn,
    )
    return _run_correction_stage(inputs).to_legacy_dict()


def _run_correction_stage(inputs: CorrectionInputs) -> CorrectionOutputs:
    # Purku pidetaan alussa, jotta vanha runko sailyy lahes koskemattomana.
    cfg = inputs.cfg
    freq_axis = inputs.freq_axis
    f_in = inputs.f_in
    m_in = inputs.m_in
    reflections = inputs.reflections
    st = inputs.st
    m_anal = inputs.m_anal
    m_plot_db = inputs.m_plot_db
    is_psy = inputs.is_psy
    cmp = inputs.cmp
    analysis_mode = inputs.analysis_mode
    gain_db = inputs.gain_db
    conf_mask = inputs.conf_mask
    complex_meas = inputs.complex_meas
    stereo_link_ctx = inputs.stereo_link_ctx
    logger = inputs.logger
    interpolate_response_fn = inputs.interpolate_response_fn
    apply_confidence_weighted_target_pull_fn = inputs.apply_confidence_weighted_target_pull_fn
    stage_probe_fn = inputs.stage_probe_fn
    cfg_float_allow_zero_fn = inputs.cfg_float_allow_zero_fn

    # A) Input normalisointi & akselit
    freq_axis = _sanitize_freq_axis(freq_axis)
    interpolate_response = interpolate_response_fn
    apply_confidence_weighted_target_pull = apply_confidence_weighted_target_pull_fn
    _stage_probe = stage_probe_fn
    _cfg_float_allow_zero = cfg_float_allow_zero_fn

    # Baseline-vaihe: mittauksen, targetin ja leveloinnin peruskonteksti.
    baseline = _prepare_correction_baseline(
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
        logger=logger,
        interpolate_response=interpolate_response,
        _cfg_float_allow_zero=_cfg_float_allow_zero,
        stereo_link_ctx=stereo_link_ctx,
    )
    current_rt60 = baseline.current_rt60
    rt60_bands = baseline.rt60_bands
    band_avg = baseline.band_avg
    target_mags = baseline.target_mags
    hpf_f = baseline.hpf_f
    hpf_order = baseline.hpf_order
    target_level_db = baseline.target_level_db
    calc_offset_db = baseline.calc_offset_db
    meas_level_db_window = baseline.meas_level_db_window
    target_level_db_window = baseline.target_level_db_window
    offset_method = baseline.offset_method
    s_min = baseline.s_min
    s_max = baseline.s_max
    target_shift_db = baseline.target_shift_db
    cmp = baseline.cmp
    analysis_mode = baseline.analysis_mode
    gain_db = baseline.gain_db

    # Magnitude-korjaus: confidence, regularisointi, smoothing, rajat ja metriikat.
    mag_inputs = _MagPipelineInputs(
        cfg=cfg,
        freq_axis=freq_axis,
        st=st,
        m_anal=m_anal,
        m_interp=baseline.m_interp,
        target_mags=target_mags,
        calc_offset_db=calc_offset_db,
        conf_mask=conf_mask,
        complex_meas=complex_meas,
        logger=logger,
        gain_db=gain_db,
        analysis_mode=analysis_mode,
        cmp=cmp,
        stage_probe=_stage_probe,
        cfg_float_allow_zero=_cfg_float_allow_zero,
        apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
    )
    mag = _run_mag_correction_pipeline(mag_inputs)

    gain_db = mag.gain_db
    afdw_on = mag.afdw_on
    base_sigma = mag.base_sigma
    _filter_smooth = mag.filter_smooth
    df_mode = mag.df_mode
    raw_g = mag.raw_g
    final_g = mag.final_g
    mask_c = mag.mask_c
    stage_probes = mag.stage_probes
    use_bassfirst = mag.use_bassfirst
    bf_room_mode = mag.bf_room_mode
    bf_rel = mag.bf_rel
    bf_conf_for_smoothing = mag.bf_conf_for_smoothing
    boost_peak_db = mag.boost_peak_db
    cut_peak_db = mag.cut_peak_db
    n_boost = mag.n_boost
    boost_cand_peak = mag.boost_cand_peak
    boost_cand_min_hz = mag.boost_cand_min_hz
    n_boost_cand = mag.n_boost_cand
    n_boost_cand_low = mag.n_boost_cand_low
    n_boost_cand_exc = mag.n_boost_cand_exc
    softclip_boost_bins = mag.softclip_boost_bins
    softclip_cut_bins = mag.softclip_cut_bins
    over_boost = mag.over_boost
    over_cut = mag.over_cut
    hardclamp_boost_bins = mag.hardclamp_boost_bins
    hardclamp_cut_bins = mag.hardclamp_cut_bins
    hard_over_boost = mag.hard_over_boost
    hard_over_cut = mag.hard_over_cut
    clamp_dominance_level = mag.clamp_dominance_level

    return CorrectionOutputs(
        current_rt60=float(current_rt60),
        rt60_bands=dict(rt60_bands) if isinstance(rt60_bands, dict) else {},
        band_avg=float(band_avg),
        target_mags=target_mags,
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
        analysis_mode=analysis_mode,
        gain_db=gain_db,
        afdw_on=bool(afdw_on),
        base_sigma=base_sigma,
        filter_smooth=_filter_smooth,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        mask_c=mask_c,
        stage_probes=stage_probes,
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
    )
