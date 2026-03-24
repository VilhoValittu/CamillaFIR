from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class BaselineNativeTelemetry:
    """Typed baseline-telemetria natiiville stats-rakenteelle."""

    analysis_mode: str
    freq_axis: np.ndarray | None
    measured_mags: np.ndarray | None
    target_mags: np.ndarray | None
    target_env_lo: np.ndarray | None
    target_env_hi: np.ndarray | None
    target_env_pivot_hz: float | None
    target_shift_db: float
    eff_target_db: float
    target_level_db_window: float
    meas_level_db_window: float
    offset_db: float
    offset_method: str
    smart_scan_range: tuple[float, float]


@dataclass
class BaselineComparisonTelemetry:
    """Typed baseline-telemetria comparison-stats-rakenteelle."""

    analysis_mode: str
    target_mags: np.ndarray | None
    measured_mags: np.ndarray | None
    filter_mags: np.ndarray | None
    eff_target_db: float
    offset_db: float
    smart_scan_range: tuple[float, float]
    meas_level_db_window: float
    target_level_db_window: float
    offset_method: str
    target_shift_db: float
@dataclass
class CorrectionInputs:
    """Korjausvaiheen syotepaketti, jolla pidetaan funktioraja siistina."""

    cfg: Any
    freq_axis: np.ndarray
    f_in: np.ndarray
    m_in: np.ndarray
    reflections: Any
    st: Any
    m_anal: np.ndarray
    m_plot_db: np.ndarray | None
    is_psy: bool
    cmp: Any
    analysis_mode: str
    gain_db: np.ndarray
    conf_mask: np.ndarray
    complex_meas: np.ndarray
    stereo_link_ctx: Any | None
    logger: Any
    interpolate_response_fn: Callable[..., np.ndarray]
    apply_confidence_weighted_target_pull_fn: Callable[..., Any]
    stage_probe_fn: Callable[..., Any]
    cfg_float_allow_zero_fn: Callable[[Any, str, float], float]


@dataclass
class CorrectionOutputs:
    """Korjausvaiheen typed tuotospaketti."""

    current_rt60: float
    rt60_bands: dict[str, Any]
    band_avg: float
    target_mags: Any
    hpf_f: float
    hpf_order: int
    target_level_db: float
    calc_offset_db: float
    meas_level_db_window: float
    target_level_db_window: float
    offset_method: str
    s_min: float
    s_max: float
    target_shift_db: float
    cmp: Any
    analysis_mode: str
    gain_db: np.ndarray
    afdw_on: bool
    base_sigma: Any
    filter_smooth: Any
    df_mode: Any
    raw_g: Any
    final_g: Any
    mask_c: np.ndarray
    stage_probes: dict[str, Any]
    use_bassfirst: bool
    bf_room_mode: Any
    bf_rel: Any
    bf_conf_for_smoothing: Any
    boost_peak_db: float
    cut_peak_db: float
    n_boost: int
    boost_cand_peak: float
    boost_cand_min_hz: float
    n_boost_cand: int
    n_boost_cand_low: int
    n_boost_cand_exc: int
    softclip_boost_bins: int
    softclip_cut_bins: int
    over_boost: float
    over_cut: float
    hardclamp_boost_bins: int
    hardclamp_cut_bins: int
    hard_over_boost: float
    hard_over_cut: float
    clamp_dominance_level: str


@dataclass
class _BaselineContext:
    """Sisainen baseline-vaiheen konteksti."""

    # Mittaus interpoloituna korjausakselille (kaytetaan myos bassfirst-haarassa).
    m_interp: np.ndarray
    current_rt60: float
    rt60_bands: dict[str, Any]
    band_avg: float
    target_mags: np.ndarray
    hpf_f: float
    hpf_order: int
    target_level_db: float
    calc_offset_db: float
    meas_level_db_window: float
    target_level_db_window: float
    offset_method: str
    s_min: float
    s_max: float
    target_shift_db: float
    cmp: Any
    analysis_mode: str
    gain_db: np.ndarray
    native_telemetry: BaselineNativeTelemetry | None
    comparison_telemetry: BaselineComparisonTelemetry | None


@dataclass
class _MagCorrectionContext:
    """Sisainen mag-correction-vaiheen konteksti."""

    afdw_on: bool
    base_sigma: Any
    filter_smooth: Any
    df_mode: Any
    raw_g: Any
    final_g: Any
    mask_c: np.ndarray
    stage_probes: dict[str, Any]
    use_bassfirst: bool
    bf_room_mode: Any
    bf_rel: Any
    bf_conf_for_smoothing: Any
    boost_peak_db: float
    cut_peak_db: float
    n_boost: int
    boost_cand_peak: float
    boost_cand_min_hz: float
    n_boost_cand: int
    n_boost_cand_low: int
    n_boost_cand_exc: int
    softclip_boost_bins: int
    softclip_cut_bins: int
    over_boost: float
    over_cut: float
    hardclamp_boost_bins: int
    hardclamp_cut_bins: int
    hard_over_boost: float
    hard_over_cut: float
    clamp_dominance_level: str
    gain_db: np.ndarray


@dataclass
class _MagPipelineInputs:
    """Sisainen syotepaketti mag-korjausputkelle."""

    cfg: Any
    freq_axis: np.ndarray
    st: Any
    m_anal: np.ndarray
    m_interp: np.ndarray
    target_mags: np.ndarray
    calc_offset_db: float
    conf_mask: np.ndarray
    complex_meas: np.ndarray
    logger: Any
    gain_db: np.ndarray
    analysis_mode: str
    cmp: Any
    stage_probe: Callable[..., Any]
    cfg_float_allow_zero: Callable[[Any, str, float], float]
    apply_confidence_weighted_target_pull: Callable[..., Any]


@dataclass
class _MagPostProcessInputs:
    """Sisainen syotepaketti mag-pipelinen post-limit-vaiheelle."""

    cfg: Any
    freq_axis: np.ndarray
    st: Any
    logger: Any
    stage_probe: Callable[..., Any]
    cfg_float_allow_zero: Callable[[Any, str, float], float]
    mask_c: np.ndarray
    gain_db: np.ndarray
    gain_apply: np.ndarray
    raw_g: np.ndarray
    final_g: np.ndarray
    pre_bass_adapt_g: np.ndarray | None
    raw_safe_ref: np.ndarray | None
    conf_mask: np.ndarray
    filter_smooth: float
    debug_stage_stats: bool
    stage_probes: dict[str, Any]
    apply_confidence_weighted_target_pull: Callable[..., Any]
    m_anal: np.ndarray
    target_mags: np.ndarray
    calc_offset_db: float


@dataclass
class _MagPostProcessOutputs:
    """Post-limit-vaiheen tuotokset yhdessa paketissa."""

    gain_db: np.ndarray
    stage_probes: dict[str, Any]
    boost_peak_db: float
    cut_peak_db: float
    n_boost: int
    boost_cand_peak: float
    boost_cand_min_hz: float
    n_boost_cand: int
    n_boost_cand_low: int
    n_boost_cand_exc: int
    softclip_boost_bins: int
    softclip_cut_bins: int
    over_boost: float
    over_cut: float
    hardclamp_boost_bins: int
    hardclamp_cut_bins: int
    hard_over_boost: float
    hard_over_cut: float
    clamp_dominance_level: str


@dataclass
class _MagCoreOutputs:
    """Mag-pipelinen core-vaiheen tuotokset post-vaihetta varten."""

    mag_enabled: bool
    debug_stage_stats: bool
    afdw_on: bool
    base_sigma: Any
    filter_smooth: Any
    df_mode: Any
    raw_g: Any
    final_g: Any
    mask_c: np.ndarray
    stage_probes: dict[str, Any]
    use_bassfirst: bool
    bf_room_mode: Any
    bf_rel: Any
    bf_conf_for_smoothing: Any
    pre_bass_adapt_g: Any
    gain_db: np.ndarray
    gain_apply: np.ndarray
    raw_safe_ref: np.ndarray | None


@dataclass
class _MagRawStageOutputs:
    """Core-vaiheen ensimmaisen osan tuotokset (error/smoothing/reg)."""

    mag_enabled: bool
    debug_stage_stats: bool
    afdw_on: bool
    afdw_base: float
    afdw_min: float
    filter_smooth: Any
    base_sigma: Any
    df_mode: Any
    raw_g: Any
    final_g: Any
    gain_db: np.ndarray
    stage_probes: dict[str, Any]


@dataclass
class _MagAdaptiveStageOutputs:
    """Core-vaiheen toisen osan tuotokset (bassfirst/afdw/confidence)."""

    final_g: Any
    mask_c: np.ndarray
    stage_probes: dict[str, Any]
    use_bassfirst: bool
    bf_room_mode: Any
    bf_rel: Any
    bf_conf_for_smoothing: Any
    pre_bass_adapt_g: Any
    gain_db: np.ndarray
    gain_apply: np.ndarray
    raw_safe_ref: np.ndarray | None
