from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


def _telemetry_list(value: Any) -> list[Any] | None:
    """Normalisoi numpy/list-datan stats-adapteria varten."""
    if value is None:
        return None
    try:
        return np.asarray(value).tolist()
    except Exception:
        return None


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


def apply_baseline_telemetry_to_stats(
    *,
    st: Any,
    cmp: Any,
    native: BaselineNativeTelemetry | None,
    comparison: BaselineComparisonTelemetry | None,
) -> None:
    """Sovittaa typed baseline-telemetrian vanhoihin st/cmp-sanakirjoihin."""

    if isinstance(st, dict) and native is not None:
        st["analysis_mode"] = str(native.analysis_mode)
        if native.freq_axis is not None:
            st["freq_axis"] = _telemetry_list(native.freq_axis)
        if native.measured_mags is not None:
            st["measured_mags"] = _telemetry_list(native.measured_mags)
        if native.target_mags is not None:
            st["target_mags"] = _telemetry_list(native.target_mags)
        if native.target_env_lo is not None and native.target_env_hi is not None:
            st["target_env_lo"] = _telemetry_list(native.target_env_lo)
            st["target_env_hi"] = _telemetry_list(native.target_env_hi)
            st["target_env_pivot_hz"] = (
                float(native.target_env_pivot_hz)
                if native.target_env_pivot_hz is not None
                else None
            )
        st["target_shift_db"] = float(native.target_shift_db)
        st["eff_target_db"] = float(native.eff_target_db)
        st["target_level_db_window"] = float(native.target_level_db_window)
        st["meas_level_db_window"] = float(native.meas_level_db_window)
        st["offset_db"] = float(native.offset_db)
        st["offset_method"] = str(native.offset_method)
        st["smart_scan_range"] = [
            float(native.smart_scan_range[0]),
            float(native.smart_scan_range[1]),
        ]

    if isinstance(cmp, dict) and comparison is not None:
        cmp["analysis_mode"] = str(comparison.analysis_mode)
        if comparison.target_mags is not None:
            cmp["cmp_target_mags"] = _telemetry_list(comparison.target_mags)
        if comparison.measured_mags is not None:
            cmp["cmp_measured_mags"] = _telemetry_list(comparison.measured_mags)
        if comparison.filter_mags is not None:
            cmp["cmp_filter_mags"] = _telemetry_list(comparison.filter_mags)
        cmp["cmp_eff_target_db"] = float(comparison.eff_target_db)
        cmp["cmp_offset_db"] = float(comparison.offset_db)
        cmp["cmp_smart_scan_range"] = [
            float(comparison.smart_scan_range[0]),
            float(comparison.smart_scan_range[1]),
        ]
        cmp["cmp_meas_level_db_window"] = float(comparison.meas_level_db_window)
        cmp["cmp_target_level_db_window"] = float(comparison.target_level_db_window)
        cmp["cmp_offset_method"] = str(comparison.offset_method)
        cmp["cmp_target_shift_db"] = float(comparison.target_shift_db)


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
    """Korjausvaiheen tuotospaketti. Tama pidetaan viela yhteensopivana vanhan dict-API:n kanssa."""

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

    def to_legacy_dict(self) -> dict[str, Any]:
        """Siltametodi: palauttaa vanhan rakenteen ilman kutsujamuutoksia."""
        return {
            "current_rt60": self.current_rt60,
            "rt60_bands": self.rt60_bands,
            "band_avg": self.band_avg,
            "target_mags": self.target_mags,
            "hpf_f": self.hpf_f,
            "hpf_order": self.hpf_order,
            "target_level_db": self.target_level_db,
            "calc_offset_db": self.calc_offset_db,
            "meas_level_db_window": self.meas_level_db_window,
            "target_level_db_window": self.target_level_db_window,
            "offset_method": self.offset_method,
            "s_min": self.s_min,
            "s_max": self.s_max,
            "target_shift_db": self.target_shift_db,
            "cmp": self.cmp,
            "analysis_mode": self.analysis_mode,
            "gain_db": self.gain_db,
            "afdw_on": self.afdw_on,
            "base_sigma": self.base_sigma,
            "_filter_smooth": self.filter_smooth,
            "df_mode": self.df_mode,
            "raw_g": self.raw_g,
            "final_g": self.final_g,
            "mask_c": self.mask_c,
            "stage_probes": self.stage_probes,
            "use_bassfirst": self.use_bassfirst,
            "bf_room_mode": self.bf_room_mode,
            "bf_rel": self.bf_rel,
            "bf_conf_for_smoothing": self.bf_conf_for_smoothing,
            "boost_peak_db": self.boost_peak_db,
            "cut_peak_db": self.cut_peak_db,
            "n_boost": self.n_boost,
            "boost_cand_peak": self.boost_cand_peak,
            "boost_cand_min_hz": self.boost_cand_min_hz,
            "n_boost_cand": self.n_boost_cand,
            "n_boost_cand_low": self.n_boost_cand_low,
            "n_boost_cand_exc": self.n_boost_cand_exc,
            "softclip_boost_bins": self.softclip_boost_bins,
            "softclip_cut_bins": self.softclip_cut_bins,
            "over_boost": self.over_boost,
            "over_cut": self.over_cut,
            "hardclamp_boost_bins": self.hardclamp_boost_bins,
            "hardclamp_cut_bins": self.hardclamp_cut_bins,
            "hard_over_boost": self.hard_over_boost,
            "hard_over_cut": self.hard_over_cut,
            "clamp_dominance_level": self.clamp_dominance_level,
        }


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
