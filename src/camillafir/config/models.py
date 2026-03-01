from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Union

@dataclass
class FilterConfig:
    
    
    fs: int = 44100
    num_taps: int = 65536
    filter_type_str: str = "Linear Phase"
    global_gain_db: float = 0.0
    
    mag_c_min: float = 10.0
    mag_c_max: float = 200.0
    max_boost_db: float = 5.0
    max_cut_db: float = 15.0
    phase_limit: float = 1000.0
    phase_safe_2058: bool = False
    enable_mag_correction: bool = True
    
    plot_smoothing_level: Union[str, int] = "Psychoacoustic"
    filter_smooth: int = 12
    fdw_cycles: float = 10.0
    reg_strength: float = 30.0
    max_slope_db_per_oct: float = 12.0
    max_slope_boost_db_per_oct: float = 0.0
    max_slope_cut_db_per_oct: float = 0.0
    df_smoothing: bool = False
    bass_smooth_adaptive: bool = True
    bass_smooth_hz: float = 200.0
    bass_smooth_sigma_scale: float = 1.4
    bass_smooth_conf_floor: float = 0.3
    bass_adaptive_isolation_mode: bool = False
    mid_refit_enable: bool = True
    mid_refit_hz_lo: float = 200.0
    mid_refit_hz_hi: float = 2000.0
    mid_refit_k: float = 0.45
    mid_refit_smooth_oct: float = 0.60
    mid_refit_conf_min_avg: float = 0.20
    bass_boost_cap_enable: bool = True
    bass_boost_cap_hz: float = 200.0
    bass_boost_cap_extra_db: float = 2.0
    bass_boost_cap_conf_min: float = 0.55
    bass_boost_post_restore_enable: bool = True
    bass_boost_post_restore_strength: float = 0.60
    
    comparison_mode: bool = True
    comparison_ref_fs: int = 44100
    comparison_ref_taps: int = 65536

    enable_tdc: bool = True
    tdc_strength: float = 50.0
    tdc_max_reduction_db: float = 9.0
    tdc_slope_db_per_oct: float = 6.0
    enable_afdw: bool = True

    ir_export_window_mode: str = "auto"
    ir_export_window_shape: str = "hann"
    ir_export_tukey_alpha: float = 0.25
    ir_anchor_mode: str = "min_causal"
    min_causal_ms: float = 80.0
    auto_asym_left_ratio: float = 0.35
    auto_asym_left_max_ms: float = 25.0
    ir_window_ms: float = 500.0
    ir_window_ms_left: float = 85.0
    ir_window_right: float = 500.0
    ir_window_left: float = 85.0
    mixed_split_freq: float = 180.0
    trans_width: float = 100.0
    mixed_transition_mode: str = "width_based"
    mixed_confidence_blend_enable: bool = False
    mixed_confidence_power: float = 1.5
    mixed_phase_budget_lf_deg: float = 45.0
    mixed_phase_budget_hf_deg: float = 22.5
    mixed_min_tilt_comp_enable: bool = True
    excess_phase_strength: float = 0.9
    low_freq_full_correction_hz: float = 140.0
    high_freq_no_correction_hz: float = 900.0
    phase_boundary_smooth_sigma_bins: float = 1.2
    phase_tail_monotonic_enable: bool = True
    phase_tail_start_ratio: float = 0.72
    phase_tail_abs_smooth_sigma_bins: float = 2.5
    phase_tail_cosine_strength: float = 0.85
    linear_phase_blend_start_ratio: float = 0.65
    enable_ir_pre_energy_guard: bool = True
    pre_energy_ratio_max: float = 0.25
    pre_energy_guard_strength: float = 0.8
    max_pre_ringing_db: float = -35.0
    max_excess_delay_ms: float = 2.5
    gd_grad_limit_ms_per_oct: float = 20.0
    bass_first_ai: bool = False
    bass_first_mode_max_hz: float = 200.0
    bass_first_smooth_floor_lo: float = 0.75
    bass_first_smooth_floor_hi: float = 0.35
    bass_first_k_mode_cut: float = 0.6
    bass_first_k_mode_boost: float = 0.9
    is_wav_source: bool = False

   

    hpf_settings: Optional[Dict] = None
    crossovers: List[Dict] = field(default_factory=list)
    house_freqs: Optional[List[float]] = None
    house_mags: Optional[List[float]] = None
    
    lvl_mode: str = "Auto"
    lvl_algo: str = "Median"
    lvl_manual_db: float = 0.0
    lvl_min: float = 200.0
    lvl_max: float = 3000.0
    stereo_link: bool = False
    lvl_force_window: Optional[Tuple[float, float]] = None
    lvl_force_offset_db: Optional[float] = None

    do_normalize: bool = False
    exc_prot: bool = False
    exc_freq: float = 40.0
    low_bass_cut_hz: float = 40.0

    conf_pull_floor: float = 0.15
    conf_pull_ceil: float = 0.95
    conf_pull_max_hz: Optional[float] = 200.0
    conf_pull_gamma_cut: float = 0.55
    conf_pull_gamma_boost: float = 1.35

    conf_pull_conf_smooth_sigma: float = 2.0
    conf_pull_bass_floor_hz: float = 120.0
    conf_pull_bass_floor_min: float = 0.25
    conf_pull_bass_boost_floor_hz: float = 200.0
    conf_pull_bass_boost_floor_min: float = 0.45
    conf_pull_bass_boost_restore: float = 0.55

    low_bass_cut_enable: bool = True
    low_bass_cut_strength: float = 1.0
