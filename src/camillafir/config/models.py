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
    max_pre_ringing_db: float = -35.0
    max_excess_delay_ms: float = 2.5
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

    low_bass_cut_enable: bool = True
    low_bass_cut_strength: float = 1.0
