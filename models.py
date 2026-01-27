from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

@dataclass
class FilterConfig:
    
    
    # --- 1. BASIC SETTINGS ---
    fs: int = 44100                 # Sample rate
    num_taps: int = 65536           # Filter length (taps)
    filter_type_str: str = "Linear Phase" # Filter type (Linear, Min, Mixed, Asymmetric)
    global_gain_db: float = 0.0     # Overall gain
    
    # --- 2. CORRECTION LIMITS (MAGNITUDE & PHASE) ---
    mag_c_min: float = 10.0         # Magnitude correction lower limit
    mag_c_max: float = 200.0        # Magnitude correction upper limit
    max_boost_db: float = 5.0       # Maximum allowed boost
    max_cut_db: float = 15.0        # Maximum allowed attenuation (negative gain, dB)
    phase_limit: float = 1000.0     # Phase correction upper limit
    phase_safe_2058: bool = False   # 2058-safe: no room phase correction (only -theo_xo / minphase)
    enable_mag_correction: bool = True # Is magnitude correction enabled
    
    # --- 3. SMOOTHING AND FDW ---
    smoothing_type: str = "Psychoacoustic" # Smoothing type (Standard, Psychoacoustic)
    smoothing_level: int = 12       # Smoothing strength (e.g. 1/12 octave)
    fdw_cycles: float = 10.0        # FDW (Frequency Dependent Windowing) cycles
    reg_strength: float = 30.0      # Regularization (dB), prevents sharp corrections
    max_slope_db_per_oct: float = 12.0  # Max change dB per octave (0 = off)
    # NEW: separate slope limits for boosts and cuts.
    # <= 0 => inherits max_slope_db_per_oct (backward compatible).
    # Idea: boosts often need gentler/lighter limits than cuts.
    max_slope_boost_db_per_oct: float = 0.0
    max_slope_cut_db_per_oct: float = 0.0
    df_smoothing: bool = False      # Raw_g gaussian smoothing: keep constant Hz width across fs/taps
    
    # --- 3B. COMPARISON MODE (locked analysis grid for scoring/reporting) ---
    comparison_mode: bool = False   # Lock score/match analysis to a fixed reference grid
    comparison_ref_fs: int = 44100  # Reference analysis sample rate
    comparison_ref_taps: int = 65536 # Reference analysis FFT length (taps -> rfft bins)

    # --- 4. ADVANCED FEATURES ---
    enable_tdc: bool = True         # Temporal Decay Control (TDC)
    tdc_strength: float = 50.0      # TDC strength in percent
    tdc_max_reduction_db: float = 9.0       # Max total TDC reduction (dB) per frequency bin
    tdc_slope_db_per_oct: float = 6.0       # Optional slope limit for TDC reduction curve (dB/oct), 0 = off
    enable_afdw: bool = True        # Adaptive FDW (A-FDW)
    ir_window_ms: float = 500.0     # Right side time window
    ir_window_ms_left: float = 10  # Left side time window (Pre-ringing)
    mixed_split_freq: float = 300.0 # Mixed Phase filter split frequency
    trans_width: float = 100.0      # Transition width at upper limit
    bass_first_ai: bool = False
    bass_first_mode_max_hz: float = 200.0
    bass_first_smooth_floor_lo: float = 0.75
    bass_first_smooth_floor_hi: float = 0.35
    bass_first_k_mode_cut: float = 0.6
    bass_first_k_mode_boost: float = 0.9
    # Input/source hint for analysis heuristics (affects Bass-first reliability masking)
    # True when the measurement data comes from WAV/IR-derived response rather than REW text/API.
    is_wav_source: bool = False

   

    # --- 5. STRUCTURAL SETTINGS (HPF, XO, TARGET) ---
    hpf_settings: Optional[Dict] = None  # High-pass filter settings (enabled, freq, order)
    crossovers: List[Dict] = field(default_factory=list) # Crossover linearization
    house_freqs: Optional[List[float]] = None # Target curve frequencies
    house_mags: Optional[List[float]] = None  # Target curve magnitudes
    
    # --- 6. LEVELING ---
    lvl_mode: str = "Auto"          # Mode selection (Auto, Manual)
    lvl_algo: str = "Median"        # Algoritmi (Median, Average)
    lvl_manual_db: float = 75.0     # Manuaalinen tavoitetaso
    lvl_min: float = 200.0          # Tasonsovituksen hakualueen alku
    lvl_max: float = 3000.0         # Tasonsovituksen hakualueen loppu
    stereo_link: bool = False
    lvl_force_window: Optional[Tuple[float, float]] = None
    lvl_force_offset_db: Optional[float] = None

    # --- 7. SUOJAUKSET JA OPTIMOINNIT ---
    do_normalize: bool = False       # Normalisointi -1 dBFS tasoon
    exc_prot: bool = False          # Ekskursiosuojaus (basson suojelu)
    exc_freq: float = 40.0          # Suojataajuus
    low_bass_cut_hz: float = 40.0  # Below this frequency only cuts allowed (no boosts)
