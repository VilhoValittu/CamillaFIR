from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class FilterConfig:
    
    
    # --- 1. PERUSASETUKSET ---
    fs: int = 44100                 # Näytetaajuus
    num_taps: int = 65536           # Filtterin pituus (taps)
    filter_type_str: str = "Linear Phase" # Suodintyyppi (Linear, Min, Mixed, Asymmetric)
    global_gain_db: float = 0.0     # Yleinen vahvistus
    
    # --- 2. KORJAUSRAJAT (MAGNITUDE & PHASE) ---
    mag_c_min: float = 10.0         # Magnitudikorjauksen alaraja
    mag_c_max: float = 200.0        # Magnitudikorjauksen yläraja
    max_boost_db: float = 5.0       # Suurin sallittu korostus    
    max_cut_db: float = 15.0        # Suurin sallittu vaimennus (negatiivinen gain, dB)
    phase_limit: float = 1000.0     # Vaihekorjauksen yläraja
    enable_mag_correction: bool = True # Onko magnitudikorjaus päällä
    
    # --- 3. TASOITUS JA FDW ---
    smoothing_type: str = "Psychoacoustic" # Tasoitustyyppi (Standard, Psychoacoustic)
    smoothing_level: int = 12       # Tasoituksen voimakkuus (esim. 1/12 oktaavia)
    fdw_cycles: float = 15.0        # FDW (Frequency Dependent Windowing) syklit
    reg_strength: float = 30.0      # Regularisointi (dB), estää jyrkät korjaukset
    max_slope_db_per_oct: float = 12.0  # Max muutos dB per oktaavi (0 = pois)
    df_smoothing: bool = False      # Raw_g gaussian smoothing: keep constant Hz width across fs/taps
    
    # --- 3B. COMPARISON MODE (locked analysis grid for scoring/reporting) ---
    comparison_mode: bool = False   # Lock score/match analysis to a fixed reference grid
    comparison_ref_fs: int = 44100  # Reference analysis sample rate
    comparison_ref_taps: int = 65536 # Reference analysis FFT length (taps -> rfft bins)

    # --- 4. EDISTYNEET OMINAISUUDET ---
    enable_tdc: bool = True         # Temporal Decay Control (TDC)
    tdc_strength: float = 50.0      # TDC:n voimakkuus prosentteina
    enable_afdw: bool = True        # Adaptiivinen FDW (A-FDW)
    ir_window_ms: float = 500.0     # Oikean puolen aikaikkuna (Right)
    ir_window_ms_left: float = 10  # Vasemman puolen aikaikkuna (Left / Pre-ringing)
    mixed_split_freq: float = 300.0 # Mixed Phase -suodattimen jakotaajuus
    trans_width: float = 100.0      # Siirtymäalueen leveys ylärajalla

    # --- 5. RAKENTEELLISET ASETUKSET (HPF, XO, TARGET) ---
    hpf_settings: Optional[Dict] = None  # Ylipäästösuotimen asetukset (enabled, freq, order)
    crossovers: List[Dict] = field(default_factory=list) # Jakosuotimien linearisointi
    house_freqs: Optional[List[float]] = None # Tavoitekäyrän taajuudet
    house_mags: Optional[List[float]] = None  # Tavoitekäyrän voimakkuudet
    
    # --- 6. TASONSOVITUS (LEVELING) ---
    lvl_mode: str = "Auto"          # Tilan valinta (Auto, Manual)
    lvl_algo: str = "Median"        # Algoritmi (Median, Average)
    lvl_manual_db: float = 75.0     # Manuaalinen tavoitetaso
    lvl_min: float = 200.0          # Tasonsovituksen hakualueen alku
    lvl_max: float = 3000.0         # Tasonsovituksen hakualueen loppu
    
    # --- 7. SUOJAUKSET JA OPTIMOINNIT ---
    do_normalize: bool = False       # Normalisointi -1 dBFS tasoon
    exc_prot: bool = False          # Ekskursiosuojaus (basson suojelu)
    exc_freq: float = 40.0          # Suojataajuus
    low_bass_cut_hz: float = 40.0  # Alle tämän taajuuden sallitaan vain leikkaus (ei boostia)
