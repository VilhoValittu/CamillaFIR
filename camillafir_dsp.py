import numpy as np
import scipy.signal
import scipy.fft
import scipy.ndimage
import logging
logger = logging.getLogger("CamillaFIR.dsp")
from models import FilterConfig
#CamillaFIR DSP Engine v1.0.5
#1.0.2 Fix comma mistake at HPF
#1.03 Fix at phase calculation that caused "spikes"
#1.04 All features works at different configurations
#1.05 Multiplier changes

def apply_smart_tdc(freq_axis, target_mags, reflections, rt60_avg, base_strength=0.5):
    adjusted_target = np.copy(target_mags)
    ref_rt60 = rt60_avg if rt60_avg > 0.1 else 0.4
    
    for rev in reflections:
        f_res = rev['freq']
        # FIXED: Changed 'error_ms' to 'gd_error' to match analyze_acoustic_confidence
        error_ms = rev['gd_error'] 
        
        # HERKEMPI KYNNYS: Reagoidaan jo 80% kohdalla keskimääräisestä RT60:stä
        excess_ratio = error_ms / (ref_rt60 * 1000.0 + 1e-12)
        
        if excess_ratio > 0.8: 
            # Dynaaminen kerroin
            dynamic_mult = np.clip(excess_ratio * base_strength, 0.2, 3.0)
            
            # Kapeampi ja kohdistetumpi kaistanleveys (BW)
            bw = f_res / (error_ms / 15.0) 
            dist = np.abs(freq_axis - f_res)
            kernel = np.exp(-0.5 * (dist / bw)**2)
            
            # Voimakkaampi vaimennus tavoitteeseen
            reduction_db = dynamic_mult * 4.0 
            adjusted_target -= (kernel * reduction_db)
            
    return adjusted_target

def apply_hpf_to_mags(freqs, mags, cutoff, order):
    """Soveltaa Butterworth-ylipäästösuodatusta magnitudivasteeseen (dB)."""
    if cutoff <= 0 or order <= 0:
        return mags
    # Butterworth vaste: 1 / sqrt(1 + (fc/f)^(2*order))
    # Muutetaan desibeleiksi: -10 * log10(1 + (fc/f)^(2*order))
    with np.errstate(divide='ignore'):
        attenuation = -10 * np.log10(1 + (cutoff / (freqs + 1e-12))**(2 * order))
    return mags + attenuation

def soft_clip_boost(gain_db, max_boost):
    """Pehmentää korostukset tanh-funktiolla, jotta max_boost ei ylity rajusti."""
    if gain_db <= 0: return gain_db
    return max_boost * np.tanh(gain_db / max_boost)

def calculate_minimum_phase(mags_lin_fft):
    """Laskee minimivaiheen Hilbert-muunnoksella. 1e-10 suojaus estää NaN-virheet."""
    n_fft = (len(mags_lin_fft) - 1) * 2
    ln_mag = np.log(np.maximum(np.abs(mags_lin_fft), 1e-10))
    full_ln_mag = np.concatenate((ln_mag, ln_mag[-2:0:-1]))
    analytic = scipy.signal.hilbert(full_ln_mag)
    min_phase_rad = -np.imag(analytic)
    return min_phase_rad[:len(mags_lin_fft)]

def psychoacoustic_smoothing(freqs, mags, oct_bw=1/3.0):
    """Yhdistää raskaan ja kevyen tasoituksen parhaan visualisoinnin saamiseksi."""
    mags_heavy, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), oct_bw)
    mags_light, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), 1/12.0)
    return np.maximum(mags_heavy, mags_light)

def apply_fdw_smoothing(freqs, phases, cycles):
    """Soveltaa taajuusriippuvaista ikkunointia (FDW) vaiheelle."""
    safe_cycles = max(cycles, 1.0)
    phase_u = np.unwrap(np.deg2rad(phases))
    oct_width = 2.0 / safe_cycles
    dummy_mags = np.zeros_like(freqs)
    _, smoothed_phase_deg = apply_smoothing_std(freqs, dummy_mags, np.rad2deg(phase_u), oct_width)
    return np.deg2rad(smoothed_phase_deg)

def apply_adaptive_fdw(freqs, mags, confidence_mask, base_cycles=15.0, min_cycles=5.0):
    """
    Sovelletaan adaptiivista magnitudin tasoitusta luottamuksen perusteella.
    Korkea luottamus = enemmän syklejä (tarkempi korjaus).
    Matala luottamus = vähemmän syklejä (raskaampi tasoitus).
    """
    # Lasketaan taajuuskohtainen syylimäärä confidence_maskin perusteella
    adaptive_cycles = min_cycles + (confidence_mask * (base_cycles - min_cycles))
    oct_widths = 2.0 / np.maximum(adaptive_cycles, 1.0)
    
    smoothed_mags = np.copy(mags)
    
    # Tehdään tasoitus usealla eri tarkkuudella ja yhdistetään ne maskin avulla
    # Tämä on nopeampi ja vakaampi tapa kuin taajuuskohtainen silmukka
    for bw in [1/3, 1/6, 1/12, 1/24, 1/48]:
        sm, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), bw)
        # Valitaan tälle taajuudelle sopiva tasoitusleveys
        mask = (oct_widths >= bw * 0.7) & (oct_widths < bw * 1.5)
        smoothed_mags[mask] = sm[mask]
    
    return smoothed_mags

def apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0):
    """Vakioitu oktaavitasoitus logaritmisella näytteistyksellä."""
    if octave_fraction <= 0: return mags, phases
    f_min = max(freqs[0], 1.0)
    f_max = freqs[-1]
    
    
    points_per_octave = 384 
    
    num_points = int(np.log2(f_max / f_min) * points_per_octave)
    num_points = max(num_points, 10)
    
    log_freqs = np.geomspace(f_min, f_max, num_points)
    log_mags = np.interp(log_freqs, freqs, mags)
    phase_unwrap = np.unwrap(np.deg2rad(phases))
    log_phases = np.interp(log_freqs, freqs, phase_unwrap)
    
    window_size = int(points_per_octave * octave_fraction)
    window_size = max(window_size, 1) # Varmistetaan vähintään 1
    window = np.ones(window_size) / window_size
    
    pad_len = window_size // 2
    m_padded = np.pad(log_mags, (pad_len, pad_len), mode='edge')
    p_padded = np.pad(log_phases, (pad_len, pad_len), mode='edge')
    
    if pad_len > 0:
        sm_mags = np.convolve(m_padded, window, mode='same')[pad_len:-pad_len]
        sm_phases = np.convolve(p_padded, window, mode='same')[pad_len:-pad_len]
    else:
        sm_mags = np.convolve(m_padded, window, mode='same')
        sm_phases = np.convolve(p_padded, window, mode='same')
        
    return np.interp(freqs, log_freqs, sm_mags), np.rad2deg(np.interp(freqs, log_freqs, sm_phases))

def calculate_theoretical_phase(freq_axis, crossovers, hpf_freq=None, hpf_slope=None):
    """Laskee jakosuotimien ja ylipäästön (HPF) aiheuttaman teoreettisen vaihesiirron."""
    total_phase_rad = np.zeros_like(freq_axis)
    
    # 1. Lisätään HPF:n vaihevaikutus analogisella Butterworth-mallilla
    if hpf_freq and hpf_slope and hpf_freq > 0:
        hpf_order = int(hpf_slope / 6)
        b, a = scipy.signal.butter(hpf_order, 2 * np.pi * hpf_freq, btype='high', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))

    # 2. Lisätään jakosuotimien vaihe
    for xo in crossovers:
        if xo.get('freq') is None: continue
        order = xo.get('order', int(xo.get('slope', 12) / 6))
        b, a = scipy.signal.butter(order, 2 * np.pi * xo['freq'], btype='low', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))
        
    return total_phase_rad

def interpolate_response(input_freqs, input_values, target_freqs):
    """Interpoloi vasteen lineaarisesti kohdetaajuuksille."""
    return np.interp(target_freqs, input_freqs, input_values)

def calculate_group_delay(freqs, phases_deg):
    """Laskee ryhmäviiveen (ms) vaiheen gradientista."""
    phase_rad = np.unwrap(np.deg2rad(phases_deg))
    d_phi_d_f = np.gradient(phase_rad, freqs)
    gd_ms = -d_phi_d_f / (2 * np.pi) * 1000.0
    return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=3)
def combine_mixed_phase(ir_lin, ir_min, fs, split_freq=300):
    """Yhdistää Linear Phase basson ja Minimum Phase diskantin."""
    ntaps = len(ir_lin)
    fir_lp = scipy.signal.firwin(2047, split_freq, fs=fs, pass_zero=True, window='blackman')
    fir_hp = -fir_lp
    fir_hp[1023] += 1.0
    
    idx_lin = np.argmax(np.abs(ir_lin))
    idx_min = np.argmax(np.abs(ir_min))
    shift = idx_lin - idx_min
    ir_min_aligned = np.roll(ir_min, shift)
    
    filt_bass = scipy.signal.fftconvolve(ir_lin, fir_lp, mode='same')
    filt_treble = scipy.signal.fftconvolve(ir_min_aligned, fir_hp, mode='same')
    return filt_bass + filt_treble

def remove_time_of_flight(freq_axis, phase_rad):
    """Etsii ja poistaa linear phase slopen (viiveen) mittauksesta."""
    mask = (freq_axis >= 1000) & (freq_axis <= 10000)
    if not np.any(mask): return phase_rad, 0.0
    poly = np.polyfit(freq_axis[mask], phase_rad[mask], 1)
    return phase_rad - (poly[0] * freq_axis), poly[0]

def analyze_acoustic_confidence(freq_axis, complex_meas, fs):
    """Analysoi akustisen luottamuksen skaalautuvalla resoluutiolla."""
    phase_rad = np.unwrap(np.angle(complex_meas))
    df = np.gradient(freq_axis) + 1e-12
    gd_s = -np.gradient(phase_rad) / (2 * np.pi * df)
    gd_ms = gd_s * 1000.0
    
    # KORJAUS: Skaalataan sigma näytetaajuuden mukaan (perustaso 44100 Hz)
    # Tämä pitää tasoituksen leveyden (Hz) vakiona näytetaajuudesta riippumatta.
    sigma_scaling = fs / 44100.0
    gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=10 * sigma_scaling)
    gd_diff = np.abs(gd_ms - gd_smooth)

    # Luottamusmaski
    threshold_ms = 2.5
    confidence_mask = 1.0 / (1.0 + np.exp(1.5 * (gd_diff - threshold_ms)))
    
    reflection_nodes = []
    valid_idx = np.where(freq_axis > 20)[0] 
    
    if len(valid_idx) > 0:
        # 2. PIIKKIEN ETSINTÄ (Herkkyyden lisäys)
        # Lasketaan korkeusvaatimus 3.0 -> 2.0, jotta pienemmätkin moodit löytyvät
        # Lyhennetään distance 50 -> 30, jotta lähekkäiset resonanssit tunnistetaan
       peaks, props = scipy.signal.find_peaks(gd_diff[valid_idx], height=2.0, distance=100)
    
    raw_nodes = []
    for p in peaks:
        idx = valid_idx[p]
        raw_nodes.append({
            'freq': round(freq_axis[idx], 1), 
            'gd_error': round(gd_diff[idx], 2),
            'dist': round((gd_diff[idx] / 1000.0 * 343.0) / 2.0, 2), 
            'type': "Resonance" if freq_axis[idx] < 200 else "Reflection"
        })
    
    # POIMITAAN VAIN 15 VOIMAKKAINTA (estää pisteytyksen romahtamisen)
    reflection_nodes = sorted(raw_nodes, key=lambda x: x['gd_error'], reverse=True)[:15]
            
    return confidence_mask, reflection_nodes, gd_ms


def find_stable_level_window(freq_axis, magnitudes, target_mags, f_min, f_max, window_size_octaves=1.0, hpf_freq=0.0):
    """
    Etsii alueen, jossa mittaus seuraa tavoitekäyrän muotoa vakaimmin.
    """
    try:
        safe_f_min = max(f_min, hpf_freq * 1.5)
        if safe_f_min >= f_max * 0.8: safe_f_min = f_min

        mask = (freq_axis >= safe_f_min) & (freq_axis <= f_max)
        f_search = freq_axis[mask]
        
        # --- MUUTOS: Tarkastellaan erotusta tavoitteeseen, ei pelkkää tasoa ---
        # Tämä poistaa tavoitekäyrän "kallistuksen" (tilt) vaikutuksen analyysista.
        m_search = (magnitudes - target_mags)[mask]
        
        if len(f_search) < 50: return float(f_min), float(f_max)
            
        best_std = float('inf')
        res_min, res_max = float(safe_f_min), float(f_max)
        
        current_f = safe_f_min
        step = 2**(1/24.0) # Hieman tarkempi askellus (oli 1/12)
        
        while current_f * (2**window_size_octaves) <= f_max:
            w_start, w_end = current_f, current_f * (2**window_size_octaves)
            w_mask = (f_search >= w_start) & (f_search <= w_end)
            if np.any(w_mask):
                current_std = np.std(m_search[w_mask])
                # Painotus: Suositaan matalampia taajuuksia hieman vähemmän aggressiivisesti
                weighted_std = current_std * (1.0 + 0.1 * np.log10(w_start / 1000.0))
                
                if weighted_std < best_std:
                    best_std = weighted_std
                    res_min, res_max = float(w_start), float(w_end)
            current_f *= step
        return res_min, res_max
    except:
        return float(f_min), float(f_max)
    
def calculate_rt60(impulse, fs):
    """Laskee RT60-estimaatin T10-menetelmällä korkean resoluution datasta."""
    try:
        imp = np.array(impulse)
        # Etsitään huippu ja neliöidään
        peak_idx = np.argmax(np.abs(imp))
        decay = np.abs(imp[peak_idx:])**2
        
        # Schroederin integraali
        schroeder = np.flip(np.cumsum(np.flip(decay)))
        # Normalisoidaan ja muutetaan desibeleiksi
        edc = 10 * np.log10(schroeder / (np.max(schroeder) + 1e-12) + 1e-12)
        
        # Etsitään -5dB ja -15dB (T10)
        idx5 = np.where(edc <= -5)[0][0]
        idx15 = np.where(edc <= -15)[0][0]
        
        rt60 = ((idx15 - idx5) / fs) * 6.0
        # Huoneen RT60 on yleensä 0.2s - 1.5s välillä
        return round(rt60, 2) if 0.05 < rt60 < 5.0 else 0.0
    except:
        return 0.0

def get_min_phase_impulse(mags_db, n_fft):
    """Luo minimivaiheisen impulssivasteen voimakkuusvasteesta."""
    # Muunnetaan dB -> lineaarinen ja luodaan symmetrinen spektri
    amp = 10**(mags_db / 20.0)
    # Hilbert-muunnos vaatii logaritmi-amplitudin
    l_amp = np.log(amp + 1e-12)
    # Lasketaan minimivaihe käyttämällä FFT:tä ja Hilbertin periaatetta
    h = scipy.fft.ifft(l_amp)
    n = len(h)
    window = np.zeros(n)
    window[0] = 1
    window[1:n//2] = 2
    window[n//2] = 1
    # Muodostetaan minimivaiheinen vaste
    min_phase = np.exp(scipy.fft.fft(h * window))
    return np.real(scipy.fft.ifft(min_phase))

        




#----------- Filtteri ---------------------


import numpy as np
import scipy.signal
import scipy.fft
import scipy.ndimage
import logging
from models import FilterConfig

logger = logging.getLogger("CamillaFIR.dsp")

#----- FILTTERI

def generate_filter(freqs, meas_mags, raw_phases, cfg: FilterConfig):
    # --- 1. DATAN TASAUS ---
    min_len = min(len(freqs), len(meas_mags), len(raw_phases))
    f_in, m_in, p_in = freqs[:min_len], meas_mags[:min_len], raw_phases[:min_len]

    # --- 2. AKSELIT ---
    n_fft = cfg.num_taps if cfg.num_taps % 2 != 0 else cfg.num_taps + 1
    freq_axis = np.linspace(0, cfg.fs/2.0, n_fft // 2 + 1)
    
    # --- 3. TASOITUS (Skaalautuva resoluutio) ---
    oct_frac = 1.0 / float(cfg.smoothing_level) if cfg.smoothing_level > 0 else 1/12.0
    is_psy = 'Psy' in str(cfg.smoothing_type).lower()
    
    # Magnituditasoitus
    m_smooth, _ = apply_smoothing_std(f_in, m_in, np.zeros_like(m_in), oct_frac * (1.5 if is_psy else 1.0))
    
    # KORJAUS: Dynaaminen vaihetasoitus näytetaajuuden mukaan
    # Korkeilla taajuuksilla (>96kHz) interpolointi luo vaiheeseen "kulmia", 
    # jotka siistitään raskaammalla (1/12 oct) tasoituksella.
    p_smooth_oct = 1/12.0 if cfg.fs > 96000 else 1/24.0
    p_smooth, _ = apply_smoothing_std(f_in, p_in, np.zeros_like(p_in), p_smooth_oct)

    # --- 4. TOF & INTERPOLOINTI ---
    m_interp = np.interp(freq_axis, f_in, m_in)
    p_rad_raw = np.deg2rad(np.interp(freq_axis, f_in, p_in))
    p_rad_interp, delay_slope = remove_time_of_flight(freq_axis, p_rad_raw)

    # --- 5. ANALYYSI (Skaalattu luottamusmaski) ---
    m_anal = np.interp(freq_axis, f_in, m_smooth)
    p_anal_rad = np.deg2rad(np.interp(freq_axis, f_in, p_smooth))
    p_anal_rad, _ = remove_time_of_flight(freq_axis, p_anal_rad)
    complex_anal = 10**(m_anal/20.0) * np.exp(1j * p_anal_rad)
    
    # Confidence mask ja heijastusanalyysi skaalautuvalla sigmalla
    conf_mask, reflections, _ = analyze_acoustic_confidence(freq_axis, complex_anal, cfg.fs)

    # --- 6. RT60 & TARGET ---
    m_rt_lin = np.interp(np.linspace(0, cfg.fs/2, 65537), freq_axis, np.interp(freq_axis, f_in, m_in))
    current_rt60 = calculate_rt60(get_min_phase_impulse(m_rt_lin, 131072), cfg.fs)
    
    target_mags = interpolate_response(cfg.house_freqs, cfg.house_mags, freq_axis)
    if cfg.hpf_settings and cfg.hpf_settings.get('enabled'):
        target_mags = apply_hpf_to_mags(freq_axis, target_mags, cfg.hpf_settings['freq'], cfg.hpf_settings['order'])
    
    if cfg.enable_tdc:
        target_mags = apply_smart_tdc(freq_axis, target_mags, reflections, current_rt60, cfg.tdc_strength/100.0)

    # --- 7. TASONSOVITUS ---
    target_level_db = 0.0
    if 'Manual' in str(cfg.lvl_mode):
        s_min, s_max = float(cfg.lvl_min), float(cfg.lvl_max)
        mask = (freq_axis >= s_min) & (freq_axis <= s_max)
        if np.any(mask): target_level_db = np.mean(m_anal[mask])
        calc_offset_db = target_level_db - cfg.lvl_manual_db
    else:
        h_f = cfg.hpf_settings['freq'] if cfg.hpf_settings else 0
        s_min, s_max = find_stable_level_window(freq_axis, m_anal, target_mags, cfg.lvl_min, cfg.lvl_max, hpf_freq=h_f)
        
        mask = (freq_axis >= s_min) & (freq_axis <= s_max)
        if np.any(mask):
            target_level_db = np.mean(m_anal[mask])
            diffs = m_anal[mask] - target_mags[mask]
            # Käytetään mediaania, se on immuuni yksittäisille piikeille
            calc_offset_db = np.median(diffs)

    # --- 8. KORJAUS ---
    gain_db = np.zeros_like(freq_axis)
    if cfg.enable_mag_correction:
        raw_g = target_mags - (m_anal - calc_offset_db)
        sigma_scaling = cfg.fs / 44100.0
        base_sigma = 60 // (cfg.smoothing_level / 12 if cfg.smoothing_level > 0 else 1)
        sigma = max(2, int(base_sigma * sigma_scaling))
        
        sm_g = scipy.ndimage.gaussian_filter1d(raw_g, sigma=sigma)
        final_g = raw_g - (raw_g - sm_g) * (cfg.reg_strength / 100.0)
        
        mask_c = (freq_axis >= (0 if cfg.hpf_settings else cfg.mag_c_min)) & (freq_axis <= cfg.mag_c_max)
        eff_conf = np.where(freq_axis < 100, np.maximum(conf_mask, 0.6), conf_mask)
        gain_db[mask_c] = [soft_clip_boost(g, cfg.max_boost_db) for g in (final_g * eff_conf)[mask_c]]
        
        f_start = max(cfg.mag_c_max - cfg.trans_width, cfg.mag_c_min)
        
        f_mask = (freq_axis > f_start) & (freq_axis <= cfg.mag_c_max)
        # Varmistetaan jakolasku (ettei jaeta nollalla, jos trans_width on 0)
        fade_len = cfg.mag_c_max - f_start
        if np.any(f_mask) and fade_len > 0: 
            gain_db[f_mask] *= (cfg.mag_c_max - freq_axis[f_mask]) / fade_len
        if cfg.exc_prot:
            # Määritetään siirtymäalue (n. 1/2 oktaavia, kerroin 1.41)
            f_start = cfg.exc_freq
            f_end = cfg.exc_freq * 1.41
            
            # 1. Täysi suojaus f_start alapuolella: Pakotetaan boost nollaan, sallitaan leikkaus
            prot_mask = freq_axis < f_start
            gain_db[prot_mask] = np.minimum(gain_db[prot_mask], 0.0)
            
            # 2. Pehmeä siirtymäalue f_start -> f_end
            # Tällä alueella sallittu boost kasvaa lineaarisesti 0 dB -> max_boost_db
            trans_mask = (freq_axis >= f_start) & (freq_axis <= f_end)
            if np.any(trans_mask):
                # Lasketaan häivytyskerroin (0.0 -> 1.0)
                fade = (freq_axis[trans_mask] - f_start) / (f_end - f_start)
                # Sallittu maksimiboost tällä taajuudella
                allowed_boost = fade * cfg.max_boost_db
                # Rajoitetaan boost, mutta pidetään kaikki vaimennukset (leikkaukset)
                gain_db[trans_mask] = np.minimum(gain_db[trans_mask], allowed_boost)
            
            logger.info(f"Exc Prot: Full protection < {f_start}Hz, Soft fade up to {f_end:.1f}Hz.")

    # --- 9. VAIHEEN GENERONTI ---
    theo_xo = calculate_theoretical_phase(freq_axis, cfg.crossovers, 
                                        cfg.hpf_settings['freq'] if cfg.hpf_settings else None,
                                        (cfg.hpf_settings['order']*6) if cfg.hpf_settings else None)
    
    total_mag = 10**((gain_db + cfg.global_gain_db) / 20.0)
    min_p = calculate_minimum_phase(total_mag)
    
    if 'Min' in cfg.filter_type_str: final_phase = min_p
    elif 'Mixed' in cfg.filter_type_str:
        # Lasketaan dynaaminen siirtymäalue oktaaveina (n. 1.0 oktaavia)
        # f_start = f_split / 1.41, f_end = f_split * 1.41
        f_center = cfg.mixed_split_freq
        
        # Estetään jakolasku nollalla ja varmistetaan järkevä taajuusresoluutio
        safe_freqs = np.maximum(freq_axis, 1.0)
        
        # Logaritminen siirtymämaski (Octave-based transition)
        # Lasketaan kuinka monta oktaavia kukin taajuus on keskipisteestä
        octave_dist = np.log2(safe_freqs / f_center)
        
        # Skaalataan maski välille -0.5 ... 0.5 oktaavia ja limitoidaan 0...1
        # Tämä tekee siirtymästä identtisen kaikilla näytetaajuuksilla
        mask = np.clip((octave_dist + 0.5), 0, 1)
        
        # Smoothstep-interpolaatio (3x^2 - 2x^3) takaa jatkuvuuden
        sm_mask = 3 * mask**2 - 2 * mask**3
        
        # Yhdistetään vaiheet: 
        # Lows: -theo_xo (Linear Phase correction for crossovers)
        # Highs: min_p (Minimum Phase for natural transient response)
        final_phase = (1 - sm_mask) * (-theo_xo) + sm_mask * min_p
        
        logger.info(f"Mixed Phase blend: Transition centered at {f_center}Hz over 1.0 octave.")
    else: final_phase = -theo_xo

    raw_imp = scipy.fft.irfft(total_mag * np.exp(1j * final_phase), n=n_fft)
    
    # Alustava sijoittelu
    if 'Asym' in cfg.filter_type_str:
        shift = min(int(cfg.ir_window_ms_left * cfg.fs / 1000.0), int(n_fft * 0.4))
        impulse = np.roll(raw_imp, shift)
    elif 'Min' in cfg.filter_type_str: impulse = raw_imp
    else: impulse = np.roll(raw_imp, n_fft // 2)

    # --- 10. KORJATTU IKKUNOINTI (PEAK-CENTRIC) ---
    peak_idx = np.argmax(np.abs(impulse))
    n = len(impulse)
    window = np.zeros(n)
    
    # Ikkunan rajojen laskenta (Näytteet)
    s_left = int(cfg.ir_window_ms_left * cfg.fs / 1000.0)
    s_right = int(cfg.ir_window_ms * cfg.fs / 1000.0)
    
    # Logiikka:
    # Asym/Min: Käytä käyttäjän antamia arvoja kirjaimellisesti.
    # Linear/Mixed: Pakota symmetria (ir_window määrää säteen), jotta lineaarinen vaihe säilyy.
    if not ('Asym' in cfg.filter_type_str or 'Min' in cfg.filter_type_str):
        radius = s_right
        s_left = radius
        s_right = radius

    # Rakennetaan ikkuna piikin ympärille
    # 1. Vasen puoli (Nousu nollasta ykköseen)
    if s_left > 0:
        win_rise = np.sin(np.linspace(0, np.pi/2, s_left + 1))[:-1]**2
        start_idx = peak_idx - s_left
        
        # Sijoitus puskuriin (varovasti reunojen kanssa)
        if start_idx >= 0:
            window[start_idx : peak_idx] = win_rise
        else:
            # Jos piikki on liian lähellä alkua, leikataan ikkunan alkupää
            offset = -start_idx
            window[0 : peak_idx] = win_rise[offset:]
            
    # 2. Oikea puoli (Lasku ykkösestä nollaan)
    if s_right > 0:
        win_fall = np.cos(np.linspace(0, np.pi/2, s_right + 1))[1:]**2
        end_idx = peak_idx + 1 + s_right
        
        if end_idx <= n:
            window[peak_idx + 1 : end_idx] = win_fall
        else:
            len_available = n - (peak_idx + 1)
            window[peak_idx + 1 : n] = win_fall[:len_available]
            
    # Piikki on aina 1.0
    if 0 <= peak_idx < n:
        window[peak_idx] = 1.0
        
    # Sovelletaan ikkuna (nollaa kaiken datan ikkunan ulkopuolelta)
    impulse *= window
    impulse -= np.mean(impulse) # DC-poisto
    
    
    # --- 9. CLIP PREVENTION & HEADROOM (UUSI OSIO) ---
    # Lasketaan suodattimen suurin mahdollinen vahvistus (Boost + Global Gain)
    current_peak_gain = np.max(gain_db + cfg.global_gain_db)
    
    # Automaattinen headroom: Jos suodin korostaa, vaimennetaan kokonaistasoa
    # vastaavasti, jotta huippu on tasan 0 dB (tai -0.1 dB varmuuden vuoksi).
    auto_headroom_db = 0.0
    if current_peak_gain > 0:
        auto_headroom_db = -current_peak_gain - 0.1
        logger.info(f"Clip Prevention: Applied {auto_headroom_db:.2f} dB headroom.")

    # Lasketaan lopullinen magnitudivaste sisältäen suojavaran
    final_gain_total = gain_db + cfg.global_gain_db + auto_headroom_db
    total_mag = 10**(final_gain_total / 20.0)
    
    # Jatka vaiheen generointiin kuten ennen, mutta käytä total_mag
    min_p = calculate_minimum_phase(total_mag)
    
    
    # --- 11. STATS & RETURN ---
    max_peak = np.max(np.abs(impulse))
    if cfg.do_normalize and max_peak > 0: impulse *= (0.89 / max_peak)

    stats = {
        'freq_axis': freq_axis.tolist(),
        'target_mags': target_mags.tolist(),
        'measured_mags': (m_anal - calc_offset_db).tolist(),
        'filter_mags': gain_db.tolist(),
        'confidence_mask': conf_mask.tolist(),
        'reflections': reflections,
        'smart_scan_range': [float(s_min), float(s_max)],
        'eff_target_db': float(target_level_db),
        'offset_db': float(calc_offset_db),
        'rt60_val': float(current_rt60),
        'avg_confidence': float(np.mean(conf_mask)*100),
        'delay_samples': float((delay_slope * cfg.fs) / (2 * np.pi)) if 'delay_slope' in locals() else 0.0,
        'peak_before_norm': float(20*np.log10(max_peak + 1e-12)),
        'auto_headroom_db': float(auto_headroom_db),
        'peak_gain_db': float(current_peak_gain),
        'final_max_db': float(np.max(final_gain_total))
    }
    return impulse, stats
