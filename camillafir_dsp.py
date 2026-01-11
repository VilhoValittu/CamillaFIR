import numpy as np
import scipy.signal
import scipy.fft
import scipy.ndimage
#CamillaFIR DSP Engine v1.0.3
#1.0.2 Fix comma mistake at HPF
#1.03 Fix at phase calculation that caused "spikes"


#def apply_temporal_decay_control(freq_axis, target_mags, reflections, strength=0.5):
#    """
#    Luo ajallisen vaimennuksen korjauskäyrän havaittujen resonanssien perusteella.
#    """
#    # Tehdään kopio, jotta emme muuta alkuperäistä tavoitekäyrää pysyvästi
#    adjusted_target = np.copy(target_mags)
    
#    for rev in reflections:
#        f_res = rev['freq']
#        error_ms = rev['error_ms']
#        
#        # TDC keskittyy merkittäviin resonansseihin (soivat pitkään)
#        if error_ms > 100: 
#            # Lasketaan dynaaminen Q-arvo (leveys) virheen pituuden mukaan
#            # Mitä pitempi virhe, sitä kapeampi ja syvempi TDC-isku tarvitaan
#            bw = f_res / (error_ms / 15.0) 
#            
#            # Luodaan Gaussin kello-suodatin kumoamaan ajallinen virhe
#            dist = np.abs(freq_axis - f_res)
#            kernel = np.exp(-0.5 * (dist / bw)**2)
#            
#            # Voimakkuuskerroin (perustuu virheen pituuteen ja käyttäjän asetukseen)
#            # Max vaimennus TDC:llä on yleensä -3..-6dB, ettei vaste muutu oudoksi
#            impact = (error_ms / 2000.0) * strength * 6.0
#            
#            adjusted_target -= (kernel * impact)
#            
#    return adjusted_target

def apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0):
    """Standardi oktaavitasoitus logaritmisella asteikolla."""
    if octave_fraction <= 0: return mags, phases
    f_min, f_max = max(freqs[0], 1.0), freqs[-1]
    log_freqs = np.geomspace(f_min, f_max, int(np.log2(f_max / f_min) * 192))
    log_mags = np.interp(log_freqs, freqs, mags)
    log_phases = np.interp(log_freqs, freqs, np.unwrap(np.deg2rad(phases)))
    w_size = max(1, int(192 * octave_fraction))
    sm_mags = np.convolve(np.pad(log_mags, w_size//2, mode='edge'), np.ones(w_size)/w_size, mode='same')[w_size//2:-w_size//2]
    sm_phases = np.convolve(np.pad(log_phases, w_size//2, mode='edge'), np.ones(w_size)/w_size, mode='same')[w_size//2:-w_size//2]
    return np.interp(freqs, log_freqs, sm_mags), np.rad2deg(np.interp(freqs, log_freqs, sm_phases))

def psychoacoustic_smoothing(freqs, mags):
    """1/3 oktaavin tasoitus kuvaajia varten."""
    return apply_smoothing_std(freqs, mags, np.zeros_like(mags), 1/3.0)[0]


def apply_smart_tdc(freq_axis, target_mags, reflections, rt60_avg, base_strength=0.5):
    adjusted_target = np.copy(target_mags)
    ref_rt60 = rt60_avg if rt60_avg > 0.1 else 0.4
    
    for rev in reflections:
        f_res = rev['freq']
        error_ms = rev['error_ms']
        
        # HERKEMPI KYNNYS: Reagoidaan jo 80% kohdalla keskimääräisestä RT60:stä
        excess_ratio = error_ms / (ref_rt60 * 1000.0 + 1e-12)
        
        if excess_ratio > 0.8: # Laskettu 1.0 -> 0.8 herkyyden lisäämiseksi
            # Dynaaminen kerroin: Voimistetaan korjausta suhteessa resonanssin pituuteen
            dynamic_mult = np.clip(excess_ratio * base_strength, 0.2, 3.0)
            
            # Kapeampi ja kohdistetumpi kaistanleveys (BW)
            # error_ms / 15.0 tekee korjauksesta tarkemman kuin error_ms / 10.0
            bw = f_res / (error_ms / 15.0) 
            dist = np.abs(freq_axis - f_res)
            kernel = np.exp(-0.5 * (dist / bw)**2)
            
            # Voimakkaampi vaimennus tavoitteeseen (max n. 12dB)
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
    """Vakioitu oktavitasoitus logaritmisella näytteistyksellä."""
    f_min = max(freqs[0], 1.0)
    f_max = freqs[-1]
    points_per_octave = 96
    num_points = int(np.log2(f_max / f_min) * points_per_octave)
    if num_points < 10: num_points = 10
    
    log_freqs = np.geomspace(f_min, f_max, num_points)
    log_mags = np.interp(log_freqs, freqs, mags)
    phase_unwrap = np.unwrap(np.deg2rad(phases))
    log_phases = np.interp(log_freqs, freqs, phase_unwrap)
    
    window_size = int(points_per_octave * octave_fraction)
    if window_size < 1: window_size = 1
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
def calculate_theoretical_phase(freq_axis, crossovers):
    """Laskee jakosuotimien aiheuttaman teoreettisen vaihesiirron."""
    total_phase_rad = np.zeros_like(freq_axis)
    for xo in crossovers:
        if xo['freq'] is None: continue
        b, a = scipy.signal.butter(xo['order'], 2 * np.pi * xo['freq'], btype='low', analog=True)
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
    """Analysoi akustisen luottamuksen ja erottelee heijastukset resonansseista herkemmin."""
    phase_rad = np.unwrap(np.angle(complex_meas))
    df = np.gradient(freq_axis) + 1e-12
    gd_s = -np.gradient(phase_rad) / (2 * np.pi * df)
    gd_ms = gd_s * 1000.0
    
    # Tasoitus: sigma=20 on hyvä perusarvo, joka erottaa piikit taustasta
    gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=20)
    gd_diff = np.abs(gd_ms - gd_smooth)

    # 1. CONFIDENCE MASK (Luottamusmaski)
    # Pidetään 2.5ms kynnys, se toimii hyvin magnitudikorjauksen vaimennuksessa
    threshold_ms = 2.5
    confidence_mask = 1.0 / (1.0 + np.exp(1.5 * (gd_diff - threshold_ms)))
    
    reflection_nodes = []
    valid_idx = np.where(freq_axis > 20)[0] 
    
    if len(valid_idx) > 0:
        # 2. PIIKKIEN ETSINTÄ (Herkkyyden lisäys)
        # Lasketaan korkeusvaatimus 3.0 -> 2.0, jotta pienemmätkin moodit löytyvät
        # Lyhennetään distance 50 -> 30, jotta lähekkäiset resonanssit tunnistetaan
        peaks, props = scipy.signal.find_peaks(gd_diff[valid_idx], height=2.0, distance=30)
        
        for p in peaks:
            idx = valid_idx[p]
            f_res = freq_axis[idx]
            error_ms = gd_diff[idx]
            
            # Lasketaan etäisyys heijastukselle (jos se on sellainen)
            dist_m = (error_ms / 1000.0 * 343.0) / 2.0 
            
            # 3. DYNAAMINEN LUOKITTELU (Acoustic Intelligence)
            # Bassotaajuuksilla (< 200 Hz) lähes kaikki piikit ovat resonansseja (huonemoodeja)
            # Korkeammilla taajuuksilla ne ovat heijastuksia pinnoista
            if f_res < 200:
                event_type = "Resonance"
            else:
                event_type = "Reflection" if error_ms < 15.0 else "Resonance"

            reflection_nodes.append({
                'freq': round(f_res, 1), 
                'gd_error': round(error_ms, 2),
                'dist': round(dist_m, 2), 
                'type': event_type
            })
            
    return confidence_mask, reflection_nodes, gd_ms

def generate_filter(freqs, meas_mags, raw_phases, crossovers, 
                    phase_c_min, phase_c_max, mag_c_min, mag_c_max,
                    house_freqs, house_mags, fs, num_taps, fine_phase_limit, 
                    max_boost_db, global_gain_db, hpf_settings, enable_mag_correction,
                    smoothing_type='Psychoacoustic', fdw_cycles=15, is_min_phase=False, filter_type_str='Linear',
                    lvl_mode='Auto', l_match_min=500, l_match_max=2000, lvl_manual_db=75.0, lvl_algo='Average',
                    do_normalize=True, reg_strength=30.0, exc_prot=False, exc_freq=25.0, 
                    ir_window_ms=500.0, mixed_split_freq=300.0, ir_window_ms_left=100.0, 
                    phase_limit=20000.0, enable_afdw=False, enable_tdc=False, tdc_strength=50.0,
                    trans_width=100.0, smoothing_level=12.0):

    # --- 1. DATAN PITUUKSIEN TASAUS ---
    min_len = min(len(freqs), len(meas_mags), len(raw_phases))
    f_in, m_in, p_in = freqs[:min_len], meas_mags[:min_len], raw_phases[:min_len]

    # --- 2. SISÄISET AKSELIT ---
    n_fft = num_taps if num_taps % 2 != 0 else num_taps + 1
    freq_axis = np.linspace(0, fs/2.0, n_fft // 2 + 1)
    actual_phase_limit = phase_limit if phase_limit < 19000 else phase_c_max

    # --- 3. TASOITUS ---
    m_smooth, _ = apply_smoothing_std(f_in, m_in, np.zeros_like(m_in), 1/12.0)
    p_smooth, _ = apply_smoothing_std(f_in, p_in, np.zeros_like(p_in), 1/24.0)

    # --- 4. INTERPOLOINTI JA TOF-POISTO ---
    m_interp = np.interp(freq_axis, f_in, m_in)
    p_rad_raw = np.deg2rad(np.interp(freq_axis, f_in, p_in))
    
    # Poistetaan etäisyysviive (TOF) heti alussa, jotta analyysi on tarkka
    p_rad_interp, delay_slope = remove_time_of_flight(freq_axis, p_rad_raw)
    
    # Tasoitetut versiot analyysia varten
    m_anal_interp = np.interp(freq_axis, f_in, m_smooth)
    # Poistetaan viive myös tasoitetusta vaiheesta
    p_anal_rad_raw = np.deg2rad(np.interp(freq_axis, f_in, p_smooth))
    p_anal_rad, _ = remove_time_of_flight(freq_axis, p_anal_rad_raw)

    # --- 5. CONFIDENCE JA AKUSTISET TAPAHTUMAT ---
    # Nyt p_anal_rad on "puhdas", joten luottamusmaski ja resonanssit lasketaan oikein
    complex_anal = 10**(m_anal_interp/20.0) * np.exp(1j * p_anal_rad)
    conf_mask, reflections, _ = analyze_acoustic_confidence(freq_axis, complex_anal, fs)

    # --- 6. RT60 REKONSTRUOKTIO ---
    n_rt = 131072 
    f_lin_rt = np.linspace(0, fs/2, n_rt//2 + 1)
    m_rt_lin = np.interp(f_lin_rt, freq_axis, m_interp)
    room_impulse_raw = get_min_phase_impulse(m_rt_lin, n_rt)
    current_rt60 = calculate_rt60(room_impulse_raw, fs)

    # --- 7. TAVOITEKÄYRÄ JA OFFSET ---
    target_mags = interpolate_response(house_freqs, house_mags, freq_axis)
    
    if hpf_settings and hpf_settings.get('enabled'):
        target_mags = apply_hpf_to_mags(freq_axis, target_mags, 
                                        hpf_settings['freq'], 
                                        hpf_settings['order'])
    
    if enable_tdc:
        target_mags = apply_smart_tdc(freq_axis, target_mags, reflections, current_rt60, base_strength=tdc_strength/100.0)

# --- A. TASONSOVITUS (KORJATTU) ---
    mask_lvl = (freq_axis >= l_match_min) & (freq_axis <= l_match_max)

    if 'Manual' in str(lvl_mode):
        calc_offset_db = np.mean(m_anal_interp[mask_lvl]) - lvl_manual_db
    else:
        # Lasketaan kuinka paljon mittaus on tavoitetta kovempi
        diffs = m_anal_interp[mask_lvl] - target_mags[mask_lvl]
        calc_offset_db = np.median(diffs) if 'Median' in str(lvl_algo) else np.mean(diffs)

# --- 8. MAGNITUDIKORJAUS ---
    gain_final_db = np.zeros_like(freq_axis)
    if enable_mag_correction:
        mask_c = (freq_axis >= 15.0) & (freq_axis <= mag_c_max)
        
        # KORJAUS: Aligned measurement = m_interp - calc_offset_db
        # Tällöin mittaus siirtyy tavoitekäyrän päälle.
        raw_gain = target_mags - (m_interp - calc_offset_db)
        
        smooth_gain = scipy.ndimage.gaussian_filter1d(raw_gain, sigma=15)
        final_g = raw_gain - (raw_gain - smooth_gain) * (reg_strength / 3000.0)
        
        effective_conf = np.where(freq_axis < 100, np.maximum(conf_mask, 0.6), conf_mask)
        gain_final_db[mask_c] = [soft_clip_boost(g, max_boost_db) for g in (final_g * effective_conf)[mask_c]]
        if exc_prot: gain_final_db[freq_axis < exc_freq] = np.minimum(gain_final_db[freq_axis < exc_freq], 0.0)

# --- 9. VAIHEENKORJAUS (Vakaa 2058-logiikka) ---
    theoretical_xo = calculate_theoretical_phase(freq_axis, crossovers)
    
    # Alustetaan final_phase kuten ehjässä 2058-tiedostossa
    final_phase = -theoretical_xo 

    if not is_min_phase:
        # Lasketaan nämä vain, jotta ne ovat olemassa (vältetään not accessed)
        meas_min_p = calculate_minimum_phase(10**(m_interp/20.0))
        ex_p_deg = np.rad2deg(np.unwrap(p_rad_interp - meas_min_p))
        
        # TÄRKEÄ VALINTA: 
        # Jos haluat 2058:n vakauden, pidä alla oleva rivi KOMMENTOITUNA.
        # Jos haluat kokeilla huonekorjausta, poista # merkit.
        
        # p_damping = conf_mask * np.exp(-np.maximum(0, freq_axis - actual_phase_limit) / 500.0)
        # final_phase = (-np.deg2rad(apply_fdw_smoothing(freq_axis, ex_p_deg, fdw_cycles)) * p_damping) - theoretical_xo
    

# --- 10. GENEROINTI (KORJATTU) ---
    
    # KORJAUS: Koska gain_final_db on jo laskettu suhteessa kohdistettuun vasteeseen,
    # emme enää vähennä offsetia tässä. Lisäämme vain globaalin gainin.
    # Tämä palauttaa suodattimen tason normaaliksi (-3...-6 dB paikkeille).
    total_mag = 10**((gain_final_db + global_gain_db) / 20.0)
    
    if 'Min' in filter_type_str:
        min_p_spec = calculate_minimum_phase(total_mag)
        h_complex = total_mag * np.exp(1j * min_p_spec)
        impulse = scipy.fft.irfft(h_complex, n=n_fft)
    elif 'Mixed' in filter_type_str:
        h_lin = total_mag * np.exp(1j * final_phase)
        ir_lin = np.roll(scipy.fft.irfft(h_lin, n=n_fft), n_fft // 2)
        h_min = total_mag * np.exp(1j * calculate_minimum_phase(total_mag))
        ir_min = scipy.fft.irfft(h_min, n=n_fft)
        impulse = combine_mixed_phase(ir_lin, ir_min, fs, split_freq=mixed_split_freq)
    else: 
        h_complex = total_mag * np.exp(1j * final_phase)
        shift = int(n_fft * 0.15) if 'Asym' in filter_type_str else n_fft // 2
        impulse = np.roll(scipy.fft.irfft(h_complex, n=n_fft), shift)

   
# --- 11. IKKUNOINTI ---
    if 'Asym' in filter_type_str or 'Min' in filter_type_str:
        # Tukey jättää 15% kohdalla olevan piikin ehjäksi (alpha=0.2)
        window = scipy.signal.windows.tukey(len(impulse), alpha=0.2)
        impulse *= window
    else:
        # Symmetrinen Hann sopii Linear Phaselle
        impulse *= scipy.signal.windows.hann(len(impulse))

    # --- 12. NORMALISOINTI JA STATS ---
    max_peak = np.max(np.abs(impulse))
    if do_normalize and max_peak > 0.0: impulse *= (0.89 / max_peak)

    stats = {
        'freq_axis': freq_axis.tolist(),
        'target_mags': target_mags.tolist(),
        
        # TÄRKEÄ: Lähetetään kohdistettu mittaus (m_interp - calc_offset_db).
        # Tämä saa viivat kohtaamaan Dashboardilla täsmälleen.
        'measured_mags': (m_interp - calc_offset_db).tolist(),
        
        'filter_mags': gain_final_db.tolist(),
        'confidence': conf_mask.tolist(),
        ##'eff_target_db': float(avg_t), # TÄMÄ ON KRIITTINEN VIIVOJEN KOHTAAMISELLE
        # Lisätään tasonsovituksen rajat kuvaajaa varten ("Step")
        'match_range': [float(l_match_min), float(l_match_max)],
        'offset_db': float(calc_offset_db),
        'impulse': impulse,
        'room_impulse': room_impulse_raw,
        'confidence_mask': conf_mask,
        'offset_db': calc_offset_db,
        'eff_target_db': np.mean(target_mags[mask_lvl]),
        'target_mags': target_mags,
        'freq_axis': freq_axis,
        'peak_before_norm': 20 * np.log10(max_peak + 1e-12),
        'avg_confidence': np.mean(conf_mask) * 100.0,
        'reflections': reflections,
        'rt60_val': current_rt60,
        'delay_samples': (delay_slope * fs) / (2 * np.pi), # Nyt delay_slope on käytössä!
        'rt60': float(current_rt60),
        'delay_samples': float((delay_slope * fs) / (2 * np.pi))
    }
    return impulse, stats

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
