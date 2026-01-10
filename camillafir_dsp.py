import numpy as np
import scipy.signal
import scipy.fft
import scipy.ndimage
#CamillaFIR DSP Engine v1.0
# --- v2.6.2 Stable: PERUSTYÖKALUT ---
def apply_temporal_decay_control(freq_axis, target_mags, reflections, strength=0.5):
    """
    Luo ajallisen vaimennuksen korjauskäyrän havaittujen resonanssien perusteella.
    """
    # Tehdään kopio, jotta emme muuta alkuperäistä tavoitekäyrää pysyvästi
    adjusted_target = np.copy(target_mags)
    
    for rev in reflections:
        f_res = rev['freq']
        error_ms = rev['error_ms']
        
        # TDC keskittyy merkittäviin resonansseihin (soivat pitkään)
        if error_ms > 100: 
            # Lasketaan dynaaminen Q-arvo (leveys) virheen pituuden mukaan
            # Mitä pitempi virhe, sitä kapeampi ja syvempi TDC-isku tarvitaan
            bw = f_res / (error_ms / 15.0) 
            
            # Luodaan Gaussin kello-suodatin kumoamaan ajallinen virhe
            dist = np.abs(freq_axis - f_res)
            kernel = np.exp(-0.5 * (dist / bw)**2)
            
            # Voimakkuuskerroin (perustuu virheen pituuteen ja käyttäjän asetukseen)
            # Max vaimennus TDC:llä on yleensä -3..-6dB, ettei vaste muutu oudoksi
            impact = (error_ms / 2000.0) * strength * 6.0
            
            adjusted_target -= (kernel * impact)
            
    return adjusted_target

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
    """Analysoi akustisen luottamuksen ja erottelee heijastukset resonansseista."""
    phase_rad = np.unwrap(np.angle(complex_meas))
    df = np.gradient(freq_axis) + 1e-12
    gd_s = -np.gradient(phase_rad) / (2 * np.pi * df)
    gd_ms = gd_s * 1000.0
    gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=20)
    gd_diff = np.abs(gd_ms - gd_smooth)

    # Korjattu herkempi kynnys heijastusten havaitsemiseen (2.5ms)
    threshold_ms = 2.5
    confidence_mask = 1.0 / (1.0 + np.exp(1.5 * (gd_diff - threshold_ms)))
    reflection_nodes = []
    valid_idx = np.where(freq_axis > 20)[0] 
    
    if len(valid_idx) > 0:
        # Pienempi korkeusvaatimus (3.0) löytää huonemoodit paremmin
        peaks, _ = scipy.signal.find_peaks(gd_diff[valid_idx], height=3.0, distance=50)
        for p in peaks:
            idx = valid_idx[p]
            dist_m = (gd_diff[idx] / 1000.0 * 343.0) / 2.0 
            reflection_nodes.append({
                'freq': freq_axis[idx], 'gd_error': gd_diff[idx],
                'dist': dist_m, 'type': "Reflection" if dist_m < 15.0 else "Resonance"
            })
    return confidence_mask, reflection_nodes, gd_ms

def generate_filter(freqs, raw_mags, raw_phases, crossovers, 
                    phase_c_min, phase_c_max, mag_c_min, mag_c_max,
                    house_freqs, house_mags, fs, num_taps, fine_phase_limit, 
                    max_boost_db, global_gain_db, hpf_settings, enable_mag_correction,
                    smoothing_type='Psychoacoustic', fdw_cycles=15, is_min_phase=False, filter_type_str='Linear',
                    lvl_mode='Auto', l_match_min=500, l_match_max=2000, lvl_manual_db=75.0, lvl_algo='Average',
                    do_normalize=True, reg_strength=30.0, exc_prot=False, exc_freq=25.0, 
                    ir_window_ms=500.0, mixed_split_freq=300.0, ir_window_ms_left=100.0, 
                    phase_limit=20000.0, enable_afdw=False, enable_tdc=False, tdc_strength=50.0):
    
    n_fft = num_taps if num_taps % 2 != 0 else num_taps + 1
    freq_axis = np.linspace(0, fs/2.0, n_fft // 2 + 1)
    
    # 1. Tasoitus (Luodaan perusvaste analyysia varten)
    if 'Psy' in str(smoothing_type):
        smoothed_mags = psychoacoustic_smoothing(freqs, raw_mags)
    else:
        smoothed_mags, _ = apply_smoothing_std(freqs, raw_mags, raw_phases, 1/48.0)
    meas_mags = interpolate_response(freqs, smoothed_mags, freq_axis)
    
    # 2. Akustinen analyysi ja luottamus
    meas_p_rad_raw = np.unwrap(np.deg2rad(interpolate_response(freqs, raw_phases, freq_axis)))
    complex_meas = 10**(meas_mags/20.0) * np.exp(1j * meas_p_rad_raw)
    conf_mask, reflections, gd_raw = analyze_acoustic_confidence(freq_axis, complex_meas, fs)

    # --- VAIHE 2.5: ADAPTIIVINEN FDW (A-FDW) ---
    if enable_afdw:
        meas_mags = apply_adaptive_fdw(freq_axis, meas_mags, conf_mask, 
                                       base_cycles=fdw_cycles, min_cycles=5.0)

    # --- VAIHE 3: TAVOITEKÄYRÄ JA TDC ---
    # KORJAUS: Siirretty tavoitekäyrän luonti tänne, jotta TDC:llä on mitä muokata
    hc_interp = interpolate_response(house_freqs, house_mags, freq_axis)
    target_mags = np.copy(hc_interp)

    # --- VAIHE 2.6: TEMPORAL DECAY CONTROL (TDC) ---
    if enable_tdc:
        # TDC muokkaa tavoitevastetta "syömällä" energiaa resonanssikohdista
        target_mags = apply_temporal_decay_control(
            freq_axis, 
            target_mags, 
            reflections, 
            strength=tdc_strength / 100.0
        )

    # 4. Offsetin ja Gainin laskenta
    mask_lvl = (freq_axis >= l_match_min) & (freq_axis <= l_match_max)
    if 'Manual' in str(lvl_mode):
        calc_offset_db = lvl_manual_db - np.mean(target_mags[mask_lvl])
    else:
        diffs = target_mags[mask_lvl] - meas_mags[mask_lvl]
        calc_offset_db = np.median(diffs) if 'Median' in str(lvl_algo) else np.mean(diffs)
    
    # ... jatkuu kuten ennen ...

    target_mags = hc_interp 
    meas_mags_shifted = meas_mags + calc_offset_db

    # --- UUSI VAIHE 2.6: TEMPORAL DECAY CONTROL (TDC) ---
    if enable_tdc:
        # TDC muokkaa tavoitevastetta "syömällä" energiaa sieltä, missä huone soi liian pitkään
        target_mags = apply_temporal_decay_control(
            freq_axis, 
            target_mags, 
            reflections, 
            strength=tdc_strength / 100.0
        )
    # -------------------------------------------------------

    # 3. Tavoitekäyrä ja Offset
    hc_interp = interpolate_response(house_freqs, house_mags, freq_axis)
    mask_lvl = (freq_axis >= l_match_min) & (freq_axis <= l_match_max)
    # ... jatkuu kuten ennen ...
    if 'Manual' in str(lvl_mode):
        calc_offset_db = lvl_manual_db - np.mean(hc_interp[mask_lvl])
    else:
        diffs = hc_interp[mask_lvl] - meas_mags[mask_lvl]
        calc_offset_db = np.median(diffs) if 'Median' in str(lvl_algo) else np.mean(diffs)

    target_mags = hc_interp 
    meas_mags_shifted = meas_mags + calc_offset_db
    # 4. Gain ja Beta-ohjaus
    raw_gain_db = target_mags - meas_mags_shifted
    smooth_gain_db = scipy.ndimage.gaussian_filter1d(raw_gain_db, sigma=50)
    beta_mask = 5.0 + (1.0 - conf_mask) * 7.0
    
    gain_final_db = np.zeros_like(freq_axis)
    if enable_mag_correction:
        mask_corr = (freq_axis >= mag_c_min) & (freq_axis <= mag_c_max)
        req_g = raw_gain_db.copy()
        req_g -= (req_g - smooth_gain_db) * (reg_strength / 100.0)
        gain_final_db[mask_corr] = [soft_clip_boost(g, max_boost_db) for g in req_g[mask_corr]]
        if exc_prot: gain_final_db[freq_axis < exc_freq] = np.minimum(gain_final_db[freq_axis < exc_freq], 0.0)

    total_mag_lin = 10**((gain_final_db + global_gain_db)/20.0)
    
    # 5. Vaiheenkorjaus
    hpf_complex = np.ones_like(freq_axis, dtype=complex)
    if hpf_settings and hpf_settings['enabled']:
        b, a = scipy.signal.butter(hpf_settings['order'], 2 * np.pi * hpf_settings['freq'], btype='high', analog=True)
        _, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        hpf_complex = h; hpf_complex[0] = 0.0

    filt_min_p = calculate_minimum_phase(total_mag_lin * np.abs(hpf_complex))
    meas_p_no_tof, _ = remove_time_of_flight(freq_axis, meas_p_rad_raw)
    meas_min_p = calculate_minimum_phase(10**(meas_mags/20.0))
    ex_p_rad = apply_fdw_smoothing(freq_axis, np.rad2deg(meas_p_no_tof - meas_min_p), fdw_cycles)
    
    phase_damping = conf_mask * np.exp(-np.maximum(0, freq_axis - phase_c_max) / 1000.0)
    phase_corr_rad = -ex_p_rad * phase_damping
    theoretical_xo = calculate_theoretical_phase(freq_axis, crossovers)
    limit_rad = np.deg2rad(fine_phase_limit)
    final_phase = np.zeros_like(freq_axis)
    mask_p_range = (freq_axis >= phase_c_min) & (freq_axis <= phase_c_max)
    final_phase[mask_p_range] = np.clip(phase_corr_rad[mask_p_range] - theoretical_xo[mask_p_range], -limit_rad, limit_rad)

    if 'Min' in filter_type_str:
        h_complex = total_mag_lin * np.exp(1j * filt_min_p) * hpf_complex
        impulse = scipy.fft.irfft(h_complex, n=n_fft)
    elif 'Mixed' in filter_type_str:
        ir_min = scipy.fft.irfft(total_mag_lin * np.exp(1j * filt_min_p) * hpf_complex, n=n_fft)
        ir_lin = np.roll(scipy.fft.irfft(total_mag_lin * np.exp(1j * final_phase) * hpf_complex, n=n_fft), n_fft // 2)
        impulse = combine_mixed_phase(ir_lin, ir_min, fs, split_freq=mixed_split_freq)
    else:
        h_complex = total_mag_lin * np.exp(1j * final_phase) * hpf_complex
        impulse = np.roll(scipy.fft.irfft(h_complex, n=n_fft), n_fft // 2)

    # 5. Ikkunointi ja Normalisointi
    # Lasketaan dynaaminen Beta luottamuksen mukaan
    avg_beta = np.clip(np.mean(beta_mask[(freq_axis > 20) & (freq_axis < 1000)]), 5, 12)
    
    # KORJAUS: Valitaan ikkunointitapa suotimen tyypin mukaan
    if 'Min' in filter_type_str:
        # Minimum Phase: Ei saa vaimentaa alkua! Ikkunoidaan vain viimeinen 10%
        window = np.ones(len(impulse))
        fade_len = int(len(impulse) * 0.1)
        fade_out = np.hanning(fade_len * 2)[fade_len:]
        window[-fade_len:] = fade_out
        impulse *= window
    else:
        # Linear ja Mixed Phase: Symmetrinen Kaiser on optimaalinen
        impulse *= scipy.signal.windows.kaiser(len(impulse), beta=avg_beta)
    # --- 6. IMPULSSIN GENEROINTI JA IKKUNOINTI (v2.6.3) ---
    if 'Asymmetric' in filter_type_str:
        # Lasketaan huipun sijoitus: Left Window (oletus 100ms)
        peak_idx = int((fs * ir_window_ms_left) / 1000)
        h_complex = total_mag_lin * np.exp(1j * final_phase) * hpf_complex
        impulse_raw = scipy.fft.irfft(h_complex, n=n_fft)
        # Siirretään huippu lähelle alkua viiveen minimoimiseksi
        impulse = np.roll(impulse_raw, peak_idx)
        
        # Asymmetrinen ikkuna: 20ms fade-in, ir_window_ms fade-out
        window = np.ones(n_fft)
        l_fade = min(peak_idx, int((fs * 20) / 1000)) 
        window[:l_fade] = np.hanning(l_fade * 2)[:l_fade]
        r_fade = int((fs * ir_window_ms) / 1000)
        if r_fade > (n_fft - peak_idx): r_fade = n_fft - peak_idx
        window[-r_fade:] = np.hanning(r_fade * 2)[r_fade:]
        impulse *= window
    elif 'Min' in filter_type_str:
        h_complex = total_mag_lin * np.exp(1j * filt_min_p) * hpf_complex
        impulse = scipy.fft.irfft(h_complex, n=n_fft)
        window = np.ones(len(impulse))
        fade_len = int(len(impulse) * 0.1)
        window[-fade_len:] = np.hanning(fade_len * 2)[fade_len:]
        impulse *= window
    elif 'Mixed' in filter_type_str:
        ir_min = scipy.fft.irfft(total_mag_lin * np.exp(1j * filt_min_p) * hpf_complex, n=n_fft)
        ir_lin = np.roll(scipy.fft.irfft(total_mag_lin * np.exp(1j * final_phase) * hpf_complex, n=n_fft), n_fft // 2)
        impulse = combine_mixed_phase(ir_lin, ir_min, fs, split_freq=mixed_split_freq)
        impulse *= scipy.signal.windows.kaiser(len(impulse), beta=avg_beta)
    else: # Standard Linear
        h_complex = total_mag_lin * np.exp(1j * final_phase) * hpf_complex
        impulse = np.roll(scipy.fft.irfft(h_complex, n=n_fft), n_fft // 2)
        impulse *= scipy.signal.windows.kaiser(len(impulse), beta=avg_beta)
    
    max_peak = np.max(np.abs(impulse))
    if do_normalize and max_peak > 0.891: 
        impulse *= (0.891 / max_peak)
    # Korjatut tilastot: mukana gd_min/max ja confidence_mask
    w, h_filt = scipy.signal.freqz(impulse, 1, worN=2048, fs=fs)
    gd_final = calculate_group_delay(w, np.rad2deg(np.unwrap(np.angle(h_filt))))
    
    stats = {
        'offset_db': calc_offset_db, 'eff_target_db': np.mean(target_mags[mask_lvl]), 
        'target_mags': target_mags, 'freq_axis': freq_axis,
        'peak_before_norm': 20 * np.log10(max_peak + 1e-12),
        'normalized': max_peak > 0.891, 
        'gd_min': np.min(gd_final), 'gd_max': np.max(gd_final),
        'confidence_mask': conf_mask, # Tärkeä Dashboardille
        'avg_confidence': np.mean(conf_mask) * 100.0,
        'reflections': reflections # Tärkeä Summarylle
    }
    return impulse, stats['gd_min'], stats['gd_max'], stats
