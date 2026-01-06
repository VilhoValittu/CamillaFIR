import numpy as np
import scipy.signal
import scipy.fft
import scipy.ndimage

def soft_clip_boost(gain_db, max_boost):
    if gain_db <= 0: return gain_db
    # Soft knee limiting for boosts
    return max_boost * np.tanh(gain_db / max_boost)

def clean_measurement_ir(freqs, mags, phases, window_ms=500.0, fs=48000):
    n_fft = 131072
    freq_axis_lin = np.linspace(0, fs/2, n_fft//2 + 1)
    
    # Interpolate to linear grid
    m_lin = np.interp(freq_axis_lin, freqs, mags)
    p_lin = np.interp(freq_axis_lin, freqs, phases)
    
    # Create complex spectrum
    complex_spec = 10**(m_lin/20.0) * np.exp(1j * np.deg2rad(p_lin))
    
    # IFFT to time domain
    ir = scipy.fft.irfft(complex_spec, n=n_fft)
    
    # Center the impulse
    peak_idx = np.argmax(np.abs(ir))
    shift_amount = (n_fft // 2) - peak_idx
    ir_centered = np.roll(ir, shift_amount)
    
    # Windowing
    center_idx = n_fft // 2
    pre_window_samples = int(0.010 * fs) 
    post_window_samples = int((window_ms/1000.0) * fs)
    
    start_idx = max(0, center_idx - pre_window_samples)
    end_idx = min(n_fft, center_idx + post_window_samples)
    
    window = np.zeros_like(ir)
    window[start_idx:end_idx] = 1.0
    
    # Fade edges
    fade_len = int(0.005 * fs)
    hann_fade = scipy.signal.windows.hann(2*fade_len)
    
    if start_idx > 0 and start_idx + fade_len < end_idx:
        window[start_idx:start_idx+fade_len] = hann_fade[:fade_len]
    
    if end_idx < len(ir) and end_idx - fade_len > start_idx:
        window[end_idx-fade_len:end_idx] = hann_fade[fade_len:]
        
    ir_windowed = ir_centered * window
    
    # Back to Frequency Domain
    spec_cleaned = scipy.fft.rfft(ir_windowed)
    mags_out = 20 * np.log10(np.abs(spec_cleaned) + 1e-12)
    phases_out = np.rad2deg(np.angle(spec_cleaned))
    
    return freq_axis_lin, mags_out, phases_out

def calculate_minimum_phase(mags_lin_fft):
    # Hilbert transform based minimum phase calculation
    n_fft = (len(mags_lin_fft) - 1) * 2
    # Ensure no zeros for log
    ln_mag = np.log(np.maximum(np.abs(mags_lin_fft), 1e-10))
    # Create symmetric spectrum for Hilbert
    full_ln_mag = np.concatenate((ln_mag, ln_mag[-2:0:-1]))
    analytic = scipy.signal.hilbert(full_ln_mag)
    min_phase_rad = -np.imag(analytic)
    return min_phase_rad[:len(mags_lin_fft)]

def psychoacoustic_smoothing(freqs, mags, oct_bw=1/3.0):
    # Simulates human hearing integration
    mags_heavy, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), oct_bw)
    mags_light, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), 1/12.0)
    # Peak hold smoothing approach
    return np.maximum(mags_heavy, mags_light)

def apply_fdw_smoothing(freqs, phases, cycles):
    # Frequency Dependent Windowing for Phase
    safe_cycles = max(cycles, 1.0)
    phase_u = np.unwrap(np.deg2rad(phases))
    # Variable smoothing width
    oct_width = 2.0 / safe_cycles
    dummy_mags = np.zeros_like(freqs)
    # Use magnitude smoothing logic for phase curve
    _, smoothed_phase_deg = apply_smoothing_std(freqs, dummy_mags, np.rad2deg(phase_u), oct_width)
    return np.deg2rad(smoothed_phase_deg)

def apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0):
    # Standard Fractional Octave Smoothing
    f_min = max(freqs[0], 1.0)
    f_max = freqs[-1]
    points_per_octave = 96
    num_points = int(np.log2(f_max / f_min) * points_per_octave)
    if num_points < 10: num_points = 10
    
    # Logarithmic grid for convolution
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
    total_phase_rad = np.zeros_like(freq_axis)
    for xo in crossovers:
        if xo['freq'] is None: continue
        # Linkwitz-Riley / Butterworth analog phase response
        b, a = scipy.signal.butter(xo['order'], 2 * np.pi * xo['freq'], btype='low', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))
    return total_phase_rad

def interpolate_response(input_freqs, input_values, target_freqs):
    return np.interp(target_freqs, input_freqs, input_values)

def combine_mixed_phase(ir_lin, ir_min, fs, split_freq=300):
    # Combine Linear (Low) and Minimum (High) Phase IRs
    sos_lp = scipy.signal.butter(4, split_freq, fs=fs, btype='low', output='sos')
    sos_hp = scipy.signal.butter(4, split_freq, fs=fs, btype='high', output='sos')
    
    # Align peaks
    idx_lin = np.argmax(np.abs(ir_lin))
    idx_min = np.argmax(np.abs(ir_min))
    ir_min_shifted = np.roll(ir_min, idx_lin - idx_min)
    
    # Filter and sum
    return scipy.signal.sosfilt(sos_lp, ir_lin) + scipy.signal.sosfilt(sos_hp, ir_min_shifted)

def calculate_group_delay(freqs, phases_deg):
    phase_rad = np.unwrap(np.deg2rad(phases_deg))
    d_phi_d_f = np.gradient(phase_rad, freqs)
    gd_ms = -d_phi_d_f / (2 * np.pi) * 1000.0
    return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=3)

def remove_time_of_flight(freq_axis, phase_rad):
    # Estimate slope (time delay) from stable region 1kHz - 10kHz
    mask = (freq_axis >= 1000) & (freq_axis <= 10000)
    if not np.any(mask):
        return phase_rad, 0.0
        
    # Fit linear model: phase = slope * freq + intercept
    # Slope is related to time delay: delay = -slope / (2*pi)
    poly = np.polyfit(freq_axis[mask], phase_rad[mask], 1)
    slope = poly[0]
    
    # We only want to remove the slope component (TOF), not the intercept
    correction = slope * freq_axis
    return phase_rad - correction, slope

def generate_filter(freqs, raw_mags, raw_phases, crossovers, 
                    phase_c_min, phase_c_max, mag_c_min, mag_c_max,
                    house_freqs, house_mags, fs, num_taps, fine_phase_limit, 
                    max_boost_db, global_gain_db, hpf_settings, enable_mag_correction,
                    smoothing_type='Standard', fdw_cycles=15, is_min_phase=False, filter_type_str='Linear',
                    lvl_mode='Auto', l_match_min=500, l_match_max=2000, lvl_manual_db=75.0, lvl_algo='Average',
                    do_normalize=True, reg_strength=0.0, exc_prot=False, exc_freq=25.0, ir_window_ms=500.0,
                    mixed_split_freq=300.0):
    
    # --- DSP INIT ---
    n_fft = num_taps if num_taps % 2 != 0 else num_taps + 1
    nyquist = fs / 2.0
    freq_axis = np.linspace(0, nyquist, n_fft // 2 + 1)
    
    # --- SMOOTHING ---
    if 'Psy' in str(smoothing_type):
        smoothed_mags = psychoacoustic_smoothing(freqs, raw_mags)
    else:
        smoothed_mags, _ = apply_smoothing_std(freqs, raw_mags, raw_phases, 1/48.0)
    
    meas_mags = interpolate_response(freqs, smoothed_mags, freq_axis)
    target_mags = np.zeros_like(freq_axis)
    calc_offset_db = 0.0 
    
    # --- TARGET CURVE & LEVEL MATCH ---
    if house_freqs is not None:
        hc_interp = interpolate_response(house_freqs, house_mags, freq_axis)
        a_start = max(l_match_min, 10.0)
        a_end = l_match_max
        mask_align = (freq_axis >= a_start) & (freq_axis <= a_end)
        
        if 'Manual' in str(lvl_mode):
            if np.any(mask_align):
                avg_hc = np.mean(hc_interp[mask_align])
                shift = lvl_manual_db - avg_hc
                target_mags = hc_interp + shift
                calc_offset_db = shift 
            else:
                target_mags = hc_interp + (lvl_manual_db - np.mean(hc_interp))
        else:
            mags_1_1, _ = apply_smoothing_std(freqs, raw_mags, raw_phases, 1.0)
            meas_mags_1_1 = interpolate_response(freqs, mags_1_1, freq_axis)
            if np.any(mask_align):
                diffs = meas_mags_1_1[mask_align] - hc_interp[mask_align]
                calc_offset_db = np.median(diffs) if 'Median' in str(lvl_algo) else np.mean(diffs)
                target_mags = hc_interp + calc_offset_db
            else:
                target_mags = hc_interp + calc_offset_db
    
    eff_target_db = np.mean(target_mags)
    
    # --- HPF PREPARATION ---
    hpf_complex = np.ones_like(freq_axis, dtype=complex)
    if hpf_settings and hpf_settings['enabled']:
        b, a = scipy.signal.butter(hpf_settings['order'], 2 * np.pi * hpf_settings['freq'], btype='high', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        hpf_complex = h
        hpf_complex[0] = 0.0

    # --- GAIN CALCULATION ---
    gain_linear = np.ones_like(freq_axis)
    raw_gain_db = target_mags - meas_mags
    
    # --- FREQUENCY DEPENDENT REGULARIZATION ---
    base_reg = reg_strength / 100.0
    reg_curve = np.ones_like(freq_axis) * base_reg 
    
    bass_zone = freq_axis < 200
    mid_trans = (freq_axis >= 200) & (freq_axis < 2000)
    treble_zone = freq_axis >= 2000
    
    reg_curve[bass_zone] = base_reg * 0.5 
    if np.any(mid_trans):
        ramp = np.linspace(0.5, 1.0, np.sum(mid_trans))
        reg_curve[mid_trans] = base_reg * ramp
    if np.any(treble_zone):
        ramp_high = np.linspace(1.0, 2.0, np.sum(treble_zone))
        high_reg = base_reg * ramp_high
        reg_curve[treble_zone] = np.clip(high_reg, 0.0, 0.95)

    smooth_gain_db = scipy.ndimage.gaussian_filter1d(raw_gain_db, sigma=50)
    
    for i, f in enumerate(freq_axis):
        if f > 0:
            g_db = 0.0
            if enable_mag_correction and (mag_c_min <= f <= mag_c_max):
                req_g = raw_gain_db[i]
                if req_g > 0:
                    diff = req_g - smooth_gain_db[i] 
                    if diff > 0:
                        req_g = req_g - (diff * reg_curve[i])
                if exc_prot and f < exc_freq: req_g = min(req_g, 0.0)
                g_db = soft_clip_boost(req_g, min(max_boost_db, 8.0))
            
            g_db += global_gain_db
            gain_linear[i] = 10.0 ** (g_db / 20.0)

    # --- PHASE CORRECTION (TOF AWARE) ---
    total_mag_response = gain_linear * np.abs(hpf_complex)
    filt_min_phase_rad = calculate_minimum_phase(total_mag_response)
    
    # 1. Get Measured Phase
    meas_phase_rad_raw = np.deg2rad(interpolate_response(freqs, raw_phases, freq_axis))
    meas_phase_rad_unwrapped = np.unwrap(meas_phase_rad_raw)
    
    # 2. REMOVE TIME OF FLIGHT (Delay) from Measurement
    # This aligns the measurement mathematically to t=0 before calculating excess phase
    meas_phase_no_tof, _ = remove_time_of_flight(freq_axis, meas_phase_rad_unwrapped)
    
    # 3. Calculate Minimum Phase of Measurement
    meas_min_phase_rad = calculate_minimum_phase(10**(meas_mags/20.0))
    
    # 4. Excess Phase (Distortion) = Cleaned Measurement - Minimum Phase
    # This now contains ONLY group delay distortions, not distance delay
    excess_phase_deg = np.rad2deg(meas_phase_no_tof - meas_min_phase_rad)
    excess_phase_fdw_rad = apply_fdw_smoothing(freq_axis, excess_phase_deg, fdw_cycles)
    
    theoretical_xo_phase = calculate_theoretical_phase(freq_axis, crossovers)
    
    phase_corr_rad = np.zeros_like(freq_axis)
    limit_rad = np.deg2rad(fine_phase_limit)
    
    for i, f in enumerate(freq_axis):
        if f > 0:
            if phase_c_min <= f <= phase_c_max:
                val = -excess_phase_fdw_rad[i]
                val = np.clip(val, -limit_rad, limit_rad)
                
                fade_start = phase_c_max * 0.8
                fade_factor = 1.0
                if f > fade_start:
                    fade_factor = (phase_c_max - f) / (phase_c_max - fade_start)
                    fade_factor = np.clip(fade_factor, 0.0, 1.0)
                
                phase_corr_rad[i] = -theoretical_xo_phase[i] + (val * fade_factor)
    
    # --- GENERATE IMPULSE ---
    impulse_out = None
    
    if 'Min' in filter_type_str:
        final_complex = total_mag_response * np.exp(1j * filt_min_phase_rad)
        impulse_out = scipy.fft.irfft(final_complex, n=n_fft)
        
    elif 'Mixed' in filter_type_str:
        complex_min = total_mag_response * np.exp(1j * filt_min_phase_rad)
        ir_min = scipy.fft.irfft(complex_min, n=n_fft)
        
        complex_lin = total_mag_response * np.exp(1j * phase_corr_rad) 
        ir_lin = scipy.fft.irfft(complex_lin, n=n_fft)
        ir_lin = np.roll(ir_lin, n_fft // 2)
        
        impulse_out = combine_mixed_phase(ir_lin, ir_min, fs, split_freq=mixed_split_freq)
        impulse_out *= scipy.signal.windows.tukey(n_fft, alpha=0.05)
        
    else: # Linear Phase
        correction_complex = gain_linear * np.exp(1j * phase_corr_rad)
        final_complex = correction_complex * hpf_complex
        impulse = scipy.fft.irfft(final_complex, n=n_fft)
        impulse_out = np.roll(impulse, n_fft // 2) * scipy.signal.windows.tukey(n_fft, alpha=0.05)

    # --- NORMALIZATION ---
    max_peak = np.max(np.abs(impulse_out))
    if do_normalize and max_peak > 0:
        target_amp = 0.891
        if max_peak > target_amp:
            impulse_out *= (target_amp / max_peak)

    # --- STATS CALCULATION ---
    w, h_filt = scipy.signal.freqz(impulse_out, 1, worN=2048, fs=fs)
    freqs_gd = w
    phase_filt = np.rad2deg(np.unwrap(np.angle(h_filt)))
    gd_ms = calculate_group_delay(freqs_gd, phase_filt)
    mask_gd = (freqs_gd > 20) & (freqs_gd < 20000)
    gd_clean = gd_ms[mask_gd]
    
    stats = {
        'offset_db': calc_offset_db,
        'eff_target_db': eff_target_db, 
        'correction_enabled': enable_mag_correction,
        'target_mags': target_mags, 
        'freq_axis': freq_axis,     
        'l_match_min': l_match_min,
        'l_match_max': l_match_max,
        'peak_before_norm': 20 * np.log10(max_peak + 1e-12),
        'normalized': (do_normalize and max_peak > 0.891),
        'gd_min': np.min(gd_clean) if len(gd_clean) > 0 else 0.0,
        'gd_max': np.max(gd_clean) if len(gd_clean) > 0 else 0.0
    }
    
    return impulse_out, stats['gd_min'], stats['gd_max'], stats