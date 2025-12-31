import numpy as np
import scipy.signal
import scipy.io.wavfile
import scipy.fft
import sys
import os
from datetime import datetime

# --- CONSTANTS ---
INPUT_FILE_L = 'L.txt'
INPUT_FILE_R = 'R.txt'
FINE_TUNE_LIMIT = 45.0   # Max phase fine-tune correction (degrees)

def check_files_exist():
    """Checks if required measurement files exist immediately."""
    missing = []
    if not os.path.exists(INPUT_FILE_L):
        missing.append(INPUT_FILE_L)
    if not os.path.exists(INPUT_FILE_R):
        missing.append(INPUT_FILE_R)
    
    if missing:
        print("\n" + "="*40)
        print(" CRITICAL ERROR: MISSING FILES")
        print("="*40)
        print("The following measurement files were not found:")
        for f in missing:
            print(f" - {f}")
        print("\nPlease export your measurements from REW as text files")
        print(f"and name them '{INPUT_FILE_L}' and '{INPUT_FILE_R}'.")
        print("Place them in the same folder as this program.")
        print("="*40 + "\n")
        input("Press Enter to exit...")
        sys.exit(1)

def parse_measurements(filename):
    try:
        data = np.loadtxt(filename)
        freqs = data[:, 0]
        mags = data[:, 1]
        phases = data[:, 2]
        return freqs, mags, phases
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        input("Press Enter to close...")
        sys.exit(1)

def parse_house_curve(filename):
    try:
        if not os.path.exists(filename):
            print(f"Warning: House curve file '{filename}' not found.")
            return None, None
        data = np.loadtxt(filename)
        return data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Error reading house curve: {e}")
        return None, None

def ask_taps():
    """Asks user to select filter length (taps)."""
    options = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    print("\n--- Select Filter Length (Taps) ---")
    print("Higher taps = Better bass resolution but more latency.")
    
    for i, taps in enumerate(options):
        latency_ms = (taps / 2 / 48000) * 1000
        default_mark = " (Default)" if taps == 65536 else ""
        print(f"{i+1}) {taps:<7} (~{latency_ms:.0f} ms latency){default_mark}")

    while True:
        try:
            val = input("Selection (number): ").strip()
            if not val: return 65536
            idx = int(val) - 1
            if 0 <= idx < len(options): return options[idx]
            print("Invalid number.")
        except ValueError: print("Please enter a number.")

def ask_max_boost():
    """Asks user for maximum boost in dB."""
    print("\n--- Max Boost limit (dB) ---")
    print("Limits how much the filter can amplify dips in response.")
    print("Range: 0 - 15 dB. Default: 5 dB.")
    
    while True:
        try:
            val = input("Max Boost dB [5]: ").strip()
            if not val: return 5.0
            
            boost = float(val)
            if 0.0 <= boost <= 15.0:
                return boost
            print("Please enter a value between 0 and 15.")
        except ValueError:
            print("Invalid input. Enter a number.")

def ask_global_gain():
    """Asks user for global output gain in dB."""
    print("\n--- Global Output Gain (dB) ---")
    print("Adjusts the overall level of the generated WAV file.")
    print("If REW shows 117dB and you want 0dB, enter -117.")
    print("Default: 0 dB (Unity Gain).")
    
    while True:
        try:
            val = input("Gain dB [0]: ").strip()
            if not val: return 0.0
            return float(val)
        except ValueError:
            print("Invalid input. Enter a number.")

def ask_output_options():
    """Asks user for output format and channel layout."""
    print("\n--- Output File Format ---")
    print("1) WAV (Standard for convolvers)")
    print("2) CSV (Text format for Equalizer APO)")
    
    fmt = 'wav'
    while True:
        val = input("Selection [1]: ").strip()
        if not val or val == '1': 
            fmt = 'wav'
            break
        if val == '2': 
            fmt = 'csv'
            break
        print("Invalid input.")

    print("\n--- Channel Layout ---")
    print("1) Mono files (Separate L and R files)")
    print("2) Stereo file (One file containing both channels)")
    
    is_stereo = False
    while True:
        val = input("Selection [1]: ").strip()
        if not val or val == '1':
            is_stereo = False
            break
        if val == '2':
            is_stereo = True
            break
        print("Invalid input.")

    return fmt, is_stereo

def ask_highpass():
    """Asks user for High Pass Filter settings."""
    print("\n--- High Pass Filter (Subsonic) ---")
    print("Protect the speakers/subs from low frequencies.")
    if input("Enable High Pass Filter (10-60Hz)? (y/n) [n]: ").strip().lower() == 'y':
        while True:
            try:
                freq = float(input("  Frequency (10-60 Hz): "))
                if 10 <= freq <= 60:
                    break
                print("Frequency must be between 10 and 60 Hz.")
            except ValueError: print("Invalid number.")
        
        while True:
            try:
                slope = int(input("  Slope (6, 12, 24, 48 dB/oct): "))
                if slope in [6, 12, 24, 48]:
                    order = slope // 6
                    break
                print("Invalid slope. Choose 6, 12, 24 or 48.")
            except ValueError: print("Invalid number.")
            
        return {'freq': freq, 'order': order}
    return None

def ask_house_curve_file():
    files = [f for f in os.listdir('.') if f.endswith('.txt')]
    ignore_files = [INPUT_FILE_L, INPUT_FILE_R, 'requirements.txt', 'gemini.txt']
    candidates = [f for f in files if f not in ignore_files]

    if not candidates:
        print("\nNo suitable House Curve (.txt) files found in the folder.")
        return None

    print("\n--- Select House Curve ---")
    print("0) No correction (Phase linearization only)")
    for i, fname in enumerate(candidates):
        print(f"{i+1}) {fname}")

    while True:
        try:
            val = input("Selection (number): ").strip()
            sel = int(val)
            if sel == 0: return None
            if 1 <= sel <= len(candidates): return candidates[sel-1]
            print("Invalid number.")
        except ValueError: print("Please enter a number.")

def ask_crossovers():
    crossovers = []
    print("\n--- Crossover Configuration (Phase Linearization) ---")
    print("Define IIR crossovers to be linearized.")
    while True:
        try:
            val = input("How many crossover points (0, 1, 2...)? [0]: ").strip()
            if not val: count = 0
            else: count = int(val)
            break
        except ValueError: print("Invalid number.")
            
    for i in range(count):
        print(f"\nCrossover #{i+1}:")
        while True:
            try:
                freq = float(input("  Frequency (Hz): "))
                slope = int(input("  Slope (dB/oct, e.g., 12, 24): "))
                crossovers.append({'freq': freq, 'order': slope // 6})
                break
            except ValueError: print("Invalid input.")
    return crossovers

def calculate_theoretical_phase(freq_axis, crossovers):
    if not crossovers: return np.zeros_like(freq_axis)
    total_phase_rad = np.zeros_like(freq_axis)
    for xo in crossovers:
        b, a = scipy.signal.butter(xo['order'], 2 * np.pi * xo['freq'], btype='low', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))
    return total_phase_rad

def apply_smoothing(freqs, mags, phases, octave_fraction=1.0):
    f_min = max(freqs[0], 1.0)
    f_max = freqs[-1]
    points_per_octave = 192 
    num_points = int(np.log2(f_max / f_min) * points_per_octave)
    log_freqs = np.geomspace(f_min, f_max, num_points)
    
    log_mags = np.interp(log_freqs, freqs, mags)
    phase_unwrap = np.unwrap(np.deg2rad(phases))
    log_phases = np.interp(log_freqs, freqs, phase_unwrap)
    
    window_size = int(points_per_octave * octave_fraction)
    if window_size < 1: window_size = 1
    window = np.ones(window_size) / window_size
    pad_len = window_size // 2
    
    m_padded = np.pad(log_mags, (pad_len, pad_len), mode='edge')
    sm_mags = np.convolve(m_padded, window, mode='same')[pad_len:-pad_len]
    
    p_padded = np.pad(log_phases, (pad_len, pad_len), mode='edge')
    sm_phases = np.convolve(p_padded, window, mode='same')[pad_len:-pad_len]
        
    new_mags = np.interp(freqs, log_freqs, sm_mags)
    new_phases_rad = np.interp(freqs, log_freqs, sm_phases)
    return new_mags, np.rad2deg(new_phases_rad)

def interpolate_response(input_freqs, input_values, target_freqs):
    return np.interp(target_freqs, input_freqs, input_values)

def detrend_phase(freqs, unwrap_phase_rad):
    valid_mask = (freqs > 100) & (freqs < freqs[-1] * 0.9)
    if np.sum(valid_mask) < 10: valid_mask = freqs > 0
    p = np.polyfit(freqs[valid_mask], unwrap_phase_rad[valid_mask], 1)
    linear_trend = np.polyval(p, freqs)
    return unwrap_phase_rad - linear_trend

def find_zero_crossing_raw(freqs, phases, search_min=20, search_max=1000):
    mask = (freqs >= search_min) & (freqs <= search_max)
    f_sub = freqs[mask]
    p_sub = phases[mask]
    if len(f_sub) < 2: return search_min
    for i in range(len(p_sub) - 1):
        if np.sign(p_sub[i]) != np.sign(p_sub[i+1]):
            if abs(p_sub[i] - p_sub[i+1]) < 90.0: return f_sub[i]
    return f_sub[np.argmin(np.abs(p_sub))]

def find_closest_to_zero_raw(freqs, phases, search_min=800, search_max=2000):
    mask = (freqs >= search_min) & (freqs <= search_max)
    f_sub = freqs[mask]
    p_sub = phases[mask]
    if len(f_sub) == 0: return search_max
    return f_sub[np.argmin(np.abs(p_sub))]

def generate_filter(freqs, smoothed_mags, smoothed_phases, crossovers, 
                    phase_c_min, phase_c_max, mag_c_min, mag_c_max,
                    house_freqs, house_mags, fs, num_taps, fine_phase_limit, 
                    max_boost_db, global_gain_db, hpf_settings):
    
    n_fft = num_taps if num_taps % 2 != 0 else num_taps + 1
    nyquist = fs / 2.0
    freq_axis = np.linspace(0, nyquist, n_fft // 2 + 1)
    
    meas_mags = interpolate_response(freqs, smoothed_mags, freq_axis)
    meas_phases_excess = detrend_phase(freq_axis, interpolate_response(freqs, np.deg2rad(smoothed_phases), freq_axis))
    theoretical_xo_phase = calculate_theoretical_phase(freq_axis, crossovers)
    
    target_mags = np.zeros_like(freq_axis)
    use_house_curve = (house_freqs is not None and house_mags is not None)
    if use_house_curve:
        hc_interp = interpolate_response(house_freqs, house_mags, freq_axis)
        idx_align = np.argmin(np.abs(freq_axis - mag_c_max))
        target_mags = hc_interp + (meas_mags[idx_align] - hc_interp[idx_align])
    
    # Calculate HPF complex response (if enabled)
    hpf_complex = np.ones_like(freq_axis, dtype=complex)
    if hpf_settings:
        b, a = scipy.signal.butter(hpf_settings['order'], 2 * np.pi * hpf_settings['freq'], btype='high', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        hpf_complex = h
        hpf_complex[0] = 0.0 # DC

    gain_linear = np.ones_like(freq_axis)
    phase_corr_rad = np.zeros_like(freq_axis)
    limit_rad = np.deg2rad(fine_phase_limit)
    
    for i, f in enumerate(freq_axis):
        if f > 0:
            # 1. Frequency Response Correction
            g_db = 0.0
            if use_house_curve and (mag_c_min <= f <= mag_c_max):
                g_db = np.clip(target_mags[i] - meas_mags[i], -15.0, max_boost_db)
            
            # 2. Apply Global Gain
            g_db += global_gain_db
            
            gain_linear[i] = 10.0 ** (g_db / 20.0)
            
            # 3. Phase Correction
            fine_correction = 0.0
            if phase_c_min <= f <= phase_c_max:
                fine_correction = np.clip(-(meas_phases_excess[i] - theoretical_xo_phase[i]), -limit_rad, limit_rad)
            phase_corr_rad[i] = -theoretical_xo_phase[i] + fine_correction

    # Combine Correction + HPF
    correction_complex = gain_linear * np.exp(1j * phase_corr_rad)
    final_complex = correction_complex * hpf_complex

    impulse = scipy.fft.irfft(final_complex, n=n_fft)
    window = scipy.signal.windows.blackman(n_fft)
    return np.roll(impulse, n_fft // 2) * window

# --- MAIN PROGRAM ---
try:
    print("--- Speaker Corrector (EXE Version) ---")
    print("    By: VilhoValittu & GeminiPro")
    
    check_files_exist()

    # 1. Output Format & Layout Selection
    output_fmt, is_stereo = ask_output_options()
    ext = ".wav" if output_fmt == 'wav' else ".csv"

    # 2. Filter Taps
    num_taps = ask_taps()
    print(f"--> Taps: {num_taps}")

    # 3. Sample Rate (Now needed before filename generation)
    print("\nSelect Sample Rate:")
    print("1) 44100 Hz, 2) 48000 Hz, 3) 96000 Hz")
    sr_map = {'1': 44100, '2': 48000, '3': 96000}
    sample_rate = sr_map.get(input("Selection (default 2): ").strip(), 48000)
    print(f"--> FS: {sample_rate} Hz")

    # 4. Generate Filenames
    now = datetime.now()
    timestamp = now.strftime('%d%m%y_%H%M')
    output_filename_l = f"L_corr_{sample_rate}Hz_{timestamp}{ext}"
    output_filename_r = f"R_corr_{sample_rate}Hz_{timestamp}{ext}"
    output_filename_stereo = f"Stereo_corr_{sample_rate}Hz_{timestamp}{ext}"

    # 5. Crossovers
    crossovers = ask_crossovers()

    # 6. High Pass Filter
    hpf_settings = ask_highpass()
    if hpf_settings:
        print(f"--> High Pass: {hpf_settings['freq']} Hz, {hpf_settings['order']*6} dB/oct")

    # 7. House Curve Selection & Boost
    selected_hc_file = ask_house_curve_file()
    
    hc_freqs = None
    hc_mags = None
    mag_min, mag_max = 10, 200
    user_max_boost = 5.0 

    if selected_hc_file:
        print(f"Reading: {selected_hc_file}...")
        hc_freqs, hc_mags = parse_house_curve(selected_hc_file)
        
        if hc_freqs is not None:
            user_max_boost = ask_max_boost()
            
            if input(f"Apply freq correction {mag_min}-{mag_max} Hz? (y/n) [y]: ").strip().lower() != 'n':
                try:
                    user_min = input(f"  Lower limit Hz [{mag_min}]: ").strip()
                    if user_min: mag_min = float(user_min)
                    user_max = input(f"  Upper limit Hz [{mag_max}]: ").strip()
                    if user_max: mag_max = float(user_max)
                except: pass
                print(f"--> Mag Correction: {mag_min}-{mag_max} Hz (Max Boost {user_max_boost} dB)")
            else:
                hc_freqs = None 
    else:
        print("--> No frequency response correction.")

    # 8. Global Gain (Skip if CSV)
    if output_fmt == 'wav':
        global_gain = ask_global_gain()
        print(f"--> Global Gain: {global_gain} dB")
    else:
        global_gain = 0.0

    # 9. Loading Data
    print("\nReading measurements (L.txt, R.txt)...")
    freqs_l, mags_l, phases_l = parse_measurements(INPUT_FILE_L)
    freqs_r, mags_r, phases_r = parse_measurements(INPUT_FILE_R)

    # 10. Auto Limits
    print("Analyzing phase data for auto-limits...")
    c_min_auto = (find_zero_crossing_raw(freqs_l, phases_l) + find_zero_crossing_raw(freqs_r, phases_r)) / 2.0
    c_max_auto = (find_closest_to_zero_raw(freqs_l, phases_l) + find_closest_to_zero_raw(freqs_r, phases_r)) / 2.0
    print(f" - Phase Correction Range (auto): {c_min_auto:.0f} - {c_max_auto:.0f} Hz")

    # 11. Smoothing
    print("Smoothing data (Mag=1/48 oct, Phase=1/1 oct)...")
    sm_mags_l, _ = apply_smoothing(freqs_l, mags_l, phases_l, 1/48.0)
    sm_mags_r, _ = apply_smoothing(freqs_r, mags_r, phases_r, 1/48.0)
    _, sm_phases_l = apply_smoothing(freqs_l, mags_l, phases_l, 1.0)
    _, sm_phases_r = apply_smoothing(freqs_r, mags_r, phases_r, 1.0)

    # 12. Generation
    print(f"\nGenerating filters...")
    l_imp = generate_filter(freqs_l, sm_mags_l, sm_phases_l, crossovers, c_min_auto, c_max_auto, mag_min, mag_max, hc_freqs, hc_mags, sample_rate, num_taps, FINE_TUNE_LIMIT, user_max_boost, global_gain, hpf_settings)
    r_imp = generate_filter(freqs_r, sm_mags_r, sm_phases_r, crossovers, c_min_auto, c_max_auto, mag_min, mag_max, hc_freqs, hc_mags, sample_rate, num_taps, FINE_TUNE_LIMIT, user_max_boost, global_gain, hpf_settings)

    if output_fmt == 'wav':
        if is_stereo:
            # Stack for stereo WAV
            stereo_data = np.column_stack((l_imp, r_imp))
            scipy.io.wavfile.write(output_filename_stereo, sample_rate, stereo_data.astype(np.float32))
            print(f"\nDone! Saved: {output_filename_stereo}")
        else:
            # Mono WAVs
            scipy.io.wavfile.write(output_filename_l, sample_rate, l_imp.astype(np.float32))
            scipy.io.wavfile.write(output_filename_r, sample_rate, r_imp.astype(np.float32))
            print(f"\nDone! Saved: {output_filename_l}, {output_filename_r}")
    else:
        # Save as CSV
        if is_stereo:
            stereo_data = np.column_stack((l_imp, r_imp))
            np.savetxt(output_filename_stereo, stereo_data, fmt='%.18f', delimiter=' ')
            print(f"\nDone! Saved: {output_filename_stereo}")
        else:
            np.savetxt(output_filename_l, l_imp, fmt='%.18f')
            np.savetxt(output_filename_r, r_imp, fmt='%.18f')
            print(f"\nDone! Saved: {output_filename_l}, {output_filename_r}")

except Exception as e:
    print(f"\nERROR: {e}")
finally:
    input("\nPress Enter to exit...")