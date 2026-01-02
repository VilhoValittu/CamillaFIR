import numpy as np
import scipy.signal
import scipy.io.wavfile
import scipy.fft
import sys
import os
import io
import json
from datetime import datetime

# --- MATPLOTLIB SETUP ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- GUI LIBRARY ---
from pywebio.input import *
from pywebio.output import *
from pywebio import start_server, config

# --- CONSTANTS ---
INPUT_FILE_L = 'L.txt'
INPUT_FILE_R = 'R.txt'
CONFIG_FILE = 'config.json'
FINE_TUNE_LIMIT = 45.0
MAX_SAFE_BOOST = 8.0
VERSION = "v1.1.0 (Auto-Save Config)"
PROGRAM_NAME = "CamillaFIR"

# --- CONFIG MANAGEMENT ---
def load_config():
    """Loads settings from config.json if exists, else returns defaults."""
    default_conf = {
        'fmt': 'WAV', 'layout': 'Stereo (Single file)', 'fs': 48000, 'taps': 65536,
        'gain': 0.0, 'hc_file': "Built-in ('not Dr Toole')", 'mag_correct': ['Enable Frequency Response Correction'],
        'hc_min': 10, 'hc_max': 200, 'max_boost': 5.0,
        'hpf_enable': [], 'hpf_freq': None, 'hpf_slope': 24,
        'xo1_f': None, 'xo1_s': 12, 'xo2_f': None, 'xo2_s': 12,
        'xo3_f': None, 'xo3_s': 12, 'xo4_f': None, 'xo4_s': 12, 'xo5_f': None, 'xo5_s': 12
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_conf = json.load(f)
                default_conf.update(saved_conf) # Merge saved into defaults
        except:
            pass
    return default_conf

def save_config(data):
    """Saves current settings to config.json."""
    # Filter out sensitive or non-serializable data if any
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Failed to save config: {e}")

# --- DATA PARSING ---
def get_default_house_curve():
    freqs = np.array([20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 20000.0])
    mags = np.array([6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5, 1.4, 0.4, 0.0, -4.0])
    return freqs, mags

def parse_measurements(filename):
    try:
        data = np.loadtxt(filename)
        if data.shape[1] < 3: return None, None, None
        return data[:, 0], data[:, 1], data[:, 2] 
    except: return None, None, None

def parse_house_curve(filename):
    try:
        data = np.loadtxt(filename)
        return data[:, 0], data[:, 1]
    except: return None, None

# --- DSP LOGIC ---
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
    window_size = max(1, int(points_per_octave * octave_fraction))
    window = np.ones(window_size) / window_size
    pad_len = window_size // 2
    m_padded = np.pad(log_mags, (pad_len, pad_len), mode='edge')
    sm_mags = np.convolve(m_padded, window, mode='same')[pad_len:-pad_len]
    p_padded = np.pad(log_phases, (pad_len, pad_len), mode='edge')
    sm_phases = np.convolve(p_padded, window, mode='same')[pad_len:-pad_len]
    return np.interp(freqs, log_freqs, sm_mags), np.rad2deg(np.interp(freqs, log_freqs, sm_phases))

def interpolate_response(input_freqs, input_values, target_freqs):
    return np.interp(target_freqs, input_freqs, input_values)

def detrend_phase(freqs, unwrap_phase_rad):
    valid_mask = (freqs > 100) & (freqs < freqs[-1] * 0.9)
    if np.sum(valid_mask) < 10: valid_mask = freqs > 0
    p = np.polyfit(freqs[valid_mask], unwrap_phase_rad[valid_mask], 1)
    return unwrap_phase_rad - np.polyval(p, freqs)

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
                    max_boost_db, global_gain_db, hpf_settings, enable_mag_correction):
    
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
    
    hpf_complex = np.ones_like(freq_axis, dtype=complex)
    if hpf_settings and hpf_settings['enabled']:
        b, a = scipy.signal.butter(hpf_settings['order'], 2 * np.pi * hpf_settings['freq'], btype='high', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        hpf_complex = h
        hpf_complex[0] = 0.0

    gain_linear = np.ones_like(freq_axis)
    phase_corr_rad = np.zeros_like(freq_axis)
    limit_rad = np.deg2rad(fine_phase_limit)
    
    for i, f in enumerate(freq_axis):
        if f > 0:
            g_db = 0.0
            if enable_mag_correction and use_house_curve and (mag_c_min <= f <= mag_c_max):
                safe_max_boost = min(max_boost_db, MAX_SAFE_BOOST)
                g_db = np.clip(target_mags[i] - meas_mags[i], -15.0, safe_max_boost)
            g_db += global_gain_db
            gain_linear[i] = 10.0 ** (g_db / 20.0)
            
            fine_correction = 0.0
            if phase_c_min <= f <= phase_c_max:
                fine_correction = np.clip(-(meas_phases_excess[i] - theoretical_xo_phase[i]), -limit_rad, limit_rad)
            phase_corr_rad[i] = -theoretical_xo_phase[i] + fine_correction

    correction_complex = gain_linear * np.exp(1j * phase_corr_rad)
    final_complex = correction_complex * hpf_complex
    final_phase_unwrapped = np.unwrap(np.angle(final_complex))
    gd_samples = -np.gradient(final_phase_unwrapped) * (len(freq_axis) / np.pi) 
    gd_ms = (gd_samples / fs) * 1000.0
    mask = (freq_axis >= 20) & (freq_axis <= 20000)
    if np.sum(mask) > 0:
        valid_gd = gd_ms[mask]
        gd_min, gd_max = np.min(valid_gd), np.max(valid_gd)
    else:
        gd_min, gd_max = 0.0, 0.0

    impulse = scipy.fft.irfft(final_complex, n=n_fft)
    window = scipy.signal.windows.blackman(n_fft)
    return np.roll(impulse, n_fft // 2) * window, gd_min, gd_max

def save_summary(filename, settings, l_stats, r_stats):
    with open(filename, 'w') as f:
        f.write(f"=== {PROGRAM_NAME} - Filter Generation Summary ===\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("--- Settings ---\n")
        for key, val in settings.items():
            f.write(f"{key}: {val}\n")
        f.write("\n--- Analysis (20Hz - 20kHz) ---\n")
        f.write(f"Left Channel GD: {l_stats['gd_min']:.2f} ms to {l_stats['gd_max']:.2f} ms\n")
        f.write(f"Right Channel GD: {r_stats['gd_min']:.2f} ms to {r_stats['gd_max']:.2f} ms\n")

def generate_prediction_plot(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title):
    try:
        n_fft = len(filt_ir)
        freq_axis_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt_complex = scipy.fft.rfft(filt_ir)
        
        orig_mags_lin = np.interp(freq_axis_lin, orig_freqs, orig_mags)
        orig_phases_lin = np.interp(freq_axis_lin, orig_freqs, orig_phases)
        
        orig_complex_lin = 10**(orig_mags_lin/20.0) * np.exp(1j * np.deg2rad(orig_phases_lin))
        total_complex = orig_complex_lin * h_filt_complex
        
        ir_total = scipy.fft.irfft(total_complex, n=n_fft)
        peak_idx = np.argmax(np.abs(ir_total))
        ir_centered = np.roll(ir_total, -peak_idx)
        total_complex_centered = scipy.fft.rfft(ir_centered)
        
        final_mags_lin = 20 * np.log10(np.abs(total_complex) + 1e-12)
        final_phases_rad = np.angle(total_complex_centered)
        final_phases_deg = np.rad2deg(final_phases_rad)
        
        final_mags_plot = np.interp(orig_freqs, freq_axis_lin, final_mags_lin)
        final_phases_plot = np.interp(orig_freqs, freq_axis_lin, final_phases_deg)
        
        _, plot_phase_orig = apply_smoothing(orig_freqs, orig_mags, orig_phases, 1.0)
        _, plot_phase_pred = apply_smoothing(orig_freqs, final_mags_plot, final_phases_plot, 1.0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ticks_val = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        ticks_lab = ['20', '50', '100', '200', '500', '1k', '2k', '5k', '10k', '20k']
        x_min = max(orig_freqs[0], 10)
        x_max = orig_freqs[-1]

        ax1.semilogx(orig_freqs, orig_mags, label='Original', color='blue', alpha=0.5)
        ax1.semilogx(orig_freqs, final_mags_plot, label='Predicted', color='orange', linewidth=2)
        ax1.set_title(f"{title} - Frequency Response")
        ax1.set_ylabel("Amplitude (dB)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend()
        ax1.set_xlim(x_min, x_max)
        ax1.set_xticks(ticks_val)
        ax1.set_xticklabels(ticks_lab)
        
        plot_phase_orig_wrapped = (plot_phase_orig + 180) % 360 - 180
        plot_phase_pred_wrapped = (plot_phase_pred + 180) % 360 - 180

        ax2.semilogx(orig_freqs, plot_phase_orig_wrapped, label='Original (Smoothed)', color='blue', alpha=0.5)
        ax2.semilogx(orig_freqs, plot_phase_pred_wrapped, label='Predicted (Delay Removed)', color='orange', linewidth=2)
        
        ax2.set_title("Phase Response (Peak Aligned)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (deg)")
        ax2.grid(True, which="both", alpha=0.3)
        ax2.set_xlim(x_min, x_max)
        ax2.set_xticks(ticks_val)
        ax2.set_xticklabels(ticks_lab)
        ax2.set_ylim(-360, 360)
        ax2.set_yticks(np.arange(-360, 361, 90))
        ax2.legend()
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        print(f"Plotting Error: {e}")
        return None

# --- MAIN GUI APP ---
@config(theme="dark")
def main():
    put_markdown(f"# 🎛️ {PROGRAM_NAME} {VERSION}")
    put_markdown("### By VilhoValittu & GeminiPro")
    
    if not os.path.exists(INPUT_FILE_L) or not os.path.exists(INPUT_FILE_R):
        put_error("CRITICAL ERROR: Measurement files not found!")
        put_text(f"Please place '{INPUT_FILE_L}' and '{INPUT_FILE_R}' in this folder.")
        return

    hc_files = [f for f in os.listdir('.') if f.endswith('.txt') and f not in [INPUT_FILE_L, INPUT_FILE_R, 'requirements.txt']]
    hc_options = ['None', "Built-in ('not Dr Toole')"] + hc_files

    # LOAD DEFAULTS
    defaults = load_config()

    inputs_list = [
        select("Output Format", options=['WAV', 'CSV'], value=defaults['fmt'], name='fmt'),
        radio("Channel Layout", options=['Mono (Separate files)', 'Stereo (Single file)'], value=defaults['layout'], name='layout'),
        select("Sample Rate", options=[44100, 48000, 96000], value=defaults['fs'], name='fs', type=NUMBER),
        select("Taps", options=[2048, 4096, 8192, 16384, 32768, 65536, 131072], value=defaults['taps'], name='taps', type=NUMBER),
        input("Global Gain (dB)", value=defaults['gain'], type=FLOAT, name='gain', help_text="Adjusts volume. Use negative values (e.g. -3dB) if boosting."),
        select("House Curve", options=hc_options, value=defaults['hc_file'], name='hc_file'),
        checkbox("Correction Mode", options=['Enable Frequency Response Correction'], value=defaults['mag_correct'], name='mag_correct'),
        input("Correction Min Freq (Hz)", value=defaults['hc_min'], type=FLOAT, name='hc_min'),
        input("Correction Max Freq (Hz)", value=defaults['hc_max'], type=FLOAT, name='hc_max'),
        input("Max Boost (dB)", value=defaults['max_boost'], type=FLOAT, name='max_boost', help_text=f"Note: Boost is safely limited to max {MAX_SAFE_BOOST} dB internally."),
        checkbox("High Pass Filter", options=['Enable'], value=defaults['hpf_enable'], name='hpf_enable'),
        input("HPF Freq (Hz)", value=defaults['hpf_freq'], type=FLOAT, name='hpf_freq', placeholder="e.g. 20"), 
        select("HPF Slope (dB/oct)", options=[6, 12, 18, 24, 36, 48], value=defaults['hpf_slope'], type=NUMBER, name='hpf_slope')
    ]

    slope_opts = [6, 12, 18, 24, 36, 48]
    for i in range(1, 6):
        label_txt = f"XO {i} Freq (Hz)"
        if i == 1: label_txt += " (Phase Linearization)"
        inputs_list.append(input(label_txt, value=defaults[f'xo{i}_f'], name=f'xo{i}_f', type=FLOAT, placeholder="Leave empty if unused"))
        inputs_list.append(select(f"XO {i} Slope (dB/oct)", options=slope_opts, value=defaults[f'xo{i}_s'], name=f'xo{i}_s'))

    data = input_group("Filter Settings", inputs_list)
    
    # SAVE CONFIG
    save_config(data)

    put_processbar('bar')
    set_processbar('bar', 0.2)
    put_text("Reading measurements...")
    
    do_mag_correct = 'Enable Frequency Response Correction' in data['mag_correct']
    crossovers = []
    for i in range(1, 6):
        if data[f'xo{i}_f']:
            crossovers.append({'freq': float(data[f'xo{i}_f']), 'order': int(data[f'xo{i}_s']) // 6})

    hpf_settings = None
    if 'Enable' in data['hpf_enable'] and data['hpf_freq']:
        hpf_settings = {'enabled': True, 'freq': data['hpf_freq'], 'order': data['hpf_slope'] // 6}

    hc_freqs, hc_mags = None, None
    if data['hc_file'] == "Built-in ('not Dr Toole')":
        hc_freqs, hc_mags = get_default_house_curve()
    elif data['hc_file'] != 'None':
        hc_freqs, hc_mags = parse_house_curve(data['hc_file'])
    
    if hc_freqs is None:
        data['hc_min'], data['hc_max'] = 0, 0

    freqs_l, mags_l, phases_l = parse_measurements(INPUT_FILE_L)
    freqs_r, mags_r, phases_r = parse_measurements(INPUT_FILE_R)
    
    if freqs_l is None:
        put_error("Failed to read L.txt/R.txt")
        return

    set_processbar('bar', 0.5)
    put_text("Analyzing and Generating Filters...")

    c_min = (find_zero_crossing_raw(freqs_l, phases_l) + find_zero_crossing_raw(freqs_r, phases_r)) / 2
    c_max = (find_closest_to_zero_raw(freqs_l, phases_l) + find_closest_to_zero_raw(freqs_r, phases_r)) / 2

    sm_mags_l, sm_phases_l = apply_smoothing(freqs_l, mags_l, phases_l, 1/48.0)
    sm_mags_r, sm_phases_r = apply_smoothing(freqs_r, mags_r, phases_r, 1/48.0)
    _, sm_phases_l_wide = apply_smoothing(freqs_l, mags_l, phases_l, 1.0)
    _, sm_phases_r_wide = apply_smoothing(freqs_r, mags_r, phases_r, 1.0)

    l_imp, l_min, l_max = generate_filter(
        freqs_l, sm_mags_l, sm_phases_l_wide, crossovers, c_min, c_max,
        data['hc_min'], data['hc_max'], hc_freqs, hc_mags, data['fs'], data['taps'],
        FINE_TUNE_LIMIT, data['max_boost'], data['gain'], hpf_settings, do_mag_correct
    )
    r_imp, r_min, r_max = generate_filter(
        freqs_r, sm_mags_r, sm_phases_r_wide, crossovers, c_min, c_max,
        data['hc_min'], data['hc_max'], hc_freqs, hc_mags, data['fs'], data['taps'],
        FINE_TUNE_LIMIT, data['max_boost'], data['gain'], hpf_settings, do_mag_correct
    )

    set_processbar('bar', 0.8)

    now = datetime.now()
    ts = now.strftime('%d%m%y_%H%M')
    ext = ".wav" if data['fmt'] == 'WAV' else ".csv"
    is_stereo = 'Stereo' in data['layout']
    
    fn_l = f"L_corr_{data['fs']}Hz_{ts}{ext}"
    fn_r = f"R_corr_{data['fs']}Hz_{ts}{ext}"
    fn_s = f"Stereo_corr_{data['fs']}Hz_{ts}{ext}"
    fn_sum = f"Summary_{ts}.txt"

    if data['fmt'] == 'WAV':
        if is_stereo:
            scipy.io.wavfile.write(fn_s, data['fs'], np.column_stack((l_imp, r_imp)).astype(np.float32))
            put_success(f"Saved: {fn_s}")
        else:
            scipy.io.wavfile.write(fn_l, data['fs'], l_imp.astype(np.float32))
            scipy.io.wavfile.write(fn_r, data['fs'], r_imp.astype(np.float32))
            put_success(f"Saved: {fn_l}, {fn_r}")
    else:
        if is_stereo:
            np.savetxt(fn_s, np.column_stack((l_imp, r_imp)), fmt='%.18f', delimiter=' ')
            put_success(f"Saved: {fn_s}")
        else:
            np.savetxt(fn_l, l_imp, fmt='%.18f')
            np.savetxt(fn_r, r_imp, fmt='%.18f')
            put_success(f"Saved: {fn_l}, {fn_r}")

    settings_dict = data.copy()
    settings_dict['Magnitude Correction'] = "Enabled" if do_mag_correct else "Disabled"
    settings_dict['Crossovers'] = str(crossovers)
    settings_dict['Phase Range'] = f"{c_min:.0f}-{c_max:.0f} Hz"
    save_summary(fn_sum, settings_dict, {'gd_min': l_min, 'gd_max': l_max}, {'gd_min': r_min, 'gd_max': r_max})
    put_info(f"Summary saved to {fn_sum}")
    
    try:
        img_l = generate_prediction_plot(freqs_l, mags_l, phases_l, l_imp, data['fs'], "Left Channel")
        img_r = generate_prediction_plot(freqs_r, mags_r, phases_r, r_imp, data['fs'], "Right Channel")
        
        if img_l and img_r:
            put_tabs([
                {'title': 'Left Channel', 'content': put_image(img_l)},
                {'title': 'Right Channel', 'content': put_image(img_r)}
            ])
        else:
            put_error("Could not generate plots.")
    except Exception as e:
        put_error(f"Failed to generate plots: {e}")
        
    set_processbar('bar', 1.0)
    put_markdown("### Done! You can close this window.")

if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)
