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
CONFIG_FILE = 'config.json'
FINE_TUNE_LIMIT = 45.0
MAX_SAFE_BOOST = 8.0
VERSION = "v1.4beta"
PROGRAM_NAME = "CamillaFIR"

# --- HELPER: STATUS UPDATE ---
def update_status(msg):
    with use_scope('status_area', clear=True):
        put_text(msg).style('font-weight: bold; color: #4CAF50;')

# --- CONFIG MANAGEMENT ---
def load_config():
    default_conf = {
        'fmt': 'WAV', 'layout': 'Stereo (Single file)', 'fs': 48000, 'taps': 65536,
        'gain': 0.0, 'hc_mode': "Built-in ('not Dr Toole')", 
        'mag_correct': ['Enable Frequency Response Correction'],
        'smoothing_type': 'Psychoacoustic',
        'fdw_cycles': 15,
        'hc_min': 20, 'hc_max': 200, 
        'ref_min': 500, 'ref_max': 2000,
        'max_boost': 5.0,
        'hpf_enable': [], 'hpf_freq': None, 'hpf_slope': 24,
        'xo1_f': None, 'xo1_s': 12, 'xo2_f': None, 'xo2_s': 12,
        'xo3_f': None, 'xo3_s': 12, 'xo4_f': None, 'xo4_s': 12, 'xo5_f': None, 'xo5_s': 12
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved_conf = json.load(f)
                default_conf.update(saved_conf)
        except: pass
    return default_conf

def save_config(data):
    save_data = data.copy()
    for key in ['file_l', 'file_r', 'hc_custom_file']:
        if key in save_data: del save_data[key]
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(save_data, f, indent=4)
    except Exception as e: print(f"Failed to save config: {e}")

# --- DATA PARSING ---
def get_default_house_curve():
    freqs = np.array([20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 20000.0])
    mags = np.array([6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5, 1.4, 0.4, 0.0, -4.0])
    return freqs, mags

def parse_measurements_from_bytes(file_content):
    try:
        f = io.BytesIO(file_content)
        data = np.loadtxt(f, comments=['#', '*'])
        if data.shape[1] < 3: return None, None, None
        return data[:, 0], data[:, 1], data[:, 2] 
    except: return None, None, None

def parse_house_curve_from_bytes(file_content):
    try:
        f = io.BytesIO(file_content)
        data = np.loadtxt(f, comments=['#', '*'])
        return data[:, 0], data[:, 1]
    except: return None, None

# --- DSP FUNCTIONS ---

def calculate_minimum_phase(mags_lin_fft):
    n_fft = (len(mags_lin_fft) - 1) * 2
    log_mag = np.log(np.maximum(mags_lin_fft, 1e-10))
    cepstrum = scipy.fft.irfft(log_mag, n=n_fft)
    window = np.zeros_like(cepstrum)
    window[0] = 1.0
    window[1:n_fft//2] = 2.0
    window[n_fft//2] = 1.0
    cepstrum_mp = cepstrum * window
    min_phase_complex = scipy.fft.rfft(cepstrum_mp)
    return np.angle(min_phase_complex)

def psychoacoustic_smoothing(freqs, mags, oct_bw=1/3.0):
    mags_heavy, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), oct_bw)
    mags_light, _ = apply_smoothing_std(freqs, mags, np.zeros_like(mags), 1/12.0)
    return np.maximum(mags_heavy, mags_light)

def apply_fdw_smoothing(freqs, phases, cycles):
    phase_u = np.unwrap(np.deg2rad(phases))
    oct_width = 2.0 / max(cycles, 1.0) 
    dummy_mags = np.zeros_like(freqs)
    _, smoothed_phase_deg = apply_smoothing_std(freqs, dummy_mags, np.rad2deg(phase_u), oct_width)
    return np.deg2rad(smoothed_phase_deg)

def apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0):
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
    sm_mags = np.convolve(m_padded, window, mode='same')[pad_len:-pad_len]
    p_padded = np.pad(log_phases, (pad_len, pad_len), mode='edge')
    sm_phases = np.convolve(p_padded, window, mode='same')[pad_len:-pad_len]
        
    return np.interp(freqs, log_freqs, sm_mags), np.rad2deg(np.interp(freqs, log_freqs, sm_phases))

def calculate_theoretical_phase(freq_axis, crossovers):
    if not crossovers: return np.zeros_like(freq_axis)
    total_phase_rad = np.zeros_like(freq_axis)
    for xo in crossovers:
        b, a = scipy.signal.butter(xo['order'], 2 * np.pi * xo['freq'], btype='low', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))
    return total_phase_rad

def interpolate_response(input_freqs, input_values, target_freqs):
    return np.interp(target_freqs, input_freqs, input_values)

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

def generate_filter(freqs, raw_mags, raw_phases, crossovers, 
                    phase_c_min, phase_c_max, mag_c_min, mag_c_max,
                    ref_c_min, ref_c_max, 
                    house_freqs, house_mags, fs, num_taps, fine_phase_limit, 
                    max_boost_db, global_gain_db, hpf_settings, enable_mag_correction,
                    smoothing_type='Standard', fdw_cycles=15):
    
    n_fft = num_taps if num_taps % 2 != 0 else num_taps + 1
    nyquist = fs / 2.0
    freq_axis = np.linspace(0, nyquist, n_fft // 2 + 1)
    
    # 1. SMOOTHING
    if smoothing_type == 'Psychoacoustic':
        smoothed_mags = psychoacoustic_smoothing(freqs, raw_mags)
    else:
        smoothed_mags, _ = apply_smoothing_std(freqs, raw_mags, raw_phases, 1/48.0)
    
    meas_mags = interpolate_response(freqs, smoothed_mags, freq_axis)
    
    # 2. PHASE (Minimum & Excess)
    meas_min_phase_rad = calculate_minimum_phase(10**(meas_mags/20.0))
    meas_phase_rad_raw = np.deg2rad(interpolate_response(freqs, raw_phases, freq_axis))
    meas_phase_rad_unwrapped = np.unwrap(meas_phase_rad
