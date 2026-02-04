import numpy as np
import scipy.signal
import scipy.fft


def limit_phase_deg(phase_rad, max_deg=45.0):
    """
    Limit phase to +/- max_deg (degrees).
    Keeps correction "safe" by preventing extreme phase rotations.
    """
    if max_deg is None:
        return phase_rad
    try:
        max_deg = float(max_deg)
    except Exception:
        max_deg = 45.0
    max_rad = np.deg2rad(abs(max_deg))
    return np.clip(phase_rad, -max_rad, max_rad)


def calculate_minimum_phase(mags_lin_fft, max_phase_deg=45.0):
    """Calculates minimum phase using Hilbert transform. 1e-10 protection prevents NaN errors."""
    n_fft = (len(mags_lin_fft) - 1) * 2
    ln_mag = np.log(np.maximum(np.abs(mags_lin_fft), 1e-10))
    full_ln_mag = np.concatenate((ln_mag, ln_mag[-2:0:-1]))
    analytic = scipy.signal.hilbert(full_ln_mag)
    min_phase_rad = -np.imag(analytic)[:len(mags_lin_fft)]
    # unwrap first, then limit
    min_phase_rad = np.unwrap(min_phase_rad)
    return limit_phase_deg(min_phase_rad, max_phase_deg)


def calculate_theoretical_phase(freq_axis, crossovers, hpf_freq=None, hpf_slope=None, max_phase_deg=45.0):
    """Calculates theoretical phase shift caused by crossovers and high-pass filter (HPF)."""
    total_phase_rad = np.zeros_like(freq_axis)

    # 1. Add HPF phase effect using analog Butterworth model
    if hpf_freq and hpf_slope and hpf_freq > 0:
        hpf_order = int(hpf_slope / 6)
        b, a = scipy.signal.butter(hpf_order, 2 * np.pi * hpf_freq, btype="high", analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))

    # 2. Add crossover filter phase
    for xo in crossovers:
        if xo.get("freq") is None:
            continue
        order = xo.get("order", int(xo.get("slope", 12) / 6))
        b, a = scipy.signal.butter(order, 2 * np.pi * xo["freq"], btype="low", analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))

    # unwrap first, then limit
    total_phase_rad = np.unwrap(total_phase_rad)
    return limit_phase_deg(total_phase_rad, max_phase_deg)


def combine_mixed_phase(ir_lin, ir_min, fs, split_freq=300):
    """Combines Linear Phase bass with Minimum Phase treble."""
    ntaps = len(ir_lin)
    fir_lp = scipy.signal.firwin(2047, split_freq, fs=fs, pass_zero=True, window="blackman")
    fir_hp = -fir_lp
    fir_hp[1023] += 1.0

    idx_lin = np.argmax(np.abs(ir_lin))
    idx_min = np.argmax(np.abs(ir_min))
    shift = idx_lin - idx_min
    ir_min_aligned = np.roll(ir_min, shift)

    filt_bass = scipy.signal.fftconvolve(ir_lin, fir_lp, mode="same")
    filt_treble = scipy.signal.fftconvolve(ir_min_aligned, fir_hp, mode="same")
    return filt_bass + filt_treble


def remove_time_of_flight(freq_axis, phase_rad):
    """Find and remove linear phase slope (delay) from measurement."""
    mask = (freq_axis >= 1000) & (freq_axis <= 10000)
    if not np.any(mask):
        return phase_rad, 0.0
    poly = np.polyfit(freq_axis[mask], phase_rad[mask], 1)
    return phase_rad - (poly[0] * freq_axis), poly[0]


def get_min_phase_impulse(mags_db, n_fft):
    """Create minimum-phase impulse response from magnitude response."""
    # Muunnetaan dB -> lineaarinen ja luodaan symmetrinen spektri
    amp = 10 ** (mags_db / 20.0)
    # Hilbert-muunnos vaatii logaritmi-amplitudin
    l_amp = np.log(amp + 1e-12)
    # Calculate minimum phase using FFT and Hilbert principle
    h = scipy.fft.ifft(l_amp)
    n = len(h)
    window = np.zeros(n)
    window[0] = 1
    window[1 : n // 2] = 2
    window[n // 2] = 1
    # Muodostetaan minimivaiheinen vaste
    min_phase = np.exp(scipy.fft.fft(h * window))
    return np.real(scipy.fft.ifft(min_phase))