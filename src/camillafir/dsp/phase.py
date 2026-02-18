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


def _shift_zeropad(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift by integer samples with zero padding (no wrap-around)."""
    x = np.asarray(x)
    y = np.zeros_like(x)
    if shift == 0:
        return x.copy()
    if shift > 0:
        # move right
        y[shift:] = x[:-shift]
    else:
        # move left
        s = -shift
        y[:-s] = x[s:]
    return y

def _raised_cosine_lp(freqs: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """
    Low-pass weight:
      1 below f0
      0 above f1
      raised-cosine transition between [f0, f1]
    """
    w = np.ones_like(freqs, dtype=float)
    w[freqs >= f1] = 0.0
    mid = (freqs > f0) & (freqs < f1)
    if np.any(mid):
        x = (freqs[mid] - f0) / (f1 - f0)
        w[mid] = 0.5 * (1.0 + np.cos(np.pi * x))  # 1 -> 0
    return w

def combine_mixed_phase(ir_lin, ir_min, fs, split_freq=120.0, transition_hz=120.0):
    """
    Combine linear-phase bass with minimum-phase treble.

    New implementation (replaces old):
    - Frequency-domain blending (no extra crossover FIR convolution)
    - Smooth transition band around split_freq
    - Proper peak alignment without circular wrap-around
    """
    ir_lin = np.asarray(ir_lin, dtype=float)
    ir_min = np.asarray(ir_min, dtype=float)

    n = len(ir_lin)
    if len(ir_min) != n:
        raise ValueError("ir_lin and ir_min must have the same length")

    if n < 8:
        return ir_lin.copy()

    # Align peaks (integer) WITHOUT wrap-around
    idx_lin = int(np.argmax(np.abs(ir_lin)))
    idx_min = int(np.argmax(np.abs(ir_min)))
    shift = idx_lin - idx_min
    ir_min_aligned = _shift_zeropad(ir_min, shift)

    # FFTs
    H_lin = np.fft.rfft(ir_lin)
    H_min = np.fft.rfft(ir_min_aligned)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    # Smooth blend band
    transition_hz = float(transition_hz)
    if transition_hz <= 0:
        # hard split (still frequency-domain, no extra FIR)
        W_lp = (freqs <= float(split_freq)).astype(float)
    else:
        f0 = max(0.0, float(split_freq) - transition_hz / 2.0)
        f1 = float(split_freq) + transition_hz / 2.0
        W_lp = _raised_cosine_lp(freqs, f0, f1)

    W_hp = 1.0 - W_lp

    # Phase-only style blend:
    # - blend phase trajectories directly (avoids complex-vector cancellation notches)
    # - blend magnitudes separately for smooth handover
    phi_lin = np.unwrap(np.angle(H_lin))
    phi_min = np.unwrap(np.angle(H_min))
    phi = (W_lp * phi_lin) + (W_hp * phi_min)

    mag_lin = np.abs(H_lin)
    mag_min = np.abs(H_min)
    mag = np.maximum((W_lp * mag_lin) + (W_hp * mag_min), 1e-12)
    H = mag * np.exp(1j * phi)

    # Back to time domain
    ir = np.fft.irfft(H, n=n)

    return ir
    


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
