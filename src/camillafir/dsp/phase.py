import numpy as np
import scipy.signal
import scipy.fft


def limit_phase_deg(phase_rad, max_deg=45.0):
    """
    Rajaa vaiheen arvoalueen asteina annettuun maksimiin.

    Syote oletetaan radiaaneina ja palautus on radiaaneina. Jos `max_deg` on
    `None`, vaihetta ei rajata.
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
    """
    Laskee minimivaiheen amplitudispektrista Hilbert-muunnoksen avulla.

    Lopputulos unwrapataan ja rajataan turvalliseen vaihealueeseen
    `limit_phase_deg()`-funktion kautta.
    """
    n_fft = (len(mags_lin_fft) - 1) * 2
    ln_mag = np.log(np.maximum(np.abs(mags_lin_fft), 1e-10))
    full_ln_mag = np.concatenate((ln_mag, ln_mag[-2:0:-1]))
    analytic = scipy.signal.hilbert(full_ln_mag)
    min_phase_rad = -np.imag(analytic)[:len(mags_lin_fft)]
    min_phase_rad = np.unwrap(min_phase_rad)
    return limit_phase_deg(min_phase_rad, max_phase_deg)


def calculate_theoretical_phase(freq_axis, crossovers, hpf_freq=None, hpf_slope=None, max_phase_deg=45.0):
    """
    Laskee teoreettisen vaiheen jakosuodattimista ja mahdollisesta HPF:sta.

    Jokainen XO- ja HPF-vaihe mallinnetaan analogisena Butterworth-suodattimena,
    summataan yhteen, unwrapataan ja rajataan lopuksi aste-rajaan.
    """
    total_phase_rad = np.zeros_like(freq_axis)

    if hpf_freq and hpf_slope and hpf_freq > 0:
        hpf_order = int(hpf_slope / 6)
        b, a = scipy.signal.butter(hpf_order, 2 * np.pi * hpf_freq, btype="high", analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))

    for xo in crossovers:
        if xo.get("freq") is None:
            continue
        order = xo.get("order", int(xo.get("slope", 12) / 6))
        b, a = scipy.signal.butter(order, 2 * np.pi * xo["freq"], btype="low", analog=True)
        w, h = scipy.signal.freqs(b, a, worN=2 * np.pi * freq_axis)
        total_phase_rad += np.unwrap(np.angle(h))

    total_phase_rad = np.unwrap(total_phase_rad)
    return limit_phase_deg(total_phase_rad, max_phase_deg)


def _shift_zeropad(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Siirtaa 1D-signaalia kokonaisilla naytteilla nollataytolla.

    Positiivinen siirto liikuttaa signaalia oikealle, negatiivinen vasemmalle.
    Ei kayta kiertavaa wrap-around-kaytosta.
    """
    x = np.asarray(x)
    y = np.zeros_like(x)
    if shift == 0:
        return x.copy()
    if shift > 0:
        y[shift:] = x[:-shift]
    else:
        s = -shift
        y[:-s] = x[s:]
    return y

def _raised_cosine_lp(freqs: np.ndarray, f0: float, f1: float) -> np.ndarray:
    """
    Muodostaa raised-cosine -alipaasopainon valille [f0, f1].

    Paino on 1 alle f0, 0 yli f1 ja pehmea kosinisiirtyma niiden valissa.
    """
    w = np.ones_like(freqs, dtype=float)
    w[freqs >= f1] = 0.0
    mid = (freqs > f0) & (freqs < f1)
    if np.any(mid):
        x = (freqs[mid] - f0) / (f1 - f0)
        w[mid] = 0.5 * (1.0 + np.cos(np.pi * x))
    return w

def combine_mixed_phase(ir_lin, ir_min, fs, split_freq=120.0, transition_hz=120.0):
    """
    Yhdistaa lineaarivaiheisen ja minimivaiheisen impulssivasteen.

    Menetelma kohdistaa ensin huiput aikatasossa, sekoittaa sitten vaihe- ja
    amplituditiedon taajuustasossa `split_freq`-kohdan ymparilla ja palauttaa
    yhdistetyn aikatasoisen IR:n.
    """
    ir_lin = np.asarray(ir_lin, dtype=float)
    ir_min = np.asarray(ir_min, dtype=float)

    n = len(ir_lin)
    if len(ir_min) != n:
        raise ValueError("ir_lin and ir_min must have the same length")

    if n < 8:
        return ir_lin.copy()

    idx_lin = int(np.argmax(np.abs(ir_lin)))
    idx_min = int(np.argmax(np.abs(ir_min)))
    shift = idx_lin - idx_min
    ir_min_aligned = _shift_zeropad(ir_min, shift)

    H_lin = np.fft.rfft(ir_lin)
    H_min = np.fft.rfft(ir_min_aligned)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs))

    transition_hz = float(transition_hz)
    if transition_hz <= 0:
        W_lp = (freqs <= float(split_freq)).astype(float)
    else:
        f0 = max(0.0, float(split_freq) - transition_hz / 2.0)
        f1 = float(split_freq) + transition_hz / 2.0
        W_lp = _raised_cosine_lp(freqs, f0, f1)

    W_hp = 1.0 - W_lp

    phi_lin = np.unwrap(np.angle(H_lin))
    phi_min = np.unwrap(np.angle(H_min))
    phi = (W_lp * phi_lin) + (W_hp * phi_min)

    mag_lin = np.abs(H_lin)
    mag_min = np.abs(H_min)
    mag = np.maximum((W_lp * mag_lin) + (W_hp * mag_min), 1e-12)
    H = mag * np.exp(1j * phi)

    ir = np.fft.irfft(H, n=n)

    return ir
    


def remove_time_of_flight(freq_axis, phase_rad):
    """
    Poistaa mittauksesta lineaarisen viivekomponentin vaiheesta.

    Viive arvioidaan lineaarisella sovitteella alueelta 1-10 kHz ja vahennetaan
    koko vaihekayrasta.
    """
    mask = (freq_axis >= 1000) & (freq_axis <= 10000)
    if not np.any(mask):
        return phase_rad, 0.0
    poly = np.polyfit(freq_axis[mask], phase_rad[mask], 1)
    return phase_rad - (poly[0] * freq_axis), poly[0]


def get_min_phase_impulse(mags_db, n_fft):
    """
    Muodostaa minimivaiheisen impulssivasteen dB-amplitudista.

    Toteutus kayttaa kepstraalimenetelmaa (log-amplitudi -> IFFT/FFT -> IFFT).
    Parametri `n_fft` sailyy rajapinnassa yhteensopivuuden vuoksi.
    """
    amp = 10 ** (mags_db / 20.0)
    l_amp = np.log(amp + 1e-12)
    h = scipy.fft.ifft(l_amp)
    n = len(h)
    window = np.zeros(n)
    window[0] = 1
    window[1 : n // 2] = 2
    window[n // 2] = 1
    min_phase = np.exp(scipy.fft.fft(h * window))
    return np.real(scipy.fft.ifft(min_phase))
