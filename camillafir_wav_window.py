# camillafir_wav_window.py
# IR WAV -> windowed FFT (freq, mag_db, phase_deg)
# Goal: resemble REW .txt export as closely as practical.

from __future__ import annotations
import numpy as np

try:
    import scipy.signal
except Exception:  # pragma: no cover
    scipy = None  # type: ignore


def _tukey_window(n: int, alpha: float = 0.25) -> np.ndarray:
    """
    Tukey window (tapered cosine). alpha=0 -> rectangular, alpha=1 -> Hann.
    Uses scipy if available; otherwise a small local implementation.
    """
    n = int(n)
    if n <= 1:
        return np.ones(max(n, 1), dtype=np.float32)

    a = float(alpha)
    a = 0.0 if a < 0.0 else (1.0 if a > 1.0 else a)

    if scipy is not None:
        try:
            return scipy.signal.windows.tukey(n, alpha=a).astype(np.float32)
        except Exception:
            pass

    # Fallback implementation
    w = np.ones(n, dtype=np.float32)
    if a <= 0.0:
        return w
    if a >= 1.0:
        # Hann
        return np.hanning(n).astype(np.float32)

    t = np.linspace(0.0, 1.0, n, endpoint=True, dtype=np.float64)
    edge = a / 2.0
    # rising cosine
    m1 = t < edge
    w[m1] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * t[m1] / a - 1.0)))
    # falling cosine
    m2 = t > (1.0 - edge)
    w[m2] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * t[m2] / a - 2.0 / a + 1.0)))
    return w.astype(np.float32)


def _next_pow2(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _octave_smooth_loggrid(freqs_hz: np.ndarray, mags_db: np.ndarray, smoothing_level: int) -> np.ndarray:
    """
    Same idea as camillafir.py _octave_smooth_loggrid:
    smooth in log-frequency with a gaussian kernel.
    (Kept local to avoid imports/refactor.)
    """
    f = np.asarray(freqs_hz, dtype=float)
    m = np.asarray(mags_db, dtype=float)
    sl = int(smoothing_level)

    if sl <= 0 or f.size != m.size or f.size < 64:
        return m.astype(float)

    # avoid f=0 for log
    mask = f > 0.0
    if np.count_nonzero(mask) < 64:
        return m.astype(float)

    ff = f[mask]
    mm = m[mask]
    logf = np.log10(ff)

    # uniform log grid
    n = ff.size
    grid = np.linspace(logf.min(), logf.max(), n)
    mg = np.interp(grid, logf, mm)

    # sigma in points: tie to octave smoothing level (1/sl oct)
    # Roughly: heavier smoothing for smaller sl. Keep stable + simple.
    # (This mirrors the intent of the existing implementation.)
    sigma_pts = max(1.0, (n / 2000.0) * (24.0 / max(float(sl), 1.0)))

    half = int(max(3, round(4.0 * sigma_pts)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_pts) ** 2)
    k /= np.sum(k)

    mg_s = np.convolve(mg, k, mode="same")
    mm_s = np.interp(logf, grid, mg_s)

    out = m.copy()
    out[mask] = mm_s
    return out.astype(float)


def ir_wav_to_freq_response(
    fs: int,
    x: np.ndarray,
    *,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
    window: str = "tukey",
    tukey_alpha: float = 0.25,
    zero_pad_pow2: bool = True,
    detrend: str = "linear",
    phase_hf_hold: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert impulse response WAV (time domain) to (freq, mag_db, phase_deg).

    Defaults are chosen to resemble REW IR-windowed export better than plain Hann:
      - asymmetric crop around peak (pre_ms/post_ms)
      - DC removal + optional linear detrend
      - Tukey window with modest taper
      - zero-pad to next pow2 for smoother spectrum
      - unwrap phase
    """
    fs_i = int(fs) if fs else 0
    if fs_i <= 0:
        raise ValueError("Invalid WAV sample rate.")

    sig = np.asarray(x, dtype=np.float32).copy()
    if sig.size < 64:
        raise ValueError("WAV too short.")

    # DC removal
    sig -= float(np.mean(sig))

    # Find peak (absolute)
    peak = int(np.argmax(np.abs(sig)))

    pre_s = int(round((float(pre_ms) / 1000.0) * fs_i))
    post_s = int(round((float(post_ms) / 1000.0) * fs_i))
    pre_s = max(pre_s, 0)
    post_s = max(post_s, 64)

    i0 = max(0, peak - pre_s)
    i1 = min(sig.size, peak + post_s)
    seg = sig[i0:i1]
    if seg.size < 64:
        seg = sig

    # Detrend (helps match REW baseline handling)
    dt = (detrend or "").lower().strip()
    if dt == "linear":
        try:
            seg = seg - np.linspace(float(seg[0]), float(seg[-1]), seg.size, dtype=np.float32)
        except Exception:
            pass
    elif dt == "mean":
        seg = seg - float(np.mean(seg))

    # Windowing
    wtype = (window or "").lower().strip()
    try:
        if wtype == "hann" or wtype == "hanning":
            w = np.hanning(seg.size).astype(np.float32)
        elif wtype == "rect" or wtype == "rectangular" or wtype == "none":
            w = np.ones(seg.size, dtype=np.float32)
        else:
            w = _tukey_window(seg.size, alpha=float(tukey_alpha))
        seg = seg * w
    except Exception:
        pass

    # Zero pad
    n_fft = int(seg.size)
    if zero_pad_pow2:
        n_fft = _next_pow2(n_fft)
    if n_fft < seg.size:
        n_fft = seg.size

    # FFT
    spec = np.fft.rfft(seg, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs_i))

    mag = np.abs(spec)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    phase_rad = np.unwrap(np.angle(spec))
    phase_deg = np.rad2deg(phase_rad)

    # Optional octave smoothing on magnitude (keeps behavior consistent with TXT path)
    if smoothing_level is not None:
        try:
            sl = int(smoothing_level)
            if sl > 0:
                mag_db = _octave_smooth_loggrid(freqs, mag_db, sl)
        except Exception:
            pass

    # Optional: hold HF phase constant (matches your existing safety behavior)
    if phase_hf_hold:
        try:
            hf = freqs > min(0.45 * fs_i, 18000.0)
            if np.any(hf) and np.any(~hf):
                phase_deg[hf] = phase_deg[np.where(~hf)[0][-1]]
        except Exception:
            pass

    return freqs.astype(float), mag_db.astype(float), phase_deg.astype(float)
# End of camillafir_wav_window.py