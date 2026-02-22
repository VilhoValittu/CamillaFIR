# camillafir_io/measurements_wav.py
import io
import os
import numpy as np
import scipy.io.wavfile

try:
    # Preferred: local package import.
    from .camillafir_wav_window import ir_wav_to_freq_response as _wav_ir_to_fr
except Exception:
    try:
        # Fallback for some script execution contexts.
        from src.camillafir.io.camillafir_wav_window import ir_wav_to_freq_response as _wav_ir_to_fr
    except Exception:
        _wav_ir_to_fr = None

# TXT-baseline compatibility policy:
# WAV input parsing should stay deterministic and not depend on user UI knobs.
_TXT_BASELINE_PRE_MS = 120.0
_TXT_BASELINE_POST_MS = 500.0
_TXT_BASELINE_SMOOTHING = 0
_TXT_BASELINE_BIN_HZ_MAX = 0.4


def _txt_baseline_params():
    return float(_TXT_BASELINE_PRE_MS), float(_TXT_BASELINE_POST_MS), int(_TXT_BASELINE_SMOOTHING)


def _txt_baseline_min_n_fft(fs: int) -> int:
    fs_i = int(fs) if fs else 0
    if fs_i <= 0:
        return 131072
    target = int(np.ceil(float(fs_i) / float(_TXT_BASELINE_BIN_HZ_MAX)))
    n = 1 << (max(1, target) - 1).bit_length()
    return int(max(131072, n))


def _wav_to_float(sig: np.ndarray) -> np.ndarray:
    x = np.asarray(sig)
    if x.dtype.kind == "f":
        return x.astype(np.float32, copy=False)
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0)
    if x.dtype == np.int32:
        return (x.astype(np.float32) / 2147483648.0)
    return x.astype(np.float32)


def _octave_smooth_loggrid(freqs: np.ndarray, mags_db: np.ndarray, smoothing_level: int) -> np.ndarray:
    try:
        f = np.asarray(freqs, dtype=float)
        m = np.asarray(mags_db, dtype=float)
        if f.size < 8 or m.size != f.size:
            return m

        N = int(smoothing_level)
        if N <= 0:
            return m

        mask = f > 0
        if np.count_nonzero(mask) < 8:
            return m

        f2 = f[mask]
        m2 = m[mask]

        logf = np.log2(f2)
        step = 1.0 / 96.0
        g0, g1 = float(logf[0]), float(logf[-1])
        if g1 <= g0 + step:
            return m

        grid = np.arange(g0, g1 + step, step, dtype=float)
        mg = np.interp(grid, logf, m2)

        fwhm_oct = 1.0 / float(N)
        sigma_oct = fwhm_oct / 2.355
        sigma_pts = max(1.0, sigma_oct / step)

        half = int(max(3, round(4.0 * sigma_pts)))
        x = np.arange(-half, half + 1, dtype=float)
        k = np.exp(-0.5 * (x / sigma_pts) ** 2)
        k /= np.sum(k)

        mg_s = np.convolve(mg, k, mode="same")
        m2_s = np.interp(logf, grid, mg_s)

        out = m.copy()
        out[mask] = m2_s
        return out
    except Exception:
        return np.asarray(mags_db, dtype=float)


def _ir_wav_to_freq_response(
    fs: int,
    x: np.ndarray,
    *,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
):
    fs_i = int(fs) if fs else 0
    if fs_i <= 0:
        raise ValueError("Invalid WAV sample rate.")

    sig = np.asarray(x, dtype=np.float32).copy()
    if sig.size < 64:
        raise ValueError("WAV too short.")

    sig -= float(np.mean(sig))
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

    n_fft = 1 << (int(seg.size) - 1).bit_length()
    if n_fft < seg.size:
        n_fft = int(seg.size)
    mn = _txt_baseline_min_n_fft(fs_i)
    if n_fft < mn:
        n_fft = int(mn)

    spec = np.fft.rfft(seg, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(fs_i))
    mag = np.abs(spec)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    phase_rad = np.unwrap(np.angle(spec))
    phase_deg = np.rad2deg(phase_rad)

    if smoothing_level is not None:
        try:
            sl = int(smoothing_level)
            if sl > 0:
                mag_db = _octave_smooth_loggrid(freqs, mag_db, sl)
        except Exception:
            pass

    hf = freqs > min(0.45 * fs_i, 18000.0)
    if np.any(hf) and np.any(~hf):
        phase_deg[hf] = phase_deg[np.where(~hf)[0][-1]]

    return freqs.astype(float), mag_db.astype(float), phase_deg.astype(float)


def parse_measurements_from_wav_bytes(
    file_content: bytes,
    *,
    channel_index: int = 0,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
    logger=None,
):
    try:
        bio = io.BytesIO(file_content)
        fs, sig = scipy.io.wavfile.read(bio)
        sig = _wav_to_float(sig)

        if sig.ndim == 2:
            ch = int(channel_index)
            ch = 0 if ch < 0 else ch
            ch = (sig.shape[1] - 1) if ch >= sig.shape[1] else ch
            sig = sig[:, ch]

        # Keep WAV->FR conversion aligned to TXT baseline (no user-tunable parse knobs).
        pre_ms, post_ms, smoothing_level = _txt_baseline_params()
        min_n_fft = _txt_baseline_min_n_fft(int(fs))

        if _wav_ir_to_fr is not None:
            return _wav_ir_to_fr(
                int(fs),
                sig,
                pre_ms=float(pre_ms),
                post_ms=float(post_ms),
                smoothing_level=smoothing_level,
                window="none",
                detrend="none",
                zero_pad_pow2=True,
                min_n_fft=int(min_n_fft),
                phase_hf_hold=True,
            )
        return _ir_wav_to_freq_response(int(fs), sig, pre_ms=float(pre_ms), post_ms=float(post_ms), smoothing_level=smoothing_level)
    except Exception as e:
        if logger:
            logger.error(f"WAV parse failed: {e}")
        return None, None, None


def parse_measurements_from_wav_path(
    path: str,
    *,
    channel_index: int = 0,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
    logger=None,
):
    try:
        p = str(path or "").strip().strip('"').strip("'")
        if not p:
            return None, None, None
        if not os.path.exists(p):
            if logger:
                logger.error(f"WAV file not found: {p}")
            return None, None, None

        fs, sig = scipy.io.wavfile.read(p)
        sig = _wav_to_float(sig)

        if sig.ndim == 2:
            ch = int(channel_index)
            ch = 0 if ch < 0 else ch
            ch = (sig.shape[1] - 1) if ch >= sig.shape[1] else ch
            sig = sig[:, ch]

        # Keep WAV->FR conversion aligned to TXT baseline (no user-tunable parse knobs).
        pre_ms, post_ms, smoothing_level = _txt_baseline_params()
        min_n_fft = _txt_baseline_min_n_fft(int(fs))

        if _wav_ir_to_fr is not None:
            return _wav_ir_to_fr(
                int(fs),
                sig,
                pre_ms=float(pre_ms),
                post_ms=float(post_ms),
                smoothing_level=smoothing_level,
                window="none",
                detrend="none",
                zero_pad_pow2=True,
                min_n_fft=int(min_n_fft),
                phase_hf_hold=True,
            )
        return _ir_wav_to_freq_response(int(fs), sig, pre_ms=float(pre_ms), post_ms=float(post_ms), smoothing_level=smoothing_level)
    except Exception as e:
        if logger:
            logger.error(f"WAV path parse failed ({path}): {e}")
        return None, None, None
