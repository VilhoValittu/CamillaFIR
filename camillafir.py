import io
import json
import locale
import logging
import os
import sys
import zipfile
from datetime import datetime
from textwrap import dedent

import numpy as np
import scipy.io.wavfile
from pywebio import config, start_server
from pywebio.input import *
from pywebio.output import *
from pywebio.pin import *
from pywebio.session import set_env

import camillafir_dsp as dsp
import camillafir_plot as plots
import models
from models import FilterConfig
#from camillafir_rew_api import *
print("USING models.py      =", models.__file__)
print("USING camillafir_dsp =", dsp.__file__)
print("USING camillafir_plot=", plots.__file__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("CamillaFIR")

CONFIG_FILE = 'config.json'
TRANS_FILE = 'translations.json'

VERSION = "v2.8.0"  # fix custom house curve upload (would silently fail)

# v2.8.0: [UI] removed html dashboard export (now PNG only)
# v2.7.9: [UI] fix custom house curve upload
# v2.7.8: [IO] fix WAV parsing – phase unwrap
# v2.7.7: [DSP] fix HF phase handling
# v2.7.6: [IO] fix WAV parsing smoothing

PROGRAM_NAME = "CamillaFIR"
FINE_TUNE_LIMIT = 45.0
MAX_SAFE_BOOST = 8.0
FORCE_SINGLE_PLOT_FS_HZ = 48000




def scale_taps_with_fs(
    fs: int,
    base_fs: int = 44100,
    base_taps: int = 65536,
    allowed_taps=(
        512, 1024, 2048, 4096, 8192, 16384,
        32768, 65536, 131072, 262144, 524288,
        1048576
   ),
) -> int:
    """Scale FIR taps with sample rate so that filter *time length* stays constant.

    Reference: 44.1 kHz -> 65 536 taps.
    For other rates, choose the next allowed taps value >= scaled target.
    """
    try:
        fs_i = int(fs)
        if fs_i <= 0:
            return int(base_taps)
        target = float(base_taps) * (float(fs_i) / float(base_fs))
        for t in allowed_taps:
            if int(t) >= target:
                return int(t)
        return int(allowed_taps[-1])
    except Exception:
        return int(base_taps)
    
def update_taps_auto_info(_=None):
    """
    UI helper: show Auto-taps mapping when multi-rate is enabled.
    Uses reference 44.1kHz -> 65536 taps (constant time-length).
    """
    # put_checkbox() -> pin['multi_rate_opt'] on lista valituista arvoista:
    # [] = off, [True] = on (tässä projektissa)
    try:
        mr = bool(pin['multi_rate_opt'])
    except Exception:
        mr = False

    for scope_name in ('taps_auto_info_scope_files', 'taps_auto_info_scope_basic'):
        with use_scope(scope_name, clear=True):
            if not mr:
                # Näytä jotain myös OFF-tilassa, jotta käyttäjä näkee että UI oikeasti päivittyy
                put_markdown(f"_{t('auto_taps_title')}: OFF_")
                continue

            rates = [44100, 48000, 88200, 96000, 176400, 192000]
            lines = [f"- **{r/1000:.1f} kHz** → **{scale_taps_with_fs(r)}** taps" for r in rates]

            put_markdown(
                f"### {t('auto_taps_title')}\n"
                f"{t('auto_taps_body')}\n\n"
                f"{t('auto_taps_ref')}\n\n"
                + "\n".join(lines)
            )


def get_resource_path(relative_path):
    """ Palauttaa polun resurssiin, oli se EXE-paketin sisällä tai kehityskoneella. """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
TRANS_FILE = get_resource_path('translations.json')


def parse_measurements_from_path(path):
    """Lukee mittausdatan paikallisesta tiedostopolusta (REW .txt export) robustisti."""
    try:
        if not path: return None, None, None

        # 1. Siivotaan polku (poistetaan lainausmerkit ja välilyönnit)
        p = path.strip().strip('"').strip("'")
        
        if not os.path.exists(p):
            logger.error(f"Tiedostoa ei löydy: {p}")
            return None, None, None
            
        # WAV import: if local path ends with .wav -> parse IR wav to FR
        ext = os.path.splitext(p)[1].lower()
        if ext == ".wav":
            # Use UI IR windows + smoothing for WAV parsing
            try:
                pre_ms = float(getattr(pin, "ir_window_ms_left", None) or pin.get("ir_window_ms_left") or 50.0)
            except Exception:
                pre_ms = 50.0
            try:
                post_ms = float(getattr(pin, "ir_window_ms", None) or pin.get("ir_window_ms") or 500.0)
            except Exception:
                post_ms = 500.0
            try:
                sl = int(getattr(pin, "smoothing_level", None) or pin.get("smoothing_level") or 0)
            except Exception:
                sl = 0
            return parse_measurements_from_wav_path(
                p,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=sl
            )

        # 2. Avataan tiedosto (TXT)
        with open(p, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        freqs, mags, phases = [], [], []

        for line in lines:
            line = line.strip()
            
            # Ohitetaan kommentit ja tyhjät
            if not line or line.startswith(('*', '#', ';')):
                continue
            
            # Ohitetaan rivit, jotka eivät ala numerolla
            if not line[0].isdigit() and line[0] != '-':
                continue

            # --- ÄLYKÄS EROTTIMEN TUNNISTUS ---
            # Jos rivillä on sekä piste että pilkku (esim. "0.36, 41.8"), pilkku on erotin -> vaihdetaan väliksi
            if ',' in line and '.' in line:
                line = line.replace(',', ' ')
            else:
                # Jos rivillä on vain pilkkuja, ne ovat todennäköisesti desimaaleja (Suomi) -> vaihdetaan pisteeksi
                line = line.replace(',', '.')
            # ----------------------------------

            parts = line.split()

            if len(parts) >= 2:
                try:
                    f_val = float(parts[0])
                    m_val = float(parts[1])
                    p_val = float(parts[2]) if len(parts) > 2 else 0.0
                    
                    freqs.append(f_val)
                    mags.append(m_val)
                    phases.append(p_val)
                except ValueError:
                    continue 
        
        if len(freqs) == 0:
            logger.warning(f"Tiedostosta {p} ei löytynyt dataa.")
            return None, None, None
            
        return np.array(freqs), np.array(mags), np.array(phases)

    except Exception as e:
        logger.error(f"Kriittinen virhe polun luvussa ({path}): {e}")
        return None, None, None
    
def _wav_to_float(sig: np.ndarray) -> np.ndarray:
    """
    Convert WAV PCM/float arrays into float32 [-1..1] approx.
    Handles int16/int32/float32/float64.
    """
    x = np.asarray(sig)
    if x.dtype.kind == "f":
        return x.astype(np.float32, copy=False)
    # PCM
    if x.dtype == np.int16:
        return (x.astype(np.float32) / 32768.0)
    if x.dtype == np.int32:
        # int32 WAV often uses full scale; map to [-1..1]
        return (x.astype(np.float32) / 2147483648.0)
    # fallback
    return x.astype(np.float32)



def _octave_smooth_loggrid(freqs: np.ndarray, mags_db: np.ndarray, smoothing_level: int) -> np.ndarray:
    """
    Apply ~1/N-oct smoothing on a uniform log2 frequency grid, then resample back.
    smoothing_level = N (e.g. 12 -> 1/12 octave).
    """
    try:
        f = np.asarray(freqs, dtype=float)
        m = np.asarray(mags_db, dtype=float)
        if f.size < 8 or m.size != f.size:
            return m

        N = int(smoothing_level)
        if N <= 0:
            return m

        # valid positive freqs only
        mask = f > 0
        if np.count_nonzero(mask) < 8:
            return m

        f2 = f[mask]
        m2 = m[mask]

        logf = np.log2(f2)
        # choose a reasonably fine grid in octaves
        step = 1.0 / 96.0  # 1/96 oct grid
        g0, g1 = float(logf[0]), float(logf[-1])
        if g1 <= g0 + step:
            return m

        grid = np.arange(g0, g1 + step, step, dtype=float)
        # interpolate mags onto log grid
        mg = np.interp(grid, logf, m2)

        # Gaussian smoothing with FWHM = 1/N oct -> sigma = fwhm/2.355
        fwhm_oct = 1.0 / float(N)
        sigma_oct = fwhm_oct / 2.355
        sigma_pts = max(1.0, sigma_oct / step)

        # build gaussian kernel
        half = int(max(3, round(4.0 * sigma_pts)))
        x = np.arange(-half, half + 1, dtype=float)
        k = np.exp(-0.5 * (x / sigma_pts) ** 2)
        k /= np.sum(k)

        mg_s = np.convolve(mg, k, mode="same")
        # resample back to original freqs
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert impulse response WAV (time domain) to (freq, mag_db, phase_deg).
    - Finds peak, windows around it: [peak-pre_ms .. peak+post_ms]
    - Applies Hann window to reduce FFT leakage.
    """
    fs_i = int(fs) if fs else 0
    if fs_i <= 0:
        raise ValueError("Invalid WAV sample rate.")

    sig = np.asarray(x, dtype=np.float32).copy()
    if sig.size < 64:
        raise ValueError("WAV too short.")

    # DC removal (helps phase stability)
    sig -= float(np.mean(sig))

    # Find impulse peak
    peak = int(np.argmax(np.abs(sig)))

    pre_s = int(round((float(pre_ms) / 1000.0) * fs_i))
    post_s = int(round((float(post_ms) / 1000.0) * fs_i))
    pre_s = max(pre_s, 0)
    post_s = max(post_s, 64)

    i0 = max(0, peak - pre_s)
    i1 = min(sig.size, peak + post_s)
    seg = sig[i0:i1]
    if seg.size < 64:
        # fallback: use full signal
        seg = sig

    # Window to reduce leakage
    try:
        w = np.hanning(seg.size).astype(np.float32)
        seg = seg * w
    except Exception:
        pass

    seg -= np.linspace(seg[0], seg[-1], seg.size, dtype=np.float32)
    # FFT
    spec = np.fft.rfft(seg)
    freqs = np.fft.rfftfreq(seg.size, d=1.0 / float(fs_i))
    mag = np.abs(spec)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    # Phase unwrap (critical for stable GD/phase plots and confidence logic)
    phase_rad = np.unwrap(np.angle(spec))
    phase_deg = np.rad2deg(phase_rad)

    # Apply octave smoothing to magnitude if requested (match TXT pipeline behavior)
    if smoothing_level is not None:
        try:
            sl = int(smoothing_level)
            if sl > 0:
                mag_db = _octave_smooth_loggrid(freqs, mag_db, sl)
        except Exception:
            pass
    
        hf = freqs > min(0.45 * fs_i, 18000.0)
        phase_deg[hf] = phase_deg[np.where(~hf)[0][-1]]

    return freqs.astype(float), mag_db.astype(float), phase_deg.astype(float)

    """
    Parse REW IR WAV export -> frequency response arrays.
    Returns freqs (Hz), mags (dB), phases (deg).
    """

try:
    # New: dedicated IR WAV window+FFT helper (closer to REW .txt export)
    from camillafir_wav_window import ir_wav_to_freq_response as _wav_ir_to_fr
except Exception:
    _wav_ir_to_fr = None
    
def parse_measurements_from_wav_bytes(
    file_content: bytes,
    *,
    channel_index: int = 0,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
):
    """
    Parse REW IR WAV export -> frequency response arrays.
    Returns freqs (Hz), mags (dB), phases (deg).
    """
    try:
        bio = io.BytesIO(file_content)
        fs, sig = scipy.io.wavfile.read(bio)
        sig = _wav_to_float(sig)

        if sig.ndim == 2:
            ch = int(channel_index)
            ch = 0 if ch < 0 else ch
            ch = (sig.shape[1] - 1) if ch >= sig.shape[1] else ch
            sig = sig[:, ch]

        if _wav_ir_to_fr is not None:
            f, m, p = _wav_ir_to_fr(
                int(fs),
                sig,
                pre_ms=float(pre_ms),
                post_ms=float(post_ms),
                smoothing_level=smoothing_level,
            )
        else:
            f, m, p = _ir_wav_to_freq_response(
                int(fs),
                sig,
                pre_ms=float(pre_ms),
                post_ms=float(post_ms),
                smoothing_level=smoothing_level,
            )
        return f, m, p
    except Exception as e:
        logger.error(f"WAV parse failed: {e}")
        return None, None, None


def parse_measurements_from_wav_path(
    path: str,
    *,
    channel_index: int = 0,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
):
    """
    Parse local IR WAV path -> frequency response arrays.
    """
    try:
        fs, sig = scipy.io.wavfile.read(path)
        sig = _wav_to_float(sig)
        if sig.ndim == 2:
            ch = int(channel_index)
            ch = 0 if ch < 0 else ch
            ch = (sig.shape[1] - 1) if ch >= sig.shape[1] else ch
            sig = sig[:, ch]

        if _wav_ir_to_fr is not None:
            f, m, p = _wav_ir_to_fr(
                int(fs),
                sig,
                pre_ms=float(pre_ms),
                post_ms=float(post_ms),
                smoothing_level=smoothing_level,
            )
        else:
            f, m, p = _ir_wav_to_freq_response(
                int(fs),
                sig,
                pre_ms=float(pre_ms),
                post_ms=float(post_ms),
                smoothing_level=smoothing_level,
            )
        return f, m, p
    except Exception as e:
        logger.error(f"WAV path parse failed ({path}): {e}")
        return None, None, None



def parse_measurements_from_upload(
    file_dict,
    *,
    channel_index: int = 0,
    pre_ms: float = 5.0,
    post_ms: float = 500.0,
    smoothing_level: int | None = None,
):
    try:
        if not file_dict:
            return None, None, None
        name = str(file_dict.get("filename", "") or "")
        content = file_dict.get("content", None)
        if content is None:
            return None, None, None
        ext = os.path.splitext(name)[1].lower()
        if ext == ".wav":
            return parse_measurements_from_wav_bytes(
                content,
                channel_index=channel_index,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
            )
        # fallback: try wav by header "RIFF"
        if isinstance(content, (bytes, bytearray)) and len(content) >= 4 and content[:4] == b"RIFF":
            return parse_measurements_from_wav_bytes(
                content,
                channel_index=channel_index,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
            )
        return parse_measurements_from_bytes(content)
    except Exception:
        return None, None, None


def load_translations():
    try:
        with open(TRANS_FILE, 'r', encoding='utf-8') as f: return json.load(f)
    except: return {}

TRANSLATIONS = load_translations()

def t(key):
    lang = locale.getlocale()[0]
    lang = 'fi' if lang and 'fi' in lang.lower() else 'en'
    if key == 'zoom_hint':
        return "(Vinkki: Voit zoomata hiirellä kuvaajaa)" if lang == 'fi' else "(Hint: Use mouse to zoom)"
    if key == 'lvl_algo_help':
        return "Mediaani on suositeltu: se on immuuni huonemoodeille. Keskiarvo sopii kaiuttimen lähimittauksiin." if lang == 'fi' else "Median is recommended."
    return TRANSLATIONS.get(lang, TRANSLATIONS.get('en', {})).get(key, key)

def update_status(msg):
    with use_scope('status_area', clear=True):
        put_text(msg).style('font-weight: bold; color: #4CAF50; margin-bottom: 10px;')

def parse_measurements_from_bytes(file_content):
    """Lukee mittausdatan (REW export) tavuista robustisti."""
    try:
        # Dekoodataan tavut tekstiksi
        content_str = file_content.decode('utf-8', errors='ignore')
        lines = content_str.split('\n')
        
        freqs, mags, phases = [], [], []
        
        for line in lines:
            line = line.strip()
            # Ohitetaan tyhjät rivit ja kommentit (* tai #)
            if not line or line.startswith(('*', '#')):
                continue
            
            # Yritetään tunnistaa onko rivi dataa (alkaa numerolla)
            if not line[0].isdigit() and line[0] != '-':
                continue
                
            # Korvataan pilkku pisteellä (Suomi-yhteensopivuus)
            line = line.replace(',', '.')
            
            # Pilkotaan välilyöntien/tabulaattorien perusteella
            parts = line.split()
            
            if len(parts) >= 3:
                try:
                    f = float(parts[0])
                    m = float(parts[1])
                    p = float(parts[2])
                    
                    freqs.append(f)
                    mags.append(m)
                    phases.append(p)
                except ValueError:
                    continue
        
        if len(freqs) == 0:
            return None, None, None
            
        return np.array(freqs), np.array(mags), np.array(phases)

    except Exception as e:
        print(f"Error parsing file: {e}")
        return None, None, None

# clear() voi olla output.clear, mutta use_scope(clear=True) hoitaa sen

def update_lvl_ui(_=None):
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        mode = str(_p('lvl_mode', 'Auto') or 'Auto')
        is_manual = ('Manual' in mode)

        # Sanity: lvl_min <= lvl_max
        vmin = float(_p('lvl_min', 500.0) or 500.0)
        vmax = float(_p('lvl_max', 2000.0) or 2000.0)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
            pin_update('lvl_min', value=vmin)
            pin_update('lvl_max', value=vmax)
        
        # lvl_manual_db: näytä AINA, mutta Auto-tilassa harmaana + ei-interaktiivinen
        with use_scope('lvl_manual_scope', clear=True):
            w = put_input(
                'lvl_manual_db',
                label=t('lvl_target_db'),
                type=FLOAT,
                value=float(_p('lvl_manual_db', 75.0) or 75.0),
                help_text=(
                    t('lvl_manual_help')
                    if is_manual
                    else (t('lvl_manual_help'))
                )
            )
            if not is_manual:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")

    except Exception:
        pass

def apply_tdc_preset(name: str):
    """
    PyWebIO preset buttons for TDC knobs.
    Note: put_checkbox stores list values ([] / [True]) in this project.
    """
    presets = {
        "Safe":       {"enable": True, "strength": 35.0, "max_red": 6.0,  "slope": 3.0},
        "Normal":     {"enable": True, "strength": 50.0, "max_red": 9.0,  "slope": 6.0},
        "Aggressive": {"enable": True, "strength": 70.0, "max_red": 12.0, "slope": 0.0},
    }
    p = presets.get(name)
    if not p:
        return

    # enable_tdc is a checkbox -> list convention in this UI: [] / [True]
    pin_update("enable_tdc", value=[True] if p["enable"] else [])
    pin_update("tdc_strength", value=float(p["strength"]))
    pin_update("tdc_max_reduction_db", value=float(p["max_red"]))
    pin_update("tdc_slope_db_per_oct", value=float(p["slope"]))

    # small feedback
    toast(f"TDC preset applied: {name}", color="success", duration=1.5)

def _normalize_hc_mode_key(v) -> str:
    """
    Convert UI label / legacy saved strings into a stable preset key.
    This fixes the bug where translated labels caused preset matching to fall back.
    """
    try:
        s = str(v or "")
    except Exception:
        s = ""

    # Already a valid key?
    known = {
        "Harman6", "Harman8", "Harman4", "Harman10",
        "Studio", "Nearfield", "HiFi", "Speech",
        "Toole", "BK", "Flat", "Cinema", "Custom",
    }
    if s in known:
        return s

    # Legacy / label heuristics (robust against language + spacing)
    n = s.lower().replace(" ", "")
    # "upload"/"custom" options in various languages (keep simple + safe)
    if "custom" in n or "lataa" in n or "upload" in n:
        return "Custom"
    if "cinema" in n:
        return "Cinema"
    if "flat" in n:
        return "Flat"
    if "toole" in n:
        return "Toole"
    if "speech" in n or "broadcast" in n:
        return "Speech"
    if "nearfield" in n or "desk" in n:
        return "Nearfield"
    if "hifi" in n or "loudness" in n:
        return "HiFi"
    if "studio" in n or "tilt" in n:
        return "Studio"
    if "harman" in n:
        if "+10db" in n or "10db" in n or "subheavy" in n:
            return "Harman10"
        if "+8db" in n or "8db" in n:
            return "Harman8"
        if "+4db" in n or "4db" in n:
            return "Harman4"
        return "Harman6"

    # Safe fallback
    return "Harman6"


def get_house_curve_by_name(name):

    # --- Common full-band frequency axis ---
    full_freqs = np.array([
        0.0,
        20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0,
        160.0, 200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0,
        8000.0, 16000.0, 20000.0
    ])

    # --- Harman variants ---
    if 'Harman8' in name or '+8dB' in name:
        freqs = full_freqs
        mags = np.array([
            8.0,
            8.0, 7.9, 7.8, 7.6, 7.3, 6.9, 6.3, 5.5, 4.5,
            3.4, 1.4, 0.0, -0.5, -1.0, -1.8, -2.8,
            -4.0, -5.5, -6.0
        ])

    elif 'Harman4' in name or '+4dB' in name:
        freqs = full_freqs
        mags = np.array([
            4.0,
            4.0, 3.9, 3.8, 3.6, 3.3, 2.9, 2.3, 1.5, 0.8,
            0.2, 0.0, 0.0, -0.3, -0.6, -1.2, -2.0,
            -3.0, -4.5, -5.0
        ])

    elif 'Harman10' in name or 'SubHeavy' in name:
        freqs = full_freqs
        mags = np.array([
            10.0,
            10.0, 9.8, 9.5, 9.0, 8.2, 7.2, 6.0, 4.8, 3.5,
            2.2, 0.8, 0.0, -0.5, -1.0, -1.8, -2.8,
            -4.0, -5.5, -6.0
        ])

    # --- Research / reference ---
    elif 'Toole' in name:
        freqs = np.array([
            0.0,
            20.0, 63.0, 100.0, 200.0, 400.0,
            1000.0, 2000.0, 4000.0, 10000.0, 20000.0
        ])
        mags = np.array([
            2.5,
            2.5, 2.0, 1.5, 1.0, 0.5,
            0.0, -1.0, -2.0, -4.0, -6.0
        ])

    elif 'Studio' in name or 'Tilt' in name:
        freqs = full_freqs
        mags = np.array([
            3.0,
            3.0, 2.6, 2.2, 1.8, 1.4, 1.0, 0.6, 0.2, 0.0,
            -0.4, -0.8, -1.2, -1.8, -2.4, -3.0, -3.8,
            -4.8, -6.0, -6.5
        ])

    # --- Listening use cases ---
    elif 'Nearfield' in name or 'Desk' in name:
        freqs = full_freqs
        mags = np.array([
            2.5,
            2.5, 2.4, 2.2, 2.0, 1.8, 1.4, 1.0, 0.6, 0.2,
            0.0, 0.0, 0.0, -0.2, -0.5, -1.0, -1.8,
            -3.0, -4.5, -5.0
        ])

    elif 'HiFi' in name or 'Loudness' in name:
        freqs = full_freqs
        mags = np.array([
            6.0,
            6.0, 5.8, 5.5, 5.0, 4.3, 3.5, 2.6, 1.8, 1.0,
            0.4, 0.0, -0.2, -0.6, -1.0, -1.6, -2.6,
            -3.6, -5.0, -5.5
        ])

    elif 'Speech' in name or 'Broadcast' in name:
        freqs = full_freqs
        mags = np.array([
            -2.0,
            -2.0, -1.8, -1.5, -1.2, -1.0, -0.6, -0.2, 0.4, 1.0,
            1.5, 1.8, 2.0, 2.0, 1.0, 0.0, -1.5,
            -3.5, -6.0, -8.0
        ])

    # --- Cinema / special ---
    elif 'Cinema' in name:
        freqs = np.array([
            0.0, 20.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0
        ])
        mags = np.array([
            0.0, 0.0, 0.0, -3.0, -9.0, -15.0, -18.0
        ])

    elif 'Flat' in name:
        freqs = full_freqs
        mags = np.zeros_like(freqs)

    # --- Default fallback ---
    else:
        freqs = full_freqs
        mags = np.array([
            6.0,
            6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5,
            1.4, 0.4, 0.0, -0.5, -1.0, -1.8, -2.8,
            -4.0, -5.5, -6.0
        ])
    return freqs, mags



def load_target_curve(file_content):
    """Lukee tavoitekäyrän tekstitiedostosta ja varmistaa järjestyksen."""
    try:
        content_str = file_content.decode('utf-8')
        lines = content_str.split('\n')
        freqs, mags = [], []
        for line in lines:
            # Poistetaan kommentit ja tyhjät
            line = line.split('#')[0].strip()
            if not line: continue
            
            # Tuetaan pilkkua ja pistettä, sekä tabulaattoria ja välilyöntiä
            parts = line.replace(',', '.').split()
            if len(parts) >= 2:
                try:
                    f = float(parts[0])
                    m = float(parts[1])
                    # Estetään nollataajuudet ja negatiiviset taajuudet
                    if f > 0:
                        freqs.append(f)
                        mags.append(m)
                except ValueError:
                    continue

        if len(freqs) < 2:
            return None, None

        # --- TÄRKEÄ KORJAUS: LAJITTELU ---
        # Varmistetaan, että taajuudet ovat nousevassa järjestyksessä.
        # Jos eivät ole, np.interp tekee "sahalaitaa" tai monttuja.
        freqs = np.array(freqs)
        mags = np.array(mags)
        if np.mean(mags) > 30:
            mags -= np.mean(mags)

        # Lajittelu taajuuden mukaan (tärkeää interpoloinnille)
        sort_idx = np.argsort(freqs)
        
        return freqs[sort_idx], mags[sort_idx]
        
    except Exception as e:
        print(f"Error loading target curve: {e}")
        return None, None


def load_config():
    default_conf = {
        'fmt': 'WAV', 'layout': 'Mono', 'fs': 44100, 'taps': 65536,
        'filter_type': 'Linear Phase', 'gain': 0.0, 
        'hc_mode': 'Harman6', 'mag_correct': True,
        'smoothing_type': 'smooth_psy', 'fdw_cycles': 15.0,
        'mag_c_min': 10.0, 'mag_c_max': 200.0, 'max_boost': 5.0,
        'lvl_mode': 'Auto', 'lvl_algo': 'Median', 
        'lvl_manual_db': 75.0, 'lvl_min': 300.0, 'lvl_max': 3000.0,
        'normalize_opt': False, 'align_opt': True, 'multi_rate_opt': False,
        'reg_strength': 30.0, 'stereo_link': True, 
        'exc_prot': True, 'exc_freq': 20.0, 
        'low_bass_cut_hz': 40.0,    # alle tämän taajuuden sallitaan vain leikkaus (ei boostia)
        'hpf_enable': False, 'hpf_freq': 20.0, 'hpf_slope': 24,
        'local_path_l': '', 'local_path_r': '',
        'input_source': 'file',             # 'file' | 'rew_api'
        'rew_api_base_url': 'http://127.0.0.1:4735',
        'rew_meas_left': '',
        'rew_meas_right': '',
        'xo1_f': None, 'xo1_s': 12, 'xo2_f': None, 'xo2_s': 12,
        'xo3_f': None, 'xo3_s': 12, 'xo4_f': None, 'xo4_s': 12, 'xo5_f': None, 'xo5_s': 12,
        'mixed_freq': 300.0, 'phase_limit': 600.0,
        'phase_safe_2058': False, # TUPE-mode
        'ir_window': 500.0,       # Oikea ikkuna (Right)
        'ir_window_left': 50.0,  # Vasen ikkuna (Left) - UUSI
        'enable_tdc': True,       # TDC oletuksena päälle
        'tdc_strength': 50.0,     # TDC voimakkuus 50%
        'enable_afdw': True,      # Adaptiivinen FDW oletuksena päälle
        'max_cut_db': 30.0,              # max vaimennus (dB)
        'max_slope_db_per_oct': 24.0,    # max jyrkkyys (dB/okt), 0 = pois
        'max_slope_boost_db_per_oct': 0.0,
        'max_slope_cut_db_per_oct': 0.0,
        'df_smoothing': False,
        'comparison_mode': False,         # LOCK score/match analysis to 44.1k reference grid
        'tdc_max_reduction_db': 9.0,
        'tdc_slope_db_per_oct': 6.0,
        "bass_first_ai": True,
        "bass_first_mode_max_hz": 200.0,
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                saved = json.load(f)
                for k in ['mag_correct', 'normalize_opt', 'align_opt', 'multi_rate_opt', 'stereo_link', 'exc_prot', 'hpf_enable', 'df_smoothing', 'comparison_mode', 'phase_safe_2058']:
                    if k in saved and isinstance(saved[k], list): saved[k] = bool(saved[k])
                default_conf.update(saved)
        except: pass
    return default_conf

def save_config(data):
    try:
        clean_data = {k: v for k, v in data.items() if not k.startswith('file_')}
        with open(CONFIG_FILE, 'w') as f: json.dump(clean_data, f, indent=4)
    except: pass

def put_guide_section():
    # Tämä lista ohjaa, mitä oppaita näytetään ja missä järjestyksessä
    guides = [
        ('guide_taps', t('guide_taps_title')),
        ('guide_ft', t('guide_ft_title')),
        ('guide_sigma', t('guide_sigma_title')),
        ('guide_mix', t('guide_mix_title')),
        ('guide_fdw', t('guide_fdw_title')),
        ('guide_reg', t('guide_reg_title')),
        ('guide_lvl', t('guide_lvl_title')),
        ('guide_sl', t('guide_sl_title')),
        ('guide_ep', t('guide_ep_title')),
        ('guide_asy', t('guide_asy_title')),
        ('guide_ai', t('guide_ai_title')),
        ('guide_summary', t('guide_summary_title')),
        ('guide_afdw', t('guide_afdw_title'))
    ]
    content = [put_collapse(t(g_key + '_title') if t(g_key + '_title') != (g_key + '_title') else g_title, [put_markdown(t(g_key + '_body') if t(g_key + '_body') != (g_key + '_body') else "Info text here")]) for g_key, g_title in guides]
    put_collapse("❓ CamillaFIR User Guides", content)

@config(theme="dark")
def main():
    set_env(output_max_width='1850px') 
    put_markdown(f"# 🎛️ {PROGRAM_NAME} {VERSION}")
    put_markdown(f"### {t('subtitle')}")
    put_guide_section(); put_markdown("---")
    d = load_config(); get_val = lambda k, def_v: d.get(k, def_v)
    hc_opts = [
        {'label': t('hc_harman'),        'value': 'Harman6'},   # default
        {'label': t('hc_harman8'),       'value': 'Harman8'},
        {'label': t('hc_harman4'),       'value': 'Harman4'},
        {'label': t('hc_harman10'),      'value': 'Harman10'},
        {'label': t('hc_studio_tilt'),   'value': 'Studio'},
        {'label': t('hc_nearfield'),     'value': 'Nearfield'},
        {'label': t('hc_hifi_loudness'), 'value': 'HiFi'},
        {'label': t('hc_speech'),        'value': 'Speech'},
        {'label': t('hc_toole'),         'value': 'Toole'},
        {'label': t('hc_bk'),            'value': 'BK'},
        {'label': t('hc_flat'),          'value': 'Flat'},
        {'label': t('hc_cinema'),        'value': 'Cinema'},
        {'label': t('hc_mode_upload'),   'value': 'Custom'},
    ]
    fs_opts = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]; taps_opts = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]; slope_opts = [6, 12, 18, 24, 36, 48]
    
#--- #1 Tiedostot
    
    tab_files = [
        put_markdown(f"### 📂 {t('tab_files')}"),
        put_markdown("---"),
        put_markdown(t('wav_recommended_info')),
        put_markdown("---"),
        put_markdown(f"### 🧾 {t('input_files_title')}"),
        put_html(
            f"<div style='opacity:0.75; font-size:13px; margin-top:-6px;'>"
            f"{t('input_files_help')}"
            f"</div>"
        ),
        put_file_upload('file_l', label=t('upload_l'), accept='.txt,.wav'), 
        put_input('local_path_l', label=t('path_l'), value=get_val('local_path_l', ''), help_text=t('path_help')),
        put_file_upload('file_r', label=t('upload_r'), accept='.txt,.wav'), 
        put_input('local_path_r', label=t('path_r'), value=get_val('local_path_r', ''), help_text=t('path_help')),
        put_select('fmt', label=t('fmt'), options=['WAV', 'TXT'], value=get_val('fmt', 'WAV'), help_text=t('fmt_help')),
        put_radio('layout', label=t('layout'), options=[t('layout_mono'), t('layout_stereo')], value=get_val('layout', t('layout_stereo')), inline=True),
        put_checkbox('multi_rate_opt', options=[{'label': t('multi_rate'), 'value': True}], value=[True] if get_val('multi_rate_opt', False) else [], help_text=t('multi_rate_help')),
        put_checkbox('comparison_mode',
                     options=[{'label': t('comparison_mode'), 'value': True}],
                     value=[True] if get_val('comparison_mode', True) else [],
                     help_text=t('comparison_mode_help')),
        put_scope('taps_auto_info_scope_files'),
    ]
    
#--- #2 Perusasetukset

    tab_basic = [
        put_markdown(f"### ⚙️ {t('tab_basic')}"),
        
        # Rivi 1: Näytteenottotaajuus ja Tapit          
        put_row([
            put_select('fs', label=t('fs'), options=fs_opts, value=get_val('fs', 44100), help_text=t('fs_help')), 
            put_select('taps', label=t('taps'), options=taps_opts, value=get_val('taps', 65536), help_text=t('taps_help'))
        ]),


        # Auto-taps info (näkyy vain kun multi_rate_opt päällä)
        put_scope('taps_auto_info_scope_basic'),
        
        # Rivi 2: Suodintyyppi ja Mixed-taajuus
        put_row([
            put_radio('filter_type', label=t('filter_type'), 
                    options=[t('ft_linear'), t('ft_min'), t('ft_mixed'), t('ft_asymmetric')], 
                    value=get_val('filter_type', t('ft_linear')), help_text=t('ft_help')), 
            put_input('mixed_freq', label=t('mixed_freq'), type=FLOAT, value=get_val('mixed_freq', 300.0), help_text=t('mixed_freq_help'))
        ]),
        
        put_input('gain', label=t('gain'), type=FLOAT, value=get_val('gain', 0.0), help_text=t('gain_help')),
        
        put_select('lvl_algo', label="Algo", options=['Median', 'Average'], value=get_val('lvl_algo', 'Median'), help_text=t('lvl_algo_help')),
        put_select(
                    'smoothing_type',
                    label=t('smooth_type'),
                    options=[
                        {'label': t('smooth_std'), 'value': 'Standard'},
                        {'label': t('smooth_psy'), 'value': 'Psychoacoustic'}
                    ],
                    value=get_val('smoothing_type', 'Psychoacoustic'),
                    help_text=t('smooth_help')
                    ),
        # Rivi 3: Tilan valinta ja tavoitetaso (jaettu kahteen osaan luettavuuden vuoksi)
        # Level match range (help_text tulee oikeaan paikkaan suoraan kenttien alle)
        put_row([
            put_input(
                'lvl_min',
                label=t('lvl_min'),
                type=FLOAT,
                value=get_val('lvl_min', 500.0),
                help_text=t('lvl_min_help_auto')  # default Auto-tila
            ),
            put_input(
                'lvl_max',
                label=t('lvl_max'),
                type=FLOAT,
                value=get_val('lvl_max', 2000.0),
                help_text=t('lvl_max_help_auto')  # default Auto-tila
            ),
        ]),

        # lvl_mode + lvl_manual_db (näytetään aina, mutta Auto-tilassa lukittu)
        put_row([
            put_select(
                'lvl_mode',
                label=t('lvl_mode'),
                options=[
                    {'label': t('lvl_mode_auto'), 'value': 'Auto'},
                    {'label': t('lvl_mode_manual'), 'value': 'Manual'},
                ],
                value=get_val('lvl_mode', 'Auto')
            ),
            put_scope('lvl_manual_scope'),
        ]),


        
            ]
#--- #3 Target
    
    tab_target = [
        put_markdown(f"### 🎯 {t('tab_target')}"),
        put_select(
            'hc_mode',
            label=t('hc_mode'),
            options=hc_opts,
            value=_normalize_hc_mode_key(get_val('hc_mode', 'Harman6')),
            help_text=t('hc_mode_help')
        ),
        
        
        put_file_upload('hc_custom_file', label=t('hc_custom'), accept='.txt', help_text=t('hc_custom_help')),
        put_markdown("---"),
        put_checkbox('mag_correct', options=[{'label': t('enable_corr'), 'value': True}], value=[True] if get_val('mag_correct', True) else []),
        put_markdown("---"),
        put_row([
            put_input('mag_c_min', label=t('min_freq'), type=FLOAT, value=get_val('mag_c_min', 10.0), help_text=t('hc_range_help')), 
            put_input('mag_c_max', label=t('max_freq'), type=FLOAT, value=get_val('mag_c_max', 200.0), help_text=t('hc_range_help'))
        ]),
        put_input('max_boost', label=t('max_boost'), type=FLOAT, value=get_val('max_boost', 5.0), help_text=t('max_boost_help')),
        put_row([
            put_input('max_cut_db', label=t('max_cut_db'), type=FLOAT, value=get_val('max_cut_db', 30.0),
                      help_text=t('max_cut_db_help')),
            put_input('max_slope_db_per_oct', label=t('max_slope_db_per_oct'), type=FLOAT, value=get_val('max_slope_db_per_oct', 12.0),
                      help_text=t('max_slope_db_per_oct_help'))
        ]),
        put_row([
            put_input('max_slope_boost_db_per_oct', label=t('max_slope_boost_db_per_oct'), type=FLOAT,
                      value=get_val('max_slope_boost_db_per_oct', 0.0),
                      help_text=t('max_slope_boost_db_per_oct_help')),
            put_input('max_slope_cut_db_per_oct', label=t('max_slope_cut_db_per_oct'), type=FLOAT,
                      value=get_val('max_slope_cut_db_per_oct', 0.0),
                      help_text=t('max_slope_cut_db_per_oct_help'))
         ]),
        
        put_input('trans_width', type=NUMBER, label="1/1 Transition Width (Hz)", value=100, help_text=t('trans_width')),
        put_markdown("---"),
        put_select(
                    'smoothing_level',
                    label=t('smoothing_level'),
                    options=[
                        {'label': '1/1 Octave', 'value': 1},
                        {'label': '1/3 Octave', 'value': 3},
                        {'label': '1/6 Octave', 'value': 6},
                        {'label': '1/12 Octave (Standard)', 'value': 12},
                        {'label': '1/24 Octave (Fine)', 'value': 24},
                        {'label': '1/48 Octave (Ultra)', 'value': 48},
                        {'label': '1/96 Octave (HC)', 'value': 96},
                    ],
                    value=get_val('smoothing_level', 12),
                    help_text=t('smoothing_level_help'),
                ),
              
        
        put_input('phase_limit', label=t('phase_limit'), type=FLOAT, value=get_val('phase_limit', 1000.0), help_text=t('phase_limit_help')),
        
        put_checkbox(
            'phase_safe_2058',
            options=[{'label': t('phase_safe_2058'), 'value': True}],
            value=[True] if get_val('phase_safe_2058', False) else [],
            help_text=t('phase_safe_2058_help')
        ),
        
    ]
#--- #4 Edistyneet
    tab_adv = [
        put_markdown(f"### 🛠️ {t('tab_adv')}"),
        
        put_markdown("#### ⏱️ Asymmetric Linear -ikkunointi"),
        put_row([
            put_input('ir_window_left', label="Left Window (ms)", type=FLOAT, value=get_val('ir_window_left', 100.0), help_text=t('ir_matala')),
            put_input('ir_window', label="Right Window (ms)", type=FLOAT, value=get_val('ir_window', 500.0), help_text=t('ir_korkea'))
        ]),
        put_markdown("---"),

        # Afdw
        put_checkbox('enable_afdw', options=[{'label': t('enable_afdw'), 'value': True}], 
             value=[True] if get_val('enable_afdw', True) else [], help_text=t('afdw_help')),
        put_row([
            put_input('fdw_cycles', label=t('fdw'), type=FLOAT, value=get_val('fdw_cycles', 15.0), help_text=t('fdw_help'))
        ]),
        put_markdown("---"),
        # --- TDC aka Trinnov-mode (PyWebIO)

        put_markdown("#### ⏳ Temporal Decay Control (TDC)"),
        put_row([
            put_buttons(
                [
                    {"label": t("tdc_preset_safe"), "value": "Safe"},
                    {"label": t("tdc_preset_normal"), "value": "Normal"},
                    {"label": t("tdc_preset_aggressive"), "value": "Aggressive"},
                ],
                onclick=lambda preset: apply_tdc_preset(preset),
                small=True,
            ),
        ]),
        put_html(f"<div style='opacity:0.65; font-size:13px'>{t('tdc_preset_help')}</div>"),


        put_checkbox(
            'enable_tdc',
            options=[{'label': t('enable_tdc'), 'value': True}],
            value=[True] if get_val('enable_tdc', True) else [],
            help_text=t('tdc_help')
        ),

        put_row([
            put_input(
                'tdc_strength',
                label=t('tdc_strength'),
                type=FLOAT,
                value=get_val('tdc_strength', 50.0),
                help_text=t('tdc_help')
            ),
            put_input(
                'tdc_max_reduction_db',
                label=t('tdc_max_reduction_db'),
                type=FLOAT,
                value=get_val('tdc_max_reduction_db', 9.0),
                help_text=t('tdc_max_reduction_db_help')
            ),
            put_input(
                'tdc_slope_db_per_oct',
                label=t('tdc_slope_db_per_oct'),
                type=FLOAT,
                value=get_val('tdc_slope_db_per_oct', 6.0),
                help_text=t('tdc_slope_db_per_oct_help')
            ),
        ]),
        
            put_markdown(f"#### 🧠 {t('bass_first_title')}"),
            put_checkbox(
                'bass_first_ai',
                options=[{'label': t('bass_first_enable_label'), 'value': True}],
                value=[True] if get_val('bass_first_ai', False) else [],
                help_text=t('bass_first_enable_help')
            ),
            put_input(
                'bass_first_mode_max_hz',
                label=t('bass_first_max_hz_label'),
                type=FLOAT,
                value=float(get_val('bass_first_mode_max_hz', 200.0) or 200.0),
                help_text=t('bass_first_max_hz_help')
            ),
            
put_markdown("---"),

        put_checkbox('df_smoothing', options=[{'label': f"{t('df_smoothing_label')} {t('badge_experimental')}", 'value': True}],
             value=[True] if get_val('df_smoothing', False) else [],
             help_text=t('df_smoothing_help')),
        put_markdown("---"),
        
        put_input('reg_strength', label=t('reg_strength'), type=FLOAT, value=get_val('reg_strength', 30.0), help_text=t('reg_help')),
        put_markdown("---"),
        
        

        put_row([
            put_checkbox('normalize_opt', options=[{'label': t('enable_norm'), 'value': True}], value=[True] if get_val('normalize_opt', True) else [], help_text=t('norm_help')), 
            put_checkbox('align_opt', options=[{'label': t('enable_align'), 'value': True}], value=[True] if get_val('align_opt', True) else [], help_text=t('align_help')), 
            put_checkbox('stereo_link', options=[{'label': t('enable_link'), 'value': True}], value=[True] if get_val('stereo_link', False) else [], help_text=t('link_help'))
        ]),
        
        # --- Bass Safety (Advanced tab) ---
put_markdown("### 🛡️ Bass Safety"),
put_markdown("---"),

        # 1) Excursion Protection (Driver Safety)
        put_row([
            put_checkbox(
                'exc_prot',
                options=[{'label': t('exc_prot_title'), 'value': True}],
                value=[True] if get_val('exc_prot', False) else [],
                help_text=t('exc_prot_help_ui')
            ),
            put_input(
                'exc_freq',
                label=t('exc_freq'),
                type=FLOAT,
                value=get_val('exc_freq', 25.0),
                help_text=t('exc_freq_help_ui')
            ),
        ]),

        # micro-hint (small, grey)
        put_html(
            f"<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>"
            f"{t('exc_prot_hint')}"
            f"</div>"
        ),

        # guide (collapsible)
        put_collapse(
            t('guide_exc_prot_title'),
            [put_markdown(t('guide_exc_prot_body'))]
        ),

        # spacing between the two tools
        put_html("<div style='height:12px'></div>"),

        # 2) Low-bass boost lock (policy limiter)
        put_input(
            'low_bass_cut_hz',
            label=t('low_bass_cut_hz'),
            type=FLOAT,
            value=get_val('low_bass_cut_hz', 40.0),
            help_text=t('low_bass_cut_hz_help')
        ),

        # micro-hint (small, grey)
        put_html(
            f"<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>"
            f"{t('low_bass_cut_hint')}"
            f"</div>"
        ),

        # guide (collapsible)
        put_collapse(
            t('guide_low_bass_cut_title'),
            [put_markdown(t('guide_low_bass_cut_body'))]
        ),

        
        put_markdown("---"),
        put_row([
            put_checkbox('hpf_enable', options=[{'label': t('hpf_enable'), 'value': True}], value=[True] if get_val('hpf_enable', False) else []), 
            put_input('hpf_freq', label=t('hpf_freq'), type=FLOAT, value=get_val('hpf_freq', 20.0), help_text=t('hpf_freq_help')), 
            put_select('hpf_slope', label=t('hpf_slope'), options=slope_opts, value=get_val('hpf_slope', 24))
        ])
    ]

#--- #5 XO
    tab_xo = [
        put_markdown(f"### ❌ {t('tab_xo')}"), 
        put_grid([[
            put_input(f'xo{i}_f', label=f"XO {i} Hz", type=FLOAT, value=get_val(f'xo{i}_f', None), help_text=t('xo_freq_help')), 
            put_select(f'xo{i}_s', label="dB/oct", options=slope_opts, value=get_val(f'xo{i}_s', 12), help_text=t('xo_slope_help'))
        ] for i in range(1, 6)])
    ]

    # Piirretään välilehdet
    put_tabs([
        {'title': t('tab_files'), 'content': tab_files}, 
        {'title': t('tab_basic'), 'content': tab_basic}, 
        {'title': t('tab_target'), 'content': tab_target}, 
        {'title': t('tab_adv'), 'content': tab_adv}, 
        {'title': t('tab_xo'), 'content': tab_xo}
    ])
    pin_on_change('lvl_mode', onchange=update_lvl_ui)
    pin_on_change('lvl_min', onchange=update_lvl_ui)
    pin_on_change('lvl_max', onchange=update_lvl_ui)


    update_lvl_ui()


    # Auto-taps UI updater: react when multi-rate toggles (tab_files) or basic changes
    pin_on_change('multi_rate_opt', onchange=update_taps_auto_info)
    pin_on_change('fs', onchange=update_taps_auto_info)
    pin_on_change('taps', onchange=update_taps_auto_info)
    update_taps_auto_info()

    put_markdown("---")

    
    # Napin päivitys: Täysin puhdas teksti ilman taustaa tai kehyksiä
    put_button("🚀 START", onclick=process_run).style("""
        width: 100%; 
        margin-top: 30px; 
        padding: 15px; 
        font-size: 24px; 
        font-weight: 900; 
        letter-spacing: 3px;
        
        background-color: transparent;  /* Ei taustaväriä */
        border: none;                  /* Poistaa kehykset kokonaan */
        color: #ffffff;                /* Teksti on puhdas valkoinen */
        
        transition: 0.3s;
        cursor: pointer;
    """)
    
def _collect_ui_data():
    p_keys = [
        'fs', 'taps', 'filter_type', 'mixed_freq', 'gain', 'hc_mode',
        'mag_c_min', 'mag_c_max', 'input_source', 'rew_api_base_url', 'rew_meas_left', 'rew_meas_right', 'max_boost', 'max_cut_db', 'max_slope_db_per_oct',
        'max_slope_boost_db_per_oct', 'max_slope_cut_db_per_oct', 'phase_limit', 'phase_safe_2058', 'mag_correct',
        'lvl_mode', 'reg_strength', 'normalize_opt', 'align_opt',
        'stereo_link', 'exc_prot', 'exc_freq', 'low_bass_cut_hz', 'hpf_enable', 'hpf_freq',
        'hpf_slope', 'multi_rate_opt', 'ir_window', 'ir_window_left',
        'local_path_l', 'local_path_r', 'fmt', 'lvl_manual_db',
        'lvl_min', 'lvl_max', 'lvl_algo', 'smoothing_type', 'fdw_cycles',
        'trans_width', 'smoothing_level', 'enable_tdc', 'tdc_strength', 'tdc_max_reduction_db',
        'tdc_slope_db_per_oct', 'enable_afdw', 'df_smoothing', 'comparison_mode', 'bass_first_ai', 'bass_first_mode_max_hz'
    ]

    data = {}
    for k in p_keys:
        try:
            data[k] = pin[k]
        except Exception:
            data[k] = None

    for k in ['mag_correct', 'normalize_opt', 'align_opt', 'multi_rate_opt', 'stereo_link', 'exc_prot', 'hpf_enable', 'df_smoothing', 'comparison_mode', 'bass_first_ai']:
        try:
            if isinstance(data.get(k, None), list):
                data[k] = bool(data[k])
        except Exception:
            pass

    for i in range(1, 6):
        data[f'xo{i}_f'] = pin[f'xo{i}_f']
        data[f'xo{i}_s'] = pin[f'xo{i}_s']
        data['max_cut_db'] = abs(float(data.get('max_cut_db', 15.0) or 15.0))
        data['max_slope_db_per_oct'] = max(0.0, float(data.get('max_slope_db_per_oct', 24.0) or 24.0))
        data['max_slope_boost_db_per_oct'] = max(0.0, float(data.get('max_slope_boost_db_per_oct', 0.0) or 0.0))
        data['max_slope_cut_db_per_oct'] = max(0.0, float(data.get('max_slope_cut_db_per_oct', 0.0) or 0.0))        
        data['lvl_manual_db'] = float(data.get('lvl_manual_db', 75.0) or 75.0)

    return data


def _log_df_smoothing_toggle():
    try:
        df_on = bool(pin['df_smoothing'])
    except Exception:
        df_on = False
    logger.info(f"DF smoothing: {'ON' if df_on else 'OFF'}")
    return df_on


def _load_measurements(data):
    # --- REW (API) FIRST: if selected, skip file parsing entirely ---
    if str(data.get('input_source') or 'file') == 'rew_api':
        try:
            from camillafir_rew_api import RewApiClient, RewMeasurementMeta
        except Exception as e:
            raise RuntimeError(f"Missing camillafir_rew_api.py: {e}")

        base_url = str(data.get('rew_api_base_url') or "http://127.0.0.1:4735")
        left_id  = str(data.get('rew_meas_left') or "")
        right_id = str(data.get('rew_meas_right') or "")

        if not left_id or not right_id:
            # keep same return-contract as file path: (f_l,m_l,p_l,f_r,m_r,p_r)
            return None, None, None, None, None, None

        client = RewApiClient(base_url=base_url)
        if not client.ping():
            raise RuntimeError("REW API not reachable. Open REW and enable API.")

        client.discover_operation_ids()

        # Fetch only FR for now (freq/mag/phase). Keep it symmetric with txt/wav pipeline.
        mL = RewMeasurementMeta(id=left_id, name="Left")
        mR = RewMeasurementMeta(id=right_id, name="Right")

        # --- Fetch FR explicitly via discovered operationId ---
        if not client.op_get_fr:
            raise RuntimeError("REW API: FR operation not found in OpenAPI spec.")

                # --- Use robust client parser (handles dict-indexed arrays "1","2",... etc.) ---
        fr_obj_L = client.get_frequency_response(left_id)
        fr_obj_R = client.get_frequency_response(right_id)

        if fr_obj_L is None or fr_obj_R is None:
            raise RuntimeError(
                f"REW did not return usable FR data. L={left_id}, R={right_id}"
            )

        f_l = np.asarray(fr_obj_L.freqs_hz, dtype=float)
        m_l = np.asarray(fr_obj_L.mag_db, dtype=float)
        p_l = np.asarray(fr_obj_L.phase_rad, dtype=float) if fr_obj_L.phase_rad is not None else np.zeros_like(m_l)

        f_r = np.asarray(fr_obj_R.freqs_hz, dtype=float)
        m_r = np.asarray(fr_obj_R.mag_db, dtype=float)
        p_r = np.asarray(fr_obj_R.phase_rad, dtype=float) if fr_obj_R.phase_rad is not None else np.zeros_like(m_r)

        return f_l, m_l, p_l, f_r, m_r, p_r



    # UI-driven IR windows (ms) + smoothing for WAV parsing
    try:
        pre_ms = float(data.get("ir_window_left", 50.0) or 50.0)
    except Exception:
        pre_ms = 50.0
    ...

    # UI-driven IR windows (ms) + smoothing for WAV parsing
    try:
        pre_ms = float(data.get("ir_window_left", 50.0) or 50.0)
    except Exception:
        pre_ms = 50.0
    try:
        post_ms = float(data.get("ir_window", 500.0) or 500.0)
    except Exception:
        post_ms = 500.0
    try:
        sl = int(data.get("smoothing_level", 0) or 0)
    except Exception:
        sl = 0

    # First: local paths (can be .txt or .wav; parse_measurements_from_path handles both)
    f_l, m_l, p_l = parse_measurements_from_path(data["local_path_l"]) if data.get("local_path_l") else (None, None, None)
    f_r, m_r, p_r = parse_measurements_from_path(data["local_path_r"]) if data.get("local_path_r") else (None, None, None)

    # Then: upload priority (WAV wins; then fallback)
    # LEFT
    if pin.file_l and str(pin.file_l.get("filename", "")).lower().endswith(".wav"):
        f_l, m_l, p_l = parse_measurements_from_upload(
            pin.file_l,
            channel_index=0,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=sl,
        )
    elif f_l is None and pin.file_l:
        # TXT upload fallback (or unknown ext)
        f_l, m_l, p_l = parse_measurements_from_upload(
            pin.file_l,
            channel_index=0,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=sl,
        )

    # RIGHT
    if pin.file_r and str(pin.file_r.get("filename", "")).lower().endswith(".wav"):
        f_r, m_r, p_r = parse_measurements_from_upload(
            pin.file_r,
            channel_index=0,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=sl,
        )
    elif f_r is None and pin.file_r:
        f_r, m_r, p_r = parse_measurements_from_upload(
            pin.file_r,
            channel_index=0,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=sl,
        )

    return f_l, m_l, p_l, f_r, m_r, p_r



def _load_house_curve(data):
    hc_f, hc_m = None, None
    hc_source = "Preset"

    if pin.hc_custom_file:
        hc_f, hc_m = load_target_curve(pin.hc_custom_file['content'])
        hc_source = "Upload"

    if hc_f is None and data.get('local_path_house'):
        try:
            hc_f, hc_m, _ = parse_measurements_from_path(data['local_path_house'])
            if hc_f is not None:
                s_idx = np.argsort(hc_f)
                hc_f, hc_m = hc_f[s_idx], hc_m[s_idx]
                hc_source = "LocalFile"
        except Exception:
            hc_f, hc_m = None, None

    if hc_f is None:
        preset_key = _normalize_hc_mode_key(data.get('hc_mode'))
        # Custom selection means "use uploaded/local file"; if none present, fall back safely.
        preset_key = "Flat" if preset_key == "Custom" else preset_key
        hc_f, hc_m = get_house_curve_by_name(preset_key)
        hc_source = f"Preset ({preset_key})"

    return hc_f, hc_m, hc_source


def _build_xos_hpf(data):
    xos = [{'freq': data[f'xo{i}_f'], 'order': data[f'xo{i}_s'] // 6} for i in range(1, 6) if data[f'xo{i}_f']]
    hpf = {'enabled': data['hpf_enable'], 'freq': data['hpf_freq'], 'order': data['hpf_slope'] // 6} if data['hpf_enable'] else None
    return xos, hpf


def _filter_type_short(filter_type):
    if "Asymmetric" in filter_type:
        return "Asymmetric"
    if "Min" in filter_type:
        return "Minimum"
    if "Mixed" in filter_type:
        return "Mixed"
    return "Linear"

#filtterin teko
def _build_filter_config(fs_v, taps_v, data, xos, hpf, hc_f, hc_m):
    # Source hint for analysis heuristics (Bass-first reliability masking)
    # WAV/IR-derived measurements tend to have noisier phase unwrap/jitter than REW text/API.
    try:
        is_wav_src = str(data.get('fmt', 'TXT') or 'TXT').strip().upper() == 'WAV'
    except Exception:
        is_wav_src = False

    return FilterConfig(
        fs=fs_v,
        num_taps=taps_v,
        df_smoothing=bool(pin['df_smoothing']),
        **({"comparison_mode": True} if hasattr(FilterConfig, "comparison_mode") else {}),
        filter_type_str=data['filter_type'],
        mixed_split_freq=data['mixed_freq'],
        global_gain_db=data['gain'],
        mag_c_min=data['mag_c_min'],
        mag_c_max=data['mag_c_max'],
        max_boost_db=data['max_boost'],
        max_cut_db=data.get('max_cut_db', 30.0),
        max_slope_db_per_oct=data.get('max_slope_db_per_oct', 24.0),
        max_slope_boost_db_per_oct=data.get('max_slope_boost_db_per_oct', 0.0),
        max_slope_cut_db_per_oct=data.get('max_slope_cut_db_per_oct', 0.0),
        phase_limit=data['phase_limit'],
        phase_safe_2058=bool(data.get('phase_safe_2058', False)),
        enable_mag_correction=bool(data.get('mag_correct', True)),
        lvl_mode=data['lvl_mode'],
        reg_strength=float(data.get('reg_strength', 30.0)),
        do_normalize=bool(data['normalize_opt']),
        exc_prot=bool(data['exc_prot']),
        exc_freq=data['exc_freq'],
        low_bass_cut_hz=float(data.get('low_bass_cut_hz', 40.0) or 40.0),
        ir_window_ms=data['ir_window'],
        ir_window_ms_left=data.get('ir_window_left', 100.0),
        enable_afdw=bool(pin.enable_afdw),
        enable_tdc=bool(pin.enable_tdc),
        tdc_strength=data.get('tdc_strength', 50.0),
        tdc_max_reduction_db=float(pin['tdc_max_reduction_db']),
        tdc_slope_db_per_oct=float(pin['tdc_slope_db_per_oct']),
        smoothing_type=data['smoothing_type'],
        fdw_cycles=data['fdw_cycles'],
        lvl_manual_db=data['lvl_manual_db'],
        lvl_min=data['lvl_min'],
        lvl_max=data['lvl_max'],
        lvl_algo=data['lvl_algo'],
        stereo_link=bool(data.get('stereo_link', False)),
        smoothing_level=int(pin.smoothing_level),
        crossovers=xos,
        hpf_settings=hpf,
        house_freqs=hc_f,
        house_mags=hc_m,
        trans_width=data.get('trans_width', 100.0),
        bass_first_ai=bool(data.get('bass_first_ai', False)),
        bass_first_mode_max_hz=float(data.get('bass_first_mode_max_hz') or 200.0),
        bass_first_smooth_floor_lo=float(data.get('bass_first_smooth_floor_lo') or 0.75),
        bass_first_smooth_floor_hi=float(data.get('bass_first_smooth_floor_hi') or 0.35),
        bass_first_k_mode_cut=float(data.get('bass_first_k_mode_cut') or 0.6),
        bass_first_k_mode_boost=float(data.get('bass_first_k_mode_boost') or 0.9),
    )


def _log_df_smoothing_for_fs(cfg, fs_v, df_on):
    if df_on:
        try:
            base_sigma = 60 // (cfg.smoothing_level / 12 if cfg.smoothing_level > 0 else 1)
            df_ref = 44100.0 / 65536.0
            sigma_hz = base_sigma * df_ref
            df_cur = (fs_v / cfg.num_taps)
            sigma_bins = sigma_hz / df_cur if df_cur > 0 else base_sigma

            logger.info(
                f"{fs_v//1000} kHz -> DF smoothing ON "
                f"(sigma = {sigma_bins:.1f} bins -> {sigma_hz:.1f} Hz)"
            )
        except Exception:
            logger.info(f"{fs_v//1000} kHz -> DF smoothing ON")
    else:
        logger.info(f"{fs_v//1000} kHz -> DF smoothing OFF")


def _append_dsp_effective_params(summary_content, data, fs_v):
    try:
        enable_afdw = bool(pin['enable_afdw']) if 'enable_afdw' in pin else bool(data.get('enable_afdw', False))
        enable_tdc = bool(pin['enable_tdc']) if 'enable_tdc' in pin else bool(data.get('enable_tdc', False))
        tdc_strength = float(data.get('tdc_strength', 0.0) or 0.0)
        fdw_cycles = float(data.get('fdw_cycles', 15.0) or 15.0)
        fdw_oct_width = (2.0 / fdw_cycles) if fdw_cycles > 0 else 0.0
        afdw_min = max(3.0, fdw_cycles / 3.0)
        afdw_min_oct_width = (2.0 / afdw_min) if afdw_min > 0 else 0.0

        df_on = bool(pin['df_smoothing']) if 'df_smoothing' in pin else bool(data.get('df_smoothing', False))
        df_ref = 44100.0 / 65536.0
        base_sigma = 60 // (data.get('smoothing_level', 12) / 12 if (data.get('smoothing_level', 12) or 0) > 0 else 1)
        sigma_hz = float(base_sigma) * df_ref
        df_cur = (float(fs_v) / float(data.get('taps', 65536) or 65536))
        sigma_bins = (sigma_hz / df_cur) if (df_cur and df_cur > 0) else float(base_sigma)

        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += f"Sample rate: {int(fs_v)} Hz\n"

        if enable_afdw:
            summary_content += "FDW mode: Adaptive (A-FDW)\n"
            summary_content += f"FDW base cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"
            summary_content += f"FDW min cycles:  {afdw_min:.2f}  (oct width -> {afdw_min_oct_width:.3f})\n"
            summary_content += "Note: A-FDW adapts per frequency/confidence; values above are the configured baseline.\n"
        else:
            summary_content += "FDW mode: Fixed\n"
            summary_content += f"FDW cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"

        summary_content += f"TDC: {'ON' if enable_tdc else 'OFF'}\n"
        if enable_tdc:
            summary_content += f"TDC strength: {tdc_strength:.1f}% (base_strength = {tdc_strength/100.0:.3f})\n"

        summary_content += f"DF smoothing: {'ON' if df_on else 'OFF'}\n"
        if df_on:
            summary_content += f"DF smoothing sigma: {sigma_bins:.1f} bins -> {sigma_hz:.2f} Hz\n"
    except Exception:
        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += "Could not compute effective params (unexpected data/pin state).\n"

    return summary_content

def _append_acoustic_events(summary_content, l_st, r_st):
    for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
        reflections = st.get('reflections') or []
        if reflections:
            summary_content += f"\n=== ACOUSTIC EVENTS ({side}) ===\n"
            summary_content += f"{'Freq (Hz)':<10} {'Type':<12} {'Error (ms)':<12} {'Dist (m)':<10}\n"
            summary_content += "-" * 50 + "\n"
            for rev in reflections:
                freq = float(rev.get('freq', 0) or 0)
                ev_type = str(rev.get('type', 'Event') or 'Event')
                gd_error = float(rev.get('gd_error', 0) or 0)
                dist = float(rev.get('dist', 0) or 0)
                summary_content += f"{freq:<10} {ev_type:<12} {gd_error:<12} {dist:<10}\n"
        # Always report headroom/normalization per side (even if no events)
        summary_content += f"\n=== HEADROOM MANAGEMENT ({side}) ===\n"
        summary_content += f"Normalize: {'ON' if bool(st.get('do_normalize', False)) else 'OFF'}\n"
        summary_content += f"Peak Gain (pre-headroom): {float(st.get('peak_gain_db', 0.0)):.2f} dB\n"
        summary_content += f"Applied Headroom: {float(st.get('auto_headroom_db', 0.0)):.2f} dB\n"
        summary_content += f"Final Max (gain+global+headroom): {float(st.get('final_max_db', 0.0)):.2f} dB\n"
        # Diagnostics for boost/cut processing
        summary_content += f"\n=== BOOST/CUT DIAGNOSTICS ({side}) ===\n"
        summary_content += f"Config: max_boost_db={float(st.get('max_boost_db', 0.0)):.2f} dB, "
        summary_content += f"max_cut_db={float(st.get('max_cut_db', 0.0)):.2f} dB\n"
        summary_content += f"Config: low_bass_cut_hz={float(st.get('low_bass_cut_hz', 0.0)):.1f} Hz, "
        summary_content += f"exc_prot={'ON' if bool(st.get('exc_prot', False)) else 'OFF'}, "
        summary_content += f"exc_freq={float(st.get('exc_freq', 0.0)):.1f} Hz, "
        summary_content += f"max_slope_db_per_oct={float(st.get('max_slope_db_per_oct', 0.0)):.1f}\n"
        summary_content += f"Result (post-clamp): boost_peak={float(st.get('boost_peak_db', 0.0)):.2f} dB, "
        summary_content += f"cut_peak={float(st.get('cut_peak_db', 0.0)):.2f} dB, "
        summary_content += f"boost_bins={int(st.get('boost_bins', 0))}\n"
        summary_content += f"Net boost peak (post global/headroom): {float(st.get('net_boost_peak_db', 0.0)):.2f} dB\n"
        summary_content += f"Candidate (pre-softclip): boost_peak={float(st.get('boost_candidate_peak_db', 0.0)):.2f} dB, "
        summary_content += f"boost_bins={int(st.get('boost_candidate_bins', 0))}, "
        summary_content += f"lowbass_boost_bins={int(st.get('boost_candidate_bins_lowbass', 0))}, "
        summary_content += f"excprot_boost_bins={int(st.get('boost_candidate_bins_excprot', 0))}\n"
        summary_content += f"Boost blocked reason: {str(st.get('boost_blocked_reason', 'n/a'))}\n"
        summary_content += f"\n=== CLAMP DIAGNOSTICS ({side}) ===\n"
        summary_content += f"{str(st.get('clamp_summary', 'n/a'))}\n"
        summary_content += (
            f"soft_clip: boost_bins={int(st.get('softclip_boost_bins', 0))}, "
            f"cut_bins={int(st.get('softclip_cut_bins', 0))}, "
            f"worst_over_boost={float(st.get('softclip_worst_over_boost_db', 0.0)):.2f} dB, "
            f"worst_over_cut={float(st.get('softclip_worst_over_cut_db', 0.0)):.2f} dB\n"
        )
        summary_content += (
            f"hard_clamp: boost_bins={int(st.get('hardclamp_boost_bins', 0))}, "
            f"cut_bins={int(st.get('hardclamp_cut_bins', 0))}, "
            f"worst_over_boost={float(st.get('hardclamp_worst_over_boost_db', 0.0)):.2f} dB, "
            f"worst_over_cut={float(st.get('hardclamp_worst_over_cut_db', 0.0)):.2f} dB\n"
        )


        # --- Stage checkpoints table ---
        probes = st.get("stage_probes") or {}
        if isinstance(probes, dict) and probes:
            summary_content += f"\n=== STAGE CHECKPOINTS ({side}) ===\n"
            summary_content += f"{'Stage':<22} {'BoostPk':>8} {'CutPk':>8} {'BoostBins':>10} {'CutBins':>8} {'NetBoostPk':>11}\n"
            summary_content += "-" * 75 + "\n"
            order = [
                "after_gain_apply",
                "after_lowbass_policy",
                "after_slope",
                "after_fade",
                "pre_softclip",
                "post_softclip",
                "post_hardclamp",
            ]
            for key in order:
                p = probes.get(key)
                if not isinstance(p, dict):
                    continue
                stage = str(p.get("stage", key))
                bpk = float(p.get("boost_peak_db", 0.0) or 0.0)
                cpk = float(p.get("cut_peak_db", 0.0) or 0.0)
                bb  = int(p.get("boost_bins", 0) or 0)
                cb  = int(p.get("cut_bins", 0) or 0)
                nbp = float(p.get("net_boost_peak_db", 0.0) or 0.0)
                summary_content += f"{stage:<22} {bpk:>8.2f} {cpk:>8.2f} {bb:>10d} {cb:>8d} {nbp:>11.2f}\n"

            summary_content += f"\n=== BASS-FIRST AI ({side}) ===\n"
            summary_content += f"Bass-first AI active: {'YES' if bool(st.get('bass_first_ai', False)) else 'NO'}\n"

            # --- Mode peak (robust formatting; fixes lost 'n/a' line) ---
            pk_hz = st.get('bass_first_mode_peak_hz', None)
            pk_sc = st.get('bass_first_mode_peak_score', None)
            if (pk_hz is not None) and (pk_sc is not None):
                summary_content += f"Mode peak: {float(pk_hz):.1f} Hz (score {float(pk_sc):.2f})\n"
            else:
                summary_content += "Mode peak: n/a\n"

            summary_content += f"Smoothing conf floor applied: {'YES' if bool(st.get('bass_first_conf_floor_applied', False)) else 'NO'}\n"

            # --- BF debug stats (if present) ---
            rm_max = st.get('bass_first_roommode_max_20_200', None)
            rel_mean = st.get('bass_first_rel_mean_20_200', None)
            rel_min = st.get('bass_first_rel_min_20_200', None)
            conf_eff_mean = st.get('bass_first_conf_eff_mean_20_200', None)
            conf_eff_min = st.get('bass_first_conf_eff_min_20_200', None)
            floor_applied = bool(st.get('bass_first_conf_floor_applied', False))
            if (rm_max is not None) or (rel_mean is not None) or (rel_min is not None) or (conf_eff_mean is not None):
                summary_content += (
                    f"BF masks (20–200): "
                    f"roommode_max={float(rm_max or 0.0):.3f}, "
                    f"rel_mean(raw)={float(rel_mean or 0.0):.3f}, "
                    f"rel_min(raw)={float(rel_min or 0.0):.3f}, "
                    f"conf_eff_mean={float(conf_eff_mean or 0.0):.3f}, "
                    f"conf_eff_min={float(conf_eff_min or 0.0):.3f}, "
                    f"conf_floor_applied={'YES' if floor_applied else 'NO'}\n"
                )

            # --- Optional source tag (only if caller stored it in stats) ---
            # e.g. st["bass_first_source"] = "WAV" or "TXT/REW"
            src = st.get("bass_first_source", None)
            if isinstance(src, str) and src.strip():
                summary_content += f"BassFirst source: {src.strip()}\n"


 

    return summary_content

def _write_fs_outputs(
    zf,
    data,
    fs_v,
    ft_short,
    file_ts,
    f_l,
    m_l,
    p_l,
    l_imp,
    l_st,
    f_r,
    m_r,
    p_r,
    r_imp,
    r_st,
    *,
    write_dashboards: bool = True,
):
    sum_name = f"Summary_{ft_short}_{fs_v}Hz.txt"
    l_dash_name = f"L_Dashboard_{ft_short}_{fs_v}Hz.png"
    r_dash_name = f"R_Dashboard_{ft_short}_{fs_v}Hz.png"

    summary_content = plots.format_summary_content(data, l_st, r_st)
    # Include explicit house-curve provenance (preset vs upload/local file)
    try:
        hc_src = str(data.get('hc_source', '') or '').strip()
        if hc_src:
            summary_content = f"House curve: {hc_src}\n" + summary_content
    except Exception:
        pass
    summary_content = _append_dsp_effective_params(summary_content, data, fs_v)
    summary_content = _append_acoustic_events(summary_content, l_st, r_st)

    if 'auto_align' in l_st:
        res = l_st['auto_align']
        summary_content += "\n=== AUTO-ALIGN ===\n"
        summary_content += f"Delay: {res['delay_ms']} ms\n"
        summary_content += f"Distance Diff: {res['distance_cm']} cm\n"
        summary_content += f"Gain Diff: {res['gain_diff_db']} dB\n"

    zf.writestr(sum_name, summary_content)

    # Policy: ZIP size control
    # We store only ONE dashboard pair into the ZIP (forced, no UI choice).
    # Dashboard format: PNG only (no HTML), so it opens everywhere without Plotly JS.
    if bool(write_dashboards):
        html_l, fig_l = plots.generate_prediction_plot(
            f_l, m_l, p_l, l_imp, fs_v, "Left",
            None, l_st, data['mixed_freq'], "low",
            create_full_html=False,
            return_fig=True
        )
        if fig_l is not None:
            zf.writestr(l_dash_name, plots.plotly_fig_to_png(fig_l, scale=2))
        else:
            # keep a clue in the ZIP if plotly rendering failed
            zf.writestr(l_dash_name.replace(".png", ".txt"), str(html_l))

        html_r, fig_r = plots.generate_prediction_plot(
            f_r, m_r, p_r, r_imp, fs_v, "Right",
            None, r_st, data['mixed_freq'], "low",
            create_full_html=False,
            return_fig=True
        )
        if fig_r is not None:
            zf.writestr(r_dash_name, plots.plotly_fig_to_png(fig_r, scale=2))
        else:
            zf.writestr(r_dash_name.replace(".png", ".txt"), str(html_r))

    hlc_cfg = generate_hlc_config(fs_v, ft_short, file_ts)
    zf.writestr(f"Config_{ft_short}_{fs_v}Hz.cfg", hlc_cfg)
    yaml_content = generate_raspberry_yaml(
        fs_v,
        ft_short,
        file_ts,
        master_gain_db=float(data.get('gain', 0.0) or 0.0)
    )
    zf.writestr(f"camilladsp_{ft_short}_{fs_v}Hz.yml", yaml_content)

def _render_results(data, f_l, m_l, p_l, f_r, m_r, p_r, l_imp_f, r_imp_f, l_st_f, r_st_f, fname, zip_buffer):
    update_status(t('stat_plot'))
    set_processbar('bar', 1.0)

    with use_scope('results', clear=True):
        if l_st_f is None or r_st_f is None:
            put_error("Error: No results captured.")
            return

        put_success(t('done_msg'))

        # --- Acoustic Intelligence UI (single source of truth: SAME as Summary.txt) ---
        # No separate "measured vs filtered" logic in UI. We display the Summary-based result.
        l_ai = plots.calc_ai_summary_from_stats(l_st_f)
        r_ai = plots.calc_ai_summary_from_stats(r_st_f)

        l_score = float(l_ai.get("score") or 0.0)
        r_score = float(r_ai.get("score") or 0.0)
        avg_pred = (l_score + r_score) / 2.0
        avg_orig = avg_pred
        improvement = 0.0

        l_match = l_ai.get("match")
        r_match = r_ai.get("match")
        if (l_match is None) or (r_match is None):
            avg_match = 0.0
        else:
            avg_match = (float(l_match) + float(r_match)) / 2.0


        put_table([
            ['Speaker', 'L', 'R'],
            ['Target Level', f"{l_st_f.get('eff_target_db', 0):.1f} dB", f"{r_st_f.get('eff_target_db', 0):.1f} dB"],
            ['Smart Scan Range',
             f"{l_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{l_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz",
             f"{r_st_f.get('smart_scan_range', [0,0])[0]:.0f}-{r_st_f.get('smart_scan_range', [0,0])[1]:.0f} Hz"],
            ['Offset to Meas.', f"{l_st_f.get('offset_db', 0):.1f} dB", f"{r_st_f.get('offset_db', 0):.1f} dB"],
            ['Acoustic Confidence', f"{l_st_f.get('avg_confidence', 0):.1f}%", f"{r_st_f.get('avg_confidence', 0):.1f}%"],
            ['Estimated RT60', f"{l_st_f.get('rt60_val', 0):.2f} s", f"{r_st_f.get('rt60_val', 0):.2f} s"],
            ['TDC (Temporal Decay Control)',
             (
                 f"ON ({float(data.get('tdc_strength', 0)):.0f}%, "
                 f"−{float(data.get('tdc_max_reduction_db', 0)):.1f} dB)"
                 if bool(data.get('enable_tdc', False)) else "OFF"
             ),
             (
                 f"ON ({float(data.get('tdc_strength', 0)):.0f}%, "
                 f"−{float(data.get('tdc_max_reduction_db', 0)):.1f} dB)"
                 if bool(data.get('enable_tdc', False)) else "OFF"
             )
            ]
        ])

        put_markdown(f"###  {t('rep_header')}")
        with put_collapse(" DSP info"):
            put_markdown(dedent(f"""
            - **Lenght:** {data['taps']} taps ({data['taps']/data['fs']*1000:.1f} ms)
            - **Resolution:** {data['fs']/data['taps']:.2f} Hz
            - **IR window:** {data['ir_window']} ms
            - **FDW:** {data['fdw_cycles']}
            - **House curve:** {data['hc_mode']} — {data.get('hc_source', 'Unknown')} ({data['mag_c_min']}-{data['mag_c_max']} Hz)
            - **Filter type:** {data['filter_type']}
            - **Smoothing:** {data['lvl_algo']}
            """))

        put_tabs([
            {'title': 'Left Channel', 'content': put_html(plots.generate_prediction_plot(f_l, m_l, p_l, l_imp_f, data['fs'], "Left", None, l_st_f, data['mixed_freq'], "low", create_full_html=False))},
            {'title': 'Right Channel', 'content': put_html(plots.generate_prediction_plot(f_r, m_r, p_r, r_imp_f, data['fs'], "Right", None, r_st_f, data['mixed_freq'], "low", create_full_html=False))}
        ])
        put_file(fname, zip_buffer.getvalue(), label=" DOWNLOAD FILTER ZIP")


def process_run():
    # 1) UI -> data dict (new unified collector)
    data = _collect_ui_data()
    save_config(data)

    # 2) Measurements (file OR REW API) via the new loader
    f_l, m_l, p_l, f_r, m_r, p_r = _load_measurements(data)
    if f_l is None or f_r is None:
        toast("Mittaustiedostot / REW valinnat puuttuvat!", color='red')
        return

    # 3) Target / house curve
    hc_f, hc_m, hc_source = _load_house_curve(data)
    data['hc_source'] = hc_source
    logger.info(f"House curve source: {hc_source}")
    # 4) XO + HPF
    xos, hpf = _build_xos_hpf(data)

    # 5) (Optional) DF smoothing log
    df_on = _log_df_smoothing_toggle()

    # 6) Sample rates list
    target_rates = (
        [44100, 48000, 88200, 96000, 176400, 192000]
        if bool(data.get('multi_rate_opt'))
        else [int(data.get('fs') or 44100)]
    )

    # Forced policy: when multi-rate is enabled, include dashboards only for ONE fs.
    multi_rate_on = bool(data.get('multi_rate_opt'))
    dash_fs = int(FORCE_SINGLE_PLOT_FS_HZ) if multi_rate_on else int(data.get('fs') or 44100)
    if multi_rate_on and dash_fs not in target_rates:
        dash_fs = int(target_rates[0])

    put_processbar('bar')
    put_scope('status_area')
    update_status(t('stat_reading'))
    set_processbar('bar', 0.2)
    zip_buffer = io.BytesIO()
    ts = datetime.now().strftime('%d%m%y_%H%M')
    file_ts = datetime.now().strftime('%H%M_%d%m%y')
    ft_short = _filter_type_short(data['filter_type'])
    split, zoom = data['mixed_freq'], t('zoom_hint')
    l_st_f, r_st_f, l_imp_f, r_imp_f = None, None, None, None

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, fs_v in enumerate(target_rates):
            if data['multi_rate_opt']:
                taps_v = scale_taps_with_fs(fs_v)
                logger.info(f"Auto taps: {int(fs_v)} Hz -> {int(taps_v)} taps (ref 44100 Hz -> 65536 taps)")
            else:
                taps_v = int(data['taps'])
            update_status(f"Lasketaan {fs_v}Hz...")
            set_processbar('bar', 0.2 + 0.6 * (i/len(target_rates)))

            cfg = _build_filter_config(fs_v, taps_v, data, xos, hpf, hc_f, hc_m)


            # Tag measurement source for DSP (bass-first AI tuning).
            # WAV/IR-derived responses often have noisier phase/GD derivatives than REW exports.
            try:
                src = str(data.get('input_source', 'file') or 'file')
            except Exception:
                src = 'file'
            is_wav = False
            if src == 'file':
                try:
                    lp_l = str(data.get('local_path_l', '') or '').lower()
                    lp_r = str(data.get('local_path_r', '') or '').lower()
                except Exception:
                    lp_l, lp_r = '', ''
                try:
                    up_l = str(pin.file_l.get('filename', '') or '').lower() if getattr(pin, 'file_l', None) else ''
                    up_r = str(pin.file_r.get('filename', '') or '').lower() if getattr(pin, 'file_r', None) else ''
                except Exception:
                    up_l, up_r = '', ''
                is_wav = (lp_l.endswith('.wav') or lp_r.endswith('.wav') or up_l.endswith('.wav') or up_r.endswith('.wav') or str(data.get('fmt','')).upper() == 'WAV')
            try:
                setattr(cfg, "is_wav_source", bool(is_wav))
            except Exception:
                pass

            _log_df_smoothing_for_fs(cfg, fs_v, df_on)

            l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, cfg)
            r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, cfg)

            if bool(data.get('stereo_link', False)):
                try:
                    from camillafir_leveling import compute_leveling

                    if isinstance(l_st, dict) and isinstance(r_st, dict):
                        fx_l = np.asarray(l_st.get('freq_axis') or [], dtype=float)
                        fx_r = np.asarray(r_st.get('freq_axis') or [], dtype=float)
                        if fx_l.size > 32 and fx_l.size == fx_r.size and np.allclose(fx_l, fx_r, rtol=0, atol=1e-9):
                            # reconstruct m_anal (A-FDW already applied inside DSP):
                            mL = np.asarray(l_st.get('measured_mags') or [], dtype=float) + float(l_st.get('offset_db', 0.0) or 0.0)
                            mR = np.asarray(r_st.get('measured_mags') or [], dtype=float) + float(r_st.get('offset_db', 0.0) or 0.0)
                            tgt = np.asarray(l_st.get('target_mags') or [], dtype=float)
                            if mL.size == fx_l.size and mR.size == fx_l.size and tgt.size == fx_l.size:
                                m_avg = 0.5 * (mL + mR)

                                (
                                    _tl,
                                    off,
                                    _mlw,
                                    _tlw,
                                    _meth,
                                    smin,
                                    smax,
                                ) = compute_leveling(cfg, fx_l, m_avg, tgt)

                                # Force identical window+offset for both channels
                                cfg.stereo_link = True
                                cfg.lvl_force_window = (float(smin), float(smax))
                                cfg.lvl_force_offset_db = float(off)

                                # Regenerate with forced leveling
                                l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, cfg)
                                r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, cfg)

                                # Tag method in stats for visibility
                                if isinstance(l_st, dict):
                                    l_st['offset_method'] = str(l_st.get('offset_method') or '') + " (StereoLink)"
                                if isinstance(r_st, dict):
                                    r_st['offset_method'] = str(r_st.get('offset_method') or '') + " (StereoLink)"
                except Exception as e:
                    logger.warning(f"Stereo link leveling failed (continuing without link): {e}")


            # ------------------------------------------------------------------
            # RT60 reliability tagging for scoring:
            # - WAV/IR path: RT60 is IR-derived => higher reliability
            # - TXT/REW FR path: RT60 is proxy/estimate => lower reliability
            # This is used by the new Acoustic Score formula (bonus is weighted).
            # ------------------------------------------------------------------
            try:
                rt_rel = 1.0 if bool(is_wav) else 0.25
                rt_src = "WAV" if bool(is_wav) else "TXT/REW"
                if isinstance(l_st, dict):
                    l_st["rt60_reliability"] = float(rt_rel)
                    l_st["rt60_source"] = rt_src
                if isinstance(r_st, dict):
                    r_st["rt60_reliability"] = float(rt_rel)
                    r_st["rt60_source"] = rt_src
            except Exception:
                pass


            l_st = _ensure_scoring_keys(l_st, f_l, m_l, hc_f, hc_m)
            r_st = _ensure_scoring_keys(r_st, f_r, m_r, hc_f, hc_m)
            # Build comparison grid per sample-rate (needed for correct UI scoring with WAV)
            if bool(data.get("comparison_mode", False)):
                try:
                    l_st = plots._make_comparison_stats(l_st, int(fs_v), int(taps_v))
                    r_st = plots._make_comparison_stats(r_st, int(fs_v), int(taps_v))
                except Exception as e:
                    logger.warning(f"Comparison-mode stats failed: {e}")

            # ------------------------------------------------------------------
            # Time alignment
            #
            # TXT-compatible behavior: if generate_filter() produced explicit
            # delay estimates (delay_samples), prefer those for alignment.
            # This avoids the "peak-pick" method drifting when the main impulse
            # peak is not stable (common with heavy LF energy / long tails).
            #
            # If delay_samples are missing, fall back to the legacy peak-pick.
            # ------------------------------------------------------------------
            if data['align_opt']:
                d_s = None

                # Prefer delay_samples (TXT-compatible)
                try:
                    dl = l_st.get('delay_samples', None) if isinstance(l_st, dict) else None
                    dr = r_st.get('delay_samples', None) if isinstance(r_st, dict) else None
                    if dl is not None and dr is not None:
                        dl_i = int(round(float(dl)))
                        dr_i = int(round(float(dr)))
                        d_s = dl_i - dr_i
                except Exception:
                    d_s = None

                # Fallback: align by impulse peak
                if d_s is None:
                    d_s = int(np.argmax(np.abs(l_imp)) - np.argmax(np.abs(r_imp)))

                if d_s > 0:
                    r_imp = np.roll(r_imp, d_s)
                elif d_s < 0:
                    l_imp = np.roll(l_imp, -d_s)

            # UI "results" view: show the same fs as the (single) dashboard fs in multi-rate.
            if fs_v == dash_fs:
                l_st_f, r_st_f, l_imp_f, r_imp_f = l_st, r_st, l_imp, r_imp

            

            if 'delay_samples' in l_st and 'delay_samples' in r_st:
                diff_samples = r_st['delay_samples'] - l_st['delay_samples']
                delay_ms = round((diff_samples / fs_v) * 1000, 3)
                distance_cm = round((delay_ms / 1000) * 34300, 2)
                gain_diff = round(l_st['offset_db'] - r_st['offset_db'], 2)
                l_st['auto_align'] = {'delay_ms': delay_ms, 'distance_cm': distance_cm, 'gain_diff_db': gain_diff}

            wav_l, wav_r = io.BytesIO(), io.BytesIO()
            scipy.io.wavfile.write(wav_l, fs_v, l_imp.astype(np.float32))
            scipy.io.wavfile.write(wav_r, fs_v, r_imp.astype(np.float32))
            zf.writestr(f"L_{ft_short}_{fs_v}Hz_{file_ts}.wav", wav_l.getvalue())
            zf.writestr(f"R_{ft_short}_{fs_v}Hz_{file_ts}.wav", wav_r.getvalue())

            _write_fs_outputs(
                zf,
                data,
                fs_v,
                ft_short,
                file_ts,
                f_l,
                m_l,
                p_l,
                l_imp,
                l_st,
                f_r,
                m_r,
                p_r,
                r_imp,
                r_st,
                write_dashboards=(not multi_rate_on) or (int(fs_v) == int(dash_fs))
            )

    fname = f"CamillaFIR_{ft_short}_{ts}.zip"
    try:
        with open(fname, "wb") as f:
            f.write(zip_buffer.getvalue())
        save_msg = f"Tallennettu: {os.path.abspath(fname)}"
    except Exception:
        save_msg = "Tallennus epäonnistui."


    # --- Ensure UI has stats even if fs selection didn't hit (e.g. WAV/local path quirks) ---
    if l_st_f is None:
        l_st_f = l_st
    if r_st_f is None:
        r_st_f = r_st
    if l_imp_f is None:
        l_imp_f = l_imp
    if r_imp_f is None:
        r_imp_f = r_imp

    # --- Ensure UI scoring has filter_mags (so Measured != Filtered) ---
    try:
        fs_sel = int(data.get('fs') or 44100)
    except Exception:
        fs_sel = 44100
    _inject_filter_mags_for_ui(l_st_f, l_imp_f, fs_sel)
    _inject_filter_mags_for_ui(r_st_f, r_imp_f, fs_sel)

    logger.info(f"UI stats mode L/R: {l_st_f.get('analysis_mode')}/{r_st_f.get('analysis_mode')} | "
                f"len cmp f/m/t = {len(l_st_f.get('cmp_freq_axis',[]))}/{len(l_st_f.get('cmp_measured_mags',[]))}/{len(l_st_f.get('cmp_target_mags',[]))}")

    _render_results(data, f_l, m_l, p_l, f_r, m_r, p_r, l_imp_f, r_imp_f, l_st_f, r_st_f, fname, zip_buffer)

#snipet
def generate_raspberry_yaml(fs, ft_short, file_ts, master_gain_db=0.0):
    import textwrap

    # FIR .wav files (CamillaDSP replaces $samplerate$ at runtime)
    l_wav = f"../coeffs/L_{ft_short}_$samplerate$Hz_{file_ts}.wav"
    r_wav = f"../coeffs/R_{ft_short}_$samplerate$Hz_{file_ts}.wav"

    # sanitize
    try:
        g = float(master_gain_db)
    except Exception:
        g = 0.0

    return textwrap.dedent(f"""
    description: null
    devices:
      capture:
        type: Stdin
        channels: 2
        format: S32LE
      playback:
        type: Alsa
        device: plughw:0,0
        channels: 2
        format: S32LE
      samplerate: {int(fs)}
      chunksize: 4096
      queuelimit: 1
      volume_ramp_time: 150

    filters:
      ir_left:
        type: Conv
        parameters:
          type: Wav
          filename: {l_wav}
          channel: 0

      ir_right:
        type: Conv
        parameters:
          type: Wav
          filename: {r_wav}
          channel: 0

      mastergain:
        type: Gain
        parameters:
          gain: {g:.6g}

    mixers:
      stereo:
        channels:
          in: 2
          out: 2
        mapping:
          - dest: 0
            sources:
              - channel: 0
                gain: 0
          - dest: 1
            sources:
              - channel: 1
                gain: 0

    pipeline:
      - type: Mixer
        name: stereo
      - type: Filter
        channels: [0]
        names: [mastergain, ir_left]
      - type: Filter
        channels: [1]
        names: [mastergain, ir_right]

    processors: null
    title: {ft_short}
    """).strip()




def generate_hlc_config(fs, ft_short, file_ts):
    """
    Luo standardin .cfg konfiguraatiotiedoston (HLC, Convolver VST, BruteFIR).
    Generoi tiedostonimet sisäisesti samoilla säännöillä kuin YAML-funktio.
    """
    # Generoidaan tiedostonimet täsmälleen samalla kaavalla kuin tallennuksessa
    l_name = f"L_{ft_short}_{fs}Hz_{file_ts}.wav"
    r_name = f"R_{ft_short}_{fs}Hz_{file_ts}.wav"

    config = [
        f"{int(fs)} 2 2 0",  # Header: SampleRate, 2 In, 2 Out, 0 Offset
        "0 0",
        "0 0",
        f"{l_name}",         # Vasen tiedosto
        "0",                 # Input Index (L)
        "0.0",
        "0.0",                 # Output Index (L)
        f"{r_name}",         # Oikea tiedosto
        "0",                 # Input Index (R)
        "1.0",
        "1.0"                  # Output Index (R)
    ]
    return "\n".join(config)


def _ui_pick(stats, key):
    """
    UI helper: pick comparison-grid data if analysis_mode == 'comparison'
    """
    if not stats:
        return None
    mode = str(stats.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)


def _pick_cmp(stats, key):
    """
    Return comparison-mode arrays for UI scoring if available.
    """
    if not stats:
        return None
    if str(stats.get("analysis_mode", "native")).lower() == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)


def _ensure_scoring_keys(st, f_in, m_in, hc_f, hc_m):
    """
    Ensure UI scoring keys exist in stats dict (WAV/TXT safe).
    - freq_axis, measured_mags, target_mags, confidence_mask
    """
    try:
        if st is None:
            return st

        f = np.asarray(f_in or [], dtype=float)
        m = np.asarray(m_in or [], dtype=float)
        if f.size > 1 and m.size > 1:
            if st.get("freq_axis") is None:
                st["freq_axis"] = f
            if st.get("measured_mags") is None:
                st["measured_mags"] = m

        # target mags (fallback from house curve if missing)
        if st.get("target_mags") is None:
            try:
                hf = np.asarray(hc_f or [], dtype=float)
                hm = np.asarray(hc_m or [], dtype=float)
                if f.size > 1 and hf.size > 1 and hm.size > 1:
                    st["target_mags"] = np.interp(f, hf, hm)
            except Exception:
                pass

        # confidence mask (fallback to ones if missing)
        if st.get("confidence_mask") is None:
            if f.size > 1:
                st["confidence_mask"] = np.ones_like(f, dtype=float)
        return st
    except Exception:
        return st

_HOUSE_FREQS = np.array([
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0
], dtype=float)

def _resample_to_freq_axis(freqs_dst: np.ndarray, arr: np.ndarray, freqs_src: np.ndarray) -> np.ndarray:
    """Safe 1D interpolation in log-frequency domain."""
    if arr.size == 0 or freqs_src.size == 0 or freqs_dst.size == 0:
        return arr
    # clip to valid region
    f1 = np.maximum(freqs_src.astype(float), 1.0)
    f2 = np.maximum(freqs_dst.astype(float), 1.0)
    lf1 = np.log10(f1)
    lf2 = np.log10(f2)
    # Ensure monotonic source
    order = np.argsort(lf1)
    lf1 = lf1[order]
    a1 = arr.astype(float)[order]
    return np.interp(lf2, lf1, a1, left=a1[0], right=a1[-1])


def calculate_target_match(st):
    """Laskee kuinka hyvin korjattu vaste seuraa tavoitekäyrää (0-100%)."""
    if not st:
        return 0.0

    freqs = np.asarray(_ui_pick(st, 'freq_axis') or [], dtype=float)
    meas  = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    filt  = np.asarray(_ui_pick(st, 'filter_mags') or [], dtype=float)

    if freqs.size == 0 or meas.size == 0 or target.size == 0:
        return 0.0

    # WAV-polulla filter_mags voi puuttua -> tulkitaan 0 dB korjaukseksi
    if filt.size == 0:
        filt = np.zeros_like(meas, dtype=float)
    # If filter mags are missing (common in some UI paths), treat as 0 dB correction
    if filt.size == 0:
        filt = np.zeros_like(meas, dtype=float)

    # If WAV measurement: measured/target are dense FFT grid, but filter may be on 19-point house grid.
    # Resample target/filter to the measurement freq_axis when shapes differ.
    if target.size != freqs.size:
        # common case: target on house grid
        if target.size == _HOUSE_FREQS.size:
            target = _resample_to_freq_axis(freqs, target, _HOUSE_FREQS)
        else:
            # last resort: truncate
            n = min(freqs.size, meas.size, target.size)
            freqs, meas, target = freqs[:n], meas[:n], target[:n]

    if filt.size != freqs.size:
        if filt.size == _HOUSE_FREQS.size:
            filt = _resample_to_freq_axis(freqs, filt, _HOUSE_FREQS)
        else:
            n = min(freqs.size, meas.size, filt.size, target.size)
            freqs, meas, target, filt = freqs[:n], meas[:n], target[:n], filt[:n]

    # RMS virhe (dB) korjatusta vasteesta
    diff = (meas + filt) - target
    rms = float(np.sqrt(np.mean(diff * diff)))

    # Sama muunnos kuin Summaryssä (sigmoidi)
    m0 = 3.2   # dB @ 50%
    s0 = 0.9   # jyrkkyys
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    return float(np.clip(match_pct, 0.0, 100.0))



def _avg_confidence_pct(st: dict) -> float:
    """
    UI helper: returns average confidence in percent.
    Supports comparison-mode keys (cmp_avg_confidence / cmp_confidence_mask).
    """
    if not st:
        return 0.0
    mode = str(st.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        v = st.get("cmp_avg_confidence", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
        cm = np.asarray(st.get("cmp_confidence_mask", []) or [], dtype=float)
        if cm.size:
            return float(np.mean(cm) * 100.0)
        return 0.0
    # native
    v = st.get("avg_confidence", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    cm = np.asarray(st.get("confidence_mask", []) or [], dtype=float)
    if cm.size:
        return float(np.mean(cm) * 100.0)
    return 0.0


def calculate_target_match_unfiltered(st: dict) -> float:
    """
    Target match for *unfiltered* response (measured vs target).
    Uses the same sigmoid mapping as calculate_target_match().
    """
    if not st:
        return 0.0
    meas = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    if meas.size == 0 or target.size == 0:
        return 0.0
    n = min(meas.size, target.size)
    meas, target = meas[:n], target[:n]
    diff = meas - target
    rms = float(np.sqrt(np.mean(diff * diff)))
    m0 = 3.2
    s0 = 0.9
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    return float(np.clip(match_pct, 0.0, 100.0))

def _inject_filter_mags_for_ui(st: dict, filt_ir, fs: int):
    """Ensure st has filter_mags on the same freq_axis as measured, so UI can score 'Filtered' correctly.

    Some pipelines didn't store filter_mags into stats; then UI 'Measured' and 'Filtered' collapse to the same value.
    This computes |FFT(filter_ir)| and interpolates it to st['freq_axis'] (or cmp_freq_axis if in comparison mode),
    storing it as (cmp_)filter_mags in dB.
    """
    try:
        if st is None or filt_ir is None:
            return
        mode = str(st.get("analysis_mode", "native") or "native").lower()
        key_f = "cmp_freq_axis" if mode == "comparison" else "freq_axis"
        key_g = "cmp_filter_mags" if mode == "comparison" else "filter_mags"

        if st.get(key_g) is not None:
            return

        f_axis = np.asarray(st.get(key_f, []) or [], dtype=float)
        if f_axis.size < 4:
            return

        ir = np.asarray(filt_ir, dtype=float).flatten()
        if ir.size < 8:
            return

        fs_i = int(fs) if fs else 0
        if fs_i <= 0:
            return

        h = np.fft.rfft(ir)
        f_fft = np.fft.rfftfreq(ir.size, d=1.0 / fs_i)
        g_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))

        f_min = float(np.min(f_fft))
        f_max = float(np.max(f_fft))
        f_q = np.clip(f_axis, f_min, f_max)
        st[key_g] = np.interp(f_q, f_fft, g_db).tolist()
    except Exception:
        return


def calculate_score(st, is_predicted=False):
    """UI score (0..99) for Measured / Filtered.

    Note: this is *not* the same as Target Curve Match.
    It combines:
      - target match (sigmoid RMS mapping)
      - acoustic confidence
      - optional RT60 room-quality bonus/penalty (scaled by reliability)
    """
    if not st:
        return 0.0

    conf = float(st.get('cmp_avg_confidence', st.get('avg_confidence', 0.0)) or 0.0)
    conf = float(np.clip(conf, 0.0, 100.0))

    meas = np.asarray(_ui_pick(st, 'measured_mags') or [], dtype=float)
    target = np.asarray(_ui_pick(st, 'target_mags') or [], dtype=float)
    filt = np.asarray(_ui_pick(st, 'filter_mags') or [], dtype=float)

    if meas.size == 0 or target.size == 0:
        return float(np.clip(conf, 0.0, 99.0))

    n = min(meas.size, target.size)
    meas, target = meas[:n], target[:n]

    if is_predicted:
        if filt.size >= n:
            filt = filt[:n]
        elif filt.size > 0:
            filt = np.pad(filt, (0, n - filt.size), mode='edge')
        else:
            filt = np.zeros(n, dtype=float)
        diff = (meas + filt) - target
    else:
        diff = meas - target

    rms = float(np.sqrt(np.mean(diff * diff)))
    m0 = 3.2
    s0 = 0.9
    match_pct = 100.0 / (1.0 + np.exp((rms - m0) / s0))
    if rms <= 0.4:
        match_pct = 99.0
    match_pct = float(np.clip(match_pct, 0.0, 100.0))

    base = 0.55 * match_pct + 0.35 * conf  # 0..90

    rt_bonus = 0.0
    try:
        rt = float(st.get('rt60_val', None)) if st.get('rt60_val', None) is not None else None
    except Exception:
        rt = None
    try:
        rel = float(st.get('rt60_reliability', 0.0) or 0.0)
    except Exception:
        rel = 0.0
    rel = float(np.clip(rel, 0.0, 1.0))

    if rt is not None and rt > 0:
        if rt <= 0.35:
            rt_bonus = ((0.35 - rt) / 0.25) * 15.0
        elif rt >= 0.55:
            rt_bonus = -min(15.0, ((rt - 0.55) / 0.35) * 15.0)
        rt_bonus *= rel

    events = st.get('cmp_reflections', st.get('reflections', [])) or []
    penalty_mult = 0.5 if is_predicted else 1.0
    event_penalty = min(8.0, float(len(events)) * 1.0) * penalty_mult

    score = base + rt_bonus - event_penalty
    return float(np.clip(score, 0.0, 99.0))


if __name__ == '__main__':
    start_server(main, port=8080, debug=True, auto_open_webbrowser=True)
