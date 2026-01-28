# camillafir_housecurve.py
import numpy as np
from pywebio.pin import pin


def _normalize_hc_mode_key(v) -> str:
    """
    Convert UI label / legacy saved strings into a stable preset key.
    This fixes the bug where translated labels caused preset matching to fall back.
    """
    try:
        s = str(v or "")
    except Exception:
        s = ""

    # Already a valid keys
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


def load_target_curve(file_content: bytes):
    """
    Reads target curve from text file and ensures correct ordering.
    """
    try:
        content_str = file_content.decode("utf-8")
        lines = content_str.split("\n")
        freqs, mags = [], []
        for line in lines:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.replace(",", ".").split()
            if len(parts) >= 2:
                try:
                    f = float(parts[0])
                    m = float(parts[1])
                    if f > 0:
                        freqs.append(f)
                        mags.append(m)
                except ValueError:
                    continue

        if len(freqs) < 2:
            return None, None

        freqs = np.array(freqs)
        mags = np.array(mags)
        if np.mean(mags) > 30:
            mags -= np.mean(mags)

        sort_idx = np.argsort(freqs)
        return freqs[sort_idx], mags[sort_idx]
    except Exception:
        return None, None


def load_house_curve(data: dict, *, parse_measurements_from_path=None):
    """
    1) Upload (pin.hc_custom_file) -> load_target_curve()
    2) local_path_house -> parse_measurements_from_path() (callback)
    3) preset -> get_house_curve_by_name()

    Returns: (hc_f, hc_m, hc_source)
    """
    hc_f, hc_m = None, None
    hc_source = "Preset"

    # 1) Upload
    try:
        if getattr(pin, "hc_custom_file", None):
            hc_f, hc_m = load_target_curve(pin.hc_custom_file["content"])
            hc_source = "Upload"
    except Exception:
        pass

    # 2) Local file
    if hc_f is None and data.get("local_path_house"):
        if callable(parse_measurements_from_path):
            try:
                hc_f, hc_m, _ = parse_measurements_from_path(data["local_path_house"])
                if hc_f is not None:
                    s_idx = np.argsort(hc_f)
                    hc_f, hc_m = hc_f[s_idx], hc_m[s_idx]
                    hc_source = "LocalFile"
            except Exception:
                hc_f, hc_m = None, None

    # 3) Preset fallback
    if hc_f is None:
        preset_key = _normalize_hc_mode_key(data.get("hc_mode"))
        preset_key = "Flat" if preset_key == "Custom" else preset_key
        hc_f, hc_m = get_house_curve_by_name(preset_key)
        hc_source = f"Preset ({preset_key})"

    return hc_f, hc_m, hc_source
