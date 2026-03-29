import numpy as np


def _normalize_hc_mode_key(v) -> str:
    """Sisainen apufunktio: normalize hc mode key."""
    try:
        s = str(v or "")
    except Exception:
        s = ""

    known = {
        "Harman6", "Harman8", "Harman4", "Harman10", "Harman12",
        "Studio", "Nearfield", "HiFi", "Speech",
        "Toole", "BK_Light", "BK_Medium", "BK_Strong", "Flat", "Cinema", "Custom",
        "Adaptive",
    }
    if s in known:
        return s

    n = s.lower().replace(" ", "")
    if "adaptive" in n:
        return "Adaptive"
    if "custom" in n or "lataa" in n or "upload" in n:
        return "Upload"
    if "cinema" in n:
        return "Cinema"
    if "flat" in n:
        return "Flat"
    if "toole" in n:
        return "Toole"
    if "b&k" in n or "bk" in n:
        if "light" in n:
            return "BK_Light"
        if "strong" in n:
            return "BK_Strong"
        return "BK_Medium"
    if "speech" in n or "broadcast" in n:
        return "Speech"
    if "nearfield" in n or "desk" in n:
        return "Nearfield"
    if "hifi" in n or "loudness" in n:
        return "HiFi"
    if "studio" in n or "tilt" in n:
        return "Studio"
    if "harman" in n:
        if "+12db" in n or "12db" in n:
            return "Harman12"
        if "+10db" in n or "10db" in n or "subheavy" in n:
            return "Harman10"
        if "+8db" in n or "8db" in n:
            return "Harman8"
        if "+4db" in n or "4db" in n:
            return "Harman4"
        return "Harman6"

    return "Harman6"


def get_house_curve_by_name(name):

    full_freqs = np.array([
        0.0,
        20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0,
        160.0, 200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0,
        8000.0, 16000.0, 20000.0
    ])

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

    elif 'Harman12' in name or '+12dB' in name:
        freqs = full_freqs
        mags = np.array([
            12.0,
            12.0, 11.8, 11.4, 10.8, 9.8, 8.6, 7.2, 5.8, 4.2,
            2.8, 1.0, 0.0, -0.5, -1.0, -1.8, -2.8,
            -4.0, -5.5, -6.0
        ])

    elif 'BK_Light' in name or 'B&K Light' in name:
        freqs = full_freqs
        mags = np.array([
            2.5,
            2.5, 2.4, 2.3, 2.1, 1.9, 1.6, 1.3, 1.0, 0.7,
            0.4, 0.1, -0.2, -0.6, -1.0, -1.6, -2.3,
            -3.1, -4.0, -4.4
        ])

    elif 'BK_Strong' in name or 'B&K Strong' in name:
        freqs = full_freqs
        mags = np.array([
            4.5,
            4.5, 4.4, 4.2, 3.9, 3.5, 3.0, 2.4, 1.8, 1.2,
            0.8, 0.3, -0.2, -0.8, -1.5, -2.3, -3.2,
            -4.3, -5.4, -6.0
        ])

    elif 'BK_Medium' in name or 'BK' in name or 'B&K' in name:
        freqs = full_freqs
        mags = np.array([
            3.5,
            3.5, 3.4, 3.2, 3.0, 2.7, 2.3, 1.9, 1.4, 1.0,
            0.6, 0.2, -0.2, -0.7, -1.3, -2.0, -2.8,
            -3.8, -4.8, -5.3
        ])

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

    elif 'Cinema' in name:
        freqs = np.array([
            0.0, 20.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0
        ])
        mags = np.array([
            0.0, 0.0, 0.0, -3.0, -9.0, -15.0, -18.0
        ])

    elif 'Flat' in name or 'Adaptive' in name:
        freqs = full_freqs
        mags = np.zeros_like(freqs)

    else:
        freqs = full_freqs
        mags = np.array([
            6.0,
            6.0, 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5,
            1.4, 0.4, 0.0, -0.5, -1.0, -1.8, -2.8,
            -4.0, -5.5, -6.0
        ])
    return freqs, mags
