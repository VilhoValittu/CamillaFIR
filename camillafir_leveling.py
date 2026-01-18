"""
CamillaFIR DSP - Tasosovitus (leveling) eriytettynä moduulina.

Tavoite
- Yksi paikka kaikelle tasosovituksen/SmartScanin logiikalle
- Robustit fallbackit (tyhjät maskit, NaN/inf, out-of-range)
- Helppo yksikkötestata

Huom
- Tämä moduuli ei tee I/O:ta eikä riipu CamillaFIR:n muista sisäisistä funktioista.
- Palautusarvot ovat aina määriteltyjä ja finittejä (erityisesti calc_offset_db).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

__all__ = [
    "find_stable_level_window",
    "compute_leveling",
]


def _to_float(x, default: float) -> float:
    """Parsi x floatiksi; jos epäonnistuu tai ei-finite, palauta default."""
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def find_stable_level_window(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray,
    f_min: float,
    f_max: float,
    window_size_octaves: float = 1.0,
    hpf_freq: float = 0.0,
) -> Tuple[float, float]:
    """
    Etsii alueen, jossa mittaus seuraa tavoitekäyrän muotoa vakaimmin.

    Palauttaa (s_min, s_max). Fallbackaa (f_min, f_max) jos ei löydy kelvollista ikkunaa.
    """
    try:
        f_min = _to_float(f_min, 0.0)
        f_max = _to_float(f_max, 0.0)
        hpf_freq = _to_float(hpf_freq, 0.0)
        window_size_octaves = _to_float(window_size_octaves, 1.0)

        if f_min <= 0 or f_max <= 0 or f_min >= f_max:
            return float(f_min), float(f_max)

        # Väistä HPF-aluetta jos mahdollista
        safe_f_min = max(float(f_min), float(hpf_freq) * 1.5)
        if safe_f_min >= float(f_max) * 0.8:
            safe_f_min = float(f_min)

        mask = (freq_axis >= safe_f_min) & (freq_axis <= float(f_max))
        f_search = freq_axis[mask]

        # Tarkastellaan erotusta tavoitteeseen (poistaa tilt-vaikutuksen)
        m_search = (magnitudes - target_mags)[mask]

        # Liian vähän pisteitä => ei luotettavaa ikkunaa
        if f_search.size < 50:
            return float(f_min), float(f_max)

        best_score = float("inf")
        res_min, res_max = float(safe_f_min), float(f_max)

        # liu'uta ikkunaa log-asteikolla
        current_f = float(safe_f_min)
        step = 2 ** (1 / 24.0)  # ~1/24 oktaavi

        while current_f * (2 ** float(window_size_octaves)) <= float(f_max):
            w_start = current_f
            w_end = current_f * (2 ** float(window_size_octaves))
            w_mask = (f_search >= w_start) & (f_search <= w_end)
            if np.any(w_mask):
                # std erotuskäyrästä = "vakaisuus"
                std = float(np.std(m_search[w_mask]))

                # kevyt painotus kohti keskialuetta (ettei valita vain matalinta)
                # (pieni vaikutus, mutta auttaa outoihin dataihin)
                weight = 1.0 + 0.05 * abs(np.log10(max(w_start, 1.0) / 1000.0))
                score = std * weight

                if score < best_score:
                    best_score = score
                    res_min, res_max = float(w_start), float(w_end)

            current_f *= step

        # jos ei löytynyt mitään järkevää
        if not np.isfinite(best_score):
            return float(f_min), float(f_max)

        return float(res_min), float(res_max)

    except Exception:
        return float(f_min), float(f_max)


def compute_leveling(cfg, freq_axis: np.ndarray, m_anal: np.ndarray, target_mags: np.ndarray):
    """
    Laskee tasosovituksen (offset) robustisti.

    Palauttaa:
      target_level_db, calc_offset_db, meas_level_db_window, target_level_db_window,
      offset_method, s_min, s_max

    Huom:
      - calc_offset_db on aina määritelty ja finitti
      - tyhjälle maskille ei tehdä mean/median-opeja
    """
    # perusarvot (aina määritelty)
    target_level_db = 0.0
    calc_offset_db = 0.0
    meas_level_db_window = 0.0
    target_level_db_window = 0.0
    offset_method = "Unknown"

    manual_target_db = _to_float(getattr(cfg, "lvl_manual_db", 75.0), 75.0)

    # user-range (käytetään myös palautuksessa)
    s_min = _to_float(getattr(cfg, "lvl_min", 500.0), 500.0)
    s_max = _to_float(getattr(cfg, "lvl_max", 2000.0), 2000.0)

    # perusvalidointi
    if s_min <= 0 or s_max <= 0 or s_min >= s_max:
        # viimeinen järkevä fallback
        s_min, s_max = 500.0, 2000.0

    mode = str(getattr(cfg, "lvl_mode", "Auto"))

    # ---------- Manual ----------
    if "Manual" in mode:
        mask = (freq_axis >= s_min) & (freq_axis <= s_max)

        if np.any(mask):
            meas_level_db_window = float(np.median(m_anal[mask]))
            target_level_db_window = float(np.median(target_mags[mask]))
            calc_offset_db = float(np.median(m_anal[mask] - target_mags[mask]))
            offset_method = "ManualMedian"
        else:
            calc_offset_db = 0.0
            offset_method = "ManualNoMask"

        # raportoinnin/plotin perusta
        target_level_db = float(manual_target_db)

        if not np.isfinite(calc_offset_db):
            calc_offset_db = 0.0

        return (
            float(target_level_db),
            float(calc_offset_db),
            float(meas_level_db_window),
            float(target_level_db_window),
            str(offset_method),
            float(s_min),
            float(s_max),
        )

    # ---------- Auto / SmartScan ----------
    hpf_freq = 0.0
    hpf_settings = getattr(cfg, "hpf_settings", None)
    if hpf_settings:
        try:
            hpf_freq = _to_float(hpf_settings.get("freq", 0.0), 0.0)
        except Exception:
            hpf_freq = 0.0

    ss_min, ss_max = find_stable_level_window(
        freq_axis,
        m_anal,
        target_mags,
        s_min,
        s_max,
        window_size_octaves=1.0,
        hpf_freq=float(hpf_freq),
    )

    # Validoi ja clampaa user-rangeen
    ss_min = _to_float(ss_min, s_min)
    ss_max = _to_float(ss_max, s_max)
    if (ss_min <= 0) or (ss_max <= 0) or (ss_min >= ss_max):
        ss_min, ss_max = s_min, s_max

    ss_min = max(s_min, ss_min)
    ss_max = min(s_max, ss_max)

    mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)

    # Jos ikkuna liian pieni, yritä perus "hifi" fallback user-rangeen clämpättynä
    if np.count_nonzero(mask) < 20:
        fb_min = max(s_min, 350.0)
        fb_max = min(s_max, 5000.0)
        if fb_min < fb_max:
            ss_min, ss_max = fb_min, fb_max
            mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)

    # Viimeinen fallback: koko user-range
    if np.count_nonzero(mask) < 20:
        ss_min, ss_max = s_min, s_max
        mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)

    if np.any(mask):
        meas_level_db_window = float(np.median(m_anal[mask]))
        target_level_db_window = float(np.median(target_mags[mask]))
        calc_offset_db = float(np.median(m_anal[mask] - target_mags[mask]))
        offset_method = "SmartScanMedian"
    else:
        # TÄRKEÄ: älä tee mitään tyhjille taulukoille.
        calc_offset_db = 0.0
        meas_level_db_window = 0.0
        target_level_db_window = 0.0
        offset_method = "SmartScanNoMask"

    # Plot/raportti-basis: käytä käyttäjän tavoitetasoa
    target_level_db = float(manual_target_db)

    # Safety: pakota finitteys
    if not np.isfinite(calc_offset_db):
        calc_offset_db = 0.0

    return (
        float(target_level_db),
        float(calc_offset_db),
        float(meas_level_db_window),
        float(target_level_db_window),
        str(offset_method),
        float(ss_min),
        float(ss_max),
    )
