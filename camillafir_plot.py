import os
import sys
import io, scipy.signal, scipy.fft, scipy.ndimage
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import copy
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from datetime import datetime
# Tuodaan tarvittavat funktiot DSP-moduulista
from camillafir_dsp import apply_smoothing_std, psychoacoustic_smoothing, calculate_rt60

# Version 1.2.0


def _maybe_shift_to_abs(mags_db, avg_t_db):
    """
    target_stats may contain either:
      A) absolute SPL-like mags already aligned to eff_target_db (preferred, new native path)
      B) relative mags around ~0 dB that still need +avg_t (older paths)

    Heuristic:
      - if median looks "small" (< 40 dB), treat as relative and shift by avg_t
      - otherwise treat as already absolute and return as-is
    """
    try:
        a = np.asarray(mags_db, dtype=float)
        if a.size == 0:
            return a
        med = float(np.nanmedian(a))
        if np.isfinite(med) and med < 40.0:
            return a + float(avg_t_db)
        return a
    except Exception:
        return np.asarray(mags_db, dtype=float)
    
def _align_meas_to_target_window(freqs_hz, meas_db, targ_db, f_min_hz, f_max_hz):
    """
    Force measured & target to overlap in the chosen window.
    Robust: uses median(meas-target) within window.
    Returns meas_db shifted by -median(meas-target) (so window overlap is exact).
    """
    try:
        f = np.asarray(freqs_hz, dtype=float)
        m = np.asarray(meas_db, dtype=float)
        t = np.asarray(targ_db, dtype=float)
        if f.size < 16 or m.size != f.size or t.size != f.size:
            return m
        f_min = float(f_min_hz); f_max = float(f_max_hz)
        if not (np.isfinite(f_min) and np.isfinite(f_max) and f_min > 0 and f_max > f_min):
            return m
        mask = (f >= f_min) & (f <= f_max) & np.isfinite(m) & np.isfinite(t)
        if np.count_nonzero(mask) < 20:
            return m
        off = float(np.median(m[mask] - t[mask]))
        if not np.isfinite(off):
            return m
        return m - off
    except Exception:
        return np.asarray(meas_db, dtype=float)



def _resource_path(rel_path: str) -> str:
    """
    Resource path that works both in dev and PyInstaller (onedir/onefile).
    - In PyInstaller: sys._MEIPASS points to extracted / bundled base.
    - In dev: use directory of this file.
    """
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS  # type: ignore[attr-defined]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, rel_path)

def _plotly_js_path() -> str | None:
    """
    Returns absolute path to local Plotly JS if present, else None.
    """
    p = _resource_path(os.path.join("assets", "plotly.min.js"))
    return p if os.path.isfile(p) else None


def smooth_complex(freqs, spec, oct_frac=1.0):
    """Tasoittaa kompleksisen vasteen Real ja Imag osat erikseen vaiheen säilyttämiseksi."""
    real_parts = np.nan_to_num(np.real(spec))
    imag_parts = np.nan_to_num(np.imag(spec))
    real_s, _ = apply_smoothing_std(freqs, real_parts, np.zeros_like(freqs), oct_frac)
    imag_s, _ = apply_smoothing_std(freqs, imag_parts, np.zeros_like(freqs), oct_frac)
    return real_s + 1j * imag_s

def calculate_clean_gd(freqs, complex_resp):
    """Laskee ryhmäviiveen (ms) tasoitetusta kompleksisesta vasteesta."""
    phase_rad = np.unwrap(np.angle(complex_resp))
    df = np.gradient(freqs) + 1e-12
    gd_ms = -np.gradient(phase_rad) / (2 * np.pi * df) * 1000.0
    gd_ms = np.nan_to_num(gd_ms, nan=0.0, posinf=0.0, neginf=0.0)
    return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=8)

def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float(lo)

def calc_acoustic_score(conf_pct: float, match_pct: float, rt60_s: float | None = None, rt60_rel: float | None = None) -> float:
    """
    Combine confidence + target match into one 0..100 score.
    Weighting: 60% match, 40% confidence.
    (Module-level so UI code can call it.)
    Acoustic Score v2:
      - Base: 55% match + 35% confidence
      - Bonus: up to +15 points for fast decay (low RT60), BUT weighted by RT60 reliability

    RT60 bonus:
      rt_bonus = 15 * clamp((0.35 - rt60) / 0.25, 0..1)
      rt_bonus_eff = rt_bonus * rt60_rel

    Notes:
      - WAV/IR path should tag rt60_rel ~ 1.0
      - TXT/REW FR path should tag rt60_rel ~ 0.25 (proxy / less trustworthy)
    """

    conf = _clamp(conf_pct, 0.0, 100.0)
    match = _clamp(match_pct, 0.0, 100.0)

    # Base score (keeps match dominant but not equal)
    base = 0.55 * match + 0.35 * conf

    # RT60 bonus (optional + reliability-weighted)
    rt_bonus_eff = 0.0
    try:
        if rt60_s is not None:
            rt60 = float(rt60_s)
            if rt60 > 0:
                rel = 1.0 if rt60_rel is None else _clamp(float(rt60_rel), 0.0, 1.0)
                rt_bonus = 15.0 * _clamp((0.35 - rt60) / 0.25, 0.0, 1.0)
                rt_bonus_eff = rt_bonus * rel
    except Exception:
        rt_bonus_eff = 0.0

    return _clamp(base + rt_bonus_eff, 0.0, 100.0)

# Backward-compat aliases (older callers use underscore names)
_calc_acoustic_score = calc_acoustic_score




def calc_ai_summary_from_stats(stats: dict) -> dict:
    """
    Single source of truth for UI + Summary:
      - confidence (%)
      - target match (%), rms (dB)
      - acoustic score (/100)
    Uses the exact same basis as format_summary_content().
    """
    stats = stats or {}
    conf = float(stats.get('cmp_avg_confidence', stats.get('avg_confidence', 0.0)) or 0.0)
    rms, match = calc_target_match_from_stats(stats)
    if match is None:
        return {"conf": conf, "rms": None, "match": None, "score": None}
    # RT60 bonus: reliability-weighted (tagged by camillafir.py)
    rt60 = stats.get("rt60_val", None)
    rt_rel = stats.get("rt60_reliability", None)
    score = calc_acoustic_score(conf, float(match), rt60_s=rt60, rt60_rel=rt_rel)
    return {
        "conf": conf,
        "rms": float(rms) if rms is not None else None,
        "match": float(match),
        "score": score,
        "rt60": float(rt60) if rt60 is not None else None,
        "rt60_rel": float(rt_rel) if rt_rel is not None else None,
    }

def _calc_target_match(stats):
    """
    Palauttaa (rms_db, match_pct) tai (None, None) jos ei dataa.
    Sama logiikka kuin Summary.txt:ssä.
    """
    def _as_np(stats, key):
        v = stats.get(key, None)
        if v is None:
            return None
        try:
            return np.asarray(v, dtype=float)
        except Exception:
            return None

    def _pick(stats, base_key: str):
        if not stats:
            return base_key
        mode = str(stats.get("analysis_mode", "native") or "native").lower()
        if mode == "comparison":
            ck = "cmp_" + base_key
            if ck in stats and stats.get(ck) is not None:
                return ck
        return base_key

    f = _as_np(stats, _pick(stats, 'freq_axis'))
    t = _as_np(stats, _pick(stats, 'target_mags'))
    m = _as_np(stats, _pick(stats, 'measured_mags'))
    c = _as_np(stats, _pick(stats, 'confidence_mask'))

    if f is None or t is None or m is None:
        return None, None

    rng = stats.get(_pick(stats, 'smart_scan_range'), None)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        fmin, fmax = float(rng[0]), float(rng[1])
    else:
        fmin, fmax = 200.0, 5000.0

    mask = (f >= fmin) & (f <= fmax)
    if np.count_nonzero(mask) < 10:
        return None, None

    diff = (m - t)[mask]

    if c is not None and c.shape == f.shape:
        w = np.clip(c[mask], 0.0, 1.0)
        w = np.maximum(w, 0.05)
        rms = float(np.sqrt(np.sum(w * diff * diff) / np.sum(w)))
    else:
        rms = float(np.sqrt(np.mean(diff * diff)))

    m50 = 3.2
    s = 0.9
    match_pct = 100.0 / (1.0 + np.exp((rms - m50) / s))
    match_pct = float(np.clip(match_pct, 0.0, 100.0))
    if rms <= 0.4:
        match_pct = 99.0

    return rms, match_pct

def calc_target_match_from_stats(stats: dict):
    """
    Public wrapper for target-match calculation.
    Returns (rms_db, match_pct) or (None, None) if insufficient data.
    """
    try:
        return _calc_target_match(stats or {})
    except Exception:
        return None, None




def format_summary_content(settings, l_stats, r_stats):
    """Luo Summary.txt sisältäen RT60, confidence, target match ja acoustic score."""
    from datetime import datetime
    import numpy as np

    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}

    lines = [
        "=== CamillaFIR - Filter Generation Summary ===",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    ]

    # --- Settings ---
    lines.append("--- Settings ---")
    for k, v in settings.items():
        if 'file' not in str(k):
            lines.append(f"{k}: {v}")

    # --- Temporal Decay Control (TDC) ---
    try:
        tdc_enabled = bool(settings.get("enable_tdc", False))
        lines.append("\n--- Temporal Decay Control (TDC) ---")
        lines.append(f"TDC enabled: {'YES' if tdc_enabled else 'NO'}")
        if tdc_enabled:
            lines.append(f"TDC strength: {float(settings.get('tdc_strength', 0)):.0f} %")
            lines.append(f"TDC max reduction: {float(settings.get('tdc_max_reduction_db', 0)):.1f} dB")
            slope = float(settings.get('tdc_slope_db_per_oct', 0))
            if slope > 0:
                lines.append(f"TDC slope limit: {slope:.1f} dB/oct")
    except Exception:
        pass

    lines.append("\n--- Acoustic Intelligence (v2.6.3) ---")
    lines.append(f"Analysis mode L: {str((l_stats or {}).get('analysis_mode','native'))} | R: {str((r_stats or {}).get('analysis_mode','native'))}")
    if (l_stats or {}).get('analysis_mode','native') == 'comparison':
        lines.append(f"Comparison grid (L): fs={float(l_stats.get('cmp_ref_fs', 0) or 0):.0f} taps={float(l_stats.get('cmp_ref_taps', 0) or 0):.0f}")
    if (r_stats or {}).get('analysis_mode','native') == 'comparison':
        lines.append(f"Comparison grid (R): fs={float(r_stats.get('cmp_ref_fs', 0) or 0):.0f} taps={float(r_stats.get('cmp_ref_taps', 0) or 0):.0f}")
    # --- Correction guards (reporting) ---
    # Nämä voivat tulla settingsistä tai puuttua (jos UI ei vielä aseta).
    max_cut_db = float(settings.get('max_cut_db', 15.0) or 15.0)
    max_slope = float(settings.get('max_slope_db_per_oct', 12.0) or 12.0)
    # optional (new): separate boost/cut slope; if missing, fall back to legacy
    max_slope_boost = float(settings.get('max_slope_boost_db_per_oct', 0.0) or 0.0) or max_slope
    max_slope_cut   = float(settings.get('max_slope_cut_db_per_oct', 0.0) or 0.0) or max_slope
    low_bass_cut_hz = float(settings.get('low_bass_cut_hz', 40.0) or 40.0)
    if max_slope_boost != max_slope_cut:

        if abs(max_slope_boost - max_slope_cut) > 1e-9:
            lines.append(
                f"Max cut: -{max_cut_db:.1f} dB | "
                f"Slope: boost {max_slope_boost:.1f} / cut {max_slope_cut:.1f} dB/oct | "
                f"Low-bass cut: <{low_bass_cut_hz:.1f} Hz (cuts only)"
            )
        else:
            lines.append(
                f"Max cut: -{max_cut_db:.1f} dB | "
                f"Max slope: {max_slope:.1f} dB/oct | "
                f"Low-bass cut: <{low_bass_cut_hz:.1f} Hz (cuts only)"
            )


    # --- A-FDW debug: active + effective BW range ---
    def _bw_frac(bw_oct: float) -> str:
        # represent ~1/N if close
        try:
            if bw_oct <= 0:
                return "-"
            n = int(round(1.0 / float(bw_oct)))
            if n <= 0:
                return "-"
            # show only for sensible small denominators
            if n in (3, 6, 12, 24, 48):
                return f"~1/{n}"
            return f"~1/{n}"
        except Exception:
            return "-"

    def _afdw_line(side: str, st: dict) -> str:
        st = st or {}
        active = bool(st.get("afdw_active", False))
        if not active:
            # fall back to settings toggle if stats missing
            active = bool((settings or {}).get("enable_afdw", False))
        if not active:
            return f"{side} A-FDW active: NO"

        mn = st.get("afdw_bw_min_oct", None)
        me = st.get("afdw_bw_mean_oct", None)
        mx = st.get("afdw_bw_max_oct", None)
        fmn = st.get("afdw_bw_min_hz", None)
        fmx = st.get("afdw_bw_max_hz", None)
        if mn is None or me is None or mx is None:
            return f"{side} A-FDW active: YES (effective BW not available)"

        return (
            f"{side} A-FDW active: YES | "
            f"BW(min/mean/max)={float(mn):.4f}/{float(me):.4f}/{float(mx):.4f} oct "
            f"({ _bw_frac(float(mn)) } / { _bw_frac(float(me)) } / { _bw_frac(float(mx)) }) | "
            f"min@{float(fmn or 0.0):.0f}Hz max@{float(fmx or 0.0):.0f}Hz"
        )

    lines.append("\n--- A-FDW Debug (effective bandwidth) ---")
    lines.append(_afdw_line("Left", l_stats))
    lines.append(_afdw_line("Right", r_stats))


    # ---- Helpers ----
    def _as_np(stats, key):
        v = stats.get(key, None)
        if v is None:
            return None
        try:
            return np.asarray(v, dtype=float)
        except Exception:
            return None
        

    def _pick(stats, base_key: str):
        if not stats:
            return base_key
        mode = str(stats.get("analysis_mode", "native") or "native").lower()
        if mode == "comparison":
            ck = "cmp_" + base_key
            if ck in stats and stats.get(ck) is not None:
                return ck
            return base_key
        return base_key


    def _band_rt60_line(stats):
        rt = float(stats.get('rt60_val', 0.0) or 0.0)
        band_avg = float(stats.get('rt60_band_avg', 0.0) or 0.0)
        return rt, band_avg

    def _fmt_bands(bands):
        if not bands:
            return "-"
        # näytä muutama tuttu kaista
        picks = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0]
        keys = [float(k) for k in bands.keys()]
        out = []
        for p in picks:
            k = min(keys, key=lambda x: abs(x - p))
            # bands avaimet voi olla float tai str -> hae molemmat
            if k in bands:
                val = bands[k]
            elif str(k) in bands:
                val = bands[str(k)]
            else:
                # fallback: hae lähin oikea avain stringeinäkin
                kk = min(bands.keys(), key=lambda x: abs(float(x) - p))
                val = bands[kk]
                k = float(kk)
            out.append(f"{k:.0f}Hz:{float(val):.2f}s")
        return " | ".join(out) if out else "-"

    

    def _calc_acoustic_score(conf_pct, match_pct, rt60_s=None, rt60_reliability=None):
        """
        Local wrapper for legacy Summary.txt.
        Delegates to module-level calc_acoustic_score (v2).
        """
        try:
            return globals()["calc_acoustic_score"](
                float(conf_pct),
                float(match_pct),
                rt60_s=rt60_s,
                rt60_rel=rt60_reliability
            )
        except Exception:
            conf_pct = float(np.clip(float(conf_pct), 0.0, 100.0))
            match_pct = float(np.clip(float(match_pct), 0.0, 100.0))
            return float(np.clip(0.60 * match_pct + 0.40 * conf_pct, 0.0, 100.0))
    # --- RT60 + Confidence ---
    l_rt, l_band_avg = _band_rt60_line(l_stats)
    r_rt, r_band_avg = _band_rt60_line(r_stats)

    lines.append(f"Left RT60 (wideband): {l_rt:.2f}s | Right RT60 (wideband): {r_rt:.2f}s")
    if (l_band_avg > 0) or (r_band_avg > 0):
        lines.append(f"RT60 band avg (125–4kHz): L {l_band_avg:.2f}s | R {r_band_avg:.2f}s")

    l_bands = l_stats.get('rt60_bands', {}) or {}
    r_bands = r_stats.get('rt60_bands', {}) or {}
    if l_bands or r_bands:
        lines.append(f"Band RT60 L: {_fmt_bands(l_bands)}")
        lines.append(f"Band RT60 R: {_fmt_bands(r_bands)}")

    l_conf = float(l_stats.get('cmp_avg_confidence', l_stats.get('avg_confidence', 0.0)) or 0.0)
    r_conf = float(r_stats.get('cmp_avg_confidence', r_stats.get('avg_confidence', 0.0)) or 0.0)
    lines.append(f"Left Confidence: {l_conf:.1f}% | Right: {r_conf:.1f}%")

    # Offset method
    l_om = (l_stats or {}).get('cmp_offset_method', (l_stats or {}).get('offset_method', '')) or ''
    r_om = (r_stats or {}).get('cmp_offset_method', (r_stats or {}).get('offset_method', '')) or ''
    if l_om or r_om:
        lines.append(f"Offset method: L {l_om or '-'} | R {r_om or '-'}")
    # Level window (diagnostiikka)
    l_win = l_stats.get('cmp_smart_scan_range', l_stats.get('smart_scan_range', None))
    r_win = r_stats.get('cmp_smart_scan_range', r_stats.get('smart_scan_range', None))
    l_mw = float(l_stats.get('cmp_meas_level_db_window', l_stats.get('meas_level_db_window', 0.0)) or 0.0)
    r_mw = float(r_stats.get('cmp_meas_level_db_window', r_stats.get('meas_level_db_window', 0.0)) or 0.0)
    l_tw = float(l_stats.get('cmp_target_level_db_window', l_stats.get('target_level_db_window', 0.0)) or 0.0)
    r_tw = float(r_stats.get('cmp_target_level_db_window', r_stats.get('target_level_db_window', 0.0)) or 0.0)
    if l_win or r_win:
        lines.append(f"Level window L: {l_win} | meas≈{l_mw:.2f} dB, target≈{l_tw:.2f} dB")
        lines.append(f"Level window R: {r_win} | meas≈{r_mw:.2f} dB, target≈{r_tw:.2f} dB")

    # --- Target Curve Match ---
    l_rms, l_match = _calc_target_match(l_stats)
    r_rms, r_match = _calc_target_match(r_stats)

    lines.append("\n--- Target Curve Match ---")
    if l_rms is not None:
        lines.append(f"Left Match:  {l_match:.1f}% | RMS error: {l_rms:.2f} dB")
    else:
        lines.append("Left Match:  (insufficient data)")
    if r_rms is not None:
        lines.append(f"Right Match: {r_match:.1f}% | RMS error: {r_rms:.2f} dB")
    else:
        lines.append("Right Match: (insufficient data)")

    # --- Acoustic Score ---
    lines.append("\n--- Acoustic Score ---")
    if (l_match is not None):
        l_score = _calc_acoustic_score(
            l_conf, l_match,
            (l_stats or {}).get("rt60_val", None),
            (l_stats or {}).get("rt60_reliability", None)
        )
        lines.append(f"Left Acoustic Score:  {l_score:.1f}/100")
    else:
        lines.append("Left Acoustic Score:  (insufficient data)")
    if (r_match is not None):
        r_score = _calc_acoustic_score(
            r_conf, r_match,
            (r_stats or {}).get("rt60_val", None),
            (r_stats or {}).get("rt60_reliability", None)
        )
        lines.append(f"Right Acoustic Score: {r_score:.1f}/100")
    else:
        lines.append("Right Acoustic Score: (insufficient data)")

    # --- Events ---
    def print_refs(refs):
        if not refs:
            return "   (None detected)"
        r_txt = []
        for ref in sorted(refs, key=lambda x: float(x.get('gd_error', 0) or 0), reverse=True)[:10]:
            f = float(ref.get('freq', 0) or 0)
            e = float(ref.get('gd_error', 0) or 0)
            d = float(ref.get('dist', 0) or 0)
            t = str(ref.get('type', 'Event') or 'Event')
            r_txt.append(f" - {f:>5.0f} Hz: {t:<10} | Virhe: {e:>6.2f}ms | Etäisyys: {d:>5.2f}m")
        return "\n".join(r_txt)

    lines.append("\nDetected Acoustic Events (Left):")
    lines.append(print_refs(l_stats.get('reflections', []) or []))
    lines.append("\nDetected Acoustic Events (Right):")
    lines.append(print_refs(r_stats.get('reflections', []) or []))

    # --- Alignment & Peaks ---
    lines.append("\n--- Alignment & Peaks ---")
    lines.append(f"L Peak (pre-norm): {float(l_stats.get('peak_before_norm', 0) or 0):.2f} dB")
    lines.append(f"R Peak (pre-norm): {float(r_stats.get('peak_before_norm', 0) or 0):.2f} dB")
    lines.append(f"Global Offset applied: {float(l_stats.get('offset_db', 0) or 0):.2f} dB")

    # --- Applied gain (UI / CamillaDSP mastergain) ---
    try:
        ui_gain_db = float((settings or {}).get('gain', 0.0) or 0.0)
    except Exception:
        ui_gain_db = 0.0
    lines.append(f"Applied global gain / mastergain: {ui_gain_db:.2f} dB")

    return "\n".join(lines)

# ======================================================================
# FINAL OVERRIDE: Comparison-mode wrapper (locks analysis grid to 44.1 kHz)
# This is appended at EOF on purpose so it always wins even if the file
# contains multiple legacy copies of format_summary_content().
# ======================================================================
_format_summary_content_legacy = format_summary_content

def _make_comparison_stats(stats: dict, ref_fs: int = 44100, ref_taps: int = 65536) -> dict:
    """
    Builds cmp_* fields by resampling native stats onto a fixed reference grid.
    This stabilizes Target Match and Score vs fs/taps changes without requiring DSP changes.
    """
    stats = stats or {}
    out = copy.deepcopy(stats)

    # If DSP already produced coherent cmp-set, keep it.
    if str(out.get("analysis_mode", "native")).lower() == "comparison" and ("cmp_freq_axis" in out):
        return out
    f = out.get("freq_axis", None)
    m = out.get("measured_mags", None)
    t = out.get("target_mags", None)
    g = out.get("filter_mags", None)
    c = out.get("confidence_mask", None)

    if f is None or m is None or t is None:
        return out  
    
    # --- Robustness: remove NaN/inf before interpolation (np.interp propagates NaNs) ---
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    if g is not None:
        g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    if c is not None:
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)


    try:
        f = np.asarray(f, dtype=float)
        m = np.asarray(m, dtype=float)
        t = np.asarray(t, dtype=float)
        g = np.asarray(g, dtype=float) if g is not None else None
        c = np.asarray(c, dtype=float) if c is not None else None
    except Exception:
        return out

    if f.ndim != 1 or f.size < 32 or m is None or t is None:
        return out
    # Require consistent lengths for interpolation
    if (m.ndim != 1) or (t.ndim != 1) or (m.size != f.size) or (t.size != f.size):
        return out
    if (g is not None) and ((g.ndim != 1) or (g.size != f.size)):
        g = None
    if (c is not None) and ((c.ndim != 1) or (c.size != f.size)):
        c = None

    # Reference grid: 0..ref_fs/2 with N=rfft(ref_taps)
    nfft = int(ref_taps)
    if nfft < 1024:
        nfft = 1024
    if (nfft % 2) != 0:
        nfft += 1
    fmax = min(float(ref_fs) / 2.0, float(np.max(f)))
    if fmax <= 10.0:
        return out

    freq_cmp = np.linspace(0.0, fmax, nfft // 2 + 1)

    def _interp(y):
        y = np.asarray(y, dtype=float)
        if y.shape != f.shape:
            return None
        return np.interp(freq_cmp, f, y)

    m_cmp = _interp(m)
    t_cmp = _interp(t)
    g_cmp = _interp(g) if g is not None and g.shape == f.shape else None
    c_cmp = _interp(c) if c is not None and c.shape == f.shape else None
    if m_cmp is None or t_cmp is None:
        return out
    

    # --- Comparison-leveling: re-align measured to target on the comparison grid ---
    # Use smart_scan_range if present; otherwise default to 200..5000 Hz.
    rng = out.get("smart_scan_range", None)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        fmin, fmax_rng = float(rng[0]), float(rng[1])
    else:
        fmin, fmax_rng = 200.0, 5000.0
    mask = (freq_cmp >= fmin) & (freq_cmp <= fmax_rng)
    if np.count_nonzero(mask) >= 20:
        # Median offset is robust vs room modes/outliers
        cmp_offset_db = float(np.median((m_cmp - t_cmp)[mask]))
    else:
        cmp_offset_db = 0.0
    m_cmp = m_cmp - cmp_offset_db

    out["analysis_mode"] = "comparison"
    out["cmp_ref_fs"] = int(ref_fs)
    out["cmp_ref_taps"] = int(ref_taps)
    out["cmp_freq_axis"] = freq_cmp.tolist()
    out["cmp_measured_mags"] = m_cmp.tolist()
    out["cmp_target_mags"] = t_cmp.tolist()
    if g_cmp is not None:
        out["cmp_filter_mags"] = g_cmp.tolist()
    if c_cmp is not None:
        out["cmp_confidence_mask"] = np.clip(c_cmp, 0.0, 1.0).tolist()
        out["cmp_avg_confidence"] = float(np.mean(np.clip(c_cmp, 0.0, 1.0)) * 100.0)
        # A-FDW BW to comparison grid (for BW panel / overlays)
    bw = out.get("afdw_bw_oct", None)
    bw_cmp = _interp(bw) if bw is not None and np.asarray(bw).shape == f.shape else None
    if bw_cmp is not None:
        out["cmp_afdw_bw_oct"] = np.clip(bw_cmp, 1.0/48.0, 1.0/3.0).tolist()
        out["cmp_offset_db"] = float(cmp_offset_db)

    # Keep scan range in Hz (same numbers), but provide cmp_ key so legacy code can use it.
    if "smart_scan_range" in out and isinstance(out["smart_scan_range"], (list, tuple)) and len(out["smart_scan_range"]) == 2:
        out["cmp_smart_scan_range"] = [float(out["smart_scan_range"][0]), float(out["smart_scan_range"][1])]

    # Average confidence for display if present
    if c_cmp is not None:
        out["cmp_avg_confidence"] = float(np.mean(np.clip(c_cmp, 0.0, 1.0)) * 100.0)


    # --- Preserve effective target level from DSP ---
    # Comparison re-alignment must NOT erase the actual target level.
    if "eff_target_db" in stats and stats.get("eff_target_db") is not None:
        try:
            v = float(stats.get("eff_target_db"))
            if np.isfinite(v):
                out["eff_target_db"] = v
                out["cmp_eff_target_db"] = v
        except Exception:
            pass

    if "target_level_db_window" in stats:
        out["cmp_target_level_db_window"] = stats.get("target_level_db_window")

    return out

def format_summary_content(settings, l_stats, r_stats):
    """
    Wrapper that forces comparison-mode analysis (locked to 44.1k grid)
    when settings['comparison_mode'] is True.
    """
    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}

    if bool(settings.get("comparison_mode", False)):
        l_stats = _make_comparison_stats(l_stats, 44100, 65536)
        r_stats = _make_comparison_stats(r_stats, 44100, 65536)

    return _format_summary_content_legacy(settings, l_stats, r_stats)


def generate_prediction_plot(
    orig_freqs, orig_mags, orig_phases, filt_ir, fs, title,
    save_filename=None, target_stats=None, mixed_split=None,
    zoom_hint="", create_full_html=True, return_fig: bool = False
):
    """Luo optimoidun HTML-dashboardin (Pieni tiedostokoko, korkea resoluutio)."""
    try:
        # 1. LASKENTA (Korkea resoluutio)
        MIN_FFT_SIZE = 131072 
        n_fft = max(len(filt_ir) * 4, MIN_FFT_SIZE)
        f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt = scipy.fft.rfft(filt_ir, n=n_fft)
        
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        # Overlap window MUST follow Smart Scan / Manual range if available
        if target_stats and 'smart_scan_range' in target_stats:
            match_range = target_stats.get('smart_scan_range', [500, 2000])
        else:
            match_range = target_stats.get('match_range', [500, 2000]) if target_stats else [500, 2000]
        try:
            f_win_min = float(match_range[0])
            f_win_max = float(match_range[1])
        except Exception:
            f_win_min, f_win_max = 500.0, 2000.0

        # Valmistellaan data lineaarisella akselilla (Heavy)
        if target_stats and 'measured_mags' in target_stats:
            f_stats = np.asarray(target_stats.get('freq_axis', []), dtype=float)
            m_stats = _maybe_shift_to_abs(target_stats.get('measured_mags', []), avg_t)
            t_stats = _maybe_shift_to_abs(target_stats.get('target_mags', []), avg_t) if 'target_mags' in target_stats else None

            m_interp = np.interp(f_lin, f_stats, m_stats)
            # FORCE overlap in chosen window so different targets remain meaningful
            if t_stats is not None and np.asarray(t_stats).size == f_stats.size:
                t_interp = np.interp(f_lin, f_stats, np.asarray(t_stats, dtype=float))
                m_interp = _align_meas_to_target_window(f_lin, m_interp, t_interp, f_win_min, f_win_max)

            m_lin_clean = psychoacoustic_smoothing(f_lin, m_interp)
        else:
            m_raw = np.interp(f_lin, orig_freqs, orig_mags)
            m_lin_clean = psychoacoustic_smoothing(f_lin, m_raw)

        p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10**(m_lin_clean/20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt
        
        # Lasketaan muut käyrät (Heavy)
        p_sm = psychoacoustic_smoothing(f_lin, 20*np.log10(np.abs(total_spec)+1e-12))
        spec_sm = smooth_complex(f_lin, total_spec, 3.0)
        ph_sm = (np.rad2deg(np.angle(spec_sm)) + 180) % 360 - 180
        gd_sm = calculate_clean_gd(f_lin, spec_sm)
        filt_db = 20 * np.log10(np.abs(h_filt) + 1e-12)

        # 2. OPTIMOINTI (Resampling visualisointia varten)
        VIS_POINTS = 4000
        f_vis = np.geomspace(2, fs/2, VIS_POINTS)
        
        m_vis = np.interp(f_vis, f_lin, m_lin_clean)
        p_vis = np.interp(f_vis, f_lin, p_sm)
        ph_vis = np.interp(f_vis, f_lin, ph_sm)
        gd_vis = np.interp(f_vis, f_lin, gd_sm)
        filt_vis = np.interp(f_vis, f_lin, filt_db)

        # --- PIIRTO ---
        fig = make_subplots(
            rows=6, cols=1, vertical_spacing=0.045,
            subplot_titles=(
                "<b>Magnitude & Alignment</b>",
                "<b>Phase</b>",
                "<b>Group Delay</b>",
                "<b>Filter (dB)</b>",
                "<b>Step Response</b>",
                "<b>A-FDW Effective BW (oct)</b>",
            )
        )

        # Smart Scan Range
        if target_stats and 'smart_scan_range' in target_stats:
            s_min, s_max = target_stats['smart_scan_range']
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=s_min, x1=s_max,
                          y0=avg_t-40, y1=avg_t+60,
                          fillcolor="rgba(200, 200, 200, 0.15)", layer="below", line_width=0, row=1, col=1)

        # --- Level reference line (Smart Scan / Manual target level) ---
        # This line shows the level-matching reference that measured & target
        # are aligned to within the chosen window.
        try:
            ref_level = float(avg_t)

            # Determine window text for legend (Smart Scan / Manual)
            if target_stats and 'smart_scan_range' in target_stats:
                _r = target_stats.get('smart_scan_range', None)
            else:
                _r = target_stats.get('match_range', None)

            if isinstance(_r, (list, tuple)) and len(_r) == 2:
                win_label = f"{int(round(_r[0]))}–{int(round(_r[1]))} Hz"
            else:
                win_label = "level window"

            fig.add_shape(
                type="line",
                xref="x", yref="y",
                x0=2.0, x1=fs / 2.0,
                y0=ref_level, y1=ref_level,
                line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"),
                row=1, col=1
            )

            # Add legend entry for level reference (Plotly shapes don't appear in legend)
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=f"Level reference ({win_label})",
                    line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=True
                ),
                row=1,
                col=1
            )
        except Exception:
            pass
        # Correction band (mag correction active range)
        if target_stats:
            try:
                cmin = float(target_stats.get('mag_c_min', 0.0) or 0.0)
                cmax = float(target_stats.get('mag_c_max', 0.0) or 0.0)
                if np.isfinite(cmin) and np.isfinite(cmax) and cmin > 0 and cmax > cmin:
                    fig.add_shape(
                        type="rect", xref="x", yref="y",
                        x0=cmin, x1=cmax,
                        y0=avg_t-40, y1=avg_t+60,
                        fillcolor="rgba(80, 140, 255, 0.08)", layer="below", line_width=0,
                        row=1, col=1
                    )
            except Exception:
                pass


        # Confidence
        if target_stats and 'confidence_mask' in target_stats:
            c_freqs = np.array(target_stats['freq_axis'])
            c_mask = np.array(target_stats['confidence_mask'])
            conf_line = (avg_t - 15) + (c_mask * 10)
            fig.add_trace(go.Scatter(x=c_freqs, y=conf_line, name='Confidence', 
                                     line=dict(color='magenta', width=1), opacity=0.3, hoverinfo='skip'), row=1, col=1)
            # Shade "unreliable" regions (low confidence). This is a *visual cue* only.
            # Threshold is intentionally conservative to avoid over-shading.
            try:
                thr = 0.35
                bad = np.asarray(c_mask, dtype=float) < float(thr)
                if bad.size == c_freqs.size and bad.size > 8:
                    in_seg = False
                    seg_start = None
                    for fx, is_bad in zip(c_freqs, bad):
                        if is_bad and not in_seg:
                            in_seg = True
                            seg_start = float(fx)
                        elif (not is_bad) and in_seg:
                            in_seg = False
                            seg_end = float(fx)
                            if seg_start is not None and seg_end > seg_start:
                                fig.add_shape(
                                    type="rect", xref="x", yref="y",
                                    x0=seg_start, x1=seg_end,
                                    y0=avg_t-40, y1=avg_t+60,
                                    fillcolor="rgba(255, 0, 0, 0.06)", layer="below", line_width=0,
                                    row=1, col=1
                                )
                    if in_seg and seg_start is not None:
                        seg_end = float(c_freqs[-1])
                        if seg_end > seg_start:
                            fig.add_shape(
                                type="rect", xref="x", yref="y",
                                x0=seg_start, x1=seg_end,
                                y0=avg_t-40, y1=avg_t+60,
                                fillcolor="rgba(255, 0, 0, 0.06)", layer="below", line_width=0,
                                row=1, col=1
                            )
            except Exception:
                pass


        # A. MITATTU (Käytetään optimoitua f_vis dataa)
        fig.add_trace(go.Scatter(x=f_vis, y=m_vis, name='Measured', 
                                 line=dict(color='rgba(0,0,255,0.4)', width=1.5)), row=1, col=1)

        # B. TARGET (Alkuperäinen kevyt data + avg_t korjaus)
        if target_stats and 'target_mags' in target_stats:
            t_mags = _maybe_shift_to_abs(target_stats.get('target_mags', []), avg_t)
            fig.add_trace(go.Scatter(x=target_stats['freq_axis'], y=t_mags,
                                     name='Target', line=dict(color='green', dash='dash', width=2.0)), row=1, col=1)

        # C. ENNUSTETTU (Käytetään optimoitua f_vis dataa)
        fig.add_trace(go.Scatter(x=f_vis, y=p_vis, name='Predicted', 
                                 line=dict(color='orange', width=1.5)), row=1, col=1)

        # Muut paneelit
        fig.add_trace(go.Scatter(x=f_vis, y=ph_vis, name="Phase", line=dict(color='orange'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=f_vis, y=gd_vis, name="Group Delay", line=dict(color='orange'), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=f_vis, y=filt_vis, name="Filter dB", line=dict(color='red', width=1.2), showlegend=False), row=4, col=1)
    

        # Mirror correction band hint on the filter panel as well
        if target_stats:
            try:
                cmin = float(target_stats.get('mag_c_min', 0.0) or 0.0)
                cmax = float(target_stats.get('mag_c_max', 0.0) or 0.0)
                if np.isfinite(cmin) and np.isfinite(cmax) and cmin > 0 and cmax > cmin:
                    fig.add_shape(
                        type="rect", xref="x", yref="y",
                        x0=cmin, x1=cmax,
                        y0=-15, y1=10,
                        fillcolor="rgba(80, 140, 255, 0.06)", layer="below", line_width=0,
                        row=4, col=1
                    )
            except Exception:
                pass

        
        # Step Response
        step_resp = np.cumsum(filt_ir)
        step_resp /= (np.max(np.abs(step_resp)) + 1e-12)
        time_axis_ms = (np.arange(len(filt_ir)) / fs) * 1000.0
        fig.add_trace(go.Scatter(x=time_axis_ms[:int(fs*0.05)], y=step_resp[:int(fs*0.05)], name="Step Resp", line=dict(color='yellow')), row=5, col=1)
        fig.update_xaxes(matches=None, row=5, col=1)
        
       
# --- A-FDW BW panel (row 6) ---
        bw_vis = None
        bw_dbg = ""

        mode = "native"
        if target_stats:
            mode = str(target_stats.get("analysis_mode", "native")).lower()

        try:
            if target_stats:
                if mode == "comparison":
                    fx_raw = target_stats.get("cmp_freq_axis")
                    bw_raw = target_stats.get("cmp_afdw_bw_oct")
                else:
                    fx_raw = target_stats.get("freq_axis")
                    bw_raw = target_stats.get("afdw_bw_oct")

                if fx_raw is not None and bw_raw is not None:
                    fx = np.asarray(fx_raw, dtype=float)
                    bw = np.asarray(bw_raw, dtype=float)

                    if fx.size == bw.size and fx.size > 16:
                        bw_vis = np.interp(f_vis, fx, bw)
                        bw_vis = np.clip(bw_vis, 1.0/48.0, 1.0/3.0)
                        bw_vis_smooth = scipy.ndimage.gaussian_filter1d(bw_vis, sigma=5.0)
                        fig.add_trace(
                            go.Scatter(
                                x=f_vis,
                                y=bw_vis_smooth,
                                mode="lines",
                                fill="tozeroy",
                                opacity=0.6,
                                line=dict(width=2),
                                showlegend=False,
                                name="A-FDW BW",
                            ),
                            row=6, col=1
                        )
                    else:
                        bw_dbg = f"shape mismatch: fx={fx.size} bw={bw.size}"
                else:
                    bw_dbg = "missing afdw bw data"
            else:
                bw_dbg = "target_stats is None"
        except Exception as e:
            bw_dbg = f"{type(e).__name__}: {e}"

        if bw_vis is None:
            fig.add_annotation(
                text=f"No A-FDW BW data ({bw_dbg})",
                x=0.5,
                y=0.5,
                showarrow=False,
                row=6,
                col=1
            )


        # Asetukset
        t_vals = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for r in (1, 2, 3, 4, 6):
            fig.update_xaxes(matches="x", row=r, col=1)
            fig.update_xaxes(type="log", range=[np.log10(2), np.log10(20000)], tickvals=t_vals, row=r, col=1)


        fig.update_yaxes(range=[avg_t-20, avg_t+30], row=1, col=1)
        fig.update_yaxes(range=[-180, 180], row=2, col=1)
        fig.update_yaxes(range=[-30, 12], row=4, col=1)
        # Y-axis for A-FDW BW panel
        if bw_vis is not None and len(bw_vis) > 0:
            bw_lo = max(1.0/48.0, float(np.min(bw_vis)) * 0.9)
            bw_hi = min(1.0/3.0,  float(np.max(bw_vis)) * 1.1)
            if bw_hi - bw_lo < 1e-6:
                bw_lo, bw_hi = (1.0/48.0, 1.0/3.0)
            fig.update_yaxes(range=[bw_lo, bw_hi], row=6, col=1) 
        else:
            fig.update_yaxes(range=[1.0/48.0, 1.0/3.0], row=6, col=1)

        fig.update_yaxes(title_text="oct", row=6, col=1)

        fig.update_layout(
            height=1780,
            width=1750,
            template="plotly_white",
            title_text=f"{title} Analysis",
            # Keep UI state stable across redraws (doesn't stop doubleclick by itself,
            # but prevents other "reset weirdness" when page re-renders)
            uirevision="keep"
        )
        
        # Use local Plotly JS when generating full HTML (offline-safe).
        # If local JS is missing, fall back to CDN.
        if create_full_html:
            if _plotly_js_path():
                js_mode = "plotly.min.js"   # suhteellinen polku
            else:
                js_mode = "cdn"
        else:
            js_mode = "require"

        # Plotly UI config:
        # - Disable double-click autoscale/reset (it breaks with matched log axes)
        # - Keep scroll-zoom enabled for easier navigation
        config = {
            "responsive": True,
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": False
        }

        html = fig.to_html(
            include_plotlyjs=js_mode,
            full_html=create_full_html,
            config=config
        )
        if bool(return_fig):
            return html, fig
        return html

    except Exception as e:
        msg = f"Visual Engine Error: {str(e)}"
        if bool(return_fig):
            return msg, None
        return msg

def plotly_fig_to_png(fig, *, scale=2):
    """
    Export Plotly figure to PNG bytes (same as HTML modebar download).
    Uses Plotly 6.x default Kaleido backend.
    """
    try:
        import plotly.io as pio
        # Plotly 6.x: kaleido is implicit backend
        return pio.to_image(fig, format="png", scale=int(scale))
    except Exception as e:
        raise RuntimeError(
            f"Plotly PNG export failed: {e}"
        )


def generate_combined_plot_mpl(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, target_stats=None):
    """Luo staattisen PNG-kuvan."""
    try:
        n_fft = len(filt_ir); f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs); h_filt = scipy.fft.rfft(filt_ir)
        offset = target_stats.get('offset_db', 0) if target_stats else 0
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        m_lin = np.interp(f_lin, orig_freqs, orig_mags); p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10**((m_lin + offset)/20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18))
        ax1.semilogx(orig_freqs, orig_mags + offset, 'b:', alpha=0.3)
        ax1.semilogx(f_lin, psychoacoustic_smoothing(f_lin, 20*np.log10(np.abs(total_spec)+1e-12)), 'orange', linewidth=2)
        if target_stats: ax1.semilogx(target_stats['freq_axis'], target_stats['target_mags'], 'g--')
        
        # Haetaan rajat stats-sanakirjasta.
        if target_stats and 'smart_scan_range' in target_stats:
            f_min, f_max = target_stats['smart_scan_range']
            ax1.axvline(f_min, color='red', linestyle='--', alpha=0.6, label=f'Final Min: {f_min:.0f}Hz')
            ax1.axvline(f_max, color='green', linestyle='--', alpha=0.6, label=f'Final Max: {f_max:.0f}Hz')
            ax1.legend(loc='upper right', fontsize='small')
        
        # KORJATTU: Poistettu NameErroria aiheuttaneet ax1.axvline(final_min/max) rivit.
        
        ax1.set_ylim(avg_t-15, avg_t+15)
        ax3.semilogx(f_lin, calculate_clean_gd(f_lin, total_spec), 'orange')
        ax4.semilogx(f_lin, 20*np.log10(np.abs(h_filt)+1e-12), 'r')
        
        for ax in [ax1, ax2, ax3, ax4]: ax.set_xscale('log'); ax.set_xlim(20, 20000); ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"Virhe visualisoinnissa ({title}): {e}")
        return b""
    
