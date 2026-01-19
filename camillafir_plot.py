import io, scipy.signal, scipy.fft, scipy.ndimage
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import copy
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
# Tuodaan tarvittavat funktiot DSP-moduulista
from camillafir_dsp import apply_smoothing_std, psychoacoustic_smoothing, calculate_rt60

#--- Plot v.2.7.4
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

    def _calc_target_match(stats):
        """
        Palauttaa (rms_db, match_pct) tai (None, None) jos ei dataa.
        match_pct = 100 - 10*rms_db, clamp 0..100 (helppo ja intuitiivinen)
        """
        f = _as_np(stats, _pick(stats, 'freq_axis'))
        t = _as_np(stats, _pick(stats, 'target_mags'))
        m = _as_np(stats, _pick(stats, 'measured_mags'))
        c = _as_np(stats, _pick(stats, 'confidence_mask'))

        if f is None or t is None or m is None:
            return None, None

        # Käytä smart_scan_range jos löytyy, muuten järkevä oletus
        rng = stats.get(_pick(stats, 'smart_scan_range')) if ('smart_scan_range' in stats or 'cmp_smart_scan_range' in stats) else None
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            fmin, fmax = float(rng[0]), float(rng[1])
        else:
            fmin, fmax = 200.0, 5000.0

        mask = (f >= fmin) & (f <= fmax)
        if np.count_nonzero(mask) < 10:
            return None, None

        diff = (m - t)[mask]

        # Painota confidence:lla kevyesti (ettei heikko alue dominoi)
        if c is not None and c.shape == f.shape:
            w = np.clip(c[mask], 0.0, 1.0)
            # estä nollapainot (numeriikka)
            w = np.maximum(w, 0.05)
            rms = float(np.sqrt(np.sum(w * diff * diff) / np.sum(w)))
        else:
            rms = float(np.sqrt(np.mean(diff * diff)))

        # Fiksumpi match-muunnos RMS(dB) -> prosentti:
        # Sigmoidi: pehmeä pudotus, realistisempi kuuntelun kannalta.
        m = 3.2   # dB @ 50%
        s = 0.9   # jyrkkyys (pienempi = jyrkempi)
        match_pct = 100.0 / (1.0 + np.exp((rms - m) / s))
        match_pct = float(np.clip(match_pct, 0.0, 100.0))
        if rms <= 0.4:
            match_pct = 99.0

        return rms, match_pct

    def _calc_acoustic_score(conf_pct, match_pct):
        """
        Yhdistää confidence + target match yhdeksi pisteeksi 0..100.
        Painotus: 60% match, 40% confidence (match on "kuultava", confidence kertoo riskistä).
        """
        conf_pct = float(np.clip(conf_pct, 0.0, 100.0))
        match_pct = float(np.clip(match_pct, 0.0, 100.0))
        score = 0.60 * match_pct + 0.40 * conf_pct
        return float(np.clip(score, 0.0, 100.0))

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
        l_score = _calc_acoustic_score(l_conf, l_match)
        lines.append(f"Left Acoustic Score:  {l_score:.1f}/100")
    else:
        lines.append("Left Acoustic Score:  (insufficient data)")
    if (r_match is not None):
        r_score = _calc_acoustic_score(r_conf, r_match)
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

    out["cmp_offset_db"] = float(cmp_offset_db)

    # Keep scan range in Hz (same numbers), but provide cmp_ key so legacy code can use it.
    if "smart_scan_range" in out and isinstance(out["smart_scan_range"], (list, tuple)) and len(out["smart_scan_range"]) == 2:
        out["cmp_smart_scan_range"] = [float(out["smart_scan_range"][0]), float(out["smart_scan_range"][1])]

    # Average confidence for display if present
    if c_cmp is not None:
        out["cmp_avg_confidence"] = float(np.mean(np.clip(c_cmp, 0.0, 1.0)) * 100.0)

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


def generate_prediction_plot(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, save_filename=None, target_stats=None, mixed_split=None, zoom_hint="", create_full_html=True):
    """Luo optimoidun HTML-dashboardin (Pieni tiedostokoko, korkea resoluutio)."""
    try:
        # 1. LASKENTA (Korkea resoluutio)
        MIN_FFT_SIZE = 131072 
        n_fft = max(len(filt_ir) * 4, MIN_FFT_SIZE)
        f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt = scipy.fft.rfft(filt_ir, n=n_fft)
        
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        match_range = target_stats.get('match_range', [500, 2000])

        # Valmistellaan data lineaarisella akselilla (Heavy)
        if target_stats and 'measured_mags' in target_stats:
            m_stats = np.array(target_stats['measured_mags'])
            f_stats = np.array(target_stats['freq_axis'])
            m_interp = np.interp(f_lin, f_stats, m_stats) + avg_t
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
        fig = make_subplots(rows=5, cols=1, vertical_spacing=0.05, 
                            subplot_titles=("<b>Magnitude & Alignment</b>", "<b>Phase</b>", "<b>Group Delay</b>", "<b>Filter (dB)</b>", "<b>Step Response</b>"))

        # Smart Scan Range
        if target_stats and 'smart_scan_range' in target_stats:
            s_min, s_max = target_stats['smart_scan_range']
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=s_min, x1=s_max,
                          y0=avg_t-40, y1=avg_t+40,
                          fillcolor="rgba(200, 200, 200, 0.15)", layer="below", line_width=0, row=1, col=1)

        # Confidence
        if target_stats and 'confidence_mask' in target_stats:
            c_freqs = np.array(target_stats['freq_axis'])
            c_mask = np.array(target_stats['confidence_mask'])
            conf_line = (avg_t - 15) + (c_mask * 10)
            fig.add_trace(go.Scatter(x=c_freqs, y=conf_line, name='Confidence', 
                                     line=dict(color='magenta', width=1), opacity=0.3, hoverinfo='skip'), row=1, col=1)

        # A. MITATTU (Käytetään optimoitua f_vis dataa)
        fig.add_trace(go.Scatter(x=f_vis, y=m_vis, name='Measured', 
                                 line=dict(color='rgba(0,0,255,0.4)', width=1.5)), row=1, col=1)

        # B. TARGET (Alkuperäinen kevyt data + avg_t korjaus)
        if target_stats and 'target_mags' in target_stats:
            t_mags = np.array(target_stats['target_mags']) + avg_t
            fig.add_trace(go.Scatter(x=target_stats['freq_axis'], y=t_mags,
                                     name='Target', line=dict(color='green', dash='dash', width=2.0)), row=1, col=1)

        # C. ENNUSTETTU (Käytetään optimoitua f_vis dataa)
        fig.add_trace(go.Scatter(x=f_vis, y=p_vis, name='Predicted', 
                                 line=dict(color='orange', width=1.5)), row=1, col=1)

        # Muut paneelit
        fig.add_trace(go.Scatter(x=f_vis, y=ph_vis, name="Phase", line=dict(color='orange'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=f_vis, y=gd_vis, name="Group Delay", line=dict(color='orange'), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=f_vis, y=filt_vis, name="Filter dB", line=dict(color='red', width=1.2), showlegend=False), row=4, col=1)
        
        # Step Response
        step_resp = np.cumsum(filt_ir)
        step_resp /= (np.max(np.abs(step_resp)) + 1e-12)
        time_axis_ms = (np.arange(len(filt_ir)) / fs) * 1000.0
        fig.add_trace(go.Scatter(x=time_axis_ms[:int(fs*0.05)], y=step_resp[:int(fs*0.05)], name="Step Resp", line=dict(color='yellow')), row=5, col=1)

        # Asetukset
        t_vals = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for r in range(1, 5):
            fig.update_xaxes(type="log", range=[np.log10(2), np.log10(20000)], tickvals=t_vals, row=r, col=1)

        fig.update_yaxes(range=[avg_t-20, avg_t+20], row=1, col=1)
        fig.update_yaxes(range=[-180, 180], row=2, col=1)
        fig.update_yaxes(range=[-15, 10], row=4, col=1)

        fig.update_layout(height=1600, width=1750, template="plotly_white", title_text=f"{title} Analysis")
        
        # Use local Plotly JS when generating full HTML (offline-safe).
        # If local JS is missing, fall back to CDN.
        if create_full_html:
            local_js = _plotly_js_path()
            js_mode = local_js if local_js else "cdn"
        else:
            # Embedded mode: keep legacy behavior
            js_mode = "require"

        return fig.to_html(include_plotlyjs=js_mode, full_html=create_full_html)

        
    except Exception as e: return f"Visual Engine Error: {str(e)}"
        


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
    
