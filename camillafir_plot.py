import io, scipy.signal, scipy.fft, scipy.ndimage
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
# Tuodaan tasoitusfunktiot DSP-moduulista
from camillafir_dsp import apply_smoothing_std, psychoacoustic_smoothing

# --- GLOBAALIT APUFUNKTIOT ---

def smooth_complex(freqs, spec, oct_frac=1.0):
    """Tasoittaa kompleksisen vasteen Real ja Imag osat erikseen vaiheen säilyttämiseksi."""
    real_parts = np.nan_to_num(np.real(spec))
    imag_parts = np.nan_to_num(np.imag(spec))
    
    real_s, _ = apply_smoothing_std(freqs, real_parts, np.zeros_like(freqs), oct_frac)
    imag_s, _ = apply_smoothing_std(freqs, imag_parts, np.zeros_like(freqs), oct_frac)
    
    return real_s + 1j * imag_s

def calculate_clean_gd(freqs, complex_resp):
    """Laskee ryhmäviiveen (ms) siististi tasoitetusta kompleksisesta vasteesta."""
    phase_rad = np.unwrap(np.angle(complex_resp))
    df = np.gradient(freqs) + 1e-12
    gd_ms = -np.gradient(phase_rad) / (2 * np.pi * df) * 1000.0
    
    # Poistetaan poikkeamat ja tasoitetaan visualisointia varten
    gd_ms = np.nan_to_num(gd_ms, nan=0.0, posinf=0.0, neginf=0.0)
    return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=8)
def format_summary_content(settings, l_stats, r_stats):
    """Luo kattavan Summary.txt raportin sisältäen akustisen analyysin tulokset."""
    lines = [f"=== CamillaFIR - Filter Generation Summary (v2.6.2 Stable) ===", 
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
    
    lines.append("--- Settings ---")
    for k, v in settings.items():
        if 'file' not in k: lines.append(f"{k}: {v}")
    
    lines.append("\n--- Acoustic Intelligence (v2.6.2) ---")
    lines.append(f"Left Confidence: {l_stats.get('avg_confidence', 0):.1f}% | Right: {r_stats.get('avg_confidence', 0):.1f}%")
    
    def print_refs(refs):
        if not refs: return "   (None detected)"
        r_txt = []
        # Listataan 10 merkittävintä akustista tapahtumaa
        for ref in sorted(refs, key=lambda x: x.get('gd_error', 0), reverse=True)[:10]:
            f = ref.get('freq', 0)
            e = ref.get('gd_error', 0)
            d = ref.get('dist', 0)
            t = ref.get('type', 'Event')
            r_txt.append(f" - {f:>5.0f} Hz: {t:<10} | Virhe: {e:>5.2f}ms | Etäisyys: {d:>5.2f}m")
        return "\n".join(r_txt)

    lines.append("\nDetected Acoustic Events (Left Channel):")
    lines.append(print_refs(l_stats.get('reflections', [])))
    lines.append("\nDetected Acoustic Events (Right Channel):")
    lines.append(print_refs(r_stats.get('reflections', [])))
    
    lines.append("\n--- Alignment & Peaks ---")
    lines.append(f"L Peak (pre-norm): {l_stats.get('peak_before_norm', 0):.2f} dB")
    lines.append(f"R Peak (pre-norm): {r_stats.get('peak_before_norm', 0):.2f} dB")
    lines.append(f"Global Offset applied: {l_stats.get('offset_db', 0):.2f} dB")
    
    return "\n".join(lines)
def generate_prediction_plot(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, save_filename=None, target_stats=None, mixed_split=None, zoom_hint=""):
    """Luo interaktiivisen HTML-dashboardin heijastusmerkinnöillä."""
    try:
        n_fft = len(filt_ir)
        f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt = scipy.fft.rfft(filt_ir)
        
        offset = target_stats.get('offset_db', 0) if target_stats else 0
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        
        m_lin = np.interp(f_lin, orig_freqs, orig_mags)
        p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        
        # Lasketaan ennustettu vaste
        total_spec = 10**((m_lin + offset)/20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt
        spec_sm = smooth_complex(f_lin, total_spec, 1.0)
        
        p_sm = psychoacoustic_smoothing(f_lin, 20*np.log10(np.abs(total_spec)+1e-12))
        ph_sm = (np.rad2deg(np.angle(spec_sm)) + 180) % 360 - 180
        gd_sm = calculate_clean_gd(f_lin, spec_sm)
        filt_db = 20 * np.log10(np.abs(h_filt) + 1e-12)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.08, 
                            subplot_titles=(f"<b>{title} - Magnitude & Confidence</b>", "<b>Phase (1/1 Oct)</b>", "<b>Group Delay (ms)</b>", "<b>Filter Correction (dB)</b>"))

        # Visualisoidaan akustinen luottamus
        if target_stats and 'confidence_mask' in target_stats:
            conf_line = (avg_t - 20) + (target_stats['confidence_mask'] * 10)
            fig.add_trace(go.Scatter(x=target_stats['freq_axis'], y=conf_line, name='Acoustic Confidence', line=dict(color='magenta', width=2), opacity=0.3), row=1, col=1)
            
        # Piirretään heijastukset/moodit (punaiset timantit)
        if target_stats and 'reflections' in target_stats:
            ref_freqs = [r['freq'] for r in target_stats['reflections']]
            ref_mags = [np.interp(r['freq'], f_lin, p_sm) for r in target_stats['reflections']]
            fig.add_trace(go.Scatter(x=ref_freqs, y=ref_mags, mode='markers', marker=dict(symbol='diamond', color='red', size=8), name='Detected Events'), row=1, col=1)
            
            ref_gd = [np.interp(r['freq'], f_lin, gd_sm) for r in target_stats['reflections']]
            fig.add_trace(go.Scatter(x=ref_freqs, y=ref_gd, mode='markers', marker=dict(symbol='diamond', color='red', size=8), showlegend=False), row=3, col=1)

        fig.add_trace(go.Scatter(x=orig_freqs, y=orig_mags + offset, name='Original (Shifted)', line=dict(color='rgba(0,0,255,0.2)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=f_lin, y=p_sm, name='Predicted', line=dict(color='orange', width=3)), row=1, col=1)
        if target_stats: fig.add_trace(go.Scatter(x=target_stats['freq_axis'], y=target_stats['target_mags'], name='Target', line=dict(color='green', dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=f_lin, y=ph_sm, line=dict(color='orange', width=2), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=f_lin, y=gd_sm, line=dict(color='orange', width=2), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=f_lin, y=filt_db, line=dict(color='red', width=1), showlegend=False), row=4, col=1)

        t_vals = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for r in range(1, 5): 
            fig.update_xaxes(type="log", range=[np.log10(20), np.log10(20000)], tickvals=t_vals, row=r, col=1)
        
        fig.update_yaxes(range=[avg_t-40, avg_t+20], row=1, col=1)
        fig.update_yaxes(range=[-180, 180], row=2, col=1)
        fig.update_yaxes(range=[-10, 40], row=3, col=1)
        fig.update_layout(height=1400, width=1750, template="plotly_white", title_text=f"{title} Analysis Dashboard {zoom_hint}")
        return fig.to_html(include_plotlyjs=True, full_html=True)
    except Exception as e: return f"Visual Engine Error: {e}"

def generate_combined_plot_mpl(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, target_stats=None):
    """Luo staattisen PNG-kuvan arkistointiin."""
    try:
        # PNG-kuvassa käytetään logaritmista asteikkoa kaikissa paneeleissa
        n_fft = len(filt_ir); f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs); h_filt = scipy.fft.rfft(filt_ir)
        offset = target_stats.get('offset_db', 0) if target_stats else 0
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        m_lin = np.interp(f_lin, orig_freqs, orig_mags); p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10**((m_lin + offset)/20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt
        spec_sm = smooth_complex(f_lin, total_spec, 1.0)
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18))
        ax1.semilogx(orig_freqs, orig_mags + offset, 'b:', alpha=0.3); ax1.semilogx(f_lin, psychoacoustic_smoothing(f_lin, 20*np.log10(np.abs(total_spec)+1e-12)), 'orange', linewidth=2)
        if target_stats: ax1.semilogx(target_stats['freq_axis'], target_stats['target_mags'], 'g--')
        
        # Piirretään heijastukset myös PNG-kuvaan
        if target_stats and 'reflections' in target_stats:
            rf = [r['freq'] for r in target_stats['reflections']]
            rm = [np.interp(r['freq'], f_lin, psychoacoustic_smoothing(f_lin, 20*np.log10(np.abs(total_spec)+1e-12))) for r in target_stats['reflections']]
            ax1.scatter(rf, rm, color='red', marker='D', s=30, label='Acoustic Events')

        ax1.set_ylim(avg_t-40, avg_t+20); ax2.scatter(f_lin, (np.rad2deg(np.angle(spec_sm)) + 180) % 360 - 180, s=0.5, color='orange')
        ax3.semilogx(f_lin, calculate_clean_gd(f_lin, spec_sm), 'orange'); ax4.semilogx(f_lin, 20*np.log10(np.abs(h_filt)+1e-12), 'r')
        
        for ax in [ax1, ax2, ax3, ax4]: ax.set_xscale('log'); ax.set_xlim(20, 20000); ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except: return None
