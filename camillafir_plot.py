import io, scipy.signal, scipy.fft, scipy.ndimage
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
# Tuodaan tarvittavat funktiot DSP-moduulista
from camillafir_dsp import apply_smoothing_std, psychoacoustic_smoothing, calculate_rt60

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
    """Luo kattavan Summary.txt raportin sisältäen RT60-analyysin."""
    from datetime import datetime
    lines = [f"=== CamillaFIR - Filter Generation Summary ===", 
             f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"]
    
    lines.append("--- Settings ---")
    for k, v in settings.items():
        if 'file' not in k: lines.append(f"{k}: {v}")
    
# 2. RT60-laskenta (Käytetään huoneen vastetta suodattimen sijaan)
    fs = settings.get('fs', 44100)
    l_rt = l_stats.get('rt60_val', 0.0)
    r_rt = r_stats.get('rt60_val', 0.0)

    lines.append("\n--- Acoustic Intelligence (v2.6.3) ---")
    lines.append(f"Left RT60 (T20): {l_rt}s | Right RT60 (T20): {r_rt}s")
    lines.append(f"Left Confidence: {l_stats.get('avg_confidence', 0):.1f}% | Right: {r_stats.get('avg_confidence', 0):.1f}%")
    
    def print_refs(refs):
        if not refs: return "   (None detected)"
        r_txt = []
        # KORJAUS: Käytetään avainta 'gd_error' joka tulee dsp-moottorista
        for ref in sorted(refs, key=lambda x: x.get('gd_error', 0), reverse=True)[:10]:
            f = ref.get('freq', 0)
            e = ref.get('gd_error', 0)
            d = ref.get('dist', 0)
            t = ref.get('type', 'Event')
            r_txt.append(f" - {f:>5.0f} Hz: {t:<10} | Virhe: {e:>6.2f}ms | Etäisyys: {d:>5.2f}m")
        return "\n".join(r_txt)

    lines.append("\nDetected Acoustic Events (Left):")
    lines.append(print_refs(l_stats.get('reflections', [])))
    lines.append("\nDetected Acoustic Events (Right):")
    lines.append(print_refs(r_stats.get('reflections', [])))
    
    lines.append("\n--- Alignment & Peaks ---")
    lines.append(f"L Peak (pre-norm): {l_stats.get('peak_before_norm', 0):.2f} dB")
    lines.append(f"R Peak (pre-norm): {r_stats.get('peak_before_norm', 0):.2f} dB")
    lines.append(f"Global Offset applied: {l_stats.get('offset_db', 0):.2f} dB")
    return "\n".join(lines)

def generate_prediction_plot(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, save_filename=None, target_stats=None, mixed_split=None, zoom_hint=""):
    """Luo 5-paneelisen HTML-dashboardin."""
    try:
        n_fft = len(filt_ir)
        f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt = scipy.fft.rfft(filt_ir)
        offset = target_stats.get('offset_db', 0) if target_stats else 0
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        
        m_lin = np.interp(f_lin, orig_freqs, orig_mags)
        p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10**((m_lin + offset)/20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt
        spec_sm = smooth_complex(f_lin, total_spec, 1.0)
        p_sm = psychoacoustic_smoothing(f_lin, 20*np.log10(np.abs(total_spec)+1e-12))
        ph_sm = (np.rad2deg(np.angle(spec_sm)) + 180) % 360 - 180
        gd_sm = calculate_clean_gd(f_lin, spec_sm)
        filt_db = 20 * np.log10(np.abs(h_filt) + 1e-12)

        fig = make_subplots(rows=5, cols=1, vertical_spacing=0.05, 
                            subplot_titles=("<b>Magnitude & Confidence</b>", "<b>Phase</b>", "<b>Group Delay</b>", "<b>Filter (dB)</b>", "<b>Step Response</b>"))

        # Akustinen luottamus
        if target_stats and 'confidence_mask' in target_stats:
            conf_line = (avg_t - 20) + (target_stats['confidence_mask'] * 10)
            fig.add_trace(go.Scatter(x=target_stats['freq_axis'], y=conf_line, name='Confidence', line=dict(color='magenta', width=2), opacity=0.3), row=1, col=1)

        fig.add_trace(go.Scatter(x=orig_freqs, y=orig_mags + offset, name='Original', line=dict(color='rgba(0,0,255,0.2)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=f_lin, y=p_sm, name='Predicted', line=dict(color='orange', width=3)), row=1, col=1)
        if target_stats: fig.add_trace(go.Scatter(x=target_stats['freq_axis'], y=target_stats['target_mags'], name='Target', line=dict(color='green', dash='dash')), row=1, col=1)

        fig.add_trace(go.Scatter(x=f_lin, y=ph_sm, line=dict(color='orange'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=f_lin, y=gd_sm, line=dict(color='orange'), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=f_lin, y=filt_db, line=dict(color='red'), showlegend=False), row=4, col=1)
        
        # Step Response - KORJATTU
        step_resp = np.cumsum(filt_ir)
        step_resp /= np.max(np.abs(step_resp))
        time_axis_ms = (np.arange(len(filt_ir)) / fs) * 1000.0
        fig.add_trace(go.Scatter(x=time_axis_ms[:int(fs*0.2)], y=step_resp[:int(fs*0.2)], name="Step", line=dict(color='yellow')), row=5, col=1)

        t_vals = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for r in range(1, 5): 
            fig.update_xaxes(type="log", range=[np.log10(20), np.log10(20000)], tickvals=t_vals, row=r, col=1)
        
        fig.update_yaxes(range=[avg_t-40, avg_t+20], row=1, col=1)
        fig.update_yaxes(range=[-180, 180], row=2, col=1)
        fig.update_layout(height=1600, width=1750, template="plotly_white", title_text=f"{title} Analysis {zoom_hint}")
        return fig.to_html(include_plotlyjs=True, full_html=True)
    except Exception as e: return f"Visual Engine Error: {e}"

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
        
        ax1.set_ylim(avg_t-40, avg_t+20)
        ax3.semilogx(f_lin, calculate_clean_gd(f_lin, total_spec), 'orange')
        ax4.semilogx(f_lin, 20*np.log10(np.abs(h_filt)+1e-12), 'r')
        
        for ax in [ax1, ax2, ax3, ax4]: ax.set_xscale('log'); ax.set_xlim(20, 20000); ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=120); plt.close(fig); buf.seek(0)
        return buf.getvalue()
    except: return None
