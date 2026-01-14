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

#--- Plot v.1.1.1 (Fixed NameErrors)

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
    
    fs = settings.get('fs', 44100)
    l_rt = l_stats.get('rt60_val', 0.0)
    r_rt = r_stats.get('rt60_val', 0.0)

    lines.append("\n--- Acoustic Intelligence (v2.6.3) ---")
    lines.append(f"Left RT60 (T20): {l_rt}s | Right RT60 (T20): {r_rt}s")
    lines.append(f"Left Confidence: {l_stats.get('avg_confidence', 0):.1f}% | Right: {r_stats.get('avg_confidence', 0):.1f}%")
    
    def print_refs(refs):
        if not refs: return "   (None detected)"
        r_txt = []
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
    """Luo 5-paneelisen HTML-dashboardin, jossa käyrät on kohdistettu Smart Scan -alueelle."""
    try:
        n_fft = len(filt_ir)
        f_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt = scipy.fft.rfft(filt_ir)
        
        # 1. Määritetään tasonsovituksen referenssi (avg_t)
        # Tämä on taso (esim. 75dB), jonka ympärille kaikki piirretään.
        avg_t = target_stats.get('eff_target_db', 75) if target_stats else 75
        match_range = target_stats.get('match_range', [500, 2000]) # Smart Scan range

        # 2. Haetaan mittausdata. Jos stats on käytössä, käytetään DSP:n valmiiksi kohdistamaa dataa.
        if target_stats and 'measured_mags' in target_stats:
            # DSP on jo laskenut: measured_mags = Raw - offset_db
            # Joten tämä on n. 0 dB tasolla (suhteessa Targettiin).
            m_stats = np.array(target_stats['measured_mags'])
            f_stats = np.array(target_stats['freq_axis'])
            
            # Nostetaan visuaaliselle tasolle (+ avg_t)
            m_interp = np.interp(f_lin, f_stats, m_stats) + avg_t
            m_lin_clean = psychoacoustic_smoothing(f_lin, m_interp)
        else:
            # Fallback ilman stats-dataa
            m_raw = np.interp(f_lin, orig_freqs, orig_mags)
            m_lin_clean = psychoacoustic_smoothing(f_lin, m_raw)

        # 3. Lasketaan ennustettu vaste (Predicted)
        p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        # Predicted = Aligned Meas + Filter
        total_spec = 10**(m_lin_clean/20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt
        p_sm = psychoacoustic_smoothing(f_lin, 20*np.log10(np.abs(total_spec)+1e-12))
        
        # 4. Muut kuvaajat (Phase, GD, Step)
        spec_sm = smooth_complex(f_lin, total_spec, 3.0)
        ph_sm = (np.rad2deg(np.angle(spec_sm)) + 180) % 360 - 180
        gd_sm = calculate_clean_gd(f_lin, spec_sm)
        filt_db = 20 * np.log10(np.abs(h_filt) + 1e-12)

        # --- PIIRTO ---
        fig = make_subplots(rows=5, cols=1, vertical_spacing=0.05, 
                            subplot_titles=("<b>Magnitude & Alignment</b>", "<b>Phase</b>", "<b>Group Delay</b>", "<b>Filter (dB)</b>", "<b>Step Response</b>"))

        # Harmaa laatikko näyttämään Smart Scan -alueen
        if target_stats and 'smart_scan_range' in target_stats:
            s_min, s_max = target_stats['smart_scan_range']
            fig.add_shape(type="rect", xref="x", yref="y",
                          x0=s_min, x1=s_max,
                          y0=avg_t-40, y1=avg_t+40,
                          fillcolor="rgba(200, 200, 200, 0.15)", layer="below", line_width=0,
                          row=1, col=1)
            # Pystyviivat rajoille
            for val, label in zip([s_min, s_max], ["Start", "End"]):
                fig.add_vline(x=val, line_dash="dot", line_color="gray", line_width=1, row=1, col=1)

        # Confidence-viiva (Lisäinfo)
        if target_stats and 'confidence_mask' in target_stats:
            conf_mask = np.array(target_stats['confidence_mask'])
            conf_line = (avg_t - 15) + (conf_mask * 10) # Piirretään alas
            fig.add_trace(go.Scatter(x=target_stats['freq_axis'], y=conf_line, name='Confidence', 
                                     line=dict(color='magenta', width=1), opacity=0.3, hoverinfo='skip'), row=1, col=1)

        # A. MITATTU (Sininen)
        fig.add_trace(go.Scatter(x=f_lin, y=m_lin_clean, name='Measured', 
                                 line=dict(color='rgba(0,0,255,0.4)', width=1.5)), row=1, col=1)

        # B. TARGET (Vihreä katkoviiva)
        if target_stats and 'target_mags' in target_stats:
            # TÄRKEÄ: Lisätään avg_t, jotta se nousee 0dB -> 75dB
            t_mags = np.array(target_stats['target_mags']) + avg_t
            fig.add_trace(go.Scatter(x=target_stats['freq_axis'], y=t_mags,
                                     name='Target', line=dict(color='green', dash='dash', width=2.0)), row=1, col=1)

        # C. ENNUSTETTU (Oranssi)
        fig.add_trace(go.Scatter(x=f_lin, y=p_sm, name='Predicted', 
                                 line=dict(color='orange', width=2.5)), row=1, col=1)

        # Muut paneelit
        fig.add_trace(go.Scatter(x=f_lin, y=ph_sm, name="Phase", line=dict(color='orange'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=f_lin, y=gd_sm, name="Group Delay", line=dict(color='orange'), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=f_lin, y=filt_db, name="Filter dB", line=dict(color='red'), showlegend=False), row=4, col=1)
        
        step_resp = np.cumsum(filt_ir)
        step_resp /= (np.max(np.abs(step_resp)) + 1e-12)
        time_axis_ms = (np.arange(len(filt_ir)) / fs) * 1000.0
        fig.add_trace(go.Scatter(x=time_axis_ms[:int(fs*0.05)], y=step_resp[:int(fs*0.05)], name="Step Resp", line=dict(color='yellow')), row=5, col=1)

        # Asetukset ja zoom
        t_vals = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        for r in range(1, 5):
            fig.update_xaxes(type="log", range=[np.log10(20), np.log10(20000)], tickvals=t_vals, row=r, col=1)

        # Zoomataan Y-akseli tavoitetason (avg_t) ympärille
        fig.update_yaxes(range=[avg_t-20, avg_t+20], row=1, col=1)
        fig.update_yaxes(range=[-180, 180], row=2, col=1)
        fig.update_yaxes(range=[-15, 10], row=3, col=1)
        fig.update_yaxes(range=[-15, 10], row=4, col=1)

        fig.update_layout(height=1600, width=1750, template="plotly_white", title_text=f"{title} Analysis")
        return fig.to_html(include_plotlyjs=True, full_html=True)
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
