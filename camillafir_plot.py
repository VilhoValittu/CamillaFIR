import io
import scipy.signal
import scipy.fft
import scipy.ndimage
import numpy as np
import matplotlib
matplotlib.use('Agg') # Estää ikkunoiden aukeamisen palvelimella
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# IMPORT DSP HELPER FOR SMOOTHING
from camillafir_dsp import apply_smoothing_std, psychoacoustic_smoothing, calculate_group_delay

def format_summary_content(settings, l_stats, r_stats):
    lines = []
    lines.append(f"=== CamillaFIR - Filter Generation Summary ===")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("--- Settings ---")
    for key, val in settings.items():
        if 'file' not in key:
            lines.append(f"{key}: {val}")
    lines.append("\n--- Analysis (20Hz - 20kHz) ---")
    
    lines.append(f"Left Channel GD: {l_stats.get('gd_min',0):.2f} ms to {l_stats.get('gd_max',0):.2f} ms")
    lines.append(f"Right Channel GD: {r_stats.get('gd_min',0):.2f} ms to {r_stats.get('gd_max',0):.2f} ms")
    
    lines.append(f"Left Target Level: {l_stats.get('eff_target_db',0):.2f} dB")
    lines.append(f"Right Target Level: {r_stats.get('eff_target_db',0):.2f} dB")
    
    lines.append(f"Left: Peak={l_stats.get('peak_before_norm',0):.2f}dB, Norm={l_stats.get('normalized', False)}")
    lines.append(f"Right: Peak={r_stats.get('peak_before_norm',0):.2f}dB, Norm={r_stats.get('normalized', False)}")
    return "\n".join(lines)

def generate_filter_plot_plotly(filt_ir, fs, title, zoom_hint=""):
    try:
        n_fft = len(filt_ir)
        n_plot = max(n_fft, 65536) 
        w, h = scipy.signal.freqz(filt_ir, 1, worN=n_plot, fs=fs)
        freqs = w
        mags_db = 20 * np.log10(np.abs(h) + 1e-12)
        phases_rad = np.unwrap(np.angle(h))
        phases_deg = np.rad2deg(phases_rad)
        
        fig = make_subplots(rows=2, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.15, 
                            subplot_titles=(f"{title} - Filter Magnitude (dB)", f"{title} - Filter Phase (deg)"))

        fig.add_trace(go.Scatter(x=freqs, y=mags_db, mode='lines', name='Magnitude', line=dict(color='red', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=freqs, y=phases_deg, mode='lines', name='Phase', line=dict(color='blue', width=1.5)), row=2, col=1)

        fig.update_xaxes(type="log", title_text="Frequency (Hz)", showticklabels=True, range=[np.log10(20), np.log10(20000)], row=1, col=1)
        fig.update_xaxes(type="log", title_text="Frequency (Hz)", range=[np.log10(20), np.log10(20000)], row=2, col=1)
        fig.update_yaxes(title_text="dB", row=1, col=1)
        fig.update_yaxes(title_text="Deg", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=True, template="plotly_white", width=1100, title_text=f"{title} {zoom_hint}")
        return fig.to_html(include_plotlyjs=True, full_html=True) 
    except Exception as e:
        print(f"Filter Plot Error: {e}")
        return None

def generate_prediction_plot(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, save_filename=None, target_stats=None, mixed_split=None, zoom_hint=""):
    try:
        n_fft = len(filt_ir)
        freq_axis_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt_complex = scipy.fft.rfft(filt_ir)
        orig_mags_lin = np.interp(freq_axis_lin, orig_freqs, orig_mags)
        orig_phases_lin = np.interp(freq_axis_lin, orig_freqs, orig_phases)
        orig_complex_lin = 10**(orig_mags_lin/20.0) * np.exp(1j * np.deg2rad(orig_phases_lin))
        total_complex = orig_complex_lin * h_filt_complex
        
        ir_total = scipy.fft.irfft(total_complex, n=n_fft)
        peak_idx = np.argmax(np.abs(ir_total))
        ir_centered = np.roll(ir_total, -peak_idx)
        total_complex_centered = scipy.fft.rfft(ir_centered)
        
        final_mags_lin = 20 * np.log10(np.abs(total_complex) + 1e-12)
        final_phases_rad = np.angle(total_complex_centered)
        final_phases_unwrap = np.unwrap(final_phases_rad)
        final_phases_deg = np.rad2deg(final_phases_unwrap)
        
        final_mags_plot = np.interp(orig_freqs, freq_axis_lin, final_mags_lin)
        final_phases_plot = np.interp(orig_freqs, freq_axis_lin, final_phases_deg)
        
        plot_orig_var = psychoacoustic_smoothing(orig_freqs, orig_mags)
        plot_pred_var = psychoacoustic_smoothing(orig_freqs, final_mags_plot)
        
        _, plot_phase_orig = apply_smoothing_std(orig_freqs, orig_mags, orig_phases, 1.0)
        _, plot_phase_pred = apply_smoothing_std(orig_freqs, final_mags_plot, final_phases_plot, 1.0)
        plot_phase_orig = (plot_phase_orig + 180) % 360 - 180
        plot_phase_pred = (plot_phase_pred + 180) % 360 - 180
        
        # Local helper for GD calc
        def calc_gd(freqs, phases_deg):
             phase_rad = np.unwrap(np.deg2rad(phases_deg))
             d_phi_d_f = np.gradient(phase_rad, freqs)
             gd_sec = -d_phi_d_f / (2 * np.pi)
             gd_ms = gd_sec * 1000.0
             return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=3)

        gd_orig = calc_gd(orig_freqs, plot_phase_orig) 
        gd_pred = calc_gd(orig_freqs, plot_phase_pred)

        # 2. FILTER RESPONSE CALCULATION
        w, h = scipy.signal.freqz(filt_ir, 1, worN=max(n_fft, 65536), fs=fs)
        filt_freqs = w
        filt_mags_db = 20 * np.log10(np.abs(h) + 1e-12)
        
        # 3. CREATE PLOTLY FIGURE
        fig = make_subplots(rows=4, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.10, 
                            subplot_titles=(f"{title} - Magnitude (dB)", f"{title} - Phase (deg)", f"{title} - Group Delay (ms)", f"{title} - Filter (dB)"))

        # Row 1: Magnitude
        fig.add_trace(go.Scatter(x=orig_freqs, y=plot_orig_var, mode='lines', name='Original', line=dict(color='blue', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=orig_freqs, y=plot_pred_var, mode='lines', name='Predicted', line=dict(color='orange', width=2)), row=1, col=1)
        
        # --- TARGET CURVE LOGIC (PLOTLY) ---
        if target_stats and 'target_mags' in target_stats:
            # Check dimensions match freq_axis
            if len(target_stats['target_mags']) == len(target_stats['freq_axis']):
                t_freqs = target_stats['freq_axis']
                t_mags = target_stats['target_mags']
                mask = t_freqs > 10
                fig.add_trace(go.Scatter(x=t_freqs[mask], y=t_mags[mask], mode='lines', name='Target', line=dict(color='green', dash='dash', width=1.5)), row=1, col=1)
        
        # --- CALC AREA SHADING (PLOTLY) ---
        if target_stats and 'l_match_min' in target_stats:
            fig.add_vrect(
                x0=target_stats['l_match_min'], 
                x1=target_stats['l_match_max'],
                fillcolor="gray", opacity=0.15, layer="below", line_width=0,
                row=1, col=1
            )

        # Row 2: Phase
        fig.add_trace(go.Scatter(x=orig_freqs, y=plot_phase_orig, mode='lines', name='Orig Phase', line=dict(color='blue', width=0.5, dash='dot'), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=orig_freqs, y=plot_phase_pred, mode='lines', name='Pred Phase', line=dict(color='orange', width=1), showlegend=False), row=2, col=1)

        # Row 3: Group Delay
        fig.add_trace(go.Scatter(x=orig_freqs, y=gd_orig, mode='lines', name='Orig GD', line=dict(color='blue', width=0.5, dash='dot'), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=orig_freqs, y=gd_pred, mode='lines', name='Pred GD', line=dict(color='orange', width=1.5), showlegend=False), row=3, col=1)

        # Row 4: Filter
        fig.add_trace(go.Scatter(x=filt_freqs, y=filt_mags_db, mode='lines', name='Filter', line=dict(color='red', width=1.5)), row=4, col=1)

        # Split Point Line
        if mixed_split:
             for r in range(1, 5):
                 fig.add_vline(x=mixed_split, line_width=1, line_dash="dash", line_color="green", row=r, col=1)

        fig.update_xaxes(type="log", title_text="Frequency (Hz)", showticklabels=True, range=[np.log10(20), np.log10(20000)], row=1, col=1)
        fig.update_xaxes(type="log", title_text="Frequency (Hz)", showticklabels=True, range=[np.log10(20), np.log10(20000)], row=2, col=1)
        fig.update_xaxes(type="log", title_text="Frequency (Hz)", showticklabels=True, range=[np.log10(20), np.log10(20000)], row=3, col=1)
        fig.update_xaxes(type="log", title_text="Frequency (Hz)", range=[np.log10(20), np.log10(20000)], row=4, col=1)
        
        fig.update_yaxes(title_text="dB", row=1, col=1)
        fig.update_yaxes(title_text="Deg", row=2, col=1, range=[-180, 180])
        fig.update_yaxes(title_text="ms", row=3, col=1, range=[-20, 100])
        fig.update_yaxes(title_text="dB", row=4, col=1)
        
        fig.update_layout(height=1100, showlegend=True, template="plotly_white", width=1100, title_text=f"{title} {zoom_hint}")
        
        return fig.to_html(include_plotlyjs=True, full_html=True) 
    except Exception as e:
        print(f"Plotly Error: {e}")
        return None

def generate_combined_plot_mpl(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, target_stats=None):
    try:
        # 1. Prediction Calculation
        n_fft = len(filt_ir)
        freq_axis_lin = scipy.fft.rfftfreq(n_fft, d=1/fs)
        h_filt_complex = scipy.fft.rfft(filt_ir)
        orig_mags_lin = np.interp(freq_axis_lin, orig_freqs, orig_mags)
        orig_phases_lin = np.interp(freq_axis_lin, orig_freqs, orig_phases)
        orig_complex_lin = 10**(orig_mags_lin/20.0) * np.exp(1j * np.deg2rad(orig_phases_lin))
        total_complex = orig_complex_lin * h_filt_complex
        
        ir_total = scipy.fft.irfft(total_complex, n=n_fft)
        peak_idx = np.argmax(np.abs(ir_total))
        ir_centered = np.roll(ir_total, -peak_idx)
        total_complex_centered = scipy.fft.rfft(ir_centered)
        
        final_mags_lin = 20 * np.log10(np.abs(total_complex) + 1e-12)
        final_phases_rad = np.angle(total_complex_centered)
        final_phases_deg = np.rad2deg(np.unwrap(final_phases_rad))
        
        final_mags_plot = np.interp(orig_freqs, freq_axis_lin, final_mags_lin)
        final_phases_plot = np.interp(orig_freqs, freq_axis_lin, final_phases_deg)
        
        plot_orig_var = psychoacoustic_smoothing(orig_freqs, orig_mags)
        plot_pred_var = psychoacoustic_smoothing(orig_freqs, final_mags_plot)
        
        _, plot_phase_orig = apply_smoothing_std(orig_freqs, orig_mags, orig_phases, 1.0)
        _, plot_phase_pred = apply_smoothing_std(orig_freqs, final_mags_plot, final_phases_plot, 1.0)
        plot_phase_orig = (plot_phase_orig + 180) % 360 - 180
        plot_phase_pred = (plot_phase_pred + 180) % 360 - 180
        
        # Calculate Group Delay for MPL
        def calc_gd(freqs, phases_deg):
             phase_rad = np.unwrap(np.deg2rad(phases_deg))
             d_phi_d_f = np.gradient(phase_rad, freqs)
             gd_sec = -d_phi_d_f / (2 * np.pi)
             gd_ms = gd_sec * 1000.0
             return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=3)

        gd_orig = calc_gd(orig_freqs, plot_phase_orig)
        gd_pred = calc_gd(orig_freqs, plot_phase_pred)
        
        # 2. Filter Calculation
        w, h = scipy.signal.freqz(filt_ir, 1, worN=max(n_fft, 65536), fs=fs)
        filt_freqs = w
        filt_mags_db = 20 * np.log10(np.abs(h) + 1e-12)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
        
        # PLOT A: MAGNITUDE
        ax1.semilogx(orig_freqs, plot_orig_var, label='Original', color='blue', alpha=0.6)
        ax1.semilogx(orig_freqs, plot_pred_var, label='Predicted', color='orange')
        
        # --- TARGET CURVE LOGIC (MPL) ---
        if target_stats and 'target_mags' in target_stats:
            # Check dimensions match freq_axis
            if len(target_stats['target_mags']) == len(target_stats['freq_axis']):
                t_freqs = target_stats['freq_axis']
                t_mags = target_stats['target_mags']
                mask = t_freqs > 10
                ax1.semilogx(t_freqs[mask], t_mags[mask], label='Target', color='green', linestyle='--', alpha=0.8)
        
        # --- CALC AREA SHADING (MPL) ---
        if target_stats and 'l_match_min' in target_stats:
            ax1.axvspan(target_stats['l_match_min'], target_stats['l_match_max'], color='gray', alpha=0.15, label='Calc Area')
            
        ax1.set_title(f"{title} - Magnitude")
        ax1.set_ylabel("Amplitude (dB)")
        ax1.grid(True, which="both", alpha=0.3)
        ax1.legend(loc='upper right', fontsize='small')
        
        # PLOT B: PHASE
        ax2.semilogx(orig_freqs, plot_phase_orig, label='Orig Phase', color='blue', alpha=0.6, linestyle=':')
        ax2.semilogx(orig_freqs, plot_phase_pred, label='Pred Phase', color='orange')
        ax2.set_title("Phase")
        ax2.set_ylabel("Phase (deg)")
        ax2.set_ylim(-180, 180)
        ax2.grid(True, which="both", alpha=0.3)
        ax2.legend(loc='upper right', fontsize='small')

        # PLOT C: GROUP DELAY
        ax3.semilogx(orig_freqs, gd_orig, label='Orig GD', color='blue', alpha=0.6, linestyle=':')
        ax3.semilogx(orig_freqs, gd_pred, label='Pred GD', color='orange')
        ax3.set_title("Group Delay")
        ax3.set_ylabel("Time (ms)")
        ax3.set_ylim(-20, 100)
        ax3.grid(True, which="both", alpha=0.3)
        ax3.legend(loc='upper right', fontsize='small')

        # PLOT D: FILTER
        ax4.semilogx(filt_freqs, filt_mags_db, label='Filter', color='red')
        ax4.set_title("Filter Response")
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Gain (dB)")
        ax4.set_xlim(20, 20000)
        ax4.grid(True, which="both", alpha=0.3)
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"MPL Plot Error: {e}")
        return None
