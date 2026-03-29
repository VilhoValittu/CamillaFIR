import io

import matplotlib
import numpy as np
import scipy.fft

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_common import (
    GD_SMOOTH_OCT,
    PHASE_SMOOTH_OCT,
    _filter_focus_band,
    _robust_axis_range,
    calculate_clean_gd,
    logger,
    psychoacoustic_smoothing,
    remove_ir_peak_delay,
    smooth_complex,
)


def plotly_fig_to_png(fig, *, scale=2, width=None, height=None):
    try:
        import plotly.io as pio

        kwargs = {"format": "png", "scale": float(scale)}
        if width is not None:
            kwargs["width"] = int(width)
        if height is not None:
            kwargs["height"] = int(height)
        return pio.to_image(fig, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Plotly PNG export failed: {e}")


def generate_combined_plot_mpl(orig_freqs, orig_mags, orig_phases, filt_ir, fs, title, target_stats=None):
    try:
        n_fft = len(filt_ir)
        f_lin = scipy.fft.rfftfreq(n_fft, d=1 / fs)
        h_filt = scipy.fft.rfft(filt_ir)
        h_filt_display, _filt_delay_ms = remove_ir_peak_delay(f_lin, h_filt, filt_ir, fs)
        offset = target_stats.get("offset_db", 0) if target_stats else 0
        avg_t = target_stats.get("eff_target_db", 75) if target_stats else 75
        m_lin = np.interp(f_lin, orig_freqs, orig_mags)
        p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10 ** ((m_lin + offset) / 20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt
        filt_phase = smooth_complex(f_lin, h_filt_display, PHASE_SMOOTH_OCT)
        filt_phase_deg = (np.rad2deg(np.angle(filt_phase)) + 180) % 360 - 180
        filt_gd = calculate_clean_gd(f_lin, smooth_complex(f_lin, h_filt_display, GD_SMOOTH_OCT))
        filt_db = 20 * np.log10(np.abs(h_filt) + 1e-12)
        focus_band = _filter_focus_band(f_lin, filt_db)
        gd_range = _robust_axis_range(
            f_lin,
            filt_gd,
            focus_band=focus_band,
            q_lo=0.02,
            q_hi=0.98,
            pad_ratio=0.16,
            min_span=6.0,
            max_span=120.0,
            include_zero=True,
        )
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 18))
        ax1.semilogx(orig_freqs, orig_mags + offset, "b:", alpha=0.3)
        ax1.semilogx(f_lin, psychoacoustic_smoothing(f_lin, 20 * np.log10(np.abs(total_spec) + 1e-12)), "orange", linewidth=2)
        if target_stats:
            ax1.semilogx(target_stats["freq_axis"], target_stats["target_mags"], "g--")

        if target_stats and "smart_scan_range" in target_stats:
            f_min, f_max = target_stats["smart_scan_range"]
            ax1.axvline(f_min, color="red", linestyle="--", alpha=0.6, label=f"Final Min: {f_min:.0f}Hz")
            ax1.axvline(f_max, color="green", linestyle="--", alpha=0.6, label=f"Final Max: {f_max:.0f}Hz")
            ax1.legend(loc="upper right", fontsize="small")

        ax1.set_ylim(avg_t - 15, avg_t + 15)
        ax2.semilogx(f_lin, filt_phase_deg, "orange", linewidth=0.9)
        ax2.set_ylim(-90, 90)
        ax3.semilogx(f_lin, filt_gd, "orange", linewidth=0.9)
        if gd_range is not None:
            ax3.set_ylim(gd_range)
        ax4.semilogx(f_lin, filt_db, "r", linewidth=0.9)

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xscale("log")
            ax.set_xlim(20, 20000)
            ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"Visualization error ({title}): {e}")
        return b""
