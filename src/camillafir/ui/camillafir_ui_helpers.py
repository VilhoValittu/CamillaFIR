# camillafir_ui_helpers.py
import numpy as np

from pywebio.output import *  # needed because this PyWebIO build doesn't expose put_input/put_select as named exports
from pywebio.input import FLOAT
from pywebio.pin import pin, pin_update, put_input

from ..resources.i8n.camillafir_i18n import t
from .camillafir_modes import MODE_DEFAULTS
from .camillafir_utils import scale_taps_with_fs

def _warn_max_boost_if_over_cap(_=None):
    """
    Warn user if max_boost exceeds internal safety cap.
    Uses a simple "edge trigger" to avoid spamming toast repeatedly.
    """
    try:
        v = pin.get('max_boost', None)
        if v is None or v == '':
            return
        v = float(v)
        if not math.isfinite(v):
            return

        over = (float(MAX_SAFE_BOOST) > 0.0) and (v > float(MAX_SAFE_BOOST) + 1e-9)
        # Edge-trigger (warn only when transitioning to over-cap state)
        prev = bool(getattr(_warn_max_boost_if_over_cap, "_prev_over", False))
        if over and not prev:
            # Build message safely even if translations are missing
            try:
                cap_suffix = t('max_boost_help_cap').format(value=f"{MAX_SAFE_BOOST:.1f}")
            except Exception:
                cap_suffix = f" (capped to {MAX_SAFE_BOOST:.1f} dB)"

            msg = f"{t('max_boost')}: {v:.1f} dB > {MAX_SAFE_BOOST:.1f} dB{cap_suffix}"

            # Use default toast styling to avoid version-specific color keyword issues
            _toast(msg, duration=5)
        _warn_max_boost_if_over_cap._prev_over = over
    except Exception as e:

        try:
            logger.warning(f"max_boost toast failed: {e}")
        except Exception:
            pass
        return

def _warn_taps_if_over_cap(_=None):
    """
    Warn user if taps exceeds recommended maximum.
    Uses edge trigger to avoid repeated toasts.
    """
    try:
        v = pin.get('taps', None)
        if v is None or v == '':
            return
        v = int(v)

        over = v > int(MAX_SAFE_TAPS)
        prev = bool(getattr(_warn_taps_if_over_cap, "_prev_over", False))

        if over and not prev:
            try:
                msg = t('taps_warn_over').format(value=MAX_SAFE_TAPS)
            except Exception:
                msg = f"Taps > {MAX_SAFE_TAPS}: very high latency and diminishing returns."

            _toast(msg, duration=6)

        _warn_taps_if_over_cap._prev_over = over
    except Exception as e:
        try:
            logger.warning(f"taps warning toast failed: {e}")
        except Exception:
            pass

def _toast(msg, *, duration=5, color=None):
    """
    Safe toast wrapper for PyWebIO.
    Works even if toast is unavailable or UI context is missing.
    """
    try:
        fn = getattr(pwo, "toast", None)
        if callable(fn):
            if color is None:
                fn(msg, duration=duration)
            else:
                fn(msg, duration=duration, color=color)
    except Exception:
        pass

def _max_boost_help_with_cap():
    try:
        return (
            f"{t('max_boost_help')}"
            f"{t('max_boost_help_cap').format(value=f'{MAX_SAFE_BOOST:.1f}')}"
        )
    except Exception:
        return t('max_boost_help')

def update_mode_desc(_=None):
    """UI helper: show a short description under Mode selection."""
    try:
        m = str(pin["mode"] or "BASIC").strip().upper()
    except Exception:
        m = "BASIC"
    key = "mode_basic_desc" if m == "BASIC" else "mode_advanced_desc"
    with use_scope("mode_desc_scope", clear=True):
        put_markdown(f"**{t('mode_desc_title')}**\n\n{t(key)}")


def _as_pin_checkbox_list(v: bool):
    return [True] if bool(v) else []

def update_ir_tukey_ui(_=None):
    """UI helper: show/lock Tukey alpha depending on selected window shape."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        sh = str(_p("ir_export_window_shape", "hann") or "hann").strip().lower()
        is_tukey = (sh == "tukey")

        # Clamp alpha to 0..1 for display stability (doesn't change DSP rules)
        try:
            a = float(_p("ir_export_tukey_alpha", 0.25) or 0.25)
        except Exception:
            a = 0.25
        if not np.isfinite(a):
            a = 0.25
        a = float(np.clip(a, 0.0, 1.0))

        with use_scope("ir_tukey_alpha_scope", clear=True):
            w = put_input(
                "ir_export_tukey_alpha",
                label=t("ir_export_tukey_alpha"),
                type=FLOAT,
                value=a,
                help_text=t("ir_export_tukey_alpha_help"),
            )
            if not is_tukey:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass


def apply_mode_defaults_to_ui(_=None):
    """Apply current mode defaults to UI fields (manual button only)."""
    try:
        mode = str(pin["mode"] or "BASIC").strip().upper()
    except Exception:
        mode = "BASIC"
    if mode not in MODE_DEFAULTS:
        mode = "BASIC"

    d = MODE_DEFAULTS.get(mode, {}) or {}

    map_num = {
        "global_gain_db": "gain",
        "mag_c_min": "mag_c_min",
        "mag_c_max": "mag_c_max",
        "max_boost_db": "max_boost",
        "max_cut_db": "max_cut_db",
        "phase_limit": "phase_limit",
        "reg_strength": "reg_strength",
        "fdw_cycles": "fdw_cycles",
        "filter_smooth": "filter_smooth",
        "tdc_strength": "tdc_strength",
        "tdc_max_reduction_db": "tdc_max_reduction_db",
        "tdc_slope_db_per_oct": "tdc_slope_db_per_oct",
        "low_bass_cut_hz": "low_bass_cut_hz",
        "ir_window_ms": "ir_window",
        "ir_window_ms_left": "ir_window_left",
        "mixed_split_freq": "mixed_freq",
        "trans_width": "trans_width",
        "bass_first_mode_max_hz": "bass_first_mode_max_hz",
        "max_slope_db_per_oct": "max_slope_db_per_oct",
        "max_slope_boost_db_per_oct": "max_slope_boost_db_per_oct",
        "max_slope_cut_db_per_oct": "max_slope_cut_db_per_oct",
        "lvl_manual_db": "lvl_manual_db",
        "lvl_min": "lvl_min",
        "lvl_max": "lvl_max",
    }
    map_str = {
        "filter_type_str": "filter_type",
        "plot_smoothing_level": "plot_smoothing_level",
        "lvl_mode": "lvl_mode",
        "lvl_algo": "lvl_algo",
    }
    map_chk = {
        "enable_mag_correction": "mag_correct",
        "do_normalize": "normalize_opt",
        "exc_prot": "exc_prot",
        "enable_tdc": "enable_tdc",
        "enable_afdw": "enable_afdw",
        "df_smoothing": "df_smoothing",
        "comparison_mode": "comparison_mode",
        "bass_first_ai": "bass_first_ai",
        "phase_safe_2058": "phase_safe_2058",
        "stereo_link": "stereo_link",
    }

    for cfg_k, pin_k in map_num.items():
        if cfg_k in d:
            try:
                pin_update(pin_k, value=d[cfg_k])
            except Exception:
                pass

    for cfg_k, pin_k in map_str.items():
        if cfg_k in d:
            try:
                pin_update(pin_k, value=d[cfg_k])
            except Exception:
                pass

    for cfg_k, pin_k in map_chk.items():
        if cfg_k in d:
            try:
                pin_update(pin_k, value=_as_pin_checkbox_list(bool(d[cfg_k])))
            except Exception:
                pass

    try:
        update_lvl_ui()
    except Exception:
        pass
    try:
        update_ir_tukey_ui()
    except Exception:
        pass
    try:
        update_taps_auto_info()
    except Exception:
        pass
    update_mode_desc()

    try:
        msg = t("mode_defaults_applied_toast").replace("{mode}", mode)
    except Exception:
        msg = f"Mode defaults applied: {mode}"
    try:
        toast(msg, color="success", duration=2.0)
    except Exception:
        pass

def update_taps_auto_info(_=None):
    """UI helper: show Auto-taps mapping when multi-rate is enabled."""
    try:
        mr = bool(pin["multi_rate_opt"])
    except Exception:
        mr = False

    for scope_name in ("taps_auto_info_scope_files", "taps_auto_info_scope_basic"):
        with use_scope(scope_name, clear=True):
            if not mr:
                put_markdown(f"_{t('auto_taps_title')}: OFF_")
                continue

            rates = [44100, 48000, 88200, 96000, 176400, 192000]
            lines = [f"- **{r/1000:.1f} kHz** → **{scale_taps_with_fs(r)}** taps" for r in rates]

            put_markdown(
                f"### {t('auto_taps_title')}\n"
                f"{t('auto_taps_body')}\n\n"
                f"{t('auto_taps_ref')}\n\n"
                + "\n".join(lines)
            )


def update_lvl_ui(_=None):
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        mode = str(_p("lvl_mode", "Auto") or "Auto")
        is_manual = ("Manual" in mode)

        vmin = float(_p("lvl_min", 500.0) or 500.0)
        vmax = float(_p("lvl_max", 2000.0) or 2000.0)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
            pin_update("lvl_min", value=vmin)
            pin_update("lvl_max", value=vmax)

        with use_scope("lvl_manual_scope", clear=True):
            w = put_input(
                "lvl_manual_db",
                label=t("lvl_target_db"),
                type=FLOAT,
                value=float(_p("lvl_manual_db", 75.0) or 75.0),
                help_text=t("lvl_manual_help"),
            )
            if not is_manual:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass


def apply_tdc_preset(name: str):
    presets = {
        "Safe": {"enable": True, "strength": 35.0, "max_red": 6.0, "slope": 3.0},
        "Normal": {"enable": True, "strength": 50.0, "max_red": 9.0, "slope": 6.0},
        "Aggressive": {"enable": True, "strength": 70.0, "max_red": 12.0, "slope": 0.0},
    }
    p = presets.get(name)
    if not p:
        return

    pin_update("enable_tdc", value=[True] if p["enable"] else [])
    pin_update("tdc_strength", value=float(p["strength"]))
    pin_update("tdc_max_reduction_db", value=float(p["max_red"]))
    pin_update("tdc_slope_db_per_oct", value=float(p["slope"]))

    try:
        toast(f"TDC preset applied: {name}", color="success", duration=1.5)
    except Exception:
        pass


def apply_afdw_preset(name: str):
    presets = {
        "Tight": {"enable": True, "cycles": 5.0},
        "Balanced": {"enable": True, "cycles": 10.0},
        "Safe": {"enable": True, "cycles": 15.0},
        "Minimal": {"enable": True, "cycles": 20.0},
    }
    p = presets.get(str(name or ""))
    if not p:
        return

    try:
        pin_update("enable_afdw", value=[True] if p["enable"] else [])
    except Exception:
        pass

    try:
        pin_update("fdw_cycles", value=float(p["cycles"]))
    except Exception:
        pass

    try:
        toast(f"A-FDW preset applied: {name}", color="success", duration=1.5)
    except Exception:
        pass


def put_guide_section():
    guides = [
        ("guide_modes", t("guide_modes_title")),
        ("guide_taps", t("guide_taps_title")),
        ("guide_ft", t("guide_ft_title")),
        ("guide_sigma", t("guide_sigma_title")),
        ("guide_mix", t("guide_mix_title")),
        ("guide_tdc", t("guide_tdc_title")),
        ("guide_afdw", t("guide_afdw_title")),
        ("guide_reg", t("guide_reg_title")),
        ("guide_lvl", t("guide_lvl_title")),
        ("guide_sl", t("guide_sl_title")),
        ("guide_ep", t("guide_ep_title")),
        ("guide_asy", t("guide_asy_title")),
        ("guide_ai", t("guide_ai_title")),
        ("guide_summary", t("guide_summary_title")),
    ]

    content = [
        put_collapse(
            t(g_key + "_title") if t(g_key + "_title") != (g_key + "_title") else g_title,
            [put_markdown(t(g_key + "_body") if t(g_key + "_body") != (g_key + "_body") else "Info text here")],
        )
        for g_key, g_title in guides
    ]
    put_collapse("❓ CamillaFIR User Guides", content)
