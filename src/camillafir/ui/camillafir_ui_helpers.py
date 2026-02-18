# camillafir_ui_helpers.py
import numpy as np
import math
import tempfile
import re
from pywebio.output import *  # needed because this PyWebIO build doesn't expose put_input/put_select as named exports
from pywebio.input import FLOAT
from pywebio.pin import pin, pin_update, put_input, put_select, put_file_upload, pin_on_change, pin_update

from ..resources.i8n.camillafir_i18n import t
from .camillafir_modes import MODE_DEFAULTS, MODE_CLAMPS
from .camillafir_utils import scale_taps_with_fs

# --- House curve loaders for target preview ---
try:
    from .camillafir_housecurve import (
        _normalize_hc_mode_key,
        get_house_curve_by_name,
        load_target_curve,
        load_house_curve,
    )
except Exception:
    _normalize_hc_mode_key = None
    get_house_curve_by_name = None
    load_target_curve = None
    load_house_curve = None



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
        fn = globals().get("toast", None)
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
            f"{t('max_boost_help')}\n\n"
            f"{t('max_boost_help_cap').format(value=f'{MAX_SAFE_BOOST:.1f}')}"
        )
    except Exception:
        return (
            f"{t('max_boost_help')}\n\n"
            f"{t('max_boost_help_cap').format(value=f'{MAX_SAFE_BOOST:.1f}')}"
        )


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
        # Allow IR window controls ONLY when filter_type is Linear or Asymmetric
        ft = str(_p("filter_type", "") or "").strip().lower()
        try:
            ft_linear_label = str(t("ft_linear") or "").strip().lower()
        except Exception:
            ft_linear_label = "linear"
        is_linear = (ft == ft_linear_label) or ("linear" in ft)

        try:
            ft_asym_label = str(t("ft_asymmetric") or "").strip().lower()
        except Exception:
            ft_asym_label = "asymmetric"
        is_asym_filter = (ft == ft_asym_label) or ("asym" in ft)

        allow_ir = bool(is_linear or is_asym_filter)

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
            # If IR controls not allowed for this filter type, hide the whole alpha control
            if not allow_ir:
                return

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

def update_ir_export_window_mode_ui(_=None):
    """
    UI helper: grey out (disable) 'rew_asym' option unless filter_type is Linear.
    Also sanitizes saved/forced value: if not linear and mode==rew_asym -> auto.
    Renders the select into scope 'ir_export_window_mode_scope'.
    """
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        # HARD POLICY: BASIC mode => windowing ALWAYS auto (UI + pin sanitize)
        try:
            m = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            m = "BASIC"
        is_basic = (m == "BASIC")

        ft = str(_p("filter_type", "") or "").strip().lower()

        try:
            ft_linear_label = str(t("ft_linear") or "").strip().lower()
        except Exception:
            ft_linear_label = "linear"

        is_linear = (ft == ft_linear_label) or ("linear" in ft)

        # Allow IR window controls ONLY when filter_type is Linear or Asymmetric
        try:
            ft_asym_label = str(t("ft_asymmetric") or "").strip().lower()
        except Exception:
            ft_asym_label = "asymmetric"
        is_asym_filter = (ft == ft_asym_label) or ("asym" in ft)
        allow_ir = bool(is_linear or is_asym_filter)
        # POLICY: Asymmetric filter type => IR export window mode must be Auto
        # (left_ms still matters for asymmetric filter placement, but windowing mode stays Auto)
        lock_window_mode = bool(is_basic or (not allow_ir) or is_asym_filter)

        # Current value (from pins, may come from loaded config)
        cur = str(_p("ir_export_window_mode", "auto") or "auto").strip().lower()

        # BASIC: force auto no matter what user tries
        if is_basic and cur != "auto":
            try:
                pin_update("ir_export_window_mode", value="auto")
            except Exception:
                pass
            cur = "auto"
        # Asymmetric filter: force auto no matter what user tries
        if is_asym_filter and cur != "auto":
            try:
                pin_update("ir_export_window_mode", value="auto")
            except Exception:
                pass
            cur = "auto"

        # If IR controls not allowed for this filter type, force mode to auto
        if (not allow_ir) and (cur != "auto"):
            try:
                pin_update("ir_export_window_mode", value="auto")
            except Exception:
                pass
            cur = "auto"


        # If not linear, force away from rew_asym
        if (not is_linear) and (cur == "rew_asym"):
            try:
                pin_update("ir_export_window_mode", value="auto")
            except Exception:
                pass
            cur = "auto"

        with use_scope("ir_export_window_mode_scope", clear=True):
            w = put_select(
                "ir_export_window_mode",
                label=t("ir_export_window_mode"),
                options=[
                    {"label": t("ir_export_window_auto"), "value": "auto"},
                    {"label": t("ir_export_window_asym"), "value": "rew_asym"},
                ],
                value=cur,
                help_text=t("ir_export_window_help"),
            )
            # Lock the whole control when policy requires Auto-only
            if lock_window_mode:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            # Disable/enable the rew_asym option at DOM level (greys out in native select)
            # NOTE: do this after put_select so the element exists.
            js_disable = "true" if (lock_window_mode or (not is_linear)) else "false"
            put_html(f"""
<script>
(function() {{
  try {{
    var sel = document.querySelector('select[name="ir_export_window_mode"]');
    if(!sel) return;
    var opt = Array.from(sel.options).find(o => (o.value === "rew_asym"));
    if(!opt) return;

    opt.disabled = {js_disable};

    // Optional: add a hint to the label when disabled/enabled (avoid duplicating)
    var base = opt.textContent.replace(/\\s*\\(Linear only\\)\\s*$/,'');
    if({js_disable}) opt.textContent = base + " (Linear only)";
    else opt.textContent = base;

    // If somehow selected while disabled (older browser state), force select back to auto
    if(opt.disabled && sel.value === "rew_asym") {{
      sel.value = "auto";
    }}
  }} catch(e) {{}}
}})();
</script>
""")
            # Info text: visible always, emphasized when non-linear
            try:
                msg_ = t("ir_asym_linear_only")
                
            except Exception:
                pass

            emph = bool(is_basic or (not is_linear))
            opacity = "1.0" if emph else "0.55"
            color = "#ffb74d" if emph else "#9aa0a6"

            put_html(f"""
<div style="
  margin-top:6px;
  font-size:12.5px;
  font-weight:700;
  letter-spacing:0.4px;
  color:{color};
  opacity:{opacity};
">
  ⚠️ {msg_}<br>
  <span style="font-weight:500; letter-spacing:0;">
    
  </span>
</div>
""")
    except Exception:
        pass

def update_ir_lr_window_ui(_=None):
    """
    UI helper:
      - Controls visible always, but enabled only when filter_type is Linear or Asymmetric.
      - Left is meaningful for:
          * IR export window mode == 'rew_asym', OR
          * filter_type == Asymmetric (DSP initial placement).
      - Right is meaningful ONLY for:
          * IR export window mode == 'rew_asym'
    Renders into scope 'ir_lr_window_scope'.
    """
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        mode = str(_p("ir_export_window_mode", "auto") or "auto").strip().lower()
        is_rew_asym = (mode == "rew_asym")

        ft = str(_p("filter_type", "") or "").strip().lower()
        try:
            ft_linear_label = str(t("ft_linear") or "").strip().lower()
        except Exception:
            ft_linear_label = "linear"
        is_linear = (ft == ft_linear_label) or ("linear" in ft)

        try:
            ft_asym_label = str(t("ft_asymmetric") or "").strip().lower()
        except Exception:
            ft_asym_label = "asymmetric"
        is_asym_filter = (ft == ft_asym_label) or ("asym" in ft)

        allow_ir = bool(is_linear or is_asym_filter)
        enable_left = bool(allow_ir and (is_rew_asym or is_asym_filter))
        enable_right = bool(allow_ir and is_rew_asym)

        # Values (keep backward compat for right window via legacy 'ir_window')
        try:
            v_left = float(_p("ir_window_left", 120.0) or 120.0)
        except Exception:
            v_left = 10.0

        try:
            v_right = _p("ir_window_right", None)
            if v_right is None or v_right == "":
                v_right = _p("ir_window", 500.0)
            v_right = float(v_right or 500.0)
        except Exception:
            v_right = 500.0

        with use_scope("ir_lr_window_scope", clear=True):
            w_left = put_input(
                "ir_window_left",
                label=t("ir_window_left_label"),
                type=FLOAT,
                value=v_left,
                help_text=t("ir_matala"),
            )
            w_right = put_input(
                "ir_window_right",
                label=t("ir_window_right_label"),
                type=FLOAT,
                value=v_right,
                help_text=t("ir_korkea"),
            )

            if not enable_left:
                w_left.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            if not enable_right:
                w_right.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")

            # Optional: show generic hint only when both are locked
            if (not enable_left) and (not enable_right):
                try:
                    msg = t("ir_lr_window_hint")
                except Exception:
                    msg = ""
                if str(msg or "").strip():
                    put_html(
                        f"<div style='margin-top:6px; font-size:12.5px; color:#9aa0a6;'>"
                        f"{msg}"
                        f"</div>"
                    )
    except Exception:
        pass




def update_ir_window_shape_ui(_=None):
    """
    UI helper: grey out IR window shape selector when IR window mode == Auto.
    """
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        mode = str(_p("ir_export_window_mode", "auto") or "auto").strip().lower()
        m = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        is_basic = (m == "BASIC")
        is_auto = (mode == "auto") or is_basic

        with use_scope("ir_export_window_shape_scope", clear=True):
            w = put_select(
                'ir_export_window_shape',
                label=t('ir_export_window_shape'),
                options=[
                    {'label': t('ir_export_window_shape_hann'), 'value': 'hann'},
                    {'label': t('ir_export_window_shape_tukey'), 'value': 'tukey'},
                ],
                value=str(_p('ir_export_window_shape', 'hann') or 'hann').strip().lower(),
                help_text=t('ir_export_window_shape_help'),
            )

            if is_auto:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass

def update_mixed_freq_ui(_=None):
    """
    UI helper: enable mixed_freq ONLY when filter_type == Mixed.
    Otherwise grey out the field (visible but locked).
    """
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        ft = str(_p("filter_type", "") or "").strip().lower()

        try:
            ft_mixed_label = str(t("ft_mixed") or "").strip().lower()
        except Exception:
            ft_mixed_label = "mixed"

        is_mixed = (ft == ft_mixed_label) or ("mixed" in ft)

        with use_scope("update_mixed_freq_scope", clear=True):
            w = put_input(
                "mixed_freq",
                label=t("mixed_freq"),
                type=FLOAT,
                value=float(_p("mixed_freq", 300.0) or 300.0),
                help_text=t("mixed_freq_help"),
            )
            if not is_mixed:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass
    
def update_low_bass_cut_ui(*, pin, pin_update, get_val, t):
    """
    UI helper: Low-bass cut toggle.
    OFF -> low_bass_cut_hz = ""   (empty)
    ON  -> low_bass_cut_hz = float

    Renders into scope: 'low_bass_cut_scope'
    Uses existing pins:
      - low_bass_cut_enable (checkbox list or bool)
      - low_bass_cut_hz     (float or "")
    """

    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    # Enable: checkbox commonly returns [] / [True]
    try:
        en_raw = _p("low_bass_cut_enable", None)
        if isinstance(en_raw, (list, tuple, set)):
            enabled = (len(en_raw) > 0)
        else:
            enabled = bool(en_raw) if en_raw is not None else bool(get_val("low_bass_cut_enable", True))
    except Exception:
        enabled = bool(get_val("low_bass_cut_enable", True))

    cur = _p("low_bass_cut_hz", "")

    # Remember last numeric for nice UX when OFF
    last = _p("low_bass_cut_hz_last", get_val("low_bass_cut_hz", 40.0))

    # --- sanitize stored pin value ---
    try:
        if not enabled:
            # OFF => force empty string, keep last remembered value
            if cur not in ("", None):
                try:
                    # store last good numeric before blanking
                    pin_update("low_bass_cut_hz_last", value=float(cur))
                except Exception:
                    pass
                pin_update("low_bass_cut_hz", value="")
            disp = float(last or 40.0)
        else:
            # ON => ensure numeric, restore from last/default if currently blank
            if cur in ("", None):
                v0 = float(last or 40.0)
                pin_update("low_bass_cut_hz", value=v0)
                cur = v0
            disp = float(cur)
            if not math.isfinite(disp):
                disp = float(last or 40.0)
                pin_update("low_bass_cut_hz", value=disp)
            # refresh last
            pin_update("low_bass_cut_hz_last", value=float(disp))
    except Exception:
        disp = 40.0

    # --- render ---
    with use_scope("low_bass_cut_scope", clear=True):
        w = put_input(
            "low_bass_cut_hz",
            label=f"{t('low_bass_cut_hz')} (Hz)",
            type=FLOAT,
            value=float(disp),
            help_text=t("low_bass_cut_hz_help"),
        )

        if not enabled:
            # truly not usable when OFF
            w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            put_html(
                "<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>OFF</div>"
            )

def update_afdw_cycles_ui(*, pin, get_val, t):
    """
    UI helper: grey out FDW cycles input when A-FDW is disabled.
    Renders into scope: 'afdw_cycles_scope'
    """
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    # PyWebIO checkbox can be [] or [True]
    try:
        en_raw = _p("enable_afdw", None)
        if isinstance(en_raw, (list, tuple, set)):
            enabled = (len(en_raw) > 0)
        else:
            enabled = bool(en_raw) if en_raw is not None else bool(get_val("enable_afdw", True))
    except Exception:
        enabled = bool(get_val("enable_afdw", True))

    # Current value
    try:
        v = float(_p("fdw_cycles", get_val("fdw_cycles", 10.0)) or 10.0)
    except Exception:
        v = float(get_val("fdw_cycles", 10.0) or 10.0)

    with use_scope("afdw_cycles_scope", clear=True):
        w = put_input(
            "fdw_cycles",
            label=t("fdw"),
            type=FLOAT,
            value=v,
            help_text=t("fdw_help"),
        )
        if not enabled:
            w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            put_html(
                f"<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>"
                f"{t('afdw_disabled_hint') if 'afdw_disabled_hint' else 'OFF'}"
                f"</div>"
            )

def update_tdc_controls_ui(*, pin, get_val, t, apply_tdc_preset):
    """
    UI helper: grey out all TDC controls when TDC is disabled.
    Renders into scope: 'tdc_controls_scope'
    """
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    # Checkbox value can be [] or [True]
    try:
        en_raw = _p("enable_tdc", None)
        if isinstance(en_raw, (list, tuple, set)):
            enabled = (len(en_raw) > 0)
        else:
            enabled = bool(en_raw) if en_raw is not None else bool(get_val("enable_tdc", True))
    except Exception:
        enabled = bool(get_val("enable_tdc", True))

    def _f(name, default):
        try:
            v = _p(name, get_val(name, default))
            return float(v if v is not None else default)
        except Exception:
            return float(default)

    strength = _f("tdc_strength", 50.0)
    max_red  = _f("tdc_max_reduction_db", 9.0)
    slope    = _f("tdc_slope_db_per_oct", 6.0)

    with use_scope("tdc_controls_scope", clear=True):
        w = put_html("<div id='tdc_box' style='margin-top:6px'>")  # wrapper start

        # Presets row (ONLY buttons)
        put_row([
            put_buttons(
                [
                    {"label": t("tdc_preset_safe"),       "value": "Safe"},
                    {"label": t("tdc_preset_normal"),     "value": "Normal"},
                    {"label": t("tdc_preset_aggressive"), "value": "Aggressive"},
                ],
                onclick=lambda preset: apply_tdc_preset(preset),
                small=True,
            ),
        ])

        # Help texts as compact blocks below (no row layout)
        put_html(f"<div style='opacity:0.65; font-size:12px; line-height:1.25; margin-top:6px'>{t('tdc_preset_help')}</div>")
        put_html(f"<div style='opacity:0.70; font-size:12px; line-height:1.25; margin-top:4px'>{t('tdc_summary_hint')}</div>")

        # Inputs row
        put_row([
            put_input(
                "tdc_strength",
                label=t("tdc_strength"),
                type=FLOAT,
                value=float(get_val("tdc_strength", 50.0) or 50.0), # type: ignore
                help_text=t("tdc_help"),
            ),
            put_input(
                "tdc_max_reduction_db",
                label=t("tdc_max_reduction_db"),
                type=FLOAT,
                value=float(get_val("tdc_max_reduction_db", 9.0) or 9.0),
                help_text=t("tdc_max_reduction_db_help"),
            ),
            put_input(
                "tdc_slope_db_per_oct",
                label=t("tdc_slope_db_per_oct"),
                type=FLOAT,
                value=float(get_val("tdc_slope_db_per_oct", 6.0) or 6.0),
                help_text=t("tdc_slope_db_per_oct_help"),
            ),
        ])

        put_html("</div>")  # wrapper end

    # Re-acquire wrapper as one output for styling
    # (PyWebIO limitation: we style the whole scope content instead)
    if not enabled:
        # style the whole scope (works reliably)
        put_html("<script>document.getElementById('tdc_box').style.opacity='0.55';"
                 "document.getElementById('tdc_box').style.pointerEvents='none';"
                 "document.getElementById('tdc_box').style.filter='grayscale(1)';</script>")




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


def _pretty_plot_smoothing(v, t):
    # UI naming: Psychoacoustic == CamillaFIR Reference
    if isinstance(v, str) and v.strip().lower() == "psychoacoustic":
        return t("smooth_safe_reference")  # "CamillaFIR Reference"
    return str(v)

def _fmt_mode_value(key: str, defaults: dict, clamps: dict):
    """
    Return a human-friendly markdown string for a mode default,
    including clamp info when available.
    """
    v = defaults.get(key, None)
    if key == "plot_smoothing_level":
        v_str = _pretty_plot_smoothing(v)
    else:
        v_str = str(v)

    # Optional clamp display
    lim = clamps.get(key, None) if isinstance(clamps, dict) else None
    if isinstance(lim, tuple) and len(lim) == 2:
        lo, hi = lim
        # bool clamps are (True, True) etc.
        if isinstance(lo, bool) and isinstance(hi, bool):
            return f"**{v_str}**"
        return f"**{v_str}** _(clamped to {lo}–{hi})_"
    return f"**{v_str}**"

def _build_modes_guide_parts(t):
    """
    Returns: intro_md, basic_md, advanced_md, tip_md
    Values are read live from camillafir_modes.MODE_DEFAULTS / MODE_CLAMPS.
    Text is localized via translations.json keys.
    """
    # Import live policy
    

    d_basic = MODE_DEFAULTS.get("BASIC", {})
    c_basic = MODE_CLAMPS.get("BASIC", {})
    d_adv   = MODE_DEFAULTS.get("ADVANCED", {})
    c_adv   = MODE_CLAMPS.get("ADVANCED", {})

    def clamp_suffix(key, clamps):
        lim = clamps.get(key)
        if isinstance(lim, tuple) and len(lim) == 2:
            lo, hi = lim
            # localized " (clamped lo–hi)"
            return t("guide_modes_clamped_suffix").format(lo=lo, hi=hi)
        return ""

    # Localized intro
    intro = t("guide_modes_intro") + "\n"

    # BASIC markdown
    basic_lines = []
    basic_lines.append(t("guide_modes_basic_goal") + "\n")
    basic_lines.append(t("guide_modes_defaults_live"))

    basic_lines.append(
        t("guide_modes_line_max_boost_cut").format(
            boost=d_basic.get("max_boost_db"),
            boost_clamp=clamp_suffix("max_boost_db", c_basic),
            cut=d_basic.get("max_cut_db"),
            cut_clamp=clamp_suffix("max_cut_db", c_basic),
        )
    )
    basic_lines.append(
        t("guide_modes_line_correction_band").format(
            lo=d_basic.get("mag_c_min"),
            hi=d_basic.get("mag_c_max"),
        )
    )
    basic_lines.append(
        t("guide_modes_line_phase_limit").format(
            hz=d_basic.get("phase_limit"),
            clamp=clamp_suffix("phase_limit", c_basic),
        )
    )
    basic_lines.append(
        t("guide_modes_line_plot_smoothing").format(
            name=_pretty_plot_smoothing(d_basic.get("plot_smoothing_level"), t)
        )
    )
    basic_lines.append(
        t("guide_modes_line_filter_smoothing").format(
            n=d_basic.get("filter_smooth"),
            clamp=clamp_suffix("filter_smooth", c_basic),
        )
    )
    basic_lines.append(
        t("guide_modes_line_tdc").format(
            on=t("common_on") if d_basic.get("enable_tdc") else t("common_off"),
            strength=d_basic.get("tdc_strength"),
            maxred=d_basic.get("tdc_max_reduction_db"),
            slope=d_basic.get("tdc_slope_db_per_oct"),
        )
    )
    basic_lines.append(
        t("guide_modes_line_afdw").format(
            on=t("common_on") if d_basic.get("enable_afdw") else t("common_off"),
            cycles=d_basic.get("fdw_cycles"),
            clamp=clamp_suffix("fdw_cycles", c_basic),
        )
    )
    basic_lines.append(
        t("guide_modes_line_bass_first").format(
            on=t("common_on") if d_basic.get("bass_first_ai") else t("common_off"),
            hz=d_basic.get("bass_first_mode_max_hz"),
        )
    )
    basic_lines.append(
        t("guide_modes_line_leveling").format(
            mode=d_basic.get("lvl_mode"),
            algo=d_basic.get("lvl_algo"),
            lo=d_basic.get("lvl_min"),
            hi=d_basic.get("lvl_max"),
            stereo=t("common_on") if d_basic.get("stereo_link") else t("common_off"),
        )
    )
    basic_lines.append(
        t("guide_modes_line_low_bass").format(
            hz=d_basic.get("low_bass_cut_hz"),
            clamp=clamp_suffix("low_bass_cut_hz", c_basic),
        )
    )
    basic_md = "\n".join(basic_lines)

    # ADVANCED markdown
    adv_lines = []
    adv_lines.append(t("guide_modes_adv_goal") + "\n")
    adv_lines.append(t("guide_modes_defaults_live"))

    adv_lines.append(
        t("guide_modes_line_max_boost_cut_adv").format(
            boost=d_adv.get("max_boost_db"),
            cut=d_adv.get("max_cut_db"),
        )
    )
    adv_lines.append(
        t("guide_modes_line_correction_band").format(
            lo=d_adv.get("mag_c_min"),
            hi=d_adv.get("mag_c_max"),
        )
    )
    adv_lines.append(
        t("guide_modes_line_phase_limit_adv").format(
            hz=d_adv.get("phase_limit"),
        )
    )
    adv_lines.append(
        t("guide_modes_line_slope_limits").format(
            g=d_adv.get("max_slope_db_per_oct"),
            b=d_adv.get("max_slope_boost_db_per_oct"),
            c=d_adv.get("max_slope_cut_db_per_oct"),
        )
    )
    adv_lines.append(
        t("guide_modes_line_leveling_window").format(
            lo=d_adv.get("lvl_min"),
            hi=d_adv.get("lvl_max"),
            stereo=t("common_on") if d_adv.get("stereo_link") else t("common_off"),
        )
    )
    adv_lines.append(
        t("guide_modes_line_low_bass_adv").format(
            hz=d_adv.get("low_bass_cut_hz"),
        )
    )
    adv_md = "\n".join(adv_lines)

    tip = t("guide_modes_tip")
    return intro, basic_md, adv_md, tip



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

    content = []
    for g_key, g_title in guides:
        title = t(g_key + "_title") if t(g_key + "_title") != (g_key + "_title") else g_title

        if g_key == "guide_modes":
            intro, basic_md, adv_md, tip = _build_modes_guide_parts(t)
            content.append(
                put_collapse(
                    title,
                    [
                        put_markdown(intro),
                        put_collapse(t("mode_basic_label"), [put_markdown(basic_md)], open=False),
                        put_collapse(t("mode_advanced_label"), [put_markdown(adv_md)], open=False),
                        put_markdown(tip),
                    ],
                    open=False,
                )
            )
            continue

        body = t(g_key + "_body") if t(g_key + "_body") != (g_key + "_body") else "Info text here"
        content.append(
            put_collapse(title, [put_markdown(body)], open=False)
        )

    # Optional outer collapse (if you want everything under one)
    put_collapse(t("guide_section_title"), content, open=False)


def update_confidence_pull_ui(*, pin, get_val, t):
    """
    UI helper: Confidence-based target pull (Advanced only).
    Creates pins so pipeline/DSP can actually see user values.
    Renders into scope: 'conf_pull_scope'
    """
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        mode = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        if mode != "ADVANCED":
            with use_scope("conf_pull_scope", clear=True):
                return

        with use_scope("conf_pull_scope", clear=True):
            put_markdown("#### 🎯 Confidence Pull (Advanced)")
            put_html(
                f"<div style='opacity:0.7; font-size:12px; margin-bottom:6px'>"
                f"{t('tdc_confidence_pull_help')}</div>"
                )

            put_row([
                put_input(
                    "conf_pull_floor",
                    label=t("conf_pull_floor"),
                    type=FLOAT,
                    value=float(get_val("conf_pull_floor", 0.05)), # type: ignore
                    help_text=t("conf_pull_floor_help"),
                ),
                put_input(
                    "conf_pull_max_hz",
                    label=t("conf_pull_max_hz"),
                    type=FLOAT,
                    value=float(get_val("conf_pull_max_hz", 200.0)),
                    help_text=t("conf_pull_max_hz_help"),
                ),
            ])

            put_row([
                put_input(
                    "conf_pull_gamma_cut",
                    label=t("conf_pull_gamma_cut"),
                    type=FLOAT,
                    value=float(get_val("conf_pull_gamma_cut", 0.55)), # type: ignore
                    help_text=t("conf_pull_gamma_cut_help"),
                ),
                put_input(
                    "conf_pull_gamma_boost",
                    label=t("conf_pull_gamma_boost"),
                    type=FLOAT,
                    value=float(get_val("conf_pull_gamma_boost", 1.35)),
                    help_text=t("conf_pull_gamma_boost_help"),
                ),
            ])

            put_row([
                put_input(
                    "low_bass_cut_strength",
                    label=t("low_bass_cut_strength"),
                    type=FLOAT,
                    value=float(get_val("low_bass_cut_strength", 0.0)), # type: ignore
                    help_text=t("low_bass_cut_strength_help"),
                ),
            ])
    except Exception:
        pass


def update_target_preview_ui(_=None):
    """
    Small, fast target preview for Target tab.
    - Uses Plotly HTML (no Kaleido).
    - Supports built-in curves and uploaded custom target file.
    """
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    def _norm_key(x: str) -> str:
        s = str(x or "").strip()
        if not s:
            return ""
        # Prefer project’s canonical normalizer when available
        try:
            if callable(_normalize_hc_mode_key):
                return str(_normalize_hc_mode_key(s))
        except Exception:
            pass
        return s
    try:
        import numpy as np
        import plotly.graph_objects as go
        import plotly.io as pio

        hc_mode_raw = str(_p("hc_mode", "Harman6") or "Harman6")
        hc_mode = _norm_key(hc_mode_raw)

        mag_c_min = float(_p("mag_c_min", 10.0) or 10.0)
        mag_c_max = float(_p("mag_c_max", 200.0) or 200.0)

        # Log frequency grid for preview
        f = np.logspace(np.log10(10.0), np.log10(20000.0), 600)

        y = None
        src = "builtin"

        # Detect upload selection (be tolerant to naming)
        key_l = str(hc_mode).strip().lower()
        is_upload = key_l in ("upload", "custom", "hc_mode_upload") or ("upload" in key_l)
        def _parse_target_bytes_fallback(b: bytes):
            """
            Preview-only fallback parser for 2-column target files:
              <freq_hz> <mag_db>
            Tolerant to extra whitespace and empty lines.
            """

            try:
                s = b.decode("utf-8", errors="ignore")
            except Exception:
                s = str(b)

            freqs = []
            mags = []
            for line in s.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#") or line.startswith(";") or line.startswith("//"):
                    continue
                parts = line.replace(",", ".").split()
                if len(parts) < 2:
                    continue
                try:
                    freqs.append(float(parts[0]))
                    mags.append(float(parts[1]))
                except Exception:
                    continue

            if len(freqs) < 2:
                raise ValueError("No valid (freq, mag) pairs found")

            ff = np.asarray(freqs, dtype=float)
            yy = np.asarray(mags, dtype=float)
            mask = np.isfinite(ff) & np.isfinite(yy) & (ff > 0)
            ff = ff[mask]
            yy = yy[mask]
            if ff.size < 2:
                raise ValueError("Not enough valid samples after filtering")
            idx = np.argsort(ff)
            return ff[idx], yy[idx]


        if is_upload:
            up = _p("hc_custom_file", None)
            
            # Handle both dict and [dict] variants
            if isinstance(up, list) and len(up) > 0:
                up = up[0]

            if isinstance(up, dict) and (up.get("content") is not None):

                try:
                    # Prefer the same loader used by pipeline
                    if callable(load_target_curve):
                        tf_f, tf_y = load_target_curve(up["content"])
                    else:
                        raise RuntimeError("load_target_curve not available")

                    # load_target_curve() returns (None, None) on failure -> make it fail loudly
                    if tf_f is None or tf_y is None:
                        raise ValueError("load_target_curve() returned no data")

                    tf_f = np.asarray(tf_f, dtype=float)
                    tf_y = np.asarray(tf_y, dtype=float)
                    if tf_f.size >= 2 and tf_y.size == tf_f.size:
                        y = np.interp(f, tf_f, tf_y, left=tf_y[0], right=tf_y[-1])
                        src = "upload"
                    else:
                        raise ValueError("Target data malformed (size mismatch)")
                except Exception as e1:
                    # Fallback: parse the uploaded bytes directly (preview-only)
                    try:
                        tf_f, tf_y = _parse_target_bytes_fallback(up.get("content", b""))
                        y = np.interp(f, tf_f, tf_y, left=tf_y[0], right=tf_y[-1])
                        src = "upload"
                    except Exception as e2:
                        with use_scope("target_preview_scope", clear=True):
                            put_html(
                                "<div style='opacity:0.85; font-size:13px; padding:8px 0;'>"
                                "⚠️ Custom target could not be parsed.<br>"
                                f"<span style='opacity:0.75'>Loader error: {str(e1)}</span><br>"
                                f"<span style='opacity:0.75'>Fallback error: {str(e2)}</span>"
                                "</div>"
                            )
                        return

            # If upload mode selected but no file yet
            if not (isinstance(up, dict) and up.get("content") is not None):
                with use_scope("target_preview_scope", clear=True):
                    put_html(
                        "<div style='opacity:0.8; font-size:13px; padding:8px 0;'>"
                        "⚠️ Custom target selected, but no file loaded yet."
                        "</div>"
                    )
                return

        if y is None:
            # Built-in curve path: try get_house_curve_by_name() in a duck-typed way
            hc = None
            try:
                if callable(get_house_curve_by_name):
                    hc = get_house_curve_by_name(hc_mode)
            except Exception:
                hc = None

            # Fallback: some implementations expose a file-backed loader
            if hc is None:
                try:
                    if callable(load_house_curve):
                        hc = load_house_curve(hc_mode)
                except Exception:
                    hc = None
            if callable(hc):
                y = np.asarray(hc(f), dtype=float)
            elif isinstance(hc, (tuple, list)) and len(hc) >= 2:
                hf = np.asarray(hc[0], dtype=float)
                hy = np.asarray(hc[1], dtype=float)
                if hf.size >= 2 and hy.size == hf.size:
                    y = np.interp(f, hf, hy, left=hy[0], right=hy[-1])
            elif isinstance(hc, dict):
                # Accept several common dict shapes
                for fk, mk in (("freqs", "mags"), ("f", "y"), ("freq", "mag"), ("hz", "db")):
                    if (fk in hc) and (mk in hc):
                        hf = np.asarray(hc[fk], dtype=float)
                        hy = np.asarray(hc[mk], dtype=float)
                        if hf.size >= 2 and hy.size == hf.size:
                            y = np.interp(f, hf, hy, left=hy[0], right=hy[-1])
                            break

        if y is None:
            with use_scope("target_preview_scope", clear=True):
                put_html(
                    "<div style='opacity:0.8; font-size:13px; padding:8px 0;'>"
                    "⚠️ Target preview could not be generated (unknown curve format). "
                    "Try switching the target curve or re-uploading the custom file."
                    "</div>"
                )
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=f, y=y, mode="lines", name=f"Target ({hc_mode_raw})"))

        # Correction band markers
        fig.add_vline(x=max(1.0, mag_c_min), line_width=1, opacity=0.35)
        fig.add_vline(x=max(1.0, mag_c_max), line_width=1, opacity=0.35)

        fig.update_xaxes(
            type="log",
            title_text="Hz",
            range=[math.log10(10.0), math.log10(20000.0)],
            fixedrange=True,
        )
        fig.update_yaxes(
            title_text="dB",
            range=[-30, 20],
            fixedrange=True,
        )
        fig.update_layout(
            height=320,
            width=1800,
            margin=dict(l=40, r=20, t=30, b=35),
            showlegend=True,
            template="plotly_dark",
            uirevision="target_preview_lock"
        )

        html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)

        with use_scope("target_preview_scope", clear=True):
            put_html(
                f"<div style='opacity:0.85; font-size:12.5px; margin:6px 0 8px 0;'>"
                f"Preview source: <b>{src}</b> &nbsp;|&nbsp; Correction band: "
                f"<b>{mag_c_min:.0f}</b>–<b>{mag_c_max:.0f}</b> Hz"
                f"</div>"
            )
            put_html(html)

    except Exception as e:
        with use_scope("target_preview_scope", clear=True):
            put_html(
                "<div style='opacity:0.8; font-size:13px; padding:8px 0;'>"
                "⚠️ Target preview failed. See console/log for details."
                "</div>"
            )
        try:
            print("update_target_preview_ui error:", repr(e))
        except Exception:
            pass