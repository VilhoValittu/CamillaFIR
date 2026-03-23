import logging
import numpy as np
import math
from pywebio.output import put_button, put_buttons, put_collapse, put_html, put_markdown, put_row, use_scope
from pywebio.input import FLOAT
from pywebio.pin import pin, pin_update, put_checkbox, put_input, put_select

from ..resources.i8n.camillafir_i18n import t
from ..io.auto_mode.filter_priors import get_auto_mode_filter_auto_defaults
from ..io.auto_mode.shared import _auto_goal_forced_level_window
from .camillafir_modes import MODE_DEFAULTS, MODE_CLAMPS
from .camillafir_utils import scale_taps_with_fs
from .system_health import (
    show_toast,
    toast_afdw_preset_applied,
    toast_max_boost_over_cap,
    toast_mode_defaults_applied,
    toast_taps_over_cap,
    toast_tdc_preset_applied,
)

logger = logging.getLogger("CamillaFIR")

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


def _pin_get(name, default=None):
    """Lukee pin-arvon turvallisesti eri PyWebIO-konteksteissa."""
    try:
        return pin[name]
    except Exception:
        pass

    try:
        getter = getattr(pin, "get", None)
        if callable(getter):
            v = getter(name, None)
            return default if v is None else v
    except Exception:
        pass

    return default


def _is_auto_mode_locked() -> bool:
    """Palauttaa tosi kun automaattitila lukitsee kontrollit."""
    try:
        mode_u = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    try:
        auto_flag = bool(_pin_get("camillafir_automatic_mode", False))
    except Exception:
        auto_flag = False
    return bool(mode_u == "AUTO" or auto_flag)


def _render_preset_badges(labels: list[str]) -> None:
    """Renderoi presetit passiivisina tageina ilman klikkauksia."""
    try:
        items = [
            (
                "<span style='display:inline-block; margin:0 8px 8px 0; padding:4px 10px; "
                "border:1px solid rgba(255,255,255,0.18); border-radius:999px; "
                "background:rgba(255,255,255,0.04); color:#d7dde7; font-size:13px;'>"
                f"{str(label or '').strip()}</span>"
            )
            for label in list(labels or [])
            if str(label or "").strip()
        ]
        if items:
            put_html("".join(items))
    except Exception:
        return

def _warn_max_boost_if_over_cap(_=None):
    """Sisainen apufunktio: warn max boost if over cap."""
    try:
        v = _pin_get('max_boost', None)
        if v is None or v == '':
            toast_max_boost_over_cap(None, float(globals().get("MAX_SAFE_BOOST", 8.0) or 8.0))
            return
        v = float(v)
        if not math.isfinite(v):
            toast_max_boost_over_cap(None, float(globals().get("MAX_SAFE_BOOST", 8.0) or 8.0))
            return

        cap = float(globals().get("MAX_SAFE_BOOST", 8.0) or 8.0)
        toast_max_boost_over_cap(v, cap)
    except Exception as e:
        try:
            logger.warning(f"max_boost toast failed: {e}")
        except Exception:
            pass
        return

def _warn_taps_if_over_cap(_=None):
    """Sisainen apufunktio: warn taps if over cap."""
    try:
        v = _pin_get('taps', None)
        if v is None or v == '':
            toast_taps_over_cap(None, int(globals().get("MAX_SAFE_TAPS", 131072) or 131072))
            return
        v = int(v)
        cap = int(globals().get("MAX_SAFE_TAPS", 131072) or 131072)
        toast_taps_over_cap(v, cap)
    except Exception as e:
        try:
            logger.warning(f"taps warning toast failed: {e}")
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
    """Soveltaa tai paivittaa: update mode desc."""
    try:
        m = str(pin["mode"] or "BASIC").strip().upper()
    except Exception:
        m = "BASIC"
    if m == "ADVANCED":
        key = "mode_advanced_desc"
    elif m == "AUTO":
        key = "mode_auto_desc"
    else:
        key = "mode_basic_desc"
    with use_scope("mode_desc_scope", clear=True):
        put_markdown(f"**{t('mode_desc_title')}**\n\n{t(key)}")


def update_auto_mode_controls_ui(_=None):
    """Soveltaa tai paivittaa: auto-mode-only controls state."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            try:
                getter = getattr(pin, "get", None)
                if callable(getter):
                    v = getter(name, None)
                else:
                    v = None
                return default if v is None else v
            except Exception:
                return default

    try:
        m = str(_p("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        m = "BASIC"
    is_auto = (m == "AUTO")
    auto_target_mode = str(_p("auto_target_mode", "auto") or "auto").strip().lower()
    house_curve_locked = bool(is_auto and auto_target_mode != "selected")
    auto_only_disabled = "false" if is_auto else "true"
    auto_locked_disabled = "true" if is_auto else "false"
    house_curve_disabled = "true" if house_curve_locked else "false"

    with use_scope("auto_mode_controls_state_scope", clear=True):
        put_html(
            f"""
<script>
(function() {{
  try {{
    function setState(name, disable) {{
      var els = document.querySelectorAll('[name="' + name + '"]');
      if (!els.length) return;
      els.forEach(function(el) {{
        el.disabled = disable;
      }});
      var wrap = els[0].closest('.form-group') || els[0].closest('.custom-control') || els[0].closest('div');
      if (wrap) {{
        wrap.style.opacity = disable ? '0.55' : '';
        wrap.style.pointerEvents = disable ? 'none' : '';
        wrap.style.filter = disable ? 'grayscale(1)' : '';
      }}
    }}
    function setScopeState(scopeId, disable) {{
      var scope = document.getElementById(scopeId);
      if (!scope) return;
      scope.style.opacity = disable ? '0.55' : '';
      scope.style.filter = disable ? 'grayscale(1)' : '';
      scope.querySelectorAll('button,input,select,textarea').forEach(function(el) {{
        el.disabled = disable;
      }});
    }}
    ['auto_goal', 'auto_target_mode'].forEach(function(name) {{
      setState(name, {auto_only_disabled});
    }});
    [
      'comparison_mode',
      'gain',
      'lvl_algo',
      'lvl_min',
      'lvl_max',
      'lvl_mode',
      'lvl_manual_db',
      'mag_correct',
      'mag_c_min',
      'mag_c_max',
      'max_boost',
      'plot_smoothing_level',
      'bass_first_ai',
      'bass_first_mode_max_hz',
      'max_slope_db_per_oct',
      'max_cut_db',
      'max_slope_boost_db_per_oct',
      'max_slope_cut_db_per_oct',
      'trans_width',
      'filter_smooth',
      'phase_limit',
      'df_smoothing',
      'reg_strength',
      'stereo_link',
      'stereo_link_strategy',
      'exc_prot',
      'exc_freq',
      'low_bass_cut_enable',
      'low_bass_cut_hz',
      'ir_export_window_mode',
      'ir_export_window_shape',
      'ir_export_tukey_alpha',
      'enable_afdw',
      'fdw_cycles',
      'enable_tdc',
      'tdc_strength',
      'tdc_max_reduction_db',
      'tdc_slope_db_per_oct',
      'mixed_freq'
    ].forEach(function(name) {{
      setState(name, {auto_locked_disabled});
    }});
    ['afdw_cycles_scope', 'tdc_controls_scope'].forEach(function(scopeId) {{
      setScopeState(scopeId, {auto_locked_disabled});
    }});
    ['hc_mode', 'hc_custom_file'].forEach(function(name) {{
      setState(name, {house_curve_disabled});
    }});
  }} catch(e) {{}}
}})();
</script>
"""
        )

def update_basic_clamp_hints_ui(*, pin, pin_update, t):
    """Soveltaa tai paivittaa: update basic clamp hints ui."""
    try:
        mode_u = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    is_basic = mode_u in ("BASIC", "AUTO")

    clamps = MODE_CLAMPS.get("BASIC", {}) or {}

    def _clamp_hint(cfg_key: str) -> str:
        lim = clamps.get(cfg_key, None)
        if (not is_basic) or (lim is None):
            return ""
        try:
            lo, hi = lim
        except Exception:
            return ""
        try:
            suf = t("guide_modes_clamped_suffix").format(lo=lo, hi=hi)
        except Exception:
            suf = f" (clamped {lo}-{hi})"
        return f"BASIC{suf}"

    def _merge_help(base_help: str, cfg_key: str) -> str:
        base = str(base_help or "").strip()
        h = _clamp_hint(cfg_key)
        if not h:
            return base
        return f"{base}\n\n{h}" if base else h

    fields = [
        ("mag_c_min", "mag_c_min", lambda: t("hc_range_help")),
        ("mag_c_max", "mag_c_max", lambda: t("hc_range_help")),
        ("max_boost_db", "max_boost", lambda: _max_boost_help_with_cap()),
        ("max_cut_db", "max_cut_db", lambda: t("max_cut_db_help")),
        ("filter_smooth", "filter_smooth", lambda: t("smoothing_level_help")),
        ("reg_strength", "reg_strength", lambda: t("reg_help")),
        ("phase_limit", "phase_limit", lambda: t("phase_limit_help")),
        ("ir_export_window_mode", "ir_export_window_mode", lambda: t("ir_export_window_help")),
        ("enable_tdc", "enable_tdc", lambda: t("tdc_help")),
        ("tdc_strength", "tdc_strength", lambda: t("tdc_help")),
        ("tdc_max_reduction_db", "tdc_max_reduction_db", lambda: t("tdc_max_reduction_db_help")),
        ("tdc_slope_db_per_oct", "tdc_slope_db_per_oct", lambda: t("tdc_slope_db_per_oct_help")),
        ("enable_afdw", "enable_afdw", lambda: t("afdw_help")),
        ("fdw_cycles", "fdw_cycles", lambda: t("fdw_help")),
        ("low_bass_cut_enable", "low_bass_cut_enable", lambda: t("low_bass_cut_hint")),
        ("low_bass_cut_hz", "low_bass_cut_hz", lambda: t("low_bass_cut_hz_help")),
        ("stereo_link", "stereo_link", lambda: t("link_help")),
        ("stereo_link_strategy", "stereo_link_strategy", lambda: t("stereo_link_mode_help")),
    ]

    for cfg_key, pin_key, base_fn in fields:
        try:
            pin_update(pin_key, help_text=_merge_help(base_fn(), cfg_key))
        except Exception:
            pass


def _as_pin_checkbox_list(v: bool):
    return [True] if bool(v) else []

def update_ir_tukey_ui(_=None):
    """Soveltaa tai paivittaa: update ir tukey ui."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
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
        if is_asym_filter and sh != "tukey":
            try:
                pin_update("ir_export_window_shape", value="tukey")
            except Exception:
                pass
            sh = "tukey"
        is_tukey = (sh == "tukey")

        try:
            a = float(_p("ir_export_tukey_alpha", 0.25) or 0.25)
        except Exception:
            a = 0.25
        if not np.isfinite(a):
            a = 0.25
        if is_asym_filter and abs(float(a) - 0.25) > 1e-9:
            try:
                pin_update("ir_export_tukey_alpha", value=0.25)
            except Exception:
                pass
            a = 0.25
        a = float(np.clip(a, 0.0, 1.0))

        with use_scope("ir_tukey_alpha_scope", clear=True):
            if not allow_ir:
                return

            w = put_input(
                "ir_export_tukey_alpha",
                label=t("ir_export_tukey_alpha"),
                type=FLOAT,
                value=a,
                help_text=t("ir_export_tukey_alpha_help"),
            )
            if (not is_tukey) or is_asym_filter:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass

def update_ir_export_window_mode_ui(_=None):
    """Soveltaa tai paivittaa: update ir export window mode ui."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        try:
            m = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            m = "BASIC"
        is_basic = m in ("BASIC", "AUTO")

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
        lock_window_mode = bool(is_basic or (not allow_ir) or is_asym_filter)

        cur = str(_p("ir_export_window_mode", "auto") or "auto").strip().lower()

        if is_asym_filter:
            if cur != "rew_asym":
                try:
                    pin_update("ir_export_window_mode", value="rew_asym")
                except Exception:
                    pass
                cur = "rew_asym"
        elif is_basic and cur != "auto":
            try:
                pin_update("ir_export_window_mode", value="auto")
            except Exception:
                pass
            cur = "auto"

        if (not allow_ir) and (cur != "auto"):
            try:
                pin_update("ir_export_window_mode", value="auto")
            except Exception:
                pass
            cur = "auto"


        if (not is_linear) and (not is_asym_filter) and (cur == "rew_asym"):
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
            if lock_window_mode:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            js_linear_only = bool((not is_linear) and (not is_asym_filter))
            js_disable = "true" if (lock_window_mode or js_linear_only) else "false"
            js_suffix = "(Linear only)" if js_linear_only else "(Locked)"
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
    var base = opt.textContent.replace(/\\s*\\((Linear only|Locked)\\)\\s*$/,'');
    if({js_disable}) opt.textContent = base + " {js_suffix}";
    else opt.textContent = base;

    // If somehow selected while disabled (older browser state), force select back to auto
    if(opt.disabled && sel.value === "rew_asym" && {str(is_asym_filter).lower()} === false) {{
      sel.value = "auto";
    }}
  }} catch(e) {{}}
}})();
</script>
""")
            try:
                msg_ = t("ir_asym_linear_only")
                
            except Exception:
                msg_ = ""

            emph = bool((not is_asym_filter) and (is_basic or (not is_linear)))
            opacity = "1.0" if emph else "0.55"
            color = "#ffb74d" if emph else "#9aa0a6"

            if str(msg_ or "").strip():
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
    """Soveltaa tai paivittaa: update ir lr window ui."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        app_mode = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        is_auto_mode = bool(app_mode == "AUTO")
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

        try:
            v_left = float(_p("ir_window_left", 85.0) or 85.0)
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

            if not enable_left or is_auto_mode:
                w_left.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            if not enable_right or is_auto_mode:
                w_right.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")

            if (not enable_left and not enable_right) or is_auto_mode:
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
    """Soveltaa tai paivittaa: update ir window shape ui."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        mode = str(_p("ir_export_window_mode", "auto") or "auto").strip().lower()
        m = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        ft = str(_p("filter_type", "") or "").strip().lower()
        try:
            ft_asym_label = str(t("ft_asymmetric") or "").strip().lower()
        except Exception:
            ft_asym_label = "asymmetric"
        is_asym_filter = (ft == ft_asym_label) or ("asym" in ft)
        if is_asym_filter and str(_p("ir_export_window_shape", "hann") or "hann").strip().lower() != "tukey":
            try:
                pin_update("ir_export_window_shape", value="tukey")
            except Exception:
                pass
        is_basic = m in ("BASIC", "AUTO")
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

            if is_auto or is_asym_filter:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass

def update_mixed_freq_ui(_=None):
    """Soveltaa tai paivittaa: update mixed freq ui."""
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
                value=float(_p("mixed_freq", 180.0) or 180.0),
                help_text=t("mixed_freq_help"),
            )
            if not is_mixed:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass
    
def update_low_bass_cut_ui(*, pin, pin_update, get_val, t):
    """Soveltaa tai paivittaa: update low bass cut ui."""

    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        en_raw = _p("low_bass_cut_enable", None)
        if isinstance(en_raw, (list, tuple, set)):
            enabled = (len(en_raw) > 0)
        else:
            enabled = bool(en_raw) if en_raw is not None else bool(get_val("low_bass_cut_enable", True))
    except Exception:
        enabled = bool(get_val("low_bass_cut_enable", True))

    cur = _p("low_bass_cut_hz", "")

    last = _p("low_bass_cut_hz_last", get_val("low_bass_cut_hz", 40.0))

    try:
        if not enabled:
            if cur not in ("", None):
                try:
                    pin_update("low_bass_cut_hz_last", value=float(cur))
                except Exception:
                    pass
                pin_update("low_bass_cut_hz", value="")
            disp = float(last or 40.0)
        else:
            if cur in ("", None):
                v0 = float(last or 40.0)
                pin_update("low_bass_cut_hz", value=v0)
                cur = v0
            disp = float(cur)
            if not math.isfinite(disp):
                disp = float(last or 40.0)
                pin_update("low_bass_cut_hz", value=disp)
            pin_update("low_bass_cut_hz_last", value=float(disp))
    except Exception:
        disp = 40.0

    with use_scope("low_bass_cut_scope", clear=True):
        w = put_input(
            "low_bass_cut_hz",
            label=f"{t('low_bass_cut_hz')} (Hz)",
            type=FLOAT,
            value=float(disp),
            help_text=t("low_bass_cut_hz_help"),
        )

        if not enabled:
            w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            put_html(
                "<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>OFF</div>"
            )

def update_afdw_cycles_ui(*, pin, get_val, t):
    """Soveltaa tai paivittaa: update afdw cycles ui."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        app_mode = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        is_auto_mode = bool(app_mode == "AUTO")
        en_raw = _p("enable_afdw", None)
        if isinstance(en_raw, (list, tuple, set)):
            enabled = (len(en_raw) > 0)
        else:
            enabled = bool(en_raw) if en_raw is not None else bool(get_val("enable_afdw", True))
    except Exception:
        enabled = bool(get_val("enable_afdw", True))

    try:
        v = float(_p("fdw_cycles", get_val("fdw_cycles", 10.0)) or 10.0)
    except Exception:
        v = float(get_val("fdw_cycles", 10.0) or 10.0)

    with use_scope("afdw_cycles_scope", clear=True):
        preset_labels = [
            t("afdw_preset_tight"),
            t("afdw_preset_balanced"),
            t("afdw_preset_safe"),
            t("afdw_preset_minimal"),
        ]
        if (not enabled) or is_auto_mode:
            _render_preset_badges(preset_labels)
        else:
            put_row(
                [
                    put_buttons(
                        [
                            {"label": preset_labels[0], "value": "Tight"},
                            {"label": preset_labels[1], "value": "Balanced"},
                            {"label": preset_labels[2], "value": "Safe"},
                            {"label": preset_labels[3], "value": "Minimal"},
                        ],
                        onclick=lambda preset: apply_afdw_preset(preset),
                        small=True,
                    ),
                ]
            )
        put_html(f"<div style='opacity:0.65; font-size:13px'>{t('afdw_preset_help')}</div>")
        w = put_input(
            "fdw_cycles",
            label=t("fdw"),
            type=FLOAT,
            value=v,
            help_text=t("fdw_help"),
        )
        if (not enabled) or is_auto_mode:
            w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            try:
                disabled_hint = str(t("afdw_disabled_hint") or "").strip()
            except Exception:
                disabled_hint = ""
            if disabled_hint.lower() == "afdw_disabled_hint":
                disabled_hint = ""
            put_html(
                f"<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>"
                f"{disabled_hint or 'OFF'}"
                f"</div>"
            )
            put_html(
                "<script>"
                "(function(){"
                "var box=document.getElementById('afdw_cycles_scope');"
                "if(!box) return;"
                "box.style.opacity='0.55';"
                "box.style.filter='grayscale(1)';"
                "box.querySelectorAll('button,input,select,textarea').forEach(function(el){el.disabled=true;});"
                "})();"
                "</script>"
            )

def update_tdc_controls_ui(*, pin, get_val, t, apply_tdc_preset):
    """Soveltaa tai paivittaa: update tdc controls ui."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        app_mode = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        is_auto_mode = bool(app_mode == "AUTO")
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
        preset_labels = [
            t("tdc_preset_safe"),
            t("tdc_preset_normal"),
            t("tdc_preset_aggressive"),
        ]
        if (not enabled) or is_auto_mode:
            _render_preset_badges(preset_labels)
        else:
            put_row([
                put_buttons(
                    [
                        {"label": preset_labels[0], "value": "Safe"},
                        {"label": preset_labels[1], "value": "Normal"},
                        {"label": preset_labels[2], "value": "Aggressive"},
                    ],
                    onclick=lambda preset: apply_tdc_preset(preset),
                    small=True,
                ),
            ])

        put_html(f"<div style='opacity:0.65; font-size:12px; line-height:1.25; margin-top:6px'>{t('tdc_preset_help')}</div>")
        put_html(f"<div style='opacity:0.70; font-size:12px; line-height:1.25; margin-top:4px'>{t('tdc_summary_hint')}</div>")

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

    if (not enabled) or is_auto_mode:
        put_html(
            "<script>"
            "(function(){"
            "var box=document.getElementById('tdc_controls_scope');"
            "if(!box) return;"
            "box.style.opacity='0.55';"
            "box.style.filter='grayscale(1)';"
            "box.querySelectorAll('button,input,select,textarea').forEach(function(el){el.disabled=true;});"
            "})();"
            "</script>"
        )




def apply_mode_defaults_to_ui(_=None):
    """Soveltaa tai paivittaa: apply mode defaults to ui."""
    try:
        mode = str(pin["mode"] or "BASIC").strip().upper()
    except Exception:
        mode = "BASIC"
    if mode not in MODE_DEFAULTS:
        mode = "BASIC"

    d = MODE_DEFAULTS.get(mode, {}) or {}
    if mode == "AUTO":
        try:
            filter_type = str(_pin_get("filter_type", d.get("filter_type_str", "Asymmetric")) or "")
        except Exception:
            filter_type = str(d.get("filter_type_str", "Asymmetric") or "Asymmetric")
        prior_defaults = get_auto_mode_filter_auto_defaults(filter_type)
        if prior_defaults:
            d = dict(d)
            d.update(dict(prior_defaults))

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
        "ir_window_right": "ir_window",
        "ir_window_left": "ir_window_left",
        "mixed_split_freq": "mixed_freq",
        "trans_width": "trans_width",
        "bass_first_mode_max_hz": "bass_first_mode_max_hz",
        "max_slope_db_per_oct": "max_slope_db_per_oct",
        "max_slope_boost_db_per_oct": "max_slope_boost_db_per_oct",
        "max_slope_cut_db_per_oct": "max_slope_cut_db_per_oct",
        "lvl_manual_db": "lvl_manual_db",
        "lvl_min": "lvl_min",
        "lvl_max": "lvl_max",
        "conf_pull_floor": "conf_pull_floor",
        "conf_pull_max_hz": "conf_pull_max_hz",
        "conf_pull_gamma_cut": "conf_pull_gamma_cut",
        "conf_pull_gamma_boost": "conf_pull_gamma_boost",
        "low_bass_cut_strength": "low_bass_cut_strength",
    }
    map_str = {
        "filter_type_str": "filter_type",
        "plot_smoothing_level": "plot_smoothing_level",
        "lvl_mode": "lvl_mode",
        "lvl_algo": "lvl_algo",
        "stereo_link_strategy": "stereo_link_strategy",
    }
    map_chk = {
        "enable_mag_correction": "mag_correct",
        "unsafe_raw_dsp": "unsafe_raw_dsp",
        "exc_prot": "exc_prot",
        "enable_tdc": "enable_tdc",
        "enable_afdw": "enable_afdw",
        "df_smoothing": "df_smoothing",
        "comparison_mode": "comparison_mode",
        "bass_first_ai": "bass_first_ai",
        "phase_safe_2058": "phase_safe_2058",
        "stereo_link": "stereo_link",
        "low_bass_cut_enable": "low_bass_cut_enable",
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
    try:
        if mode == "AUTO":
            pin_update("auto_target_mode", value="auto")
    except Exception:
        pass
    update_mode_desc()

    toast_mode_defaults_applied(mode)

def update_taps_auto_info(_=None):
    """Soveltaa tai paivittaa: update taps auto info."""
    try:
        mr = bool(pin["multi_rate_opt"])
    except Exception:
        mr = False
    try:
        base_taps = int(float(pin["taps"]))
    except Exception:
        base_taps = 65536

    for scope_name in ("taps_auto_info_scope_files", "taps_auto_info_scope_basic"):
        with use_scope(scope_name, clear=True):
            if not mr:
                put_markdown(f"_{t('auto_taps_title')}: OFF_")
                continue

            rates = [44100, 48000, 88200, 96000, 176400, 192000]
            lines = [f"- **{r/1000:.1f} kHz** -> **{scale_taps_with_fs(r, base_taps=base_taps)}** taps" for r in rates]
            put_markdown(
                f"### {t('auto_taps_title')}\n"
                f"{t('auto_taps_body')}\n\n"
                f"**Reference:** 44.1 kHz -> {base_taps:,} taps\n\n"
                + "\n".join(lines)
            )


def update_lvl_ui(_=None):
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        try:
            app_mode = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            app_mode = "BASIC"
        is_basic = app_mode in ("BASIC", "AUTO")

        mode = str(_p("lvl_mode", "Auto") or "Auto")
        if is_basic:
            mode = "Auto"
        is_manual = ("Manual" in mode)

        prev_mode = getattr(update_lvl_ui, "_last_lvl_mode", None)
        try:
            cur_min = float(_p("lvl_min", 500.0) or 500.0)
            cur_max = float(_p("lvl_max", 2000.0) or 2000.0)
        except Exception:
            cur_min, cur_max = 500.0, 2000.0
        if cur_min > cur_max:
            cur_min, cur_max = cur_max, cur_min

        if not hasattr(update_lvl_ui, "_lvl_auto_range"):
            setattr(update_lvl_ui, "_lvl_auto_range", (float(cur_min), float(cur_max)))
        if not hasattr(update_lvl_ui, "_lvl_manual_range"):
            setattr(update_lvl_ui, "_lvl_manual_range", (float(cur_min), float(cur_max)))

        if prev_mode is not None and str(prev_mode) != str(mode):
            try:
                if "Manual" in str(prev_mode):
                    setattr(update_lvl_ui, "_lvl_manual_range", (float(cur_min), float(cur_max)))
                else:
                    setattr(update_lvl_ui, "_lvl_auto_range", (float(cur_min), float(cur_max)))
            except Exception:
                pass

            try:
                if is_manual:
                    r_min, r_max = getattr(update_lvl_ui, "_lvl_manual_range", (cur_min, cur_max))
                else:
                    r_min, r_max = getattr(update_lvl_ui, "_lvl_auto_range", (cur_min, cur_max))
                r_min = float(r_min)
                r_max = float(r_max)
                if r_min > r_max:
                    r_min, r_max = r_max, r_min
                pin_update("lvl_min", value=r_min)
                pin_update("lvl_max", value=r_max)
                cur_min, cur_max = r_min, r_max
            except Exception:
                pass

        setattr(update_lvl_ui, "_last_lvl_mode", str(mode))

        try:
            pin_update("lvl_min", help_text=t("lvl_min_help_manual" if is_manual else "lvl_min_help_auto"))
            pin_update("lvl_max", help_text=t("lvl_max_help_manual" if is_manual else "lvl_max_help_auto"))
        except Exception:
            pass

        vmin = float(_p("lvl_min", cur_min) or cur_min)
        vmax = float(_p("lvl_max", cur_max) or cur_max)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
            pin_update("lvl_min", value=vmin)
            pin_update("lvl_max", value=vmax)

        try:
            if is_manual:
                setattr(update_lvl_ui, "_lvl_manual_range", (float(vmin), float(vmax)))
            else:
                setattr(update_lvl_ui, "_lvl_auto_range", (float(vmin), float(vmax)))
        except Exception:
            pass

        def _step_manual_target(delta_db: float):
            try:
                cur = float(_p("lvl_manual_db", 0.0) or 0.0)
            except Exception:
                cur = 0.0
            nxt = round((float(cur) + float(delta_db)) * 10.0) / 10.0
            try:
                pin_update("lvl_manual_db", value=float(nxt))
            except Exception:
                pass
            try:
                update_target_preview_ui()
            except Exception:
                pass

        with use_scope("lvl_manual_scope", clear=True):
            try:
                _bias_hint = t("lvl_manual_bias_hint")
            except Exception:
                _bias_hint = ""

            w = put_row([
                put_input(
                    "lvl_manual_db",
                    label=t("lvl_target_db"),
                    type=FLOAT,
                    value=float(_p("lvl_manual_db", 0.0) or 0.0), # type: ignore
                    help_text=t("lvl_manual_help"),
                ),
                put_button("+", onclick=lambda: _step_manual_target(-0.1), color="secondary").style("margin-top:28px; min-width:34px; margin-right:4px;"),
                put_button("-", onclick=lambda: _step_manual_target(+0.1), color="secondary").style("margin-top:28px; min-width:34px;"),
            ], size="1fr auto auto")
            put_html(
                f"<div style='opacity:0.75; font-size:12px; margin-top:4px;'>"
                f"{_bias_hint}"
                f"</div>"
            )
            if not is_manual:
                w.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass


def apply_tdc_preset(name: str):
    if _is_auto_mode_locked():
        show_toast(
            "TDC preset locked in Automatic mode",
            color="info",
            duration=1.8,
            dedupe_key="tdc_preset_locked_auto",
        )
        return

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

    toast_tdc_preset_applied(name)


def apply_afdw_preset(name: str):
    if _is_auto_mode_locked():
        show_toast(
            "A-FDW preset locked in Automatic mode",
            color="info",
            duration=1.8,
            dedupe_key="afdw_preset_locked_auto",
        )
        return

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

    toast_afdw_preset_applied(name)


def _pretty_plot_smoothing(v, t):
    if isinstance(v, str) and v.strip().lower() == "psychoacoustic":
        return t("smooth_safe_reference")
    return str(v)

def _fmt_mode_value(key: str, defaults: dict, clamps: dict):
    """Sisainen apufunktio: fmt mode value."""
    v = defaults.get(key, None)
    if key == "plot_smoothing_level":
        v_str = _pretty_plot_smoothing(v)
    else:
        v_str = str(v)

    lim = clamps.get(key, None) if isinstance(clamps, dict) else None
    if isinstance(lim, tuple) and len(lim) == 2:
        lo, hi = lim
        if isinstance(lo, bool) and isinstance(hi, bool):
            return f"**{v_str}**"
        return f"**{v_str}** _(clamped to {lo}–{hi})_"
    return f"**{v_str}**"

def _build_modes_guide_parts(t):
    """Sisainen apufunktio: build modes guide parts."""
    

    d_basic = MODE_DEFAULTS.get("BASIC", {})
    c_basic = MODE_CLAMPS.get("BASIC", {})
    d_adv   = MODE_DEFAULTS.get("ADVANCED", {})
    c_adv   = MODE_CLAMPS.get("ADVANCED", {})

    def clamp_suffix(key, clamps):
        lim = clamps.get(key)
        if isinstance(lim, tuple) and len(lim) == 2:
            lo, hi = lim
            return t("guide_modes_clamped_suffix").format(lo=lo, hi=hi)
        return ""

    intro = t("guide_modes_intro") + "\n"

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
        ("guide_exc_prot", t("guide_exc_prot_title")),
        ("guide_low_bass_cut", t("guide_low_bass_cut_title")),
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

    put_collapse(t("guide_section_title"), content, open=False)


def update_confidence_pull_ui(*, pin, get_val, t):
    """Piilottaa confidence pull -saadot kayttoliittymasta.

    DSP-logiikka ja sisaiset oletusarvot jaavat edelleen kayttoon.
    """
    try:
        with use_scope("conf_pull_scope", clear=True):
            return
    except Exception:
        pass


def update_unsafe_raw_dsp_ui(*, pin, get_val, t):
    """Renderoi UNSAFE/Raw DSP -kytkimen vain ADVANCED-tilaan."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        mode_u = str(_p("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"

    try:
        with use_scope("unsafe_raw_dsp_scope", clear=True):
            if mode_u != "ADVANCED":
                return

            put_checkbox(
                "unsafe_raw_dsp",
                options=[{"label": t("unsafe_raw_dsp_label"), "value": True}],
                value=[True] if bool(get_val("unsafe_raw_dsp", False)) else [],
                help_text=t("unsafe_raw_dsp_help"),
            )
            put_html(
                "<div style='margin-top:6px; color:#d32f2f; font-weight:700; font-size:13px;'>"
                f"{t('unsafe_raw_dsp_warning')}"
                "</div>"
            )
    except Exception:
        pass


def update_target_preview_ui(_=None):
    """Soveltaa tai paivittaa: update target preview ui."""
    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    def _norm_key(x: str) -> str:
        s = str(x or "").strip()
        if not s:
            return ""
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
        from ..dsp.smoothing import psychoacoustic_smoothing as _psycho_smooth
        from ..io.measurements_txt import (
            parse_measurements_from_bytes as _parse_txt_bytes,
            parse_measurements_from_path as _parse_txt_path,
        )
        from ..io.measurements_wav import (
            parse_measurements_from_wav_bytes as _parse_wav_bytes,
            parse_measurements_from_wav_path as _parse_wav_path,
        )

        def _to_float(v, default):
            try:
                x = float(v)
                if np.isfinite(x):
                    return x
            except Exception:
                pass
            return float(default)

        def _normalize_curve(freqs, mags):
            try:
                ff = np.asarray(freqs, dtype=float)
                mm = np.asarray(mags, dtype=float)
                if ff.size < 8 or mm.size != ff.size:
                    return None, None
                mask = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.0)
                ff = ff[mask]
                mm = mm[mask]
                if ff.size < 8:
                    return None, None
                order = np.argsort(ff)
                ff = ff[order]
                mm = mm[order]
                uniq, idx = np.unique(ff, return_index=True)
                ff = uniq
                mm = mm[idx]
                if ff.size < 8:
                    return None, None
                return ff, mm
            except Exception:
                return None, None

        def _pick_upload(name):
            v = _p(name, None)
            if isinstance(v, list) and len(v) > 0:
                v = v[0]
            if isinstance(v, dict) and (v.get("content") is not None):
                return v
            return None

        def _parse_uploaded_measurement(up):
            if not isinstance(up, dict):
                return None, None
            content = up.get("content", None)
            if content is None:
                return None, None

            name = str(up.get("filename", "") or "").strip().lower()
            ext = "." + name.rsplit(".", 1)[1] if "." in name else ""
            pre_ms = _to_float(_p("ir_window_left", 85.0), 85.0)
            post_raw = _p("ir_window_right", None)
            if post_raw in (None, ""):
                post_raw = _p("ir_window", 500.0)
            post_ms = _to_float(post_raw, 500.0)
            try:
                sl = int(float(_p("filter_smooth", _p("smoothing_level", 0)) or 0))
            except Exception:
                sl = 0

            try:
                is_wav = (ext == ".wav") or (
                    isinstance(content, (bytes, bytearray)) and len(content) >= 4 and content[:4] == b"RIFF"
                )
                if is_wav:
                    ff, mm, _ = _parse_wav_bytes(
                        content,
                        pre_ms=pre_ms,
                        post_ms=post_ms,
                        smoothing_level=sl,
                        logger=None,
                    )
                else:
                    ff, mm, _ = _parse_txt_bytes(content)
            except Exception:
                return None, None
            return _normalize_curve(ff, mm)

        def _parse_local_measurement(path_raw):
            p = str(path_raw or "").strip().strip('"').strip("'")
            if not p:
                return None, None
            p_l = p.lower()
            pre_ms = _to_float(_p("ir_window_left", 85.0), 85.0)
            post_raw = _p("ir_window_right", None)
            if post_raw in (None, ""):
                post_raw = _p("ir_window", 500.0)
            post_ms = _to_float(post_raw, 500.0)
            try:
                sl = int(float(_p("filter_smooth", _p("smoothing_level", 0)) or 0))
            except Exception:
                sl = 0
            try:
                if p_l.endswith(".wav"):
                    ff, mm, _ = _parse_wav_path(
                        p,
                        pre_ms=pre_ms,
                        post_ms=post_ms,
                        smoothing_level=sl,
                        logger=None,
                    )
                else:
                    ff, mm, _ = _parse_txt_path(p, logger=None)
            except Exception:
                return None, None
            return _normalize_curve(ff, mm)

        def _align_to_target_window(m_curve, t_curve, freq_axis, fmin, fmax):
            try:
                m = np.asarray(m_curve, dtype=float)
                t_ = np.asarray(t_curve, dtype=float)
                fx = np.asarray(freq_axis, dtype=float)
                if m.size != fx.size or t_.size != fx.size:
                    return m
                mask = (fx >= float(fmin)) & (fx <= float(fmax)) & np.isfinite(m) & np.isfinite(t_)
                if np.count_nonzero(mask) < 16:
                    return m
                off = float(np.median(m[mask] - t_[mask]))
                if not np.isfinite(off):
                    return m
                return m - off
            except Exception:
                return np.asarray(m_curve, dtype=float)

        def _smooth_for_preview(freq_axis, mags_curve):
            try:
                return np.asarray(_psycho_smooth(freq_axis, mags_curve), dtype=float)
            except Exception:
                return np.asarray(mags_curve, dtype=float)

        hc_mode_raw = str(_p("hc_mode", "Harman6") or "Harman6")
        hc_mode = _norm_key(hc_mode_raw)

        mag_c_min = _to_float(_p("mag_c_min", 10.0), 10.0)
        mag_c_max = _to_float(_p("mag_c_max", 200.0), 200.0)

        lvl_min = _to_float(_p("lvl_min", 500.0), 500.0)
        lvl_max = _to_float(_p("lvl_max", 2000.0), 2000.0)
        if not (lvl_min > 0.0 and lvl_max > lvl_min):
            lvl_min, lvl_max = 500.0, 2000.0
        try:
            app_mode = str(_p("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            app_mode = "BASIC"
        auto_goal = str(_p("auto_goal", "balanced") or "balanced").strip().lower()
        forced_level_window = _auto_goal_forced_level_window(auto_goal) if app_mode == "AUTO" else None

        lvl_mode = str(_p("lvl_mode", "Auto") or "Auto")
        if app_mode in ("BASIC", "AUTO"):
            lvl_mode = "Auto"
        if forced_level_window is not None:
            lvl_min, lvl_max = float(forced_level_window[0]), float(forced_level_window[1])

        is_manual_level = ("manual" in lvl_mode.strip().lower())
        lvl_mode_label = t("lvl_mode_manual") if is_manual_level else t("lvl_mode_auto")
        lvl_manual_db = _to_float(_p("lvl_manual_db", 0.0), 0.0)
        preview_level_shift_db = lvl_manual_db if is_manual_level else 0.0

        f = np.logspace(np.log10(10.0), np.log10(20000.0), 600)

        y = None
        src = t("target_preview_source_builtin")

        key_l = str(hc_mode).strip().lower()
        is_upload = key_l in ("upload", "custom", "hc_mode_upload") or ("upload" in key_l)

        def _parse_target_bytes_fallback(b: bytes):
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
            if isinstance(up, list) and len(up) > 0:
                up = up[0]

            if isinstance(up, dict) and (up.get("content") is not None):
                try:
                    if callable(load_target_curve):
                        tf_f, tf_y = load_target_curve(up["content"])
                    else:
                        raise RuntimeError("load_target_curve not available")

                    if tf_f is None or tf_y is None:
                        raise ValueError("load_target_curve() returned no data")

                    tf_f = np.asarray(tf_f, dtype=float)
                    tf_y = np.asarray(tf_y, dtype=float)
                    if tf_f.size >= 2 and tf_y.size == tf_f.size:
                        y = np.interp(f, tf_f, tf_y, left=tf_y[0], right=tf_y[-1])
                        src = t("target_preview_source_upload")
                    else:
                        raise ValueError("Target data malformed (size mismatch)")
                except Exception as e1:
                    try:
                        tf_f, tf_y = _parse_target_bytes_fallback(up.get("content", b""))
                        y = np.interp(f, tf_f, tf_y, left=tf_y[0], right=tf_y[-1])
                        src = t("target_preview_source_upload")
                    except Exception as e2:
                        with use_scope("target_preview_scope", clear=True):
                            put_html(
                                "<div style='opacity:0.85; font-size:13px; padding:8px 0;'>"
                                "Custom target could not be parsed.<br>"
                                f"<span style='opacity:0.75'>Loader error: {str(e1)}</span><br>"
                                f"<span style='opacity:0.75'>Fallback error: {str(e2)}</span>"
                                "</div>"
                            )
                        return

            if not (isinstance(up, dict) and up.get("content") is not None):
                with use_scope("target_preview_scope", clear=True):
                    put_html(
                        "<div style='opacity:0.8; font-size:13px; padding:8px 0;'>"
                        "Custom target selected, but no file loaded yet."
                        "</div>"
                    )
                return

        if y is None:
            hc = None
            try:
                if callable(get_house_curve_by_name):
                    hc = get_house_curve_by_name(hc_mode)
            except Exception:
                hc = None

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
                    "Target preview could not be generated (unknown curve format). "
                    "Try switching the target curve or re-uploading the custom file."
                    "</div>"
                )
            return

        speaker_curves = {}
        for ch, up_key, path_key in (
            ("L", "file_l", "local_path_l"),
            ("R", "file_r", "local_path_r"),
        ):
            up = _pick_upload(up_key)
            ff, mm = (None, None)
            if up is not None:
                ff, mm = _parse_uploaded_measurement(up)
            if ff is None or mm is None:
                ff, mm = _parse_local_measurement(_p(path_key, ""))
            if ff is not None and mm is not None:
                speaker_curves[ch] = (ff, mm)

        speaker_interp = {}
        for ch, (ff, mm) in speaker_curves.items():
            m_raw = np.interp(f, ff, mm, left=mm[0], right=mm[-1])
            m_aligned = _align_to_target_window(m_raw, y, f, lvl_min, lvl_max)
            speaker_interp[ch] = _smooth_for_preview(f, m_aligned)

        if abs(preview_level_shift_db) > 1e-9:
            for _k in list(speaker_interp.keys()):
                speaker_interp[_k] = np.asarray(speaker_interp[_k], dtype=float) + float(preview_level_shift_db)

        speaker_avg = None
        if len(speaker_interp) > 0:
            speaker_avg = np.mean(np.vstack([speaker_interp[k] for k in sorted(speaker_interp.keys())]), axis=0)

        speaker_label = "No speaker data loaded"
        if "L" in speaker_interp and "R" in speaker_interp:
            speaker_label = f"L + R (aligned {lvl_min:.0f}-{lvl_max:.0f} Hz)"
        elif "L" in speaker_interp:
            speaker_label = f"L only (aligned {lvl_min:.0f}-{lvl_max:.0f} Hz)"
        elif "R" in speaker_interp:
            speaker_label = f"R only (aligned {lvl_min:.0f}-{lvl_max:.0f} Hz)"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=f,
                y=y,
                mode="lines",
                name=f"Target ({hc_mode_raw})",
                line=dict(color="#4caf50", width=2.0),
            )
        )

        if "L" in speaker_interp:
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=speaker_interp["L"],
                    mode="lines",
                    name="Speaker L",
                    line=dict(color="rgba(102, 187, 255, 0.55)", width=1.2),
                )
            )
        if "R" in speaker_interp:
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=speaker_interp["R"],
                    mode="lines",
                    name="Speaker R",
                    line=dict(color="rgba(255, 167, 102, 0.55)", width=1.2),
                )
            )
        if speaker_avg is not None:
            fig.add_trace(
                go.Scatter(
                    x=f,
                    y=speaker_avg,
                    mode="lines",
                    name="Speaker avg",
                    line=dict(color="#ffd166", width=2.0),
                )
            )

        fig.add_vrect(
            x0=max(1.0, lvl_min),
            x1=max(1.0, lvl_max),
            fillcolor="rgba(180, 180, 180, 0.16)",
            line_width=0,
            layer="below",
        )

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
            range=[-10.0, 20.0],
            fixedrange=True,
        )
        fig.update_layout(
            height=320,
            width=1800,
            margin=dict(l=40, r=20, t=30, b=35),
            showlegend=True,
            template="plotly_dark",
            uirevision="target_preview_lock",
        )

        html = pio.to_html(fig, include_plotlyjs=True, full_html=False)

        with use_scope("target_preview_scope", clear=True):
            put_html(
                f"<div style='opacity:0.85; font-size:12.5px; margin:6px 0 8px 0;'>"
                f"{t('target_preview_source_label')}: <b>{src}</b> &nbsp;|&nbsp; {t('target_preview_correction_band_label')}: "
                f"<b>{mag_c_min:.0f}</b>-<b>{mag_c_max:.0f}</b> Hz &nbsp;|&nbsp; Level window: <b>{lvl_min:.0f}-{lvl_max:.0f} Hz</b> "
                f"&nbsp;|&nbsp; Level mode: <b>{lvl_mode_label}</b>"
                f"{f' (target {lvl_manual_db:.1f} dB, speaker shift {preview_level_shift_db:+.1f} dB)' if is_manual_level else ''}"
                f" &nbsp;|&nbsp; Speaker data: <b>{speaker_label}</b>"
                f"</div>"
            )
            put_html(html)

    except Exception as e:
        with use_scope("target_preview_scope", clear=True):
            put_html(
                "<div style='opacity:0.8; font-size:13px; padding:8px 0;'>"
                "Target preview failed. See console/log for details."
                "</div>"
            )
        try:
            logger.warning("update_target_preview_ui error: %r", e)
        except Exception:
            pass
