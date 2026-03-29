import logging

from pywebio.output import put_html, put_markdown, use_scope
from pywebio.pin import pin, pin_update, put_checkbox

from ..io.auto_mode.filter_priors import get_auto_mode_filter_auto_defaults
from ..resources.i8n.camillafir_i18n import t
from ..config.mode_policy import MODE_CLAMPS, MODE_DEFAULTS
from .camillafir_utils import scale_taps_with_fs
from .system_health import (
    show_toast,
    toast_afdw_preset_applied,
    toast_mode_defaults_applied,
    toast_tdc_preset_applied,
)

logger = logging.getLogger("CamillaFIR")


def _pin_get(name, default=None):
    try:
        return pin[name]
    except Exception:
        pass

    try:
        getter = getattr(pin, "get", None)
        if callable(getter):
            value = getter(name, None)
            return default if value is None else value
    except Exception:
        pass

    return default


def _is_auto_mode_locked() -> bool:
    try:
        mode_u = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    try:
        auto_flag = bool(_pin_get("camillafir_automatic_mode", False))
    except Exception:
        auto_flag = False
    return bool(mode_u == "AUTO" or auto_flag)


def _as_pin_checkbox_list(v: bool):
    return [True] if bool(v) else []


def _max_boost_help_with_cap():
    cap = float(globals().get("MAX_SAFE_BOOST", 8.0) or 8.0)
    try:
        return (
            f"{t('max_boost_help')}\n\n"
            f"{t('max_boost_help_cap').format(value=f'{cap:.1f}')}"
        )
    except Exception:
        return (
            f"{t('max_boost_help')}\n\n"
            f"{t('max_boost_help_cap').format(value=f'{cap:.1f}')}"
        )


def update_mode_desc(_=None):
    """Soveltaa tai paivittaa: update mode desc."""
    try:
        mode = str(pin["mode"] or "BASIC").strip().upper()
    except Exception:
        mode = "BASIC"
    if mode == "ADVANCED":
        key = "mode_advanced_desc"
    elif mode == "AUTO":
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
                    value = getter(name, None)
                else:
                    value = None
                return default if value is None else value
            except Exception:
                return default

    try:
        mode = str(_p("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode = "BASIC"
    is_auto = mode == "AUTO"
    auto_target_mode = str(_p("auto_target_mode", "auto") or "auto").strip().lower()
    house_curve_locked = bool(is_auto and auto_target_mode in ("auto", "adaptive"))
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
    ['afdw_section_scope', 'tdc_section_scope'].forEach(function(scopeId) {{
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
        limit = clamps.get(cfg_key, None)
        if (not is_basic) or (limit is None):
            return ""
        try:
            lo, hi = limit
        except Exception:
            return ""
        try:
            suffix = t("guide_modes_clamped_suffix").format(lo=lo, hi=hi)
        except Exception:
            suffix = f" (clamped {lo}-{hi})"
        return f"BASIC{suffix}"

    def _merge_help(base_help: str, cfg_key: str) -> str:
        base = str(base_help or "").strip()
        hint = _clamp_hint(cfg_key)
        if not hint:
            return base
        return f"{base}\n\n{hint}" if base else hint

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


def apply_mode_defaults_to_ui(_=None):
    """Soveltaa tai paivittaa: apply mode defaults to ui."""
    from .controls_ir_window import update_ir_tukey_ui, update_lvl_ui

    try:
        mode = str(pin["mode"] or "BASIC").strip().upper()
    except Exception:
        mode = "BASIC"
    if mode not in MODE_DEFAULTS:
        mode = "BASIC"

    defaults = MODE_DEFAULTS.get(mode, {}) or {}
    if mode == "AUTO":
        try:
            filter_type = str(_pin_get("filter_type", defaults.get("filter_type_str", "Asymmetric")) or "")
        except Exception:
            filter_type = str(defaults.get("filter_type_str", "Asymmetric") or "Asymmetric")
        prior_defaults = get_auto_mode_filter_auto_defaults(filter_type)
        if prior_defaults:
            defaults = dict(defaults)
            defaults.update(dict(prior_defaults))

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

    for cfg_key, pin_key in map_num.items():
        if cfg_key in defaults:
            try:
                pin_update(pin_key, value=defaults[cfg_key])
            except Exception:
                pass

    for cfg_key, pin_key in map_str.items():
        if cfg_key in defaults:
            try:
                pin_update(pin_key, value=defaults[cfg_key])
            except Exception:
                pass

    for cfg_key, pin_key in map_chk.items():
        if cfg_key in defaults:
            try:
                pin_update(pin_key, value=_as_pin_checkbox_list(bool(defaults[cfg_key])))
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
        multi_rate = bool(pin["multi_rate_opt"])
    except Exception:
        multi_rate = False
    try:
        base_taps = int(float(pin["taps"]))
    except Exception:
        base_taps = 65536

    for scope_name in ("taps_auto_info_scope_files", "taps_auto_info_scope_basic"):
        with use_scope(scope_name, clear=True):
            if not multi_rate:
                put_markdown(f"_{t('auto_taps_title')}: OFF_")
                continue

            rates = [44100, 48000, 88200, 96000, 176400, 192000]
            lines = [
                f"- **{rate/1000:.1f} kHz** -> **{scale_taps_with_fs(rate, base_taps=base_taps)}** taps"
                for rate in rates
            ]
            put_markdown(
                f"### {t('auto_taps_title')}\n"
                f"{t('auto_taps_body')}\n\n"
                f"**Reference:** 44.1 kHz -> {base_taps:,} taps\n\n"
                + "\n".join(lines)
            )


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
    preset = presets.get(name)
    if not preset:
        return

    pin_update("enable_tdc", value=[True] if preset["enable"] else [])
    pin_update("tdc_strength", value=float(preset["strength"]))
    pin_update("tdc_max_reduction_db", value=float(preset["max_red"]))
    pin_update("tdc_slope_db_per_oct", value=float(preset["slope"]))
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
    preset = presets.get(str(name or ""))
    if not preset:
        return

    try:
        pin_update("enable_afdw", value=[True] if preset["enable"] else [])
    except Exception:
        pass
    try:
        pin_update("fdw_cycles", value=float(preset["cycles"]))
    except Exception:
        pass

    toast_afdw_preset_applied(name)


def update_confidence_pull_ui(*, pin, get_val, t):
    """Piilottaa confidence pull -saadot kayttoliittymasta."""
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


__all__ = [
    "apply_afdw_preset",
    "apply_mode_defaults_to_ui",
    "apply_tdc_preset",
    "update_auto_mode_controls_ui",
    "update_basic_clamp_hints_ui",
    "update_confidence_pull_ui",
    "update_mode_desc",
    "update_taps_auto_info",
    "update_unsafe_raw_dsp_ui",
]
