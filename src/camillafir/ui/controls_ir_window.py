import math

import numpy as np
from pywebio.input import FLOAT
from pywebio.output import put_button, put_buttons, put_html, put_row, use_scope
from pywebio.pin import pin, pin_update, put_checkbox, put_input, put_select

from ..resources.i8n.camillafir_i18n import t


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


def _render_preset_badges(labels: list[str]) -> None:
    try:
        items = [
            (
                "<span style='display:inline-block; margin:0 4px 0 0; padding:0 10px; "
                "min-height:30px; line-height:28px; "
                "border:1px solid rgba(255,255,255,0.18); border-radius:999px; "
                "background:rgba(255,255,255,0.04); color:#d7dde7; font-size:12px;'>"
                f"{str(label or '').strip()}</span>"
            )
            for label in list(labels or [])
            if str(label or "").strip()
        ]
        if items:
            put_html(
                "<div style='display:flex; flex-wrap:wrap; gap:4px; margin-bottom:4px;'>"
                + "".join(items)
                + "</div>"
            )
    except Exception:
        return


def update_ir_tukey_ui(_=None):
    """Soveltaa tai paivittaa: update ir tukey ui."""
    try:
        ft = str(_pin_get("filter_type", "") or "").strip().lower()
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

        shape = str(_pin_get("ir_export_window_shape", "hann") or "hann").strip().lower()
        if is_asym_filter and shape != "tukey":
            try:
                pin_update("ir_export_window_shape", value="tukey")
            except Exception:
                pass
            shape = "tukey"
        is_tukey = shape == "tukey"

        try:
            alpha = float(_pin_get("ir_export_tukey_alpha", 0.25) or 0.25)
        except Exception:
            alpha = 0.25
        if not np.isfinite(alpha):
            alpha = 0.25
        if is_asym_filter and abs(float(alpha) - 0.25) > 1e-9:
            try:
                pin_update("ir_export_tukey_alpha", value=0.25)
            except Exception:
                pass
            alpha = 0.25
        alpha = float(np.clip(alpha, 0.0, 1.0))

        with use_scope("ir_tukey_alpha_scope", clear=True):
            if not allow_ir:
                return

            widget = put_input(
                "ir_export_tukey_alpha",
                label=t("ir_export_tukey_alpha"),
                type=FLOAT,
                value=alpha,
                help_text=t("ir_export_tukey_alpha_help"),
            )
            if (not is_tukey) or is_asym_filter:
                widget.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass


def update_ir_export_window_mode_ui(_=None):
    """Soveltaa tai paivittaa: update ir export window mode ui."""
    try:
        try:
            app_mode = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            app_mode = "BASIC"
        is_basic = app_mode in ("BASIC", "AUTO")

        ft = str(_pin_get("filter_type", "") or "").strip().lower()
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

        cur = str(_pin_get("ir_export_window_mode", "auto") or "auto").strip().lower()

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

        if (not allow_ir) and cur != "auto":
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
            widget = put_select(
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
                widget.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            js_linear_only = bool((not is_linear) and (not is_asym_filter))
            js_disable = "true" if (lock_window_mode or js_linear_only) else "false"
            js_suffix = "(Linear only)" if js_linear_only else "(Locked)"
            put_html(
                f"""
<script>
(function() {{
  try {{
    var sel = document.querySelector('select[name="ir_export_window_mode"]');
    if(!sel) return;
    var opt = Array.from(sel.options).find(function(o) {{ return o.value === "rew_asym"; }});
    if(!opt) return;

    opt.disabled = {js_disable};

    var base = opt.textContent.replace(/\\s*\\((Linear only|Locked)\\)\\s*$/,'');
    if({js_disable}) opt.textContent = base + " {js_suffix}";
    else opt.textContent = base;

    if(opt.disabled && sel.value === "rew_asym" && {str(is_asym_filter).lower()} === false) {{
      sel.value = "auto";
    }}
  }} catch(e) {{}}
}})();
</script>
"""
            )
            try:
                msg = t("ir_asym_linear_only")
            except Exception:
                msg = ""

            emph = bool((not is_asym_filter) and (is_basic or (not is_linear)))
            opacity = "1.0" if emph else "0.55"
            color = "#ffb74d" if emph else "#9aa0a6"

            if str(msg or "").strip():
                put_html(
                    f"""
<div style="
  margin-top:6px;
  font-size:12.5px;
  font-weight:700;
  letter-spacing:0.4px;
  color:{color};
  opacity:{opacity};
">
  &#9888; {msg}<br>
  <span style="font-weight:500; letter-spacing:0;"></span>
</div>
"""
                )
    except Exception:
        pass


def update_ir_lr_window_ui(_=None):
    """Soveltaa tai paivittaa: update ir lr window ui."""
    try:
        app_mode = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
        is_auto_mode = bool(app_mode == "AUTO")
        mode = str(_pin_get("ir_export_window_mode", "auto") or "auto").strip().lower()
        is_rew_asym = mode == "rew_asym"

        ft = str(_pin_get("filter_type", "") or "").strip().lower()
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
            value_left = float(_pin_get("ir_window_left", 85.0) or 85.0)
        except Exception:
            value_left = 10.0

        try:
            value_right = _pin_get("ir_window_right", None)
            if value_right is None or value_right == "":
                value_right = _pin_get("ir_window", 500.0)
            value_right = float(value_right or 500.0)
        except Exception:
            value_right = 500.0

        with use_scope("ir_lr_window_scope", clear=True):
            w_left = put_input(
                "ir_window_left",
                label=t("ir_window_left_label"),
                type=FLOAT,
                value=value_left,
                help_text=t("ir_matala"),
            )
            w_right = put_input(
                "ir_window_right",
                label=t("ir_window_right_label"),
                type=FLOAT,
                value=value_right,
                help_text=t("ir_korkea"),
            )

            if (not enable_left) or is_auto_mode:
                w_left.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            if (not enable_right) or is_auto_mode:
                w_right.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")

            if ((not enable_left) and (not enable_right)) or is_auto_mode:
                try:
                    msg = t("ir_lr_window_hint")
                except Exception:
                    msg = ""
                if str(msg or "").strip():
                    put_html(
                        f"<div style='margin-top:6px; font-size:12.5px; color:#9aa0a6;'>{msg}</div>"
                    )
    except Exception:
        pass


def update_ir_window_shape_ui(_=None):
    """Soveltaa tai paivittaa: update ir window shape ui."""
    try:
        mode = str(_pin_get("ir_export_window_mode", "auto") or "auto").strip().lower()
        app_mode = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
        ft = str(_pin_get("filter_type", "") or "").strip().lower()
        try:
            ft_asym_label = str(t("ft_asymmetric") or "").strip().lower()
        except Exception:
            ft_asym_label = "asymmetric"
        is_asym_filter = (ft == ft_asym_label) or ("asym" in ft)
        if is_asym_filter and str(_pin_get("ir_export_window_shape", "hann") or "hann").strip().lower() != "tukey":
            try:
                pin_update("ir_export_window_shape", value="tukey")
            except Exception:
                pass
        is_basic = app_mode in ("BASIC", "AUTO")
        is_auto = (mode == "auto") or is_basic

        with use_scope("ir_export_window_shape_scope", clear=True):
            widget = put_select(
                "ir_export_window_shape",
                label=t("ir_export_window_shape"),
                options=[
                    {"label": t("ir_export_window_shape_hann"), "value": "hann"},
                    {"label": t("ir_export_window_shape_tukey"), "value": "tukey"},
                ],
                value=str(_pin_get("ir_export_window_shape", "hann") or "hann").strip().lower(),
                help_text=t("ir_export_window_shape_help"),
            )

            if is_auto or is_asym_filter:
                widget.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass


def update_mixed_freq_ui(_=None):
    """Soveltaa tai paivittaa: update mixed freq ui."""
    try:
        ft = str(_pin_get("filter_type", "") or "").strip().lower()

        try:
            ft_mixed_label = str(t("ft_mixed") or "").strip().lower()
        except Exception:
            ft_mixed_label = "mixed"

        is_mixed = (ft == ft_mixed_label) or ("mixed" in ft)

        with use_scope("update_mixed_freq_scope", clear=True):
            widget = put_input(
                "mixed_freq",
                label=t("mixed_freq"),
                type=FLOAT,
                value=float(_pin_get("mixed_freq", 180.0) or 180.0),
                help_text=t("mixed_freq_help"),
            )
            if not is_mixed:
                widget.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass


def update_bass_first_ui(*, pin, get_val, t):
    """Soveltaa tai paivittaa: bass_first_mode_max_hz input tila checkboxin mukaan."""

    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        enabled_raw = _p("bass_first_ai", None)
        if isinstance(enabled_raw, (list, tuple, set)):
            enabled = len(enabled_raw) > 0
        else:
            enabled = bool(enabled_raw) if enabled_raw is not None else bool(get_val("bass_first_ai", False))
    except Exception:
        enabled = bool(get_val("bass_first_ai", False))

    try:
        value = float(_pin_get("bass_first_mode_max_hz", get_val("bass_first_mode_max_hz", 200.0)) or 200.0)
    except Exception:
        value = float(get_val("bass_first_mode_max_hz", 200.0) or 200.0)

    with use_scope("bass_first_max_hz_scope", clear=True):
        widget = put_input(
            "bass_first_mode_max_hz",
            label=t("bass_first_max_hz_label"),
            type=FLOAT,
            value=value,
            help_text=t("bass_first_max_hz_help"),
        )
        if not enabled:
            widget.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")


def update_low_bass_cut_ui(*, pin, pin_update, get_val, t):
    """Soveltaa tai paivittaa: update low bass cut ui."""

    def _p(name, default=None):
        try:
            return pin[name]
        except Exception:
            return default

    try:
        enabled_raw = _p("low_bass_cut_enable", None)
        if isinstance(enabled_raw, (list, tuple, set)):
            enabled = len(enabled_raw) > 0
        else:
            enabled = bool(enabled_raw) if enabled_raw is not None else bool(get_val("low_bass_cut_enable", True))
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
                value0 = float(last or 40.0)
                pin_update("low_bass_cut_hz", value=value0)
                cur = value0
            disp = float(cur)
            if not math.isfinite(disp):
                disp = float(last or 40.0)
                pin_update("low_bass_cut_hz", value=disp)
            pin_update("low_bass_cut_hz_last", value=float(disp))
    except Exception:
        disp = 40.0

    with use_scope("low_bass_cut_scope", clear=True):
        widget = put_input(
            "low_bass_cut_hz",
            label=f"{t('low_bass_cut_hz')} (Hz)",
            type=FLOAT,
            value=float(disp),
            help_text=t("low_bass_cut_hz_help"),
        )

        if not enabled:
            widget.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
            put_html("<div style='margin-top:6px; color:#9aa0a6; font-size:13px;'>OFF</div>")


def update_afdw_cycles_ui(*, pin, get_val, t):
    """Soveltaa tai paivittaa: update afdw cycles ui."""
    try:
        app_mode = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
        is_auto_mode = bool(app_mode == "AUTO")
        enabled_raw = _pin_get("enable_afdw", None)
        if isinstance(enabled_raw, (list, tuple, set)):
            enabled = len(enabled_raw) > 0
        else:
            enabled = bool(enabled_raw) if enabled_raw is not None else bool(get_val("enable_afdw", True))
    except Exception:
        enabled = bool(get_val("enable_afdw", True))

    try:
        value = float(_pin_get("fdw_cycles", get_val("fdw_cycles", 10.0)) or 10.0)
    except Exception:
        value = float(get_val("fdw_cycles", 10.0) or 10.0)

    with use_scope("afdw_section_scope", clear=True):
        put_checkbox(
            "enable_afdw",
            options=[{"label": "\u23f3 Adaptive Frequency-Domain Windowing (A-FDW)", "value": True}],
            value=[True] if enabled else [],
            help_text=t("afdw_help"),
        )
        preset_labels = [
            t("afdw_preset_tight"),
            t("afdw_preset_balanced"),
            t("afdw_preset_safe"),
            t("afdw_preset_minimal"),
        ]
        if (not enabled) or is_auto_mode:
            _render_preset_badges(preset_labels)
        else:
            from .controls_mode import apply_afdw_preset

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
        widget = put_input(
            "fdw_cycles",
            label=t("fdw"),
            type=FLOAT,
            value=value,
            help_text=t("fdw_help"),
        )
        if (not enabled) or is_auto_mode:
            widget.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
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
                "var box=document.getElementById('afdw_section_scope');"
                "if(!box) return;"
                "box.style.opacity='0.55';"
                "box.style.filter='grayscale(1)';"
                "box.querySelectorAll('button,input,select,textarea').forEach(function(el){el.disabled=true;});"
                "})();"
                "</script>"
            )


def update_tdc_controls_ui(*, pin, get_val, t, apply_tdc_preset):
    """Soveltaa tai paivittaa: update tdc controls ui."""
    try:
        app_mode = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
        is_auto_mode = bool(app_mode == "AUTO")
        enabled_raw = _pin_get("enable_tdc", None)
        if isinstance(enabled_raw, (list, tuple, set)):
            enabled = len(enabled_raw) > 0
        else:
            enabled = bool(enabled_raw) if enabled_raw is not None else bool(get_val("enable_tdc", True))
    except Exception:
        enabled = bool(get_val("enable_tdc", True))

    def _f(name, default):
        try:
            value = _pin_get(name, get_val(name, default))
            return float(value if value is not None else default)
        except Exception:
            return float(default)

    strength = _f("tdc_strength", 50.0)
    max_red = _f("tdc_max_reduction_db", 9.0)
    slope = _f("tdc_slope_db_per_oct", 6.0)

    with use_scope("tdc_section_scope", clear=True):
        put_checkbox(
            "enable_tdc",
            options=[{"label": "\u23f3 Temporal Decay Control (TDC)", "value": True}],
            value=[True] if enabled else [],
            help_text=t("tdc_help"),
        )
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
                value=float(strength),
                help_text=t("tdc_help"),
            ),
            put_input(
                "tdc_max_reduction_db",
                label=t("tdc_max_reduction_db"),
                type=FLOAT,
                value=float(max_red),
                help_text=t("tdc_max_reduction_db_help"),
            ),
            put_input(
                "tdc_slope_db_per_oct",
                label=t("tdc_slope_db_per_oct"),
                type=FLOAT,
                value=float(slope),
                help_text=t("tdc_slope_db_per_oct_help"),
            ),
        ])

        if (not enabled) or is_auto_mode:
            put_html(
                "<script>"
                "(function(){"
                "var box=document.getElementById('tdc_section_scope');"
                "if(!box) return;"
                "box.style.opacity='0.55';"
                "box.style.filter='grayscale(1)';"
                "box.querySelectorAll('button,input,select,textarea').forEach(function(el){el.disabled=true;});"
                "})();"
                "</script>"
            )


def update_lvl_ui(_=None):
    try:
        try:
            app_mode = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            app_mode = "BASIC"
        is_basic = app_mode in ("BASIC", "AUTO")

        mode = str(_pin_get("lvl_mode", "Auto") or "Auto")
        if is_basic:
            mode = "Auto"
        is_manual = "Manual" in mode

        prev_mode = getattr(update_lvl_ui, "_last_lvl_mode", None)
        try:
            cur_min = float(_pin_get("lvl_min", 500.0) or 500.0)
            cur_max = float(_pin_get("lvl_max", 2000.0) or 2000.0)
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

        vmin = float(_pin_get("lvl_min", cur_min) or cur_min)
        vmax = float(_pin_get("lvl_max", cur_max) or cur_max)
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
            from .controls_target_preview import update_target_preview_ui

            try:
                cur = float(_pin_get("lvl_manual_db", 0.0) or 0.0)
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
                bias_hint = t("lvl_manual_bias_hint")
            except Exception:
                bias_hint = ""

            widget = put_row([
                put_input(
                    "lvl_manual_db",
                    label=t("lvl_target_db"),
                    type=FLOAT,
                    value=float(_pin_get("lvl_manual_db", 0.0) or 0.0),
                    help_text=t("lvl_manual_help"),
                ),
                put_button("+", onclick=lambda: _step_manual_target(-0.1), color="secondary").style("margin-top:28px; min-width:34px; margin-right:4px;"),
                put_button("-", onclick=lambda: _step_manual_target(+0.1), color="secondary").style("margin-top:28px; min-width:34px;"),
            ], size="1fr auto auto")
            put_html(
                f"<div style='opacity:0.75; font-size:12px; margin-top:4px;'>{bias_hint}</div>"
            )
            if not is_manual:
                widget.style("opacity:0.55; pointer-events:none; filter:grayscale(1);")
    except Exception:
        pass


__all__ = [
    "update_afdw_cycles_ui",
    "update_bass_first_ui",
    "update_ir_export_window_mode_ui",
    "update_ir_lr_window_ui",
    "update_ir_tukey_ui",
    "update_ir_window_shape_ui",
    "update_low_bass_cut_ui",
    "update_lvl_ui",
    "update_mixed_freq_ui",
    "update_tdc_controls_ui",
]
