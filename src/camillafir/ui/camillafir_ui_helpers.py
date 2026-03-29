import logging
import math

from pywebio.output import put_collapse, put_html, put_markdown
from pywebio.pin import pin

from ..resources.i8n.camillafir_i18n import t
from ..config.mode_policy import MODE_CLAMPS, MODE_DEFAULTS
from .controls_ir_window import (
    update_afdw_cycles_ui,
    update_ir_export_window_mode_ui,
    update_ir_lr_window_ui,
    update_ir_tukey_ui,
    update_ir_window_shape_ui,
    update_low_bass_cut_ui,
    update_lvl_ui,
    update_mixed_freq_ui,
    update_tdc_controls_ui,
)
from .controls_mode import (
    apply_afdw_preset,
    apply_mode_defaults_to_ui,
    apply_tdc_preset,
    update_auto_mode_controls_ui,
    update_basic_clamp_hints_ui,
    update_confidence_pull_ui,
    update_mode_desc,
    update_taps_auto_info,
    update_unsafe_raw_dsp_ui,
)
from .controls_target_preview import update_target_preview_ui
from .system_health import (
    toast_max_boost_over_cap,
    toast_taps_over_cap,
)

logger = logging.getLogger("CamillaFIR")


def _pin_get(name, default=None):
    """Lukee pin-arvon turvallisesti eri PyWebIO-konteksteissa."""
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
    """Palauttaa tosi kun automaattitila lukitsee kontrollit."""
    try:
        mode_u = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
    except (TypeError, ValueError):
        mode_u = "BASIC"
    try:
        auto_flag = bool(_pin_get("camillafir_automatic_mode", False))
    except (TypeError, ValueError):
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
        # Decorative guide UI only; malformed labels must not block rendering the rest of the page.
        logger.debug("Preset badge rendering failed", exc_info=True)
        return


def _warn_max_boost_if_over_cap(_=None):
    """Sisainen apufunktio: warn max boost if over cap."""
    try:
        value = _pin_get("max_boost", None)
        cap = float(globals().get("MAX_SAFE_BOOST", 8.0) or 8.0)
        if value is None or value == "":
            toast_max_boost_over_cap(None, cap)
            return
        value = float(value)
        if not math.isfinite(value):
            toast_max_boost_over_cap(None, cap)
            return
        toast_max_boost_over_cap(value, cap)
    except Exception as exc:
        try:
            logger.warning("max_boost toast failed: %s", exc)
        except Exception:
            pass


def _warn_taps_if_over_cap(_=None):
    """Sisainen apufunktio: warn taps if over cap."""
    try:
        value = _pin_get("taps", None)
        cap = int(globals().get("MAX_SAFE_TAPS", 131072) or 131072)
        if value is None or value == "":
            toast_taps_over_cap(None, cap)
            return
        value = int(value)
        toast_taps_over_cap(value, cap)
    except Exception as exc:
        try:
            logger.warning("taps warning toast failed: %s", exc)
        except Exception:
            pass


def _pretty_plot_smoothing(v, t):
    if isinstance(v, str) and v.strip().lower() == "psychoacoustic":
        return t("smooth_safe_reference")
    return str(v)


def _fmt_mode_value(key: str, defaults: dict, clamps: dict):
    """Sisainen apufunktio: fmt mode value."""
    value = defaults.get(key, None)
    if key == "plot_smoothing_level":
        value_str = _pretty_plot_smoothing(value, t)
    else:
        value_str = str(value)

    limit = clamps.get(key, None) if isinstance(clamps, dict) else None
    if isinstance(limit, tuple) and len(limit) == 2:
        lo, hi = limit
        if isinstance(lo, bool) and isinstance(hi, bool):
            return f"**{value_str}**"
        return f"**{value_str}** _(clamped to {lo}-{hi})_"
    return f"**{value_str}**"


def _build_modes_guide_parts(t):
    """Sisainen apufunktio: build modes guide parts."""
    d_basic = MODE_DEFAULTS.get("BASIC", {})
    c_basic = MODE_CLAMPS.get("BASIC", {})
    d_adv = MODE_DEFAULTS.get("ADVANCED", {})
    c_adv = MODE_CLAMPS.get("ADVANCED", {})

    def clamp_suffix(key, clamps):
        limit = clamps.get(key)
        if isinstance(limit, tuple) and len(limit) == 2:
            lo, hi = limit
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
        content.append(put_collapse(title, [put_markdown(body)], open=False))

    put_collapse(t("guide_section_title"), content, open=False)


__all__ = [
    "_is_auto_mode_locked",
    "_pin_get",
    "_render_preset_badges",
    "_warn_max_boost_if_over_cap",
    "_warn_taps_if_over_cap",
    "apply_afdw_preset",
    "apply_mode_defaults_to_ui",
    "apply_tdc_preset",
    "put_guide_section",
    "update_afdw_cycles_ui",
    "update_auto_mode_controls_ui",
    "update_basic_clamp_hints_ui",
    "update_confidence_pull_ui",
    "update_ir_export_window_mode_ui",
    "update_ir_lr_window_ui",
    "update_ir_tukey_ui",
    "update_ir_window_shape_ui",
    "update_low_bass_cut_ui",
    "update_lvl_ui",
    "update_mixed_freq_ui",
    "update_mode_desc",
    "update_taps_auto_info",
    "update_target_preview_ui",
    "update_tdc_controls_ui",
    "update_unsafe_raw_dsp_ui",
]
