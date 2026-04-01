"""NiceGUI Basic tab builder.

Replaces build_filter_section() from layout_builders.py.
"""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl

_FS_OPTS = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]
_TAPS_OPTS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]


def _normalize_filter_type_value(value) -> str:
    raw = str(value or "").strip().lower()
    if "asym" in raw or "low-latency" in raw or "causal" in raw:
        return "Asymmetric"
    if "mix" in raw:
        return "Mixed"
    if "min" in raw:
        return "Minimum"
    if "lin" in raw:
        return "Linear"
    return "Asymmetric"


def build_basic_tab(*, t: Callable, get_val: Callable, max_safe_boost: float) -> None:
    from nicegui import ui

    mode_value = str(get_val("mode", "AUTO") or "AUTO").strip().upper()
    if mode_value not in ("BASIC", "ADVANCED", "AUTO"):
        mode_value = "AUTO"
    if bool(get_val("camillafir_automatic_mode", False)):
        mode_value = "AUTO"

    auto_goal_value = str(get_val("auto_goal", "balanced") or "balanced").strip().lower()
    if auto_goal_value not in ("room-safe", "balanced", "low-ripple", "flat", "subwoofers"):
        auto_goal_value = "balanced"

    auto_target_mode_value = str(get_val("auto_target_mode", "auto") or "auto").strip().lower()
    if auto_target_mode_value not in ("auto", "adaptive", "selected"):
        auto_target_mode_value = "auto"

    ui.markdown(f"#### {t('tab_basic')}")
    ui.separator()

    # Mode row
    with ui.row().classes("w-full gap-4 items-end"):
        ctrl.register(
            "mode",
            ui.select(
                options={
                    "AUTO":     t("mode_auto_label"),
                    "BASIC":    t("mode_basic_label"),
                    "ADVANCED": t("mode_advanced_label"),
                },
                value=mode_value,
                label=t("mode_label"),
            ).props("dense outlined").classes("flex-1"),
        )
        ui.button(
            t("mode_apply_defaults_btn"),
            on_click=lambda: _apply_mode_defaults(t=t, get_val=get_val),
        ).props('color="secondary" outline')

    ui.label(t("mode_apply_defaults_help")).classes("text-xs text-gray-400 italic")

    # Mode description container
    mode_desc = ui.column().classes("w-full")
    ctrl.register_container("mode_desc_scope", mode_desc)

    ui.separator()

    # AUTO-mode options
    auto_mode_col = ui.column().classes("w-full gap-4")
    ctrl.register_container("auto_mode_scope", auto_mode_col)
    with auto_mode_col:
        ctrl.register(
            "auto_goal",
            ui.select(
                options={
                    "balanced":   t("auto_goal_balanced"),
                    "room-safe":  t("auto_goal_room_safe"),
                    "subwoofers": t("auto_goal_subwoofers"),
                    "low-ripple": t("auto_goal_low_ripple"),
                    "flat":       t("auto_goal_flat"),
                },
                value=auto_goal_value,
                label=t("auto_goal_label"),
            ).props("dense outlined").classes("w-full"),
        )
        ctrl.register(
            "auto_target_mode",
            ui.select(
                options={
                    "auto":     t("auto_target_mode_auto"),
                    "adaptive": t("auto_target_mode_adaptive"),
                    "selected": t("auto_target_mode_selected"),
                },
                value=auto_target_mode_value,
                label=t("auto_target_mode_label"),
            ).props("dense outlined").classes("w-full"),
        )
    auto_mode_col.set_visibility(mode_value == "AUTO")

    ui.separator()
    ui.markdown(f"#### 🧱 {t('ui_fir_engine')}")

    # Sampling rate + taps
    with ui.row().classes("w-full gap-4"):
        ctrl.register(
            "fs",
            ui.select(
                _FS_OPTS,
                value=get_val("fs", 44100),
                label=t("fs"),
            ).props("dense outlined").classes("flex-1"),
        )
        ctrl.register(
            "taps",
            ui.select(
                _TAPS_OPTS,
                value=get_val("taps", 65536),
                label=t("taps"),
            ).props("dense outlined").classes("flex-1"),
        )

    # Engine metrics label (latency + resolution)
    ctrl.register("engine_metrics_label", ui.label("").classes("text-xs text-gray-400"))

    # Multi-rate info container
    ctrl.register_container("taps_auto_info_scope_basic", ui.column().classes("w-full"))

    ui.separator()
    ui.markdown(f"#### 🧭 {t('ui_filter_type')}")

    with ui.row().classes("w-full gap-4 items-start"):
        ftype_opts = {
            "Linear": t("ft_linear"),
            "Minimum": t("ft_min"),
            "Mixed": t("ft_mixed"),
            "Asymmetric": t("ft_asymmetric"),
        }
        ctrl.register(
            "filter_type",
            ui.select(
                ftype_opts,
                value=_normalize_filter_type_value(get_val("filter_type", "Asymmetric")),
                label=t("filter_type"),
            ).props("dense outlined").classes("flex-1"),
        )
        # Mixed-phase split frequency container (shown when filter_type=mixed)
        mixed_scope = ui.column().classes("flex-1")
        ctrl.register_container("update_mixed_freq_scope", mixed_scope)
        with mixed_scope:
            ctrl.register(
                "mixed_freq",
                ui.number(
                    label=t("mixed_split_hz_label"),
                    value=get_val("mixed_freq", 200.0),
                    format="%.1f",
                ).props("dense outlined").classes("w-full"),
            )
        # Will be hidden/shown by mode callback based on filter_type


def _apply_mode_defaults(*, t: Callable, get_val: Callable) -> None:
    """Apply mode defaults when user clicks the button."""
    try:
        from ..config.mode_policy import MODE_DEFAULTS  # noqa: PLC0415
        from .ng_health import toast_mode_defaults_applied  # noqa: PLC0415

        mode = str(ctrl.value("mode", "BASIC") or "BASIC").upper()
        defaults = dict(MODE_DEFAULTS.get(mode, {}))

        for cfg_key, ui_key in _MODE_DEFAULTS_KEY_MAP.items():
            v = defaults.get(cfg_key, None)
            if v is not None:
                ctrl.set_value(ui_key, v)

        toast_mode_defaults_applied(mode)
    except Exception:
        import logging
        logging.getLogger("CamillaFIR").debug("_apply_mode_defaults failed", exc_info=True)


# Map mode-defaults config keys → ng_controls element names
_MODE_DEFAULTS_KEY_MAP: dict[str, str] = {
    "mag_c_min": "mag_c_min",
    "mag_c_max": "mag_c_max",
    "max_boost_db": "max_boost",
    "max_cut_db": "max_cut_db",
    "max_slope_db_per_oct": "max_slope_db_per_oct",
    "phase_limit": "phase_limit",
    "reg_strength": "reg_strength",
    "filter_smooth": "filter_smooth",
    "bass_first_ai": "bass_first_ai",
    "stereo_link": "stereo_link",
    "exc_prot": "exc_prot",
    "hpf_enable": "hpf_enable",
    "low_bass_cut_enable": "low_bass_cut_enable",
    "enable_tdc": "enable_tdc",
    "enable_afdw": "enable_afdw",
}
