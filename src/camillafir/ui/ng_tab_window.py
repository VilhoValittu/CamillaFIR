"""NiceGUI IR Window / TDC / A-FDW tab builder.

Replaces build_export_section() + controls_ir_window.py dynamic rendering.

All controls are created upfront; ng_mode_controls handles show/hide.
"""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl


def build_window_tab(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui

    ui.label(f"🪟 {t('tab_window_tdc')}").classes("text-lg font-semibold")
    ui.separator()

    # ---- IR Export Window ----
    ui.label("🪟 IR Export Window").classes("text-sm font-semibold")

    ctrl.register(
        "ir_export_window_mode",
        ui.select(
            options={
                "auto":    t("ir_export_window_auto"),
                "rew_asym": t("ir_export_window_asym"),
            },
            value=str(get_val("ir_export_window_mode", "auto") or "auto").strip().lower(),
            label=t("ir_export_window_mode"),
        ).props("dense outlined").classes("w-full"),
    )
    ctrl.register_container("ir_export_window_mode_scope", ui.column().classes("w-full"))

    with ui.row().classes("w-full gap-4"):
        ctrl.register(
            "ir_export_window_shape",
            ui.select(
                options={
                    "hann":  t("ir_export_window_shape_hann"),
                    "tukey": t("ir_export_window_shape_tukey"),
                },
                value=str(get_val("ir_export_window_shape", "hann") or "hann").strip().lower(),
                label=t("ir_export_window_shape"),
            ).props("dense outlined").classes("flex-1"),
        )

        # Tukey alpha (shown only when shape=tukey)
        tukey_col = ui.column().classes("flex-1")
        ctrl.register_container("ir_tukey_alpha_scope", tukey_col)
        with tukey_col:
            ctrl.register(
                "ir_export_tukey_alpha",
                ui.number(
                    label=t("ir_export_tukey_alpha"),
                    value=float(get_val("ir_export_tukey_alpha", 0.25) or 0.25),
                    format="%.3f",
                    min=0.0,
                    max=1.0,
                ).props("dense outlined").classes("w-full"),
            )
        tukey_col.set_visibility(False)

    # L/R window lengths (shown for linear/asymmetric + manual mode)
    lr_col = ui.column().classes("w-full gap-2")
    ctrl.register_container("ir_lr_window_scope", lr_col)
    with lr_col:
        with ui.row().classes("w-full gap-4"):
            ctrl.register(
                "ir_window_left",
                ui.number(
                    label=t("ir_window_left_label"),
                    value=float(get_val("ir_window_left", 85.0) or 85.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )
            _ir_right_def = get_val("ir_window_right", None) or get_val("ir_window", 500.0)
            ctrl.register(
                "ir_window_right",
                ui.number(
                    label=t("ir_window_right_label"),
                    value=float(_ir_right_def or 500.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )
            ctrl.register(
                "ir_window",
                ctrl._ValueHolder(float(get_val("ir_window", 500.0) or 500.0)),
            )
    lr_col.set_visibility(False)

    with ui.expansion(t("ir_export_window_help_long_title")).classes("w-full"):
        ui.markdown(t("ir_export_window_help_long"))

    ui.separator()

    # ---- A-FDW ----
    ui.label("⏳ Temporal Processing").classes("text-sm font-semibold")

    afdw_col = ui.column().classes("w-full gap-1")
    ctrl.register_container("afdw_section_scope", afdw_col)
    with afdw_col:
        ctrl.register(
            "enable_afdw",
            ui.checkbox(
                "⏳ Adaptive Frequency-Domain Windowing (A-FDW)",
                value=bool(get_val("enable_afdw", True)),
            ),
        )

        # Preset buttons
        with ui.row().classes("gap-2 flex-wrap") as afdw_presets_row:
            ctrl.register_container("afdw_presets_row", afdw_presets_row)
            for preset_key, preset_name in [
                ("afdw_preset_tight", "Tight"),
                ("afdw_preset_balanced", "Balanced"),
                ("afdw_preset_safe", "Safe"),
                ("afdw_preset_minimal", "Minimal"),
            ]:
                label = t(preset_key)
                ui.button(
                    label,
                    on_click=lambda n=preset_name: _apply_afdw_preset(n),
                ).props('size="sm" outline')

        ctrl.register(
            "fdw_cycles",
            ui.number(
                label=t("fdw"),
                value=float(get_val("fdw_cycles", 10.0) or 10.0),
                format="%.1f",
            ).props("dense outlined").classes("w-full"),
        )

    ui.separator()

    # ---- TDC ----
    tdc_col = ui.column().classes("w-full gap-1")
    ctrl.register_container("tdc_section_scope", tdc_col)
    with tdc_col:
        ctrl.register(
            "enable_tdc",
            ui.checkbox(
                "⏳ Temporal Decay Control (TDC)",
                value=bool(get_val("enable_tdc", True)),
            ),
        )

        # Preset buttons
        with ui.row().classes("gap-2 flex-wrap"):
            for preset_key, preset_name in [
                ("tdc_preset_safe", "Safe"),
                ("tdc_preset_normal", "Normal"),
                ("tdc_preset_aggressive", "Aggressive"),
            ]:
                label = t(preset_key)
                ui.button(
                    label,
                    on_click=lambda n=preset_name: _apply_tdc_preset(n),
                ).props('size="sm" outline')

        with ui.row().classes("w-full gap-4"):
            ctrl.register(
                "tdc_strength",
                ui.number(
                    label=t("tdc_strength"),
                    value=float(get_val("tdc_strength", 50.0) or 50.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )
            ctrl.register(
                "tdc_max_reduction_db",
                ui.number(
                    label=t("tdc_max_reduction_db"),
                    value=float(get_val("tdc_max_reduction_db", 9.0) or 9.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )
            ctrl.register(
                "tdc_slope_db_per_oct",
                ui.number(
                    label=t("tdc_slope_db_per_oct"),
                    value=float(get_val("tdc_slope_db_per_oct", 6.0) or 6.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )


def _apply_tdc_preset(name: str) -> None:
    from .ng_health import show_toast  # noqa: PLC0415

    _PRESETS = {
        "Safe":       {"enable": True, "strength": 35.0, "max_red": 6.0,  "slope": 3.0},
        "Normal":     {"enable": True, "strength": 50.0, "max_red": 9.0,  "slope": 6.0},
        "Aggressive": {"enable": True, "strength": 70.0, "max_red": 12.0, "slope": 0.0},
    }
    p = _PRESETS.get(name)
    if p is None:
        return
    mode = str(ctrl.value("mode", "BASIC") or "BASIC").upper()
    if mode == "AUTO":
        show_toast("TDC preset locked in AUTO mode", color="info", duration=1.8)
        return
    ctrl.set_value("enable_tdc", p["enable"])
    ctrl.set_value("tdc_strength", p["strength"])
    ctrl.set_value("tdc_max_reduction_db", p["max_red"])
    ctrl.set_value("tdc_slope_db_per_oct", p["slope"])
    show_toast(f"TDC preset applied: {name}", color="success", duration=1.5)


def _apply_afdw_preset(name: str) -> None:
    from .ng_health import show_toast  # noqa: PLC0415

    _PRESETS = {
        "Tight":    {"enable": True, "cycles": 4.0},
        "Balanced": {"enable": True, "cycles": 10.0},
        "Safe":     {"enable": True, "cycles": 18.0},
        "Minimal":  {"enable": True, "cycles": 30.0},
    }
    p = _PRESETS.get(name)
    if p is None:
        return
    mode = str(ctrl.value("mode", "BASIC") or "BASIC").upper()
    if mode == "AUTO":
        show_toast("A-FDW preset locked in AUTO mode", color="info", duration=1.8)
        return
    ctrl.set_value("enable_afdw", p["enable"])
    ctrl.set_value("fdw_cycles", p["cycles"])
    show_toast(f"A-FDW preset applied: {name}", color="success", duration=1.5)
