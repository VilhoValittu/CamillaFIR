"""NiceGUI IR Window / TDC / A-FDW tab builder.

Replaces build_export_section() + controls_ir_window.py dynamic rendering.

All controls are created upfront; ng_mode_controls handles show/hide.
"""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl
from ..ui_i18n import (
    AFDW_PRESET_BALANCED,
    AFDW_PRESET_LABEL_KEYS,
    AFDW_PRESET_MINIMAL,
    AFDW_PRESET_SAFE,
    AFDW_PRESET_TIGHT,
    TDC_PRESET_AGGRESSIVE,
    TDC_PRESET_LABEL_KEYS,
    TDC_PRESET_NORMAL,
    TDC_PRESET_SAFE,
    normalize_afdw_preset_key,
    normalize_tdc_preset_key,
)


def build_window_tab(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui

    ui.label(f"{t('tab_window_tdc')}").classes("text-lg font-semibold")
    ui.separator()

    # ---- IR Export Window ----
    ui.label(t("ui_ir_export_window_title")).classes("text-sm font-semibold")

    ctrl.register(
        "ir_export_window_mode",
        ui.select(
            options={
                "auto": t("ir_export_window_auto"),
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
                    "hann": t("ir_export_window_shape_hann"),
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
    ui.label(t("ui_temporal_processing_title")).classes("text-sm font-semibold")

    afdw_col = ui.column().classes("w-full gap-1")
    ctrl.register_container("afdw_section_scope", afdw_col)
    with afdw_col:
        ctrl.register(
            "enable_afdw",
            ui.checkbox(
                t("enable_afdw"),
                value=bool(get_val("enable_afdw", True)),
            ),
        )
        afdw_details_col = ui.column().classes("w-full gap-1")
        ctrl.register_container("afdw_details_scope", afdw_details_col)
        with afdw_details_col:
            with ui.row().classes("gap-2 flex-wrap") as afdw_presets_row:
                ctrl.register_container("afdw_presets_row", afdw_presets_row)
                for preset_key in (
                    AFDW_PRESET_TIGHT,
                    AFDW_PRESET_BALANCED,
                    AFDW_PRESET_SAFE,
                    AFDW_PRESET_MINIMAL,
                ):
                    ui.button(
                        t(AFDW_PRESET_LABEL_KEYS[preset_key]),
                        on_click=lambda key=preset_key: _apply_afdw_preset(key, t=t),
                    ).props('size="sm" outline')

            ctrl.register(
                "fdw_cycles",
                ui.number(
                    label=t("fdw"),
                    value=float(get_val("fdw_cycles", 10.0) or 10.0),
                    format="%.1f",
                ).props("dense outlined").classes("w-full"),
            )
        afdw_details_col.set_visibility(bool(get_val("enable_afdw", True)))

    ui.separator()

    # ---- TDC ----
    tdc_col = ui.column().classes("w-full gap-1")
    ctrl.register_container("tdc_section_scope", tdc_col)
    with tdc_col:
        ctrl.register(
            "enable_tdc",
            ui.checkbox(
                t("enable_tdc"),
                value=bool(get_val("enable_tdc", True)),
            ),
        )
        tdc_details_col = ui.column().classes("w-full gap-1")
        ctrl.register_container("tdc_details_scope", tdc_details_col)
        with tdc_details_col:
            with ui.row().classes("gap-2 flex-wrap"):
                for preset_key in (
                    TDC_PRESET_SAFE,
                    TDC_PRESET_NORMAL,
                    TDC_PRESET_AGGRESSIVE,
                ):
                    ui.button(
                        t(TDC_PRESET_LABEL_KEYS[preset_key]),
                        on_click=lambda key=preset_key: _apply_tdc_preset(key, t=t),
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
        tdc_details_col.set_visibility(bool(get_val("enable_tdc", True)))


def _apply_tdc_preset(preset_key: str, t: Callable | None = None) -> None:
    from .ng_health import show_toast  # noqa: PLC0415
    from ..resources.i8n.camillafir_i18n import t as _t  # noqa: PLC0415

    if t is None:
        t = _t

    preset_key = normalize_tdc_preset_key(preset_key, t=t)
    presets = {
        TDC_PRESET_SAFE: {"enable": True, "strength": 35.0, "max_red": 6.0, "slope": 3.0},
        TDC_PRESET_NORMAL: {"enable": True, "strength": 50.0, "max_red": 9.0, "slope": 6.0},
        TDC_PRESET_AGGRESSIVE: {"enable": True, "strength": 70.0, "max_red": 12.0, "slope": 0.0},
    }
    preset = presets.get(preset_key or "")
    if preset is None or preset_key is None:
        return

    mode = str(ctrl.value("mode", "BASIC") or "BASIC").upper()
    if mode == "AUTO":
        show_toast(t("toast_tdc_preset_locked_auto"), color="info", duration=1.8)
        return

    ctrl.set_value("enable_tdc", preset["enable"])
    ctrl.set_value("tdc_strength", preset["strength"])
    ctrl.set_value("tdc_max_reduction_db", preset["max_red"])
    ctrl.set_value("tdc_slope_db_per_oct", preset["slope"])
    show_toast(
        t("toast_tdc_preset_applied").format(name=t(TDC_PRESET_LABEL_KEYS[preset_key])),
        color="success",
        duration=1.5,
    )


def _apply_afdw_preset(preset_key: str, t: Callable | None = None) -> None:
    from .ng_health import show_toast  # noqa: PLC0415
    from ..resources.i8n.camillafir_i18n import t as _t  # noqa: PLC0415

    if t is None:
        t = _t

    preset_key = normalize_afdw_preset_key(preset_key, t=t)
    presets = {
        AFDW_PRESET_TIGHT: {"enable": True, "cycles": 4.0},
        AFDW_PRESET_BALANCED: {"enable": True, "cycles": 10.0},
        AFDW_PRESET_SAFE: {"enable": True, "cycles": 18.0},
        AFDW_PRESET_MINIMAL: {"enable": True, "cycles": 30.0},
    }
    preset = presets.get(preset_key or "")
    if preset is None or preset_key is None:
        return

    mode = str(ctrl.value("mode", "BASIC") or "BASIC").upper()
    if mode == "AUTO":
        show_toast(t("toast_afdw_preset_locked_auto"), color="info", duration=1.8)
        return

    ctrl.set_value("enable_afdw", preset["enable"])
    ctrl.set_value("fdw_cycles", preset["cycles"])
    show_toast(
        t("toast_afdw_preset_applied").format(name=t(AFDW_PRESET_LABEL_KEYS[preset_key])),
        color="success",
        duration=1.5,
    )
