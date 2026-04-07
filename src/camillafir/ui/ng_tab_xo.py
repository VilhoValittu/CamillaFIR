"""NiceGUI Crossover tab builder.

Replaces build_results_section() from layout_builders.py.
"""
from __future__ import annotations

from typing import Callable

from ..config.camillafir_pipeline import filter_type_supports_xo_phase_model
from . import ng_controls as ctrl

_SLOPE_OPTS = [6, 12, 18, 24, 36, 48]


def build_xo_tab(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui

    mode_value = str(get_val("mode", "BASIC") or "BASIC").strip().upper()
    if bool(get_val("camillafir_automatic_mode", False)):
        mode_value = "AUTO"
    bass_integration_visible = bool(
        mode_value == "AUTO" and bool(get_val("bass_integration_enable", False))
    )
    xo_enabled = bool(filter_type_supports_xo_phase_model(get_val("filter_type", "Asymmetric")))

    with ui.column().classes("w-full gap-0") as xo_content:
        ui.markdown(f"#### ✖ {t('tab_xo')}")
        ui.label(t("tab_xo_help")).classes("text-sm text-gray-400 -mt-2 mb-2")
        with ui.column().classes("w-full gap-1") as bass_xo_scope:
            ui.label(t("bass_integration_xo_phase_model_info")).classes("text-xs text-gray-400")
            ui.label(t("bass_integration_xo_avr_separate_info")).classes("text-xs text-gray-400")
        ctrl.register_container("bass_integration_xo_info_scope", bass_xo_scope)
        bass_xo_scope.set_visibility(bass_integration_visible)
        ui.separator()

        for i in range(1, 6):
            with ui.row().classes("w-full gap-4 items-end"):
                ctrl.register(
                    f"xo{i}_f",
                    ui.number(
                        label=f"XO {i} Hz",
                        value=get_val(f"xo{i}_f", None),
                        format="%.1f",
                    ).props("dense outlined").classes("flex-1"),
                )
                ctrl.register(
                    f"xo{i}_s",
                    ui.select(
                        _SLOPE_OPTS,
                        value=get_val(f"xo{i}_s", 12),
                        label="dB/oct",
                    ).props("dense outlined").classes("w-32"),
                )
    ctrl.register_container("xo_tab_content_scope", xo_content)
    xo_content.set_visibility(xo_enabled)
