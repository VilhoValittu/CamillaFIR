"""NiceGUI Crossover tab builder.

Replaces build_results_section() from layout_builders.py.
"""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl

_SLOPE_OPTS = [6, 12, 18, 24, 36, 48]


def build_xo_tab(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui

    ui.markdown(f"#### ✖ {t('tab_xo')}")
    ui.label(t("tab_xo_help")).classes("text-sm text-gray-400 -mt-2 mb-2")
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
