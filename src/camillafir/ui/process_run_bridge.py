"""UI adapter for workflow/process_run_flow.py.

Workflow calls these callables instead of importing PyWebIO or UI modules directly.
"""
from __future__ import annotations

import time
import typing
from dataclasses import dataclass

from pywebio.output import put_processbar, set_processbar

from ..ui.camillafir_export import build_export_zip, save_export_bundle
from ..ui.camillafir_housecurve import load_house_curve
from ..ui.camillafir_ui import _render_results
from ..ui.system_health import (
    compute_health,
    toast_health_gate_result,
    toast_measurement_files_missing,
)


@dataclass(frozen=True)
class ProcessRunCallbacks:
    status: typing.Callable[[str], None]
    set_auto_selected_bar: typing.Callable[[typing.Any], None]


@dataclass(frozen=True)
class ProcessRunUiBridge:
    ensure_progress_bar: typing.Callable[[], None]
    set_progress: typing.Callable[[float], None]
    compute_health: typing.Callable[..., typing.Any]
    toast_health_gate_result: typing.Callable[..., bool]
    toast_measurement_files_missing: typing.Callable[[], None]
    load_house_curve: typing.Callable[..., tuple]
    render_results: typing.Callable[..., None]
    build_export_zip: typing.Callable[..., tuple]
    save_export_bundle: typing.Callable[..., tuple]
    make_callbacks: typing.Callable[[float], ProcessRunCallbacks]


def _default_ensure_progress_bar() -> None:
    try:
        set_processbar("bar", 0.0)
    except Exception:
        put_processbar("bar")
        set_processbar("bar", 0.0)


def _default_set_progress(v: float) -> None:
    set_processbar("bar", float(v))


def _default_make_callbacks(run_started_at: float) -> ProcessRunCallbacks:
    from ..ui.camillafir_ui import (
        update_auto_selected_bar as auto_selected_bar_cb,
        update_status as status_cb,
    )

    def _elapsed() -> float:
        try:
            return max(0.0, float(time.perf_counter() - run_started_at))
        except Exception:
            return 0.0

    def _status(msg: str) -> None:
        if callable(status_cb):
            try:
                status_cb(f"{msg} | {_elapsed():.1f} s")
            except Exception:
                pass

    def _set_auto_selected_bar(msg: typing.Any = "") -> None:
        if callable(auto_selected_bar_cb):
            try:
                auto_selected_bar_cb(msg)
            except Exception:
                pass

    return ProcessRunCallbacks(status=_status, set_auto_selected_bar=_set_auto_selected_bar)


def build_default_ui_bridge() -> ProcessRunUiBridge:
    return ProcessRunUiBridge(
        ensure_progress_bar=_default_ensure_progress_bar,
        set_progress=_default_set_progress,
        compute_health=compute_health,
        toast_health_gate_result=toast_health_gate_result,
        toast_measurement_files_missing=toast_measurement_files_missing,
        load_house_curve=load_house_curve,
        render_results=_render_results,
        build_export_zip=build_export_zip,
        save_export_bundle=save_export_bundle,
        make_callbacks=_default_make_callbacks,
    )
