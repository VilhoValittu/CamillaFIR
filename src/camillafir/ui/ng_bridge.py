"""NiceGUI UI adapter for workflow/process_run_flow.py.

Drop-in replacement for process_run_bridge.py.  The dataclass interfaces
(ProcessRunCallbacks, ProcessRunUiBridge) are identical so workflow code needs
no changes when switching to the NiceGUI bridge.

Progress element integration
-----------------------------
The NiceGUI progress bar element is created in ng_run_section.py as part of
the page layout.  Call set_progress_element_getter() once after building the
layout to wire it in:

    import ng_bridge
    ng_bridge.set_progress_element_getter(lambda: _progress_element)
"""
from __future__ import annotations

import time
import typing
from dataclasses import dataclass

from . import ui_state


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


# ---------------------------------------------------------------------------
# Progress element holder
# ---------------------------------------------------------------------------

_progress_getter: typing.Callable[[], typing.Any] | None = None


def set_progress_element_getter(fn: typing.Callable[[], typing.Any]) -> None:
    """Register a callable that returns the live NiceGUI LinearProgress element."""
    global _progress_getter
    _progress_getter = fn if callable(fn) else None


def _default_ensure_progress_bar() -> None:
    el = _progress_getter() if callable(_progress_getter) else None
    if el is not None:
        try:
            el.set_value(0.0)
        except Exception:
            pass


def _default_set_progress(v: float) -> None:
    el = _progress_getter() if callable(_progress_getter) else None
    if el is not None:
        try:
            el.set_value(float(v))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------------

def _default_make_callbacks(run_started_at: float) -> ProcessRunCallbacks:
    def _elapsed() -> float:
        try:
            return max(0.0, float(time.perf_counter() - run_started_at))
        except Exception:
            return 0.0

    def _status(msg: str) -> None:
        try:
            ui_state.update_status(f"{msg} | {_elapsed():.1f} s")
        except Exception:
            pass

    def _set_auto_selected_bar(msg: typing.Any = "") -> None:
        try:
            ui_state.update_auto_selected_bar(msg)
        except Exception:
            pass

    return ProcessRunCallbacks(status=_status, set_auto_selected_bar=_set_auto_selected_bar)


# ---------------------------------------------------------------------------
# Lazy render_results – ng_results_sections.py is created in Phase 4
# ---------------------------------------------------------------------------

def _lazy_render_results(*args: typing.Any, **kwargs: typing.Any) -> None:
    from .ng_results_sections import render_results  # noqa: PLC0415
    render_results(*args, **kwargs)


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------

def build_default_ui_bridge() -> ProcessRunUiBridge:
    from .ng_health import (  # noqa: PLC0415
        compute_health,
        toast_health_gate_result,
        toast_measurement_files_missing,
    )
    from ..ui.camillafir_housecurve import load_house_curve  # noqa: PLC0415
    from ..ui.camillafir_export import build_export_zip, save_export_bundle  # noqa: PLC0415

    return ProcessRunUiBridge(
        ensure_progress_bar=_default_ensure_progress_bar,
        set_progress=_default_set_progress,
        compute_health=compute_health,
        toast_health_gate_result=toast_health_gate_result,
        toast_measurement_files_missing=toast_measurement_files_missing,
        load_house_curve=load_house_curve,
        render_results=_lazy_render_results,
        build_export_zip=build_export_zip,
        save_export_bundle=save_export_bundle,
        make_callbacks=_default_make_callbacks,
    )
