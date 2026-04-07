"""NiceGUI toast backend for the application health service.

Patches the application health service to return a NiceGUI `ui.notify`
callable, then re-exports the public API for UI callers.

Toasts may be triggered from background threads (DSP pipeline runs in a
worker thread). NiceGUI's `ui.notify` uses `asyncio.current_task()` to
locate the client slot stack, which is unavailable in a plain thread.

To bridge the gap: `_notify` pushes pending notifications into a
thread-safe deque. A `ui.timer` (installed once per page load) drains the
deque from the UI event loop where the client context is valid.
"""
from __future__ import annotations

import collections
import threading
from typing import NamedTuple

from ..application import health_service as _sh


class _PendingToast(NamedTuple):
    msg: str
    duration: float
    color: str | None


_pending: collections.deque[_PendingToast] = collections.deque()
_pending_lock = threading.Lock()


def _ng_toast_callable():
    _COLOR_MAP = {
        "success": "positive",
        "warn": "warning",
        "warning": "warning",
        "error": "negative",
        "info": "info",
    }

    def _notify(msg: str, *, duration: float = 5.0, color: str | None = None) -> None:
        with _pending_lock:
            _pending.append(_PendingToast(msg=str(msg), duration=float(duration), color=color))

    return _notify


def _drain_pending_toasts() -> None:
    """Called by ui.timer from the UI thread — drains the pending queue."""
    from nicegui import ui  # noqa: PLC0415

    _COLOR_MAP = {
        "success": "positive",
        "warn": "warning",
        "warning": "warning",
        "error": "negative",
        "info": "info",
    }

    with _pending_lock:
        items = list(_pending)
        _pending.clear()

    for item in items:
        ng_type = _COLOR_MAP.get(str(item.color or ""), "info")
        timeout = max(1000, int(item.duration * 1000))
        try:
            ui.notify(item.msg, type=ng_type, timeout=timeout, position="top")
        except Exception:
            pass


def install_toast_timer() -> None:
    """Install the drain timer. Call once after the page UI is built."""
    from nicegui import ui  # noqa: PLC0415

    ui.timer(0.25, _drain_pending_toasts)


# Patch the application health service to use the queued notify backend.
_sh._get_toast_callable = _ng_toast_callable  # type: ignore[attr-defined]

from ..application.health_service import (  # noqa: E402
    HealthResult,
    Issue,
    Level,
    compute_health,
    format_health_summary,
    show_toast,
    toast_afdw_preset_applied,
    toast_health_gate_result,
    toast_max_boost_over_cap,
    toast_mode_defaults_applied,
    toast_taps_over_cap,
    toast_tdc_preset_applied,
)

__all__ = [
    "HealthResult",
    "Issue",
    "Level",
    "compute_health",
    "format_health_summary",
    "install_toast_timer",
    "show_toast",
    "toast_afdw_preset_applied",
    "toast_health_gate_result",
    "toast_max_boost_over_cap",
    "toast_mode_defaults_applied",
    "toast_taps_over_cap",
    "toast_tdc_preset_applied",
]
