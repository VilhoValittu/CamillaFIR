"""NiceGUI toast backend for the application health service.

Patches the application health service to return a NiceGUI `ui.notify`
callable, then re-exports the public API for UI callers.
"""
from __future__ import annotations

from ..application import health_service as _sh


def _ng_toast_callable():
    from nicegui import ui  # noqa: PLC0415

    _COLOR_MAP = {
        "success": "positive",
        "warn": "warning",
        "warning": "warning",
        "error": "negative",
        "info": "info",
    }

    def _notify(msg: str, *, duration: float = 5.0, color: str | None = None) -> None:
        ng_type = _COLOR_MAP.get(str(color or ""), "info")
        timeout = max(1000, int(float(duration) * 1000))
        ui.notify(msg, type=ng_type, timeout=timeout, position="top")

    return _notify


# Patch the application health service to use NiceGUI toast notifications.
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
    toast_measurement_files_missing,
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
    "show_toast",
    "toast_afdw_preset_applied",
    "toast_health_gate_result",
    "toast_max_boost_over_cap",
    "toast_measurement_files_missing",
    "toast_mode_defaults_applied",
    "toast_taps_over_cap",
    "toast_tdc_preset_applied",
]
