"""NiceGUI toast backend for system_health.

Patches system_health._get_toast_callable to return a NiceGUI ui.notify wrapper,
then re-exports all public symbols so callers can import from here instead of
system_health directly.

Usage:
    from ..ui.ng_health import (
        compute_health, toast_health_gate_result, toast_measurement_files_missing, ...
    )
"""
from __future__ import annotations

from . import system_health as _sh


def _ng_toast_callable():
    from nicegui import ui  # noqa: PLC0415 – deferred to avoid import at module level

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


# Patch system_health to use NiceGUI toast instead of PyWebIO toast.
# _get_toast_callable() is called lazily inside show_toast(), so this
# override takes effect for every subsequent call.
_sh._get_toast_callable = _ng_toast_callable  # type: ignore[attr-defined]

# Re-export the full public API so callers only need to import from ng_health.
from .system_health import (  # noqa: E402 – after the patch
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
