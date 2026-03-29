"""NiceGUI cross-tab reactive callbacks.

Replaces callbacks.py.  This module is called AFTER all tab builders have
registered their elements in ng_controls, so all elements are guaranteed to
exist when register_callbacks() runs.

Phase 3 will implement each callback.  This stub allows ng_app.py to import
without errors during development.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from . import ng_controls as ctrl

logger = logging.getLogger("CamillaFIR")


def register_callbacks(*, t: Callable, get_val: Callable, max_safe_boost: float) -> None:
    """Register all reactive callbacks on form elements.

    Called once per page load, after all tab builders have run.
    Phase 3 fills in each individual callback below.
    """
    _register_mode_callbacks(t=t)
    _register_target_callbacks(t=t)
    _register_lvl_callbacks(t=t)
    _register_ir_window_callbacks(t=t)
    _register_bass_callbacks(t=t)
    _register_tdc_afdw_callbacks(t=t)
    _register_stereo_callbacks(t=t)
    _register_metric_callbacks(t=t)
    _initial_state_sync(t=t, get_val=get_val)


# ---------------------------------------------------------------------------
# Stubs – each will be implemented in Phase 3
# ---------------------------------------------------------------------------

def _register_mode_callbacks(*, t: Callable) -> None:
    """mode → lvl_mode options, desc, auto controls, raw dsp visibility."""

    def _on_mode_change(v: Any) -> None:
        from .ng_mode_controls import on_mode_change
        on_mode_change(mode=str(v or "BASIC").upper(), t=t)
        _update_target_preview()

    ctrl.on_change("mode", _on_mode_change)


def _register_target_callbacks(*, t: Callable) -> None:
    """hc_mode, hc_custom_file, auto_goal, auto_target_mode → target preview."""
    def _sync_hc_upload_visibility(v: Any) -> None:
        upload_col = ctrl.get_container("hc_custom_upload_col")
        if upload_col is None:
            return
        try:
            upload_col.set_visibility(str(v or "").strip().lower() == "upload")
        except Exception:
            logger.debug("hc custom upload visibility update failed", exc_info=True)

    _preview_fields = [
        "hc_mode", "hc_custom_file", "auto_goal", "auto_target_mode",
        "mag_c_min", "mag_c_max", "file_l", "file_r",
        "local_path_l", "local_path_r", "lvl_min", "lvl_max",
        "ir_window_left", "ir_window_right", "ir_window",
        "filter_smooth", "smoothing_level",
    ]
    for field in _preview_fields:
        ctrl.on_change(field, lambda v, f=field: _update_target_preview())

    def _on_hc_mode_change(v: Any) -> None:
        _sync_hc_upload_visibility(v)
        if str(v or "").strip().lower() != "upload":
            ctrl.set_value("hc_custom_file", None)
        _update_target_preview()

    ctrl.on_change("hc_mode", _on_hc_mode_change)
    _sync_hc_upload_visibility(ctrl.value("hc_mode", "Harman6"))


def _register_lvl_callbacks(*, t: Callable) -> None:
    """lvl_mode → show/hide manual dB; lvl_min/max → swap if inverted."""

    def _on_lvl_mode(v: Any) -> None:
        from .ng_mode_controls import update_lvl_ui
        update_lvl_ui(t=t)
        _update_target_preview()

    ctrl.on_change("lvl_mode", _on_lvl_mode)

    def _on_lvl_range(v: Any) -> None:
        from .ng_mode_controls import update_lvl_range
        update_lvl_range()

    for field in ("lvl_min", "lvl_max"):
        ctrl.on_change(field, _on_lvl_range)
    ctrl.on_change("lvl_manual_db", lambda v: _update_target_preview())


def _register_ir_window_callbacks(*, t: Callable) -> None:
    """ir_export_window_mode, ir_export_window_shape → show/hide sub-controls."""

    def _refresh(v: Any) -> None:
        from .ng_mode_controls import update_ir_window_controls, update_mixed_freq_ui
        update_ir_window_controls(t=t)
        update_mixed_freq_ui(t=t)

    ctrl.on_change("ir_export_window_mode", _refresh)
    ctrl.on_change("ir_export_window_shape", _refresh)
    ctrl.on_change("filter_type", _refresh)


def _register_bass_callbacks(*, t: Callable) -> None:
    """low_bass_cut_enable, bass_first_ai → show/hide sub-controls."""

    def _on_bass_cut(v: Any) -> None:
        from .ng_mode_controls import update_low_bass_cut_ui
        update_low_bass_cut_ui()

    ctrl.on_change("low_bass_cut_enable", _on_bass_cut)

    def _on_bass_first(v: Any) -> None:
        from .ng_mode_controls import update_bass_first_ui
        update_bass_first_ui()

    ctrl.on_change("bass_first_ai", _on_bass_first)


def _register_tdc_afdw_callbacks(*, t: Callable) -> None:
    """enable_tdc, enable_afdw → show/hide sub-controls + clamp hints."""

    def _on_tdc(v: Any) -> None:
        from .ng_mode_controls import update_tdc_controls_ui
        update_tdc_controls_ui(t=t)

    def _on_afdw(v: Any) -> None:
        from .ng_mode_controls import update_afdw_cycles_ui
        update_afdw_cycles_ui(t=t)

    ctrl.on_change("enable_tdc", _on_tdc)
    ctrl.on_change("enable_afdw", _on_afdw)


def _register_stereo_callbacks(*, t: Callable) -> None:
    """stereo_link → enable/disable stereo_link_strategy."""

    def _on_stereo(v: Any) -> None:
        mode = str(ctrl.value("mode", "BASIC") or "BASIC").upper()
        enabled = bool(v) and mode != "AUTO"
        ctrl.set_enabled("stereo_link_strategy", enabled)

    ctrl.on_change("stereo_link", _on_stereo)
    ctrl.on_change("mode", lambda v: _on_stereo(ctrl.value("stereo_link", False)))


def _register_metric_callbacks(*, t: Callable) -> None:
    """fs, taps, multi_rate_opt → latency/resolution display."""

    def _refresh(v: Any) -> None:
        from .ng_mode_controls import update_engine_metrics_ui, update_taps_auto_info
        update_engine_metrics_ui(t=t)
        update_taps_auto_info(t=t)

    for field in ("fs", "taps", "multi_rate_opt", "filter_type"):
        ctrl.on_change(field, _refresh)


def _initial_state_sync(*, t: Callable, get_val: Callable) -> None:
    """Apply initial show/hide state for all dynamic sections."""
    try:
        from .ng_mode_controls import (  # noqa: PLC0415
            on_mode_change,
            update_ir_window_controls,
            update_lvl_ui,
            update_mixed_freq_ui,
            update_taps_auto_info,
        )
        mode = str(ctrl.value("mode", get_val("mode", "BASIC")) or "BASIC").upper()
        on_mode_change(mode=mode, t=t)
        update_lvl_ui(t=t)
        update_ir_window_controls(t=t)
        update_mixed_freq_ui(t=t)
        update_taps_auto_info(t=t)
        _update_target_preview()
    except Exception:
        logger.debug("_initial_state_sync failed", exc_info=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _update_target_preview() -> None:
    """Update target curve preview.  Implemented in Phase 3 (ng_tab_target)."""
    try:
        from .ng_tab_target import refresh_target_preview  # noqa: PLC0415
        refresh_target_preview()
    except Exception:
        pass
