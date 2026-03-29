"""NiceGUI mode-driven UI update functions.

Replaces controls_mode.py + controls_ir_window.py + parts of callbacks.py.

Each function reads the current state from ng_controls and updates
visibility/options/values of dependent elements accordingly.

Phase 3 fills in the full implementations as tab elements are created.
"""
from __future__ import annotations

import logging
from typing import Callable

from . import ng_controls as ctrl
from .camillafir_utils import scale_taps_with_fs

logger = logging.getLogger("CamillaFIR")

_MULTI_RATE_RATES = [44100, 48000, 88200, 96000, 176400, 192000]


# ---------------------------------------------------------------------------
# Master mode change
# ---------------------------------------------------------------------------

def on_mode_change(*, mode: str, t: Callable) -> None:
    """Cascade all mode-driven UI changes.

    Mirrors PyWebIO _on_mode_change() from callbacks.py.
    """
    mode_u = str(mode or "BASIC").strip().upper()
    is_auto = mode_u == "AUTO"
    is_basic = mode_u == "BASIC"
    is_advanced = mode_u == "ADVANCED"

    # lvl_mode options
    if is_advanced:
        ctrl.set_options("lvl_mode", {
            "Auto":   t("lvl_mode_auto"),
            "Manual": t("lvl_mode_manual"),
        })
    else:
        ctrl.set_options("lvl_mode", {"Auto": t("lvl_mode_auto")})
        ctrl.set_value("lvl_mode", "Auto")

    # Mode description label
    key_map = {"AUTO": "mode_auto_desc", "ADVANCED": "mode_advanced_desc"}
    desc_key = key_map.get(mode_u, "mode_basic_desc")
    desc_container = ctrl.get_container("mode_desc_scope")
    if desc_container is not None:
        try:
            desc_container.clear()
            from nicegui import ui  # noqa: PLC0415
            with desc_container:
                ui.markdown(f"**{t('mode_desc_title')}**\n\n{t(desc_key)}")
        except Exception:
            logger.debug("mode desc update failed", exc_info=True)

    # Raw DSP section visibility (ADVANCED only)
    ctrl.set_visibility("unsafe_raw_dsp_scope", is_advanced)

    # Confidence pull (ADVANCED only)
    ctrl.set_visibility("conf_pull_scope", is_advanced)

    # AUTO-mode controls disable
    _update_auto_mode_fields_state(is_auto=is_auto, t=t)

    # Apply mode-driven IR window default
    _apply_mode_ir_window_default(mode_u=mode_u)

    # Cascade dependent updates
    update_lvl_ui(t=t)
    update_mixed_freq_ui(t=t)
    update_engine_metrics_ui(t=t)


# ---------------------------------------------------------------------------
# Level UI
# ---------------------------------------------------------------------------

def update_lvl_ui(*, t: Callable) -> None:
    """Show/hide manual dB input depending on lvl_mode value."""
    lvl = str(ctrl.value("lvl_mode", "Auto") or "Auto").strip()
    is_manual = lvl.lower() == "manual"
    ctrl.set_visibility("lvl_manual_scope", is_manual)


def update_lvl_range() -> None:
    """Swap lvl_min/lvl_max if inverted."""
    try:
        lo = float(ctrl.value("lvl_min", 0.0) or 0.0)
        hi = float(ctrl.value("lvl_max", 0.0) or 0.0)
        if lo >= hi:
            ctrl.set_value("lvl_min", hi - 1.0)
            ctrl.set_value("lvl_max", lo + 1.0)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# IR window controls
# ---------------------------------------------------------------------------

def update_ir_window_controls(*, t: Callable) -> None:
    """Show/hide IR window sub-controls based on mode + shape selections."""
    mode = str(ctrl.value("ir_export_window_mode", "") or "").strip().lower()
    shape = str(ctrl.value("ir_export_window_shape", "") or "").strip().lower()
    ftype = str(ctrl.value("filter_type", "") or "").strip().lower()

    is_manual = mode in ("manual", "energy", "peak")
    is_asym = "asym" in ftype

    ctrl.set_visibility("ir_lr_window_scope", is_manual or is_asym)
    ctrl.set_visibility("ir_tukey_alpha_scope", shape == "tukey")

    update_ir_tukey_ui()


def update_ir_tukey_ui() -> None:
    """Show Tukey alpha input only when shape=tukey."""
    shape = str(ctrl.value("ir_export_window_shape", "") or "").strip().lower()
    ctrl.set_visibility("ir_tukey_alpha_scope", shape == "tukey")


# ---------------------------------------------------------------------------
# Bass controls
# ---------------------------------------------------------------------------

def update_low_bass_cut_ui() -> None:
    enabled = bool(ctrl.value("low_bass_cut_enable", False))
    ctrl.set_visibility("low_bass_cut_scope", enabled)


def update_bass_first_ui() -> None:
    enabled = bool(ctrl.value("bass_first_ai", False))
    ctrl.set_visibility("bass_first_max_hz_scope", enabled)


# ---------------------------------------------------------------------------
# Mixed-phase controls
# ---------------------------------------------------------------------------

def update_mixed_freq_ui(*, t: Callable) -> None:
    """Show mixed split only for BASIC/ADVANCED when mixed-phase is selected."""
    try:
        mode_u = str(ctrl.value("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    ftype = str(ctrl.value("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ftype
    show = mode_u in ("BASIC", "ADVANCED") and is_mixed

    ctrl.set_visibility("update_mixed_freq_scope", show)
    ctrl.set_enabled("mixed_freq", show)


# ---------------------------------------------------------------------------
# TDC / A-FDW
# ---------------------------------------------------------------------------

def update_tdc_controls_ui(*, t: Callable) -> None:
    enabled = bool(ctrl.value("enable_tdc", False))
    ctrl.set_visibility("tdc_section_scope", enabled)


def update_afdw_cycles_ui(*, t: Callable) -> None:
    enabled = bool(ctrl.value("enable_afdw", False))
    ctrl.set_visibility("afdw_section_scope", enabled)


# ---------------------------------------------------------------------------
# Engine metrics  (latency + resolution display)
# ---------------------------------------------------------------------------

def update_engine_metrics_ui(*, t: Callable) -> None:
    """Update latency / Hz-per-bin info label."""
    try:
        fs = int(ctrl.value("fs", 48000) or 48000)
        taps = int(ctrl.value("taps", 65536) or 65536)
        if fs <= 0 or taps <= 0:
            return
        latency_ms = (taps / 2.0) / float(fs) * 1000.0
        bin_hz = float(fs) / float(taps)
        ftype = str(ctrl.value("filter_type", "") or "").lower()
        is_min = "min" in ftype
        is_asym = "asym" in ftype
        if is_min or is_asym:
            lat_str = t("health_low_latency_mode")
        else:
            lat_str = f"{latency_ms:.0f} ms"
        metrics_el = ctrl.get("engine_metrics_label")
        if metrics_el is not None:
            metrics_el.set_text(f"{lat_str}  |  {bin_hz:.2f} Hz/bin")
    except Exception:
        logger.debug("update_engine_metrics_ui failed", exc_info=True)


def update_taps_auto_info(*, t: Callable) -> None:
    """Render multi-rate taps info into both NiceGUI scope containers."""
    try:
        multi_rate = bool(ctrl.value("multi_rate_opt", False))
    except Exception:
        multi_rate = False

    try:
        base_taps = int(float(ctrl.value("taps", 65536) or 65536))
    except Exception:
        base_taps = 65536

    markdown = _build_taps_auto_info_markdown(
        multi_rate=multi_rate,
        base_taps=base_taps,
        t=t,
    )

    try:
        from nicegui import ui  # noqa: PLC0415
    except Exception:
        logger.debug("update_taps_auto_info failed to import NiceGUI", exc_info=True)
        return

    for scope_name in ("taps_auto_info_scope_files", "taps_auto_info_scope_basic"):
        container = ctrl.get_container(scope_name)
        if container is None:
            continue
        try:
            container.clear()
            with container:
                ui.markdown(markdown).classes("w-full")
        except Exception:
            logger.debug("update_taps_auto_info failed for %s", scope_name, exc_info=True)


def _build_taps_auto_info_markdown(*, multi_rate: bool, base_taps: int, t: Callable) -> str:
    """Build the multi-rate taps help text shown in Files and Basic tabs."""
    if not multi_rate:
        return f"_{t('auto_taps_title')}: OFF_"

    lines = [
        f"- **{rate / 1000:.1f} kHz** -> **{scale_taps_with_fs(rate, base_taps=base_taps):,}** taps"
        for rate in _MULTI_RATE_RATES
    ]
    return (
        f"### {t('auto_taps_title')}\n"
        f"{t('auto_taps_body')}\n\n"
        f"**Reference:** 44.1 kHz -> {base_taps:,} taps\n\n"
        + "\n".join(lines)
    )


# ---------------------------------------------------------------------------
# AUTO-mode field enable/disable
# ---------------------------------------------------------------------------

_AUTO_LOCKED_FIELDS = [
    "comparison_mode", "gain", "lvl_algo", "lvl_min", "lvl_max",
    "lvl_mode", "lvl_manual_db", "mag_correct", "mag_c_min", "mag_c_max",
    "max_boost", "bass_first_ai", "bass_first_mode_max_hz",
    "max_slope_db_per_oct", "max_slope_boost_db_per_oct",
    "max_slope_cut_db_per_oct", "trans_width", "filter_smooth",
    "phase_limit", "df_smoothing", "reg_strength",
    "stereo_link", "stereo_link_strategy",
    "exc_prot", "exc_freq", "low_bass_cut_enable",
    "ir_export_window_mode", "ir_export_window_shape", "ir_export_tukey_alpha",
    "enable_afdw", "fdw_cycles", "enable_tdc",
    "tdc_strength", "tdc_max_reduction_db", "tdc_slope_db_per_oct",
    "mixed_freq", "plot_smoothing_level",
]
_AUTO_ONLY_FIELDS = ["auto_goal", "auto_target_mode"]
_HC_LOCKED_FIELDS = ["hc_mode", "hc_custom_file"]


def _update_auto_mode_fields_state(*, is_auto: bool, t: Callable) -> None:
    for name in _AUTO_LOCKED_FIELDS:
        ctrl.set_enabled(name, not is_auto)
    for name in _AUTO_ONLY_FIELDS:
        ctrl.set_enabled(name, is_auto)

    auto_target_mode = str(ctrl.value("auto_target_mode", "auto") or "auto").lower()
    hc_locked = is_auto and auto_target_mode in ("auto", "adaptive")
    for name in _HC_LOCKED_FIELDS:
        ctrl.set_enabled(name, not hc_locked)


# ---------------------------------------------------------------------------
# Mode-driven IR export window default
# ---------------------------------------------------------------------------

def _apply_mode_ir_window_default(*, mode_u: str) -> None:
    try:
        from ..config.mode_policy import MODE_DEFAULTS  # noqa: PLC0415
        key = "BASIC" if mode_u in ("BASIC", "AUTO") else "ADVANCED"
        v = MODE_DEFAULTS.get(key, {}).get("ir_export_window_mode", None)
        if v:
            ctrl.set_value("ir_export_window_mode", str(v).strip())
            update_ir_window_controls(t=lambda k: k)
    except Exception:
        logger.debug("_apply_mode_ir_window_default failed", exc_info=True)
