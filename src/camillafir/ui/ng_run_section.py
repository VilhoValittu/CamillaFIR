"""NiceGUI Run tab: START button, progress bar, status area, results container.

Replaces PyWebIO build_run_section() + the status-area rendering in app.py.

Status updates
--------------
Instead of using run_js() DOM patches, this module polls ui_state every
500 ms via a NiceGUI timer. This is simpler and avoids all threading/session
concerns.

External access
---------------
ng_bridge.py calls set_progress_element_getter to wire up the progress bar.
ng_results_sections.py calls get_results_container() to get the container
it should render results into.
"""
from __future__ import annotations

import logging
import threading
import time

from . import ui_state
from ..resources.i8n.camillafir_i18n import t

logger = logging.getLogger("CamillaFIR")

_results_container_ref = None
_progress_ref = None
_run_clock: dict = {"started_at": None, "active": False, "elapsed_s": None}


def get_results_container():
    return _results_container_ref


def get_progress_element():
    return _progress_ref


def _format_progress_percent(value) -> str:
    try:
        pct = int(round(max(0.0, min(1.0, float(value))) * 100.0))
    except Exception:
        pct = 0
    return f"{pct}%"


def _format_elapsed_clock(value) -> str:
    try:
        elapsed = max(0.0, float(value))
    except Exception:
        return ""
    total_s = int(elapsed)
    hours, rem = divmod(total_s, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def build_global_progress_bar() -> None:
    """Render the progress bar and status area in the sticky header.

    Must be called inside the cf-top-shell column context (ng_app._build_header).
    Creates the progress bar, status labels, and the 500 ms polling timer.
    """
    global _progress_ref

    from nicegui import ui
    from . import ng_bridge

    progress = ui.linear_progress(value=0.0, size="24px", show_value=False).classes("w-full")
    with progress:
        with ui.row().classes("absolute-full items-center no-wrap px-3 gap-3"):
            progress_phase_label = ui.label("").classes("text-xs text-white font-medium truncate min-w-0 grow")
            with ui.row().classes("items-center no-wrap gap-3 shrink-0"):
                progress_elapsed_label = ui.label("").classes("text-xs text-white font-medium whitespace-nowrap")
                ui.label("").classes("text-sm text-white font-medium whitespace-nowrap").bind_text_from(
                    progress,
                    "value",
                    backward=_format_progress_percent,
                )
    progress.visible = False
    _progress_ref = progress

    ng_bridge.set_progress_element_getter(get_progress_element)

    with ui.column().classes("w-full gap-1"):
        info_box = ui.label("").classes("cf-status-info")
        info_box.visible = False
        auto_bar = ui.label("").classes("cf-auto-bar")
        auto_bar.visible = False
        with ui.expansion(t("run_auto_details_title")).classes("w-full text-xs") as auto_details_exp:
            auto_details_exp.visible = False
            auto_details_label = ui.label("").classes("whitespace-pre text-xs")

    def _refresh_status() -> None:
        snap = ui_state.get_status_snapshot()

        phase_txt = snap.get("status_base_message", "") or ""
        progress_phase_label.set_text(phase_txt)
        if _run_clock["active"] and _run_clock["started_at"] is not None:
            elapsed_text = _format_elapsed_clock(time.perf_counter() - float(_run_clock["started_at"]))
        else:
            elapsed_text = _format_elapsed_clock(_run_clock["elapsed_s"])
        progress_elapsed_label.set_text(elapsed_text)

        info = snap.get("status_info_text", "") or ""
        info_box.set_text(info)
        info_box.set_visibility(bool(info))

        auto_txt = snap.get("auto_selected_bar_text", "") or ""
        auto_bar.set_text(auto_txt)
        auto_bar.set_visibility(bool(auto_txt))

        details = snap.get("auto_status_detail_body", "") or ""
        auto_details_label.set_text(details)
        auto_details_exp.set_visibility(bool(details))

    ui.timer(0.5, _refresh_status)


def build_info_panel() -> None:
    """Compact config/score panel for the sticky header (top-right)."""
    from nicegui import ui
    from . import ng_controls

    with ui.element("div").classes("cf-info-panel"):
        line1 = ui.label("").classes("cf-info-line-dim")
        line2 = ui.label("").classes("cf-info-line-dim")
        line_meas = ui.label("").classes("cf-info-line-dim")
        line3 = ui.label("")
        line3.set_visibility(False)

    def _fmt_fs(raw) -> str:
        try:
            hz = int(raw)
            return f"{hz // 1000} kHz" if hz % 1000 == 0 else f"{hz / 1000:.1f} kHz"
        except (TypeError, ValueError):
            return "\u2014"

    def _fmt_latency(taps_raw, fs_raw, ftype_str: str) -> str:
        ftype_low = ftype_str.lower()
        if "min" in ftype_low or "asym" in ftype_low:
            return t("health_low_latency_mode")
        if "linear" not in ftype_low:
            return "\u2014"
        try:
            ms = (int(taps_raw) / 2.0 / int(fs_raw)) * 1000.0
            return f"~{ms:.0f} ms"
        except (TypeError, ValueError, ZeroDivisionError):
            return "\u2014"

    def _has_meas(file_key: str, path_key: str) -> bool:
        file_val = ng_controls.value(file_key)
        path_val = ng_controls.value(path_key) or ""
        return bool(file_val) or bool(str(path_val).strip())

    def _refresh_info() -> None:
        mode = ng_controls.value("mode") or "\u2014"
        fs_raw = ng_controls.value("fs")
        taps_raw = ng_controls.value("taps")
        ftype = ng_controls.value("filter_type") or "\u2014"
        hc_mode = ng_controls.value("hc_mode") or ""

        mode_str = str(mode).strip().upper()
        ftype_str = str(ftype).strip()
        taps_str = str(int(taps_raw)) if taps_raw is not None else "\u2014"
        lat_str = _fmt_latency(taps_raw, fs_raw, ftype_str)
        hc_str = str(hc_mode).strip() if hc_mode else "\u2014"
        line1.set_text(f"{mode_str} \u00b7 {_fmt_fs(fs_raw)} \u00b7 {taps_str} taps")
        line2.set_text(f"{ftype_str} \u00b7 {lat_str} \u00b7 {hc_str}")

        bass_int = bool(ng_controls.value("bass_integration_enable"))
        if bass_int:
            l_ok = _has_meas("file_l_main", "local_path_l_main")
            r_ok = _has_meas("file_r_main", "local_path_r_main")
        else:
            l_ok = _has_meas("file_l", "local_path_l")
            r_ok = _has_meas("file_r", "local_path_r")
        l_sym = "\u2713" if l_ok else "\u2013"
        r_sym = "\u2713" if r_ok else "\u2013"
        meas_label = t("info_panel_meas")
        line_meas.set_text(f"{meas_label}  L {l_sym}  R {r_sym}")
        if l_ok and r_ok:
            line_meas.classes(add="cf-info-line-ok", remove="cf-info-line-dim cf-info-line-warn")
        elif l_ok or r_ok:
            line_meas.classes(add="cf-info-line-warn", remove="cf-info-line-dim cf-info-line-ok")
        else:
            line_meas.classes(add="cf-info-line-dim", remove="cf-info-line-ok cf-info-line-warn")

        info = ui_state.get_last_run_info()
        if info:
            score = info.get("score")
            match = info.get("match")
            conf = info.get("conf")
            parts = []
            parts.append(
                t("run_info_score").format(score=score)
                if score is not None else
                t("run_info_score_missing")
            )
            parts.append(
                t("run_info_match").format(match=match)
                if match is not None else
                t("run_info_match_missing")
            )
            if conf is not None:
                parts.append(t("run_info_conf").format(conf=conf))
            line3.set_text(" \u00b7 ".join(parts))
            line3.classes(add="cf-info-line-score", remove="cf-info-line-dim")
            line3.set_visibility(True)
        else:
            line3.set_visibility(False)

    ui.timer(1.0, _refresh_info)


def build_run_section(*, on_start_click) -> None:
    """Build the Run tab content. Must be called inside a ui.tab_panel context."""
    global _results_container_ref

    from nicegui import ui

    with ui.column().classes("w-full gap-3"):
        start_btn = ui.button(
            t("run_start_button"),
            on_click=lambda: _handle_start(on_start_click, start_btn, _run_clock),
        ).classes("w-full text-2xl font-bold tracking-widest py-4").props(
            'color="positive" unelevated'
        )

        _results_container_ref = ui.column().classes("w-full gap-2")


def _clear_previous_run_output() -> None:
    ui_state.set_last_run_info({})
    container = get_results_container()
    if container is None:
        return
    try:
        container.clear()
    except Exception:
        logger.debug("Failed to clear previous run output", exc_info=True)


def _handle_start(on_start_click, start_btn, run_clock) -> None:
    """Run the DSP pipeline in a background thread so the UI stays responsive."""
    from nicegui import ui

    start_btn.disable()
    run_clock["started_at"] = time.perf_counter()
    run_clock["elapsed_s"] = 0.0
    run_clock["active"] = True
    if _progress_ref is not None:
        _progress_ref.set_value(0.0)
        _progress_ref.set_text_color("primary")
        _progress_ref.set_visibility(True)

    _clear_previous_run_output()
    ui_state.update_status(t("stat_reading"))
    ui_state.update_status_notices(summary_text="", info_text="")
    ui_state.update_auto_selected_bar("")
    ui_state.reset_auto_status_details()

    def _run() -> None:
        try:
            on_start_click()
        except Exception:
            logger.exception("DSP run failed")
        finally:
            try:
                run_clock["elapsed_s"] = max(0.0, float(time.perf_counter() - float(run_clock["started_at"])))
            except Exception:
                pass
            run_clock["active"] = False
            try:
                start_btn.enable()
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()
