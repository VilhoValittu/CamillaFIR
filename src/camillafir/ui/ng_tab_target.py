"""NiceGUI Target tab builder.

Replaces build_target_section() from layout_builders.py.
"""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl
from .target_preview_common import apply_manual_target_preview_adjustments
from .target_preview_interaction import (
    build_draggable_tilt_handle_shape,
    build_draggable_target_shape,
    build_level_window_trace,
    build_target_curve_path,
    build_tilt_handle_path,
    build_vertical_marker_trace,
    extract_target_tilt_from_shape_relayout,
    extract_vertical_shift_from_shape_relayout,
    parse_svg_path_points,
)

_HC_OPTS = {
    "Harman6":   "Harman 6 dB",
    "Harman8":   "Harman 8 dB",
    "Harman4":   "Harman 4 dB",
    "Harman10":  "Harman 10 dB",
    "Harman12":  "Harman 12 dB",
    "Studio":    "Studio Tilt",
    "Nearfield": "Nearfield",
    "HiFi":      "HiFi Loudness",
    "Speech":    "Speech",
    "Toole":     "Toole",
    "BK_Light":  "BK Light",
    "BK_Medium": "BK Medium",
    "BK_Strong": "BK Strong",
    "Flat":      "Flat",
    "Cinema":    "Cinema",
    "Upload":    "Upload Custom",
}

_TARGET_PREVIEW_REFRESH_TOKEN = 0
_TARGET_PREVIEW_DRAG_BASE_POINTS: list[tuple[float, float]] = []
_TARGET_PREVIEW_TILT_HANDLE_POINTS: list[tuple[float, float]] = []
_TARGET_PREVIEW_DRAG_ACTIVE = False


def _step_manual_target(delta_db: float) -> None:
    try:
        cur = float(ctrl.value("lvl_manual_db", 0.0) or 0.0)
    except Exception:
        cur = 0.0

    nxt = round((float(cur) + float(delta_db)) * 10.0) / 10.0
    ctrl.set_value("lvl_manual_db", float(nxt))
    refresh_target_preview()


def _step_manual_target_tilt(delta_db_per_oct: float) -> None:
    try:
        cur = float(ctrl.value("manual_target_tilt_db_per_oct", 0.0) or 0.0)
    except Exception:
        cur = 0.0

    nxt = round((float(cur) + float(delta_db_per_oct)) * 10.0) / 10.0
    ctrl.set_value("manual_target_tilt_db_per_oct", float(nxt))
    refresh_target_preview()


def build_target_tab(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui
    from ..application.house_curve_service import _normalize_hc_mode_key  # noqa: PLC0415

    ui.markdown(f"#### {t('tab_target')}")
    ui.separator()
    ui.markdown(f"#### {t('ui_target_preview')}")

    # Target preview container
    preview_col = ui.column().classes("w-full")
    ctrl.register_container("target_preview_scope", preview_col)

    ui.separator()

    # House curve selector + custom upload holder
    _hc_file = ctrl._ValueHolder(get_val("hc_custom_file", None))
    ctrl.register("hc_custom_file", _hc_file)

    hc_value = _normalize_hc_mode_key(get_val("hc_mode", "Harman6"))
    ctrl.register(
        "hc_mode",
        ui.select(
            options=_HC_OPTS,
            value=hc_value,
            label=t("hc_mode"),
        ).props("dense outlined").classes("w-full"),
    )

    # Custom file upload (visible only when hc_mode=Upload)
    with ui.column().classes("w-full") as hc_upload_col:
        ui.label(t("hc_custom")).classes("text-sm font-medium")

        async def _on_hc_upload(e) -> None:
            _hc_file.value = {
                "filename": e.file.name,
                "content": await e.file.read(),
                "mime_type": getattr(e.file, "content_type", ""),
            }
            refresh_target_preview()

        ui.upload(
            label=t("hc_custom"),
            on_upload=_on_hc_upload,
            auto_upload=True,
        ).props('accept=".txt"').classes("w-full")
    ctrl.register_container("hc_custom_upload_col", hc_upload_col)
    hc_upload_col.set_visibility(hc_value == "Upload")

    ui.separator()
    ui.markdown(f"#### 🎚 {t('ui_leveling_gain')}")

    with ui.row().classes("w-full gap-4"):
        ctrl.register(
            "lvl_algo",
            ui.select(
                ["Median", "Average"],
                value=get_val("lvl_algo", "Median"),
                label=t("lvl_algo"),
            ).props("dense outlined").classes("flex-1"),
        )
        ctrl.register(
            "gain",
            ui.number(
                label=t("gain"),
                value=float(get_val("gain", 0.0) or 0.0),
                format="%.1f",
            ).props("dense outlined").classes("flex-1"),
        )

    with ui.row().classes("w-full gap-4"):
        ctrl.register(
            "lvl_min",
            ui.number(
                label=t("lvl_min"),
                value=float(get_val("lvl_min", 500.0) or 500.0),
                format="%.0f",
            ).props("dense outlined").classes("flex-1"),
        )
        ctrl.register(
            "lvl_max",
            ui.number(
                label=t("lvl_max"),
                value=float(get_val("lvl_max", 2000.0) or 2000.0),
                format="%.0f",
            ).props("dense outlined").classes("flex-1"),
        )

    with ui.row().classes("w-full gap-4 items-start"):
        ctrl.register(
            "lvl_mode",
            ui.select(
                options={
                    "Auto":   t("lvl_mode_auto"),
                    "Manual": t("lvl_mode_manual"),
                },
                value=get_val("lvl_mode", "Auto"),
                label=t("lvl_mode"),
            ).props("dense outlined").classes("flex-1"),
        )
        # Manual dB scope (shown only when lvl_mode=Manual)
        lvl_manual_col = ui.column().classes("flex-1 gap-1")
        ctrl.register_container("lvl_manual_scope", lvl_manual_col)
        with lvl_manual_col:
            with ui.row().classes("w-full gap-2 items-end"):
                ctrl.register(
                    "lvl_manual_db",
                    ui.number(
                        label=t("lvl_target_db"),
                        value=float(get_val("lvl_manual_db", 0.0) or 0.0),
                        format="%.1f",
                    ).props("dense outlined step=0.1").classes("flex-1"),
                )
                ui.button(
                    "+",
                    on_click=lambda: _step_manual_target(+0.1),
                ).props('color="secondary" outline').style("min-width:34px;")
                ui.button(
                    "-",
                    on_click=lambda: _step_manual_target(-0.1),
                ).props('color="secondary" outline').style("min-width:34px;")
            with ui.row().classes("w-full gap-2 items-end"):
                ctrl.register(
                    "manual_target_tilt_db_per_oct",
                    ui.number(
                        label=t("manual_target_tilt"),
                        value=float(get_val("manual_target_tilt_db_per_oct", 0.0) or 0.0),
                        format="%.1f",
                    ).props("dense outlined step=0.1").classes("flex-1"),
                )
                ui.button(
                    "+",
                    on_click=lambda: _step_manual_target_tilt(+0.1),
                ).props('color="secondary" outline').style("min-width:34px;")
                ui.button(
                    "-",
                    on_click=lambda: _step_manual_target_tilt(-0.1),
                ).props('color="secondary" outline').style("min-width:34px;")
            ui.label(t("lvl_manual_help")).classes("text-xs text-gray-400")
            ui.label(t("manual_target_tilt_help")).classes("text-xs text-gray-400")
            ui.label(t("lvl_manual_bias_hint")).classes("text-xs text-gray-400")
            ui.label(t("lvl_manual_drag_hint_curve")).classes("text-xs text-gray-400")
            ui.label(t("manual_target_tilt_drag_hint")).classes("text-xs text-gray-400")
        lvl_manual_col.set_visibility(False)

    ui.separator()

    # Magnitude correction
    ctrl.register(
        "mag_correct",
        ui.checkbox(t("enable_corr"), value=bool(get_val("mag_correct", True))),
    )

    ui.label(t("magnitude_correction_limits")).classes("text-sm font-semibold mt-4")

    with ui.row().classes("w-full gap-4"):
        ctrl.register(
            "mag_c_min",
            ui.number(
                label=t("min_freq"),
                value=float(get_val("mag_c_min", 10.0) or 10.0),
                format="%.1f",
            ).props("dense outlined").classes("flex-1"),
        )
        ctrl.register(
            "mag_c_max",
            ui.number(
                label=t("max_freq"),
                value=float(get_val("mag_c_max", 200.0) or 200.0),
                format="%.1f",
            ).props("dense outlined").classes("flex-1"),
        )

    ui.separator()

    ctrl.register(
        "max_boost",
        ui.number(
            label=t("max_boost"),
            value=float(get_val("max_boost", 5.0) or 5.0),
            format="%.1f",
        ).props("dense outlined").classes("w-full"),
    )


def refresh_target_preview() -> None:
    """Regenerate the target curve preview plot (NiceGUI version).

    Reads values from ng_controls instead of PyWebIO pin.
    """
    global _TARGET_PREVIEW_DRAG_ACTIVE, _TARGET_PREVIEW_DRAG_BASE_POINTS, _TARGET_PREVIEW_TILT_HANDLE_POINTS

    preview_col = ctrl.get_container("target_preview_scope")
    if preview_col is None:
        return

    fig, drag_base_points, tilt_handle_points = _build_target_preview_fig()
    _TARGET_PREVIEW_DRAG_BASE_POINTS = drag_base_points
    _TARGET_PREVIEW_TILT_HANDLE_POINTS = tilt_handle_points
    preview_col.clear()
    if fig is not None:
        from nicegui import ui  # noqa: PLC0415

        with preview_col:
            plot = ui.plotly(fig).classes("w-full")
            plot.on("plotly_relayout", _on_target_preview_relayout)
    _TARGET_PREVIEW_DRAG_ACTIVE = False


def _schedule_target_preview_refresh(delay_s: float = 0.10) -> None:
    global _TARGET_PREVIEW_REFRESH_TOKEN

    preview_col = ctrl.get_container("target_preview_scope")
    if preview_col is None:
        return

    _TARGET_PREVIEW_REFRESH_TOKEN += 1
    token = _TARGET_PREVIEW_REFRESH_TOKEN

    from nicegui import ui  # noqa: PLC0415

    def _run() -> None:
        global _TARGET_PREVIEW_DRAG_ACTIVE

        if token != _TARGET_PREVIEW_REFRESH_TOKEN:
            return
        try:
            refresh_target_preview()
        finally:
            _TARGET_PREVIEW_DRAG_ACTIVE = False

    with preview_col:
        ui.timer(delay_s, _run, once=True, immediate=False)


def _on_target_preview_relayout(e) -> None:
    global _TARGET_PREVIEW_DRAG_ACTIVE

    app_mode = str(ctrl.value("mode", "BASIC") or "BASIC").upper()
    lvl_mode = str(ctrl.value("lvl_mode", "Auto") or "Auto").strip().lower()
    if app_mode in ("BASIC", "AUTO") or lvl_mode != "manual":
        return

    payload = getattr(e, "args", None)
    if not isinstance(payload, dict):
        return

    new_tilt = extract_target_tilt_from_shape_relayout(
        payload,
        _TARGET_PREVIEW_TILT_HANDLE_POINTS,
    )
    if new_tilt is not None:
        _TARGET_PREVIEW_DRAG_ACTIVE = True
        ctrl.set_value("manual_target_tilt_db_per_oct", new_tilt, emit=False)
        _schedule_target_preview_refresh()
        return

    new_db = extract_vertical_shift_from_shape_relayout(
        payload,
        _TARGET_PREVIEW_DRAG_BASE_POINTS,
    )
    if new_db is None:
        return

    _TARGET_PREVIEW_DRAG_ACTIVE = True
    ctrl.set_value("lvl_manual_db", new_db, emit=False)
    _schedule_target_preview_refresh()


def _build_target_preview_fig():
    """Build the target curve preview Plotly figure from current ctrl values.

    Returns a Plotly dict plus the base points of the draggable target curve and tilt handle.
    """
    try:
        import math  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import plotly.graph_objects as go  # noqa: PLC0415
        from ..auto_mode.shared import _auto_goal_forced_level_window  # noqa: PLC0415
        from ..dsp.smoothing import psychoacoustic_smoothing as _psycho_smooth  # noqa: PLC0415
        from ..io.measurements_txt import (  # noqa: PLC0415
            parse_measurements_from_bytes as _parse_txt_bytes,
            parse_measurements_from_path as _parse_txt_path,
        )
        from ..io.measurements_wav import (  # noqa: PLC0415
            parse_measurements_from_wav_bytes as _parse_wav_bytes,
            parse_measurements_from_wav_path as _parse_wav_path,
        )
        from ..application.house_curve_service import (  # noqa: PLC0415
            _normalize_hc_mode_key,
            load_house_curve,
            load_target_curve,
        )

        global _TARGET_PREVIEW_DRAG_ACTIVE

        def _cv(name, default=None):
            return ctrl.value(name, default)

        def _to_float(v, default):
            try:
                x = float(v)
                if math.isfinite(x):
                    return x
            except Exception:
                pass
            return float(default)

        def _normalize_curve(freqs, mags):
            try:
                ff = np.asarray(freqs, dtype=float)
                mm = np.asarray(mags, dtype=float)
                if ff.size < 8 or mm.size != ff.size:
                    return None, None
                mask = np.isfinite(ff) & np.isfinite(mm) & (ff > 0)
                ff, mm = ff[mask], mm[mask]
                order = np.argsort(ff)
                ff, mm = ff[order], mm[order]
                uniq, idx = np.unique(ff, return_index=True)
                return uniq if uniq.size >= 8 else None, mm[idx] if uniq.size >= 8 else None
            except Exception:
                return None, None

        def _smooth_for_preview(freq_axis, m):
            try:
                return _psycho_smooth(freq_axis, np.asarray(m, dtype=float))
            except Exception:
                return m

        # --- collect ctrl values ---
        hc_mode_raw = str(_cv("hc_mode", "Harman6") or "Harman6")
        hc_mode = str(_normalize_hc_mode_key(hc_mode_raw))
        hc_file = _cv("hc_custom_file", None)
        lvl_min = _to_float(_cv("lvl_min", 500.0), 500.0)
        lvl_max = _to_float(_cv("lvl_max", 2000.0), 2000.0)
        mag_c_min = _to_float(_cv("mag_c_min", 10.0), 10.0)
        mag_c_max = _to_float(_cv("mag_c_max", 200.0), 200.0)
        auto_goal = str(_cv("auto_goal", "balanced") or "balanced")
        app_mode = str(_cv("mode", "BASIC") or "BASIC").upper()
        lvl_mode = str(_cv("lvl_mode", "Auto") or "Auto")
        if app_mode in ("BASIC", "AUTO"):
            lvl_mode = "Auto"
        is_manual_level = "manual" in lvl_mode.strip().lower()
        lvl_manual_db = _to_float(_cv("lvl_manual_db", 0.0), 0.0)
        manual_target_tilt_db_per_oct = _to_float(_cv("manual_target_tilt_db_per_oct", 0.0), 0.0)
        preview_level_shift_db = lvl_manual_db if is_manual_level else 0.0
        preview_manual_target_tilt_db_per_oct = manual_target_tilt_db_per_oct if is_manual_level else 0.0
        pre_ms = _to_float(_cv("ir_window_left", 85.0), 85.0)
        post_ms = _to_float(_cv("ir_window_right") or _cv("ir_window", 500.0), 500.0)
        smoothing_level = int(float(_cv("filter_smooth", _cv("smoothing_level", 0)) or 0))

        freq_axis = np.logspace(math.log10(10.0), math.log10(20000.0), 400)

        # --- build target curve ---
        target_curve = None
        src = hc_mode_raw

        if hc_mode == "Upload" and isinstance(hc_file, dict) and hc_file.get("content"):
            try:
                tf_f, tf_m = load_target_curve(hc_file["content"])
                if tf_f is not None and tf_m is not None:
                    tf_f = np.asarray(tf_f, dtype=float)
                    tf_m = np.asarray(tf_m, dtype=float)
                    if tf_f.size >= 2 and tf_m.size == tf_f.size:
                        target_curve = np.interp(freq_axis, tf_f, tf_m, left=tf_m[0], right=tf_m[-1])
                        src = "Custom upload"
            except Exception:
                pass

        if target_curve is None:
            hc_f, hc_m, _ = load_house_curve({"hc_mode": hc_mode})
            if hc_f is not None and hc_m is not None:
                hc_f = np.asarray(hc_f, dtype=float)
                hc_m = np.asarray(hc_m, dtype=float)
                target_curve = np.interp(freq_axis, hc_f, hc_m, left=hc_m[0], right=hc_m[-1])

        if target_curve is None:
            return None, [], []

        # AUTO mode forced level window
        if app_mode in ("BASIC", "AUTO"):
            forced = _auto_goal_forced_level_window(auto_goal)
            if forced is not None:
                lvl_min, lvl_max = float(forced[0]), float(forced[1])

        # --- speaker measurements ---
        def _parse_upload(up):
            if not isinstance(up, dict) or not up.get("content"):
                return None, None
            content = up["content"]
            name = str(up.get("filename", "") or "").lower()
            try:
                is_wav = name.endswith(".wav") or (
                    isinstance(content, (bytes, bytearray)) and len(content) >= 4 and content[:4] == b"RIFF"
                )
                if is_wav:
                    ff, mm, _ = _parse_wav_bytes(content, pre_ms=pre_ms, post_ms=post_ms,
                                                  smoothing_level=smoothing_level, logger=None)
                else:
                    ff, mm, _ = _parse_txt_bytes(content)
                return _normalize_curve(ff, mm)
            except Exception:
                return None, None

        def _parse_path(path_raw):
            path = str(path_raw or "").strip().strip('"').strip("'")
            if not path:
                return None, None
            try:
                if path.lower().endswith(".wav"):
                    ff, mm, _ = _parse_wav_path(path, pre_ms=pre_ms, post_ms=post_ms,
                                                  smoothing_level=smoothing_level, logger=None)
                else:
                    ff, mm, _ = _parse_txt_path(path, logger=None)
                return _normalize_curve(ff, mm)
            except Exception:
                return None, None

        def _align(m_curve, t_curve, fa, fmin, fmax):
            try:
                mask = (fa >= fmin) & (fa <= fmax)
                if not mask.any():
                    return m_curve
                diff = np.nanmedian(t_curve[mask] - np.asarray(m_curve, dtype=float)[mask])
                return np.asarray(m_curve, dtype=float) + diff
            except Exception:
                return m_curve

        speaker_interp = {}
        for ch, up_key, path_key in (("L", "file_l", "local_path_l"), ("R", "file_r", "local_path_r")):
            ff, mm = _parse_upload(_cv(up_key, None))
            if ff is None:
                ff, mm = _parse_path(_cv(path_key, ""))
            if ff is not None and mm is not None:
                m_interp = np.interp(freq_axis, ff, mm, left=mm[0], right=mm[-1])
                m_aligned = _align(m_interp, target_curve, freq_axis, lvl_min, lvl_max)
                speaker_interp[ch] = _smooth_for_preview(freq_axis, m_aligned)

        target_curve_display = apply_manual_target_preview_adjustments(
            freq_axis,
            target_curve,
            preview_level_shift_db,
            preview_manual_target_tilt_db_per_oct,
        )
        target_curve_path = build_target_curve_path(freq_axis, target_curve_display)
        drag_base_points = []
        tilt_handle_points = []
        if is_manual_level and target_curve_path:
            drag_base_points = [
                (float(x), float(y))
                for x, y in zip(freq_axis, target_curve_display)
                if math.isfinite(float(x)) and math.isfinite(float(y))
            ]
            tilt_handle_y = float(np.interp(16000.0, freq_axis, target_curve_display))
            tilt_handle_points = parse_svg_path_points(build_tilt_handle_path(tilt_handle_y))

        # --- build figure ---
        fig = go.Figure()
        fig.add_trace(build_level_window_trace(lvl_min, lvl_max))
        fig.add_trace(build_vertical_marker_trace(mag_c_min))
        fig.add_trace(build_vertical_marker_trace(mag_c_max))

        target_trace = dict(
            x=freq_axis,
            y=target_curve_display,
            mode="lines",
            name=f"Target ({hc_mode_raw})",
            line=dict(color="#4caf50", width=2.0),
        )
        if is_manual_level:
            target_trace["opacity"] = 0.45
            target_trace["hoverinfo"] = "skip"
        fig.add_trace(go.Scatter(**target_trace))
        if is_manual_level:
            fig.add_trace(go.Scatter(
                x=[freq_axis[0], freq_axis[-1]],
                y=[lvl_manual_db, lvl_manual_db],
                mode="lines",
                name=f"Manual level ({lvl_manual_db:+.1f} dB)",
                line=dict(color="rgba(255,255,255,0.70)", width=1.2, dash="dot"),
                hoverinfo="skip",
                visible="legendonly",
            ))
            fig.add_trace(go.Scatter(
                x=[freq_axis[0], freq_axis[-1]],
                y=[0.0, 0.0],
                mode="lines",
                name=f"Manual tilt ({manual_target_tilt_db_per_oct:+.1f} dB/oct @ 1 kHz)",
                line=dict(color="#f4a261", width=1.8, dash="dot"),
                hoverinfo="skip",
                visible="legendonly",
            ))
        if "L" in speaker_interp:
            fig.add_trace(go.Scatter(
                x=freq_axis, y=speaker_interp["L"], mode="lines",
                name="Speaker L", line=dict(color="rgba(102,187,255,0.55)", width=1.2),
            ))
        if "R" in speaker_interp:
            fig.add_trace(go.Scatter(
                x=freq_axis, y=speaker_interp["R"], mode="lines",
                name="Speaker R", line=dict(color="rgba(255,167,102,0.55)", width=1.2),
            ))
        if len(speaker_interp) > 0:
            avg = np.mean(np.vstack([speaker_interp[k] for k in sorted(speaker_interp)]), axis=0)
            fig.add_trace(go.Scatter(
                x=freq_axis, y=avg, mode="lines",
                name="Speaker avg", line=dict(color="#ffd166", width=2.0),
            ))
        fig.update_xaxes(type="log", title_text="Hz",
                         range=[math.log10(10.0), math.log10(20000.0)], fixedrange=True)
        fig.update_yaxes(title_text="dB", range=[-10.0, 20.0], fixedrange=True)
        fig.update_layout(height=320, margin=dict(l=40, r=20, t=30, b=35),
                          showlegend=True, template="plotly_dark",
                          uirevision="target_preview_lock")
        if is_manual_level and target_curve_path:
            drag_shape = build_draggable_target_shape(freq_axis, target_curve_display)
            drag_shape["path"] = target_curve_path
            tilt_handle_shape = build_draggable_tilt_handle_shape(float(np.interp(16000.0, freq_axis, target_curve_display)))
            fig.update_layout(shapes=[drag_shape, tilt_handle_shape])

        fig_dict = fig.to_plotly_json()
        layout = fig_dict.setdefault("layout", {})
        layout["uirevision"] = "target_preview_lock"
        layout["editrevision"] = (
            f"target_preview_manual_{_TARGET_PREVIEW_REFRESH_TOKEN}_{int(_TARGET_PREVIEW_DRAG_ACTIVE)}"
            if is_manual_level else
            "target_preview_static"
        )
        fig_dict["config"] = {
            "responsive": True,
            "displayModeBar": False,
            "editable": False,
            "edits": {"shapePosition": bool(is_manual_level)},
            "doubleClick": False,
            "scrollZoom": False,
        }
        return fig_dict, drag_base_points, tilt_handle_points

    except Exception:
        import logging  # noqa: PLC0415
        logging.getLogger("CamillaFIR").warning("_build_target_preview_fig failed", exc_info=True)
        return None, [], []
