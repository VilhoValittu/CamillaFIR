"""NiceGUI Files tab builder.

Replaces build_input_section() from layout_builders.py.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Callable

from . import ng_controls as ctrl
from ..ui_i18n import LAYOUT_MONO, LAYOUT_OPTION_LABEL_KEYS, normalize_layout_value, tr_options


def _normalize_layout_value(value: Any, t: Callable[[str], str] | None = None) -> str:
    return normalize_layout_value(value, t)


def _guess_upload_format(file_data: dict[str, Any] | None) -> str:
    if not isinstance(file_data, dict):
        return "Unknown"
    name = str(file_data.get("filename", "") or "").strip().lower()
    content = file_data.get("content", b"")
    if name.endswith(".wav") or (
        isinstance(content, (bytes, bytearray)) and len(content) >= 4 and bytes(content[:4]) == b"RIFF"
    ):
        return "WAV"
    if name.endswith(".txt"):
        return "TXT"
    return "Unknown"


def _normalize_local_path_value(value: Any) -> str:
    try:
        return str(value or "").strip().strip('"').strip("'")
    except Exception:
        return ""


def _describe_local_path(path_raw: Any) -> dict[str, Any]:
    path = _normalize_local_path_value(path_raw)
    if not path:
        return {
            "entered": False,
            "exists": False,
            "path": "",
            "filename": "",
            "format": "Unknown",
            "size_bytes": 0,
        }

    try:
        exists = bool(os.path.isfile(path))
    except OSError:
        exists = False

    size_bytes = 0
    if exists:
        try:
            size_bytes = int(os.path.getsize(path))
        except OSError:
            size_bytes = 0

    return {
        "entered": True,
        "exists": exists,
        "path": path,
        "filename": os.path.basename(path) or path,
        "format": "WAV" if path.lower().endswith(".wav") else ("TXT" if path.lower().endswith(".txt") else "Unknown"),
        "size_bytes": size_bytes,
    }


def _format_upload_size(size_bytes: Any) -> str:
    try:
        size = float(size_bytes)
    except Exception:
        size = 0.0
    if size <= 0:
        return "0 KB"
    if size >= 1024.0 * 1024.0:
        return f"{size / (1024.0 * 1024.0):.2f} MB"
    return f"{size / 1024.0:.1f} KB"


def _build_upload_payload(*, filename: str, content: bytes, mime_type: str = "") -> dict[str, Any]:
    content_bytes = bytes(content or b"")
    return {
        "filename": filename,
        "content": content_bytes,
        "mime_type": str(mime_type or ""),
        "size_bytes": len(content_bytes),
        "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
    }


def build_files_tab(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui

    ui.markdown(f"### {t('tab_files')}")
    ui.separator()
    ui.markdown(t("wav_recommended_info"))
    ui.separator()
    ui.markdown(f"#### {t('input_files_title')}")
    ui.label(t("input_files_help")).classes("text-sm text-gray-400 -mt-2 mb-2")

    # File uploads – store as {"filename": ..., "content": bytes} in holders
    _file_l = ctrl._ValueHolder(get_val("file_l", None))
    _file_r = ctrl._ValueHolder(get_val("file_r", None))
    ctrl.register("file_l", _file_l)
    ctrl.register("file_r", _file_r)

    def _refresh_target_preview() -> None:
        try:
            from .ng_tab_target import refresh_target_preview  # noqa: PLC0415

            refresh_target_preview()
        except Exception:
            pass

    def _render_file_status(*, channel_label: str, holder, scope_name: str, path_key: str) -> None:
        scope = ctrl.get_container(scope_name)
        if scope is None:
            return

        file_data = holder.value if isinstance(holder.value, dict) else None
        upload_loaded = bool(file_data and file_data.get("content"))
        local_path_info = _describe_local_path(ctrl.value(path_key, ""))
        if upload_loaded:
            preview_source_text = t("file_status_preview_upload")
        elif local_path_info["entered"]:
            preview_source_text = t("file_status_preview_path")
        else:
            preview_source_text = t("file_status_preview_none")
        header_loaded = bool(upload_loaded or local_path_info["exists"])

        def _clear_uploaded_file() -> None:
            holder.value = None
            _render_file_status(channel_label=channel_label, holder=holder, scope_name=scope_name, path_key=path_key)
            _refresh_target_preview()

        scope.clear()
        with scope:
            with ui.card().classes("w-full gap-2"):
                with ui.row().classes("w-full items-start justify-between gap-3"):
                    with ui.column().classes("gap-1"):
                        ui.label(channel_label).classes("text-xs font-medium text-gray-400")
                        ui.label(
                            t("file_status_loaded") if header_loaded else t("file_status_not_loaded")
                        ).classes("text-sm font-semibold")
                    if upload_loaded:
                        ui.button(
                            t("file_status_clear"),
                            on_click=_clear_uploaded_file,
                        ).props('color="secondary" outline size="sm"')
                with ui.grid(columns=2).classes("w-full gap-x-3 gap-y-1"):
                    ui.label(t("file_status_preview_source")).classes("text-xs text-gray-400")
                    ui.label(preview_source_text).classes("text-xs")

                    if upload_loaded:
                        upload_format = _guess_upload_format(file_data)
                        if upload_format == "Unknown":
                            upload_format = t("file_status_unknown")
                        upload_size_bytes = int(file_data.get("size_bytes") or len(file_data.get("content", b"") or b""))

                        ui.label(t("file_status_upload")).classes("text-xs text-gray-400")
                        ui.label(str(file_data.get("filename", "") or t("health_not_set"))).classes("text-sm")
                        ui.label(t("file_status_format")).classes("text-xs text-gray-400")
                        ui.label(upload_format).classes("text-xs")
                        ui.label(t("file_status_size")).classes("text-xs text-gray-400")
                        ui.label(_format_upload_size(upload_size_bytes)).classes("text-xs")

                    if local_path_info["entered"]:
                        path_format = str(local_path_info["format"] or "Unknown")
                        if path_format == "Unknown":
                            path_format = t("file_status_unknown")

                        ui.label(t("file_status_local_path")).classes("text-xs text-gray-400")
                        ui.label(str(local_path_info["path"] or t("health_not_set"))).classes("text-xs break-all")
                        ui.label(t("file_status_on_disk")).classes("text-xs text-gray-400")
                        ui.label(
                            t("file_status_path_found") if local_path_info["exists"] else t("file_status_path_missing")
                        ).classes("text-xs")
                        ui.label(t("file_status_format")).classes("text-xs text-gray-400")
                        ui.label(path_format).classes("text-xs")
                        ui.label(t("file_status_size")).classes("text-xs text-gray-400")
                        ui.label(
                            _format_upload_size(local_path_info["size_bytes"])
                            if local_path_info["exists"]
                            else t("health_not_set")
                        ).classes("text-xs")

                    if not upload_loaded and not local_path_info["entered"]:
                        ui.label(t("file_status_name")).classes("text-xs text-gray-400")
                        ui.label(t("health_not_set")).classes("text-xs")
                        ui.label(t("file_status_format")).classes("text-xs text-gray-400")
                        ui.label(t("health_not_set")).classes("text-xs")
                        ui.label(t("file_status_size")).classes("text-xs text-gray-400")
                        ui.label(t("health_not_set")).classes("text-xs")

    async def _on_upload_l(e) -> None:
        _file_l.value = _build_upload_payload(
            filename=e.file.name,
            content=await e.file.read(),
            mime_type=getattr(e.file, "content_type", ""),
        )
        _render_file_status(channel_label=t("upload_l"), holder=_file_l, scope_name="file_l_status_scope", path_key="local_path_l")
        _refresh_target_preview()

    async def _on_upload_r(e) -> None:
        _file_r.value = _build_upload_payload(
            filename=e.file.name,
            content=await e.file.read(),
            mime_type=getattr(e.file, "content_type", ""),
        )
        _render_file_status(channel_label=t("upload_r"), holder=_file_r, scope_name="file_r_status_scope", path_key="local_path_r")
        _refresh_target_preview()

    with ui.row().classes("w-full gap-4"):
        with ui.column().classes("flex-1 gap-2"):
            ui.label(t("upload_l")).classes("text-sm font-medium")
            ui.upload(
                label=t("upload_l"),
                on_upload=_on_upload_l,
                auto_upload=True,
            ).props('accept=".txt,.wav"').classes("w-full")
            file_l_status_scope = ui.column().classes("w-full")
            ctrl.register_container("file_l_status_scope", file_l_status_scope)
        with ui.column().classes("flex-1 gap-2"):
            ui.label(t("upload_r")).classes("text-sm font-medium")
            ui.upload(
                label=t("upload_r"),
                on_upload=_on_upload_r,
                auto_upload=True,
            ).props('accept=".txt,.wav"').classes("w-full")
            file_r_status_scope = ui.column().classes("w-full")
            ctrl.register_container("file_r_status_scope", file_r_status_scope)

    # Local paths (collapsed)
    with ui.expansion(t("ui_local_paths_optional")).classes("w-full"):
        ctrl.register(
            "local_path_l",
            ui.input(label=t("path_l"), value=get_val("local_path_l", "")).classes("w-full"),
        )
        ctrl.register(
            "local_path_r",
            ui.input(label=t("path_r"), value=get_val("local_path_r", "")).classes("w-full"),
        )
    ctrl.on_change(
        "local_path_l",
        lambda v: _render_file_status(
            channel_label=t("upload_l"),
            holder=_file_l,
            scope_name="file_l_status_scope",
            path_key="local_path_l",
        ),
    )
    ctrl.on_change(
        "local_path_r",
        lambda v: _render_file_status(
            channel_label=t("upload_r"),
            holder=_file_r,
            scope_name="file_r_status_scope",
            path_key="local_path_r",
        ),
    )

    _render_file_status(channel_label=t("upload_l"), holder=_file_l, scope_name="file_l_status_scope", path_key="local_path_l")
    _render_file_status(channel_label=t("upload_r"), holder=_file_r, scope_name="file_r_status_scope", path_key="local_path_r")

    ui.separator()

    # Filter layout
    with ui.row().classes("w-full gap-4 items-end"):
        with ui.column().classes("gap-1"):
            ui.label(t("layout")).classes("text-sm font-medium")
            ctrl.register(
                "layout",
                ui.radio(
                    tr_options(t, LAYOUT_OPTION_LABEL_KEYS),
                    value=normalize_layout_value(get_val("layout", LAYOUT_MONO), t),
                ),
            )

    ui.separator()

    # Checkboxes
    ctrl.register(
        "multi_rate_opt",
        ui.checkbox(
            t("multi_rate"),
            value=bool(get_val("multi_rate_opt", False)),
        ),
    )
    ctrl.register(
        "comparison_mode",
        ui.checkbox(
            t("comparison_mode"),
            value=bool(get_val("comparison_mode", True)),
        ),
    )

    # Dynamic multi-rate info container (replaces taps_auto_info_scope_files)
    ctrl.register_container("taps_auto_info_scope_files", ui.column().classes("w-full"))
