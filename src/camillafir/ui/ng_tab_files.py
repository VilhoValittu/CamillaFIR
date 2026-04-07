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
    mode_value = str(get_val("mode", "BASIC") or "BASIC").strip().upper()
    if bool(get_val("camillafir_automatic_mode", False)):
        mode_value = "AUTO"
    bass_integration_visible = bool(mode_value == "AUTO")
    bass_integration_enabled = bool(get_val("bass_integration_enable", False))
    bass_integration_active = bool(bass_integration_visible and bass_integration_enabled)
    is_direct_dac = (
        bass_integration_active
        and str(get_val("bass_integration_mode", "") or "").strip() == "direct_dac"
    )

    ui.markdown(f"### {t('tab_files')}")
    ui.separator()
    ui.markdown(t("wav_recommended_info"))
    ui.separator()
    ui.markdown(f"#### {t('input_files_title')}")
    ui.label(t("input_files_help")).classes("text-sm text-gray-400 -mt-2 mb-2")

    # File uploads – store as {"filename": ..., "content": bytes} in holders
    file_holders = {
        "file_l": ctrl._ValueHolder(get_val("file_l", None)),
        "file_r": ctrl._ValueHolder(get_val("file_r", None)),
        "file_l_main": ctrl._ValueHolder(get_val("file_l_main", None)),
        "file_r_main": ctrl._ValueHolder(get_val("file_r_main", None)),
        "file_l_sub": ctrl._ValueHolder(get_val("file_l_sub", None)),
        "file_r_sub": ctrl._ValueHolder(get_val("file_r_sub", None)),
    }
    for key, holder in file_holders.items():
        ctrl.register(key, holder)

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

    async def _handle_upload(e, *, upload_key: str, channel_label: str, scope_name: str, path_key: str) -> None:
        file_holders[upload_key].value = _build_upload_payload(
            filename=e.file.name,
            content=await e.file.read(),
            mime_type=getattr(e.file, "content_type", ""),
        )
        _render_file_status(
            channel_label=channel_label,
            holder=file_holders[upload_key],
            scope_name=scope_name,
            path_key=path_key,
        )
        _refresh_target_preview()

    def _build_measurement_slot(
        *,
        upload_key: str,
        path_key: str,
        channel_label_key: str,
        path_label_key: str,
    ) -> None:
        channel_label = t(channel_label_key)
        scope_name = f"{upload_key}_status_scope"

        with ui.column().classes("flex-1 gap-2"):
            ui.label(channel_label).classes("text-sm font-medium")

            async def _on_upload(
                e,
                *,
                _upload_key=upload_key,
                _channel_label=channel_label,
                _scope_name=scope_name,
                _path_key=path_key,
            ) -> None:
                await _handle_upload(
                    e,
                    upload_key=_upload_key,
                    channel_label=_channel_label,
                    scope_name=_scope_name,
                    path_key=_path_key,
                )

            ui.upload(
                label=channel_label,
                on_upload=_on_upload,
                auto_upload=True,
            ).props('accept=".txt,.wav"').classes("w-full")
            status_scope = ui.column().classes("w-full")
            ctrl.register_container(scope_name, status_scope)
            ctrl.register(
                path_key,
                ui.input(label=t(path_label_key), value=get_val(path_key, "")).classes("w-full"),
            )
        ctrl.on_change(
            path_key,
            lambda v, _label=channel_label, _holder=file_holders[upload_key], _scope_name=scope_name, _path_key=path_key: _render_file_status(
                channel_label=_label,
                holder=_holder,
                scope_name=_scope_name,
                path_key=_path_key,
            ),
        )
        _render_file_status(
            channel_label=channel_label,
            holder=file_holders[upload_key],
            scope_name=scope_name,
            path_key=path_key,
        )

    with ui.column().classes("w-full gap-4") as legacy_scope:
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l",
                path_key="local_path_l",
                channel_label_key="upload_l",
                path_label_key="path_l",
            )
            _build_measurement_slot(
                upload_key="file_r",
                path_key="local_path_r",
                channel_label_key="upload_r",
                path_label_key="path_r",
            )
    ctrl.register_container("files_legacy_topology_scope", legacy_scope)
    legacy_scope.set_visibility(not bass_integration_active)

    with ui.column().classes("w-full gap-4") as bi_scope:
        ui.label(t("bass_integration_requires_wav")).classes("text-xs text-gray-400")
        ui.label(t("bass_integration_wav_format")).classes("text-xs text-gray-400")
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l_main",
                path_key="local_path_l_main",
                channel_label_key="upload_l_main",
                path_label_key="path_l_main",
            )
            _build_measurement_slot(
                upload_key="file_r_main",
                path_key="local_path_r_main",
                channel_label_key="upload_r_main",
                path_label_key="path_r_main",
            )
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l_sub",
                path_key="local_path_l_sub",
                channel_label_key="upload_l_sub",
                path_label_key="path_l_sub",
            )
            _build_measurement_slot(
                upload_key="file_r_sub",
                path_key="local_path_r_sub",
                channel_label_key="upload_r_sub",
                path_label_key="path_r_sub",
            )
    ctrl.register_container("files_bass_integration_topology_scope", bi_scope)
    bi_scope.set_visibility(bass_integration_active and not is_direct_dac)

    with ui.column().classes("w-full gap-4") as direct_dac_scope:
        ui.label(t("bi_direct_sub_help")).classes("text-xs text-gray-400")
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l_main",
                path_key="local_path_l_main",
                channel_label_key="upload_l_main",
                path_label_key="path_l_main",
            )
            _build_measurement_slot(
                upload_key="file_r_main",
                path_key="local_path_r_main",
                channel_label_key="upload_r_main",
                path_label_key="path_r_main",
            )
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l_sub",
                path_key="local_path_l_sub",
                channel_label_key="upload_l_sub",
                path_label_key="path_l_sub",
            )
            _build_measurement_slot(
                upload_key="file_r_sub",
                path_key="local_path_r_sub",
                channel_label_key="upload_r_sub",
                path_label_key="path_r_sub",
            )
    ctrl.register_container("files_direct_dac_topology_scope", direct_dac_scope)
    direct_dac_scope.set_visibility(bool(bass_integration_active and is_direct_dac))

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
