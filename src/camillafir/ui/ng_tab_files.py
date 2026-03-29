"""NiceGUI Files tab builder.

Replaces build_input_section() from layout_builders.py.
"""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl


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

    async def _on_upload_l(e) -> None:
        _file_l.value = {
            "filename": e.file.name,
            "content": await e.file.read(),
            "mime_type": getattr(e.file, "content_type", ""),
        }
        try:
            from .ng_tab_target import refresh_target_preview  # noqa: PLC0415
            refresh_target_preview()
        except Exception:
            pass

    async def _on_upload_r(e) -> None:
        _file_r.value = {
            "filename": e.file.name,
            "content": await e.file.read(),
            "mime_type": getattr(e.file, "content_type", ""),
        }
        try:
            from .ng_tab_target import refresh_target_preview  # noqa: PLC0415
            refresh_target_preview()
        except Exception:
            pass

    with ui.row().classes("w-full gap-4"):
        with ui.column().classes("flex-1"):
            ui.label(t("upload_l")).classes("text-sm font-medium")
            ui.upload(
                label=t("upload_l"),
                on_upload=_on_upload_l,
                auto_upload=True,
            ).props('accept=".txt,.wav"').classes("w-full")
        with ui.column().classes("flex-1"):
            ui.label(t("upload_r")).classes("text-sm font-medium")
            ui.upload(
                label=t("upload_r"),
                on_upload=_on_upload_r,
                auto_upload=True,
            ).props('accept=".txt,.wav"').classes("w-full")

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

    ui.separator()

    # Filter layout
    with ui.row().classes("w-full gap-4 items-end"):
        with ui.column().classes("gap-1"):
            ui.label(t("layout")).classes("text-sm font-medium")
            ctrl.register(
                "layout",
                ui.radio(
                    [t("layout_mono"), t("layout_stereo")],
                    value=get_val("layout", t("layout_mono")),
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
