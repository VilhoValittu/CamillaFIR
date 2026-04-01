"""NiceGUI app configuration for CamillaFIR.

Public API:
    configure_app(*, process_run, PROGRAM_NAME, VERSION, MAX_SAFE_BOOST)
    build_app(*, process_run, PROGRAM_NAME, VERSION, MAX_SAFE_BOOST)
    update_status(msg)
    update_status_notices(*, summary_text, info_text)
    update_auto_selected_bar(msg)
"""
from __future__ import annotations

import base64
import importlib.resources as pkgres
import logging
import sys
from pathlib import Path

from ..config.camillafir_config import load_config
from ..resources.i8n.camillafir_i18n import t
from . import ui_state
from .ng_theme import apply_theme

logger = logging.getLogger("CamillaFIR")

_PROCESS_RUN = None
PROGRAM_NAME = "CamillaFIR"
VERSION = ""
MAX_SAFE_BOOST = 8.0


def _external_link_head_html() -> str:
    return (
        "<script>"
        "document.addEventListener('click',function(e){"
        "var a=e.target.closest('a[href]');"
        "if(a&&/^https?:\\/\\//i.test(a.getAttribute('href'))){"
        "e.preventDefault();e.stopPropagation();"
        "window.open(a.href,'_blank','noopener,noreferrer');"
        "}"
        "});"
        "</script>"
    )


def _load_user_manual_text() -> str:
    manual_path = _resolve_user_manual_path()
    if manual_path is None:
        logger.warning("User manual not found. Tried: %s", ", ".join(str(p) for p in _user_manual_path_candidates()))
        return "_User Manual not found._"
    try:
        return manual_path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Failed to read user manual: %s", manual_path)
        return "_User Manual not available._"


def _user_manual_path_candidates() -> list[Path]:
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        try:
            candidates.append(Path(sys._MEIPASS) / "docs" / "User_Manual.md")  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to resolve bundled user manual path")
    candidates.append(Path(__file__).resolve().parents[3] / "docs" / "User_Manual.md")
    return candidates


def _resolve_user_manual_path() -> Path | None:
    for path in _user_manual_path_candidates():
        try:
            if path.is_file():
                return path
        except OSError:
            logger.exception("Failed to inspect user manual path: %s", path)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float) -> None:
    """Wire runtime globals and register the NiceGUI main page."""
    g = globals()
    g["_PROCESS_RUN"] = process_run
    g["PROGRAM_NAME"] = PROGRAM_NAME
    g["VERSION"] = VERSION
    g["MAX_SAFE_BOOST"] = float(MAX_SAFE_BOOST)
    register_main_page()


def build_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float):
    """Backward-compatible wrapper around the current NiceGUI app setup."""
    configure_app(
        process_run=process_run,
        PROGRAM_NAME=PROGRAM_NAME,
        VERSION=VERSION,
        MAX_SAFE_BOOST=MAX_SAFE_BOOST,
    )
    return lambda *a, **kw: None


def update_status(msg) -> None:
    ui_state.update_status(msg)


def update_status_notices(*, summary_text=None, info_text=None) -> None:
    ui_state.update_status_notices(summary_text=summary_text, info_text=info_text)


def update_auto_selected_bar(msg) -> None:
    ui_state.update_auto_selected_bar(msg)


def get_status_base_message(default: str = "CamillaFIR running") -> str:
    return ui_state.get_status_base_message(default=default)


def set_run_wall_clock_text(value) -> None:
    ui_state.set_run_wall_clock_text(value)


def get_run_wall_clock_text(default: str = "") -> str:
    return ui_state.get_run_wall_clock_text(default=default)


# ---------------------------------------------------------------------------
# Page registration
# ---------------------------------------------------------------------------

def register_main_page() -> None:
    from nicegui import ui
    from . import ng_controls

    @ui.page("/")
    def _page() -> None:
        ng_controls.reset()
        apply_theme()

        d = load_config()
        get_val = lambda k, def_v: d.get(k, def_v)

        with ui.column().classes("w-full gap-0 cf-brand-shell"):
            _build_brand_header(version=VERSION)

        with ui.column().classes("w-full gap-0 cf-tabs-shell"):
            from .ng_run_section import build_global_progress_bar  # noqa: PLC0415

            build_global_progress_bar()
            with ui.tabs().classes("w-full") as tabs:
                tab_files = ui.tab(t("tab_files"))
                tab_basic = ui.tab(t("tab_basic"))
                tab_target = ui.tab(t("tab_target"))
                tab_advanced = ui.tab(t("tab_adv"))
                tab_export = ui.tab(t("tab_window_tdc"))
                tab_xo = ui.tab(t("tab_xo"))
                tab_run = ui.tab(t("tab_run"))

        with ui.tab_panels(tabs, value=tab_files).classes("w-full"):
            with ui.tab_panel(tab_files):
                from .ng_tab_files import build_files_tab  # noqa: PLC0415

                build_files_tab(t=t, get_val=get_val)

            with ui.tab_panel(tab_basic):
                from .ng_tab_basic import build_basic_tab  # noqa: PLC0415

                build_basic_tab(t=t, get_val=get_val, max_safe_boost=float(MAX_SAFE_BOOST))

            with ui.tab_panel(tab_target):
                from .ng_tab_target import build_target_tab  # noqa: PLC0415

                build_target_tab(t=t, get_val=get_val)

            with ui.tab_panel(tab_advanced):
                from .ng_tab_advanced import build_advanced_tab  # noqa: PLC0415

                build_advanced_tab(t=t, get_val=get_val, max_safe_boost=float(MAX_SAFE_BOOST))

            with ui.tab_panel(tab_export):
                from .ng_tab_window import build_window_tab  # noqa: PLC0415

                build_window_tab(t=t, get_val=get_val)

            with ui.tab_panel(tab_xo):
                from .ng_tab_xo import build_xo_tab  # noqa: PLC0415

                build_xo_tab(t=t, get_val=get_val)

            with ui.tab_panel(tab_run):
                from .ng_run_section import build_run_section  # noqa: PLC0415

                build_run_section(on_start_click=_on_start_click)

        # Register cross-tab callbacks after all elements exist.
        from .ng_callbacks import register_callbacks  # noqa: PLC0415

        register_callbacks(t=t, get_val=get_val, max_safe_boost=float(MAX_SAFE_BOOST))


# ---------------------------------------------------------------------------
# Start handler
# ---------------------------------------------------------------------------

def _on_start_click() -> None:
    """Called from ng_run_section in a background thread."""
    if not callable(_PROCESS_RUN):
        logger.warning("_on_start_click: _PROCESS_RUN not configured")
        return
    try:
        _PROCESS_RUN()
    except Exception:
        logger.exception("process_run raised")


# ---------------------------------------------------------------------------
# Header rendering
# ---------------------------------------------------------------------------

def _build_brand_header(*, version: str) -> None:
    from nicegui import ui

    ui.add_head_html(_external_link_head_html())
    manual_text = _load_user_manual_text()

    try:
        with pkgres.files("camillafir.ui.assets").joinpath("camillafir_logo.png").open("rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        logo_html = (
            f'<img src="data:image/png;base64,{img_b64}" '
            f'style="height:80px; width:auto;" />'
        )
    except Exception:
        logo_html = ""

    with ui.row().classes("items-center gap-4 my-4 w-full justify-between"):
        with ui.row().classes("items-center gap-4"):
            if logo_html:
                ui.html(logo_html)
            with ui.column().classes("gap-0"):
                ui.label("CamillaFIR").classes("text-3xl font-bold tracking-wide text-white")
                ui.label(version).classes("text-sm text-gray-400 mt-1")
        from .ng_run_section import build_info_panel  # noqa: PLC0415

        build_info_panel()

    ui.separator()
    # About / guide (collapsed by default)
    with ui.expansion(t("about_title")).classes("w-full"):
        ui.markdown(t("about_body"))
        with ui.dialog() as manual_dlg, ui.card().classes("w-full max-w-3xl max-h-[80vh] overflow-y-auto cf-modal-card"):
            ui.markdown(manual_text).classes("w-full")
            ui.button(t("manual_close_btn"), on_click=manual_dlg.close).props("flat")
        ui.button(t("open_manual_btn"), on_click=manual_dlg.open).props("flat").classes("mt-2")

    ui.separator()
