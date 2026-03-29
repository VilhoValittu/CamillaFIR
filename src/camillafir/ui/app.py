import html
import logging
import json

from pywebio import config
from pywebio.output import put_html, use_scope
from pywebio.session import set_env

from ..config.camillafir_config import load_config
from ..resources.i8n.camillafir_i18n import t
from . import callbacks, layout_sections, ui_state

logger = logging.getLogger("CamillaFIR")

_PROCESS_RUN = None
PROGRAM_NAME = "CamillaFIR"
VERSION = ""
MAX_SAFE_BOOST = 8.0


def build_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float):
    g = globals()
    g["_PROCESS_RUN"] = process_run
    g["PROGRAM_NAME"] = PROGRAM_NAME
    g["VERSION"] = VERSION
    g["MAX_SAFE_BOOST"] = float(MAX_SAFE_BOOST)
    ui_state.set_status_renderer(_dispatch_status_state_event)
    callbacks.configure_engine_hooks(process_run=process_run)
    return main


def get_status_base_message(default: str = "CamillaFIR running") -> str:
    return ui_state.get_status_base_message(default=default)


def set_run_wall_clock_text(value) -> None:
    ui_state.set_run_wall_clock_text(value)


def get_run_wall_clock_text(default: str = "") -> str:
    return ui_state.get_run_wall_clock_text(default=default)


def update_status(msg):
    ui_state.update_status(msg)


def update_auto_selected_bar(msg):
    ui_state.update_auto_selected_bar(msg)


def update_status_notices(*, summary_text=None, info_text=None):
    ui_state.update_status_notices(summary_text=summary_text, info_text=info_text)


def reset_auto_status_details():
    ui_state.reset_auto_status_details()


def activate_tab(tab_title: str) -> None:
    try:
        from pywebio.session import run_js
    except ImportError:
        run_js = None

    if not callable(run_js):
        return

    try:
        run_js(
            """
            const wanted = String(title || '').trim();
            if (!wanted) return;
            const links = Array.from(document.querySelectorAll('.nav-tabs .nav-link'));
            const match = links.find((el) => (el.textContent || '').trim() === wanted);
            if (match) {
              match.click();
              match.scrollIntoView({behavior: 'smooth', block: 'nearest', inline: 'nearest'});
            }
            """,
            title=str(tab_title or ""),
        )
    except Exception:
        # Tab switching is a UI nicety; do not fail if JS hooks are unavailable in the current session.
        logger.debug("activate_tab JS hook failed for %r", tab_title, exc_info=True)
        return


def activate_run_tab() -> None:
    activate_tab(t("tab_run"))


def _render_status_area(snapshot: dict | None = None):
    snap = dict(snapshot or ui_state.get_status_snapshot() or {})
    safe_text = html.escape(str(snap.get("status_last_text", "") or ""))
    summary_txt = str(snap.get("status_summary_text", "") or "")
    summary_safe = html.escape(summary_txt)
    summary_display = "block" if summary_txt else "none"
    info_txt = str(snap.get("status_info_text", "") or "")
    info_safe = html.escape(info_txt)
    info_display = "block" if info_txt else "none"
    auto_txt = str(snap.get("auto_selected_bar_text", "") or "")
    auto_safe = html.escape(auto_txt)
    auto_display = "block" if auto_txt else "none"
    detail_body = str(snap.get("auto_status_detail_body", "") or "")
    detail_safe = html.escape(detail_body)
    detail_display = "block" if detail_body else "none"
    with use_scope("status_area", clear=True):
        put_html(
            f'<div id="cf_status_summary" '
            f'style="display:{summary_display}; margin:0 0 10px 0; padding:12px 14px; '
            f'border:1px solid rgba(34,197,94,0.28); border-radius:12px; '
            f'background:rgba(34,197,94,0.10); color:rgba(235,255,245,0.92); '
            f'font-weight:700;">{summary_safe}</div>'
            f'<div id="cf_status_info" '
            f'style="display:{info_display}; margin:0 0 10px 0; padding:12px 14px; '
            f'border:1px solid rgba(59,130,246,0.25); border-radius:12px; '
            f'background:rgba(59,130,246,0.10); color:rgba(235,245,255,0.92); '
            f'font-weight:650;">{info_safe}</div>'
            f'<div id="cf_status_text" '
            f'style="font-weight:bold; color:#4CAF50; margin-bottom:10px;">'
            f"{safe_text}</div>"
            f'<div id="cf_auto_selected_bar" '
            f'style="display:{auto_display}; margin-top:4px; padding:7px 10px; '
            f'border:1px solid rgba(76,175,80,0.55); border-radius:8px; '
            f'background:rgba(76,175,80,0.12); color:#d8f5de; font-weight:600;">'
            f"{auto_safe}</div>"
            f'<div id="cf_auto_status_details_wrap" style="display:{detail_display}; margin-top:8px;">'
            f'<details id="cf_auto_status_details" style="border:1px solid rgba(255,255,255,0.16); '
            f'border-radius:8px; padding:6px 10px; background:rgba(255,255,255,0.03);">'
            f'<summary style="cursor:pointer; font-weight:600; color:#cfd8e3;">'
            f'Automatic mode details</summary>'
            f'<div id="cf_auto_status_details_body" style="margin-top:8px; color:#b8c2d1; '
            f'font-size:12px; white-space:pre-wrap; line-height:1.35;">{detail_safe}</div>'
            f"</details></div>"
        )


def _dispatch_status_state_event(*, event: str, snapshot: dict | None = None) -> None:
    snap = dict(snapshot or ui_state.get_status_snapshot() or {})
    event_name = str(event or "")
    dom_ready = bool(snap.get("status_dom_ready", False))
    render_text = str(
        snap.get("status_last_text", "")
        or snap.get("status_base_message", "")
        or ""
    )
    if not dom_ready:
        if event_name != "status":
            return
        _render_status_area(snap)
        ui_state.mark_status_dom_ready(True)
        return

    try:
        from pywebio.session import run_js
    except ImportError:
        run_js = None

    if event_name == "status" and callable(run_js):
        try:
            run_js(
                "const summary=document.getElementById('cf_status_summary');"
                "if(summary){"
                "const txt=%s;"
                "summary.textContent=txt;"
                "summary.style.display=txt?'block':'none';"
                "}"
                "const info=document.getElementById('cf_status_info');"
                "if(info){"
                "const txt=%s;"
                "info.textContent=txt;"
                "info.style.display=txt?'block':'none';"
                "}"
                "const el=document.getElementById('cf_status_text');"
                "if(el){el.textContent=%s;}"
                "const wrap=document.getElementById('cf_auto_status_details_wrap');"
                "const body=document.getElementById('cf_auto_status_details_body');"
                "if(wrap&&body){"
                "const det=%s;"
                "body.textContent=det;"
                "wrap.style.display=det?'block':'none';"
                "}"
                % (
                    json.dumps(str(snap.get("status_summary_text", "") or "")),
                    json.dumps(str(snap.get("status_info_text", "") or "")),
                    json.dumps(render_text),
                    json.dumps(str(snap.get("auto_status_detail_body", "") or "")),
                )
            )
            return
        except Exception:
            # UI failsafe: status updates must not break the page if the DOM/session changed mid-run.
            logger.debug("Status JS update failed; falling back to full render", exc_info=True)
            pass

    if event_name == "auto_selected_bar":
        if not callable(run_js):
            _render_status_area(snap)
            ui_state.mark_status_dom_ready(True)
            return
        try:
            run_js(
                "const bar=document.getElementById('cf_auto_selected_bar');"
                "if(bar){"
                "const txt=%s;"
                "bar.textContent=txt;"
                "bar.style.display=txt?'block':'none';"
                "}"
                % json.dumps(str(snap.get("auto_selected_bar_text", "") or ""))
            )
            return
        except Exception:
            logger.debug("Auto-selected-bar JS update failed; falling back to full render", exc_info=True)
            pass

    if event_name == "status_notices":
        if not callable(run_js):
            _render_status_area(snap)
            ui_state.mark_status_dom_ready(True)
            return
        try:
            run_js(
                "const summary=document.getElementById('cf_status_summary');"
                "if(summary){"
                "const txt=%s;"
                "summary.textContent=txt;"
                "summary.style.display=txt?'block':'none';"
                "}"
                "const info=document.getElementById('cf_status_info');"
                "if(info){"
                "const txt=%s;"
                "info.textContent=txt;"
                "info.style.display=txt?'block':'none';"
                "}"
                % (
                    json.dumps(str(snap.get("status_summary_text", "") or "")),
                    json.dumps(str(snap.get("status_info_text", "") or "")),
                )
            )
            return
        except Exception:
            logger.debug("Status-notices JS update failed; falling back to full render", exc_info=True)
            pass

    if event_name == "reset_auto_status_details":
        if not callable(run_js):
            _render_status_area(snap)
            ui_state.mark_status_dom_ready(True)
            return
        try:
            run_js(
                "const wrap=document.getElementById('cf_auto_status_details_wrap');"
                "const body=document.getElementById('cf_auto_status_details_body');"
                "if(body){body.textContent='';}"
                "if(wrap){wrap.style.display='none';}"
            )
            return
        except Exception:
            logger.debug("Auto-status-details reset JS update failed; falling back to full render", exc_info=True)
            pass

    _render_status_area(snap)
    ui_state.mark_status_dom_ready(True)


@config(theme="dark")
def main():
    set_env(output_max_width="1850px")

    d = load_config()
    get_val = lambda k, def_v: d.get(k, def_v)

    layout_sections.build_header(t=t, version=VERSION)
    layout_sections.build_tabs(
        t=t,
        get_val=get_val,
        max_safe_boost=float(MAX_SAFE_BOOST),
        on_mode_apply_defaults=callbacks.on_mode_apply_defaults,
        on_afdw_preset=callbacks.on_afdw_preset,
        on_start_click=callbacks.on_start_click,
    )

    callbacks.register_callbacks(t=t, get_val=get_val)
