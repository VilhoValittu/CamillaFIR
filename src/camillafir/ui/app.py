import html
import logging
import json
import re

from pywebio import config
from pywebio.output import put_html, use_scope
from pywebio.session import set_env

from ..config.camillafir_config import load_config
from ..resources.i8n.camillafir_i18n import t
from . import callbacks, layout_sections

logger = logging.getLogger("CamillaFIR")

_PROCESS_RUN = None
PROGRAM_NAME = "CamillaFIR"
VERSION = ""
MAX_SAFE_BOOST = 8.0
_STATUS_BASE_MSG = ""
_STATUS_DOM_READY = False
_STATUS_LAST_TEXT = ""
_STATUS_SUMMARY_TEXT = ""
_STATUS_INFO_TEXT = ""
_AUTO_SELECTED_BAR_MSG = ""
_AUTO_STATUS_DETAILS: list[str] = []
_AUTO_STATUS_DETAIL_MAX = 80
_AUTO_STATUS_LAST_DETAIL = ""
_RUN_WALL_CLOCK_TEXT = ""


def build_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float):
    g = globals()
    g["_PROCESS_RUN"] = process_run
    g["PROGRAM_NAME"] = PROGRAM_NAME
    g["VERSION"] = VERSION
    g["MAX_SAFE_BOOST"] = float(MAX_SAFE_BOOST)
    callbacks.configure_engine_hooks(process_run=process_run)
    return main


def _status_split_elapsed_suffix(msg: str) -> tuple[str, str]:
    try:
        s = str(msg or "").strip()
    except Exception:
        s = ""
    if not s:
        return "", ""
    try:
        m = re.match(r"^(.*?)(\|\s*\d+(?:\.\d+)?\s*s)\s*$", s, flags=re.IGNORECASE)
    except Exception:
        m = None
    if not m:
        return str(s), ""
    return str(m.group(1) or "").strip(), str(m.group(2) or "").strip()


def _compact_auto_status_core(core: str) -> str:
    try:
        s = str(core or "").strip()
    except Exception:
        s = ""
    prefix = "CamillaFIR automatic mode:"
    if not s.startswith(prefix):
        return s
    after = str(s[len(prefix):] or "").strip()
    low = after.lower()
    if low.startswith("target shortlist"):
        phase = "target shortlist"
    elif low.startswith("target preselect"):
        phase = "target preselect"
    elif low.startswith("selecting target curve"):
        phase = "selecting target curve"
    elif low.startswith("target trials"):
        phase = "target trials"
    elif low.startswith("phase1 done"):
        phase = "phase1 done"
    elif low.startswith("local refine"):
        phase = "local refine"
    elif low.startswith("target finalize"):
        phase = "target finalize"
    elif low.startswith("preset search"):
        phase = "preset search"
    elif low.startswith("protection model"):
        phase = "protection model"
    elif low.startswith("hpf auto-fit"):
        phase = "hpf auto-fit"
    elif low.startswith("finalize"):
        phase = "finalize"
    elif low.startswith("target curve mode="):
        phase = "target curve mode"
    elif low.startswith("target preselect winner"):
        phase = "target preselect winner"
    elif low.startswith("init"):
        phase = "init"
    else:
        try:
            phase = re.sub(r"\s*\(.*\)\s*$", "", after).strip()
        except Exception:
            phase = after
        if not phase:
            phase = "running"
    return f"{prefix} {phase}".strip()


def _status_compact_with_detail(msg) -> tuple[str, str | None]:
    try:
        raw = str(msg or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return "CamillaFIR running", None
    core, elapsed = _status_split_elapsed_suffix(raw)
    compact_core = _compact_auto_status_core(core)
    out = compact_core
    if elapsed:
        out = f"{compact_core} {elapsed}"
    detail = None
    if (
        isinstance(core, str)
        and core.startswith("CamillaFIR automatic mode:")
        and str(core).strip() != str(compact_core).strip()
    ):
        detail = str(core).strip()
    return out, detail


def _status_base_from_text(msg) -> str:
    text, _detail = _status_compact_with_detail(msg)
    core, _elapsed = _status_split_elapsed_suffix(text)
    core = str(core or "").strip()
    return core or "CamillaFIR running"


def get_status_base_message(default: str = "CamillaFIR running") -> str:
    try:
        v = str(_STATUS_BASE_MSG or "").strip()
    except Exception:
        v = ""
    return v or str(default)


def set_run_wall_clock_text(value) -> None:
    global _RUN_WALL_CLOCK_TEXT
    try:
        _RUN_WALL_CLOCK_TEXT = str(value or "").strip()
    except Exception:
        _RUN_WALL_CLOCK_TEXT = ""


def get_run_wall_clock_text(default: str = "") -> str:
    try:
        v = str(_RUN_WALL_CLOCK_TEXT or "").strip()
    except Exception:
        v = ""
    return v or str(default or "")


def activate_tab(tab_title: str) -> None:
    try:
        from pywebio.session import run_js
    except Exception:
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
        return


def activate_run_tab() -> None:
    activate_tab(t("tab_run"))


def _normalize_auto_selected_text(msg) -> str:
    try:
        txt = str(msg or "").strip()
    except Exception:
        txt = ""
    if not txt:
        return ""
    return txt


def _normalize_status_notice_text(msg) -> str:
    try:
        txt = str(msg or "").strip()
    except Exception:
        txt = ""
    return txt


def _render_status_area(text: str):
    safe_text = html.escape(str(text or ""))
    summary_txt = _normalize_status_notice_text(_STATUS_SUMMARY_TEXT)
    summary_safe = html.escape(summary_txt)
    summary_display = "block" if summary_txt else "none"
    info_txt = _normalize_status_notice_text(_STATUS_INFO_TEXT)
    info_safe = html.escape(info_txt)
    info_display = "block" if info_txt else "none"
    auto_txt = _normalize_auto_selected_text(_AUTO_SELECTED_BAR_MSG)
    auto_safe = html.escape(auto_txt)
    auto_display = "block" if auto_txt else "none"
    detail_body = "\n".join(str(x or "") for x in list(_AUTO_STATUS_DETAILS or []))
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


def update_status(msg):
    global _STATUS_BASE_MSG, _STATUS_DOM_READY, _STATUS_LAST_TEXT, _AUTO_STATUS_DETAILS, _AUTO_STATUS_LAST_DETAIL
    text, detail = _status_compact_with_detail(msg)
    _STATUS_BASE_MSG = _status_base_from_text(text)
    _STATUS_LAST_TEXT = str(text or "")
    if isinstance(detail, str) and detail.strip():
        d = str(detail).strip()
        if d != str(_AUTO_STATUS_LAST_DETAIL or ""):
            _AUTO_STATUS_LAST_DETAIL = d
            _AUTO_STATUS_DETAILS = list(_AUTO_STATUS_DETAILS or []) + [d]
            max_n = int(max(10, _AUTO_STATUS_DETAIL_MAX))
            if len(_AUTO_STATUS_DETAILS) > max_n:
                _AUTO_STATUS_DETAILS = list(_AUTO_STATUS_DETAILS[-max_n:])

    if not _STATUS_DOM_READY:
        _render_status_area(_STATUS_LAST_TEXT)
        _STATUS_DOM_READY = True
        return

    try:
        from pywebio.session import run_js
    except Exception:
        run_js = None

    if callable(run_js):
        try:
            detail_body = "\n".join(str(x or "") for x in list(_AUTO_STATUS_DETAILS or []))
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
                    json.dumps(str(_STATUS_SUMMARY_TEXT or "")),
                    json.dumps(str(_STATUS_INFO_TEXT or "")),
                    json.dumps(_STATUS_LAST_TEXT),
                    json.dumps(detail_body),
                )
            )
            return
        except Exception:
            pass

    _render_status_area(_STATUS_LAST_TEXT)
    _STATUS_DOM_READY = True


def update_auto_selected_bar(msg):
    global _AUTO_SELECTED_BAR_MSG, _STATUS_DOM_READY
    _AUTO_SELECTED_BAR_MSG = _normalize_auto_selected_text(msg)

    if not _STATUS_DOM_READY:
        return

    try:
        from pywebio.session import run_js
    except Exception:
        run_js = None

    if callable(run_js):
        try:
            run_js(
                "const bar=document.getElementById('cf_auto_selected_bar');"
                "if(bar){"
                "const txt=%s;"
                "bar.textContent=txt;"
                "bar.style.display=txt?'block':'none';"
                "}"
                % json.dumps(str(_AUTO_SELECTED_BAR_MSG or ""))
            )
            return
        except Exception:
            pass

    _render_status_area(str(_STATUS_LAST_TEXT or _STATUS_BASE_MSG or ""))
    _STATUS_DOM_READY = True


def update_status_notices(*, summary_text=None, info_text=None):
    global _STATUS_SUMMARY_TEXT, _STATUS_INFO_TEXT, _STATUS_DOM_READY
    if summary_text is not None:
        _STATUS_SUMMARY_TEXT = _normalize_status_notice_text(summary_text)
    if info_text is not None:
        _STATUS_INFO_TEXT = _normalize_status_notice_text(info_text)

    if not _STATUS_DOM_READY:
        return

    try:
        from pywebio.session import run_js
    except Exception:
        run_js = None

    if callable(run_js):
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
                    json.dumps(str(_STATUS_SUMMARY_TEXT or "")),
                    json.dumps(str(_STATUS_INFO_TEXT or "")),
                )
            )
            return
        except Exception:
            pass

    _render_status_area(str(_STATUS_LAST_TEXT or _STATUS_BASE_MSG or ""))
    _STATUS_DOM_READY = True


def reset_auto_status_details():
    global _AUTO_STATUS_DETAILS, _AUTO_STATUS_LAST_DETAIL, _STATUS_DOM_READY
    _AUTO_STATUS_DETAILS = []
    _AUTO_STATUS_LAST_DETAIL = ""
    if not _STATUS_DOM_READY:
        return
    try:
        from pywebio.session import run_js
    except Exception:
        run_js = None
    if callable(run_js):
        try:
            run_js(
                "const wrap=document.getElementById('cf_auto_status_details_wrap');"
                "const body=document.getElementById('cf_auto_status_details_body');"
                "if(body){body.textContent='';}"
                "if(wrap){wrap.style.display='none';}"
            )
            return
        except Exception:
            pass
    _render_status_area(str(_STATUS_LAST_TEXT or _STATUS_BASE_MSG or ""))
    _STATUS_DOM_READY = True


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
