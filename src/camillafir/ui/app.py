import html
import logging
import json
import re

from pywebio import config
from pywebio.output import put_button, put_html, put_markdown, use_scope
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


def build_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float):
    g = globals()
    g["_PROCESS_RUN"] = process_run
    g["PROGRAM_NAME"] = PROGRAM_NAME
    g["VERSION"] = VERSION
    g["MAX_SAFE_BOOST"] = float(MAX_SAFE_BOOST)
    callbacks.configure_engine_hooks(process_run=process_run)
    return main


def _status_base_from_text(msg) -> str:
    try:
        s = str(msg or "").strip()
    except Exception:
        s = ""
    if not s:
        return "CamillaFIR running"
    # Remove trailing elapsed suffix like " | 123.4 s" for timer refresh.
    try:
        s = re.sub(r"\s*\|\s*\d+(?:\.\d+)?\s*s\s*$", "", s, flags=re.IGNORECASE)
    except Exception:
        pass
    s = str(s or "").strip()
    return s or "CamillaFIR running"


def get_status_base_message(default: str = "CamillaFIR running") -> str:
    try:
        v = str(_STATUS_BASE_MSG or "").strip()
    except Exception:
        v = ""
    return v or str(default)


def update_status(msg):
    global _STATUS_BASE_MSG, _STATUS_DOM_READY
    _STATUS_BASE_MSG = _status_base_from_text(msg)
    text = str(msg or "")

    if not _STATUS_DOM_READY:
        with use_scope("status_area", clear=True):
            safe_text = html.escape(text)
            put_html(
                f'<div id="cf_status_text" '
                f'style="font-weight:bold; color:#4CAF50; margin-bottom:10px;">'
                f"{safe_text}</div>"
            )
        _STATUS_DOM_READY = True
        return

    try:
        from pywebio.session import run_js
    except Exception:
        run_js = None

    if callable(run_js):
        try:
            run_js(
                "const el=document.getElementById('cf_status_text');"
                "if(el){el.textContent=%s;}" % json.dumps(text)
            )
            return
        except Exception:
            pass

    with use_scope("status_area", clear=True):
        safe_text = html.escape(text)
        put_html(
            f'<div id="cf_status_text" '
            f'style="font-weight:bold; color:#4CAF50; margin-bottom:10px;">'
            f"{safe_text}</div>"
        )
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
    )

    callbacks.register_callbacks(t=t, get_val=get_val)

    put_markdown("---")
    put_button("🚀 START", onclick=callbacks.on_start_click).style(
        """
        width: 100%;
        margin-top: 30px;
        padding: 15px;
        font-size: 24px;
        font-weight: 900;
        letter-spacing: 3px;

        background-color: transparent;
        border: none;
        color: #ffffff;

        transition: 0.3s;
        cursor: pointer;
    """
    )
