import logging

from pywebio import config
from pywebio.output import put_button, put_markdown, put_text, use_scope
from pywebio.session import set_env

from ..config.camillafir_config import load_config
from ..resources.i8n.camillafir_i18n import t
from . import callbacks, layout_sections

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
    callbacks.configure_engine_hooks(process_run=process_run)
    return main


def update_status(msg):
    with use_scope("status_area", clear=True):
        put_text(msg).style("font-weight: bold; color: #4CAF50; margin-bottom: 10px;")


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
