import base64
import importlib.resources as pkgres

from pywebio.output import put_collapse, put_html, put_markdown

from .camillafir_ui_helpers import put_guide_section
from .layout_builders import (
    build_advanced_section,
    build_export_section,
    build_filter_section,
    build_input_section,
    build_results_section,
    build_run_section,
    build_tabs,
    build_target_section,
)
from .layout_theme import _inject_dark_css


def _put_logo_header(version: str):
    with pkgres.files("camillafir.ui.assets").joinpath("camillafir_logo.png").open("rb") as f:
        img_bytes = f.read()

    img_b64 = base64.b64encode(img_bytes).decode()

    put_html(
        f"""
<div style="
    display:flex;
    align-items:center;
    gap:18px;
    margin: 18px 0 18px 0;
">
    <div class="cf-brand-logo-wrap">
        <img src="data:image/png;base64,{img_b64}"
             class="cf-brand-logo"
             style="height:96px; width:auto;" />
    </div>

    <div style="display:flex; flex-direction:column;">
        <div style="
            font-size:32px;
            font-weight:650;
            color:#ffffff;
            line-height:1.1;
        ">
            CamillaFIR
        </div>

        <div style="
            font-size:14px;
            color:#9aa4b2;
            margin-top:6px;
            letter-spacing:0.4px;
        ">
            {version}
        </div>
    </div>
</div>
"""
    )


def build_header(*, t, version: str):
    _inject_dark_css()
    put_html(
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
    _put_logo_header(version)
    put_html('<hr style="border:1px solid #1f2937; margin: 0 0 14px 0;">')
    put_collapse(
        t("about_title"),
        [put_markdown(t("about_body"))],
    )
    put_guide_section()
    put_markdown("---")


__all__ = [
    "_inject_dark_css",
    "build_header",
    "build_input_section",
    "build_filter_section",
    "build_target_section",
    "build_advanced_section",
    "build_export_section",
    "build_results_section",
    "build_run_section",
    "build_tabs",
]
