import os
import sys

if __package__ in (None, ""):
    _pkg_root = os.path.dirname(os.path.abspath(__file__))
    _src_root = os.path.dirname(_pkg_root)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    __package__ = "camillafir"

from .bootstrap import initialize_logging  # noqa: E402
from .version import VERSION as APP_VERSION  # noqa: E402

_BOOTSTRAP = initialize_logging()
logger = _BOOTSTRAP["logger"]

VERSION = APP_VERSION
PROGRAM_NAME = "CamillaFIR"
MAX_SAFE_BOOST = 8.0
FORCE_SINGLE_PLOT_FS_HZ = 48000
MAX_SAFE_TAPS = 131072
TEST_MODE = os.environ.get("CAMILLAFIR_TEST", "0") == "1"

_MAIN_APP_CONFIGURED = False


def resolve_static_dir() -> str | None:
    candidates = []

    if hasattr(sys, "_MEIPASS"):
        try:
            candidates.append(os.path.join(sys._MEIPASS, "assets"))  # type: ignore[attr-defined]
        except Exception:
            pass

    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "plotly"))

    for d in candidates:
        try:
            if d and os.path.isfile(os.path.join(d, "plotly.min.js")):
                return d
        except Exception:
            continue
    return None


def process_run():
    from .application.request_builder import build_run_request_from_pin
    from .auto_mode.api import AUTO_MODE_COMPAT_VERSION
    from .ui.ng_controls import NgPinProxy

    from .workflow.process_run_flow import ProcessRunSupport, run_process_flow
    from .workflow.process_support import (
        auto_target_mode_norm as _auto_target_mode_norm,
        auto_target_selection_method_text as _auto_target_selection_method_text,
        has_uploaded_target_file as _has_uploaded_target_file,
        pick_target_curve_label as _pick_target_curve_label,
        slugify_filename_token as _slugify_filename_token,
    )
    from .ui.ng_bridge import build_default_ui_bridge

    request = build_run_request_from_pin(
        NgPinProxy(),
        version=str(VERSION),
        auto_mode_compat_version=str(AUTO_MODE_COMPAT_VERSION),
    )

    return run_process_flow(
        request=request,
        support=ProcessRunSupport(
            version=str(VERSION),
            max_safe_boost=float(MAX_SAFE_BOOST),
            force_single_plot_fs_hz=int(FORCE_SINGLE_PLOT_FS_HZ),
            auto_target_mode_norm=_auto_target_mode_norm,
            auto_target_selection_method_text=_auto_target_selection_method_text,
            pick_target_curve_label=_pick_target_curve_label,
            slugify_filename_token=_slugify_filename_token,
            has_uploaded_target_file=_has_uploaded_target_file,
            ui_bridge=build_default_ui_bridge(),
        ),
    )


def configure_main_app() -> None:
    global _MAIN_APP_CONFIGURED
    if _MAIN_APP_CONFIGURED:
        return

    from .ui.ng_app import configure_app as _configure_ui_app

    _configure_ui_app(
        process_run=process_run,
        PROGRAM_NAME=PROGRAM_NAME,
        VERSION=VERSION,
        MAX_SAFE_BOOST=MAX_SAFE_BOOST,
    )
    _MAIN_APP_CONFIGURED = True
