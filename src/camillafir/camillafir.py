import os
import sys

if __package__ in (None, ""):
    _pkg_root = os.path.dirname(os.path.abspath(__file__))
    _src_root = os.path.dirname(_pkg_root)
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)
    __package__ = "camillafir"


def _auto_thread_budget() -> tuple[int, int]:
    """
    Valitsee automaattisen thread-budjetin: noin 50 % CPU-ytimista.

    Esim. 4 -> 3, 8 -> 6.
    """
    try:
        cores = int(os.cpu_count() or 1)
    except Exception:
        cores = 1
    cores = max(1, int(cores))
    use = max(1, int((cores * 2) // 4))
    return int(use), int(cores)


def _apply_auto_thread_env() -> tuple[int, int, list[str]]:
    """
    Asettaa numeeristen kirjastojen thread-rajoitukset, jos niita ei ole
    eksplisiittisesti asetettu ymparistomuuttujilla.
    """
    use, cores = _auto_thread_budget()
    keys = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    applied: list[str] = []
    for k in keys:
        cur = str(os.environ.get(k, "") or "").strip()
        if cur:
            try:
                if int(float(cur)) > 0:
                    continue
            except Exception:
                pass
        os.environ[k] = str(use)
        applied.append(str(k))
    return int(use), int(cores), list(applied)


_AUTO_THREADS_USE, _AUTO_THREADS_CORES, _AUTO_THREADS_ENV_APPLIED = _apply_auto_thread_env()


def _resolve_console_log_level() -> int:
    """
    Normalikäytössä konsoli pidetään hiljaisena.

    Oletus on `WARNING`, mutta kehityslokit voi ottaa takaisin kayttoon
    ymparistomuuttujalla `CAMILLAFIR_LOG_LEVEL` (esim. INFO tai DEBUG).
    """
    raw = str(os.environ.get("CAMILLAFIR_LOG_LEVEL", "WARNING") or "WARNING").strip().upper()
    return int(getattr(logging, raw, logging.WARNING))

import logging  # noqa: E402
import re   # noqa: E402
from datetime import datetime   # noqa: E402
import numpy as np  # noqa: E402
from .config.camillafir_config import save_config   # noqa: E402
from .resources.i8n.camillafir_i18n import t    # noqa: E402
from .version import VERSION as APP_VERSION  # noqa: E402
from camillafir.io.measurements_loader import load_measurements_lr  # noqa: E402
from camillafir.io.measurements_txt import parse_measurements_from_path # noqa: E402
from .config.camillafir_pipeline import (   # noqa: E402
    collect_ui_data,
    log_df_smoothing_toggle,
    build_xos_hpf,
    filter_type_short,
    choose_target_rates,
    choose_dash_fs,
    detect_is_wav_source,
)
from .common.result_postprocess import (  # noqa: E402
    _ensure_scoring_keys,
    _inject_filter_mags_for_ui,
    _irwin_tag,
    _postpolish_wav_filter_ir,
    _shift_zeropad_1d,
)
from .engine import build_config, run_pipeline, summarize_run   # noqa: E402
from .dsp import camillafir_dsp as dsp  # noqa: E402
from .dsp.target_match import target_match_from_stats  # noqa: E402
from .io.camillafir_automatic_mode import (  # noqa: E402
    AUTO_MODE_COMPAT_VERSION,
    AUTO_MODE_EXC_FROM_F6_ADD_HZ,
    AUTO_MODE_EXC_MAX_HZ,
    AUTO_MODE_EXC_MIN_HZ,
    AUTO_MODE_LOCAL_REFINE_ENABLED,
    AUTO_MODE_LOCAL_REFINE_TOP_K,
    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
    AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_REFINE_TRIALS,
    AUTO_MODE_TARGET_TOP_N,
    AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    AUTO_MODE_TRIALS,
    _auto_optuna_storage_path,
    get_auto_mode_cache_path,
    _auto_goal_norm,
    _auto_safe_float,
    _auto_select_builtin_target_curve,
    _auto_select_target_curve_with_trials,
    _estimate_auto_hpf_from_response,
    _estimate_auto_mag_c_min_hz,
    _resolve_auto_hpf_application,
    _run_auto_mode_search,
)
from .io.auto_mode.rank_score import attach_official_rank_score, official_rank_score  # noqa: E402
from .io.auto_mode.shared import _auto_goal_forced_level_window  # noqa: E402

_CONSOLE_LOG_LEVEL = _resolve_console_log_level()

logging.basicConfig(
    level=_CONSOLE_LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("CamillaFIR")
logger.setLevel(_CONSOLE_LOG_LEVEL)

try:
    logger.info(
        "CPU thread budget: "
        f"{int(_AUTO_THREADS_USE)}/{int(_AUTO_THREADS_CORES)} cores "
        "(auto 90% for DSP/NumPy backends)"
    )
    if _AUTO_THREADS_ENV_APPLIED:
        logger.info(
            "Applied thread env vars: "
            + ", ".join([f"{k}={os.environ.get(k, '')}" for k in _AUTO_THREADS_ENV_APPLIED])
        )
except Exception:
    pass

VERSION = APP_VERSION
PROGRAM_NAME = "CamillaFIR"
MAX_SAFE_BOOST = 8.0
FORCE_SINGLE_PLOT_FS_HZ = 48000
MAX_SAFE_TAPS = 131072
TEST_MODE = 0
TEST_MODE = os.environ.get("CAMILLAFIR_TEST", "0") == "1"

def resolve_static_dir() -> str | None:
    """
    Palauttaa PyWebIO:n static_dir-polun, josta paikallinen Plotly JS loytyy.

    Tarkistaa ensin PyInstaller-ympariston (`sys._MEIPASS/assets`) ja sen
    jalkeen lahdekoodin resurssipolun (`resources/plotly`).
    """
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
    from pywebio.pin import pin
    from .workflow.process_run_flow import ProcessRunSupport, run_process_flow
    from .workflow.process_support import (
        auto_target_mode_norm as _auto_target_mode_norm,
        auto_target_selection_method_text as _auto_target_selection_method_text,
        slugify_filename_token as _slugify_filename_token,
        pick_target_curve_label as _pick_target_curve_label,
        has_uploaded_target_file as _has_uploaded_target_file,
    )
    from .ui.process_run_bridge import build_default_ui_bridge

    return run_process_flow(
        pin_obj=pin,
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

def calculate_score(st, is_predicted=False):
    """
    Laskee UI:n kokonaispisteet (0..99) mitatulle tai ennustetulle vasteelle.

    Piste koostuu target-matchista, confidence-arvosta, RT60-bonuksesta
    (luotettavuudella painotettuna) seka heijastuslistan penalteista.
    """
    if not st:
        return 0.0

    conf = float(st.get('cmp_avg_confidence', st.get('avg_confidence', 0.0)) or 0.0)
    conf = float(np.clip(conf, 0.0, 100.0))

    _rms, match_pct = target_match_from_stats(
        st or {},
        include_filter=bool(is_predicted),
        use_confidence=True,
        use_smart_scan_range=True,
    )
    if match_pct is None:
        return float(np.clip(conf, 0.0, 99.0))
    match_pct = float(match_pct)

    base = 0.55 * match_pct + 0.35 * conf

    rt_bonus = 0.0
    try:
        rt = float(st.get('rt60_val', None)) if st.get('rt60_val', None) is not None else None
    except Exception:
        rt = None
    try:
        rel = float(st.get('rt60_reliability', 0.0) or 0.0)
    except Exception:
        rel = 0.0
    rel = float(np.clip(rel, 0.0, 1.0))

    if rt is not None and rt > 0:
        if rt <= 0.35:
            rt_bonus = ((0.35 - rt) / 0.25) * 15.0
        elif rt >= 0.55:
            rt_bonus = -min(15.0, ((rt - 0.55) / 0.35) * 15.0)
        rt_bonus *= rel

    events = st.get('cmp_reflections', st.get('reflections', [])) or []
    penalty_mult = 0.5 if is_predicted else 1.0
    event_penalty = min(8.0, float(len(events)) * 1.0) * penalty_mult

    score = base + rt_bonus - event_penalty
    return float(np.clip(score, 0.0, 99.0))


_MAIN_APP = None


def _get_main_app():
    global _MAIN_APP
    if _MAIN_APP is None:
        from .ui.camillafir_ui import build_app as _build_ui_app

        _MAIN_APP = _build_ui_app(
            process_run=process_run,
            PROGRAM_NAME=PROGRAM_NAME,
            VERSION=VERSION,
            MAX_SAFE_BOOST=MAX_SAFE_BOOST,
        )
    return _MAIN_APP


def main(*args, **kwargs):
    return _get_main_app()(*args, **kwargs)


if __name__ == '__main__':
    from pywebio import start_server

    start_server(
        main,
        port=8080,
        debug=bool(str(os.environ.get("CAMILLAFIR_WEB_DEBUG", "0") or "0").strip() == "1"),
        auto_open_webbrowser=True,
        static_dir=resolve_static_dir(),
    )
