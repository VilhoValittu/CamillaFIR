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
    Valitsee automaattisen thread-budjetin: noin 90 % CPU-ytimista.

    Esim. 4 -> 3, 8 -> 6.
    """
    try:
        cores = int(os.cpu_count() or 1)
    except Exception:
        cores = 1
    cores = max(1, int(cores))
    use = max(1, int((cores * 10) // 10))
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
import typing   # noqa: E402
import re   # noqa: E402
import unicodedata  # noqa: E402
import math # noqa: E402
import time # noqa: E402
from datetime import datetime   # noqa: E402
import numpy as np  # noqa: E402
from pywebio import start_server    # noqa: E402
from pywebio.output import put_processbar, set_processbar    # noqa: E402
from pywebio.pin import pin # noqa: E402
from .config.camillafir_config import save_config   # noqa: E402
from .resources.i8n.camillafir_i18n import t    # noqa: E402
from .ui.camillafir_housecurve import load_house_curve  # noqa: E402
from camillafir.io.measurements_loader import load_measurements_lr  # noqa: E402
from camillafir.io.measurements_txt import parse_measurements_from_path # noqa: E402
from .ui.system_health import (# noqa: E402
    compute_health,
    toast_health_gate_result,
    toast_measurement_files_missing,
)
from .config.camillafir_pipeline import (   # noqa: E402
    collect_ui_data,
    log_df_smoothing_toggle,
    build_xos_hpf,
    filter_type_short,
    choose_target_rates,
    choose_dash_fs,
    detect_is_wav_source,
)
from .ui.camillafir_export import build_export_zip, save_export_bundle  # noqa: E402
from .engine import build_config, run_pipeline, summarize_run   # noqa: E402
from .dsp import camillafir_dsp as dsp  # noqa: E402
from .dsp.target_match import target_match_from_stats  # noqa: E402
from .ui.camillafir_ui import _render_results # noqa: E402
from .ui.camillafir_utils import scale_taps_with_fs  # noqa: E402
from .io.camillafir_automatic_mode import (  # noqa: E402
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
    get_auto_mode_cache_path,
    _auto_goal_norm,
    _auto_safe_float,
    _auto_select_builtin_target_curve,
    _auto_select_target_curve_with_trials,
    _estimate_auto_hpf_from_response,
    _estimate_auto_mag_c_min_hz,
    _run_auto_mode_search,
)

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

VERSION = "v.3.5.5"
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

def _irwin_tag(mode: typing.Any) -> str:
    """
    Normalisoi IR-ikkunointitilan lyhyeksi tiedostonimitunnisteeksi.

    Tunnetut arvot muunnetaan muotoon `sym`, `asym`, `auto` tai `off`.
    Virheellinen syote palautuu arvoon `auto`.
    """
    try:
        m = str(mode or "auto").strip().lower()
    except Exception:
        m = "auto"
    if m == "rew_sym":
        return "sym"
    if m == "rew_asym":
        return "asym"
    if m in ("auto", "off"):
        return m
    return "auto"


def _auto_target_mode_norm(mode: typing.Any) -> str:
    """
    Normalisoi AUTO-tilan target-kayran valintatavan.

    Palauttaa joko `auto` (etsi paras target-kayra) tai
    `selected` (kayta kayttajan valitsemaa target-kayraa).
    """
    try:
        m = str(mode or "auto").strip().lower()
    except Exception:
        m = "auto"
    if m in ("selected", "manual", "fixed", "user"):
        return "selected"
    return "auto"


def _auto_target_selection_method_text(method: typing.Any) -> str:
    """
    Muuntaa auto-target-valinnan metodit luettaviksi loki- ja UI-teksteiksi.
    """
    try:
        key = str(method or "").strip().lower()
    except Exception:
        key = ""

    mapping = {
        "cache_signature_hit": "cache hit (same measurements + settings, no target re-evaluation)",
        "cache_measurement": "measurement cache seed",
        "cache_signature": "signature cache seed",
        "trial_with_cache_wildcard": "trial winner with cache wildcard",
        "top3x10_trials": "trial comparison",
        "top3x10_trials_rank_tie_composite": "trial comparison with rank tie-break",
        "fit_rms": "quick fit preselect",
    }
    return str(mapping.get(key, key or "unknown"))


def _slugify_filename_token(value: typing.Any, *, default: str = "target", max_len: int = 48) -> str:
    """Muuntaa tekstin turvalliseksi tiedostonimiosaksi."""
    try:
        raw = str(value or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return default

    try:
        txt = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    except Exception:
        txt = raw
    txt = re.sub(r"[^A-Za-z0-9]+", "-", txt).strip("-").lower()
    if not txt:
        return default
    if len(txt) > int(max_len):
        txt = txt[: int(max_len)].rstrip("-")
    return txt or default


def _pick_target_curve_label(data: dict) -> str:
    """Valitsee target curven nimen vientitiedostonimiin."""
    try:
        up = data.get("hc_custom_file")
        if isinstance(up, dict):
            for k in ("filename", "name", "file_name"):
                v = up.get(k)
                if isinstance(v, str) and v.strip():
                    return os.path.splitext(os.path.basename(v.strip()))[0]
    except Exception:
        pass

    try:
        p = str(data.get("local_path_house") or "").strip()
    except Exception:
        p = ""
    if p:
        return os.path.splitext(os.path.basename(p))[0]

    try:
        hc_mode = str(data.get("hc_mode") or "").strip()
    except Exception:
        hc_mode = ""
    if hc_mode:
        return hc_mode

    try:
        src = str(data.get("hc_source") or "").strip()
    except Exception:
        src = ""
    if src:
        return src
    return "Target"


def _has_uploaded_target_file(data: dict) -> bool:
    """Checks if UI data contains an uploaded custom target file."""
    try:
        up = data.get("hc_custom_file", None)
    except Exception:
        up = None

    def _is_uploaded_file_obj(v: typing.Any) -> bool:
        if not isinstance(v, dict):
            return False
        try:
            content = v.get("content", None)
            if isinstance(content, (bytes, bytearray)) and len(content) > 0:
                return True
        except Exception:
            pass
        for k in ("filename", "name", "file_name"):
            try:
                s = str(v.get(k, "") or "").strip()
            except Exception:
                s = ""
            if s:
                return True
        return False

    if _is_uploaded_file_obj(up):
        return True
    if isinstance(up, list):
        for item in up:
            if _is_uploaded_file_obj(item):
                return True
    return False

def _resolve_ui_stats_fs(ui_stats_fs: typing.Any, selected_fs: typing.Any) -> int:
    """
    Valitsee UI-statistiikalle oikean sample raten.

    Priorisoi dashboardille valitun analyysinopeuden (`ui_stats_fs`) ja
    kaatuu turvallisesti UI-valintaan (`selected_fs`) tai 44100 Hz:iin.
    """
    for cand in (ui_stats_fs, selected_fs, 44100):
        try:
            fs_i = int(cand)
        except Exception:
            continue
        if fs_i > 0:
            return fs_i
    return 44100

def _shift_zeropad_1d(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Siirtaa 1D-signaalia kokonaisilla naytteilla nollataytolla.

    Positiivinen siirto lisaa viivetta (siirto oikealle), negatiivinen
    aikaistaa signaalia (siirto vasemmalle) ilman wrap-around-kaytosta.
    """
    arr = np.asarray(x)
    n = int(arr.size)
    s = int(shift)
    if n == 0 or s == 0:
        return arr.copy()
    out = np.zeros_like(arr)
    if s > 0:
        if s < n:
            out[s:] = arr[:-s]
    else:
        s = -s
        if s < n:
            out[:-s] = arr[s:]
    return out


def _postpolish_wav_filter_ir(
    ir: np.ndarray,
    fs: int,
    *,
    mag_c_min: float,
    mag_c_max: float,
    trans_width: float,
) -> np.ndarray:
    """
    Viimeistelee WAV-lahteesta lasketun FIR-impulssin ennen vientia.

    Tasaa korjausalueen ylarajan ymparilla esiintyvaa ripplea taajuustasossa,
    sailyttaa vaiheen ja skaalaa ulostulon takaisin alkuperaiseen huippuun.
    """
    x = np.asarray(ir, dtype=float).reshape(-1)
    n = int(x.size)
    fs_i = int(fs) if fs else 0
    if n < 64 or fs_i <= 0:
        return x

    cmin = float(mag_c_min if np.isfinite(mag_c_min) else 0.0)
    cmax = float(mag_c_max if np.isfinite(mag_c_max) else 0.0)
    tw = float(trans_width if np.isfinite(trans_width) else 0.0)
    if cmax <= max(1.0, cmin):
        return x
    if tw <= 0.0:
        tw = max(50.0, 0.4 * cmax)

    f_lo = max(cmin, cmax - 0.95 * tw)
    f_hi = min(float(fs_i) * 0.5, cmax + 1.45 * tw)
    if f_hi <= f_lo:
        return x

    h = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / float(fs_i))
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-14))
    ph = np.angle(h)

    df = float(np.median(np.diff(freqs))) if freqs.size > 2 else 1.0
    sigma_bins = max(2.0, 8.0 / max(df, 1e-9))
    half = int(max(3, round(4.0 * sigma_bins)))
    kx = np.arange(-half, half + 1, dtype=float)
    kk = np.exp(-0.5 * (kx / sigma_bins) ** 2)
    kk /= np.sum(kk)
    mag_sm = np.convolve(mag_db, kk, mode="same")

    zone = (freqs >= f_lo) & (freqs <= f_hi)
    if int(np.count_nonzero(zone)) < 8:
        return x

    w = np.zeros_like(freqs, dtype=float)
    span = max(1e-9, float(f_hi - f_lo))
    zz = (freqs[zone] - f_lo) / span
    w[zone] = 0.5 - 0.5 * np.cos(np.pi * np.clip(zz, 0.0, 1.0))

    mix = 0.95
    mag_out = mag_db + (mag_sm - mag_db) * (mix * w)

    h2 = np.power(10.0, mag_out / 20.0) * np.exp(1j * ph)
    y = np.fft.irfft(h2, n=n)

    p0 = float(np.max(np.abs(x)))
    p1 = float(np.max(np.abs(y)))
    if p0 > 0.0 and p1 > 0.0:
        y = y * (p0 / p1)
    return y.astype(float, copy=False)


def process_run():
    from .ui.camillafir_ui import (
        update_auto_selected_bar as auto_selected_bar_cb,
        update_status as status_cb,
    )

    run_started_at = time.perf_counter()
    perf_stats = {
        "read_s": 0.0,
        "dsp_s": 0.0,
        "zip_png_s": 0.0,
    }
    per_fs_stats = {}

    def _elapsed_seconds() -> float:
        try:
            return max(0.0, float(time.perf_counter() - run_started_at))
        except Exception:
            return 0.0

    def _status(msg):
        if callable(status_cb):
            try:
                status_cb(f"{msg} | {_elapsed_seconds():.1f} s")
            except Exception:
                pass

    def _set_auto_selected_bar(msg: typing.Any = ""):
        if callable(auto_selected_bar_cb):
            try:
                auto_selected_bar_cb(msg)
            except Exception:
                pass

    def _build_auto_selected_text(run_data: dict) -> str:
        try:
            target_name = str(
                run_data.get(
                    "target_curve_name",
                    run_data.get("hc_mode", "n/a"),
                )
                or "n/a"
            ).strip()
        except Exception:
            target_name = "n/a"
        if not target_name:
            target_name = "n/a"

        f6_hz = _auto_safe_float(
            run_data.get("_auto_mag_c_min_hz", run_data.get("mag_c_min", float("nan"))),
            float("nan"),
        )
        f6_txt = f"{f6_hz:.1f} Hz" if np.isfinite(f6_hz) else "n/a"

        hpf_enabled = bool(run_data.get("hpf_enable", False))
        hpf_freq = _auto_safe_float(run_data.get("hpf_freq", float("nan")), float("nan"))
        hpf_slope = _auto_safe_float(run_data.get("hpf_slope", float("nan")), float("nan"))
        if hpf_enabled and np.isfinite(hpf_freq):
            if np.isfinite(hpf_slope):
                hpf_txt = f"{hpf_freq:.1f} Hz/{int(round(hpf_slope))} dB/oct"
            else:
                hpf_txt = f"{hpf_freq:.1f} Hz"
        else:
            hpf_txt = "off"

        ft_short = filter_type_short(str(run_data.get("filter_type", "") or ""))

        phase_hz = _auto_safe_float(run_data.get("phase_limit", float("nan")), float("nan"))
        phase_txt = f"{phase_hz:.1f} Hz" if np.isfinite(phase_hz) else "n/a"

        mixed_hz = _auto_safe_float(run_data.get("mixed_freq", float("nan")), float("nan"))
        mixed_txt = f"{mixed_hz:.1f} Hz" if np.isfinite(mixed_hz) else "n/a"

        detail_txt = ""
        if ft_short in ("Linear", "Asymmetric"):
            detail_txt = f", phase limit {phase_txt}"
        elif ft_short == "Mixed":
            detail_txt = f", mixed freq {mixed_txt}"

        return (
            "Chosen (Automatic mode): "
            f"target {target_name}, HPF {hpf_txt}, -6 dB {f6_txt}{detail_txt}"
        )

    data = collect_ui_data(pin)
    _set_auto_selected_bar("")
    data["bass_adaptive_isolation_mode"] = True
    # Target-lean trial tuning for bass adaptive smoothing.
    data["bass_smooth_sigma_scale"] = 1.20
    data["bass_smooth_conf_floor"] = 0.25
    data["bass_smooth_w_gamma"] = 2.40
    data["bass_smooth_w_max"] = 0.45
    data["program_version"] = VERSION

    try:
        mode = str(data.get("mode") or "BASIC").strip().upper()
        hr = compute_health(data, mode)
        if toast_health_gate_result(hr, mode):
            return
    except Exception:
        pass



    _iw = data.get('ir_export_window_mode')
    if not isinstance(_iw, str) or _iw.strip() == "":
        data['ir_export_window_mode'] = 'auto'
    logger.info(f"UI ir_export_window_mode={data.get('ir_export_window_mode')}")

    try:
        sh = str(data.get('ir_export_window_shape', 'hann') or 'hann').strip().lower()
    except Exception:
        sh = 'hann'
    if sh not in ('hann', 'tukey'):
        sh = 'hann'
    data['ir_export_window_shape'] = sh
    try:
        a = float(data.get('ir_export_tukey_alpha', 0.25))
    except Exception:
        a = 0.25
    if not math.isfinite(a):
        a = 0.25
    data['ir_export_tukey_alpha'] = float(np.clip(a, 0.0, 1.0))
    try:
        if filter_type_short(str(data.get("filter_type", "") or "")) == "Asymmetric":
            data["ir_export_window_mode"] = "rew_asym"
            data["ir_window_mode"] = "rew_asym"
            data["ir_export_window_shape"] = "tukey"
            data["ir_export_tukey_alpha"] = 0.25
    except Exception:
        pass

    taps_base = int(float(data.get("taps", 65536) or 65536))
    save_config(data)
    try:
        set_processbar('bar', 0.0)
    except Exception:
        put_processbar('bar')
        set_processbar('bar', 0.0)
    _t_read = time.perf_counter()
    _status(t('stat_reading'))
    f_l, m_l, p_l, f_r, m_r, p_r = load_measurements_lr(data, logger=logger)
    perf_stats["read_s"] += max(0.0, float(time.perf_counter() - _t_read))
    if f_l is None or f_r is None:
        toast_measurement_files_missing()
        return

    try:
        auto_mode_preview = bool(str(data.get("mode", "BASIC") or "BASIC").strip().upper() == "AUTO" or data.get("camillafir_automatic_mode", False))
    except Exception:
        auto_mode_preview = False
    auto_goal = _auto_goal_norm(str(data.get("auto_goal", "balanced") or "balanced"))
    data["auto_goal"] = str(auto_goal)
    auto_basis = "rank_score"
    logger.info(f"Automatic mode goal: {auto_goal} (basis: {auto_basis})")
    if auto_mode_preview:
        try:
            ft = str(data.get("filter_type", "mixed") or "mixed")
        except Exception:
            ft = "mixed"
        _status(
            "CamillaFIR automatic mode: init "
            f"(goal {auto_goal}, basis {auto_basis}, filter {ft}, taps {int(taps_base)})"
        )
    if auto_mode_preview:
        try:
            est_mag_c_min = _estimate_auto_mag_c_min_hz(
                f_l,
                m_l,
                f_r,
                m_r,
                default_hz=_auto_safe_float(data.get("mag_c_min", 25.0), 25.0),
            )
            data["mag_c_min"] = float(est_mag_c_min)
            data["_auto_mag_c_min_hz"] = float(est_mag_c_min)
            est_low_bass_cut = float(
                np.clip(
                    float(est_mag_c_min) + float(AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ),
                    float(AUTO_MODE_LOW_BASS_MIN_HZ),
                    float(AUTO_MODE_LOW_BASS_MAX_HZ),
                )
            )
            est_exc_freq = float(
                np.clip(
                    float(est_mag_c_min) + float(AUTO_MODE_EXC_FROM_F6_ADD_HZ),
                    float(AUTO_MODE_EXC_MIN_HZ),
                    float(AUTO_MODE_EXC_MAX_HZ),
                )
            )
            data["low_bass_cut_hz"] = float(round(est_low_bass_cut, 1))
            data["exc_freq"] = float(round(est_exc_freq, 1))
            data["_auto_low_bass_cut_hz"] = float(round(est_low_bass_cut, 1))
            data["_auto_exc_freq_hz"] = float(round(est_exc_freq, 1))
            data["_auto_exc_seed_freq_hz"] = float(round(est_exc_freq, 1))
            _status(
                "CamillaFIR automatic mode: protection seed "
                f"(smoothed -6 dB point {float(est_mag_c_min):.1f} Hz -> "
                f"mag_c_min {float(data['mag_c_min']):.1f} Hz, "
                f"low-cut {float(data['low_bass_cut_hz']):.1f} Hz, "
                f"exc seed {float(data['exc_freq']):.1f} Hz, "
                "final exc, mag_c_min and low_bass_cut values auto-tuned in preset search)"
            )
            user_hpf_enabled = bool(data.get("hpf_enable", False))
            auto_hpf = _estimate_auto_hpf_from_response(
                f_l,
                m_l,
                f_r,
                m_r,
                default_freq_hz=_auto_safe_float(data.get("hpf_freq", 20.0), 20.0),
                default_slope_db_oct=int(_auto_safe_float(data.get("hpf_slope", 24), 24.0)),
            )
            if isinstance(auto_hpf, dict):
                auto_hpf_conf = _auto_safe_float(auto_hpf.get("confidence", 0.0), 0.0)
                auto_hpf_method = str(auto_hpf.get("method", "") or "").strip().lower()
                auto_hpf_apply = bool(auto_hpf.get("enabled", False))
                # If user has HPF enabled in AUTO mode, always follow response-based estimate.
                if bool(user_hpf_enabled) and auto_hpf_method == "response_fit":
                    auto_hpf_apply = True
                auto_hpf["applied"] = bool(auto_hpf_apply)
                data["_auto_hpf_meta"] = dict(auto_hpf)
                if auto_hpf_apply:
                    auto_hpf_freq = _auto_safe_float(
                        auto_hpf.get("freq", data.get("hpf_freq", 20.0)),
                        _auto_safe_float(data.get("hpf_freq", 20.0), 20.0),
                    )
                    auto_hpf_slope = int(
                        round(
                            _auto_safe_float(
                                auto_hpf.get("slope_db_oct", data.get("hpf_slope", 24)),
                                _auto_safe_float(data.get("hpf_slope", 24), 24.0),
                            )
                        )
                    )
                    data["hpf_enable"] = True
                    data["hpf_freq"] = float(round(auto_hpf_freq, 1))
                    data["hpf_slope"] = int(max(6, auto_hpf_slope))
                    _status(
                        "CamillaFIR automatic mode: HPF auto-fit applied "
                        f"{float(data['hpf_freq']):.1f} Hz/{int(data['hpf_slope'])} dB/oct "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f}, "
                        f"user_hpf={'on' if bool(user_hpf_enabled) else 'off'})"
                    )
                elif user_hpf_enabled:
                    _status(
                        "CamillaFIR automatic mode: HPF auto-fit skipped "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f}) -> "
                        "keeping user HPF settings"
                    )
                else:
                    _status(
                        "CamillaFIR automatic mode: HPF auto-fit skipped "
                        f"(method {auto_hpf_method or 'n/a'}, confidence {auto_hpf_conf:.2f})"
                    )

            auto_target_mode = _auto_target_mode_norm(data.get("auto_target_mode", "auto"))
            data["auto_target_mode"] = str(auto_target_mode)
            hc_mode_raw = str(data.get("hc_mode", "") or "").strip().lower()
            has_local_target = bool(str(data.get("local_path_house", "") or "").strip())
            has_upload_target = _has_uploaded_target_file(data)
            wants_custom_target = bool(("upload" in hc_mode_raw) or ("custom" in hc_mode_raw))
            use_user_target = bool(auto_target_mode == "selected")
            if not use_user_target:
                use_user_target = bool(
                    has_local_target
                    or (wants_custom_target and has_upload_target)
                )
            if use_user_target:
                data.pop("_auto_target_seed_preset", None)
                if auto_target_mode == "selected":
                    selected_hc = str(data.get("hc_mode", "n/a") or "n/a")
                    _status(
                        "CamillaFIR automatic mode: target curve mode=selected "
                        f"(using {selected_hc}, auto target comparison disabled)"
                    )
                else:
                    selected_hc = str(data.get("hc_mode", "n/a") or "n/a")
                    _status(
                        "CamillaFIR automatic mode: target curve mode=user "
                        f"(using {selected_hc}, skip built-in target comparison)"
                    )
            else:
                if wants_custom_target and not has_local_target:
                    _status(
                        "CamillaFIR automatic mode: custom target selected but no file found, "
                        "using built-in target comparison"
                    )
                pre_target_rates = choose_target_rates(data)
                pre_fs = int(pre_target_rates[0]) if pre_target_rates else int(data.get("fs", 44100) or 44100)
                if bool(data.get("multi_rate_opt", False)):
                    pre_taps = int(scale_taps_with_fs(pre_fs, base_taps=taps_base))
                else:
                    pre_taps = int(taps_base)
                pre_xos, pre_hpf = build_xos_hpf(data)
                pre_measurements = {
                    "f_l": np.asarray(f_l, dtype=float),
                    "m_l": np.asarray(m_l, dtype=float),
                    "p_l": np.asarray(p_l, dtype=float),
                    "f_r": np.asarray(f_r, dtype=float),
                    "m_r": np.asarray(m_r, dtype=float),
                    "p_r": np.asarray(p_r, dtype=float),
                    "is_wav_source": bool(detect_is_wav_source(data, pin)),
                }
                _status(
                    "CamillaFIR automatic mode: target preselect init "
                    f"(top-{AUTO_MODE_TARGET_TOP_N}, {AUTO_MODE_TARGET_TRIALS_PER_CURVE} trials/curve, "
                    f"fs {int(pre_fs)} Hz, taps {int(pre_taps)}, "
                    f"-6 dB point {float(est_mag_c_min):.1f} Hz, goal {auto_goal})"
                )
                tc_pick = _auto_select_target_curve_with_trials(
                    base_data=data,
                    measurements=pre_measurements,
                    fs_v=int(pre_fs),
                    taps_v=int(pre_taps),
                    xos=pre_xos,
                    hpf=pre_hpf,
                    pin_obj=pin,
                    status_cb=_status,
                    top_n=int(AUTO_MODE_TARGET_TOP_N),
                    trials_per_curve=int(AUTO_MODE_TARGET_TRIALS_PER_CURVE),
                )
                if not isinstance(tc_pick, dict):
                    tc_pick = _auto_select_builtin_target_curve(
                        data,
                        f_l=f_l,
                        m_l=m_l,
                        f_r=f_r,
                        m_r=m_r,
                    )
                if isinstance(tc_pick, dict):
                    chosen_hc = str(tc_pick.get("selected_hc_mode", "Harman6") or "Harman6")
                    prev_hc = str(data.get("hc_mode", "") or "")
                    method_raw = str(tc_pick.get("selection_method", "fit_rms") or "fit_rms")
                    method_txt = _auto_target_selection_method_text(method_raw)
                    data["hc_mode"] = chosen_hc
                    target_seed_preset = dict(tc_pick.get("best_preset", {}) or {})
                    if target_seed_preset:
                        data["_auto_target_seed_preset"] = dict(target_seed_preset)
                    else:
                        data.pop("_auto_target_seed_preset", None)
                    # Auto-target selection must use built-in presets; ignore local/upload target for this run.
                    data["local_path_house"] = ""
                    data["_auto_target_curve_meta"] = dict(tc_pick)
                    _status(
                        "CamillaFIR automatic mode: target preselect winner "
                        f"{chosen_hc} (method {method_txt}, "
                        f"fit_rms {float(tc_pick.get('fit_rms_db', 0.0)):.3f} dB, "
                        f"tested {int(tc_pick.get('top_n', 0) or 0)} curves x "
                        f"{int(tc_pick.get('trials_per_curve', 0) or 0)} trials)"
                    )
                    logger.info(
                        f"Automatic mode target select: {chosen_hc} "
                        f"(fit_rms={float(tc_pick.get('fit_rms_db', 0.0)):.3f} dB, "
                        f"method={method_txt}, "
                        f"prev={prev_hc or 'n/a'})"
                    )
                    if target_seed_preset:
                        logger.info(
                            "Automatic mode target seed preset: "
                            + ", ".join([f"{k}={v}" for k, v in target_seed_preset.items()])
                        )
        except Exception as exc:
            logger.warning(f"Automatic mode target preselect failed: {type(exc).__name__}: {exc}")

    hc_f, hc_m, hc_source = load_house_curve(
        data,
        parse_measurements_from_path=parse_measurements_from_path
    )
    data['hc_source'] = hc_source
    target_curve_name = _pick_target_curve_label(data)
    target_curve_tag = _slugify_filename_token(target_curve_name, default="target")
    data["target_curve_name"] = target_curve_name
    data["target_curve_tag"] = target_curve_tag
    logger.info(f"House curve source: {hc_source}")
    logger.info(f"Export target curve tag: {target_curve_tag} (from '{target_curve_name}')")
    try:
        if hc_f is not None and hc_m is not None:
            logger.info(
                f"HC: n={len(hc_f)} f=[{hc_f[0]:.2f}..{hc_f[-1]:.2f}] "
                f"m=[{float(np.min(hc_m)):.2f}..{float(np.max(hc_m)):.2f}] mean={float(np.mean(hc_m)):.2f}"
            )
    except Exception:
        pass
    xos, hpf = build_xos_hpf(data)
    try:
        if xos:
            xo_txt = ", ".join([f"{float(x.get('freq')):.1f}Hz/{int(x.get('slope', int(x.get('order',1))*6))}dB/oct" for x in xos])
            logger.info(f"XO (UI->CFG): {xo_txt}")
        else:
            logger.info("XO (UI->CFG): off")
        if isinstance(hpf, dict) and hpf.get("enabled"):
            hf = float(hpf.get("freq", 0.0) or 0.0)
            ho = int(hpf.get("order", 0) or 0)
            logger.info(f"HPF (UI->CFG): {hf:.1f}Hz/{int(ho*6)}dB/oct")
        else:
            logger.info("HPF (UI->CFG): off")
    except Exception:
        pass
    log_df_smoothing_toggle(pin, logger)


    target_rates = choose_target_rates(data)
    multi_rate_on = bool(data.get("multi_rate_opt"))
    dash_fs = choose_dash_fs(target_rates, multi_rate_on=multi_rate_on, forced_plot_fs_hz=int(FORCE_SINGLE_PLOT_FS_HZ))
    mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    auto_mode_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    zip_dashboards_on = False

    ts = datetime.now().strftime('%d%m%y_%H%M')
    file_ts = datetime.now().strftime('%H%M_%d%m%y')
    ft_short = filter_type_short(data['filter_type'])
    l_st_f, r_st_f, l_imp_f, r_imp_f = None, None, None, None
    ui_dashboards = {}
    logger.warning(
        f"EXPORT IR (UI): shape={data.get('ir_export_window_shape')}, "
        f"alpha={data.get('ir_export_tukey_alpha')}"
    )

    val_raw = data.get('ir_export_window_mode', None)
    if not isinstance(val_raw, str) or val_raw.strip() == '':
        val_raw = data.get('ir_window_mode', 'auto')
    irw_mode = str(val_raw or 'auto').strip().lower()
    if irw_mode not in ('auto','off','rew_sym','rew_asym'):
        irw_mode = 'auto'
    data['ir_export_window_mode'] = irw_mode
    irw_tag = _irwin_tag(irw_mode)

    is_wav_source = detect_is_wav_source(data, pin)
    data["_is_wav_source"] = bool(is_wav_source)

    results_by_fs = []
    measurements = {
        "f_l": np.asarray(f_l, dtype=float),
        "m_l": np.asarray(m_l, dtype=float),
        "p_l": np.asarray(p_l, dtype=float),
        "f_r": np.asarray(f_r, dtype=float),
        "m_r": np.asarray(m_r, dtype=float),
        "p_r": np.asarray(p_r, dtype=float),
        "hc_f": hc_f,
        "hc_m": hc_m,
        "ui_data": data,
        "is_wav_source": bool(is_wav_source),
    }

    if auto_mode_enabled:
        try:
            data["comparison_mode"] = True
            auto_search_fs = int(target_rates[0]) if target_rates else int(data.get("fs", 44100) or 44100)
            if bool(data.get("multi_rate_opt", False)):
                auto_search_taps = int(scale_taps_with_fs(auto_search_fs, base_taps=taps_base))
            else:
                auto_search_taps = int(taps_base)
            set_processbar('bar', 0.10)
            _f6_hz = _auto_safe_float(
                data.get("_auto_mag_c_min_hz", data.get("mag_c_min", float("nan"))),
                float("nan"),
            )
            _f6_txt = f", -6 dB point {_f6_hz:.1f} Hz" if np.isfinite(_f6_hz) else ""
            _phase2_hint = int(AUTO_MODE_REFINE_TRIALS)
            if bool(AUTO_MODE_LOCAL_REFINE_ENABLED) and str(auto_goal) in ("balanced", "room-safe", "low-ripple"):
                _phase2_hint = int(AUTO_MODE_LOCAL_REFINE_TOP_K * AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP)
            _status(
                f"CamillaFIR automatic mode: preset search init "
                f"(phase1 {AUTO_MODE_TRIALS} + refine {_phase2_hint} trials @ {auto_search_fs} Hz{_f6_txt}, "
                f"goal {auto_goal}, basis {auto_basis}, target {str(data.get('hc_mode', 'n/a') or 'n/a')})"
            )
            auto_res = _run_auto_mode_search(
                base_data=dict(data),
                measurements=measurements,
                fs_v=int(auto_search_fs),
                taps_v=int(auto_search_taps),
                xos=xos,
                hpf=hpf,
                hc_f=hc_f,
                hc_m=hc_m,
                pin_obj=pin,
                status_cb=_status,
                n_trials=int(AUTO_MODE_TRIALS),
            )
            if isinstance(auto_res, dict):
                best_preset = dict(auto_res.get("best_preset", {}) or {})
                best_metrics = dict(auto_res.get("best_metrics", {}) or {})
                winner_explanation = dict(auto_res.get("winner_explanation", {}) or {})
                if best_preset:
                    data.update(best_preset)
                    measurements["ui_data"] = data
                best_auto_exc_hz = _auto_safe_float(
                    auto_res.get(
                        "best_auto_exc_freq_hz",
                        best_metrics.get("auto_exc_zero_penalty_hz", float("nan")),
                    ),
                    float("nan"),
                )
                if np.isfinite(best_auto_exc_hz):
                    best_auto_exc_hz = float(
                        np.clip(
                            float(best_auto_exc_hz),
                            float(AUTO_MODE_EXC_MIN_HZ),
                            float(AUTO_MODE_EXC_MAX_HZ),
                        )
                    )
                    data["exc_freq"] = float(round(best_auto_exc_hz, 1))
                    data["_auto_exc_freq_hz"] = float(round(best_auto_exc_hz, 1))
                    logger.info(
                        "CamillaFIR automatic mode: excursion protection auto-tuned "
                        f"to zero-penalty floor {float(data['_auto_exc_freq_hz']):.1f} Hz"
                    )
                sel_goal = str(auto_res.get("auto_goal", data.get("auto_goal", "balanced")) or "balanced")
                sel_basis = str(
                    auto_res.get(
                        "selection_basis",
                        "rank_score",
                    )
                    or "rank_score"
                )
                data["_auto_mode_meta"] = {
                    "enabled": True,
                    "auto_goal": str(sel_goal),
                    "selection_basis": str(sel_basis),
                    "trials_total": int(auto_res.get("trials_total", AUTO_MODE_TRIALS)),
                    "trials_ok": int(auto_res.get("trials_ok", 0)),
                    "trials_phase1_total": int(auto_res.get("trials_phase1_total", AUTO_MODE_TRIALS)),
                    "trials_phase1_ok": int(auto_res.get("trials_phase1_ok", 0)),
                    "phase1_plateau_hit": bool(auto_res.get("phase1_plateau_hit", False)),
                    "trials_phase2_total": int(auto_res.get("trials_phase2_total", AUTO_MODE_REFINE_TRIALS)),
                    "trials_phase2_ok": int(auto_res.get("trials_phase2_ok", 0)),
                    "phase2_plateau_hit": bool(auto_res.get("phase2_plateau_hit", False)),
                    "search_fs": int(auto_res.get("search_fs", auto_search_fs)),
                    "search_taps": int(auto_res.get("search_taps", auto_search_taps)),
                    "auto_exc_seed_freq_hz": (
                        float(data.get("_auto_exc_seed_freq_hz"))
                        if np.isfinite(_auto_safe_float(data.get("_auto_exc_seed_freq_hz", float("nan")), float("nan")))
                        else float("nan")
                    ),
                    "best_auto_exc_freq_hz": (
                        float(data.get("_auto_exc_freq_hz"))
                        if np.isfinite(_auto_safe_float(data.get("_auto_exc_freq_hz", float("nan")), float("nan")))
                        else float("nan")
                    ),
                    "best_metrics": best_metrics,
                    "best_preset": best_preset,
                    "winner_explanation": winner_explanation,
                    "top": list(auto_res.get("top", []) or []),
                }
                _status(
                    "CamillaFIR automatic mode: finalize "
                    f"(winner rank {_auto_safe_float(best_metrics.get('rank_score'), 0.0):.3f}/100, "
                    f"avg {_auto_safe_float(best_metrics.get('avg_score'), 0.0):.3f}, "
                    f"boost {_auto_safe_float(best_metrics.get('max_net_boost_db'), 0.0):.2f} dB, "
                    f"events {int(best_metrics.get('events_total', 0) or 0)})"
                )
                _set_auto_selected_bar(_build_auto_selected_text(data))
                logger.info(
                    "Automatic mode best: "
                    f"goal={sel_goal}, basis={sel_basis}, "
                    f"rank={_auto_safe_float(best_metrics.get('rank_score'), 0.0):.3f}/100, "
                    f"avg={_auto_safe_float(best_metrics.get('avg_score'), 0.0):.3f}, "
                    f"dsp_pen={_auto_safe_float(best_metrics.get('dsp_penalty'), 0.0):.3f}, "
                    f"exc_pen={_auto_safe_float(best_metrics.get('exc_penalty'), 0.0):.3f}, "
                    f"boost={_auto_safe_float(best_metrics.get('max_net_boost_db'), 0.0):.2f} dB, "
                    f"events={int(best_metrics.get('events_total', 0) or 0)}, "
                    f"event_sev={_auto_safe_float(best_metrics.get('events_severity'), 0.0):.2f}"
                )
            else:
                logger.warning("Automatic mode could not produce a valid best preset; using current settings.")
            set_processbar('bar', 0.18)
        except Exception as exc:
            logger.warning(f"Automatic mode failed: {type(exc).__name__}: {exc}")

    for i, fs_v in enumerate(target_rates):
        if bool(data.get('multi_rate_opt', False)):
            taps_v = scale_taps_with_fs(fs_v, base_taps=taps_base)
            logger.info(f"Auto taps: {int(fs_v)} Hz -> {int(taps_v)} taps (ref 44100 Hz -> {taps_base} taps)")
        else:
            taps_v = taps_base

        _status(f"{t('stat_calc')} {fs_v}Hz...")
        set_processbar('bar', 0.2 + 0.6 * (i / len(target_rates)))
        data["enable_residual_pass"] = bool(data.get("enable_residual_pass", False))

        cfg = build_config(
            data,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            pin=pin,
            max_safe_boost=float(MAX_SAFE_BOOST),
        )
        try:
            setattr(cfg, "bass_smooth_w_gamma", float(data.get("bass_smooth_w_gamma", 2.40)))
            setattr(cfg, "bass_smooth_w_max", float(data.get("bass_smooth_w_max", 0.45)))
        except Exception:
            pass

        _status(f"{t('stat_calc')} {int(fs_v)} Hz")
        _t_dsp = time.perf_counter()
        result = run_pipeline(cfg, measurements)
        result.metrics["summary"] = summarize_run(result)
        _dsp_dt = max(0.0, float(time.perf_counter() - _t_dsp))

        perf_stats["dsp_s"] += _dsp_dt
        _fs_k = int(fs_v)
        _slot = per_fs_stats.setdefault(_fs_k, {})
        _slot["dsp_s"] = float(_slot.get("dsp_s", 0.0)) + _dsp_dt
        results_by_fs.append(result)

        if int(fs_v) == int(dash_fs):
            l_st_f, r_st_f = result.l_st, result.r_st
            l_imp_f, r_imp_f = result.l_ir, result.r_ir

    if not results_by_fs:
        toast_measurement_files_missing()
        return

    _t_zip = time.perf_counter()
    zip_buffer, ui_dashboards, zip_perf = build_export_zip(
        data=data,
        results=results_by_fs,
        ft_short=ft_short,
        file_ts=file_ts,
        irw_tag=irw_tag,
        write_dashboards=zip_dashboards_on,
        dash_fs=int(dash_fs),
    )
    _zip_dt = max(0.0, float(time.perf_counter() - _t_zip))
    perf_stats["zip_png_s"] += max(float(zip_perf.get("zip_png_s", 0.0) or 0.0), _zip_dt)
    for fs_k, st in (zip_perf.get("per_fs_stats", {}) or {}).items():
        slot = per_fs_stats.setdefault(int(fs_k), {})
        slot["zip_png_s"] = float(slot.get("zip_png_s", 0.0)) + float(st.get("zip_png_s", 0.0) or 0.0)

    fname, saved_filters_dir, _save_msg = save_export_bundle(
        zip_buffer,
        ft_short=ft_short,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        ts=ts,
        program_version=str(data.get("program_version", VERSION) or VERSION),
    )
    auto_cache_path = None
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    if bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False)):
        try:
            auto_cache_path = str(get_auto_mode_cache_path())
        except Exception:
            auto_cache_path = None

    if l_st_f is None or r_st_f is None or l_imp_f is None or r_imp_f is None:
        fallback = results_by_fs[-1]
        l_st_f, r_st_f = fallback.l_st, fallback.r_st
        l_imp_f, r_imp_f = fallback.l_ir, fallback.r_ir

    logger.info(f"UI stats mode L/R: {l_st_f.get('analysis_mode')}/{r_st_f.get('analysis_mode')} | "
                f"len cmp f/m/t = {len(l_st_f.get('cmp_freq_axis',[]))}/{len(l_st_f.get('cmp_measured_mags',[]))}/{len(l_st_f.get('cmp_target_mags',[]))}")

    _render_results(
        data,
        f_l,
        m_l,
        p_l,
        f_r,
        m_r,
        p_r,
        l_imp_f,
        r_imp_f,
        l_st_f,
        r_st_f,
        fname,
        zip_buffer,
        dash_html_l=ui_dashboards.get("left_html"),
        dash_html_r=ui_dashboards.get("right_html"),
        run_started_at=run_started_at,
        perf_stats=perf_stats,
        per_fs_stats=per_fs_stats,
        saved_filters_dir=saved_filters_dir,
        auto_cache_path=auto_cache_path,
    )

def _ui_pick(stats, key):
    """
    Hakee UI:lle oikean arvon natiivi- tai vertailutilasta.

    Jos analyysitila on `comparison`, funktio priorisoi `cmp_<key>`-avaimen.
    Muussa tapauksessa palauttaa tavallisen `<key>`-arvon.
    """
    if not stats:
        return None
    mode = str(stats.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)


def _pick_cmp(stats, key):
    """Yhteensopiva valitsin vertailutilan (`cmp_`) ja natiivin datan valille."""
    if not stats:
        return None
    if str(stats.get("analysis_mode", "native")).lower() == "comparison":
        return stats.get("cmp_" + key, stats.get(key))
    return stats.get(key)

def view_mags_for_plot(freqs, mags, *, plot_smoothing_level="Psychoacoustic"):
    """
    Tuottaa kuvaajia varten tasoitetun amplitudikayran (vain UI-nakymaan).

    Tukee kahta tilaa:
    - `Psychoacoustic`: REW-tyylinen painotettu LF/HF-yhdistelma psyko=CamillaFIR Reference
    - numeerinen N: standardi 1/N-oktaavitasoitus
    """
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(mags, dtype=float)
    if f.size < 8 or m.size != f.size:
        return mags

    try:
        apply_smoothing_std = dsp.apply_smoothing_std
    except Exception:
        return mags

    psl = plot_smoothing_level

    if isinstance(psl, str) and ("psy" in psl.lower()):
        try:
            from .dsp.smoothing import psychoacoustic_smoothing
            return psychoacoustic_smoothing(f, m)
        except Exception:
            return m

    try:
        n = float(psl if not isinstance(psl, str) else 48.0)
        if not np.isfinite(n) or n <= 0:
            n = 48.0
        oct_frac = 1.0 / n
    except Exception:
        oct_frac = 1.0 / 48.0

    try:
        m_sm, _ = apply_smoothing_std(f, m, np.zeros_like(m), float(oct_frac))
        return m_sm
    except Exception:
        return m


def _ensure_scoring_keys(st, f_in, m_in, hc_f, hc_m):
    """
    Varmistaa, etta pisteytykseen tarvittavat avaimet ovat stats-sanakirjassa.

    Taydentaa puuttuvat kentat (`freq_axis`, `measured_mags`, `target_mags`,
    `confidence_mask`) turvallisilla oletuksilla annetuista syotteista.
    """
    try:
        if st is None:
            return st

        f = np.asarray(f_in if f_in is not None else [], dtype=float)
        m = np.asarray(m_in if m_in is not None else [], dtype=float)
        if f.size > 1 and m.size > 1:
            if st.get("freq_axis") is None:
                st["freq_axis"] = f
            if st.get("measured_mags") is None:
                st["measured_mags"] = m

        if st.get("target_mags") is None:
            try:
                hf = np.asarray(hc_f if hc_f is not None else [], dtype=float)
                hm = np.asarray(hc_m if hc_m is not None else [], dtype=float)
                if f.size > 1 and hf.size > 1 and hm.size > 1:
                    st["target_mags"] = np.interp(f, hf, hm)
            except Exception:
                pass

        if st.get("confidence_mask") is None:
            if f.size > 1:
                st["confidence_mask"] = np.ones_like(f, dtype=float)
        return st
    except Exception:
        return st

_HOUSE_FREQS = np.array([
    20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0,
    200.0, 250.0, 400.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 20000.0
], dtype=float)

def _resample_to_freq_axis(freqs_dst: np.ndarray, arr: np.ndarray, freqs_src: np.ndarray) -> np.ndarray:
    """
    Resamplaa 1D-kayran uudelle taajuusakselille log-taajuusinterpoloinnilla.

    Tarkoitettu tilanteisiin, joissa target- tai filter-kayran pisteistus
    poikkeaa mittausakselista.
    """
    if arr.size == 0 or freqs_src.size == 0 or freqs_dst.size == 0:
        return arr
    f1 = np.maximum(freqs_src.astype(float), 1.0)
    f2 = np.maximum(freqs_dst.astype(float), 1.0)
    lf1 = np.log10(f1)
    lf2 = np.log10(f2)
    order = np.argsort(lf1)
    lf1 = lf1[order]
    a1 = arr.astype(float)[order]
    return np.interp(lf2, lf1, a1, left=a1[0], right=a1[-1])


def calculate_target_match(st):
    """
    Laskee korjatun vasteen tavoitevastaavuuden prosentteina (0-100).

    Vertailu tehdaan kayrille `(measured + filter)` vs. `target`, jonka
    RMS-virhe muunnetaan sigmoidilla prosenttipisteiksi.
    """
    _rms, match_pct = target_match_from_stats(
        st or {},
        include_filter=True,
        use_confidence=True,
        use_smart_scan_range=True,
    )
    return float(match_pct) if match_pct is not None else 0.0



def _avg_confidence_pct(st: dict) -> float:
    """
    Palauttaa keskimaaraisen confidence-arvon prosentteina.

    Tukee seka natiivi- (`avg_confidence` / `confidence_mask`) etta
    vertailutilan (`cmp_avg_confidence` / `cmp_confidence_mask`) avaimia.
    """
    if not st:
        return 0.0
    mode = str(st.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        v = st.get("cmp_avg_confidence", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
        cm_src = st.get("cmp_confidence_mask", None)
        cm = np.asarray(cm_src if cm_src is not None else [], dtype=float)
        if cm.size:
            return float(np.mean(cm) * 100.0)
        return 0.0
    v = st.get("avg_confidence", None)
    if v is not None:
        try:
            return float(v)
        except Exception:
            pass
    cm_src = st.get("confidence_mask", None)
    cm = np.asarray(cm_src if cm_src is not None else [], dtype=float)
    if cm.size:
        return float(np.mean(cm) * 100.0)
    return 0.0


def calculate_target_match_unfiltered(st: dict) -> float:
    """
    Laskee tavoitevastaavuuden suodattamattomalle vasteelle.

    Vertailu tehdaan suoraan kayrille `measured` vs. `target` samalla
    sigmoidikartoituksella kuin suodatetussa target-match-laskennassa.
    """
    _rms, match_pct = target_match_from_stats(
        st or {},
        include_filter=False,
        use_confidence=True,
        use_smart_scan_range=True,
    )
    return float(match_pct) if match_pct is not None else 0.0

def _inject_filter_mags_for_ui(st: dict, filt_ir, fs: int):
    """
    Laskee ja injektoi `filter_mags`-kayran stats-rakenteeseen UI:ta varten.

    Jos putki ei ole tallentanut filter-kayraa valmiiksi, funktio muodostaa
    sen `filt_ir`-impulssista FFT:lla ja interpoloi aktiiviselle taajuusakselille.
    """
    try:
        if st is None or filt_ir is None:
            return
        mode = str(st.get("analysis_mode", "native") or "native").lower()
        key_f = "cmp_freq_axis" if mode == "comparison" else "freq_axis"
        key_g = "cmp_filter_mags" if mode == "comparison" else "filter_mags"

        f_src = st.get(key_f, None)
        f_axis = np.asarray(f_src if f_src is not None else [], dtype=float)
        if f_axis.size < 4:
            return

        ir = np.asarray(filt_ir, dtype=float).flatten()
        if ir.size < 8:
            return

        fs_i = int(fs) if fs else 0
        if fs_i <= 0:
            return

        h = np.fft.rfft(ir)
        f_fft = np.fft.rfftfreq(ir.size, d=1.0 / fs_i)
        g_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))

        f_min = float(np.min(f_fft))
        f_max = float(np.max(f_fft))
        f_q = np.clip(f_axis, f_min, f_max)
        st[key_g] = np.interp(f_q, f_fft, g_db).tolist()
        st[f"{key_g}_source"] = "ir_fft_final"
    except Exception:
        return


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


from .ui.camillafir_ui import build_app as _build_ui_app
main = _build_ui_app(
    process_run=process_run,
    PROGRAM_NAME=PROGRAM_NAME,
    VERSION=VERSION,
    MAX_SAFE_BOOST=MAX_SAFE_BOOST,
)


if __name__ == '__main__':
    start_server(
        main,
        port=8080,
        debug=bool(str(os.environ.get("CAMILLAFIR_WEB_DEBUG", "0") or "0").strip() == "1"),
        auto_open_webbrowser=True,
        static_dir=resolve_static_dir(),
    )
