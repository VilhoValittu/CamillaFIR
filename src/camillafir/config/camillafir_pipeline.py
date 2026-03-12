from __future__ import annotations

from asyncio.log import logger
from typing import Any, Dict, List, Optional, Tuple
import logging
import math
import numpy as np

from ..io.auto_mode.filter_priors import get_auto_mode_filter_auto_defaults
from ..ui.camillafir_modes import MODE_DEFAULTS


_AUTO_MODE_DEFAULT_CFG_TO_UI = {
    "global_gain_db": "gain",
    "mag_c_min": "mag_c_min",
    "mag_c_max": "mag_c_max",
    "max_boost_db": "max_boost",
    "max_cut_db": "max_cut_db",
    "phase_limit": "phase_limit",
    "reg_strength": "reg_strength",
    "fdw_cycles": "fdw_cycles",
    "filter_smooth": "filter_smooth",
    "tdc_strength": "tdc_strength",
    "tdc_max_reduction_db": "tdc_max_reduction_db",
    "tdc_slope_db_per_oct": "tdc_slope_db_per_oct",
    "low_bass_cut_hz": "low_bass_cut_hz",
    "ir_window_ms": "ir_window",
    "ir_window_ms_left": "ir_window_left",
    "ir_window_right": "ir_window",
    "ir_window_left": "ir_window_left",
    "mixed_split_freq": "mixed_freq",
    "trans_width": "trans_width",
    "bass_first_mode_max_hz": "bass_first_mode_max_hz",
    "max_slope_db_per_oct": "max_slope_db_per_oct",
    "max_slope_boost_db_per_oct": "max_slope_boost_db_per_oct",
    "max_slope_cut_db_per_oct": "max_slope_cut_db_per_oct",
    "lvl_manual_db": "lvl_manual_db",
    "lvl_min": "lvl_min",
    "lvl_max": "lvl_max",
    "conf_pull_floor": "conf_pull_floor",
    "conf_pull_max_hz": "conf_pull_max_hz",
    "conf_pull_gamma_cut": "conf_pull_gamma_cut",
    "conf_pull_gamma_boost": "conf_pull_gamma_boost",
    "low_bass_cut_strength": "low_bass_cut_strength",
    "filter_type_str": "filter_type",
    "plot_smoothing_level": "plot_smoothing_level",
    "lvl_mode": "lvl_mode",
    "lvl_algo": "lvl_algo",
    "stereo_link_strategy": "stereo_link_strategy",
    "enable_mag_correction": "mag_correct",
    "unsafe_raw_dsp": "unsafe_raw_dsp",
    "exc_prot": "exc_prot",
    "enable_tdc": "enable_tdc",
    "enable_afdw": "enable_afdw",
    "df_smoothing": "df_smoothing",
    "comparison_mode": "comparison_mode",
    "bass_first_ai": "bass_first_ai",
    "phase_safe_2058": "phase_safe_2058",
    "stereo_link": "stereo_link",
    "low_bass_cut_enable": "low_bass_cut_enable",
}


def _apply_auto_mode_managed_settings(data: Dict[str, Any]) -> None:
    """Force AUTO mode to use program-managed settings except allowed user choices."""
    try:
        filter_type = str(data.get("filter_type", "Asymmetric") or "Asymmetric")
    except Exception:
        filter_type = "Asymmetric"

    merged_defaults = dict(MODE_DEFAULTS.get("AUTO", {}) or {})
    merged_defaults.update(dict(get_auto_mode_filter_auto_defaults(filter_type) or {}))

    forced = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_mode_workers": 0,
        "mag_correct": True,
        "gain": 0.0,
        "lvl_mode": "Auto",
        "lvl_algo": "Median",
        "lvl_manual_db": 0.0,
        "normalize_opt": False,
        "align_opt": True,
        "unsafe_raw_dsp": False,
        "stereo_link": True,
        "stereo_link_strategy": "auto",
        "exc_prot": True,
        "low_bass_cut_enable": True,
        "comparison_mode": True,
        "df_smoothing": False,
        "auto_target_mode": str(data.get("auto_target_mode", "auto") or "auto"),
    }

    for cfg_key, ui_key in _AUTO_MODE_DEFAULT_CFG_TO_UI.items():
        if cfg_key in merged_defaults:
            forced[ui_key] = merged_defaults[cfg_key]

    for key, value in forced.items():
        data[key] = value

def collect_ui_data(pin) -> Dict[str, Any]:
    """Funktio: collect ui data."""
    logger = logging.getLogger("CamillaFIR")
    p_keys = [
        "mode", "auto_goal", "auto_target_mode", "auto_mode_workers", "fs", "taps", "filter_type", "mixed_freq", "gain", "hc_mode",
        "mag_c_min", "mag_c_max", "max_boost", "max_cut_db", "max_slope_db_per_oct",
        "max_slope_boost_db_per_oct", "max_slope_cut_db_per_oct", "phase_limit", "mag_correct",
        "excess_phase_strength", "low_freq_full_correction_hz", "high_freq_no_correction_hz",
        "enable_ir_pre_energy_guard", "pre_energy_ratio_max", "pre_energy_guard_strength",
        "max_pre_ringing_db", "max_excess_delay_ms", "gd_grad_limit_ms_per_oct",
        "ir_anchor_mode", "min_causal_ms", "auto_asym_left_ratio", "auto_asym_left_max_ms",
        "lvl_mode", "reg_strength", "normalize_opt", "align_opt",
        "stereo_link", "stereo_link_strategy", "exc_prot", "exc_freq", "low_bass_cut_hz", "low_bass_cut_enable", "hpf_enable", "hpf_freq",
        "hpf_slope", "multi_rate_opt", "ir_window", "ir_window_left", "ir_window_right", "ir_export_window_mode", "ir_window_mode",
        "ir_export_window_shape", "ir_export_tukey_alpha",
        "local_path_l", "local_path_r", "fmt", "lvl_manual_db",
        "lvl_min", "lvl_max", "lvl_algo", "fdw_cycles",
        "trans_width", "smoothing_level", "filter_smooth", "plot_smoothing_level",
        "bass_smooth_adaptive", "bass_smooth_hz", "bass_smooth_sigma_scale", "bass_smooth_conf_floor",
        "bass_adaptive_isolation_mode",
        "bass_boost_cap_enable", "bass_boost_cap_hz", "bass_boost_cap_extra_db", "bass_boost_cap_conf_min",
        "bass_boost_post_restore_enable", "bass_boost_post_restore_strength",
        "enable_tdc", "tdc_strength", "tdc_max_reduction_db",
        "tdc_slope_db_per_oct", "enable_afdw", "df_smoothing", "comparison_mode",
        "bass_first_ai", "bass_first_mode_max_hz",
        "local_path_house",
        "conf_pull_floor", "conf_pull_ceil", "conf_pull_max_hz",
        "conf_pull_gamma_cut", "conf_pull_gamma_boost",
        "conf_pull_conf_smooth_sigma",
        "conf_pull_bass_floor_hz", "conf_pull_bass_floor_min",
        "conf_pull_bass_boost_floor_hz", "conf_pull_bass_boost_floor_min",
        "conf_pull_bass_boost_restore",
        "low_bass_cut_strength", "hc_custom_file",
        "file_l", "file_r", "unsafe_raw_dsp",
        "camillafir_automatic_mode",
    ]

    data: Dict[str, Any] = {}
    for k in p_keys:
        try:
            data[k] = pin[k]
        except Exception:
            data[k] = None

    if data.get("ir_window_right", None) in (None, ""):
        data["ir_window_right"] = data.get("ir_window", 500.0)
    if data.get("ir_window", None) in (None, ""):
        data["ir_window"] = data.get("ir_window_right", 500.0)

    for k in [
        "mag_correct", "normalize_opt", "align_opt", "multi_rate_opt",
        "stereo_link", "exc_prot", "hpf_enable", "df_smoothing",
        "comparison_mode", "bass_first_ai", "phase_safe_2058",
        "enable_tdc", "enable_afdw", "low_bass_cut_enable", "enable_ir_pre_energy_guard",
        "bass_smooth_adaptive",
        "bass_adaptive_isolation_mode",
        "bass_boost_cap_enable",
        "bass_boost_post_restore_enable",
        "unsafe_raw_dsp",
        "camillafir_automatic_mode",
    ]:
        try:
            if isinstance(data.get(k, None), list):
                data[k] = bool(data[k])
        except Exception:
            pass

    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    if mode_u not in ("BASIC", "ADVANCED", "AUTO"):
        mode_u = "BASIC"

    is_auto_mode = (mode_u == "AUTO")
    if not is_auto_mode:
        try:
            is_auto_mode = bool(data.get("camillafir_automatic_mode", False))
        except Exception:
            is_auto_mode = False

    if is_auto_mode:
        mode_u = "AUTO"
        data["mode"] = "AUTO"
    data["camillafir_automatic_mode"] = bool(is_auto_mode)

    try:
        atm = str(data.get("auto_target_mode", "auto") or "auto").strip().lower()
    except Exception:
        atm = "auto"
    if atm in ("selected", "manual", "fixed", "user"):
        atm = "selected"
    else:
        atm = "auto"
    data["auto_target_mode"] = str(atm)

    if is_auto_mode:
        _apply_auto_mode_managed_settings(data)

    if mode_u in ("BASIC", "AUTO"):
        data["lvl_mode"] = "Auto"
        data["unsafe_raw_dsp"] = False

    try:
        sls = str(data.get("stereo_link_strategy", "") or "").strip().lower()
    except Exception:
        sls = ""
    if sls not in ("shared", "hybrid", "auto"):
        sls = "auto"
    # Default behavior across modes: auto strategy
    # (shared/hybrid remain selectable via explicit config value).
    if sls == "":
        sls = "auto"
    data["stereo_link_strategy"] = sls

    # Confidence-pull controls are hidden from UI; keep stable internal defaults.
    # ADVANCED keeps the tuned profile from mode defaults.
    hidden_conf_defaults = (
        {
            "conf_pull_floor": 0.05,
            "conf_pull_ceil": 0.95,
            "conf_pull_max_hz": 180.0,
            "conf_pull_gamma_cut": 0.554,
            "conf_pull_gamma_boost": 0.75,
            "low_bass_cut_strength": 0.0,
        }
        if mode_u == "ADVANCED"
        else {
            "conf_pull_floor": 0.05,
            "conf_pull_ceil": 0.95,
            "conf_pull_max_hz": 200.0,
            "conf_pull_gamma_cut": 0.55,
            "conf_pull_gamma_boost": 1.35,
            "low_bass_cut_strength": 0.0,
        }
    )
    for _k, _v in hidden_conf_defaults.items():
        if data.get(_k, None) in (None, ""):
            data[_k] = _v

    data["align_opt"] = True

    for i in range(1, 6):
        try:
            data[f"xo{i}_f"] = pin[f"xo{i}_f"]
        except Exception:
            data[f"xo{i}_f"] = None
        try:
            data[f"xo{i}_s"] = pin[f"xo{i}_s"]
        except Exception:
            data[f"xo{i}_s"] = None

    try:
        data["max_cut_db"] = abs(float(data.get("max_cut_db", 15.0) or 15.0))
    except Exception:
        data["max_cut_db"] = 15.0

    for k, dv in [
        ("max_slope_db_per_oct", 24.0),
        ("max_slope_boost_db_per_oct", 0.0),
        ("max_slope_cut_db_per_oct", 0.0),
    ]:
        try:
            data[k] = max(0.0, float(data.get(k, dv) or dv))
        except Exception:
            data[k] = dv
    try:
        v = float(data.get("lvl_manual_db", 0.0) or 0.0)
        data["lvl_manual_db"] = v if math.isfinite(v) else 0.0
    except Exception:
        data["lvl_manual_db"] = 0.0

    try:
        data["gain"] = max(0.0, float(data.get("gain", 0.0) or 0.0))
    except Exception:
        data["gain"] = 0.0
    try:
        data["auto_mode_workers"] = int(float(data.get("auto_mode_workers", 0) or 0))
    except Exception:
        data["auto_mode_workers"] = 0

    v_raw = data.get("ir_export_window_mode", None)
    if v_raw is None or (isinstance(v_raw, str) and v_raw.strip() == ""):
        v_raw = data.get("ir_window_mode", "auto")
    v = str(v_raw or "auto").strip().lower()
    v = v if v in ("auto", "off", "rew_sym", "rew_asym") else "auto"
    data["ir_export_window_mode"] = v
    data["ir_window_mode"] = v

    am = str(data.get("ir_anchor_mode", "min_causal") or "min_causal").strip().lower()
    if am not in ("peak", "centroid", "min_causal"):
        am = "min_causal"
    data["ir_anchor_mode"] = am

    try:
        sh_raw = data.get("ir_export_window_shape", None)
        sh = str(sh_raw or "hann").strip().lower()
    except Exception:
        sh = "hann"
    if sh not in ("hann", "tukey"):
        sh = "hann"
    data["ir_export_window_shape"] = sh

    try:
        a = float(data.get("ir_export_tukey_alpha", 0.25))
    except Exception:
        a = 0.25
    if not math.isfinite(a):
        a = 0.25
    data["ir_export_tukey_alpha"] = max(0.0, min(1.0, float(a)))

    # Force asymmetric filter to use asymmetric export windowing with fixed Tukey alpha.
    try:
        if filter_type_short(str(data.get("filter_type", "") or "")) == "Asymmetric":
            data["ir_export_window_mode"] = "rew_asym"
            data["ir_window_mode"] = "rew_asym"
            data["ir_export_window_shape"] = "tukey"
            data["ir_export_tukey_alpha"] = 0.25
    except Exception:
        pass

    try:
        logger.info(
            f"UI pins: ir_export_window_mode={data.get('ir_export_window_mode')}, "
            f"shape={data.get('ir_export_window_shape')}, alpha={data.get('ir_export_tukey_alpha')}"
        )
    except Exception:
        pass

    try:
        if data.get("filter_smooth", None) is None and data.get("smoothing_level", None) is not None:
            data["filter_smooth"] = data.get("smoothing_level")
    except Exception:
        pass

    try:
        if data.get("plot_smoothing_level", None) is None:
            data["plot_smoothing_level"] = "Psychoacoustic"
    except Exception:
        pass
    return data


def log_df_smoothing_toggle(pin, logger) -> bool:
    try:
        df_on = bool(pin["df_smoothing"])
    except Exception:
        df_on = False
    try:
        logger.info(f"DF smoothing: {'ON' if df_on else 'OFF'}")
    except Exception:
        pass
    return df_on


def build_xos_hpf(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    xos: List[Dict[str, Any]] = []
    for i in range(1, 6):
        f_raw = data.get(f"xo{i}_f", None)
        if f_raw in (None, "", 0):
            continue
        try:
            f_hz = float(f_raw)
        except Exception:
            continue
        if not math.isfinite(f_hz) or f_hz <= 0:
            continue
        s_raw = data.get(f"xo{i}_s", 12)
        try:
            slope_db_oct = int(round(float(s_raw)))
        except Exception:
            slope_db_oct = 12
        if slope_db_oct <= 0:
            slope_db_oct = 12
        order = max(1, int(round(slope_db_oct / 6.0)))
        xos.append({"freq": f_hz, "order": order, "slope": slope_db_oct, "idx": i})
    xos.sort(key=lambda d: float(d.get("freq", 0.0)))

    hpf = (
        {"enabled": bool(data.get("hpf_enable")),
         "freq": data.get("hpf_freq"),
         "order": int(data.get("hpf_slope")) // 6}
        if bool(data.get("hpf_enable"))
        else None
    )
    return xos, hpf


def filter_type_short(filter_type: str) -> str:
    s = str(filter_type or "")
    if "Asymmetric" in s:
        return "Asymmetric"
    if "Min" in s:
        return "Minimum"
    if "Mixed" in s:
        return "Mixed"
    return "Linear"


def choose_target_rates(data: Dict[str, Any]) -> List[int]:
    if bool(data.get("multi_rate_opt")):
        return [44100, 48000, 88200, 96000, 176400, 192000]
    try:
        return [int(data.get("fs") or 44100)]
    except Exception:
        return [44100]


def choose_dash_fs(target_rates: List[int], *, multi_rate_on: bool, forced_plot_fs_hz: int) -> int:
    if not target_rates:
        return forced_plot_fs_hz
    dash_fs = int(forced_plot_fs_hz) if multi_rate_on else int(target_rates[0])
    if multi_rate_on and dash_fs not in target_rates:
        dash_fs = int(target_rates[0])
    return dash_fs


def detect_is_wav_source(data: Dict[str, Any], pin) -> bool:
    try:
        lp_l_s = str(data.get("local_path_l", "") or "").lower()
        lp_r_s = str(data.get("local_path_r", "") or "").lower()
    except Exception:
        lp_l_s, lp_r_s = "", ""

    try:
        up_l_s = (
            str(pin["file_l"].get("filename", "") or "").lower()
            if isinstance(pin.get("file_l", None), dict)
            else ""
        )
        up_r_s = (
            str(pin["file_r"].get("filename", "") or "").lower()
            if isinstance(pin.get("file_r", None), dict)
            else ""
        )
    except Exception:
        up_l_s, up_r_s = "", ""

    return (
        lp_l_s.endswith(".wav")
        or lp_r_s.endswith(".wav")
        or up_l_s.endswith(".wav")
        or up_r_s.endswith(".wav")
    )


def build_filter_config(
    *,
    FilterConfig_cls,
    fs_v: int,
    taps_v: int,
    data: Dict[str, Any],
    xos,
    hpf,
    hc_f,
    hc_m,
    pin,
) -> Any:
    """Rakentaa tai generoi: build filter config."""

    def _pin_get(key: str, default=None):
        """Sisainen apufunktio: pin get."""
        try:
            if hasattr(pin, "get"):
                v = pin.get(key, None)
                if v is not None:
                    return v
        except Exception:
            pass

        try:
            return pin[key]
        except Exception:
            pass

        try:
            v = getattr(pin, key)
            if v is not None:
                return v
        except Exception:
            pass

        return default

    def _as_float(v, default=0.0) -> float:
        try:
            x = float(v)
            return x if x == x else float(default)
        except Exception:
            return float(default)

    def _as_int(v, default=0) -> int:
        try:
            return int(float(v))
        except Exception:
            return int(default)
    def _as_bool_default(v, default: bool) -> bool:
        """Sisainen apufunktio: as bool with None/empty fallback."""
        if v is None:
            return bool(default)
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "":
                return bool(default)
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return bool(default)
            if len(v) == 1:
                return _as_bool_default(v[0], default)
        try:
            return bool(v)
        except Exception:
            return bool(default)
    def _as_float_allow_zero(v, default: float) -> float:
        """Sisainen apufunktio: as float allow zero."""
        if v is None:
            return float(default)
        if isinstance(v, str) and v.strip() == "":
            return float(default)
        return _as_float(v, default)

    def _as_float_or_none(v, default: Optional[float]) -> Optional[float]:
        """Sisainen apufunktio: as float or none."""
        if v is None:
            return default
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() == "none":
                return default
        try:
            x = float(v)
        except Exception:
            return default
        if not math.isfinite(x):
            return default
        return float(x)

    conf_pull_floor = _as_float_allow_zero(data.get("conf_pull_floor", None), 0.05)
    conf_pull_ceil  = _as_float_allow_zero(data.get("conf_pull_ceil", None), 0.95)
    conf_pull_max_hz = _as_float_or_none(data.get("conf_pull_max_hz", None), 200.0)
    conf_pull_gamma_cut   = _as_float_allow_zero(data.get("conf_pull_gamma_cut", None), 0.55)
    conf_pull_gamma_boost = _as_float_allow_zero(data.get("conf_pull_gamma_boost", None), 1.35)

    conf_pull_conf_smooth_sigma = _as_float_allow_zero(data.get("conf_pull_conf_smooth_sigma", None), 2.0)
    conf_pull_bass_floor_hz     = _as_float_allow_zero(data.get("conf_pull_bass_floor_hz", None), 120.0)
    conf_pull_bass_floor_min    = _as_float_allow_zero(data.get("conf_pull_bass_floor_min", None), 0.25)

    low_bass_cut_strength = _as_float_allow_zero(data.get("low_bass_cut_strength", None), 0.0)
    low_bass_cut_strength = float(max(0.0, min(1.0, low_bass_cut_strength)))
    mixed_excess_phase_strength = _as_float_allow_zero(data.get("excess_phase_strength", None), 0.9)
    mixed_low_full_hz = _as_float_allow_zero(data.get("low_freq_full_correction_hz", None), 140.0)
    mixed_high_none_hz = _as_float_allow_zero(data.get("high_freq_no_correction_hz", None), 900.0)
    enable_ir_pre_energy_guard = bool(data.get("enable_ir_pre_energy_guard", True))
    pre_energy_ratio_max = _as_float_allow_zero(data.get("pre_energy_ratio_max", None), 0.25)
    pre_energy_guard_strength = _as_float_allow_zero(data.get("pre_energy_guard_strength", None), 0.8)
    mixed_max_pre_db = _as_float_allow_zero(data.get("max_pre_ringing_db", None), -35.0)
    mixed_max_excess_delay_ms = _as_float_allow_zero(data.get("max_excess_delay_ms", None), 2.5)
    gd_grad_limit_ms_per_oct = _as_float_allow_zero(data.get("gd_grad_limit_ms_per_oct", None), 20.0)
    ir_anchor_mode = str(data.get("ir_anchor_mode", "min_causal") or "min_causal").strip().lower()
    if ir_anchor_mode not in ("peak", "centroid", "min_causal"):
        ir_anchor_mode = "min_causal"
    min_causal_ms = _as_float_allow_zero(data.get("min_causal_ms", None), 80.0)
    auto_asym_left_ratio = _as_float_allow_zero(data.get("auto_asym_left_ratio", None), 0.35)
    auto_asym_left_max_ms = _as_float_allow_zero(data.get("auto_asym_left_max_ms", None), 25.0)
    mixed_kwargs = {}
    if hasattr(FilterConfig_cls, "excess_phase_strength"):
        mixed_kwargs["excess_phase_strength"] = float(max(0.0, min(1.0, mixed_excess_phase_strength)))
    if hasattr(FilterConfig_cls, "low_freq_full_correction_hz"):
        mixed_kwargs["low_freq_full_correction_hz"] = float(max(20.0, mixed_low_full_hz))
    if hasattr(FilterConfig_cls, "high_freq_no_correction_hz"):
        mixed_kwargs["high_freq_no_correction_hz"] = float(max(20.0, mixed_high_none_hz))
    if hasattr(FilterConfig_cls, "enable_ir_pre_energy_guard"):
        mixed_kwargs["enable_ir_pre_energy_guard"] = bool(enable_ir_pre_energy_guard)
    if hasattr(FilterConfig_cls, "pre_energy_ratio_max"):
        mixed_kwargs["pre_energy_ratio_max"] = float(max(0.0, pre_energy_ratio_max))
    if hasattr(FilterConfig_cls, "pre_energy_guard_strength"):
        mixed_kwargs["pre_energy_guard_strength"] = float(np.clip(pre_energy_guard_strength, 0.0, 1.0))
    if hasattr(FilterConfig_cls, "max_pre_ringing_db"):
        mixed_kwargs["max_pre_ringing_db"] = float(min(0.0, mixed_max_pre_db))
    if hasattr(FilterConfig_cls, "max_excess_delay_ms"):
        mixed_kwargs["max_excess_delay_ms"] = float(max(0.0, mixed_max_excess_delay_ms))
    if hasattr(FilterConfig_cls, "gd_grad_limit_ms_per_oct"):
        mixed_kwargs["gd_grad_limit_ms_per_oct"] = float(max(0.0, gd_grad_limit_ms_per_oct))
    if hasattr(FilterConfig_cls, "ir_anchor_mode"):
        mixed_kwargs["ir_anchor_mode"] = str(ir_anchor_mode)
    if hasattr(FilterConfig_cls, "min_causal_ms"):
        mixed_kwargs["min_causal_ms"] = float(max(0.0, min_causal_ms))
    if hasattr(FilterConfig_cls, "auto_asym_left_ratio"):
        mixed_kwargs["auto_asym_left_ratio"] = float(np.clip(auto_asym_left_ratio, 0.0, 1.0))
    if hasattr(FilterConfig_cls, "auto_asym_left_max_ms"):
        mixed_kwargs["auto_asym_left_max_ms"] = float(max(0.0, auto_asym_left_max_ms))
    lb_en = bool(data.get("low_bass_cut_enable", True))
    lb_raw = data.get("low_bass_cut_hz", "")
    if (not lb_en) or (lb_raw in (None, "", "None")):
        lb_hz = 0.0
    else:
        lb_hz = _as_float(lb_raw, 40.0)
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    auto_mode_locked = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    df_smoothing = _as_bool_default(
        data.get("df_smoothing", False)
        if auto_mode_locked
        else _pin_get("df_smoothing", data.get("df_smoothing", False)),
        False,
    )
    bass_smooth_adaptive = _as_bool_default(
        _pin_get("bass_smooth_adaptive", data.get("bass_smooth_adaptive", True)),
        True,
    )
    bass_smooth_hz = _as_float_allow_zero(data.get("bass_smooth_hz", None), 200.0)
    bass_smooth_sigma_scale = _as_float_allow_zero(data.get("bass_smooth_sigma_scale", None), 1.4)
    bass_smooth_conf_floor = _as_float_allow_zero(data.get("bass_smooth_conf_floor", None), 0.3)
    mid_refit_enable = _as_bool_default(
        _pin_get("mid_refit_enable", data.get("mid_refit_enable", True)),
        True,
    )
    mid_refit_hz_lo = _as_float_allow_zero(data.get("mid_refit_hz_lo", None), 200.0)
    mid_refit_hz_hi = _as_float_allow_zero(data.get("mid_refit_hz_hi", None), 2000.0)
    mid_refit_k = _as_float_allow_zero(data.get("mid_refit_k", None), 0.45)
    mid_refit_smooth_oct = _as_float_allow_zero(data.get("mid_refit_smooth_oct", None), 0.60)
    mid_refit_conf_min_avg = _as_float_allow_zero(data.get("mid_refit_conf_min_avg", None), 0.20)
    # process_run may enforce this in `data`; keep data as highest-priority source.
    bass_adaptive_isolation_mode = _as_bool_default(
        data.get("bass_adaptive_isolation_mode", _pin_get("bass_adaptive_isolation_mode", False)),
        False,
    )
    bass_smooth_kwargs = {}
    if hasattr(FilterConfig_cls, "bass_smooth_adaptive"):
        bass_smooth_kwargs["bass_smooth_adaptive"] = bool(bass_smooth_adaptive)
    if hasattr(FilterConfig_cls, "bass_smooth_hz"):
        bass_smooth_kwargs["bass_smooth_hz"] = float(max(20.0, bass_smooth_hz))
    if hasattr(FilterConfig_cls, "bass_smooth_sigma_scale"):
        bass_smooth_kwargs["bass_smooth_sigma_scale"] = float(max(1.0, bass_smooth_sigma_scale))
    if hasattr(FilterConfig_cls, "bass_smooth_conf_floor"):
        bass_smooth_kwargs["bass_smooth_conf_floor"] = float(np.clip(bass_smooth_conf_floor, 0.05, 1.0))
    if hasattr(FilterConfig_cls, "bass_adaptive_isolation_mode"):
        bass_smooth_kwargs["bass_adaptive_isolation_mode"] = bool(bass_adaptive_isolation_mode)
    if hasattr(FilterConfig_cls, "mid_refit_enable"):
        bass_smooth_kwargs["mid_refit_enable"] = bool(mid_refit_enable)
    if hasattr(FilterConfig_cls, "mid_refit_hz_lo"):
        bass_smooth_kwargs["mid_refit_hz_lo"] = float(max(20.0, mid_refit_hz_lo))
    if hasattr(FilterConfig_cls, "mid_refit_hz_hi"):
        bass_smooth_kwargs["mid_refit_hz_hi"] = float(max(max(20.0, mid_refit_hz_lo) + 1.0, mid_refit_hz_hi))
    if hasattr(FilterConfig_cls, "mid_refit_k"):
        bass_smooth_kwargs["mid_refit_k"] = float(np.clip(mid_refit_k, 0.0, 1.0))
    if hasattr(FilterConfig_cls, "mid_refit_smooth_oct"):
        bass_smooth_kwargs["mid_refit_smooth_oct"] = float(np.clip(mid_refit_smooth_oct, 1.0 / 192.0, 1.0))
    if hasattr(FilterConfig_cls, "mid_refit_conf_min_avg"):
        bass_smooth_kwargs["mid_refit_conf_min_avg"] = float(np.clip(mid_refit_conf_min_avg, 0.0, 1.0))
    bass_boost_cap_enable = _as_bool_default(
        _pin_get("bass_boost_cap_enable", data.get("bass_boost_cap_enable", True)),
        True,
    )
    bass_boost_cap_hz = _as_float_allow_zero(data.get("bass_boost_cap_hz", None), 200.0)
    bass_boost_cap_extra_db = _as_float_allow_zero(data.get("bass_boost_cap_extra_db", None), 2.0)
    bass_boost_cap_conf_min = _as_float_allow_zero(data.get("bass_boost_cap_conf_min", None), 0.55)
    bass_boost_post_restore_enable = _as_bool_default(
        _pin_get("bass_boost_post_restore_enable", data.get("bass_boost_post_restore_enable", True)),
        True,
    )
    bass_boost_post_restore_strength = _as_float_allow_zero(data.get("bass_boost_post_restore_strength", None), 0.60)
    bass_boost_cap_kwargs = {}
    if hasattr(FilterConfig_cls, "bass_boost_cap_enable"):
        bass_boost_cap_kwargs["bass_boost_cap_enable"] = bool(bass_boost_cap_enable)
    if hasattr(FilterConfig_cls, "bass_boost_cap_hz"):
        bass_boost_cap_kwargs["bass_boost_cap_hz"] = float(max(20.0, bass_boost_cap_hz))
    if hasattr(FilterConfig_cls, "bass_boost_cap_extra_db"):
        bass_boost_cap_kwargs["bass_boost_cap_extra_db"] = float(max(0.0, bass_boost_cap_extra_db))
    if hasattr(FilterConfig_cls, "bass_boost_cap_conf_min"):
        bass_boost_cap_kwargs["bass_boost_cap_conf_min"] = float(np.clip(bass_boost_cap_conf_min, 0.0, 0.99))
    if hasattr(FilterConfig_cls, "bass_boost_post_restore_enable"):
        bass_boost_cap_kwargs["bass_boost_post_restore_enable"] = bool(bass_boost_post_restore_enable)
    if hasattr(FilterConfig_cls, "bass_boost_post_restore_strength"):
        bass_boost_cap_kwargs["bass_boost_post_restore_strength"] = float(np.clip(bass_boost_post_restore_strength, 0.0, 1.0))
    enable_afdw = _as_bool_default(
        data.get("enable_afdw", False)
        if auto_mode_locked
        else _pin_get("enable_afdw", data.get("enable_afdw", False)),
        False,
    )
    enable_tdc = _as_bool_default(
        data.get("enable_tdc", False)
        if auto_mode_locked
        else _pin_get("enable_tdc", data.get("enable_tdc", False)),
        False,
    )
    tdc_max_red = _as_float(
        data.get("tdc_max_reduction_db", 9.0)
        if auto_mode_locked
        else _pin_get("tdc_max_reduction_db", data.get("tdc_max_reduction_db", 9.0)),
        9.0,
    )
    tdc_slope = _as_float(
        data.get("tdc_slope_db_per_oct", 0.0)
        if auto_mode_locked
        else _pin_get("tdc_slope_db_per_oct", data.get("tdc_slope_db_per_oct", 0.0)),
        0.0,
    )
    filter_smooth = _as_int(
        (
            data.get("filter_smooth", data.get("smoothing_level", 12))
            if auto_mode_locked
            else _pin_get("filter_smooth", data.get("filter_smooth", data.get("smoothing_level", 12)))
        ),
        12
    )
    comparison_mode = bool(data.get("comparison_mode", True))
    lvl_mode = str(data.get("lvl_mode", "Auto") or "Auto")
    if mode_u in ("BASIC", "AUTO"):
        lvl_mode = "Auto"
    sls = str(data.get("stereo_link_strategy", "auto") or "").strip().lower()
    if sls not in ("shared", "hybrid", "auto"):
        sls = "auto"
    
    cfg = FilterConfig_cls(
        fs=int(fs_v),
        num_taps=int(taps_v),
        df_smoothing=bool(df_smoothing),
        **({"comparison_mode": bool(comparison_mode)} if hasattr(FilterConfig_cls, "comparison_mode") else {}),
        filter_type_str=data["filter_type"],
        mixed_split_freq=data["mixed_freq"],
        global_gain_db=0.0,
        mag_c_min=data["mag_c_min"],
        mag_c_max=data["mag_c_max"],
        max_boost_db=data["max_boost"],
        max_cut_db=data.get("max_cut_db", 30.0),
        max_slope_db_per_oct=data.get("max_slope_db_per_oct", 24.0),
        max_slope_boost_db_per_oct=data.get("max_slope_boost_db_per_oct", 0.0),
        max_slope_cut_db_per_oct=data.get("max_slope_cut_db_per_oct", 0.0),
        phase_limit=data["phase_limit"],
        phase_safe_2058=False,
        enable_mag_correction=bool(data.get("mag_correct", True)),
        unsafe_raw_dsp=bool(data.get("unsafe_raw_dsp", False)),
        lvl_mode=lvl_mode,
        reg_strength=float(data.get("reg_strength", 30.0)),
        do_normalize=bool(data["normalize_opt"]),
        exc_prot=bool(data["exc_prot"]),
        exc_freq=data["exc_freq"],
        low_bass_cut_hz=float(lb_hz),
        ir_window_ms=data.get("ir_window_right", 500.0),
        ir_window_ms_left=data.get("ir_window_left", 85.0),
        ir_export_window_mode=data.get("ir_export_window_mode", "auto"),
        enable_afdw=bool(enable_afdw),
        enable_tdc=bool(enable_tdc),
        tdc_strength=data.get("tdc_strength", 50.0),
        tdc_max_reduction_db=float(tdc_max_red),
        tdc_slope_db_per_oct=float(tdc_slope),
        plot_smoothing_level=data.get("plot_smoothing_level", "Psychoacoustic"),
        filter_smooth=int(filter_smooth),
        fdw_cycles=data["fdw_cycles"],
        lvl_manual_db=data["lvl_manual_db"],
        lvl_min=data["lvl_min"],
        lvl_max=data["lvl_max"],
        lvl_algo=data["lvl_algo"],
        stereo_link=bool(data.get("stereo_link", False)),
        stereo_link_strategy=str(sls),
        crossovers=xos,
        hpf_settings=hpf,
        house_freqs=hc_f,
        house_mags=hc_m,
        trans_width=data.get("trans_width", 100.0),
        bass_first_ai=bool(data.get("bass_first_ai", False)),
        bass_first_mode_max_hz=float(data.get("bass_first_mode_max_hz", 200.0) or 200.0),
        conf_pull_floor=float(_as_float_allow_zero(data.get("conf_pull_floor", None), 0.05)),
        conf_pull_ceil=float(_as_float_allow_zero(data.get("conf_pull_ceil", None), 0.95)),
        conf_pull_max_hz=_as_float_or_none(data.get("conf_pull_max_hz", None), 200.0),
        conf_pull_gamma_cut=float(_as_float_allow_zero(data.get("conf_pull_gamma_cut", None), 0.55)),
        conf_pull_gamma_boost=float(_as_float_allow_zero(data.get("conf_pull_gamma_boost", None), 1.35)),
        conf_pull_conf_smooth_sigma=float(_as_float_allow_zero(data.get("conf_pull_conf_smooth_sigma", None), 2.0)),
        conf_pull_bass_floor_hz=float(_as_float_allow_zero(data.get("conf_pull_bass_floor_hz", None), 120.0)),
        conf_pull_bass_floor_min=float(_as_float_allow_zero(data.get("conf_pull_bass_floor_min", None), 0.25)),
        conf_pull_bass_boost_floor_hz=float(_as_float_allow_zero(data.get("conf_pull_bass_boost_floor_hz", None), 200.0)),
        conf_pull_bass_boost_floor_min=float(_as_float_allow_zero(data.get("conf_pull_bass_boost_floor_min", None), 0.45)),
        conf_pull_bass_boost_restore=float(_as_float_allow_zero(data.get("conf_pull_bass_boost_restore", None), 0.55)),
        low_bass_cut_enable=bool(data.get("low_bass_cut_enable", True)),
        low_bass_cut_strength=float(max(0.0, min(1.0, _as_float_allow_zero(data.get("low_bass_cut_strength", None), 0.0)))),
        **bass_smooth_kwargs,
        **bass_boost_cap_kwargs,
        **mixed_kwargs,
    )
    try:
        setattr(cfg, "auto_gain_margin_db", float(max(0.0, _as_float_allow_zero(data.get("gain", None), 0.0))))
    except Exception:
        pass
    logger.info(f"UI raw: conf_pull_floor pin={data.get('conf_pull_floor')}, low_bass_cut_strength pin={data.get('low_bass_cut_strength')}")
    
    try:
        setattr(cfg, "enable_residual_pass", bool(data.get("enable_residual_pass", False)))
    except Exception:
        pass

    try:
        setattr(cfg, "lvl_force_window", None)
        setattr(cfg, "lvl_force_offset_db", None)
    except Exception:
        pass

    return cfg
