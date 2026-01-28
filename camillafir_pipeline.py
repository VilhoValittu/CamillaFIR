# camillafir_pipeline.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def collect_ui_data(pin) -> Dict[str, Any]:
    """
    Read relevant pins into a plain dict and normalize checkbox pins (list -> bool).
    NOTE: pin is injected to keep this module testable without PyWebIO.
    """
    p_keys = [
        "mode", "fs", "taps", "filter_type", "mixed_freq", "gain", "hc_mode",
        "mag_c_min", "mag_c_max", "max_boost", "max_cut_db", "max_slope_db_per_oct",
        "max_slope_boost_db_per_oct", "max_slope_cut_db_per_oct", "phase_limit", "phase_safe_2058", "mag_correct",
        "lvl_mode", "reg_strength", "normalize_opt", "align_opt",
        "stereo_link", "exc_prot", "exc_freq", "low_bass_cut_hz", "hpf_enable", "hpf_freq",
        "hpf_slope", "multi_rate_opt", "ir_window", "ir_window_left",
        "local_path_l", "local_path_r", "fmt", "lvl_manual_db",
        "lvl_min", "lvl_max", "lvl_algo", "smoothing_type", "fdw_cycles",
        "trans_width", "smoothing_level", "enable_tdc", "tdc_strength", "tdc_max_reduction_db",
        "tdc_slope_db_per_oct", "enable_afdw", "df_smoothing", "comparison_mode",
        "bass_first_ai", "bass_first_mode_max_hz",
        "local_path_house",
    ]

    data: Dict[str, Any] = {}
    for k in p_keys:
        try:
            data[k] = pin[k]
        except Exception:
            data[k] = None

    # normalize checkbox pins saved as [] / [True]
    for k in [
        "mag_correct", "normalize_opt", "align_opt", "multi_rate_opt",
        "stereo_link", "exc_prot", "hpf_enable", "df_smoothing",
        "comparison_mode", "bass_first_ai", "phase_safe_2058",
        "enable_tdc", "enable_afdw",
    ]:
        try:
            if isinstance(data.get(k, None), list):
                data[k] = bool(data[k])
        except Exception:
            pass

    # XO pins
    for i in range(1, 6):
        try:
            data[f"xo{i}_f"] = pin[f"xo{i}_f"]
        except Exception:
            data[f"xo{i}_f"] = None
        try:
            data[f"xo{i}_s"] = pin[f"xo{i}_s"]
        except Exception:
            data[f"xo{i}_s"] = None

    # numeric clamps (same behavior as camillafir.py)
    try:
        data["max_cut_db"] = abs(float(data.get("max_cut_db", 15.0) or 15.0))
    except Exception:
        data["max_cut_db"] = 15.0

    for k, dv in [
        ("max_slope_db_per_oct", 24.0),
        ("max_slope_boost_db_per_oct", 0.0),
        ("max_slope_cut_db_per_oct", 0.0),
        ("lvl_manual_db", 75.0),
    ]:
        try:
            data[k] = max(0.0, float(data.get(k, dv) or dv))
        except Exception:
            data[k] = dv

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
    xos = [{"freq": data[f"xo{i}_f"], "order": int(data[f"xo{i}_s"]) // 6}
           for i in range(1, 6) if data.get(f"xo{i}_f")]
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
        or str(data.get("fmt", "")).upper() == "WAV"
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
    """
    Construct FilterConfig instance without importing camillafir/models inside this module.
    """
    cfg = FilterConfig_cls(
        fs=int(fs_v),
        num_taps=int(taps_v),
        df_smoothing=bool(pin["df_smoothing"]),
        **({"comparison_mode": True} if hasattr(FilterConfig_cls, "comparison_mode") else {}),
        filter_type_str=data["filter_type"],
        mixed_split_freq=data["mixed_freq"],
        global_gain_db=data["gain"],
        mag_c_min=data["mag_c_min"],
        mag_c_max=data["mag_c_max"],
        max_boost_db=data["max_boost"],
        max_cut_db=data.get("max_cut_db", 30.0),
        max_slope_db_per_oct=data.get("max_slope_db_per_oct", 24.0),
        max_slope_boost_db_per_oct=data.get("max_slope_boost_db_per_oct", 0.0),
        max_slope_cut_db_per_oct=data.get("max_slope_cut_db_per_oct", 0.0),
        phase_limit=data["phase_limit"],
        phase_safe_2058=bool(data.get("phase_safe_2058", False)),
        enable_mag_correction=bool(data.get("mag_correct", True)),
        lvl_mode=data["lvl_mode"],
        reg_strength=float(data.get("reg_strength", 30.0)),
        do_normalize=bool(data["normalize_opt"]),
        exc_prot=bool(data["exc_prot"]),
        exc_freq=data["exc_freq"],
        low_bass_cut_hz=float(data.get("low_bass_cut_hz", 40.0) or 40.0),
        ir_window_ms=data["ir_window"],
        ir_window_ms_left=data.get("ir_window_left", 100.0),
        enable_afdw=bool(pin.enable_afdw),
        enable_tdc=bool(pin.enable_tdc),
        tdc_strength=data.get("tdc_strength", 50.0),
        tdc_max_reduction_db=float(pin["tdc_max_reduction_db"]),
        tdc_slope_db_per_oct=float(pin["tdc_slope_db_per_oct"]),
        smoothing_type=data["smoothing_type"],
        fdw_cycles=data["fdw_cycles"],
        lvl_manual_db=data["lvl_manual_db"],
        lvl_min=data["lvl_min"],
        lvl_max=data["lvl_max"],
        lvl_algo=data["lvl_algo"],
        stereo_link=bool(data.get("stereo_link", False)),
        smoothing_level=int(pin.smoothing_level),
        crossovers=xos,
        hpf_settings=hpf,
        house_freqs=hc_f,
        house_mags=hc_m,
        trans_width=data.get("trans_width", 100.0),
        bass_first_ai=bool(data.get("bass_first_ai", False)),
        bass_first_mode_max_hz=float(data.get("bass_first_mode_max_hz", 200.0) or 200.0),
    )
    return cfg
