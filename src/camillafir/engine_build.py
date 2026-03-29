from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .config.camillafir_pipeline import build_filter_config, build_xos_hpf, detect_is_wav_source
from .config.mode_policy import apply_mode_to_cfg
from .config.models import FilterConfig

logger = logging.getLogger("CamillaFIR")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def build_config(
    ui_data: dict,
    preset: dict | None = None,
    *,
    fs_v: int | None = None,
    taps_v: int | None = None,
    xos: list[dict[str, Any]] | None = None,
    hpf: dict[str, Any] | None = None,
    hc_f=None,
    hc_m=None,
    filter_config_cls=FilterConfig,
    max_safe_boost: float = 8.0,
) -> FilterConfig:
    """
    Build a FilterConfig via existing pipeline builders and mode clamping.

    This function intentionally delegates to `config.camillafir_pipeline`
    to keep config behavior unchanged.
    """
    data = dict(ui_data or {})
    if isinstance(preset, dict) and preset:
        data.update(preset)

    if xos is None or hpf is None:
        xos_b, hpf_b = build_xos_hpf(data)
        if xos is None:
            xos = xos_b
        if hpf is None:
            hpf = hpf_b

    fs_eff = int(fs_v if fs_v is not None else _as_int(data.get("fs", 44100), 44100))
    taps_eff = int(taps_v if taps_v is not None else _as_int(data.get("taps", 65536), 65536))

    cfg = build_filter_config(
        FilterConfig_cls=filter_config_cls,
        fs_v=fs_eff,
        taps_v=taps_eff,
        data=data,
        xos=xos,
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
    )

    mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    try:
        apply_mode_to_cfg(cfg, mode_u, apply_defaults=False)
    except Exception as exc:
        logger.warning(f"Mode clamp apply failed ({mode_u}): {exc}")
    try:
        unsafe_raw_req = bool(data.get("unsafe_raw_dsp", False))
    except Exception:
        unsafe_raw_req = False
    unsafe_raw = bool(unsafe_raw_req and mode_u == "ADVANCED")
    try:
        setattr(cfg, "unsafe_raw_dsp", bool(unsafe_raw))
    except Exception:
        pass

    irw_raw = data.get("ir_export_window_mode", data.get("ir_window_mode", "auto"))
    irw_mode = str(irw_raw or "auto").strip().lower()
    if irw_mode not in ("auto", "off", "rew_sym", "rew_asym"):
        irw_mode = "auto"
    try:
        sh = str(data.get("ir_export_window_shape", "hann") or "hann").strip().lower()
    except Exception:
        sh = "hann"
    if sh not in ("hann", "tukey"):
        sh = "hann"
    tukey_alpha = float(np.clip(_as_float(data.get("ir_export_tukey_alpha", 0.25), 0.25), 0.0, 1.0))

    try:
        filter_type_s = str(getattr(cfg, "filter_type_str", data.get("filter_type", "")) or "").strip().lower()
    except Exception:
        filter_type_s = ""
    if "asym" in filter_type_s:
        irw_mode = "rew_asym"
        sh = "tukey"
        tukey_alpha = 0.25

    setattr(cfg, "ir_export_window_mode", irw_mode)
    setattr(cfg, "ir_export_window_shape", sh)
    setattr(cfg, "ir_export_tukey_alpha", float(tukey_alpha))

    try:
        setattr(cfg, "ir_window", float(data.get("ir_window", getattr(cfg, "ir_window", 500.0)) or 500.0))
        setattr(cfg, "ir_window_left", float(data.get("ir_window_left", getattr(cfg, "ir_window_left", 120.0)) or 120.0))
    except Exception:
        pass
    try:
        ir_anchor_mode = str(data.get("ir_anchor_mode", getattr(cfg, "ir_anchor_mode", "min_causal")) or "min_causal").strip().lower()
        if ir_anchor_mode not in ("peak", "centroid", "min_causal"):
            ir_anchor_mode = "min_causal"
        setattr(cfg, "ir_anchor_mode", ir_anchor_mode)
        setattr(cfg, "min_causal_ms", float(max(0.0, _as_float(data.get("min_causal_ms", getattr(cfg, "min_causal_ms", 80.0)), 80.0))))
        setattr(
            cfg,
            "auto_asym_left_ratio",
            float(np.clip(_as_float(data.get("auto_asym_left_ratio", getattr(cfg, "auto_asym_left_ratio", 0.35)), 0.35), 0.0, 1.0)),
        )
        setattr(
            cfg,
            "auto_asym_left_max_ms",
            float(max(0.0, _as_float(data.get("auto_asym_left_max_ms", getattr(cfg, "auto_asym_left_max_ms", 25.0)), 25.0))),
        )
    except Exception:
        pass
    try:
        setattr(
            cfg,
            "enable_ir_pre_energy_guard",
            bool(data.get("enable_ir_pre_energy_guard", getattr(cfg, "enable_ir_pre_energy_guard", True))),
        )
        setattr(
            cfg,
            "pre_energy_ratio_max",
            float(max(0.0, _as_float(data.get("pre_energy_ratio_max", getattr(cfg, "pre_energy_ratio_max", 0.25)), 0.25))),
        )
        setattr(
            cfg,
            "pre_energy_guard_strength",
            float(np.clip(_as_float(data.get("pre_energy_guard_strength", getattr(cfg, "pre_energy_guard_strength", 0.8)), 0.8), 0.0, 1.0)),
        )
    except Exception:
        pass

    try:
        user_max_boost = float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
        setattr(cfg, "max_boost_db_user", user_max_boost)
        setattr(cfg, "max_safe_boost_db", float(max_safe_boost))
        if (not unsafe_raw) and user_max_boost > 0.0 and float(max_safe_boost) > 0.0:
            eff = min(user_max_boost, float(max_safe_boost))
            if eff < user_max_boost - 1e-9:
                logger.info(
                    "Safety cap: max_boost_db "
                    f"user={user_max_boost:.2f} dB -> effective={eff:.2f} dB "
                    f"(MAX_SAFE_BOOST={float(max_safe_boost):.2f} dB)"
                )
            setattr(cfg, "max_boost_db", float(eff))
        elif unsafe_raw:
            logger.warning("UNSAFE Raw DSP: bypassing MAX_SAFE_BOOST safety cap")
    except Exception:
        pass
    if unsafe_raw:
        try:
            setattr(cfg, "max_boost_db", float(max(120.0, float(getattr(cfg, "max_boost_db", 0.0) or 0.0))))
            setattr(cfg, "max_cut_db", float(max(120.0, abs(float(getattr(cfg, "max_cut_db", 0.0) or 0.0)))))
            setattr(cfg, "max_slope_db_per_oct", 0.0)
            setattr(cfg, "max_slope_boost_db_per_oct", 0.0)
            setattr(cfg, "max_slope_cut_db_per_oct", 0.0)
            setattr(cfg, "reg_strength", 0.0)
            setattr(cfg, "low_bass_cut_enable", False)
            setattr(cfg, "low_bass_cut_hz", 0.0)
            setattr(cfg, "low_bass_cut_strength", 0.0)
            setattr(cfg, "exc_prot", False)
            setattr(cfg, "bass_boost_cap_enable", False)
            setattr(cfg, "bass_boost_post_restore_enable", False)
            setattr(cfg, "bass_smooth_adaptive", False)
            setattr(cfg, "enable_ir_pre_energy_guard", False)
            logger.warning("UNSAFE Raw DSP: guard rails disabled (FOR TEST USE ONLY)")
        except Exception:
            pass

    is_wav = False
    try:
        if "_is_wav_source" in data:
            is_wav = bool(data.get("_is_wav_source"))
        else:
            is_wav = detect_is_wav_source(data)
    except Exception:
        is_wav = False
    try:
        setattr(cfg, "is_wav_source", bool(is_wav))
    except Exception:
        pass

    return cfg
