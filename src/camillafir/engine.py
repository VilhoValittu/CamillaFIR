from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .common.comparison_stats import _make_comparison_stats
from .common.result_postprocess import (
    _ensure_scoring_keys,
    _inject_filter_gd_stats,
    _inject_filter_mags_for_ui,
    _irwin_tag,
    _postpolish_wav_filter_ir,
    _shift_zeropad_1d,
)
from .config.camillafir_pipeline import (
    build_filter_config,
    build_xos_hpf,
    detect_is_wav_source,
)
from .config.models import FilterConfig
from .config.results import FilterResult
from .dsp import camillafir_dsp as dsp
from .config.mode_policy import apply_mode_to_cfg

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


def _phase_from_ir(ir: np.ndarray, fs: int, freq_axis: np.ndarray) -> np.ndarray:
    x = np.asarray(ir, dtype=float).reshape(-1)
    f_axis = np.asarray(freq_axis, dtype=float).reshape(-1)
    if x.size < 8 or f_axis.size == 0 or int(fs) <= 0:
        return np.zeros_like(f_axis, dtype=float)
    h = np.fft.rfft(x)
    f_fft = np.fft.rfftfreq(x.size, d=1.0 / float(fs))
    p_deg = np.rad2deg(np.unwrap(np.angle(h)))
    f_q = np.clip(f_axis, float(np.min(f_fft)), float(np.max(f_fft)))
    return np.interp(f_q, f_fft, p_deg).astype(float)


def _to_axis(arr: Any, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(arr if arr is not None else [], dtype=float)
    if out.size > 1:
        return out
    return np.asarray(fallback if fallback is not None else [], dtype=float)


def _resample_to_axis(values: Any, f_src: np.ndarray, f_dst: np.ndarray) -> np.ndarray:
    v = np.asarray(values if values is not None else [], dtype=float)
    fs = np.asarray(f_src if f_src is not None else [], dtype=float)
    fd = np.asarray(f_dst if f_dst is not None else [], dtype=float)
    if fd.size == 0:
        return np.asarray([], dtype=float)
    if v.size == fd.size and fd.size > 0:
        return v.astype(float, copy=False)
    if v.size < 2 or fs.size < 2:
        return np.zeros_like(fd, dtype=float)
    f_q = np.clip(fd, float(np.min(fs)), float(np.max(fs)))
    return np.interp(f_q, fs, v).astype(float)


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

    # Hard-force asymmetric filter windowing profile.
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
            # Disable practical guard rails in magnitude path for algorithm testing.
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


def run_pipeline(
    cfg: FilterConfig,
    measurements: dict,
    *,
    debug: bool = False,
    include_response_arrays: bool = True,
) -> FilterResult:
    """
    Execute one DSP pipeline run and return normalized results.

    `measurements` must contain L/R frequency, magnitude and phase arrays:
    `f_l,m_l,p_l,f_r,m_r,p_r`.
    """
    if not isinstance(measurements, dict):
        raise TypeError("measurements must be a dict")

    f_l = _to_axis(measurements.get("f_l"), np.asarray([], dtype=float))
    m_l = _to_axis(measurements.get("m_l"), np.asarray([], dtype=float))
    p_l = _to_axis(measurements.get("p_l"), np.asarray([], dtype=float))
    f_r = _to_axis(measurements.get("f_r"), np.asarray([], dtype=float))
    m_r = _to_axis(measurements.get("m_r"), np.asarray([], dtype=float))
    p_r = _to_axis(measurements.get("p_r"), np.asarray([], dtype=float))
    if min(f_l.size, m_l.size, p_l.size, f_r.size, m_r.size, p_r.size) == 0:
        raise ValueError("Incomplete measurement data for pipeline run")

    data = dict(measurements.get("ui_data") or {})
    hc_f = measurements.get("hc_f")
    hc_m = measurements.get("hc_m")
    comparison_mode = bool(data.get("comparison_mode", getattr(cfg, "comparison_mode", False)))

    warnings: list[str] = []

    if bool(getattr(cfg, "stereo_link", False)):
        l_imp, l_st, r_imp, r_st = dsp.generate_filter_pair(f_l, m_l, p_l, f_r, m_r, p_r, cfg)
    else:
        l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, cfg)
        r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, cfg)

    is_wav = bool(measurements.get("is_wav_source", getattr(cfg, "is_wav_source", False)))
    rt_rel = 1.0 if is_wav else 0.25
    rt_src = "WAV" if is_wav else "TXT/REW"
    if isinstance(l_st, dict):
        l_st["rt60_reliability"] = float(rt_rel)
        l_st["rt60_source"] = rt_src
    if isinstance(r_st, dict):
        r_st["rt60_reliability"] = float(rt_rel)
        r_st["rt60_source"] = rt_src

    l_st = _ensure_scoring_keys(l_st, f_l, m_l, hc_f, hc_m)
    r_st = _ensure_scoring_keys(r_st, f_r, m_r, hc_f, hc_m)

    if comparison_mode:
        try:
            l_st = _make_comparison_stats(l_st, int(cfg.fs), int(cfg.num_taps))
            r_st = _make_comparison_stats(r_st, int(cfg.fs), int(cfg.num_taps))
        except Exception as exc:
            warnings.append(f"comparison_stats_failed: {exc}")
            logger.warning(f"Comparison-mode stats failed: {exc}")

    align_method = "peak"
    d_peak = int(np.argmax(np.abs(l_imp)) - np.argmax(np.abs(r_imp)))
    d_delay: int | None = None
    try:
        dl = l_st.get("delay_samples", None) if isinstance(l_st, dict) else None
        dr = r_st.get("delay_samples", None) if isinstance(r_st, dict) else None
        if dl is not None and dr is not None:
            d_delay = int(round(float(dr))) - int(round(float(dl)))
    except Exception:
        d_delay = None

    if d_delay is None:
        d_s = d_peak
        align_method = "peak"
    else:
        d_s = int(d_delay)
        align_method = "delay_samples"
        try:
            guard_samples = 8
            if abs(int(d_delay) - int(d_peak)) > int(guard_samples):
                d_s = int(d_peak)
                align_method = "peak_guard"
                logger.info(
                    f"Alignment guard: delay_samples={int(d_delay)} vs peak={int(d_peak)} "
                    f"(>{guard_samples} samp) -> using peak"
                )
        except Exception:
            pass

    if d_s > 0:
        r_imp = _shift_zeropad_1d(r_imp, d_s)
    elif d_s < 0:
        l_imp = _shift_zeropad_1d(l_imp, -d_s)

    wav_like_fft_grid = False
    try:
        fx = np.asarray(f_l if f_l is not None else [], dtype=float)
        if fx.size > 1024 and abs(float(fx[0])) < 1e-9:
            df = float(np.median(np.diff(fx[: min(int(fx.size), 4096)])))
            if np.isfinite(df) and (0.0 < df < 2.0):
                wav_like_fft_grid = True
    except Exception:
        wav_like_fft_grid = False

    if bool(is_wav) or bool(wav_like_fft_grid):
        try:
            mc_min = _as_float(getattr(cfg, "mag_c_min", data.get("mag_c_min", 10.0)), 10.0)
            mc_max = _as_float(getattr(cfg, "mag_c_max", data.get("mag_c_max", 230.0)), 230.0)
            tr_w = _as_float(getattr(cfg, "trans_width", data.get("trans_width", 100.0)), 100.0)

            l_imp = _postpolish_wav_filter_ir(
                l_imp,
                int(cfg.fs),
                mag_c_min=mc_min,
                mag_c_max=mc_max,
                trans_width=tr_w,
            )
            r_imp = _postpolish_wav_filter_ir(
                r_imp,
                int(cfg.fs),
                mag_c_min=mc_min,
                mag_c_max=mc_max,
                trans_width=tr_w,
            )
            logger.info(
                f"WAV final IR polish applied at {int(cfg.fs)} Hz "
                f"(zone approx {max(mc_min, mc_max - 0.95 * tr_w):.0f}-{mc_max + 1.45 * tr_w:.0f} Hz, "
                f"is_wav={bool(is_wav)}, wav_like_fft_grid={bool(wav_like_fft_grid)})"
            )
        except Exception as exc:
            warnings.append(f"wav_postpolish_failed: {exc}")
            logger.warning(f"WAV final IR polish failed: {exc}")

    if isinstance(l_st, dict) and isinstance(r_st, dict):
        try:
            delay_ms = round((float(d_s) / float(cfg.fs)) * 1000.0, 3)
            distance_cm = round((delay_ms / 1000.0) * 34300.0, 2)
            gain_diff = round(float(l_st.get("offset_db", 0.0)) - float(r_st.get("offset_db", 0.0)), 2)
            l_st["auto_align"] = {
                "delay_ms": delay_ms,
                "distance_cm": distance_cm,
                "gain_diff_db": gain_diff,
                "method": str(align_method),
            }
        except Exception:
            pass

    _inject_filter_gd_stats(l_st, l_imp, int(cfg.fs))
    _inject_filter_gd_stats(r_st, r_imp, int(cfg.fs))

    if bool(include_response_arrays):
        _inject_filter_mags_for_ui(l_st, l_imp, int(cfg.fs))
        _inject_filter_mags_for_ui(r_st, r_imp, int(cfg.fs))

        l_mode = str((l_st or {}).get("analysis_mode", "native")).lower()
        r_mode = str((r_st or {}).get("analysis_mode", "native")).lower()
        l_fk = "cmp_freq_axis" if l_mode == "comparison" else "freq_axis"
        r_fk = "cmp_freq_axis" if r_mode == "comparison" else "freq_axis"
        l_gk = "cmp_filter_mags" if l_mode == "comparison" else "filter_mags"
        r_gk = "cmp_filter_mags" if r_mode == "comparison" else "filter_mags"

        l_freq = _to_axis((l_st or {}).get(l_fk), f_l)
        r_freq = _to_axis((r_st or {}).get(r_fk), f_r)
        freq_axis = l_freq if l_freq.size >= r_freq.size else r_freq
        if freq_axis.size == 0:
            freq_axis = f_l if f_l.size else f_r

        l_mag = _resample_to_axis((l_st or {}).get(l_gk), l_freq, freq_axis)
        r_mag = _resample_to_axis((r_st or {}).get(r_gk), r_freq, freq_axis)
        l_phase = _phase_from_ir(np.asarray(l_imp, dtype=float), int(cfg.fs), freq_axis)
        r_phase = _phase_from_ir(np.asarray(r_imp, dtype=float), int(cfg.fs), freq_axis)
    else:
        freq_axis = np.asarray([], dtype=float)
        l_mag = np.asarray([], dtype=float)
        r_mag = np.asarray([], dtype=float)
        l_phase = np.asarray([], dtype=float)
        r_phase = np.asarray([], dtype=float)

    metrics = {
        "alignment_samples": int(d_s),
        "alignment_method": str(align_method),
        "delay_samples_delta": (int(d_delay) if d_delay is not None else None),
        "peak_delta_samples": int(d_peak),
        "is_wav_source": bool(is_wav),
        "wav_like_fft_grid": bool(wav_like_fft_grid),
        "comparison_mode": bool(comparison_mode),
        "ir_export_window_tag": _irwin_tag(getattr(cfg, "ir_export_window_mode", "auto")),
        "l_max_boost_db_effective": _as_float((l_st or {}).get("max_boost_db_effective", (l_st or {}).get("max_boost_db", 0.0)), 0.0),
        "r_max_boost_db_effective": _as_float((r_st or {}).get("max_boost_db_effective", (r_st or {}).get("max_boost_db", 0.0)), 0.0),
        "l_max_cut_db": _as_float((l_st or {}).get("max_cut_db", 0.0), 0.0),
        "r_max_cut_db": _as_float((r_st or {}).get("max_cut_db", 0.0), 0.0),
    }

    result = FilterResult(
        fs=int(cfg.fs),
        taps=int(cfg.num_taps),
        l_ir=np.asarray(l_imp, dtype=float),
        r_ir=np.asarray(r_imp, dtype=float),
        l_mag=np.asarray(l_mag, dtype=float),
        r_mag=np.asarray(r_mag, dtype=float),
        l_phase=np.asarray(l_phase, dtype=float),
        r_phase=np.asarray(r_phase, dtype=float),
        freq_axis=np.asarray(freq_axis, dtype=float),
        l_st=dict(l_st or {}),
        r_st=dict(r_st or {}),
        metrics=metrics,
        warnings=warnings,
        measurements={
            "f_l": np.asarray(f_l, dtype=float),
            "m_l": np.asarray(m_l, dtype=float),
            "p_l": np.asarray(p_l, dtype=float),
            "f_r": np.asarray(f_r, dtype=float),
            "m_r": np.asarray(m_r, dtype=float),
            "p_r": np.asarray(p_r, dtype=float),
        },
        cfg=cfg,
    )
    if debug:
        result.metrics["summary"] = summarize_run(result)
    return result


def summarize_run(result: FilterResult) -> dict:
    fs = int(result.fs) if result.fs else 0
    taps = int(result.taps) if result.taps else 0
    latency_ms = ((float(taps) / 2.0) / float(fs) * 1000.0) if fs > 0 and taps > 0 else 0.0
    resolution_hz = (float(fs) / float(taps)) if fs > 0 and taps > 0 else 0.0
    l_peak_idx = int(np.argmax(np.abs(np.asarray(result.l_ir)))) if np.asarray(result.l_ir).size else 0
    r_peak_idx = int(np.argmax(np.abs(np.asarray(result.r_ir)))) if np.asarray(result.r_ir).size else 0

    shift_samples = result.metrics.get("alignment_samples", l_peak_idx - r_peak_idx)
    try:
        shift_samples = int(shift_samples)
    except Exception:
        shift_samples = int(l_peak_idx - r_peak_idx)
    shift_ms = (float(shift_samples) / float(fs) * 1000.0) if fs > 0 else 0.0

    l_st = result.l_st or {}
    r_st = result.r_st or {}
    l_gd_enabled = bool(l_st.get("gd_limiter_enabled", l_st.get("gd_grad_limiter_enabled", False)))
    r_gd_enabled = bool(r_st.get("gd_limiter_enabled", r_st.get("gd_grad_limiter_enabled", False)))
    l_gd_limit = l_st.get("gd_limiter_limit_ms_per_oct", l_st.get("gd_grad_limit_ms_per_oct", None))
    r_gd_limit = r_st.get("gd_limiter_limit_ms_per_oct", r_st.get("gd_grad_limit_ms_per_oct", None))
    l_gd_reason = str(l_st.get("gd_limiter_reason", l_st.get("gd_grad_limiter_reason", "unknown")) or "unknown")
    r_gd_reason = str(r_st.get("gd_limiter_reason", r_st.get("gd_grad_limiter_reason", "unknown")) or "unknown")
    l_gd_grad_max = _as_float(
        l_st.get(
            "gd_limiter_max_grad_ms_per_oct",
            l_st.get(
                "gd_grad_limiter_max_grad_ms_per_oct",
                l_st.get(
                    "gd_limiter_max_grad_after_ms_per_oct",
                    l_st.get("gd_grad_limiter_max_grad_after_ms_per_oct", 0.0),
                ),
            ),
        ),
        0.0,
    )
    r_gd_grad_max = _as_float(
        r_st.get(
            "gd_limiter_max_grad_ms_per_oct",
            r_st.get(
                "gd_grad_limiter_max_grad_ms_per_oct",
                r_st.get(
                    "gd_limiter_max_grad_after_ms_per_oct",
                    r_st.get("gd_grad_limiter_max_grad_after_ms_per_oct", 0.0),
                ),
            ),
        ),
        0.0,
    )
    out = {
        "fs": fs,
        "taps": taps,
        "latency_ms": float(latency_ms),
        "resolution_hz_per_bin": float(resolution_hz),
        "l_peak_idx": int(l_peak_idx),
        "r_peak_idx": int(r_peak_idx),
        "shift_samples": int(shift_samples),
        "shift_ms": float(shift_ms),
        "alignment_method": str(result.metrics.get("alignment_method", "peak")),
        "gd_limiter_enabled_l": bool(l_gd_enabled),
        "gd_limiter_enabled_r": bool(r_gd_enabled),
        "gd_limiter_reason_l": l_gd_reason,
        "gd_limiter_reason_r": r_gd_reason,
        "gd_limiter_limit_ms_per_oct_l": None if l_gd_limit is None else _as_float(l_gd_limit, 0.0),
        "gd_limiter_limit_ms_per_oct_r": None if r_gd_limit is None else _as_float(r_gd_limit, 0.0),
        "gd_limiter_max_grad_ms_per_oct_l": float(l_gd_grad_max),
        "gd_limiter_max_grad_ms_per_oct_r": float(r_gd_grad_max),
        "max_boost_db_effective_l": _as_float(l_st.get("max_boost_db_effective", l_st.get("max_boost_db", 0.0)), 0.0),
        "max_boost_db_effective_r": _as_float(r_st.get("max_boost_db_effective", r_st.get("max_boost_db", 0.0)), 0.0),
        "max_cut_db_l": _as_float(l_st.get("max_cut_db", 0.0), 0.0),
        "max_cut_db_r": _as_float(r_st.get("max_cut_db", 0.0), 0.0),
        "boost_peak_db_l": _as_float(l_st.get("boost_peak_db", 0.0), 0.0),
        "boost_peak_db_r": _as_float(r_st.get("boost_peak_db", 0.0), 0.0),
        "cut_peak_db_l": _as_float(l_st.get("cut_peak_db", 0.0), 0.0),
        "cut_peak_db_r": _as_float(r_st.get("cut_peak_db", 0.0), 0.0),
    }
    return out
