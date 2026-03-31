from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .auto_mode.auto_mode_profile import profiled_section
from .common.comparison_stats import _make_comparison_stats
from .common.result_postprocess import (
    _ensure_scoring_keys,
    _inject_filter_gd_stats,
    _inject_filter_mags_for_ui,
    _irwin_tag,
    _postpolish_wav_filter_ir,
    _shift_zeropad_1d,
)
from .config.models import FilterConfig
from .config.results import FilterResult
from .dsp import camillafir_dsp as dsp
from .engine_build import _as_float
from .engine_summary import summarize_run

logger = logging.getLogger("CamillaFIR")


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

    with profiled_section("run_pipeline.generate_filters"):
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

    with profiled_section("run_pipeline.ensure_scoring_keys"):
        l_st = _ensure_scoring_keys(l_st, f_l, m_l, hc_f, hc_m)
        r_st = _ensure_scoring_keys(r_st, f_r, m_r, hc_f, hc_m)

    if comparison_mode:
        with profiled_section("run_pipeline.comparison_stats"):
            try:
                l_st = _make_comparison_stats(l_st, int(cfg.fs), int(cfg.num_taps))
                r_st = _make_comparison_stats(r_st, int(cfg.fs), int(cfg.num_taps))
            except Exception as exc:
                warnings.append(f"comparison_stats_failed: {exc}")
                logger.warning(f"Comparison-mode stats failed: {exc}")

    with profiled_section("run_pipeline.align"):
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
        with profiled_section("run_pipeline.wav_postpolish"):
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

    with profiled_section("run_pipeline.inject_filter_gd_stats"):
        _inject_filter_gd_stats(l_st, l_imp, int(cfg.fs))
        _inject_filter_gd_stats(r_st, r_imp, int(cfg.fs))

    if bool(include_response_arrays):
        with profiled_section("run_pipeline.response_arrays"):
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
