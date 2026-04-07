from __future__ import annotations

import dataclasses
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
from .dsp.bass_integration import sum_complex_responses
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


def _resolve_sub_measurement_for_filter(
    measurements: dict,
    *,
    f_l: np.ndarray,
    m_l: np.ndarray,
    p_l: np.ndarray,
    f_r: np.ndarray,
    m_r: np.ndarray,
    p_r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bundle = measurements.get("bass_integration_bundle", None)
    if bundle is not None:
        try:
            sub_transfer = sum_complex_responses(
                bundle.l_sub,
                bundle.r_sub,
                label="Direct-DAC sub total",
            )
            return (
                np.asarray(sub_transfer.freqs_hz, dtype=float),
                np.asarray(sub_transfer.mag_db, dtype=float),
                np.asarray(sub_transfer.phase_deg, dtype=float),
            )
        except Exception:
            logger.debug("Could not build Bass Integration sub-only transfer", exc_info=True)

    f_sub = (f_l + f_r) / 2.0 if f_l.size == f_r.size else f_l
    m_sub = (m_l + m_r) / 2.0 if m_l.size == m_r.size else m_l
    p_sub = (p_l + p_r) / 2.0 if p_l.size == p_r.size else p_l
    return (
        np.asarray(f_sub, dtype=float),
        np.asarray(m_sub, dtype=float),
        np.asarray(p_sub, dtype=float),
    )


def _interp_complex_to_axis(
    f_src: np.ndarray,
    h_src: np.ndarray,
    f_dst: np.ndarray,
) -> np.ndarray:
    fs = np.asarray(f_src if f_src is not None else [], dtype=float).reshape(-1)
    hs = np.asarray(h_src if h_src is not None else [], dtype=np.complex128).reshape(-1)
    fd = np.asarray(f_dst if f_dst is not None else [], dtype=float).reshape(-1)
    if fd.size == 0:
        return np.asarray([], dtype=np.complex128)
    if fs.size < 2 or hs.size != fs.size:
        return np.zeros(fd.shape, dtype=np.complex128)
    f_q = np.clip(fd, float(np.min(fs)), float(np.max(fs)))
    re = np.interp(f_q, fs, np.real(hs))
    im = np.interp(f_q, fs, np.imag(hs))
    return (re + 1j * im).astype(np.complex128, copy=False)


def _ir_fft_on_axis(ir: np.ndarray, fs: int, f_axis: np.ndarray) -> np.ndarray:
    x = np.asarray(ir if ir is not None else [], dtype=float).reshape(-1)
    fa = np.asarray(f_axis if f_axis is not None else [], dtype=float).reshape(-1)
    fs_i = int(fs) if fs else 0
    if x.size < 8 or fa.size == 0 or fs_i <= 0:
        return np.zeros(fa.shape, dtype=np.complex128)
    h_fft = np.fft.rfft(x)
    f_fft = np.fft.rfftfreq(x.size, d=1.0 / float(fs_i))
    return _interp_complex_to_axis(f_fft, h_fft, fa)


def _stats_level_comp_factor(st: dict | None) -> float:
    try:
        ag_db = float((st or {}).get("auto_global_gain_db", 0.0) or 0.0)
    except Exception:
        ag_db = 0.0
    try:
        ah_db = float((st or {}).get("auto_headroom_db", 0.0) or 0.0)
    except Exception:
        ah_db = 0.0
    try:
        return float(np.power(10.0, -0.05 * (float(ag_db) + float(ah_db))))
    except Exception:
        return 1.0


def _inject_direct_dac_summed_prediction_for_plot(
    *,
    st: dict | None,
    main_transfer: Any,
    sub_transfer: Any,
    main_ir: np.ndarray,
    sub_ir: np.ndarray | None,
    sub_st: dict | None,
    fs: int,
) -> None:
    if not isinstance(st, dict) or sub_ir is None:
        return
    try:
        f_axis = np.asarray(st.get("freq_axis", None), dtype=float)
        if f_axis.size < 4:
            f_axis = np.asarray(getattr(main_transfer, "freqs_hz", []), dtype=float)
        if f_axis.size < 4:
            return

        main_meas = _interp_complex_to_axis(
            np.asarray(getattr(main_transfer, "freqs_hz", []), dtype=float),
            np.asarray(getattr(main_transfer, "complex_spec", []), dtype=np.complex128),
            f_axis,
        )
        sub_meas = _interp_complex_to_axis(
            np.asarray(getattr(sub_transfer, "freqs_hz", []), dtype=float),
            np.asarray(getattr(sub_transfer, "complex_spec", []), dtype=np.complex128),
            f_axis,
        )
        main_h = _ir_fft_on_axis(np.asarray(main_ir, dtype=float), int(fs), f_axis)
        sub_h = _ir_fft_on_axis(np.asarray(sub_ir, dtype=float), int(fs), f_axis)

        total_measured = main_meas + sub_meas
        st["direct_dac_sum_measured_mags"] = (
            20.0 * np.log10(np.maximum(np.abs(total_measured), 1e-12))
        ).astype(float).tolist()

        total_export = main_meas * main_h + sub_meas * sub_h
        st["direct_dac_sum_predicted_mags"] = (
            20.0 * np.log10(np.maximum(np.abs(total_export), 1e-12))
        ).astype(float).tolist()

        main_comp = float(_stats_level_comp_factor(st))
        sub_comp = float(_stats_level_comp_factor(sub_st))
        total_comp = main_meas * (main_h * main_comp) + sub_meas * (sub_h * sub_comp)
        st["direct_dac_sum_predicted_mags_comp"] = (
            20.0 * np.log10(np.maximum(np.abs(total_comp), 1e-12))
        ).astype(float).tolist()
    except Exception:
        logger.debug("Direct-DAC summed plot prediction injection failed", exc_info=True)


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
    sub_st: dict | None = None

    with profiled_section("run_pipeline.generate_filters"):
        if bool(getattr(cfg, "stereo_link", False)):
            l_imp, l_st, r_imp, r_st = dsp.generate_filter_pair(f_l, m_l, p_l, f_r, m_r, p_r, cfg)
        else:
            l_imp, l_st = dsp.generate_filter(f_l, m_l, p_l, cfg)
            r_imp, r_st = dsp.generate_filter(f_r, m_r, p_r, cfg)

    sub_ir: np.ndarray | None = None
    sub_f: np.ndarray | None = None
    sub_m: np.ndarray | None = None
    sub_p: np.ndarray | None = None
    if bool(getattr(cfg, "sub_integration_enable", False)) and \
       bool(getattr(cfg, "sub_generate_ir", False)):
        try:
            f_sub, m_sub, p_sub = _resolve_sub_measurement_for_filter(
                measurements,
                f_l=f_l,
                m_l=m_l,
                p_l=p_l,
                f_r=f_r,
                m_r=m_r,
                p_r=p_r,
            )

            sub_xo_hz = float(getattr(cfg, "sub_crossover_hz", 80.0))
            sub_xo_order = int(getattr(cfg, "sub_crossover_order", 4))
            sub_hpf_freq = float(getattr(cfg, "sub_hpf_freq", 20.0))
            sub_hpf_order = int(getattr(cfg, "sub_hpf_order", 2))

            sub_cfg = dataclasses.replace(
                cfg,
                lpf_settings={"enabled": True, "freq": sub_xo_hz, "order": sub_xo_order},
                hpf_settings={"enabled": True, "freq": sub_hpf_freq, "order": sub_hpf_order},
                crossovers=[{"freq": sub_xo_hz, "order": sub_xo_order,
                             "slope": sub_xo_order * 6, "idx": 0}],
                mag_c_min=max(5.0, float(sub_hpf_freq)),
                mag_c_max=sub_xo_hz,
                lvl_min=20.0,
                lvl_max=200.0,
                stereo_link=False,
                sub_integration_enable=False,
                sub_generate_ir=False,
            )
            sub_imp, sub_st = dsp.generate_filter(f_sub, m_sub, p_sub, sub_cfg)
            sub_ir = np.asarray(sub_imp, dtype=float)
            sub_f = np.asarray(f_sub, dtype=float)
            sub_m = np.asarray(m_sub, dtype=float)
            sub_p = np.asarray(p_sub, dtype=float)
            logger.info(
                f"Sub pass: xo={sub_xo_hz:.0f} Hz (order {sub_xo_order}), "
                f"HPF={sub_hpf_freq:.0f} Hz (order {sub_hpf_order})"
            )
        except Exception as exc:
            warnings.append(f"sub_pass_failed: {exc}")
            logger.warning(f"Sub DSP pass failed: {exc}")
            sub_ir = None

    is_wav = bool(measurements.get("is_wav_source", getattr(cfg, "is_wav_source", False)))
    rt_rel = 1.0 if is_wav else 0.25
    rt_src = "WAV" if is_wav else "TXT/REW"
    if isinstance(l_st, dict):
        l_st["rt60_reliability"] = float(rt_rel)
        l_st["rt60_source"] = rt_src
    if isinstance(r_st, dict):
        r_st["rt60_reliability"] = float(rt_rel)
        r_st["rt60_source"] = rt_src

    _ir_align_lr: dict = {}
    _ir_align_sub: dict = {}
    _raw_ir_l = measurements.get("raw_ir_l")
    _raw_ir_r = measurements.get("raw_ir_r")
    _raw_ir_fs_l = int(measurements.get("raw_ir_fs_l") or 0)
    _raw_ir_fs_r = int(measurements.get("raw_ir_fs_r") or 0)
    _raw_ir_sub = measurements.get("raw_ir_sub")
    _raw_ir_fs_sub = int(measurements.get("raw_ir_fs_sub") or 0)
    if _raw_ir_l is not None and _raw_ir_r is not None and _raw_ir_fs_l > 0 and _raw_ir_fs_r > 0:
        try:
            from .dsp.ir_alignment_check import run_ir_alignment_check  # noqa: PLC0415
            _xo_lr = float(data.get("avr_crossover_hz", 80.0) or 80.0)
            if not (10.0 <= _xo_lr <= 500.0):
                _xo_lr = 80.0
            _ir_align_lr = run_ir_alignment_check(
                np.asarray(_raw_ir_l, dtype=float), _raw_ir_fs_l,
                np.asarray(_raw_ir_r, dtype=float), _raw_ir_fs_r,
                xo_hz=_xo_lr,
            )
        except Exception:
            logger.debug("IR alignment check L/R failed", exc_info=True)
    if _raw_ir_l is not None and _raw_ir_sub is not None and _raw_ir_fs_l > 0 and _raw_ir_fs_sub > 0:
        try:
            from .dsp.ir_alignment_check import run_ir_alignment_check  # noqa: PLC0415
            _xo_sub = float(
                data.get("avr_crossover_hz") or data.get("sub_crossover_hz", 80.0) or 80.0
            )
            if not (10.0 <= _xo_sub <= 500.0):
                _xo_sub = 80.0
            _ir_align_sub = run_ir_alignment_check(
                np.asarray(_raw_ir_l, dtype=float), _raw_ir_fs_l,
                np.asarray(_raw_ir_sub, dtype=float), _raw_ir_fs_sub,
                xo_hz=_xo_sub,
            )
        except Exception:
            logger.debug("IR alignment check main/sub failed", exc_info=True)
    if isinstance(l_st, dict):
        if _ir_align_lr:
            l_st["ir_align"] = dict(_ir_align_lr)
        if _ir_align_sub:
            l_st["ir_align_sub"] = dict(_ir_align_sub)
    if isinstance(r_st, dict):
        if _ir_align_lr:
            r_st["ir_align"] = dict(_ir_align_lr)
        if _ir_align_sub:
            r_st["ir_align_sub"] = dict(_ir_align_sub)

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

        if sub_ir is not None and np.asarray(sub_ir, dtype=float).size > 0:
            sub_imp = np.asarray(sub_ir, dtype=float)
            d_sub: int | None = None
            try:
                # Sub delay_samples is computed from the 1–10 kHz phase reference band,
                # which is outside the sub's 20–80 Hz working range → unreliable (returns ~0).
                # Always use peak-based alignment for the subwoofer IR.
                main_peak_ref = int(
                    round(
                        0.5 * (
                            int(np.argmax(np.abs(np.asarray(l_imp, dtype=float))))
                            + int(np.argmax(np.abs(np.asarray(r_imp, dtype=float))))
                        )
                    )
                )
                sub_peak = int(np.argmax(np.abs(sub_imp)))
                d_sub = int(main_peak_ref - sub_peak)
            except Exception:
                d_sub = None
            if d_sub != 0:
                sub_ir = _shift_zeropad_1d(sub_imp, int(d_sub))
                logger.info(f"Sub alignment applied: {int(d_sub)} samples")

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

            if (
                sub_ir is not None
                and bool(getattr(cfg, "sub_integration_enable", False))
                and bool(getattr(cfg, "sub_generate_ir", False))
                and str(getattr(cfg, "bass_integration_mode", "") or "").strip().lower() == "direct_dac"
            ):
                bundle = measurements.get("bass_integration_bundle", None)
                if bundle is not None:
                    try:
                        _inject_direct_dac_summed_prediction_for_plot(
                            st=l_st,
                            main_transfer=bundle.l_main,
                            sub_transfer=bundle.l_sub,
                            main_ir=np.asarray(l_imp, dtype=float),
                            sub_ir=np.asarray(sub_ir, dtype=float),
                            sub_st=sub_st,
                            fs=int(cfg.fs),
                        )
                        _inject_direct_dac_summed_prediction_for_plot(
                            st=r_st,
                            main_transfer=bundle.r_main,
                            sub_transfer=bundle.r_sub,
                            main_ir=np.asarray(r_imp, dtype=float),
                            sub_ir=np.asarray(sub_ir, dtype=float),
                            sub_st=sub_st,
                            fs=int(cfg.fs),
                        )
                    except Exception:
                        logger.debug("Direct-DAC summed prediction build failed", exc_info=True)

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
            **({
                "f_sub": np.asarray(sub_f, dtype=float),
                "m_sub": np.asarray(sub_m, dtype=float),
                "p_sub": np.asarray(sub_p, dtype=float),
            } if sub_f is not None else {}),
        },
        cfg=cfg,
        sub_ir=sub_ir,
        sub_st=dict(sub_st) if isinstance(sub_st, dict) else None,
    )
    if debug:
        result.metrics["summary"] = summarize_run(result)
    return result
