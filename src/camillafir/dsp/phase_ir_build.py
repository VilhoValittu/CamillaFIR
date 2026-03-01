from __future__ import annotations

import numpy as np
import scipy.fft

from .phase import calculate_minimum_phase, combine_mixed_phase
from .phase_ir_align import _compute_alignment_target, _shift_ir
from .phase_ir_guards import _pre_energy_guard, _tdc_postprocess
from .phase_ir_ir import _build_complex_spectrum, _ifft_to_ir, _normalize_ir
from .phase_ir_metrics import _summarize_ir_metrics
from .phase_ir_phase import _PhaseComponents, _apply_phase_model, _compute_excess_phase, _unwrap_phases
from .phase_ir_utils import _ms_value, _resolve_ir_anchor_mode, _resolve_ir_window_mode
from .phase_ir_window import _apply_fdw_if_enabled, _apply_ir_window


def _apply_midband_realized_level_match(
    impulse: np.ndarray,
    total_mag: np.ndarray,
    cfg,
    st: dict,
) -> np.ndarray:
    """Match realized IR level to total_mag in 200-2000 Hz with a global scale."""
    ir = np.asarray(impulse, dtype=float).copy()
    mag_target = np.asarray(total_mag, dtype=float).reshape(-1)
    try:
        enabled = bool(getattr(cfg, "ir_realized_level_match_enable", True))
    except Exception:
        enabled = True
    try:
        f_lo = float(getattr(cfg, "ir_realized_level_match_mid_lo_hz", 200.0) or 200.0)
    except Exception:
        f_lo = 200.0
    try:
        f_hi = float(getattr(cfg, "ir_realized_level_match_mid_hi_hz", 2000.0) or 2000.0)
    except Exception:
        f_hi = 2000.0
    try:
        min_abs_db = float(getattr(cfg, "ir_realized_level_match_min_abs_db", 0.25) or 0.25)
    except Exception:
        min_abs_db = 0.25
    try:
        max_abs_db = float(getattr(cfg, "ir_realized_level_match_max_abs_db", 3.0) or 3.0)
    except Exception:
        max_abs_db = 3.0

    if not np.isfinite(f_lo):
        f_lo = 200.0
    if not np.isfinite(f_hi) or f_hi <= f_lo:
        f_hi = max(f_lo + 1.0, 2000.0)
    if not np.isfinite(min_abs_db) or min_abs_db < 0.0:
        min_abs_db = 0.25
    if not np.isfinite(max_abs_db) or max_abs_db <= 0.0:
        max_abs_db = 3.0
    min_abs_db = float(min_abs_db)
    max_abs_db = float(max_abs_db)

    def _mark(reason: str, applied: bool, delta_raw, delta_apply, delta_after, scale):
        try:
            if isinstance(st, dict):
                st["ir_realized_level_match_enabled"] = bool(enabled)
                st["ir_realized_level_match_applied"] = bool(applied)
                st["ir_realized_level_match_reason"] = str(reason)
                st["ir_realized_level_match_mid_lo_hz"] = float(f_lo)
                st["ir_realized_level_match_mid_hi_hz"] = float(f_hi)
                st["ir_realized_level_match_delta_db_raw"] = (
                    float(delta_raw) if delta_raw is not None and np.isfinite(float(delta_raw)) else None
                )
                st["ir_realized_level_match_delta_db_applied"] = (
                    float(delta_apply) if delta_apply is not None and np.isfinite(float(delta_apply)) else 0.0
                )
                st["ir_realized_level_match_delta_db_after"] = (
                    float(delta_after) if delta_after is not None and np.isfinite(float(delta_after)) else None
                )
                st["ir_realized_level_match_scale"] = (
                    float(scale) if scale is not None and np.isfinite(float(scale)) else 1.0
                )
        except Exception:
            pass

    if not enabled:
        _mark("disabled", False, None, 0.0, None, 1.0)
        return ir

    fs = float(getattr(cfg, "fs", 0.0) or 0.0)
    if ir.size < 8 or mag_target.size < 8 or fs <= 0.0:
        _mark("invalid_input", False, None, 0.0, None, 1.0)
        return ir

    n = int(ir.size)
    h2 = scipy.fft.rfft(ir, n=n)
    mag2 = np.abs(h2)
    n2 = int(min(mag2.size, mag_target.size))
    if n2 < 8:
        _mark("insufficient_bins", False, None, 0.0, None, 1.0)
        return ir

    freq = scipy.fft.rfftfreq(n, d=1.0 / fs)[:n2]
    mag2_db = 20.0 * np.log10(np.maximum(np.asarray(mag2[:n2], dtype=float), 1e-12))
    target_db = 20.0 * np.log10(np.maximum(np.asarray(mag_target[:n2], dtype=float), 1e-12))

    valid = np.isfinite(freq) & np.isfinite(mag2_db) & np.isfinite(target_db) & (freq > 0.0)
    mid = valid & (freq >= float(f_lo)) & (freq <= float(f_hi))
    if int(np.count_nonzero(mid)) < 8:
        _mark("mid_band_sparse", False, None, 0.0, None, 1.0)
        return ir

    delta_db_raw = float(np.median(mag2_db[mid] - target_db[mid]))
    if not np.isfinite(delta_db_raw):
        _mark("delta_invalid", False, None, 0.0, None, 1.0)
        return ir

    delta_db_apply = float(np.clip(delta_db_raw, -max_abs_db, max_abs_db))
    if abs(delta_db_apply) <= float(min_abs_db):
        _mark("below_threshold", False, delta_db_raw, 0.0, delta_db_raw, 1.0)
        return ir

    scale = float(10.0 ** (-delta_db_apply / 20.0))
    ir *= scale

    try:
        h3 = scipy.fft.rfft(ir, n=n)
        mag3_db = 20.0 * np.log10(np.maximum(np.asarray(np.abs(h3[:n2]), dtype=float), 1e-12))
        delta_after = float(np.median(mag3_db[mid] - target_db[mid]))
    except Exception:
        delta_after = None

    _mark("applied", True, delta_db_raw, delta_db_apply, delta_after, scale)
    return ir


def build_phase_and_ir(
    *,
    cfg,
    freq_axis: np.ndarray,
    n_fft: int,
    gain_db: np.ndarray,
    p_rad_interp: np.ndarray,
    conf_mask,
    st: dict,
    mask_c: np.ndarray,
    use_bassfirst: bool,
    afdw_on: bool,
    logger,
    theo_xo: np.ndarray,
    auto_global_gain_db: float,
    auto_headroom_db: float,
    final_gain_total: np.ndarray,
    limit_gd_gradient_ms_per_oct_fn,
) -> dict:
    """
    Contract:
      - This stage builds phase + IR only.
      - It must not modify gain_db/autogain values.
      - Allowed side effects are st window/align/guard/metric indicators.

    Returns dict with:
      impulse: np.ndarray
      total_mag: np.ndarray
      min_phase: np.ndarray
      final_phase: np.ndarray
      mixed_split_hz: float
      mixed_transition_hz: float
    Side effects: st updates exactly like before (mixed_blend keys etc.)
    """
    total_mag = 10**(final_gain_total / 20.0)
    min_p = calculate_minimum_phase(total_mag, max_phase_deg=None)

    is_mixed = ("Mixed" in cfg.filter_type_str)
    mixed_split_hz = float(
        np.clip(float(getattr(cfg, "mixed_split_freq", 300.0) or 300.0), 20.0, float(cfg.fs) * 0.49)
    )
    mixed_transition_hz = float(getattr(cfg, "trans_width", mixed_split_hz) or mixed_split_hz)
    if not np.isfinite(mixed_transition_hz) or mixed_transition_hz < 0.0:
        mixed_transition_hz = mixed_split_hz

    raw_u, ref_u = _unwrap_phases(p_rad_interp, theo_xo)
    excess_u = _compute_excess_phase(raw_u, ref_u)

    phase_components = _PhaseComponents(
        raw_u=raw_u,
        ref_u=ref_u,
        excess_u=excess_u,
        min_phase=min_p,
        theo_xo=theo_xo,
        conf_mask=conf_mask,
        total_mag=total_mag,
        n_fft=n_fft,
        is_mixed=is_mixed,
        mixed_split_hz=mixed_split_hz,
        mixed_transition_hz=mixed_transition_hz,
        use_bassfirst=use_bassfirst,
        afdw_on=afdw_on,
        logger=logger,
        limit_gd_gradient_ms_per_oct_fn=limit_gd_gradient_ms_per_oct_fn,
    )
    final_phase = _apply_phase_model(freq_axis, cfg, st, phase_components)
    low_phase = phase_components.low_phase

    if is_mixed and low_phase is not None:
        try:
            if isinstance(st, dict):
                st["mixed_blend_split_hz"] = float(mixed_split_hz)
                st["mixed_blend_transition_hz"] = float(mixed_transition_hz)
        except Exception:
            pass
        h_lin = _build_complex_spectrum(total_mag, low_phase)
        h_min = _build_complex_spectrum(total_mag, min_p)
        ir_lin = _ifft_to_ir(h_lin, n=n_fft)
        ir_min = _ifft_to_ir(h_min, n=n_fft)
        raw_imp = combine_mixed_phase(
            ir_lin,
            ir_min,
            fs=float(cfg.fs),
            split_freq=mixed_split_hz,
            transition_hz=mixed_transition_hz,
        )
        final_phase = np.angle(scipy.fft.rfft(raw_imp))
    else:
        h_complex = _build_complex_spectrum(total_mag, final_phase)
        raw_imp = _ifft_to_ir(h_complex, n=n_fft)

    raw_imp = _normalize_ir(raw_imp, cfg)

    anchor_mode = _resolve_ir_anchor_mode(cfg)
    _, win_mode, is_min_filter = _resolve_ir_window_mode(cfg, logger=logger)
    left_ms = _ms_value(cfg, "ir_window_ms_left", "ir_window_left", 0.0)
    requested_win_mode, win_mode, is_min_filter = _resolve_ir_window_mode(cfg, logger=logger)
    left_ms = _ms_value(cfg, "ir_window_ms_left", "ir_window_left", 0.0)

    want_left_anchor = (str(requested_win_mode or "").strip().lower() == "rew_asym")
#filtterin siirto
    if "Asym" in cfg.filter_type_str:
        shift = min(int(left_ms * cfg.fs / 1000.0), int(n_fft * 0.4))
        impulse = np.roll(raw_imp, shift)

    elif "Min" in cfg.filter_type_str:
        impulse = raw_imp

    elif "Linear" in cfg.filter_type_str:
        if want_left_anchor:
            shift = min(int(left_ms * cfg.fs / 1000.0), int(n_fft * 0.4))
            impulse = np.roll(raw_imp, shift)
        else:
            impulse = np.roll(raw_imp, n_fft // 2)

    else:
        impulse = np.roll(raw_imp, n_fft // 2)

    try:
        if isinstance(st, dict):
            st["ir_anchor_mode"] = str(anchor_mode)
    except Exception:
        pass
    logger.info(f"IR init: type={cfg.filter_type_str}, peak0={int(np.argmax(np.abs(impulse)))}, n={n_fft}")
    
    pre_target = _compute_alignment_target(impulse, cfg, st, stage="pre_window")
    impulse, pre_info = _shift_ir(
        impulse,
        pre_target,
        True,
        st=st,
        key_prefix="pre_align",
        anchor_mode=anchor_mode,
    )
    if is_min_filter and win_mode == "off" and bool(pre_info.get("applied", False)):
        try:
            if isinstance(st, dict):
                st["min_off_peak_shift_samples"] = int(pre_info.get("shift_samples", 0))
                st["min_off_peak_before_samples"] = int(pre_info.get("peak_before_samples", 0))
        except Exception:
            pass
        logger.info(
            "Minimum OFF causal shift applied: "
            f"peak {int(pre_info.get('peak_before_samples', 0))} -> 0 "
            f"(shift={int(pre_info.get('shift_samples', 0))})"
        )

    impulse = _apply_ir_window(impulse, cfg, st, logger=logger)
    impulse = _apply_fdw_if_enabled(impulse, cfg, st)

    post_target = _compute_alignment_target(impulse, cfg, st, stage="post_window")
    impulse, post_info = _shift_ir(
        impulse,
        post_target,
        True,
        st=st,
        key_prefix="post_align",
        anchor_mode=anchor_mode,
    )
    try:
        if isinstance(st, dict):
            split_idx = int(post_info.get("anchor_after_samples", post_info.get("peak_after_samples", 0)))
            st["ir_energy_split_samples"] = int(np.clip(split_idx, 0, max(int(impulse.size) - 1, 0)))
            st["ir_energy_split_source"] = "post_align_anchor"
            if anchor_mode == "min_causal":
                st["min_causal_shift"] = int(post_info.get("shift_samples", 0))
                st["min_causal_peak_after_samples"] = int(post_info.get("peak_after_samples", 0))
                st["min_causal_anchor_after_samples"] = int(post_info.get("anchor_after_samples", 0))
    except Exception:
        pass
    if is_mixed and impulse.size > 0 and anchor_mode != "min_causal":
        mixed_forced_peak_ms = 90.0
        try:
            if isinstance(st, dict):
                st["mixed_forced_peak_ms"] = float(mixed_forced_peak_ms)
                st["mixed_forced_peak_samples"] = int(post_info.get("desired_samples", 0))
                st["mixed_forced_shift_samples"] = int(post_info.get("shift_samples", 0))
        except Exception:
            pass
        logger.info(
            f"Mixed forced peak shift applied: peak {int(post_info.get('peak_before_samples', 0))} "
            f"-> {int(post_info.get('desired_samples', 0))} "
            f"({mixed_forced_peak_ms:.1f} ms, shift={int(post_info.get('shift_samples', 0))})"
        )

    impulse, _ = _pre_energy_guard(impulse, cfg, st)
    impulse = _tdc_postprocess(impulse, cfg, st)
    impulse = _apply_midband_realized_level_match(impulse, total_mag, cfg, st)
    _summarize_ir_metrics(impulse, cfg, st)

    return {
        "impulse": np.asarray(impulse, dtype=float),
        "total_mag": np.asarray(total_mag, dtype=float),
        "min_phase": np.asarray(min_p, dtype=float),
        "final_phase": np.asarray(final_phase, dtype=float),
        "mixed_split_hz": float(mixed_split_hz),
        "mixed_transition_hz": float(mixed_transition_hz),
    }
