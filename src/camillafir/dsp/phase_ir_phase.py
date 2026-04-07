from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.ndimage

from .phase import combine_mixed_phase
from .phase_ir_ir import _build_complex_spectrum, _ifft_to_ir
from .phase_ir_phase_gradient import (
    gd_grad_limiter as _gd_grad_limiter_impl,
    gd_grad_metrics as _gd_grad_metrics_impl,
    max_abs_gd_gradient_ms_per_oct as _max_abs_gd_gradient_ms_per_oct_impl,
)
from .phase_ir_phase_models import (
    apply_mixed_excess_mask as _apply_mixed_excess_mask_impl,
    enforce_linear_tail_decay as _enforce_linear_tail_decay_impl,
    linear_excess_weight as _linear_excess_weight_impl,
    linear_to_minphase_blend_mask as _linear_to_minphase_blend_mask_impl,
    merge_minphase_and_excess as _merge_minphase_and_excess_impl,
    smooth_linear_boundary as _smooth_linear_boundary_impl,
)
from .phase_ir_utils import _max_abs_group_delay_ms, _mixed_excess_weight, _pre_ringing_db, _smoothstep01


@dataclass
class _PhaseComponents:
    raw_u: np.ndarray
    ref_u: np.ndarray
    excess_u: np.ndarray
    min_phase: np.ndarray
    theo_xo: np.ndarray
    conf_mask: np.ndarray | None
    total_mag: np.ndarray
    n_fft: int
    is_mixed: bool
    mixed_split_hz: float
    mixed_transition_hz: float
    use_bassfirst: bool
    afdw_on: bool
    logger: Any
    limit_gd_gradient_ms_per_oct_fn: Any
    low_phase: np.ndarray | None = None
    extra_phase: np.ndarray | None = None
    phase_mask: np.ndarray | None = None


def _unwrap_phases(raw_phase, min_phase) -> tuple[np.ndarray, np.ndarray]:
    raw_u = np.unwrap(np.asarray(raw_phase, dtype=float))
    min_u = np.unwrap(np.asarray(min_phase, dtype=float))
    return raw_u, min_u


def _compute_excess_phase(raw_phase, ref_phase) -> np.ndarray:
    raw = np.asarray(raw_phase, dtype=float)
    ref = np.asarray(ref_phase, dtype=float)
    # Keep excess phase branch-stable: if raw/ref unwrap to different 2*pi branches,
    # direct subtraction can inject large artificial zig-zag into mixed-phase correction.
    return np.angle(np.exp(1j * (raw - ref)))


def _apply_mixed_excess_mask(freq_axis, excess, cfg, st) -> np.ndarray:
    return _apply_mixed_excess_mask_impl(freq_axis, excess, cfg, st)


def _linear_excess_weight(freq_axis: np.ndarray, phase_lim_hz: float) -> np.ndarray:
    return _linear_excess_weight_impl(freq_axis, phase_lim_hz)


def _smooth_linear_boundary(freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
    return _smooth_linear_boundary_impl(freq_axis, extra_phase, phase_lim_hz, cfg, st)


def _enforce_linear_tail_decay(freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
    return _enforce_linear_tail_decay_impl(freq_axis, extra_phase, phase_lim_hz, cfg, st)


def _linear_to_minphase_blend_mask(freq_axis: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
    return _linear_to_minphase_blend_mask_impl(freq_axis, phase_lim_hz, cfg, st)


def _merge_minphase_and_excess(min_u, excess_masked) -> np.ndarray:
    return _merge_minphase_and_excess_impl(min_u, excess_masked)


def _max_abs_gd_gradient_ms_per_oct(
    freq_axis: np.ndarray,
    phase_rad: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[float, float | None]:
    return _max_abs_gd_gradient_ms_per_oct_impl(freq_axis, phase_rad, mask=mask)


def _gd_grad_metrics(
    freq_axis: np.ndarray,
    phase_rad: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    return _gd_grad_metrics_impl(freq_axis, phase_rad, mask=mask)


def _gd_grad_limiter(
    ir,
    cfg,
    st,
    *,
    freq_axis=None,
    phase_mask=None,
    use_bassfirst=False,
    afdw_on=False,
    limiter_fn=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _gd_grad_limiter_impl(
        ir,
        cfg,
        st,
        freq_axis=freq_axis,
        phase_mask=phase_mask,
        limiter_fn=limiter_fn,
    )


def _has_active_theoretical_phase_model(cfg) -> bool:
    try:
        for xo in list(getattr(cfg, "crossovers", None) or []):
            try:
                fc = float(xo.get("freq", 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
            if np.isfinite(fc) and fc > 0.0:
                return True
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        hs = getattr(cfg, "hpf_settings", None)
        if isinstance(hs, dict) and bool(hs.get("enabled", False)):
            fc = float(hs.get("freq", 0.0) or 0.0)
            order = int(hs.get("order", 0) or 0)
            if np.isfinite(fc) and fc > 0.0 and order > 0:
                return True
    except (AttributeError, TypeError, ValueError):
        pass

    return False


def _apply_phase_model(freq_axis, cfg, st, phase_components: _PhaseComponents) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    is_mixed = bool(phase_components.is_mixed)
    min_p = np.asarray(phase_components.min_phase, dtype=float)
    theo_xo = np.asarray(phase_components.theo_xo, dtype=float)
    excess_phase = np.asarray(phase_components.excess_u, dtype=float)
    logger = phase_components.logger

    if bool(getattr(cfg, "phase_safe_2058", False)):
        if "Min" in cfg.filter_type_str:
            final_phase = min_p
        elif is_mixed:
            low_phase = -theo_xo
            phase_components.low_phase = low_phase
            final_phase = low_phase
        else:
            final_phase = -theo_xo
        phase_components.extra_phase = None
        phase_components.phase_mask = None
        return final_phase

    try:
        conf_arr = (
            np.asarray(phase_components.conf_mask, dtype=float)
            if phase_components.conf_mask is not None
            else np.ones_like(f, dtype=float)
        )
        conf_s = scipy.ndimage.gaussian_filter1d(conf_arr, sigma=2)
        conf_s = np.clip(conf_s, 0.0, 1.0)
    except (TypeError, ValueError):
        conf_s = np.ones_like(f, dtype=float)

    phase_lim_hz = float(getattr(cfg, "phase_limit", 1000.0))
    phase_mask = (f > 0) & (f <= phase_lim_hz)
    phase_components.phase_mask = phase_mask
    try:
        if isinstance(st, dict):
            st["phase_limit_hz"] = float(phase_lim_hz)
    except (TypeError, ValueError):
        pass

    try:
        f1 = float(phase_lim_hz)
        f0_fade = f1 / 2.0
        if f0_fade < (f1 - 1.0):
            x = (f - f0_fade) / (f1 - f0_fade + 1e-12)
            x = np.clip(x, 0.0, 1.0)
            w_hi = 0.5 * (1.0 + np.cos(np.pi * x))
            w_hi = np.where(f <= f0_fade, 1.0, w_hi)
            w_hi = np.where(f >= f1, 0.0, w_hi)
        else:
            w_hi = np.ones_like(f, dtype=float)
    except (TypeError, ValueError, FloatingPointError):
        w_hi = np.ones_like(f, dtype=float)

    if is_mixed:
        extra_phase = -_apply_mixed_excess_mask(f, excess_phase, cfg, st)
    else:
        phase_weight = _linear_excess_weight(f, phase_lim_hz)
        phase_weight = phase_weight * phase_mask.astype(float)
        extra_phase = -excess_phase * phase_weight

    try:
        extra_phase *= w_hi
    except (TypeError, ValueError):
        pass

    try:
        conf_floor = 0.10
        conf_power = 1.25
        conf_gain = np.clip(conf_s, 0.0, 1.0) ** conf_power
        conf_gain = conf_floor + (1.0 - conf_floor) * conf_gain
        extra_phase *= conf_gain
    except (TypeError, ValueError):
        pass

    try:
        extra_phase_before = np.asarray(extra_phase, dtype=float).copy()
        if is_mixed:
            clamp_max_deg = float(getattr(cfg, "mixed_phase_budget_lf_deg", 45.0) or 45.0)
            clamp_min_deg = float(getattr(cfg, "mixed_phase_budget_hf_deg", 22.5) or 22.5)
        else:
            clamp_max_deg = 45.0
            clamp_min_deg = 15.0
        if clamp_max_deg < clamp_min_deg:
            clamp_max_deg, clamp_min_deg = clamp_min_deg, clamp_max_deg

        conf_part = np.clip(conf_s, 0.0, 1.0) ** 0.85
        if phase_lim_hz > 0.0:
            freq_rel = np.clip((phase_lim_hz - f) / max(phase_lim_hz, 1e-9), 0.0, 1.0)
        else:
            freq_rel = np.ones_like(f, dtype=float)
        freq_part = np.sqrt(freq_rel)

        blend = 0.70 * conf_part + 0.30 * freq_part
        limit_deg_arr = clamp_min_deg + (clamp_max_deg - clamp_min_deg) * blend
        limit_deg_arr = np.clip(limit_deg_arr, clamp_min_deg, clamp_max_deg)
        limit_rad_arr = np.deg2rad(limit_deg_arr)

        before_rad = float(np.max(np.abs(extra_phase)))
        extra_phase = np.clip(extra_phase, -limit_rad_arr, limit_rad_arr)
        after_rad = float(np.max(np.abs(extra_phase)))
        try:
            _cut_rad = np.maximum(np.abs(extra_phase_before) - np.abs(extra_phase), 0.0)
            _cut_frac = _cut_rad / np.maximum(np.abs(extra_phase_before), 1e-9)
            _cut_frac = np.clip(_cut_frac, 0.0, 1.0)
            if isinstance(st, dict):
                st["phase_corr_clamp_cut_frac"] = _cut_frac.tolist()
        except (TypeError, ValueError, FloatingPointError):
            pass

        before_deg = float(np.rad2deg(before_rad))
        after_deg = float(np.rad2deg(after_rad))
        clipped = bool(np.any(np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)))
        try:
            clipped_bins = int(np.sum((np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)) & phase_mask))
        except (TypeError, ValueError, FloatingPointError):
            clipped_bins = int(clipped)
        if clipped:
            msg = (
                "Phase Correction Clamp (adaptive): "
                f"max={before_deg:.1f} deg -> {after_deg:.1f} deg "
                f"(limit {clamp_min_deg:.1f}..{clamp_max_deg:.1f} deg, clipped_bins={clipped_bins})"
            )
        else:
            msg = (
                "Phase Correction Clamp (adaptive): "
                f"max={before_deg:.1f} deg (limit {clamp_min_deg:.1f}..{clamp_max_deg:.1f} deg)"
            )
        logger.info(msg)
        try:
            if isinstance(st, dict):
                st["phase_corr_clamp_deg"] = float(clamp_max_deg)
                st["phase_corr_clamp_min_deg"] = float(clamp_min_deg)
                st["phase_corr_clamp_max_deg"] = float(clamp_max_deg)
                st["phase_corr_clamp_mean_deg"] = (
                    float(np.mean(limit_deg_arr[phase_mask]))
                    if np.any(phase_mask)
                    else float(np.mean(limit_deg_arr))
                )
                st["phase_corr_max_before_deg"] = float(before_deg)
                st["phase_corr_max_after_deg"] = float(after_deg)
                st["phase_corr_clipped"] = bool(clipped)
                st["phase_corr_clipped_bins"] = int(clipped_bins)
                st["phase_corr_clamp_msg"] = str(msg)
        except (TypeError, ValueError):
            pass
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass

    if is_mixed:
        try:
            max_excess_delay_ms = float(getattr(cfg, "max_excess_delay_ms", 2.5) or 0.0)
        except (AttributeError, TypeError, ValueError):
            max_excess_delay_ms = 0.0
        if np.isfinite(max_excess_delay_ms) and max_excess_delay_ms > 0.0:
            try:
                max_gd_ms = _max_abs_group_delay_ms(f, extra_phase, phase_mask)
                if np.isfinite(max_gd_ms) and max_gd_ms > max_excess_delay_ms:
                    gd_scale = float(np.clip(max_excess_delay_ms / max(max_gd_ms, 1e-9), 0.05, 1.0))
                    extra_phase *= gd_scale
                    logger.info(
                        "Mixed phase excess-delay guard: "
                        f"max|GD|={max_gd_ms:.2f} ms -> target<={max_excess_delay_ms:.2f} ms "
                        f"(scale={gd_scale:.3f})"
                    )
                    try:
                        if isinstance(st, dict):
                            st["mixed_max_excess_delay_ms"] = float(max_excess_delay_ms)
                            st["mixed_excess_delay_before_ms"] = float(max_gd_ms)
                            st["mixed_excess_delay_scale"] = float(gd_scale)
                    except (TypeError, ValueError):
                        pass
            except (TypeError, ValueError, FloatingPointError, IndexError):
                pass

    if is_mixed:
        try:
            max_pre_db = float(getattr(cfg, "max_pre_ringing_db", -35.0) or -35.0)
        except (AttributeError, TypeError, ValueError):
            max_pre_db = -35.0
        if np.isfinite(max_pre_db):
            max_pre_db = float(min(max_pre_db, 0.0))
            extra_guard = np.asarray(extra_phase, dtype=float).copy()
            pre_before_db = None
            pre_after_db = None
            guard_scale_total = 1.0
            h_min = _build_complex_spectrum(phase_components.total_mag, min_p)
            ir_min = _ifft_to_ir(h_min, n=phase_components.n_fft)
            for i in range(3):
                h_lin_guard = _build_complex_spectrum(phase_components.total_mag, _merge_minphase_and_excess(-theo_xo, extra_guard))
                ir_lin_guard = _ifft_to_ir(h_lin_guard, n=phase_components.n_fft)
                ir_mixed_guard = combine_mixed_phase(
                    ir_lin_guard,
                    ir_min,
                    fs=float(cfg.fs),
                    split_freq=phase_components.mixed_split_hz,
                    transition_hz=phase_components.mixed_transition_hz,
                )
                pre_now_db = _pre_ringing_db(ir_mixed_guard)
                if i == 0:
                    pre_before_db = float(pre_now_db)
                pre_after_db = float(pre_now_db)
                if (not np.isfinite(pre_now_db)) or (pre_now_db <= max_pre_db):
                    break
                ratio_now = 10.0 ** (pre_now_db / 10.0)
                ratio_target = 10.0 ** (max_pre_db / 10.0)
                step_scale = float(np.clip(np.sqrt(ratio_target / max(ratio_now, 1e-30)), 0.20, 0.95))
                extra_guard *= step_scale
                guard_scale_total *= step_scale

            if guard_scale_total < 0.999:
                extra_phase = extra_guard
                logger.info(
                    "Mixed phase pre-ringing guard: "
                    f"{pre_before_db:.1f} dB -> {pre_after_db:.1f} dB "
                    f"(limit={max_pre_db:.1f} dB, scale={guard_scale_total:.3f})"
                )
            try:
                if isinstance(st, dict):
                    st["mixed_max_pre_ringing_db"] = float(max_pre_db)
                    st["mixed_pre_ringing_before_db"] = None if pre_before_db is None else float(pre_before_db)
                    st["mixed_pre_ringing_after_db"] = None if pre_after_db is None else float(pre_after_db)
                    st["mixed_pre_ringing_scale"] = float(guard_scale_total)
            except (TypeError, ValueError):
                pass

        try:
            corr_band = phase_mask & (np.abs(excess_phase) > 1e-12)
            if np.any(corr_band):
                eff = np.abs(extra_phase[corr_band]) / np.maximum(np.abs(excess_phase[corr_band]), 1e-12)
                if isinstance(st, dict):
                    st["mixed_phase_eff_strength_mean"] = float(np.mean(eff))
                    st["mixed_phase_eff_strength_max"] = float(np.max(eff))
        except (TypeError, ValueError, FloatingPointError):
            pass

    extra_phase, gd_lim_info = _gd_grad_limiter(
        extra_phase,
        cfg,
        st,
        freq_axis=f,
        phase_mask=phase_mask,
        use_bassfirst=phase_components.use_bassfirst,
        afdw_on=phase_components.afdw_on,
        limiter_fn=phase_components.limit_gd_gradient_ms_per_oct_fn,
    )
    if not is_mixed:
        extra_phase = _smooth_linear_boundary(f, extra_phase, phase_lim_hz, cfg, st)
        extra_phase = _enforce_linear_tail_decay(f, extra_phase, phase_lim_hz, cfg, st)
    try:
        gd_lim_enabled = bool(gd_lim_info.get("enabled", False))
        gd_reason = str(gd_lim_info.get("reason", "unknown"))
        gd_limit = gd_lim_info.get("limit_ms_per_oct", None)
        gd_before = gd_lim_info.get("max_grad_before_ms_per_oct", None)
        gd_after = gd_lim_info.get("max_grad_after_ms_per_oct", None)
        if gd_lim_enabled:
            if gd_limit is None:
                logger.info(
                    "GD gradient limiter: ON "
                    f"(reason={gd_reason}, max|dGD/dOct| {float(gd_before or 0.0):.2f} -> {float(gd_after or 0.0):.2f} ms/oct)"
                )
            else:
                logger.info(
                    "GD gradient limiter: ON "
                    f"(reason={gd_reason}, limit={float(gd_limit):.2f} ms/oct, "
                    f"max|dGD/dOct| {float(gd_before or 0.0):.2f} -> {float(gd_after or 0.0):.2f} ms/oct)"
                )
        else:
            logger.info(
                "GD gradient limiter: OFF "
                f"(reason={gd_reason}, max|dGD/dOct|={float(gd_before or 0.0):.2f} ms/oct)"
            )
    except (AttributeError, TypeError, ValueError):
        pass

    low_phase = _merge_minphase_and_excess(-theo_xo, extra_phase)
    phase_components.low_phase = low_phase
    phase_components.extra_phase = extra_phase

    if "Min" in cfg.filter_type_str:
        final_phase = min_p
    elif is_mixed:
        final_phase = low_phase
    else:
        sm_mask = _linear_to_minphase_blend_mask(f, phase_lim_hz, cfg, st)
        # Keep XO/HPF theoretical phase in the final baseline. phase_limit should
        # only fade out room excess-phase correction, not remove the crossover model.
        if _has_active_theoretical_phase_model(cfg):
            final_phase = (1.0 - sm_mask) * low_phase + sm_mask * (-theo_xo)
        else:
            final_phase = (1.0 - sm_mask) * low_phase + sm_mask * min_p
    return final_phase
