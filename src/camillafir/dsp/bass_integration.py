from __future__ import annotations

from typing import Any

import numpy as np
import scipy.signal

from ..auto_mode.shared import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    _auto_bass_integration_profile_norm,
    _auto_bass_integration_profile_weights,
)

AVR_CROSSOVER_CANDIDATES: tuple[float, ...] = (40.0, 60.0, 70.0, 80.0, 90.0, 110.0, 120.0, 150.0, 180.0)
DIRECT_DAC_CROSSOVER_STEP_HZ = 0.5
DIRECT_DAC_ALLPASS_FREQ_MULTIPLIERS: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00, 1.15, 1.35, 1.60)
DIRECT_DAC_ALLPASS_Q_CANDIDATES: tuple[float, ...] = (0.45, 0.60, 0.80, 1.00, 1.30, 1.70, 2.20)
DIRECT_DAC_ALLPASS_REFINE_FREQ_FACTORS: tuple[float, ...] = (0.88, 0.94, 0.98, 1.00, 1.02, 1.06, 1.12)
DIRECT_DAC_ALLPASS_REFINE_Q_FACTORS: tuple[float, ...] = (0.78, 0.90, 0.97, 1.00, 1.03, 1.10, 1.22)
DIRECT_DAC_ALLPASS_MIN_IMPROVEMENT_SCORE = 0.08
DIRECT_DAC_ALLPASS_MIN_CANCEL_IMPROVEMENT = 0.010
DIRECT_DAC_ALLPASS_MIN_RIPPLE_IMPROVEMENT_DB = 0.12
DIRECT_DAC_ALLPASS_MIN_GD_IMPROVEMENT_MS = 0.04
from ..io.measurement_bundle import BassIntegrationBundle, TransferData


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        if np.isfinite(out):
            return float(out)
    except Exception:
        pass
    return float(default)


def _band_mask(freqs_hz: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    try:
        f = np.asarray(freqs_hz, dtype=float)
    except Exception:
        return np.zeros(0, dtype=bool)
    lo = _safe_float(lo_hz, float("nan"))
    hi = _safe_float(hi_hz, float("nan"))
    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
        return np.zeros(f.shape, dtype=bool)
    return np.isfinite(f) & (f >= float(lo)) & (f <= float(hi))


def _normalize_candidate_frequencies(candidates: Any) -> tuple[float, ...]:
    if candidates is None:
        return ()
    out: list[float] = []
    seen: set[float] = set()
    try:
        iterator = tuple(candidates)
    except TypeError:
        return ()
    for candidate in iterator:
        fc = _safe_float(candidate, float("nan"))
        if not np.isfinite(fc) or fc <= 0.0:
            continue
        key = float(round(float(fc), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(float(fc))
    return tuple(out)


def _normalize_candidate_q_values(candidates: Any) -> tuple[float, ...]:
    if candidates is None:
        return ()
    out: list[float] = []
    seen: set[float] = set()
    try:
        iterator = tuple(candidates)
    except TypeError:
        return ()
    for candidate in iterator:
        q = _safe_float(candidate, float("nan"))
        if not np.isfinite(q) or q <= 0.0:
            continue
        key = float(round(float(q), 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(float(q))
    return tuple(out)


def _default_direct_dac_crossover_candidates() -> tuple[float, ...]:
    lo_hz = float(AVR_CROSSOVER_CANDIDATES[0])
    hi_hz = float(AVR_CROSSOVER_CANDIDATES[-1])
    step_hz = float(DIRECT_DAC_CROSSOVER_STEP_HZ)
    steps = max(0, int(round((hi_hz - lo_hz) / step_hz)))
    return tuple(float(lo_hz + step_hz * idx) for idx in range(steps + 1))


def _interp_complex_response(source: TransferData, target_freqs_hz: np.ndarray) -> np.ndarray:
    src_f = np.asarray(source.freqs_hz, dtype=float)
    src_c = np.asarray(source.complex_spec, dtype=np.complex128)
    dst_f = np.asarray(target_freqs_hz, dtype=float)
    if src_f.size < 2 or src_c.size != src_f.size or dst_f.size < 1:
        return np.zeros(dst_f.shape, dtype=np.complex128)
    if src_f.size == dst_f.size and np.allclose(src_f, dst_f, rtol=0.0, atol=1e-9):
        return src_c.astype(np.complex128, copy=False)
    re = np.interp(dst_f, src_f, np.real(src_c), left=np.real(src_c[0]), right=np.real(src_c[-1]))
    im = np.interp(dst_f, src_f, np.imag(src_c), left=np.imag(src_c[0]), right=np.imag(src_c[-1]))
    return np.asarray(re + 1j * im, dtype=np.complex128)


def _build_transfer_like(
    template: TransferData,
    complex_spec: np.ndarray,
    *,
    label: str,
) -> TransferData:
    spec = np.asarray(complex_spec, dtype=np.complex128)
    freqs = np.asarray(template.freqs_hz, dtype=float)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(spec), 1e-12))
    phase_deg = np.rad2deg(np.unwrap(np.angle(spec)))
    return TransferData(
        freqs_hz=freqs,
        complex_spec=spec,
        mag_db=np.asarray(mag_db, dtype=float),
        phase_deg=np.asarray(phase_deg, dtype=float),
        sample_rate=int(template.sample_rate),
        label=str(label or ""),
    )


def _sum_component_specs(template: TransferData, components: tuple[TransferData, ...]) -> np.ndarray:
    freqs = np.asarray(template.freqs_hz, dtype=float)
    total_spec = np.asarray(template.complex_spec, dtype=np.complex128).copy()
    for component in components:
        total_spec += _interp_complex_response(component, freqs)
    return np.asarray(total_spec, dtype=np.complex128)


def _sum_sub_components(
    template: TransferData,
    *subs: TransferData,
    label: str = "",
) -> TransferData:
    freqs = np.asarray(template.freqs_hz, dtype=float)
    sub_spec = np.zeros(freqs.shape, dtype=np.complex128)
    for sub in subs:
        sub_spec += _interp_complex_response(sub, freqs)
    return _build_transfer_like(template, sub_spec, label=label)


def sum_complex_responses(
    main: TransferData,
    *subs: TransferData,
    label: str = "",
) -> TransferData:
    total_spec = _sum_component_specs(main, tuple(subs))
    return _build_transfer_like(main, total_spec, label=label)


def _butterworth_complex_response(
    freqs_hz: np.ndarray,
    cutoff_hz: float,
    order: int,
    *,
    btype: str,
) -> np.ndarray:
    freqs = np.asarray(freqs_hz, dtype=float)
    cutoff = _safe_float(cutoff_hz, float("nan"))
    try:
        ord_i = int(order)
    except Exception:
        ord_i = 0
    if freqs.size == 0 or (not np.isfinite(cutoff)) or cutoff <= 0.0 or ord_i <= 0:
        return np.ones(freqs.shape, dtype=np.complex128)
    try:
        b, a = scipy.signal.butter(
            max(1, int(ord_i)),
            2.0 * np.pi * float(cutoff),
            btype=str(btype),
            analog=True,
        )
        _, h = scipy.signal.freqs(b, a, worN=2.0 * np.pi * freqs)
        return np.asarray(h, dtype=np.complex128)
    except Exception:
        return np.ones(freqs.shape, dtype=np.complex128)


def _allpass2_complex_response(freqs_hz: np.ndarray, freq_hz: float, q: float) -> np.ndarray:
    freqs = np.asarray(freqs_hz, dtype=float)
    fc = _safe_float(freq_hz, float("nan"))
    q_v = _safe_float(q, float("nan"))
    if freqs.size == 0 or (not np.isfinite(fc)) or fc <= 0.0 or (not np.isfinite(q_v)) or q_v <= 0.0:
        return np.ones(freqs.shape, dtype=np.complex128)
    omega = 2.0 * np.pi * freqs
    omega_0 = 2.0 * np.pi * float(fc)
    s = 1j * omega
    damping = float(omega_0 / max(float(q_v), 1e-9))
    num = (s ** 2) - damping * s + (omega_0 ** 2)
    den = (s ** 2) + damping * s + (omega_0 ** 2)
    den = np.where(np.abs(den) < 1e-18, 1e-18 + 0j, den)
    return np.asarray(num / den, dtype=np.complex128)


def _apply_allpass_to_transfer(
    transfer: TransferData,
    *,
    freq_hz: float,
    q: float,
    label: str,
) -> TransferData:
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    spec = np.asarray(transfer.complex_spec, dtype=np.complex128).copy()
    if spec.size != freqs.size:
        spec = _interp_complex_response(transfer, freqs)
    spec *= _allpass2_complex_response(freqs, float(freq_hz), float(q))
    return _build_transfer_like(transfer, spec, label=label)


def _apply_branch_filters(
    transfer: TransferData,
    *,
    hpf_hz: float | None = None,
    hpf_order: int | None = None,
    lpf_hz: float | None = None,
    lpf_order: int | None = None,
    label: str,
) -> TransferData:
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    spec = np.asarray(transfer.complex_spec, dtype=np.complex128).copy()
    if spec.size != freqs.size:
        spec = _interp_complex_response(transfer, freqs)
    if hpf_hz is not None and hpf_order is not None:
        spec *= _butterworth_complex_response(
            freqs,
            float(hpf_hz),
            int(hpf_order),
            btype="high",
        )
    if lpf_hz is not None and lpf_order is not None:
        spec *= _butterworth_complex_response(
            freqs,
            float(lpf_hz),
            int(lpf_order),
            btype="low",
        )
    return _build_transfer_like(transfer, spec, label=label)


def _build_direct_dac_trial_bundle(
    bundle: BassIntegrationBundle,
    *,
    fc_hz: float,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hpf_hz: float,
    sub_hpf_order: int,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
) -> BassIntegrationBundle:
    fc = _safe_float(fc_hz, 80.0)
    xo_order = max(1, int(main_hpf_order))
    lpf_order = max(1, int(sub_lpf_order))
    sub_hp_hz = max(0.0, _safe_float(sub_hpf_hz, 0.0))
    sub_hp_order = max(1, int(sub_hpf_order))

    l_main_f = _apply_branch_filters(
        bundle.l_main,
        hpf_hz=fc,
        hpf_order=xo_order,
        label="L main + HPF trial",
    )
    r_main_f = _apply_branch_filters(
        bundle.r_main,
        hpf_hz=fc,
        hpf_order=xo_order,
        label="R main + HPF trial",
    )
    l_sub_f = _apply_branch_filters(
        bundle.l_sub,
        hpf_hz=sub_hp_hz,
        hpf_order=sub_hp_order,
        lpf_hz=fc,
        lpf_order=lpf_order,
        label="L sub + LPF/HPF trial",
    )
    r_sub_f = _apply_branch_filters(
        bundle.r_sub,
        hpf_hz=sub_hp_hz,
        hpf_order=sub_hp_order,
        lpf_hz=fc,
        lpf_order=lpf_order,
        label="R sub + LPF/HPF trial",
    )
    ap_freq_hz = _safe_float(sub_allpass_freq_hz, float("nan"))
    ap_q = _safe_float(sub_allpass_q, float("nan"))
    if np.isfinite(ap_freq_hz) and ap_freq_hz > 0.0 and np.isfinite(ap_q) and ap_q > 0.0:
        l_sub_f = _apply_allpass_to_transfer(
            l_sub_f,
            freq_hz=float(ap_freq_hz),
            q=float(ap_q),
            label="L sub + LPF/HPF/AP trial",
        )
        r_sub_f = _apply_allpass_to_transfer(
            r_sub_f,
            freq_hz=float(ap_freq_hz),
            q=float(ap_q),
            label="R sub + LPF/HPF/AP trial",
        )

    l_total_f = sum_complex_responses(l_main_f, l_sub_f, r_sub_f, label="L Direct-DAC trial total")
    r_total_f = sum_complex_responses(r_main_f, l_sub_f, r_sub_f, label="R Direct-DAC trial total")
    return BassIntegrationBundle(
        l_main=l_main_f,
        r_main=r_main_f,
        l_sub=l_sub_f,
        r_sub=r_sub_f,
        l_total=l_total_f,
        r_total=r_total_f,
        avr_crossover_hz=float(fc),
        profile=str(bundle.profile or "safe"),
        diagnostics={},
    )


def _direct_dac_metric_snapshot(
    diag: dict[str, Any] | None,
    gd_cont: dict[str, Any] | None,
    *,
    enabled: bool,
    freq_hz: float,
    q: float,
) -> dict[str, float | bool]:
    diag_obj = dict(diag or {})
    gd_obj = dict(gd_cont or {})
    return {
        "allpass_enabled": bool(enabled),
        "allpass_freq_hz": float(freq_hz),
        "allpass_q": float(q),
        "cancellation_risk": _safe_float(diag_obj.get("cancellation_risk", float("nan")), float("nan")),
        "overlap_ripple_db": _safe_float(diag_obj.get("overlap_ripple_db", float("nan")), float("nan")),
        "sub_dominance_db": _safe_float(diag_obj.get("sub_dominance_db", float("nan")), float("nan")),
        "xo_gd_mismatch_ms": _safe_float(gd_obj.get("avg_gd_mismatch_ms", float("nan")), float("nan")),
        "xo_l_gd_mismatch_ms": _safe_float(gd_obj.get("l_gd_mismatch_ms", float("nan")), float("nan")),
        "xo_r_gd_mismatch_ms": _safe_float(gd_obj.get("r_gd_mismatch_ms", float("nan")), float("nan")),
        "xo_main_gd_ms": _safe_float(
            (
                _safe_float(gd_obj.get("l_main_gd_ms", float("nan")), float("nan"))
                + _safe_float(gd_obj.get("r_main_gd_ms", float("nan")), float("nan"))
            )
            / 2.0,
            float("nan"),
        ),
        "xo_sub_gd_ms": _safe_float(gd_obj.get("sub_gd_ms", float("nan")), float("nan")),
    }


def _direct_dac_alignment_objective(
    diag: dict[str, Any] | None,
    gd_cont: dict[str, Any] | None,
    *,
    ap_freq_hz: float,
    ap_q: float,
    profile: str,
) -> float:
    weights = _auto_bass_integration_profile_weights(profile)
    cancel = _safe_float(dict(diag or {}).get("cancellation_risk", float("nan")), float("nan"))
    ripple = _safe_float(dict(diag or {}).get("overlap_ripple_db", float("nan")), float("nan"))
    dominance = _safe_float(dict(diag or {}).get("sub_dominance_db", float("nan")), float("nan"))
    gd_mm = _safe_float(dict(gd_cont or {}).get("avg_gd_mismatch_ms", float("nan")), float("nan"))
    if not (np.isfinite(cancel) and np.isfinite(ripple) and np.isfinite(dominance) and np.isfinite(gd_mm)):
        return float("nan")
    q_pen = max(0.0, _safe_float(ap_q, 0.0) - 0.90) / 1.30
    penalty = (
        float(weights.get("cancellation", 8.0)) * float(cancel)
        + float(weights.get("overlap_ripple", 1.8)) * (float(ripple) / 10.0)
        + float(weights.get("xo_gd_continuity", 0.8)) * (float(gd_mm) / 3.0)
        + float(weights.get("sub_dominance", 0.9)) * (abs(float(dominance)) / 8.0)
        + 0.18 * float(q_pen)
    )
    return float(-penalty)


def compute_direct_dac_bass_integration_analysis(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, Any]:
    ap_freq_hz = _safe_float(sub_allpass_freq_hz, float("nan"))
    ap_q = _safe_float(sub_allpass_q, float("nan"))
    ap_enabled = bool(np.isfinite(ap_freq_hz) and ap_freq_hz > 0.0 and np.isfinite(ap_q) and ap_q > 0.0)
    if not ap_enabled:
        ap_freq_hz = 0.0
        ap_q = 0.707
    trial_bundle = _build_direct_dac_trial_bundle(
        bundle,
        fc_hz=float(fc_hz),
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
        sub_allpass_freq_hz=(float(ap_freq_hz) if ap_enabled else None),
        sub_allpass_q=(float(ap_q) if ap_enabled else None),
    )
    diag = compute_bass_integration_diagnostics(
        trial_bundle,
        float(fc_hz),
        profile,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
    )
    gd_cont = compute_xo_gd_continuity(trial_bundle, float(fc_hz))
    return {
        "trial_bundle": trial_bundle,
        "diagnostics": dict(diag or {}),
        "gd_continuity": dict(gd_cont or {}),
        "objective": _direct_dac_alignment_objective(
            diag,
            gd_cont,
            ap_freq_hz=float(ap_freq_hz),
            ap_q=float(ap_q),
            profile=profile,
        ),
        "allpass_enabled": bool(ap_enabled),
        "allpass_freq_hz": float(ap_freq_hz),
        "allpass_q": float(ap_q),
        "snapshot": _direct_dac_metric_snapshot(
            diag,
            gd_cont,
            enabled=bool(ap_enabled),
            freq_hz=float(ap_freq_hz),
            q=float(ap_q),
        ),
    }


def compute_direct_dac_bass_integration_diagnostics(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, Any]:
    analysis = compute_direct_dac_bass_integration_analysis(
        bundle,
        float(fc_hz),
        profile,
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
        sub_allpass_freq_hz=sub_allpass_freq_hz,
        sub_allpass_q=sub_allpass_q,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
    )
    out = dict(analysis.get("diagnostics", {}) or {})
    try:
        out["direct_dac_main_hpf_order"] = int(main_hpf_order)
        out["direct_dac_sub_lpf_order"] = int(sub_lpf_order)
        out["direct_dac_sub_hpf_hz"] = float(sub_hpf_hz)
        out["direct_dac_sub_hpf_order"] = int(sub_hpf_order)
        out["direct_dac_sub_allpass_enabled"] = bool(analysis.get("allpass_enabled", False))
        out["direct_dac_sub_allpass_freq_hz"] = float(analysis.get("allpass_freq_hz", 0.0))
        out["direct_dac_sub_allpass_q"] = float(analysis.get("allpass_q", 0.707))
    except Exception:
        pass
    return out


def _channel_overlap_metrics(
    main: TransferData,
    sub: TransferData,
    total: TransferData,
    *,
    lo_hz: float,
    hi_hz: float,
) -> dict[str, float]:
    freqs = np.asarray(total.freqs_hz, dtype=float)
    mask = _band_mask(freqs, lo_hz, hi_hz)
    if int(np.count_nonzero(mask)) < 3:
        return {
            "overlap_ratio": float("nan"),
            "overlap_ripple_db": float("nan"),
            "cancellation_risk": float("nan"),
            "sub_dominance_db": float("nan"),
        }

    main_spec = _interp_complex_response(main, freqs)
    sub_spec = _interp_complex_response(sub, freqs)
    total_spec = _interp_complex_response(total, freqs)

    main_mag = np.maximum(np.abs(main_spec[mask]), 1e-12)
    sub_mag = np.maximum(np.abs(sub_spec[mask]), 1e-12)
    total_mag = np.maximum(np.abs(total_spec[mask]), 1e-12)
    stronger = np.maximum(main_mag, sub_mag)
    weaker = np.minimum(main_mag, sub_mag)
    overlap_ratio = float(np.mean(weaker / np.maximum(stronger, 1e-12)))

    total_db = 20.0 * np.log10(np.maximum(total_mag, 1e-12))
    if total_db.size >= 5:
        ripple_db = float(np.percentile(total_db, 95.0) - np.percentile(total_db, 5.0))
    else:
        ripple_db = float(np.max(total_db) - np.min(total_db))

    main_phase = np.angle(main_spec[mask])
    sub_phase = np.angle(sub_spec[mask])
    phase_delta = np.angle(np.exp(1j * (sub_phase - main_phase)))
    phase_opposition = np.clip((-np.cos(phase_delta) - 0.15) / 0.85, 0.0, 1.0)
    depth_db = 20.0 * np.log10(np.maximum(stronger, 1e-12) / np.maximum(total_mag, 1e-12))
    depth_weight = np.clip(depth_db / 12.0, 0.0, 1.0)
    cancellation_risk = float(np.mean(phase_opposition * depth_weight))

    sub_dominance_db = float(np.median(20.0 * np.log10(sub_mag / np.maximum(main_mag, 1e-12))))
    return {
        "overlap_ratio": float(overlap_ratio),
        "overlap_ripple_db": float(ripple_db),
        "cancellation_risk": float(cancellation_risk),
        "sub_dominance_db": float(sub_dominance_db),
    }


def compute_overlap_metrics(
    main: TransferData,
    sub: TransferData,
    total: TransferData,
    *,
    fc_hz: float,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, float]:
    fc = _safe_float(fc_hz, 80.0)
    lo = max(5.0, fc * max(0.05, _safe_float(guard_lo_ratio, 0.60)))
    hi = max(lo + 1.0, fc * max(0.05, _safe_float(guard_hi_ratio, 1.40)))
    ch = _channel_overlap_metrics(main, sub, total, lo_hz=lo, hi_hz=hi)
    return {
        "guard_lo_hz": float(lo),
        "guard_hi_hz": float(hi),
        "overlap_ratio": float(ch.get("overlap_ratio", float("nan"))),
        "overlap_ripple_db": float(ch.get("overlap_ripple_db", float("nan"))),
    }


def compute_cancellation_metrics(
    main: TransferData,
    sub: TransferData,
    total: TransferData,
    *,
    fc_hz: float,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, float]:
    fc = _safe_float(fc_hz, 80.0)
    lo = max(5.0, fc * max(0.05, _safe_float(guard_lo_ratio, 0.60)))
    hi = max(lo + 1.0, fc * max(0.05, _safe_float(guard_hi_ratio, 1.40)))
    ch = _channel_overlap_metrics(main, sub, total, lo_hz=lo, hi_hz=hi)
    return {
        "guard_lo_hz": float(lo),
        "guard_hi_hz": float(hi),
        "cancellation_risk": float(ch.get("cancellation_risk", float("nan"))),
    }


def compute_sub_dominance_metrics(
    main: TransferData,
    sub: TransferData,
    *,
    fc_hz: float,
) -> dict[str, float]:
    freqs = np.asarray(main.freqs_hz, dtype=float)
    fc = _safe_float(fc_hz, 80.0)
    lo = max(5.0, 0.35 * fc)
    hi = max(lo + 1.0, 1.00 * fc)
    mask = _band_mask(freqs, lo, hi)
    if int(np.count_nonzero(mask)) < 3:
        return {
            "sub_dominance_db": float("nan"),
            "sub_dominance_lo_hz": float(lo),
            "sub_dominance_hi_hz": float(hi),
        }
    main_spec = _interp_complex_response(main, freqs)
    sub_spec = _interp_complex_response(sub, freqs)
    main_mag = np.maximum(np.abs(main_spec[mask]), 1e-12)
    sub_mag = np.maximum(np.abs(sub_spec[mask]), 1e-12)
    return {
        "sub_dominance_db": float(np.median(20.0 * np.log10(sub_mag / np.maximum(main_mag, 1e-12)))),
        "sub_dominance_lo_hz": float(lo),
        "sub_dominance_hi_hz": float(hi),
    }


def compute_bass_integration_diagnostics(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, Any]:
    fc = _safe_float(fc_hz, 80.0)
    lo_ratio = max(0.05, _safe_float(guard_lo_ratio, 0.60))
    hi_ratio = max(lo_ratio + 0.05, _safe_float(guard_hi_ratio, 1.40))
    profile_name = _auto_bass_integration_profile_norm(profile)

    l_sub_total = _sum_sub_components(
        bundle.l_main,
        bundle.l_sub,
        bundle.r_sub,
        label="L combined sub",
    )
    r_sub_total = _sum_sub_components(
        bundle.r_main,
        bundle.l_sub,
        bundle.r_sub,
        label="R combined sub",
    )

    l_metrics = _channel_overlap_metrics(
        bundle.l_main,
        l_sub_total,
        bundle.l_total,
        lo_hz=max(5.0, fc * lo_ratio),
        hi_hz=max(5.0, fc * hi_ratio),
    )
    r_metrics = _channel_overlap_metrics(
        bundle.r_main,
        r_sub_total,
        bundle.r_total,
        lo_hz=max(5.0, fc * lo_ratio),
        hi_hz=max(5.0, fc * hi_ratio),
    )

    def _avg_metric(key: str) -> float:
        vals = [
            _safe_float((l_metrics or {}).get(key), float("nan")),
            _safe_float((r_metrics or {}).get(key), float("nan")),
        ]
        vals = [float(v) for v in vals if np.isfinite(v)]
        return float(np.mean(np.asarray(vals, dtype=float))) if vals else float("nan")

    out = {
        "profile": profile_name,
        "avr_crossover_hz": float(fc),
        "guard_lo_ratio": float(lo_ratio),
        "guard_hi_ratio": float(hi_ratio),
        "guard_lo_hz": float(max(5.0, fc * lo_ratio)),
        "guard_hi_hz": float(max(5.0, fc * hi_ratio)),
        "overlap_ratio": _avg_metric("overlap_ratio"),
        "overlap_ripple_db": _avg_metric("overlap_ripple_db"),
        "cancellation_risk": _avg_metric("cancellation_risk"),
        "sub_dominance_db": _avg_metric("sub_dominance_db"),
        "channels": {
            "l": dict(l_metrics),
            "r": dict(r_metrics),
        },
    }
    return out


def _gd_ms_from_transfer(transfer: TransferData) -> np.ndarray:
    """Compute group delay (ms) from unwrapped phase_deg in TransferData."""
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    phase_rad = np.deg2rad(np.asarray(transfer.phase_deg, dtype=float))
    omega = 2.0 * np.pi * np.maximum(freqs, 1e-9)
    gd_s = -np.gradient(phase_rad, omega)
    return np.asarray(gd_s * 1000.0, dtype=float)


def _gd_ms_at_hz(transfer: TransferData, target_hz: float) -> float:
    """Interpolate group delay (ms) of a TransferData at a specific frequency."""
    freqs = np.asarray(transfer.freqs_hz, dtype=float)
    gd_ms = _gd_ms_from_transfer(transfer)
    if freqs.size < 2 or gd_ms.size != freqs.size:
        return float("nan")
    return float(np.interp(float(target_hz), freqs, gd_ms))


def _main_guard_band_drop_db(main: TransferData, fc_hz: float) -> float:
    """Return how far (dB) the main speaker has fallen in the XO guard band
    relative to its 200–600 Hz midrange reference.  Positive = rolling off."""
    freqs = np.asarray(main.freqs_hz, dtype=float)
    mag_db = np.asarray(main.mag_db, dtype=float)
    ref_mask = _band_mask(freqs, 200.0, 600.0)
    if int(np.count_nonzero(ref_mask)) < 3:
        return float("nan")
    ref_db = float(np.mean(mag_db[ref_mask]))
    fc = _safe_float(fc_hz, 80.0)
    lo = max(5.0, 0.6 * fc)
    hi = max(lo + 1.0, 1.4 * fc)
    guard_mask = _band_mask(freqs, lo, hi)
    if int(np.count_nonzero(guard_mask)) < 3:
        return float("nan")
    guard_db = float(np.mean(mag_db[guard_mask]))
    return float(ref_db - guard_db)


def compute_xo_gd_continuity(
    bundle: BassIntegrationBundle,
    fc_hz: float,
) -> dict[str, float]:
    """Compute group delay mismatch between main and combined sub at fc_hz.

    The combined sub (L+R sub complex sum) GD is compared to each main
    channel GD at the crossover frequency.  A mismatch >0 ms means timing
    is misaligned at the handoff point.

    Returns dict with:
      l_main_gd_ms, l_sub_gd_ms, l_gd_mismatch_ms
      r_main_gd_ms, r_sub_gd_ms, r_gd_mismatch_ms
      avg_gd_mismatch_ms   — average of L and R mismatch
      fc_hz                — crossover frequency used
    """
    fc = _safe_float(fc_hz, 80.0)
    freqs = np.asarray(bundle.l_main.freqs_hz, dtype=float)

    # Combined sub complex spectrum (L+R sub summed), then compute GD
    l_sub_spec = _interp_complex_response(bundle.l_sub, freqs)
    r_sub_spec = _interp_complex_response(bundle.r_sub, freqs)
    combined_sub_spec = l_sub_spec + r_sub_spec
    combined_sub_phase_rad = np.unwrap(np.angle(combined_sub_spec))
    omega = 2.0 * np.pi * np.maximum(freqs, 1e-9)
    combined_sub_gd_ms = -np.gradient(combined_sub_phase_rad, omega) * 1000.0
    sub_gd_at_fc = float(np.interp(fc, freqs, combined_sub_gd_ms))

    l_main_gd = _gd_ms_at_hz(bundle.l_main, fc)
    r_main_gd = _gd_ms_at_hz(bundle.r_main, fc)

    def _mismatch(main_gd: float, sub_gd: float) -> float:
        if np.isfinite(main_gd) and np.isfinite(sub_gd):
            return float(abs(main_gd - sub_gd))
        return float("nan")

    l_mm = _mismatch(l_main_gd, sub_gd_at_fc)
    r_mm = _mismatch(r_main_gd, sub_gd_at_fc)

    valid = [v for v in (l_mm, r_mm) if np.isfinite(v)]
    avg_mm = float(np.mean(np.asarray(valid, dtype=float))) if valid else float("nan")

    return {
        "fc_hz": float(fc),
        "l_main_gd_ms": float(l_main_gd),
        "r_main_gd_ms": float(r_main_gd),
        "sub_gd_ms": float(sub_gd_at_fc),
        "l_gd_mismatch_ms": float(l_mm),
        "r_gd_mismatch_ms": float(r_mm),
        "avg_gd_mismatch_ms": float(avg_mm),
    }


def compute_bass_integration_metric_payload(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    mode: str = "avr_lfe_main_decomposed",
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, Any]:
    mode_norm = str(mode or "avr_lfe_main_decomposed").strip().lower()
    if mode_norm == "direct_dac":
        analysis = compute_direct_dac_bass_integration_analysis(
            bundle,
            float(fc_hz),
            profile,
            main_hpf_order=int(main_hpf_order),
            sub_lpf_order=int(sub_lpf_order),
            sub_hpf_hz=float(sub_hpf_hz),
            sub_hpf_order=int(sub_hpf_order),
            sub_allpass_freq_hz=sub_allpass_freq_hz,
            sub_allpass_q=sub_allpass_q,
            guard_lo_ratio=float(guard_lo_ratio),
            guard_hi_ratio=float(guard_hi_ratio),
        )
        diag = dict(analysis.get("diagnostics", {}) or {})
        gd_cont = dict(analysis.get("gd_continuity", {}) or {})
        allpass_enabled = bool(analysis.get("allpass_enabled", False))
        allpass_freq_hz = float(analysis.get("allpass_freq_hz", 0.0))
        allpass_q = float(analysis.get("allpass_q", 0.707))
    else:
        diag = compute_bass_integration_diagnostics(
            bundle,
            float(fc_hz),
            profile,
            guard_lo_ratio=float(guard_lo_ratio),
            guard_hi_ratio=float(guard_hi_ratio),
        )
        gd_cont = compute_xo_gd_continuity(bundle, float(fc_hz))
        allpass_enabled = False
        allpass_freq_hz = 0.0
        allpass_q = 0.707
    return {
        "bass_cancellation_risk": _safe_float(diag.get("cancellation_risk", float("nan")), float("nan")),
        "bass_overlap_ripple": _safe_float(diag.get("overlap_ripple_db", float("nan")), float("nan")),
        "bass_sub_dominance": _safe_float(diag.get("sub_dominance_db", float("nan")), float("nan")),
        "bass_guard_lo_hz": _safe_float(diag.get("guard_lo_hz", float("nan")), float("nan")),
        "bass_guard_hi_hz": _safe_float(diag.get("guard_hi_hz", float("nan")), float("nan")),
        "bass_integration_profile": str(_auto_bass_integration_profile_norm(profile)),
        "bass_integration_mode": str(mode_norm),
        "bass_xo_gd_mismatch_ms": _safe_float(gd_cont.get("avg_gd_mismatch_ms", float("nan")), float("nan")),
        "bass_xo_l_gd_mismatch_ms": _safe_float(gd_cont.get("l_gd_mismatch_ms", float("nan")), float("nan")),
        "bass_xo_r_gd_mismatch_ms": _safe_float(gd_cont.get("r_gd_mismatch_ms", float("nan")), float("nan")),
        "bass_xo_main_gd_ms": _safe_float(
            (
                _safe_float(gd_cont.get("l_main_gd_ms", float("nan")), float("nan"))
                + _safe_float(gd_cont.get("r_main_gd_ms", float("nan")), float("nan"))
            )
            / 2.0,
            float("nan"),
        ),
        "bass_xo_sub_gd_ms": _safe_float(gd_cont.get("sub_gd_ms", float("nan")), float("nan")),
        "bass_allpass_enabled": bool(allpass_enabled),
        "bass_allpass_freq_hz": float(allpass_freq_hz),
        "bass_allpass_q": float(allpass_q),
    }


def recommend_avr_crossover(
    bundle: BassIntegrationBundle,
    candidates: tuple[float, ...] = AVR_CROSSOVER_CANDIDATES,
    profile: str = "safe",
) -> dict[str, Any]:
    """Score each candidate crossover Hz and return the best one.

    Returns a dict with:
      recommended_hz   – the best candidate frequency
      scores           – per-candidate dict with score and raw metrics
    """
    weights = _auto_bass_integration_profile_weights(profile)
    w_cancel = float(weights.get("cancellation", 8.0))
    w_ripple = float(weights.get("overlap_ripple", 1.8))
    w_dom = float(weights.get("sub_dominance", 0.9))
    w_main_act = float(weights.get("main_activity", 6.0))

    scores: dict[float, dict[str, float]] = {}
    for fc in candidates:
        fc = float(fc)
        diag = compute_bass_integration_diagnostics(bundle, fc, profile)
        cancel = _safe_float(diag.get("cancellation_risk", float("nan")), float("nan"))
        ripple = _safe_float(diag.get("overlap_ripple_db", float("nan")), float("nan"))
        dominance = _safe_float(diag.get("sub_dominance_db", float("nan")), float("nan"))

        # Main speaker activity: how far has the main speaker fallen in the
        # XO guard band relative to its midrange reference.  A large drop
        # means the XO is below the speaker's bass extension → penalise.
        l_drop = _main_guard_band_drop_db(bundle.l_main, fc)
        r_drop = _main_guard_band_drop_db(bundle.r_main, fc)
        drop_vals = [v for v in (l_drop, r_drop) if np.isfinite(v)]
        main_drop_norm = max(0.0, float(np.mean(np.asarray(drop_vals, dtype=float)))) / 12.0 if drop_vals else float("nan")

        if np.isfinite(cancel) and np.isfinite(ripple) and np.isfinite(dominance):
            # Normalize each metric to approximate unit range before weighting:
            #   cancel:         0..1   (already normalized)
            #   ripple:         0..20 dB  → /10
            #   dominance:      0..12 dB deviation → /6
            #   main_drop_norm: 0..∞  (12 dB drop = 1 unit)
            penalty = (
                w_cancel * cancel
                + w_ripple * (ripple / 10.0)
                + w_dom * (abs(dominance) / 6.0)
            )
            if np.isfinite(main_drop_norm):
                penalty += w_main_act * main_drop_norm
            score = float(-penalty)
        else:
            score = float("nan")

        avg_drop = float(np.mean(np.asarray(drop_vals, dtype=float))) if drop_vals else float("nan")
        scores[fc] = {
            "score": score,
            "cancellation_risk": cancel,
            "overlap_ripple_db": ripple,
            "sub_dominance_db": dominance,
            "main_activity_drop_db": avg_drop,
        }

    valid = {fc: d["score"] for fc, d in scores.items() if np.isfinite(d["score"])}
    if valid:
        best_hz = float(max(valid, key=lambda fc: valid[fc]))
    else:
        best_hz = float(bundle.avr_crossover_hz)

    return {
        "recommended_hz": best_hz,
        "scores": scores,
    }


def recommend_direct_dac_crossover(
    bundle: BassIntegrationBundle,
    candidates: tuple[float, ...] | None = None,
    profile: str = "safe",
    *,
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
) -> dict[str, Any]:
    """Score Direct-DAC XO candidates after applying trial branch filters.

    Each candidate is tested by applying a main HPF and sub HPF+LPF to the
    measured branch responses, summing the filtered branches acoustically,
    and scoring the resulting handoff.

    When no explicit candidates are provided, Direct DAC uses a dedicated
    0.5 Hz search grid instead of AVR-standard crossover steps.
    """
    weights = _auto_bass_integration_profile_weights(profile)
    w_cancel = float(weights.get("cancellation", 8.0))
    w_ripple = float(weights.get("overlap_ripple", 1.8))
    w_dom = float(weights.get("sub_dominance", 0.9))
    w_main_act = float(weights.get("main_activity", 6.0))

    try:
        hpf_order_i = max(1, int(main_hpf_order))
    except Exception:
        hpf_order_i = 4
    try:
        lpf_order_i = max(1, int(sub_lpf_order))
    except Exception:
        lpf_order_i = hpf_order_i
    try:
        sub_hpf_hz_f = max(0.0, float(sub_hpf_hz))
    except Exception:
        sub_hpf_hz_f = 20.0
    try:
        sub_hpf_order_i = max(1, int(sub_hpf_order))
    except Exception:
        sub_hpf_order_i = 2

    search_candidates = _normalize_candidate_frequencies(candidates)
    if not search_candidates:
        search_candidates = _default_direct_dac_crossover_candidates()

    scores: dict[float, dict[str, float]] = {}
    for fc in search_candidates:
        fc = float(fc)
        if not np.isfinite(fc) or fc <= 0.0 or fc <= (sub_hpf_hz_f + 1.0):
            scores[fc] = {
                "score": float("nan"),
                "cancellation_risk": float("nan"),
                "overlap_ripple_db": float("nan"),
                "sub_dominance_db": float("nan"),
                "main_activity_drop_db": float("nan"),
            }
            continue

        diag = compute_direct_dac_bass_integration_diagnostics(
            bundle,
            fc,
            profile,
            main_hpf_order=hpf_order_i,
            sub_lpf_order=lpf_order_i,
            sub_hpf_hz=sub_hpf_hz_f,
            sub_hpf_order=sub_hpf_order_i,
        )
        cancel = _safe_float(diag.get("cancellation_risk", float("nan")), float("nan"))
        ripple = _safe_float(diag.get("overlap_ripple_db", float("nan")), float("nan"))
        dominance = _safe_float(diag.get("sub_dominance_db", float("nan")), float("nan"))

        l_drop = _main_guard_band_drop_db(bundle.l_main, fc)
        r_drop = _main_guard_band_drop_db(bundle.r_main, fc)
        drop_vals = [v for v in (l_drop, r_drop) if np.isfinite(v)]
        main_drop_norm = max(0.0, float(np.mean(np.asarray(drop_vals, dtype=float)))) / 12.0 if drop_vals else float("nan")

        if np.isfinite(cancel) and np.isfinite(ripple) and np.isfinite(dominance):
            penalty = (
                w_cancel * cancel
                + w_ripple * (ripple / 10.0)
                + w_dom * (abs(dominance) / 6.0)
            )
            if np.isfinite(main_drop_norm):
                penalty += w_main_act * main_drop_norm
            score = float(-penalty)
        else:
            score = float("nan")

        avg_drop = float(np.mean(np.asarray(drop_vals, dtype=float))) if drop_vals else float("nan")
        scores[fc] = {
            "score": score,
            "cancellation_risk": cancel,
            "overlap_ripple_db": ripple,
            "sub_dominance_db": dominance,
            "main_activity_drop_db": avg_drop,
        }

    valid = {fc: d["score"] for fc, d in scores.items() if np.isfinite(d["score"])}
    if valid:
        best_hz = float(max(valid, key=lambda fc: valid[fc]))
    else:
        best_hz = float(bundle.avr_crossover_hz)

    return {
        "recommended_hz": best_hz,
        "scores": scores,
    }


def recommend_direct_dac_allpass(
    bundle: BassIntegrationBundle,
    *,
    fc_hz: float,
    profile: str,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hpf_hz: float,
    sub_hpf_order: int,
) -> dict[str, Any]:
    fc = _safe_float(fc_hz, 80.0)
    sub_hp = max(0.0, _safe_float(sub_hpf_hz, 20.0))
    baseline_analysis = compute_direct_dac_bass_integration_analysis(
        bundle,
        float(fc),
        profile,
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hp),
        sub_hpf_order=int(sub_hpf_order),
    )
    baseline = dict(baseline_analysis.get("snapshot", {}) or {})
    baseline_score = _safe_float(baseline_analysis.get("objective", float("nan")), float("nan"))
    if (not np.isfinite(fc)) or fc <= 0.0 or fc <= (sub_hp + 1.0):
        return {
            "enabled": False,
            "freq_hz": 0.0,
            "q": 0.707,
            "baseline": baseline,
            "optimized": baseline,
            "improvement_score": 0.0,
            "reason": "No meaningful improvement found.",
        }

    def _evaluate(freq_hz: float, q: float) -> dict[str, Any] | None:
        analysis = compute_direct_dac_bass_integration_analysis(
            bundle,
            float(fc),
            profile,
            main_hpf_order=int(main_hpf_order),
            sub_lpf_order=int(sub_lpf_order),
            sub_hpf_hz=float(sub_hp),
            sub_hpf_order=int(sub_hpf_order),
            sub_allpass_freq_hz=float(freq_hz),
            sub_allpass_q=float(q),
        )
        score = _safe_float(analysis.get("objective", float("nan")), float("nan"))
        if not np.isfinite(score):
            return None
        out = dict(analysis)
        out["score"] = float(score)
        return out

    coarse_freqs = _normalize_candidate_frequencies(float(fc) * mul for mul in DIRECT_DAC_ALLPASS_FREQ_MULTIPLIERS)
    coarse_qs = _normalize_candidate_q_values(DIRECT_DAC_ALLPASS_Q_CANDIDATES)
    best_candidate: dict[str, Any] | None = None

    def _consider_candidates(freqs: tuple[float, ...], qs: tuple[float, ...], current_best: dict[str, Any] | None) -> dict[str, Any] | None:
        best = current_best
        best_score = _safe_float((best or {}).get("score", float("nan")), float("nan"))
        for freq_hz in freqs:
            freq_v = _safe_float(freq_hz, float("nan"))
            if (not np.isfinite(freq_v)) or freq_v <= (sub_hp + 1.0):
                continue
            for q in qs:
                cand = _evaluate(freq_v, q)
                if cand is None:
                    continue
                cand_score = _safe_float(cand.get("score", float("nan")), float("nan"))
                if (best is None) or (not np.isfinite(best_score)) or cand_score > best_score:
                    best = cand
                    best_score = cand_score
        return best

    best_candidate = _consider_candidates(coarse_freqs, coarse_qs, None)
    if best_candidate is None:
        return {
            "enabled": False,
            "freq_hz": 0.0,
            "q": 0.707,
            "baseline": baseline,
            "optimized": baseline,
            "improvement_score": 0.0,
            "reason": "No meaningful improvement found.",
        }

    refine_freqs = _normalize_candidate_frequencies(
        float(best_candidate.get("allpass_freq_hz", fc)) * factor
        for factor in DIRECT_DAC_ALLPASS_REFINE_FREQ_FACTORS
    )
    refine_qs = _normalize_candidate_q_values(
        float(
            np.clip(
                float(best_candidate.get("allpass_q", 1.0)) * factor,
                float(min(DIRECT_DAC_ALLPASS_Q_CANDIDATES)),
                float(max(DIRECT_DAC_ALLPASS_Q_CANDIDATES)),
            )
        )
        for factor in DIRECT_DAC_ALLPASS_REFINE_Q_FACTORS
    )
    best_candidate = _consider_candidates(refine_freqs, refine_qs, best_candidate)

    optimized = dict((best_candidate or {}).get("snapshot", {}) or baseline)
    best_score = _safe_float((best_candidate or {}).get("score", float("nan")), float("nan"))
    improvement_score = (
        float(best_score - baseline_score)
        if np.isfinite(best_score) and np.isfinite(baseline_score)
        else float("nan")
    )
    baseline_cancel = _safe_float(baseline.get("cancellation_risk", float("nan")), float("nan"))
    optimized_cancel = _safe_float(optimized.get("cancellation_risk", float("nan")), float("nan"))
    cancel_improvement = (
        float(baseline_cancel - optimized_cancel)
        if np.isfinite(baseline_cancel) and np.isfinite(optimized_cancel)
        else float("nan")
    )
    ripple_improvement = (
        float(
            _safe_float(baseline.get("overlap_ripple_db", float("nan")), float("nan"))
            - _safe_float(optimized.get("overlap_ripple_db", float("nan")), float("nan"))
        )
    )
    gd_improvement = (
        float(
            _safe_float(baseline.get("xo_gd_mismatch_ms", float("nan")), float("nan"))
            - _safe_float(optimized.get("xo_gd_mismatch_ms", float("nan")), float("nan"))
        )
    )

    enabled = bool(
        np.isfinite(improvement_score)
        and improvement_score >= float(DIRECT_DAC_ALLPASS_MIN_IMPROVEMENT_SCORE)
        and (
            (np.isfinite(cancel_improvement) and cancel_improvement >= float(DIRECT_DAC_ALLPASS_MIN_CANCEL_IMPROVEMENT))
            or (np.isfinite(ripple_improvement) and ripple_improvement >= float(DIRECT_DAC_ALLPASS_MIN_RIPPLE_IMPROVEMENT_DB))
            or (np.isfinite(gd_improvement) and gd_improvement >= float(DIRECT_DAC_ALLPASS_MIN_GD_IMPROVEMENT_MS))
        )
        and (
            (not np.isfinite(baseline_cancel))
            or (not np.isfinite(optimized_cancel))
            or optimized_cancel <= (baseline_cancel + 1e-6)
        )
    )
    return {
        "enabled": bool(enabled),
        "freq_hz": float(best_candidate.get("allpass_freq_hz", 0.0)) if enabled else 0.0,
        "q": float(best_candidate.get("allpass_q", 0.707)) if enabled else 0.707,
        "baseline": baseline,
        "optimized": optimized,
        "improvement_score": float(improvement_score) if np.isfinite(improvement_score) else 0.0,
        "reason": (
            "Applied shared mono-sub allpass."
            if enabled
            else "No meaningful improvement found."
        ),
    }
