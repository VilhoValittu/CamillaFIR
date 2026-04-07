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


def compute_direct_dac_bass_integration_diagnostics(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, Any]:
    trial_bundle = _build_direct_dac_trial_bundle(
        bundle,
        fc_hz=float(fc_hz),
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
    )
    out = compute_bass_integration_diagnostics(
        trial_bundle,
        float(fc_hz),
        profile,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
    )
    try:
        out["direct_dac_main_hpf_order"] = int(main_hpf_order)
        out["direct_dac_sub_lpf_order"] = int(sub_lpf_order)
        out["direct_dac_sub_hpf_hz"] = float(sub_hpf_hz)
        out["direct_dac_sub_hpf_order"] = int(sub_hpf_order)
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
