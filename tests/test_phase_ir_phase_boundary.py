import numpy as np
import pytest

from camillafir.config.models import FilterConfig
from camillafir.dsp.phase import calculate_minimum_phase, calculate_theoretical_phase
from camillafir.dsp.phase_ir_phase import (
    _PhaseComponents,
    _apply_phase_model,
    _compute_excess_phase,
    _enforce_linear_tail_decay,
    _linear_to_minphase_blend_mask,
    _linear_excess_weight,
    _smooth_linear_boundary,
    _unwrap_phases,
)


def test_linear_excess_weight_does_not_force_zero_at_400_when_limit_is_higher():
    f = np.geomspace(20.0, 1000.0, 1024)
    w = _linear_excess_weight(f, 600.0)
    i400 = int(np.argmin(np.abs(f - 400.0)))
    i500 = int(np.argmin(np.abs(f - 500.0)))
    i600 = int(np.argmin(np.abs(f - 600.0)))

    assert float(w[i400]) > 0.0
    assert float(w[i500]) > 0.0
    assert float(w[i600]) < 1e-3


def test_smooth_linear_boundary_reduces_local_spike_near_limit():
    f = np.geomspace(20.0, 1000.0, 2048)
    x = np.zeros_like(f)
    i = int(np.argmin(np.abs(f - 430.0)))
    x[i] = 1.0

    class _Cfg:
        phase_boundary_smooth_sigma_bins = 2.0

    out = _smooth_linear_boundary(f, x, 500.0, _Cfg(), st={})

    assert float(out[i]) < float(x[i])
    # Low-frequency area should stay effectively unchanged.
    i_lf = int(np.argmin(np.abs(f - 80.0)))
    assert float(abs(out[i_lf] - x[i_lf])) < 1e-9


def test_enforce_linear_tail_decay_makes_tail_nonincreasing():
    f = np.geomspace(20.0, 1000.0, 2048)
    x = np.zeros_like(f)
    sel = (f >= 330.0) & (f <= 500.0)
    idx = np.flatnonzero(sel)
    # Craft a tail with a local bump around ~410 Hz.
    t = np.linspace(0.0, 1.0, idx.size, endpoint=True)
    tail = 0.08 * (1.0 - t)
    bump_i = int(0.45 * (idx.size - 1))
    tail[bump_i:bump_i + 4] += 0.04
    x[idx] = tail

    class _Cfg:
        phase_tail_monotonic_enable = True
        phase_tail_abs_smooth_sigma_bins = 1.0

    out = _enforce_linear_tail_decay(f, x, 500.0, _Cfg(), st={})
    out_tail = np.abs(out[idx])
    dif = np.diff(out_tail)
    assert np.all(dif <= 1e-9)


def test_enforce_linear_tail_decay_locks_single_sign_branch():
    f = np.geomspace(20.0, 1000.0, 2048)
    x = np.zeros_like(f)
    sel = (f >= 350.0) & (f <= 500.0)
    idx = np.flatnonzero(sel)
    t = np.linspace(0.0, 1.0, idx.size, endpoint=True)
    # Create a flip in sign near the boundary area.
    x[idx] = 0.04 * (1.0 - t)
    flip_from = int(0.6 * idx.size)
    x[idx[flip_from:]] *= -1.0

    class _Cfg:
        phase_tail_monotonic_enable = True
        phase_tail_abs_smooth_sigma_bins = 1.0
        phase_tail_cosine_strength = 1.0

    out = _enforce_linear_tail_decay(f, x, 500.0, _Cfg(), st={})
    out_tail = out[idx]
    non_zero = np.abs(out_tail) > 1e-12
    signs = np.sign(out_tail[non_zero])
    assert signs.size > 0
    assert np.all(signs == signs[0])


def test_linear_to_minphase_blend_mask_reaches_one_at_phase_limit():
    f = np.geomspace(20.0, 2000.0, 2048)

    class _Cfg:
        linear_phase_blend_start_ratio = 0.65

    m = _linear_to_minphase_blend_mask(f, 500.0, _Cfg(), st={})
    i450 = int(np.argmin(np.abs(f - 450.0)))
    i500 = int(np.argmin(np.abs(f - 500.0)))
    i700 = int(np.argmin(np.abs(f - 700.0)))

    assert float(m[i450]) < 1.0
    assert float(m[i500]) == 1.0
    assert float(m[i700]) == 1.0


class _SilentLogger:
    def info(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


def _identity_phase_limiter(freq_axis, phase_rad, **kwargs):
    return np.asarray(phase_rad, dtype=float)


def _render_final_phase(
    *,
    filter_type: str,
    phase_limit: float,
    crossovers: list[dict] | None = None,
    hpf_settings: dict | None = None,
    total_mag: np.ndarray | None = None,
    p_rad_interp: np.ndarray | None = None,
):
    fs = 48000
    n_fft = 32768
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    theo_xo = calculate_theoretical_phase(
        freq_axis,
        list(crossovers or []),
        hpf_freq=float((hpf_settings or {}).get("freq", 0.0) or 0.0) if bool((hpf_settings or {}).get("enabled", False)) else None,
        hpf_slope=float(int((hpf_settings or {}).get("order", 0) or 0) * 6) if bool((hpf_settings or {}).get("enabled", False)) else None,
        max_phase_deg=None,
    )
    total_mag_arr = np.asarray(total_mag if total_mag is not None else np.ones_like(freq_axis), dtype=float)
    p_used = np.asarray(p_rad_interp if p_rad_interp is not None else theo_xo, dtype=float)
    min_p = calculate_minimum_phase(total_mag_arr, max_phase_deg=None)
    raw_u, ref_u = _unwrap_phases(p_used, theo_xo)
    excess_u = _compute_excess_phase(raw_u, ref_u)
    cfg = FilterConfig(
        fs=fs,
        num_taps=n_fft,
        filter_type_str=filter_type,
        phase_limit=float(phase_limit),
        crossovers=list(crossovers or []),
        hpf_settings=dict(hpf_settings) if isinstance(hpf_settings, dict) else None,
    )
    final_phase = _apply_phase_model(
        freq_axis,
        cfg,
        {},
        _PhaseComponents(
            raw_u=raw_u,
            ref_u=ref_u,
            excess_u=excess_u,
            min_phase=min_p,
            theo_xo=theo_xo,
            conf_mask=np.ones_like(freq_axis, dtype=float),
            total_mag=total_mag_arr,
            n_fft=n_fft,
            is_mixed=False,
            mixed_split_hz=180.0,
            mixed_transition_hz=100.0,
            use_bassfirst=False,
            afdw_on=False,
            logger=_SilentLogger(),
            limit_gd_gradient_ms_per_oct_fn=_identity_phase_limiter,
        ),
    )
    return freq_axis, np.asarray(final_phase, dtype=float), np.asarray(theo_xo, dtype=float), np.asarray(min_p, dtype=float)


@pytest.mark.parametrize("filter_type", ["Asymmetric", "Linear Phase"])
def test_phase_limit_preserves_xo_baseline_for_asymmetric_and_linear(filter_type: str):
    crossovers = [
        {"freq": 500.0, "order": 4, "slope": 24, "idx": 1},
        {"freq": 2800.0, "order": 4, "slope": 24, "idx": 2},
    ]
    freq_axis, final_phase, theo_xo, _min_p = _render_final_phase(
        filter_type=filter_type,
        phase_limit=400.0,
        crossovers=crossovers,
    )

    for probe_hz in (500.0, 2800.0):
        idx = int(np.argmin(np.abs(freq_axis - probe_hz)))
        assert np.isclose(float(final_phase[idx]), float(-theo_xo[idx]), atol=1e-6)


@pytest.mark.parametrize("filter_type", ["Asymmetric", "Linear Phase"])
def test_phase_limit_without_theoretical_model_still_blends_to_minphase(filter_type: str):
    fs = 48000
    n_fft = 32768
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(fs))
    mag_db = np.where(freq_axis < 1800.0, 0.0, -9.0)
    total_mag = 10.0 ** (mag_db / 20.0)
    _, final_phase, _theo_xo, min_p = _render_final_phase(
        filter_type=filter_type,
        phase_limit=400.0,
        total_mag=total_mag,
        p_rad_interp=np.zeros_like(freq_axis, dtype=float),
    )

    idx = int(np.argmin(np.abs(freq_axis - 4000.0)))
    assert np.isclose(float(final_phase[idx]), float(min_p[idx]), atol=1e-6)
