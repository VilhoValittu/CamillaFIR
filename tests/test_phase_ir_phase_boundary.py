import numpy as np

from camillafir.dsp.phase_ir_phase import (
    _enforce_linear_tail_decay,
    _linear_to_minphase_blend_mask,
    _linear_excess_weight,
    _smooth_linear_boundary,
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
