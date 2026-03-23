from types import SimpleNamespace

import numpy as np

from camillafir.dsp.gain_policy import (
    apply_cuts_only_guard,
    build_low_frequency_guard_mask,
    clamp_gain_curve,
    resolve_gain_policy,
)


def _cfg_float_allow_zero(cfg, name: str, default: float) -> float:
    try:
        v = getattr(cfg, name, default)
        if v is None:
            return float(default)
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def test_shared_gain_policy_combines_low_bass_and_exc_guards():
    cfg = SimpleNamespace(
        max_boost_db=6.0,
        max_cut_db=15.0,
        low_bass_cut_enable=True,
        low_bass_cut_hz=40.0,
        low_bass_cut_strength=0.75,
        exc_prot=True,
        exc_freq=35.0,
    )
    freq_axis = np.asarray([10.0, 20.0, 35.0, 45.0, 50.0, 60.0], dtype=float)

    policy = resolve_gain_policy(cfg, cfg_float_allow_zero_fn=_cfg_float_allow_zero)
    guard_mask = build_low_frequency_guard_mask(freq_axis, policy)

    assert bool(policy.low_cut_enable) is True
    assert float(policy.low_cut_hz) == 40.0
    assert float(policy.exc_soft_hz) == 35.0 * 1.41
    assert np.array_equal(guard_mask, np.asarray([True, True, True, True, False, False]))


def test_shared_gain_policy_reapplies_cuts_only_floor_and_clamp():
    cfg = SimpleNamespace(
        max_boost_db=3.0,
        max_cut_db=12.0,
        low_bass_cut_enable=True,
        low_bass_cut_hz=40.0,
        low_bass_cut_strength=1.0,
        exc_prot=False,
        exc_freq=0.0,
    )
    freq_axis = np.asarray([20.0, 30.0, 50.0], dtype=float)
    mask = np.asarray([True, True, True], dtype=bool)
    curve = np.asarray([2.0, -1.0, 8.0], dtype=float)
    floor_ref = np.asarray([-2.5, -1.5, np.nan], dtype=float)

    policy = resolve_gain_policy(cfg, cfg_float_allow_zero_fn=_cfg_float_allow_zero)
    low_guard = build_low_frequency_guard_mask(freq_axis, policy, include_exc_soft=False)
    guarded, meta = apply_cuts_only_guard(curve, mask=mask, guard_mask=low_guard, floor_ref=floor_ref)
    clamped = clamp_gain_curve(guarded, policy=policy, mask=mask)

    assert np.allclose(guarded[:2], [-2.5, -1.5], atol=1e-10, rtol=0.0)
    assert int(meta["boost_clamped_bins"]) == 1
    assert int(meta["floor_reapplied_bins"]) == 2
    assert np.allclose(clamped, [-2.5, -1.5, 3.0], atol=1e-10, rtol=0.0)
