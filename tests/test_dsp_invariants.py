from types import SimpleNamespace

import numpy as np

from camillafir.dsp.camillafir_dsp import _limit_gd_gradient_ms_per_oct, apply_confidence_weighted_target_pull
from camillafir.dsp.phase_ir_phase_gradient import max_abs_gd_gradient_ms_per_oct


def test_confidence_weighted_target_pull_stays_between_target_and_measured():
    target = np.asarray([0.0, -4.0, 2.0, 6.0, -1.0], dtype=float)
    measured = np.asarray([-2.0, 0.0, 1.0, 3.0, -6.0], dtype=float)
    conf = np.asarray([0.1, 0.4, 0.7, 0.9, 0.2], dtype=float)
    freq = np.asarray([20.0, 40.0, 80.0, 160.0, 320.0], dtype=float)

    out = np.asarray(
        apply_confidence_weighted_target_pull(
            target,
            measured,
            conf,
            conf_floor=0.05,
            conf_ceil=0.95,
            freq_axis=freq,
            freq_limit_hz=500.0,
            gamma_cut=0.7,
            gamma_boost=1.2,
        ),
        dtype=float,
    )

    lo = np.minimum(target, measured)
    hi = np.maximum(target, measured)
    assert np.all(out >= lo - 1e-10)
    assert np.all(out <= hi + 1e-10)


def test_gd_gradient_limiter_never_worsens_peak_gradient_metric():
    freq = np.geomspace(20.0, 250.0, 256)
    phase = 2.5 * np.sin(np.linspace(0.0, 24.0, freq.size, dtype=float))
    mask = np.ones_like(freq, dtype=bool)

    before, _ = max_abs_gd_gradient_ms_per_oct(freq, phase, mask=mask)
    limited = np.asarray(
        _limit_gd_gradient_ms_per_oct(
            freq,
            phase,
            mask=mask,
            max_grad_ms_per_oct=18.0,
            f_min=20.0,
            f_max=250.0,
            grad_smooth_sigma=0.8,
            soft_limit=True,
        ),
        dtype=float,
    )
    after, _ = max_abs_gd_gradient_ms_per_oct(freq, limited, mask=mask)

    assert float(after) <= float(before) + 1e-9
    assert limited.shape == phase.shape
