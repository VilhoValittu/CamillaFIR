import numpy as np

from src.camillafir.dsp.dsp_phase_ir import _compute_excess_phase


def test_compute_excess_phase_is_invariant_to_2pi_branch_offset():
    raw = np.linspace(-1.0, 1.0, 32, dtype=float)
    ref = raw + (2.0 * np.pi)

    exc = _compute_excess_phase(raw, ref)

    assert np.allclose(exc, 0.0, atol=1e-12, rtol=0.0)


def test_compute_excess_phase_uses_principal_angle_for_wrapped_inputs():
    raw = np.deg2rad(np.array([179.0, -179.0, 178.0, -178.0], dtype=float))
    ref = np.deg2rad(np.array([170.0, 170.0, 170.0, 170.0], dtype=float))

    exc_deg = np.rad2deg(_compute_excess_phase(raw, ref))

    assert np.allclose(exc_deg, np.array([9.0, 11.0, 8.0, 12.0], dtype=float), atol=1e-9, rtol=0.0)
