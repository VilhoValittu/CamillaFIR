import numpy as np

from camillafir.ui.target_preview_common import (
    apply_manual_target_preview_adjustments,
    apply_manual_target_preview_shift,
)


def test_apply_manual_target_preview_shift_moves_target_curve_only():
    target_curve = np.array([-1.0, 0.0, 1.5], dtype=float)

    shifted = apply_manual_target_preview_shift(target_curve, 2.5)

    np.testing.assert_allclose(shifted, np.array([1.5, 2.5, 4.0], dtype=float))
    np.testing.assert_allclose(target_curve, np.array([-1.0, 0.0, 1.5], dtype=float))


def test_apply_manual_target_preview_shift_ignores_zero_and_invalid_shift():
    target_curve = np.array([0.5, -0.5], dtype=float)

    zero_shift = apply_manual_target_preview_shift(target_curve, 0.0)
    invalid_shift = apply_manual_target_preview_shift(target_curve, "not-a-number")

    np.testing.assert_allclose(zero_shift, target_curve)
    np.testing.assert_allclose(invalid_shift, target_curve)


def test_apply_manual_target_preview_adjustments_apply_tilt_around_1khz_then_shift():
    freq_axis = np.array([125.0, 1000.0, 8000.0], dtype=float)
    target_curve = np.array([0.0, 0.0, 0.0], dtype=float)

    adjusted = apply_manual_target_preview_adjustments(
        freq_axis,
        target_curve,
        1.0,
        0.5,
    )

    np.testing.assert_allclose(
        adjusted,
        np.array([2.5, 1.0, -0.5], dtype=float),
    )
