import numpy as np
import pytest

from camillafir.dsp.phase_ir_contracts import (
    arrays_equal_strict,
    require_allowed_keys,
    require_scalar_unchanged,
    require_unchanged,
)


def test_arrays_equal_strict_accepts_exact_match():
    lhs = np.array([1.0, 2.0, np.nan], dtype=float)
    rhs = np.array([1.0, 2.0, np.nan], dtype=float)

    assert arrays_equal_strict(lhs, rhs) is True


def test_require_unchanged_accepts_identical_arrays():
    arr = np.array([0.0, 1.0, 2.0], dtype=float)

    require_unchanged("phase_ir_test", "gain_db", arr, arr.copy())


def test_require_unchanged_raises_on_changed_arrays():
    before = np.array([0.0, 1.0, 2.0], dtype=float)
    after = np.array([0.0, 1.1, 2.0], dtype=float)

    with pytest.raises(RuntimeError, match="contract breach"):
        require_unchanged("phase_ir_test", "gain_db", before, after)


def test_require_scalar_unchanged_raises_on_changed_scalar():
    with pytest.raises(RuntimeError, match="scalar 'auto_gain' changed"):
        require_scalar_unchanged("phase_ir_test", "auto_gain", 0.0, 1.0)


def test_require_allowed_keys_rejects_unexpected_keys():
    with pytest.raises(RuntimeError, match="unexpected output keys"):
        require_allowed_keys("phase_ir_test", {"ok": 1, "extra": 2}, frozenset({"ok"}))
