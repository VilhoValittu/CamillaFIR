import numpy as np
import pytest

from camillafir.common.house_curves import _normalize_hc_mode_key, get_house_curve_by_name


def test_builtin_house_curve_lookup_from_common_module():
    freqs, mags = get_house_curve_by_name("Harman6")

    assert isinstance(freqs, np.ndarray)
    assert isinstance(mags, np.ndarray)
    assert freqs.shape == mags.shape
    assert freqs[0] == pytest.approx(0.0)
    assert mags[0] == pytest.approx(6.0)
    assert mags[-1] == pytest.approx(-6.0)


def test_normalized_house_curve_lookup_from_common_module():
    freqs, mags = get_house_curve_by_name(_normalize_hc_mode_key("B&K Strong"))

    assert freqs.shape == mags.shape
    assert mags[0] == pytest.approx(4.5)
    assert mags[-1] == pytest.approx(-6.0)
