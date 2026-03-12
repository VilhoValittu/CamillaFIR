import numpy as np

from camillafir.io.camillafir_automatic_mode import (
    _estimate_auto_hpf_from_response,
    _resolve_auto_hpf_application,
)


def _butter_hpf_mag_db(freqs: np.ndarray, fc_hz: float, order: int) -> np.ndarray:
    f = np.asarray(freqs, dtype=float)
    with np.errstate(divide="ignore"):
        return -10.0 * np.log10(1.0 + np.power(float(fc_hz) / np.maximum(f, 1e-9), 2 * int(order)))


def test_auto_hpf_estimator_recovers_butterworth_like_rolloff():
    rng = np.random.default_rng(1234)
    f = np.geomspace(10.0, 20000.0, 4096).astype(float)

    fc_true = 38.0
    order_true = 4  # 24 dB/oct
    hpf = _butter_hpf_mag_db(f, fc_true, order_true)

    # Add mild room-like structure while keeping LF roll-off dominant.
    room = 0.35 * np.sin(np.log(f) * 1.9) * np.exp(-0.5 * (np.log2(np.maximum(f, 1.0) / 120.0) / 1.8) ** 2)
    n_l = rng.normal(0.0, 0.06, size=f.size)
    n_r = rng.normal(0.0, 0.06, size=f.size)
    m_l = hpf + room + n_l
    m_r = _butter_hpf_mag_db(f, fc_true * 1.03, order_true) + room * 0.92 + n_r

    res = _estimate_auto_hpf_from_response(
        f,
        m_l,
        f,
        m_r,
        default_freq_hz=20.0,
        default_slope_db_oct=24,
    )

    assert isinstance(res, dict)
    assert str(res.get("method")) == "response_fit"
    assert bool(res.get("enabled", False))
    assert abs(float(res.get("freq", 0.0)) - fc_true) <= 6.0
    assert int(res.get("slope_db_oct", 0)) == 24
    assert float(res.get("confidence", 0.0)) >= 0.70


def test_auto_hpf_estimator_stays_off_for_flat_response():
    f = np.geomspace(10.0, 20000.0, 2048).astype(float)
    m = np.zeros_like(f, dtype=float)

    res = _estimate_auto_hpf_from_response(
        f,
        m,
        f,
        m,
        default_freq_hz=23.0,
        default_slope_db_oct=24,
    )

    assert isinstance(res, dict)
    assert not bool(res.get("enabled", True))
    assert str(res.get("method")) == "fallback_default"
    assert abs(float(res.get("freq", 0.0)) - 23.0) <= 0.1
    assert int(res.get("slope_db_oct", 0)) == 24
    assert float(res.get("confidence", 1.0)) <= 0.1


def test_auto_hpf_is_only_a_suggestion_when_user_has_not_enabled_hpf():
    auto_hpf = {
        "enabled": True,
        "freq": 18.2,
        "slope_db_oct": 24,
        "confidence": 0.81,
        "method": "response_fit",
    }

    resolved = _resolve_auto_hpf_application(auto_hpf, user_hpf_enabled=False)

    assert bool(resolved.get("enabled", False))
    assert not bool(resolved.get("applied", True))
    assert str(resolved.get("decision")) == "suggest_only"


def test_auto_hpf_applies_response_fit_when_user_hpf_is_enabled():
    auto_hpf = {
        "enabled": False,
        "freq": 22.0,
        "slope_db_oct": 18,
        "confidence": 0.52,
        "method": "response_fit",
    }

    resolved = _resolve_auto_hpf_application(auto_hpf, user_hpf_enabled=True)

    assert bool(resolved.get("applied", False))
    assert str(resolved.get("decision")) == "apply_user_hpf_fit"
