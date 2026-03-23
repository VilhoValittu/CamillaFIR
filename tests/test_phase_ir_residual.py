from types import SimpleNamespace

import numpy as np

from camillafir.dsp.phase_ir_residual import apply_residual_pass_if_enabled


class _NullLogger:
    def info(self, _msg: str) -> None:
        return None


def _cfg_float_allow_zero(cfg, name: str, default: float) -> float:
    try:
        v = getattr(cfg, name, default)
        if v is None:
            return float(default)
        if isinstance(v, str) and v.strip() == "":
            return float(default)
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def _run_residual(
    cfg: SimpleNamespace,
    gain_db: np.ndarray,
    target_mags: np.ndarray,
    freq_axis: np.ndarray,
) -> tuple[np.ndarray, object | None]:
    mask_c = np.ones_like(freq_axis, dtype=bool)
    out, telemetry = apply_residual_pass_if_enabled(
        cfg=cfg,
        freq_axis=np.asarray(freq_axis, dtype=float),
        gain_db=np.asarray(gain_db, dtype=float),
        conf_mask=None,
        m_anal=np.zeros_like(freq_axis, dtype=float),
        calc_offset_db=0.0,
        target_mags=np.asarray(target_mags, dtype=float),
        st={},
        mask_c=mask_c,
        base_sigma=6.0,
        filter_smooth=12.0,
        df_mode=False,
        raw_g=np.zeros_like(freq_axis, dtype=float),
        final_g=np.zeros_like(freq_axis, dtype=float),
        logger=_NullLogger(),
        cfg_float_allow_zero_fn=_cfg_float_allow_zero,
    )
    return np.asarray(out, dtype=float), telemetry


def test_residual_is_band_limited_with_smoothstep_edges():
    freq_axis = np.geomspace(10.0, 220.0, 256)
    gain_db = np.zeros_like(freq_axis, dtype=float)
    target = np.full_like(freq_axis, 6.0, dtype=float)
    cfg = SimpleNamespace(
        enable_residual_pass=True,
        enable_mag_correction=True,
        residual_strength=1.0,
        residual_smoothing_mult=1.0,
        residual_conf_power=1.0,
        residual_max_delta_db=6.0,
        mag_c_min=40.0,
        mag_c_max=140.0,
        trans_width=20.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        exc_prot=False,
        exc_freq=0.0,
        max_boost_db=12.0,
        max_cut_db=15.0,
    )

    out, telemetry = _run_residual(cfg, gain_db, target, freq_axis)

    assert np.allclose(out[freq_axis < 40.0], 0.0, atol=1e-10, rtol=0.0)
    assert np.allclose(out[freq_axis > 140.0], 0.0, atol=1e-10, rtol=0.0)
    assert telemetry is not None
    assert bool(getattr(telemetry, "residual_pass_enabled")) is True

    left_edge = out[int(np.argmin(np.abs(freq_axis - 44.0)))]
    center = out[int(np.argmin(np.abs(freq_axis - 90.0)))]
    right_edge = out[int(np.argmin(np.abs(freq_axis - 136.0)))]
    assert float(center) > float(left_edge)
    assert float(center) > float(right_edge)


def test_residual_cannot_increase_lf_boost_below_low_bass_cut():
    freq_axis = np.geomspace(10.0, 220.0, 256)
    gain_db = np.where(freq_axis <= 40.0, -2.0, 0.0).astype(float)
    target = np.full_like(freq_axis, 6.0, dtype=float)
    cfg = SimpleNamespace(
        enable_residual_pass=True,
        enable_mag_correction=True,
        residual_strength=1.0,
        residual_smoothing_mult=1.0,
        residual_conf_power=1.0,
        residual_max_delta_db=6.0,
        mag_c_min=20.0,
        mag_c_max=180.0,
        trans_width=20.0,
        low_bass_cut_enable=True,
        low_bass_cut_hz=40.0,
        exc_prot=False,
        exc_freq=0.0,
        max_boost_db=12.0,
        max_cut_db=15.0,
    )

    out, telemetry = _run_residual(cfg, gain_db, target, freq_axis)
    m = freq_axis <= 40.0
    assert np.any(m)
    assert np.all(out[m] <= gain_db[m] + 1e-10)
    assert telemetry is not None
    assert int(getattr(telemetry, "residual_lf_guard_bins")) > 0


def test_residual_cannot_increase_lf_boost_near_exc_protection():
    freq_axis = np.geomspace(10.0, 220.0, 256)
    gain_db = np.zeros_like(freq_axis, dtype=float)
    target = np.full_like(freq_axis, 6.0, dtype=float)
    cfg = SimpleNamespace(
        enable_residual_pass=True,
        enable_mag_correction=True,
        residual_strength=1.0,
        residual_smoothing_mult=1.0,
        residual_conf_power=1.0,
        residual_max_delta_db=6.0,
        mag_c_min=20.0,
        mag_c_max=180.0,
        trans_width=20.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        exc_prot=True,
        exc_freq=35.0,
        max_boost_db=12.0,
        max_cut_db=15.0,
    )

    out, telemetry = _run_residual(cfg, gain_db, target, freq_axis)
    guard_hz = 35.0 * 1.41
    m = freq_axis <= guard_hz
    assert np.any(m)
    assert np.all(out[m] <= gain_db[m] + 1e-10)
    assert telemetry is not None
    assert int(getattr(telemetry, "residual_lf_boost_blocked_bins")) >= 0
