from types import SimpleNamespace

import numpy as np

from camillafir.dsp.correction_mag import (
    _apply_mid_refit_pre_slope,
    _apply_bass_boost_post_restore,
    _apply_confidence_adaptive_bass_smoothing,
    _apply_hard_boost_cut_clamp,
    _apply_smoothing,
    _select_bass_adaptive_conf_mask,
)


def _mk_cfg(adaptive: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        bass_smooth_adaptive=bool(adaptive),
        bass_smooth_hz=200.0,
        bass_smooth_sigma_scale=1.4,
        bass_smooth_conf_floor=0.3,
    )


def _roughness(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=float)
    if y.size < 4:
        return 0.0
    return float(np.std(np.diff(y)))


def test_bass_adaptive_smoothing_reduces_low_conf_bass_roughness():
    freq_axis = np.geomspace(10.0, 2000.0, 1024).astype(float)
    lf = np.log2(freq_axis)
    err_db = (1.5 * np.sin(18.0 * lf) + 0.7 * np.sin(33.0 * lf)).astype(float)

    conf = np.ones_like(freq_axis, dtype=float)
    bass = (freq_axis >= 20.0) & (freq_axis <= 200.0)
    conf[bass] = 0.0

    out_base = _apply_smoothing(
        err_db=err_db,
        freq_axis=freq_axis,
        cfg=_mk_cfg(adaptive=False),
        st={},
        filter_smooth=12.0,
        df_mode=False,
        conf_mask=conf,
    )
    st_adaptive = {}
    out_adaptive = _apply_smoothing(
        err_db=err_db,
        freq_axis=freq_axis,
        cfg=_mk_cfg(adaptive=True),
        st=st_adaptive,
        filter_smooth=12.0,
        df_mode=False,
        conf_mask=conf,
    )

    assert _roughness(out_adaptive[bass]) < _roughness(out_base[bass])
    assert bool(st_adaptive.get("bass_adaptive_smoothing_enabled", False))
    assert float(st_adaptive.get("bass_adaptive_smoothing_avg_w_20_200", 0.0)) > 0.0
    assert float(st_adaptive.get("bass_adaptive_smoothing_avg_w_20_200", 1.0)) <= 0.55 + 1e-9


def test_bass_adaptive_smoothing_disables_cleanly_without_conf_mask():
    freq_axis = np.geomspace(10.0, 2000.0, 1024).astype(float)
    lf = np.log2(freq_axis)
    err_db = (1.0 * np.sin(20.0 * lf)).astype(float)

    out_base = _apply_smoothing(
        err_db=err_db,
        freq_axis=freq_axis,
        cfg=_mk_cfg(adaptive=False),
        st={},
        filter_smooth=12.0,
        df_mode=False,
        conf_mask=None,
    )
    st_adaptive = {}
    out_adaptive = _apply_smoothing(
        err_db=err_db,
        freq_axis=freq_axis,
        cfg=_mk_cfg(adaptive=True),
        st=st_adaptive,
        filter_smooth=12.0,
        df_mode=False,
        conf_mask=None,
    )

    assert np.allclose(out_adaptive, out_base, rtol=0.0, atol=1e-12)
    assert not bool(st_adaptive.get("bass_adaptive_smoothing_enabled", True))
    assert float(st_adaptive.get("bass_adaptive_smoothing_avg_w_20_200", -1.0)) == 0.0


def test_late_bass_adaptive_smoothing_produces_nonzero_delta_metrics():
    freq_axis = np.geomspace(10.0, 2000.0, 1024).astype(float)
    lf = np.log2(freq_axis)
    curve = (1.2 * np.sin(16.0 * lf) + 0.6 * np.sin(29.0 * lf)).astype(float)
    conf = np.ones_like(freq_axis, dtype=float)
    conf[(freq_axis >= 20.0) & (freq_axis <= 200.0)] = 0.0

    st = {}
    out = _apply_confidence_adaptive_bass_smoothing(
        curve_db=curve,
        freq_axis=freq_axis,
        cfg=_mk_cfg(adaptive=True),
        st=st,
        conf_mask=conf,
        stage_tag="core",
    )

    assert np.max(np.abs(out - curve)) > 0.0
    assert bool(st.get("bass_adaptive_smoothing_core_enabled", False))
    assert float(st.get("bass_adaptive_smoothing_core_delta_rms_db_20_200", 0.0)) > 0.0
    assert float(st.get("bass_adaptive_smoothing_core_delta_max_db_20_200", 0.0)) > 0.0


def test_bass_adaptive_conf_selection_prefers_raw_over_bassfirst():
    f = np.geomspace(10.0, 2000.0, 512).astype(float)
    raw = np.ones_like(f, dtype=float)
    raw[(f >= 20.0) & (f <= 200.0)] = 0.1
    # bassfirst fused mask would floor bass confidence high
    bf_fused = np.ones_like(f, dtype=float) * 0.75

    out, src = _select_bass_adaptive_conf_mask(
        conf_mask=raw,
        bf_conf_for_smoothing=bf_fused,
        use_bassfirst=True,
    )

    b = (f >= 20.0) & (f <= 200.0)
    assert src == "raw_conf"
    assert out is not None
    assert float(np.mean(out[b])) < 0.3


def test_hard_clamp_supports_per_bin_boost_cap():
    g = np.array([6.0, 6.0, 6.0, -20.0], dtype=float)
    cap = np.array([5.0, 6.0, 7.0, 5.0], dtype=float)
    cfg = SimpleNamespace(max_boost_db=5.0)

    out = _apply_hard_boost_cut_clamp(g, cfg, 15.0, boost_cap_db=cap, mask=np.ones_like(g, dtype=bool))

    assert np.allclose(out[:3], np.array([5.0, 6.0, 6.0], dtype=float), rtol=0.0, atol=1e-12)
    assert float(out[3]) == -15.0


def test_bass_boost_post_restore_moves_curve_toward_target_in_band():
    f = np.geomspace(10.0, 2000.0, 512).astype(float)
    m = (f >= 20.0) & (f <= 200.0)
    g = np.zeros_like(f, dtype=float)
    t = np.zeros_like(f, dtype=float)
    cap = np.zeros_like(f, dtype=float)
    # current curve under-corrects bass; target asks for more boost.
    g[m] = 1.0
    t[m] = 4.0
    cap[m] = 5.0

    out, meta = _apply_bass_boost_post_restore(
        g,
        t,
        cap,
        f,
        np.ones_like(f, dtype=bool),
        hz_lo=20.0,
        hz_hi=200.0,
        strength=0.5,
    )

    assert bool(meta.get("enabled", False))
    assert int(meta.get("bins", 0)) > 0
    assert float(meta.get("delta_rms_20_200", 0.0)) > 0.0
    assert float(meta.get("delta_max_20_200", 0.0)) > 0.0
    assert float(np.mean(out[m])) > float(np.mean(g[m]))


def test_mid_refit_reduces_mid_error_rms_when_conf_is_sufficient():
    f = np.geomspace(10.0, 5000.0, 1024).astype(float)
    m = np.zeros_like(f, dtype=float)
    t = np.zeros_like(f, dtype=float)
    g0 = np.zeros_like(f, dtype=float)
    mask = np.ones_like(f, dtype=bool)
    conf = np.ones_like(f, dtype=float) * 0.8
    mid = (f >= 200.0) & (f <= 2000.0)
    # Inject structured mid error for the refit step to correct.
    g0[mid] = 1.2 * np.sin(10.0 * np.log2(f[mid] / 200.0))
    cfg = SimpleNamespace(
        enable_mag_correction=True,
        mid_refit_enable=True,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        mid_refit_k=0.45,
        mid_refit_smooth_oct=0.6,
        mid_refit_conf_min_avg=0.2,
    )
    st = {}
    out = _apply_mid_refit_pre_slope(
        g0,
        f,
        mask,
        m_anal=m,
        target_mags=t,
        calc_offset_db=0.0,
        conf_mask=conf,
        cfg=cfg,
        st=st,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert float(st.get("mid_refit_err_rms_before", 0.0)) > float(st.get("mid_refit_err_rms_after", 0.0))
    assert float(st.get("mid_refit_delta_rms", 0.0)) > 0.0
    assert bool(st.get("mid_refit_enabled", False))
    assert str(st.get("mid_refit_reason", "")) == "applied"
    assert np.max(np.abs(out[mid] - g0[mid])) > 0.0


def test_mid_refit_skips_when_mid_confidence_is_too_low():
    f = np.geomspace(10.0, 5000.0, 1024).astype(float)
    m = np.zeros_like(f, dtype=float)
    t = np.zeros_like(f, dtype=float)
    g0 = np.ones_like(f, dtype=float) * 0.5
    mask = np.ones_like(f, dtype=bool)
    conf = np.ones_like(f, dtype=float)
    conf[(f >= 200.0) & (f <= 2000.0)] = 0.05
    cfg = SimpleNamespace(
        enable_mag_correction=True,
        mid_refit_enable=True,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        mid_refit_k=0.45,
        mid_refit_smooth_oct=0.6,
        mid_refit_conf_min_avg=0.2,
    )
    st = {}
    out = _apply_mid_refit_pre_slope(
        g0,
        f,
        mask,
        m_anal=m,
        target_mags=t,
        calc_offset_db=0.0,
        conf_mask=conf,
        cfg=cfg,
        st=st,
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert np.allclose(out, g0, rtol=0.0, atol=1e-12)
    assert not bool(st.get("mid_refit_enabled", True))
    assert str(st.get("mid_refit_reason", "")) == "low_mid_conf"
