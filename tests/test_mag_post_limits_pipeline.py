from types import SimpleNamespace

import numpy as np

from camillafir.dsp.correction_types import _MagPostProcessInputs
from camillafir.dsp.dsp_utils import cfg_float_allow_zero
from camillafir.dsp.mag_post_limits import apply_post_limits_and_metrics


class _NullLogger:
    def info(self, *_args, **_kwargs):
        return None


def _identity_mid_refit(gain_db, *_args, **_kwargs):
    return np.asarray(gain_db, dtype=float)


def _stage_probe(*args, **_kwargs):
    return {"stage": str(args[0]) if args else "unknown"}


def _make_inputs(
    *,
    gain_apply=None,
    gain_db=None,
    pre_bass_adapt_g=None,
    st=None,
    cfg=None,
    n: int = 128,
):
    freq_axis = np.geomspace(10.0, 1000.0, n).astype(float)
    mask_c = np.ones_like(freq_axis, dtype=bool)
    gain_apply_arr = np.zeros_like(freq_axis) if gain_apply is None else np.asarray(gain_apply, dtype=float)
    gain_db_arr = np.zeros_like(freq_axis) if gain_db is None else np.asarray(gain_db, dtype=float)
    pre_bass = None if pre_bass_adapt_g is None else np.asarray(pre_bass_adapt_g, dtype=float)
    cfg_obj = cfg or SimpleNamespace(
        max_boost_db=2.0,
        max_cut_db=15.0,
        low_bass_cut_enable=True,
        low_bass_cut_hz=40.0,
        low_bass_cut_strength=1.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=False,
        bass_boost_post_restore_enable=False,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    stats = {} if st is None else st
    return (
        freq_axis,
        stats,
        _MagPostProcessInputs(
            cfg=cfg_obj,
            freq_axis=freq_axis,
            st=stats,
            logger=_NullLogger(),
            stage_probe=_stage_probe,
            cfg_float_allow_zero=cfg_float_allow_zero,
            mask_c=mask_c,
            gain_db=gain_db_arr.copy(),
            gain_apply=gain_apply_arr.copy(),
            raw_g=np.asarray(gain_apply_arr, dtype=float).copy(),
            final_g=np.asarray(gain_apply_arr, dtype=float).copy(),
            pre_bass_adapt_g=pre_bass,
            raw_safe_ref=np.zeros_like(freq_axis),
            conf_mask=np.ones_like(freq_axis),
            filter_smooth=12.0,
            debug_stage_stats=False,
            stage_probes={},
            apply_confidence_weighted_target_pull=lambda **kwargs: (
                np.asarray(kwargs["target_db"], dtype=float),
                {},
            ),
            m_anal=np.zeros_like(freq_axis),
            target_mags=np.zeros_like(freq_axis),
            calc_offset_db=0.0,
        ),
    )


def test_apply_post_limits_and_metrics_preserves_array_shape():
    freq_axis, st, inputs = _make_inputs(gain_apply=np.linspace(-1.0, 1.0, 128))

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert out.gain_db.shape == freq_axis.shape
    assert out.stage_probes
    assert isinstance(st, dict)


def test_low_frequency_guard_blocks_forbidden_boost_cases():
    freq_axis = np.geomspace(10.0, 1000.0, 128)
    gain_apply = np.where(freq_axis <= 40.0, 6.0, 0.5)
    _, _, inputs = _make_inputs(gain_apply=gain_apply)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)
    lf_mask = freq_axis <= 40.0

    assert np.any(lf_mask)
    assert float(np.max(out.gain_db[lf_mask])) <= 1e-9


def test_realized_metric_fields_are_emitted_when_expected():
    st = {
        "mid_refit_err_rms_before_stage_local": 1.5,
        "bass_adaptive_smoothing_delta_rms_db_20_200_stage_local": 0.25,
        "bass_adaptive_smoothing_delta_max_db_20_200_stage_local": 0.5,
    }
    gain_apply = np.linspace(0.0, 1.0, 128)
    pre_bass = np.zeros(128, dtype=float)
    _, _, inputs = _make_inputs(gain_apply=gain_apply, pre_bass_adapt_g=pre_bass, st=st)

    _ = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert "bass_adaptive_smoothing_delta_rms_db_20_200_realized_pre_ir" in st
    assert "bass_adaptive_smoothing_delta_max_db_20_200_realized_pre_ir" in st
    assert "mid_refit_err_rms_after_realized_pre_ir" in st


def test_minimal_synthetic_inputs_do_not_crash():
    _, _, inputs = _make_inputs(gain_apply=np.zeros(24), gain_db=np.zeros(24), n=24)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert out is not None
    assert out.gain_db.shape == (24,)


def test_post_limit_stages_do_not_increase_boost_after_clamp_barrier():
    gain_apply = np.full(128, 8.0, dtype=float)
    cfg = SimpleNamespace(
        max_boost_db=1.0,
        max_cut_db=15.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=False,
        bass_boost_post_restore_enable=False,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    _, _, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert float(np.max(out.gain_db)) <= 1.0 + 1e-6
