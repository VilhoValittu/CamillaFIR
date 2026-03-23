from __future__ import annotations

import numpy as np

from .phase_ir_autogain import compute_auto_gain_and_headroom
from .phase_ir_build import build_phase_and_ir
from .phase_ir_phase import _compute_excess_phase
from .phase_ir_residual import apply_residual_pass_if_enabled
from .phase_ir_theoretical import compute_theoretical_phase_and_store_stats
from .phase_ir_types import (
    PhaseIRInputs,
    PhaseIROutputs,
    apply_residual_telemetry_to_stats,
)

__all__ = ("run_phase_ir_stage", "_compute_excess_phase")

_THEORETICAL_ALLOWED_KEYS = frozenset(
    {
        "theo_xo",
        "theo_on_raw",
        "theo_off_raw",
        "xo_txt",
        "hpf_txt",
        "hpf_freq",
        "hpf_slope",
        "p_rad_interp",
    }
)

_AUTOGAIN_ALLOWED_KEYS = frozenset(
    {
        "current_peak_gain",
        "gain_margin_db",
        "auto_global_gain_db",
        "auto_headroom_db",
        "final_gain_total",
    }
)

_BUILD_ALLOWED_KEYS = frozenset(
    {
        "impulse",
        "total_mag",
        "min_phase",
        "final_phase",
        "mixed_split_hz",
        "mixed_transition_hz",
    }
)


def _arrays_equal_strict(lhs, rhs) -> bool:
    a = np.asarray(lhs)
    b = np.asarray(rhs)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in ("f", "c") or b.dtype.kind in ("f", "c"):
        return bool(np.allclose(a, b, atol=0.0, rtol=0.0, equal_nan=True))
    return bool(np.array_equal(a, b))


def _require_unchanged(stage: str, value_name: str, before, after) -> None:
    if _arrays_equal_strict(before, after):
        return
    raise RuntimeError(
        f"Phase-IR contract breach in {stage}: '{value_name}' was modified, "
        "but this stage is not allowed to mutate it."
    )


def _require_scalar_unchanged(stage: str, value_name: str, before, after) -> None:
    if bool(np.isclose(float(before), float(after), atol=0.0, rtol=0.0, equal_nan=True)):
        return
    raise RuntimeError(
        f"Phase-IR contract breach in {stage}: scalar '{value_name}' changed "
        "but must remain immutable in this stage."
    )


def _require_allowed_keys(stage: str, obj, allowed: frozenset[str]) -> None:
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Phase-IR contract breach in {stage}: expected dict output, got {type(obj).__name__}."
        )
    extra = set(obj.keys()) - set(allowed)
    if extra:
        keys_txt = ", ".join(sorted(str(k) for k in extra))
        raise RuntimeError(
            f"Phase-IR contract breach in {stage}: unexpected output keys: {keys_txt}."
        )


def run_phase_ir_stage(
    *,
    cfg,
    freq_axis,
    n_fft,
    gain_db,
    p_rad_interp,
    conf_mask,
    m_anal,
    calc_offset_db,
    target_mags,
    st,
    mask_c,
    base_sigma,
    _filter_smooth,
    df_mode,
    raw_g,
    final_g,
    use_bassfirst,
    afdw_on,
    logger,
    apply_hpf_to_mags_fn,
    limit_gd_gradient_ms_per_oct_fn,
    cfg_float_allow_zero_fn,
):
    inputs = PhaseIRInputs(
        cfg=cfg,
        freq_axis=freq_axis,
        n_fft=n_fft,
        gain_db=gain_db,
        p_rad_interp=p_rad_interp,
        conf_mask=conf_mask,
        m_anal=m_anal,
        calc_offset_db=calc_offset_db,
        target_mags=target_mags,
        st=st,
        mask_c=mask_c,
        base_sigma=base_sigma,
        filter_smooth=_filter_smooth,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        use_bassfirst=use_bassfirst,
        afdw_on=afdw_on,
        logger=logger,
        apply_hpf_to_mags_fn=apply_hpf_to_mags_fn,
        limit_gd_gradient_ms_per_oct_fn=limit_gd_gradient_ms_per_oct_fn,
        cfg_float_allow_zero_fn=cfg_float_allow_zero_fn,
    )
    outputs = _run_phase_ir_stage(inputs)
    apply_residual_telemetry_to_stats(st=inputs.st, telemetry=outputs.residual_telemetry)
    return outputs.to_legacy_dict()


def _run_phase_ir_stage(inputs: PhaseIRInputs) -> PhaseIROutputs:
    cfg = inputs.cfg
    freq_axis = np.asarray(inputs.freq_axis, dtype=float)
    n_fft = int(inputs.n_fft)
    gain_db = np.asarray(inputs.gain_db, dtype=float).copy()
    p_rad_interp = np.asarray(inputs.p_rad_interp, dtype=float).copy()
    conf_mask = inputs.conf_mask
    m_anal = np.asarray(inputs.m_anal, dtype=float)
    calc_offset_db = float(inputs.calc_offset_db)
    target_mags = np.asarray(inputs.target_mags, dtype=float)
    st = inputs.st
    mask_c = np.asarray(inputs.mask_c, dtype=bool)
    base_sigma = inputs.base_sigma
    _filter_smooth = inputs.filter_smooth
    df_mode = inputs.df_mode
    raw_g = inputs.raw_g
    final_g = inputs.final_g
    use_bassfirst = bool(inputs.use_bassfirst)
    afdw_on = bool(inputs.afdw_on)
    logger = inputs.logger
    apply_hpf_to_mags = inputs.apply_hpf_to_mags_fn
    _limit_gd_gradient_ms_per_oct = inputs.limit_gd_gradient_ms_per_oct_fn
    _cfg_float_allow_zero = inputs.cfg_float_allow_zero_fn

    gain_db_before_theoretical = gain_db.copy()
    p_rad_input_for_theoretical = p_rad_interp.copy()
    theo = compute_theoretical_phase_and_store_stats(
        cfg=cfg,
        freq_axis=freq_axis,
        p_rad_interp=p_rad_input_for_theoretical,
        st=st,
        logger=logger,
    )
    _require_allowed_keys("phase_ir_theoretical", theo, _THEORETICAL_ALLOWED_KEYS)
    _require_unchanged(
        "phase_ir_theoretical",
        "gain_db",
        gain_db_before_theoretical,
        gain_db,
    )
    _require_unchanged(
        "phase_ir_theoretical",
        "p_rad_interp input",
        p_rad_interp,
        p_rad_input_for_theoretical,
    )

    theo_xo = np.asarray(theo["theo_xo"], dtype=float)
    hpf_freq = theo.get("hpf_freq", None)
    hpf_slope = theo.get("hpf_slope", None)
    p_rad_interp = np.asarray(theo.get("p_rad_interp", p_rad_interp), dtype=float)

    hs = cfg.hpf_settings
    if isinstance(hs, dict) and hs.get('enabled'):
        hpf_f = float(hs.get('freq', 0.0) or 0.0)
        hpf_order = int(hs.get('order', 0) or 0)
        if hpf_f > 0 and hpf_order > 0:
            hpf_db = apply_hpf_to_mags(freq_axis, np.zeros_like(freq_axis), hpf_f, hpf_order)
            gain_db = gain_db + hpf_db
            try:
                logger.info(
                    "HPF magnitude applied to FIR: "
                    f"fc={hpf_f:.1f} Hz, "
                    f"order={hpf_order} "
                    f"({hpf_order * 6:.0f} dB/oct)"
                )
            except Exception:
                pass

    gain_db_before_residual = gain_db.copy()
    freq_axis_before_residual = freq_axis.copy()
    p_rad_before_residual = p_rad_interp.copy()
    m_anal_before_residual = m_anal.copy()
    target_mags_before_residual = target_mags.copy()
    mask_c_before_residual = mask_c.copy()
    gain_db, residual_telemetry = apply_residual_pass_if_enabled(
        cfg=cfg,
        freq_axis=freq_axis,
        gain_db=gain_db,
        conf_mask=conf_mask,
        m_anal=m_anal,
        calc_offset_db=calc_offset_db,
        target_mags=target_mags,
        st=st,
        mask_c=mask_c,
        base_sigma=base_sigma,
        filter_smooth=_filter_smooth,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        logger=logger,
        cfg_float_allow_zero_fn=_cfg_float_allow_zero,
    )
    gain_db = np.asarray(gain_db, dtype=float)
    if gain_db.shape != gain_db_before_residual.shape:
        raise RuntimeError(
            "Phase-IR contract breach in phase_ir_residual: "
            "gain_db shape changed unexpectedly."
        )
    _require_unchanged(
        "phase_ir_residual",
        "freq_axis",
        freq_axis_before_residual,
        freq_axis,
    )
    _require_unchanged(
        "phase_ir_residual",
        "p_rad_interp",
        p_rad_before_residual,
        p_rad_interp,
    )
    _require_unchanged(
        "phase_ir_residual",
        "m_anal",
        m_anal_before_residual,
        m_anal,
    )
    _require_unchanged(
        "phase_ir_residual",
        "target_mags",
        target_mags_before_residual,
        target_mags,
    )
    _require_unchanged(
        "phase_ir_residual",
        "mask_c",
        mask_c_before_residual,
        mask_c,
    )

    gain_db_before_autogain = gain_db.copy()
    gain_db_for_autogain = gain_db.copy()
    ag = compute_auto_gain_and_headroom(
        cfg=cfg,
        gain_db=gain_db_for_autogain,
        mask_c=mask_c,
        logger=logger,
    )
    _require_allowed_keys("phase_ir_autogain", ag, _AUTOGAIN_ALLOWED_KEYS)
    _require_unchanged(
        "phase_ir_autogain",
        "gain_db input",
        gain_db_before_autogain,
        gain_db_for_autogain,
    )
    _require_unchanged(
        "phase_ir_autogain",
        "gain_db",
        gain_db_before_autogain,
        gain_db,
    )

    current_peak_gain = float(ag["current_peak_gain"])
    gain_margin_db = float(ag["gain_margin_db"])
    auto_global_gain_db = float(ag["auto_global_gain_db"])
    auto_headroom_db = float(ag["auto_headroom_db"])
    final_gain_total = np.asarray(ag["final_gain_total"], dtype=float)
    expected_final_gain_total = gain_db + auto_global_gain_db + auto_headroom_db
    _require_unchanged(
        "phase_ir_autogain",
        "final_gain_total formula",
        expected_final_gain_total,
        final_gain_total,
    )

    gain_db_before_build = gain_db.copy()
    p_rad_before_build = p_rad_interp.copy()
    final_gain_total_before_build = final_gain_total.copy()
    auto_global_gain_db_before_build = float(auto_global_gain_db)
    auto_headroom_db_before_build = float(auto_headroom_db)
    gain_db_for_build = gain_db.copy()
    p_rad_for_build = p_rad_interp.copy()
    final_gain_total_for_build = final_gain_total.copy()
    built = build_phase_and_ir(
        cfg=cfg,
        freq_axis=freq_axis,
        n_fft=n_fft,
        gain_db=gain_db_for_build,
        p_rad_interp=p_rad_for_build,
        conf_mask=conf_mask,
        st=st,
        mask_c=mask_c,
        use_bassfirst=use_bassfirst,
        afdw_on=afdw_on,
        logger=logger,
        theo_xo=theo_xo,
        auto_global_gain_db=auto_global_gain_db,
        auto_headroom_db=auto_headroom_db,
        final_gain_total=final_gain_total_for_build,
        limit_gd_gradient_ms_per_oct_fn=_limit_gd_gradient_ms_per_oct,
    )
    _require_allowed_keys("phase_ir_build", built, _BUILD_ALLOWED_KEYS)
    _require_unchanged(
        "phase_ir_build",
        "gain_db input",
        gain_db_before_build,
        gain_db_for_build,
    )
    _require_unchanged(
        "phase_ir_build",
        "p_rad_interp input",
        p_rad_before_build,
        p_rad_for_build,
    )
    _require_unchanged(
        "phase_ir_build",
        "final_gain_total input",
        final_gain_total_before_build,
        final_gain_total_for_build,
    )
    _require_unchanged(
        "phase_ir_build",
        "gain_db",
        gain_db_before_build,
        gain_db,
    )
    _require_unchanged(
        "phase_ir_build",
        "final_gain_total",
        final_gain_total_before_build,
        final_gain_total,
    )
    _require_scalar_unchanged(
        "phase_ir_build",
        "auto_global_gain_db",
        auto_global_gain_db_before_build,
        auto_global_gain_db,
    )
    _require_scalar_unchanged(
        "phase_ir_build",
        "auto_headroom_db",
        auto_headroom_db_before_build,
        auto_headroom_db,
    )

    return PhaseIROutputs(
        impulse=np.asarray(built["impulse"], dtype=float),
        gain_db=np.asarray(gain_db, dtype=float),
        auto_global_gain_db=float(auto_global_gain_db),
        gain_margin_db=float(gain_margin_db),
        auto_headroom_db=float(auto_headroom_db),
        current_peak_gain=float(current_peak_gain),
        final_gain_total=np.asarray(final_gain_total, dtype=float),
        residual_telemetry=residual_telemetry,
    )
