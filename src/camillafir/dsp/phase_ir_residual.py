from __future__ import annotations

import numpy as np
import scipy.ndimage

if __package__ in (None, ""):
    import pathlib
    import sys

    _src_root = pathlib.Path(__file__).resolve().parents[2]
    _src_root_s = str(_src_root)
    if _src_root_s not in sys.path:
        sys.path.insert(0, _src_root_s)
    from camillafir.dsp.camillafir_analysis import _sigma_bins_from_hz
    from camillafir.dsp.gain_policy import build_low_frequency_guard_mask, clamp_gain_curve, resolve_gain_policy
    from camillafir.dsp.phase_ir_types import ResidualTelemetry
    from camillafir.dsp.phase_ir_utils import _cosine_fade_out_01, _smoothstep01
    from camillafir.dsp.smoothing import smooth_gain_fractional_octave
else:
    from .camillafir_analysis import _sigma_bins_from_hz
    from .gain_policy import build_low_frequency_guard_mask, clamp_gain_curve, resolve_gain_policy
    from .phase_ir_types import ResidualTelemetry
    from .phase_ir_utils import _cosine_fade_out_01, _smoothstep01
    from .smoothing import smooth_gain_fractional_octave


def _residual_band_weight(
    *,
    freq_axis: np.ndarray,
    cfg,
    mask_c: np.ndarray,
    skip_right_transition_fade: bool = False,
) -> tuple[np.ndarray, float, float, float]:
    f = np.asarray(freq_axis, dtype=float)
    m = np.asarray(mask_c, dtype=bool)
    w = np.zeros_like(f, dtype=float)
    if f.size == 0 or m.shape != f.shape:
        return w, 0.0, 0.0, 0.0

    try:
        f_min = float(getattr(cfg, "mag_c_min", 0.0) or 0.0)
    except (AttributeError, TypeError, ValueError):
        f_min = 0.0
    try:
        f_max = float(getattr(cfg, "mag_c_max", 0.0) or 0.0)
    except (AttributeError, TypeError, ValueError):
        f_max = 0.0

    if (not np.isfinite(f_min)) or (not np.isfinite(f_max)) or (f_max <= f_min):
        idx = np.where(m)[0]
        if idx.size < 2:
            return w, 0.0, 0.0, 0.0
        f_min = float(f[int(idx[0])])
        f_max = float(f[int(idx[-1])])

    span = float(f_max - f_min)
    if span <= 1e-9:
        return w, float(f_min), float(f_max), 0.0

    edge_hz = 0.0
    try:
        edge_hz = float(getattr(cfg, "residual_band_edge_hz", 0.0) or 0.0)
    except (AttributeError, TypeError, ValueError):
        edge_hz = 0.0
    if (not np.isfinite(edge_hz)) or edge_hz <= 0.0:
        try:
            edge_hz = float(getattr(cfg, "trans_width", 0.0) or 0.0)
        except (AttributeError, TypeError, ValueError):
            edge_hz = 0.0
    if (not np.isfinite(edge_hz)) or edge_hz <= 0.0:
        edge_hz = 0.08 * span
    edge_hz = float(np.clip(edge_hz, 1e-6, 0.49 * span))

    in_band = (f >= f_min) & (f <= f_max) & m
    w[in_band] = 1.0

    left = (f >= f_min) & (f < (f_min + edge_hz)) & m
    if np.any(left):
        w[left] = _smoothstep01((f[left] - f_min) / edge_hz)

    right = (f > (f_max - edge_hz)) & (f <= f_max) & m
    if (not bool(skip_right_transition_fade)) and np.any(right):
        x_right = (f[right] - (f_max - edge_hz)) / edge_hz
        w[right] = _cosine_fade_out_01(x_right)

    return np.clip(w, 0.0, 1.0), float(f_min), float(f_max), float(edge_hz)


def apply_residual_pass_if_enabled(
    *,
    cfg,
    freq_axis: np.ndarray,
    gain_db: np.ndarray,
    conf_mask,
    m_anal: np.ndarray,
    calc_offset_db: float,
    target_mags: np.ndarray,
    st: dict,
    mask_c: np.ndarray,
    base_sigma,
    filter_smooth,
    df_mode: bool,
    raw_g,
    final_g,
    logger,
    cfg_float_allow_zero_fn,
) -> tuple[np.ndarray, ResidualTelemetry | None]:
    """
    Contract:
      - This stage may only change gain_db.
      - It must not mutate phase-domain or IR-domain arrays.
      - Telemetry is returned typed and adapted to stats outside this stage.

    Returns updated gain_db (same shape) and optional residual telemetry.
    """
    _filter_smooth = filter_smooth
    _cfg_float_allow_zero = cfg_float_allow_zero_fn
    residual_telemetry: ResidualTelemetry | None = None

    if bool(getattr(cfg, "enable_residual_pass", False)) and bool(getattr(cfg, "enable_mag_correction", True)):
        try:
            freq_axis = np.asarray(freq_axis, dtype=float)
            gain_db = np.asarray(gain_db, dtype=float).copy()
            mask_c = np.asarray(mask_c, dtype=bool)
            if gain_db.shape != freq_axis.shape or mask_c.shape != gain_db.shape:
                return gain_db, residual_telemetry
            gain_policy = resolve_gain_policy(cfg, cfg_float_allow_zero_fn=_cfg_float_allow_zero)
            measured_aligned = (m_anal - calc_offset_db)
            pred0 = measured_aligned + gain_db
            resid0 = (target_mags - pred0)

            resid = np.zeros_like(gain_db, dtype=float)
            resid[mask_c] = resid0[mask_c]

            k = float(getattr(cfg, "residual_conf_power", 2.0) or 2.0)
            try:
                if conf_mask is not None:
                    resid[mask_c] *= np.clip(conf_mask[mask_c], 0.0, 1.0) ** k
            except (TypeError, ValueError, IndexError):
                pass

            strength = float(getattr(cfg, "residual_strength", 0.6) or 0.6)
            strength = float(np.clip(strength, 0.0, 1.0))
            mult = float(getattr(cfg, "residual_smoothing_mult", 2.0) or 2.0)
            mult = max(1.0, float(mult))

            try:
                _base_sigma = float(base_sigma)
            except (TypeError, ValueError, OverflowError):
                _base_sigma = float(60 // (_filter_smooth / 12 if _filter_smooth > 0 else 1))
            if (not np.isfinite(_base_sigma)) or _base_sigma <= 0.0:
                _base_sigma = float(60 // (_filter_smooth / 12 if _filter_smooth > 0 else 1))
            _df_mode = bool(df_mode) if df_mode is not None else bool(getattr(cfg, "df_smoothing", False))
            if _df_mode:
                df_ref = 44100.0 / 65536.0
                sigma_hz = float(_base_sigma) * df_ref * mult
                sigma_bins = _sigma_bins_from_hz(
                    freq_axis,
                    sigma_hz=sigma_hz,
                    fallback_bins=max(2.0, float(_base_sigma) * mult)
                )
                resid_sm = scipy.ndimage.gaussian_filter1d(resid, sigma=float(sigma_bins))
            else:
                resid_sm = smooth_gain_fractional_octave(
                    freq_axis,
                    resid,
                    _filter_smooth,
                    mult=mult,
                )

            already = bool(isinstance(st, dict) and st.get("mag_transition_fade_applied", False))
            band_w, band_min_hz, band_max_hz, band_edge_hz = _residual_band_weight(
                freq_axis=freq_axis,
                cfg=cfg,
                mask_c=mask_c,
                skip_right_transition_fade=already,
            )
            residual_delta = np.asarray(resid_sm, dtype=float) * float(strength) * np.asarray(band_w, dtype=float)

            try:
                max_delta_db = float(getattr(cfg, "residual_max_delta_db", 1.25) or 1.25)
            except (AttributeError, TypeError, ValueError):
                max_delta_db = 1.25
            if (not np.isfinite(max_delta_db)) or max_delta_db <= 0.0:
                max_delta_db = 1.25
            residual_delta = np.clip(residual_delta, -float(max_delta_db), float(max_delta_db))

            low_guard_mask = build_low_frequency_guard_mask(freq_axis, gain_policy)

            apply_mask = mask_c & (band_w > 0.0)
            low_guard_apply_mask = apply_mask & low_guard_mask
            blocked_boost_bins = int(np.count_nonzero((residual_delta > 1e-9) & low_guard_apply_mask))
            if np.any(low_guard_apply_mask):
                residual_delta[low_guard_apply_mask] = np.minimum(residual_delta[low_guard_apply_mask], 0.0)

            gain_before = gain_db.copy()
            if np.any(apply_mask):
                gain_db[apply_mask] = gain_before[apply_mask] + residual_delta[apply_mask]

            if np.any(low_guard_apply_mask):
                gain_db[low_guard_apply_mask] = np.minimum(
                    gain_db[low_guard_apply_mask],
                    gain_before[low_guard_apply_mask],
                )

            if np.any(mask_c):
                gain_db = clamp_gain_curve(gain_db, policy=gain_policy, mask=mask_c)

            residual_telemetry = ResidualTelemetry(
                residual_pass_enabled=True,
                residual_strength=float(strength),
                residual_smoothing_mult=float(mult),
                residual_conf_power=float(k),
                residual_max_delta_db=float(max_delta_db),
                residual_band_min_hz=float(band_min_hz),
                residual_band_max_hz=float(band_max_hz),
                residual_band_edge_hz=float(band_edge_hz),
                residual_band_bins=int(np.count_nonzero(apply_mask)),
                residual_lf_guard_bins=int(np.count_nonzero(low_guard_apply_mask)),
                residual_lf_boost_blocked_bins=int(blocked_boost_bins),
                residual_right_transition_fade_skipped=bool(already),
            )
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
            pass

    return gain_db, residual_telemetry
