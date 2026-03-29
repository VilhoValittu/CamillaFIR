from __future__ import annotations

import math

import numpy as np

from ...common.acoustic_stats import calc_acoustic_score, calc_ai_summary_from_stats
from ...dsp.target_match import target_match_from_stats
from . import shared
from .rank_score import (
    OFFICIAL_RANK_SCORE_CONTEXT,
    attach_official_rank_score,
    compute_rank_score_components,
)
from .runtime_context import (
    _auto_collect_reflections,
    _auto_event_penalty_weighted,
    _auto_event_severity,
    _auto_get_top_modes_hz,
    _auto_get_worst_mode_hz,
    _auto_mode_band,
    _auto_pick_metric,
)


def _auto_dsp_quality_penalty(st: dict | None) -> tuple[float, dict]:
    st = st or {}
    penalty = 0.0
    dbg = {}

    real_rms = _auto_pick_metric(
        st,
        (
            "real_mag_error_rms",
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
            "post_to_ir_delta_rms_20_200_db",
        ),
        abs_value=True,
        nonneg=True,
    )
    if real_rms is not None:
        penalty += 6.0 * max(0.0, float(real_rms) - 0.90)
    dbg["real_rms"] = real_rms

    ripple_rms = _auto_pick_metric(
        st,
        (
            "ripple_rms",
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
        ),
        abs_value=True,
        nonneg=True,
    )
    if ripple_rms is not None:
        penalty += 4.0 * max(0.0, float(ripple_rms) - 0.50)
    dbg["ripple_rms"] = ripple_rms

    gd_grad_max = _auto_pick_metric(
        st,
        (
            "gd_grad_limiter_after_max_ms_per_oct",
            "gd_grad_limiter_before_max_ms_per_oct",
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        ),
        abs_value=True,
        nonneg=True,
    )
    if gd_grad_max is not None:
        penalty += 1.5 * max(0.0, float(gd_grad_max) - 8.0)
    dbg["gd_grad_max"] = gd_grad_max

    gd_abs_max = _auto_pick_metric(
        st,
        ("gd_abs_max_20_500_ms",),
        abs_value=True,
        nonneg=True,
    )
    if gd_abs_max is not None:
        penalty += 0.08 * max(0.0, float(gd_abs_max) - 25.0)
    dbg["gd_abs_max_20_500_ms"] = gd_abs_max

    pre_ringing_db = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_ringing_db",
            "mixed_pre_ringing_after_db",
            "ir_pre_energy_guard_after_db",
            "mixed_pre_ringing_before_db",
            "ir_pre_energy_guard_before_db",
        ),
    )
    if pre_ringing_db is not None:
        penalty += 0.70 * max(0.0, float(pre_ringing_db) + 45.0)
    dbg["pre_ringing_db"] = pre_ringing_db

    pre_post_ratio = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_post_ratio",
            "ir_pre_energy_guard_after_ratio",
            "ir_pre_energy_guard_before_ratio",
        ),
        nonneg=True,
    )
    if pre_post_ratio is not None:
        penalty += 30.0 * max(0.0, float(pre_post_ratio) - 0.015)
    dbg["ir_pre_post_ratio"] = pre_post_ratio

    phase_boundary_mdb = _auto_pick_metric(
        st,
        (
            "phase_boundary_peak_mdb",
            "phase_corr_boundary_peak_mdb",
        ),
        abs_value=True,
        nonneg=True,
    )
    if phase_boundary_mdb is not None:
        penalty += 0.015 * max(0.0, float(phase_boundary_mdb) - 120.0)
    dbg["phase_boundary_peak_mdb"] = phase_boundary_mdb

    return float(max(0.0, penalty)), dbg


def _auto_excursion_penalty(st: dict | None) -> tuple[float, dict]:
    st = st or {}
    penalty = 0.0
    dbg = {}

    exc_raw = st.get("exc_prot", None)
    exc_known = exc_raw is not None
    exc_on = bool(exc_raw) if exc_known else None
    exc_freq = shared._auto_safe_float(st.get("exc_freq", 0.0), 0.0)

    try:
        exc_bins = int(float(st.get("boost_candidate_bins_excprot", 0) or 0))
    except Exception:
        exc_bins = 0
    lf_boost_max = shared._auto_safe_float(st.get("lf_boost_max_db", 0.0), 0.0)
    pen_exc_off = 0.0
    pen_exc_invalid = 0.0
    pen_bins = 0.0
    pen_lf = 0.0

    if exc_known and (exc_on is False):
        pen_exc_off = 2.0
        penalty += float(pen_exc_off)
    if exc_known and (exc_on is True) and (not np.isfinite(exc_freq) or exc_freq <= 0.0):
        pen_exc_invalid = 0.8
        penalty += float(pen_exc_invalid)
    if exc_bins > 0:
        pen_bins = float(min(2.5, 0.10 * float(exc_bins)))
        penalty += float(pen_bins)

    pen_lf = float(min(12.0, 1.25 * max(0.0, float(lf_boost_max) - 1.5)))
    penalty += float(pen_lf)
    penalty = min(16.0, float(penalty))

    dbg["exc_known"] = bool(exc_known)
    dbg["exc_on"] = exc_on
    dbg["exc_freq"] = float(exc_freq)
    dbg["exc_bins"] = int(exc_bins)
    dbg["lf_boost_max_db"] = float(lf_boost_max)
    dbg["pen_exc_off"] = float(pen_exc_off)
    dbg["pen_exc_invalid"] = float(pen_exc_invalid)
    dbg["pen_bins"] = float(pen_bins)
    dbg["pen_lf"] = float(pen_lf)
    dbg["pen_total_pre_cap"] = float(pen_exc_off + pen_exc_invalid + pen_bins + pen_lf)
    dbg["pen_total"] = float(penalty)
    return float(max(0.0, penalty)), dbg


def _auto_exc_penalty_bins_from_dbg(exc_dbg: dict | None) -> float:
    try:
        dbg = dict(exc_dbg or {})
    except Exception:
        dbg = {}
    v = shared._auto_safe_float(dbg.get("pen_bins", float("nan")), float("nan"))
    if np.isfinite(v):
        return float(max(0.0, v))
    exc_bins = int(shared._auto_safe_float(dbg.get("exc_bins", 0), 0.0))
    return float(min(2.5, 0.10 * max(0, exc_bins)))


def _auto_exc_zero_penalty_freq_hz_from_stats(st: dict | None) -> float:
    st = dict(st or {})
    v = shared._auto_safe_float(st.get("boost_candidate_min_hz", float("nan")), float("nan"))
    if not np.isfinite(v) or float(v) <= 0.0:
        return float("nan")
    return float(
        np.clip(
            float(v),
            float(shared._auto_safe_float(shared.AUTO_MODE_EXC_MIN_HZ, 20.0)),
            float(shared._auto_safe_float(shared.AUTO_MODE_EXC_MAX_HZ, 80.0)),
        )
    )


def _auto_focus_ripple_from_stats(
    st: dict | None,
    *,
    focus_lo_hz: float,
    focus_hi_hz: float,
) -> float | None:
    st = dict(st or {})
    lo = shared._auto_safe_float(focus_lo_hz, float("nan"))
    hi = shared._auto_safe_float(focus_hi_hz, float("nan"))
    if not (np.isfinite(lo) and np.isfinite(hi)) or float(hi) <= float(lo):
        return None

    mode = str(st.get("analysis_mode", "native") or "native").strip().lower()

    def _pick_arr(base_key: str, *fallback_keys: str) -> np.ndarray:
        keys: list[str] = []
        if mode == "comparison":
            keys.append(f"cmp_{str(base_key)}")
            keys.extend([f"cmp_{str(k)}" for k in fallback_keys])
        keys.append(str(base_key))
        keys.extend([str(k) for k in fallback_keys])
        for key in keys:
            try:
                arr = np.asarray(st.get(key, []), dtype=float).reshape(-1)
            except Exception:
                arr = np.asarray([], dtype=float)
            if arr.size:
                return np.asarray(arr, dtype=float)
        return np.asarray([], dtype=float)

    # Primary metric: local corrected-response RMS vs target inside the detected focus band.
    # This keeps auto-mode decisions tied to residual acoustic error instead of IR realization fidelity.
    f = _pick_arr("freq_axis")
    m_meas = _pick_arr("measured_mags")
    t_tgt = _pick_arr("target_mags")
    g_real = _pick_arr("realized_filter_mags", "filter_mags")
    c_mask = _pick_arr("confidence_mask")
    n = int(min(f.size, m_meas.size, t_tgt.size))
    if n >= 8:
        f = np.asarray(f[:n], dtype=float)
        pred = np.asarray(m_meas[:n], dtype=float)
        if g_real.size >= n:
            pred = pred + np.asarray(g_real[:n], dtype=float)
        err = pred - np.asarray(t_tgt[:n], dtype=float)
        mask = np.isfinite(f) & np.isfinite(err) & (f >= float(lo)) & (f <= float(hi))
        if int(np.count_nonzero(mask)) >= 8:
            err_use = np.asarray(err[mask], dtype=float)
            if c_mask.size >= n:
                w = np.clip(np.asarray(c_mask[:n], dtype=float)[mask], 0.0, 1.0)
                w = np.maximum(w, 0.05)
                w_sum = float(np.sum(w))
                if np.isfinite(w_sum) and w_sum > 1e-12:
                    return float(np.sqrt(np.sum(w * err_use * err_use) / w_sum))
            return float(np.sqrt(np.mean(err_use * err_use)))

    # Fallback: if corrected-response data is incomplete, fall back to filter-realization delta.
    g_pred = _pick_arr("predicted_filter_mags")
    g_real = _pick_arr("realized_filter_mags", "filter_mags")
    n = int(min(f.size, g_pred.size, g_real.size))
    if n < 8:
        return None
    f = np.asarray(f[:n], dtype=float)
    d = np.asarray(g_real[:n], dtype=float) - np.asarray(g_pred[:n], dtype=float)
    m = np.isfinite(f) & np.isfinite(d) & (f >= float(lo)) & (f <= float(hi))
    if int(np.count_nonzero(m)) < 8:
        return None
    dv = np.asarray(d[m], dtype=float)
    off = float(np.median(dv))
    d_shape = np.asarray(dv, dtype=float) - float(off)
    return float(np.sqrt(np.mean(d_shape * d_shape)))


def _auto_score_result(
    result,
    *,
    auto_exc_freq_hz: float | None = None,
    focus_lo_hz: float | None = None,
    focus_hi_hz: float | None = None,
    base_data: dict | None = None,
) -> dict:
    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    l_ai = calc_ai_summary_from_stats(l_st)
    r_ai = calc_ai_summary_from_stats(r_st)

    def _ai_score_with_fallback(st: dict, ai: dict) -> float:
        score = shared._auto_safe_float((ai or {}).get("score"), float("nan"))
        if np.isfinite(score):
            return float(score)
        try:
            conf = shared._auto_safe_float(
                st.get("cmp_avg_confidence", st.get("avg_confidence", 0.0)),
                0.0,
            )
            _rms_fb, match_fb = target_match_from_stats(
                st,
                include_filter=False,
                use_confidence=True,
                use_smart_scan_range=True,
            )
            if match_fb is None:
                return 0.0
            rt60 = st.get("rt60_val", None)
            rt_rel = st.get("rt60_reliability", None)
            return shared._auto_safe_float(
                calc_acoustic_score(conf, float(match_fb), rt60_s=rt60, rt60_rel=rt_rel),
                0.0,
            )
        except Exception:
            return 0.0

    l_score = _ai_score_with_fallback(l_st, l_ai)
    r_score = _ai_score_with_fallback(r_st, r_ai)
    avg_score = (l_score + r_score) / 2.0
    lr_delta = abs(l_score - r_score)

    net_boost_max = max(
        shared._auto_safe_float(l_st.get("net_boost_peak_db", 0.0), 0.0),
        shared._auto_safe_float(r_st.get("net_boost_peak_db", 0.0), 0.0),
    )
    l_refs = _auto_collect_reflections(l_st)
    r_refs = _auto_collect_reflections(r_st)
    events_total = int(len(l_refs) + len(r_refs))
    events_severity_l = _auto_event_severity(l_refs)
    events_severity_r = _auto_event_severity(r_refs)
    events_severity_raw = float(events_severity_l + events_severity_r)
    events_severity = float(math.log1p(max(0.0, events_severity_raw) / 6.0))
    dsp_pen_l, dsp_dbg_l = _auto_dsp_quality_penalty(l_st)
    dsp_pen_r, dsp_dbg_r = _auto_dsp_quality_penalty(r_st)
    dsp_penalty_raw = 0.5 * (float(dsp_pen_l) + float(dsp_pen_r))
    exc_pen_l, exc_dbg_l = _auto_excursion_penalty(l_st)
    exc_pen_r, exc_dbg_r = _auto_excursion_penalty(r_st)
    exc_penalty_raw_total = 0.5 * (float(exc_pen_l) + float(exc_pen_r))
    exc_penalty_bins_raw = 0.5 * (
        float(_auto_exc_penalty_bins_from_dbg(exc_dbg_l))
        + float(_auto_exc_penalty_bins_from_dbg(exc_dbg_r))
    )
    exc_penalty_raw = float(exc_penalty_raw_total)
    exc_penalty_waived = bool(np.isfinite(shared._auto_safe_float(auto_exc_freq_hz, float("nan"))))
    auto_exc_zero_l = _auto_exc_zero_penalty_freq_hz_from_stats(l_st)
    auto_exc_zero_r = _auto_exc_zero_penalty_freq_hz_from_stats(r_st)
    auto_exc_zero_vals = [float(v) for v in (auto_exc_zero_l, auto_exc_zero_r) if np.isfinite(v)]
    auto_exc_zero_penalty_hz = float(min(auto_exc_zero_vals)) if auto_exc_zero_vals else float("nan")
    auto_exc_hz_now = shared._auto_safe_float(auto_exc_freq_hz, float("nan"))
    exc_penalty_bins_waived = False
    if (
        bool(exc_penalty_waived)
        and np.isfinite(auto_exc_zero_penalty_hz)
        and np.isfinite(auto_exc_hz_now)
        and (float(auto_exc_hz_now) + 1e-6) >= float(auto_exc_zero_penalty_hz)
    ):
        exc_penalty_raw = max(0.0, float(exc_penalty_raw_total) - float(exc_penalty_bins_raw))
        exc_penalty_bins_waived = bool(float(exc_penalty_bins_raw) > 1e-9)
    exc_penalty = float(exc_penalty_raw) * (0.35 if exc_penalty_waived else 1.0)

    boost_knee_db = 1.0
    boost_x = (float(net_boost_max) - 5.0) / float(boost_knee_db)
    boost_x = float(np.clip(boost_x, -60.0, 60.0))
    soft_hinge_db = float(boost_knee_db) * float(np.log1p(np.exp(boost_x)))
    boost_pen = min(12.0, 1.25 * soft_hinge_db)
    dsp_penalty = min(12.0, 0.30 * float(dsp_penalty_raw))
    all_events = list(l_refs) + list(r_refs)
    event_pen_raw = _auto_event_penalty_weighted(
        all_events,
        base_per_event=float(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_BASE_PER_EVENT, 0.5)),
        dt_weight=float(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_DT_WEIGHT, 0.02)),
        power=float(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_DT_POWER, 2.0)),
        dt_ref_ms=float(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_DT_REF_MS, 100.0)),
    )
    event_pen_conf_scale = 1.0
    if bool(shared.AUTO_MODE_EVENT_PEN_CONF_GATE_ENABLE):
        conf_vals = []
        for st in (l_st, r_st):
            c = shared._auto_safe_float(
                st.get("cmp_avg_confidence", st.get("avg_confidence", float("nan"))),
                float("nan"),
            )
            if not np.isfinite(c):
                continue
            c01 = float(c / 100.0) if float(c) > 1.5 else float(c)
            c01 = float(np.clip(c01, 0.0, 1.0))
            conf_vals.append(float(c01))
        if conf_vals:
            conf_mean = float(np.mean(np.asarray(conf_vals, dtype=float)))
            min_scale = float(np.clip(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_CONF_GATE_MIN_SCALE, 0.45), 0.0, 1.0))
            full_conf = float(np.clip(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_CONF_GATE_FULL_CONF, 0.85), 1e-6, 1.0))
            conf_norm = float(np.clip(conf_mean / full_conf, 0.0, 1.0))
            event_pen_conf_scale = float(min_scale + (1.0 - min_scale) * conf_norm)
    event_pen_raw *= float(event_pen_conf_scale)
    event_pen = min(12.0, max(0.0, event_pen_raw))
    lr_pen = min(4.0, 0.03 * lr_delta)
    exc_penalty = min(12.0, float(exc_penalty))
    filter_key = shared._auto_filter_cache_key(base_data)
    phase_limit_used_hz = shared._auto_safe_float((base_data or {}).get("phase_limit", float("nan")), float("nan"))
    if shared._auto_is_phase_search_filter(filter_key):
        phase_limit_used_hz = float(shared._auto_phase_limit_clip(phase_limit_used_hz, default=shared.AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ))
    phase_limit_penalty = float(
        shared._auto_phase_limit_prior_penalty(phase_limit_used_hz, filter_key=filter_key)
    )

    def _rank_scale(v: float) -> float:
        g = float(shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_GAIN, 1.0))
        b = float(shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_BIAS, 0.0))
        return float(np.clip(float(g) * float(v) + float(b), 0.0, 100.0))

    base_rank_components = compute_rank_score_components(
        avg_score=avg_score,
        boost_penalty=boost_pen,
        event_penalty=event_pen,
        lr_delta_penalty=lr_pen,
        dsp_penalty=dsp_penalty,
        exc_penalty=exc_penalty,
        phase_limit_penalty=phase_limit_penalty,
        gain=shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_GAIN, 1.0),
        bias=shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_BIAS, 0.0),
        context=OFFICIAL_RANK_SCORE_CONTEXT,
    )
    rank_raw = float(base_rank_components.get("rank_score_raw", 0.0))
    rank_score_base = float(base_rank_components.get("rank_score", 0.0))
    rank_score = float(rank_score_base)
    focus_ripple_l = None
    focus_ripple_r = None
    flo = shared._auto_safe_float(focus_lo_hz, float("nan"))
    fhi = shared._auto_safe_float(focus_hi_hz, float("nan"))
    if np.isfinite(flo) and np.isfinite(fhi) and float(fhi) > float(flo):
        focus_ripple_l = _auto_focus_ripple_from_stats(
            l_st,
            focus_lo_hz=float(flo),
            focus_hi_hz=float(fhi),
        )
        focus_ripple_r = _auto_focus_ripple_from_stats(
            r_st,
            focus_lo_hz=float(flo),
            focus_hi_hz=float(fhi),
        )
    if not (
        np.isfinite(shared._auto_safe_float(focus_ripple_l, float("nan")))
        or np.isfinite(shared._auto_safe_float(focus_ripple_r, float("nan")))
    ):
        focus_ripple_keys = (
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
            "post_to_ir_delta_rms_20_200_db",
            "ripple_rms",
        )
        focus_ripple_l = _auto_pick_metric(l_st, focus_ripple_keys, abs_value=True, nonneg=True)
        focus_ripple_r = _auto_pick_metric(r_st, focus_ripple_keys, abs_value=True, nonneg=True)
    focus_ripple_vals = []
    for v in (focus_ripple_l, focus_ripple_r):
        x = shared._auto_safe_float(v, float("nan"))
        if np.isfinite(x):
            focus_ripple_vals.append(float(x))
    focus_ripple = float(np.mean(np.asarray(focus_ripple_vals, dtype=float))) if focus_ripple_vals else 0.0

    top_modes = []
    try:
        if bool(shared.AUTO_MODE_DUAL_MODE_ENABLED):
            top_modes = _auto_get_top_modes_hz(result, top_n=int(shared.AUTO_MODE_DUAL_MODE_TOP_N))
        else:
            m1 = _auto_get_worst_mode_hz(result)
            top_modes = [float(m1)] if m1 is not None else []
    except Exception:
        top_modes = []

    mode_hz = shared._auto_safe_float(
        (top_modes[0] if len(top_modes) >= 1 else _auto_get_worst_mode_hz(result)),
        float("nan"),
    )
    mode_band_lo = float("nan")
    mode_band_hi = float("nan")
    mode_ripple_db = float("nan")
    mode2_hz = float("nan")
    mode2_band_lo = float("nan")
    mode2_band_hi = float("nan")
    mode2_ripple_db = float("nan")
    mode_band = _auto_mode_band(mode_hz, base_data=base_data) if np.isfinite(mode_hz) else None
    if isinstance(mode_band, tuple) and len(mode_band) == 2:
        mode_band_lo = float(shared._auto_safe_float(mode_band[0], float("nan")))
        mode_band_hi = float(shared._auto_safe_float(mode_band[1], float("nan")))
        if np.isfinite(mode_band_lo) and np.isfinite(mode_band_hi) and (mode_band_hi > mode_band_lo):
            mr_l = _auto_focus_ripple_from_stats(l_st, focus_lo_hz=float(mode_band_lo), focus_hi_hz=float(mode_band_hi))
            mr_r = _auto_focus_ripple_from_stats(r_st, focus_lo_hz=float(mode_band_lo), focus_hi_hz=float(mode_band_hi))
            mr_vals = []
            for mv in (mr_l, mr_r):
                x = shared._auto_safe_float(mv, float("nan"))
                if np.isfinite(x):
                    mr_vals.append(float(x))
            if mr_vals:
                mode_ripple_db = float(np.mean(np.asarray(mr_vals, dtype=float)))
    if (not np.isfinite(mode_ripple_db)) and np.isfinite(mode_hz):
        mode_ripple_db = float(shared._auto_safe_float(focus_ripple, float("nan")))

    if len(top_modes) >= 2:
        mode2_hz = float(shared._auto_safe_float(top_modes[1], float("nan")))
        mode2_band = _auto_mode_band(mode2_hz, base_data=base_data) if np.isfinite(mode2_hz) else None
        if isinstance(mode2_band, tuple) and len(mode2_band) == 2:
            mode2_band_lo = float(shared._auto_safe_float(mode2_band[0], float("nan")))
            mode2_band_hi = float(shared._auto_safe_float(mode2_band[1], float("nan")))
            if np.isfinite(mode2_band_lo) and np.isfinite(mode2_band_hi) and (mode2_band_hi > mode2_band_lo):
                mr2_l = _auto_focus_ripple_from_stats(l_st, focus_lo_hz=float(mode2_band_lo), focus_hi_hz=float(mode2_band_hi))
                mr2_r = _auto_focus_ripple_from_stats(r_st, focus_lo_hz=float(mode2_band_lo), focus_hi_hz=float(mode2_band_hi))
                mr2_vals = []
                for mv in (mr2_l, mr2_r):
                    x = shared._auto_safe_float(mv, float("nan"))
                    if np.isfinite(x):
                        mr2_vals.append(float(x))
                if mr2_vals:
                    mode2_ripple_db = float(np.mean(np.asarray(mr2_vals, dtype=float)))
        if (not np.isfinite(mode2_ripple_db)) and np.isfinite(mode2_hz):
            mode2_ripple_db = float(shared._auto_safe_float(focus_ripple, float("nan")))

    mode_r1 = shared._auto_safe_float(mode_ripple_db, float("nan"))
    mode_r2 = shared._auto_safe_float(mode2_ripple_db, float("nan"))
    mode_combined = float("nan")
    if np.isfinite(mode_r1) and np.isfinite(mode_r2):
        mode_combined = max(float(mode_r1), float(shared.AUTO_MODE_MODE_RIPPLE_SECONDARY_W) * float(mode_r2))
    elif np.isfinite(mode_r1):
        mode_combined = float(mode_r1)
    elif np.isfinite(mode_r2):
        mode_combined = float(mode_r2)

    mode_penalty = 0.0
    if np.isfinite(mode_combined):
        mode_penalty = float(shared.AUTO_MODE_MODE_RIPPLE_PENALTY_W) * max(
            0.0,
            float(mode_combined) - float(shared.AUTO_MODE_MODE_RIPPLE_OK_DB),
        )
        mode_penalty = float(np.clip(mode_penalty, 0.0, 6.0))

    if mode_penalty > 0.0:
        rank_raw = float(rank_raw - float(mode_penalty))
        rank_score = float(_rank_scale(rank_raw))
    rank_components = compute_rank_score_components(
        avg_score=avg_score,
        boost_penalty=boost_pen,
        event_penalty=event_pen,
        lr_delta_penalty=lr_pen,
        dsp_penalty=dsp_penalty,
        exc_penalty=exc_penalty,
        mode_penalty=mode_penalty,
        phase_limit_penalty=phase_limit_penalty,
        gain=shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_GAIN, 1.0),
        bias=shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_BIAS, 0.0),
        context=OFFICIAL_RANK_SCORE_CONTEXT,
    )
    rank_score = float(rank_components.get("rank_score", rank_score))
    realized_keys = (
        "post_to_ir_staged_shape_delta_rms_20_200_db",
        "post_to_ir_shape_delta_rms_20_200_db",
        "post_to_ir_delta_rms_20_200_db",
    )
    realized_l = _auto_pick_metric(l_st, realized_keys, abs_value=True, nonneg=True)
    realized_r = _auto_pick_metric(r_st, realized_keys, abs_value=True, nonneg=True)
    realized_vals = []
    for rv in (realized_l, realized_r):
        x = shared._auto_safe_float(rv, float("nan"))
        if np.isfinite(x):
            realized_vals.append(float(x))
    realized_rms_20_200 = float(np.mean(np.asarray(realized_vals, dtype=float))) if realized_vals else float("nan")

    ripple_raw_l = _auto_pick_metric(l_st, ("ripple_rms",), abs_value=True, nonneg=True)
    ripple_raw_r = _auto_pick_metric(r_st, ("ripple_rms",), abs_value=True, nonneg=True)
    ripple_raw_vals = []
    for rv in (ripple_raw_l, ripple_raw_r):
        x = shared._auto_safe_float(rv, float("nan"))
        if np.isfinite(x):
            ripple_raw_vals.append(float(x))
    ripple_raw = float(np.mean(np.asarray(ripple_raw_vals, dtype=float))) if ripple_raw_vals else float("nan")

    pre_post_keys = (
        "ir_pre_post_ratio",
        "ir_pre_energy_guard_after_ratio",
        "ir_pre_energy_guard_before_ratio",
    )
    pre_post_l = None if bool(l_st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        l_st,
        pre_post_keys,
        nonneg=True,
    )
    pre_post_r = None if bool(r_st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        r_st,
        pre_post_keys,
        nonneg=True,
    )
    pre_post_l_f = shared._auto_safe_float(pre_post_l, float("nan"))
    pre_post_r_f = shared._auto_safe_float(pre_post_r, float("nan"))
    pre_post_max = float("nan")
    pre_post_vals = []
    if np.isfinite(pre_post_l_f):
        pre_post_vals.append(float(pre_post_l_f))
    if np.isfinite(pre_post_r_f):
        pre_post_vals.append(float(pre_post_r_f))
    if pre_post_vals:
        pre_post_max = float(max(pre_post_vals))

    metrics_out = {
        "rank_score": float(rank_score),
        "rank_score_base": float(rank_score_base),
        "rank_score_official": float(rank_score),
        "rank_score_components": dict(rank_components),
        "avg_score": float(avg_score),
        "focus_ripple_db": float(focus_ripple or 0.0),
        "mode_hz": float(mode_hz) if np.isfinite(mode_hz) else float("nan"),
        "mode_band_lo": float(mode_band_lo) if np.isfinite(mode_band_lo) else float("nan"),
        "mode_band_hi": float(mode_band_hi) if np.isfinite(mode_band_hi) else float("nan"),
        "mode_ripple_db": float(mode_ripple_db) if np.isfinite(mode_ripple_db) else float("nan"),
        "mode2_hz": float(mode2_hz) if np.isfinite(mode2_hz) else float("nan"),
        "mode2_band_lo": float(mode2_band_lo) if np.isfinite(mode2_band_lo) else float("nan"),
        "mode2_band_hi": float(mode2_band_hi) if np.isfinite(mode2_band_hi) else float("nan"),
        "mode2_ripple_db": float(mode2_ripple_db) if np.isfinite(mode2_ripple_db) else float("nan"),
        "mode_ripple_combined_db": float(mode_combined) if np.isfinite(mode_combined) else float("nan"),
        "mode_penalty": float(mode_penalty),
        "realized_rms_20_200_db": float(realized_rms_20_200) if np.isfinite(realized_rms_20_200) else float("nan"),
        "ir_pre_post_energy_ratio_l": float(pre_post_l_f) if np.isfinite(pre_post_l_f) else float("nan"),
        "ir_pre_post_energy_ratio_r": float(pre_post_r_f) if np.isfinite(pre_post_r_f) else float("nan"),
        "ir_pre_post_energy_ratio_max": float(pre_post_max) if np.isfinite(pre_post_max) else float("nan"),
        "ripple_rms": float(ripple_raw) if np.isfinite(ripple_raw) else float("nan"),
        "lr_delta_score": float(lr_delta),
        "max_net_boost_db": float(net_boost_max),
        "boost_penalty": float(boost_pen),
        "events_total": int(events_total),
        "events_severity": float(events_severity),
        "events_severity_raw": float(events_severity_raw),
        "events_severity_l": float(events_severity_l),
        "events_severity_r": float(events_severity_r),
        "event_penalty": float(event_pen),
        "lr_delta_penalty": float(lr_pen),
        "dsp_penalty": float(dsp_penalty),
        "dsp_penalty_raw": float(dsp_penalty_raw),
        "dsp_penalty_l": float(dsp_pen_l),
        "dsp_penalty_r": float(dsp_pen_r),
        "exc_penalty": float(exc_penalty),
        "exc_penalty_raw": float(exc_penalty_raw),
        "exc_penalty_raw_total": float(exc_penalty_raw_total),
        "exc_penalty_bins_raw": float(exc_penalty_bins_raw),
        "exc_penalty_bins_waived": bool(exc_penalty_bins_waived),
        "exc_penalty_waived": bool(exc_penalty_waived),
        "exc_penalty_l": float(exc_pen_l),
        "exc_penalty_r": float(exc_pen_r),
        "auto_exc_zero_penalty_hz": float(auto_exc_zero_penalty_hz) if np.isfinite(auto_exc_zero_penalty_hz) else float("nan"),
        "phase_limit_hz": float(phase_limit_used_hz) if np.isfinite(phase_limit_used_hz) else float("nan"),
        "phase_limit_penalty": float(phase_limit_penalty),
        "dsp_dbg_l": dict(dsp_dbg_l),
        "dsp_dbg_r": dict(dsp_dbg_r),
        "exc_dbg_l": dict(exc_dbg_l),
        "exc_dbg_r": dict(exc_dbg_r),
    }
    return attach_official_rank_score(metrics_out, components=rank_components)


__all__ = [
    "_auto_dsp_quality_penalty",
    "_auto_exc_penalty_bins_from_dbg",
    "_auto_exc_zero_penalty_freq_hz_from_stats",
    "_auto_excursion_penalty",
    "_auto_focus_ripple_from_stats",
    "_auto_score_result",
]
