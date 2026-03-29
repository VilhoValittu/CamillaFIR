import numpy as np
import scipy.ndimage

from ..common.acoustic_stats import _clamp, calc_acoustic_score, calc_ai_summary_from_stats
from ..common.comparison_stats import _make_comparison_stats
from ..dsp.phase_ir_metrics import format_pre_energy_status
from ..dsp.quality_metrics import _mag_error_db, _rms
from ..dsp.target_match import target_match_from_stats as _target_match_from_stats_ssot

def format_band_rt60_summary(bands, *, picks=None):
    if not bands:
        return "-"
    pick_list = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0] if picks is None else list(picks)
    normalized = []
    if isinstance(bands, dict):
        items = list(bands.items())
    else:
        items = list(bands or [])
    for key, value in items:
        try:
            freq = float(key)
            rt60 = float(value)
        except Exception:
            continue
        if np.isfinite(freq) and np.isfinite(rt60):
            normalized.append((float(freq), float(rt60)))
    if not normalized:
        return "-"
    seen = set()
    out = []
    for pick in pick_list:
        freq, rt60 = min(normalized, key=lambda item: abs(float(item[0]) - float(pick)))
        label = f"{float(freq):.0f}Hz"
        if label in seen:
            continue
        seen.add(label)
        out.append(f"{label}:{float(rt60):.2f}s")
    return " | ".join(out) if out else "-"


def _float_allow_zero(v, default: float) -> float:
    """Sisainen apufunktio: float allow zero."""
    if v is None:
        return float(default)
    if isinstance(v, str) and v.strip() == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)



def _calc_target_match(stats):
    """Sisainen apufunktio: calc target match."""
    return _target_match_from_stats_ssot(
        stats or {},
        include_filter=True,
        use_confidence=True,
        use_smart_scan_range=True,
    )


def calc_target_match_from_stats(stats: dict):
    """Laskee: calc target match from stats."""
    try:
        return _calc_target_match(stats or {})
    except Exception:
        return None, None


def format_dsp_quality_report_block(settings, l_stats, r_stats):
    """Rakenna Summaryyn DSP Quality Report -blokki (L/R)."""
    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}
    debug_report = bool(settings.get("quality_report_debug", False))

    def _safe_float(v):
        try:
            x = float(v)
            if np.isfinite(x):
                return float(x)
        except Exception:
            pass
        return None

    def _as_arr(v):
        try:
            a = np.asarray(v, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([], dtype=float)
        if a.size == 0:
            return np.asarray([], dtype=float)
        return a

    def _gd_grad_max_value(st):
        for k in (
            "gd_grad_limiter_after_max_ms_per_oct",
            "gd_grad_limiter_before_max_ms_per_oct",
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None:
                return v
        return None

    def _gd_grad_max_hz(st):
        for k in (
            "gd_grad_limiter_peak_hz",
            "gd_limiter_max_grad_hz",
            "gd_grad_limiter_max_grad_hz",
            "gd_limiter_max_grad_after_hz",
            "gd_grad_limiter_max_grad_after_hz",
            "gd_limiter_max_grad_before_hz",
            "gd_grad_limiter_max_grad_before_hz",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None:
                return v
        return None

    def _gd_max_value(st):
        for k in (
            "gd_max_ms",
            "max_gd_ms",
            "mixed_excess_delay_before_ms",
            "xo_diff_raw_max_gd_ms",
            "hpf_diff_raw_max_gd_ms",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None:
                return abs(v)
        refs = st.get("reflections", []) or []
        vals = []
        for ref in refs:
            if isinstance(ref, dict):
                g = _safe_float(ref.get("gd_error", None))
                if g is not None:
                    vals.append(abs(g))
        if vals:
            return float(max(vals))
        return None

    def _pre_ringing_db(st):
        if bool(st.get("pre_energy_metric_suspect", False)):
            return None
        for k in (
            "ir_pre_ringing_db",
            "mixed_pre_ringing_after_db",
            "ir_pre_energy_guard_after_db",
            "mixed_pre_ringing_before_db",
            "ir_pre_energy_guard_before_db",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None:
                return v
        return None

    def _pre_post_ratio(st, pre_db):
        if bool(st.get("pre_energy_metric_suspect", False)):
            return None
        for k in (
            "ir_pre_post_ratio",
            "ir_pre_energy_guard_after_ratio",
            "ir_pre_energy_guard_before_ratio",
        ):
            v = _safe_float(st.get(k, None))
            if v is not None and v >= 0.0:
                return v
        if pre_db is None:
            return None
        try:
            r = float(10.0 ** (float(pre_db) / 10.0))
            return r if np.isfinite(r) else None
        except Exception:
            return None

    def _pre_metric_info(st, *, debug_report=False):
        valid = bool(st.get("pre_energy_metric_valid", not bool(st.get("pre_energy_metric_suspect", False))))
        suspect = bool(not valid)
        reason_code = str(st.get("pre_energy_metric_reason_code", "") or "").strip()
        note = "ok" if not suspect else format_pre_energy_status(reason_code, debug=debug_report)
        return suspect, note

    def _active_axes(st):
        mode = str(st.get("analysis_mode", "native") or "native").strip().lower()
        if mode == "comparison":
            f = _as_arr(st.get("cmp_freq_axis", None))
            t = _as_arr(st.get("cmp_target_mags", None))
            mm = _as_arr(st.get("cmp_mag_mask", None))
            offset_db = _safe_float(st.get("cmp_offset_db", 0.0))
            if offset_db is None:
                offset_db = 0.0
            m_raw = _as_arr(st.get("cmp_measured_mags_raw", None))
            m_corr = _as_arr(st.get("cmp_measured_mags", None))
            g_pred = _as_arr(st.get("cmp_predicted_filter_mags", None))
            g_real = _as_arr(st.get("cmp_realized_filter_mags", None))
            g_legacy = _as_arr(st.get("cmp_filter_mags", None))
            g_legacy_src = str(st.get("cmp_filter_mags_source", "") or "").strip().lower()
            if m_raw.size < 8 and m_corr.size >= 8:
                m_raw = np.asarray(m_corr, dtype=float) + float(offset_db)
            if m_corr.size < 8 and m_raw.size >= 8:
                m_corr = np.asarray(m_raw, dtype=float) - float(offset_db)
            if g_pred.size < 8 and g_legacy.size >= 8 and g_legacy_src != "ir_fft_final":
                g_pred = g_legacy
            if g_real.size < 8 and g_legacy.size >= 8 and g_legacy_src == "ir_fft_final":
                g_real = g_legacy
            if g_pred.size < 8 and g_legacy.size >= 8:
                g_pred = g_legacy
            if g_real.size < 8 and g_legacy.size >= 8:
                g_real = g_legacy
            return f, m_corr, t, g_pred, g_real, mm, float(offset_db)

        f = _as_arr(st.get("freq_axis", None))
        t = _as_arr(st.get("target_mags", None))
        mm = _as_arr(st.get("mag_mask", st.get("mask_c", None)))
        offset_db = _safe_float(st.get("offset_db", 0.0))
        if offset_db is None:
            offset_db = 0.0
        m_raw = _as_arr(st.get("measured_mags_raw", None))
        m_corr = _as_arr(st.get("measured_mags", None))
        g_pred = _as_arr(st.get("predicted_filter_mags", None))
        g_real = _as_arr(st.get("realized_filter_mags", None))
        g_legacy = _as_arr(st.get("filter_mags", None))
        g_legacy_src = str(st.get("filter_mags_source", "") or "").strip().lower()
        if m_raw.size < 8 and m_corr.size >= 8:
            m_raw = np.asarray(m_corr, dtype=float) + float(offset_db)
        if m_corr.size < 8 and m_raw.size >= 8:
            m_corr = np.asarray(m_raw, dtype=float) - float(offset_db)
        if g_pred.size < 8 and g_legacy.size >= 8 and g_legacy_src != "ir_fft_final":
            g_pred = g_legacy
        if g_real.size < 8 and g_legacy.size >= 8 and g_legacy_src == "ir_fft_final":
            g_real = g_legacy
        if g_pred.size < 8 and g_legacy.size >= 8:
            g_pred = g_legacy
        if g_real.size < 8 and g_legacy.size >= 8:
            g_real = g_legacy
        return f, m_corr, t, g_pred, g_real, mm, float(offset_db)

    def _phase_limit_hz(st):
        for k in ("phase_limit_hz", "phase_limit"):
            v = _safe_float(st.get(k, None))
            if v is not None and v > 0.0:
                return float(v)
        v = _safe_float(settings.get("phase_limit", None))
        if v is not None and v > 0.0:
            return float(v)
        return None

    def _phase_boundary_peak_mdb(freqs, filt_db, valid_mask, phase_lim_hz):
        if phase_lim_hz is None:
            return None, None
        f = np.asarray(freqs, dtype=float)
        g = np.asarray(filt_db, dtype=float)
        valid = np.asarray(valid_mask, dtype=bool)
        if f.size < 32 or g.size != f.size or valid.size != f.size:
            return None, None
        lim = float(phase_lim_hz)
        lo = max(20.0, 0.75 * lim)
        hi = min(float(np.max(f[np.isfinite(f)])) if np.any(np.isfinite(f)) else lim, 1.20 * lim)
        if hi <= (lo + 1.0):
            return None, None
        sel = valid & (f >= lo) & (f <= hi)
        if np.count_nonzero(sel) < 12:
            return None, None
        fv = f[sel]
        gv = g[sel]
        sigma = float(np.clip(float(gv.size) / 18.0, 1.0, 14.0))
        try:
            trend = scipy.ndimage.gaussian_filter1d(gv, sigma=sigma, mode="nearest")
            resid = gv - trend
            idx = int(np.argmax(np.abs(resid)))
            return float(np.abs(resid[idx]) * 1000.0), float(fv[idx])
        except Exception:
            return None, None

    def _collect(st):
        out = {
            "pred_mag_error_rms": None,
            "pred_mag_error_max": None,
            "pred_mag_error_max_hz": None,
            "pred_mag_error_rms_20_200": None,
            "pred_mag_error_max_20_200": None,
            "pred_mag_error_rms_200_2000": None,
            "real_mag_error_rms": None,
            "real_mag_error_max": None,
            "real_mag_error_max_hz": None,
            "real_mag_error_rms_20_200": None,
            "real_mag_error_max_20_200": None,
            "real_mag_error_rms_200_2000": None,
            "mid_refit_enabled": bool(st.get("mid_refit_enabled", False)),
            "mid_refit_reason": str(st.get("mid_refit_reason", "") or ""),
            "mid_refit_k": _safe_float(st.get("mid_refit_k", None)),
            "mid_refit_err_rms_before": _safe_float(st.get("mid_refit_err_rms_before", None)),
            "mid_refit_err_rms_after": _safe_float(st.get("mid_refit_err_rms_after", None)),
            "mid_refit_delta_rms": _safe_float(st.get("mid_refit_delta_rms", None)),
            "mid_refit_conf_avg_200_2000": _safe_float(st.get("mid_refit_conf_avg_200_2000", None)),
            "ripple_rms": None,
            "gd_max": _gd_max_value(st),
            "gd_grad_max": _gd_grad_max_value(st),
            "gd_grad_max_hz": _gd_grad_max_hz(st),
            "phase_boundary_peak_mdb": None,
            "phase_boundary_peak_hz": None,
            "bass_adaptive_enabled": bool(st.get("bass_adaptive_smoothing_enabled", False)),
            "bass_adaptive_conf_source": str(st.get("bass_adaptive_conf_source", "") or ""),
            "bass_adaptive_isolation_mode": bool(st.get("bass_adaptive_isolation_mode", False)),
            "bass_adaptive_sigma_scale": _safe_float(st.get("bass_adaptive_smoothing_sigma_scale", None)),
            "bass_adaptive_conf_floor": _safe_float(st.get("bass_adaptive_smoothing_conf_floor", None)),
            "bass_adaptive_w_gamma": _safe_float(st.get("bass_adaptive_smoothing_w_gamma", None)),
            "bass_adaptive_w_max": _safe_float(st.get("bass_adaptive_smoothing_w_max", None)),
            "bass_adaptive_avg_w": _safe_float(st.get("bass_adaptive_smoothing_avg_w_20_200", None)),
            "bass_adaptive_delta_rms_20_200": _safe_float(st.get("bass_adaptive_smoothing_delta_rms_db_20_200", None)),
            "bass_adaptive_delta_max_20_200": _safe_float(st.get("bass_adaptive_smoothing_delta_max_db_20_200", None)),
            "bass_adaptive_delta_max_hz_20_200": _safe_float(st.get("bass_adaptive_smoothing_delta_max_hz_20_200", None)),
            "bass_adaptive_delta_basis": str(st.get("bass_adaptive_smoothing_delta_basis", "") or ""),
            "bass_adaptive_effectiveness_pct": None,
            "post_to_ir_delta_rms_20_200": _safe_float(st.get("post_to_ir_delta_rms_20_200_db", None)),
            "post_to_ir_delta_max_20_200": _safe_float(st.get("post_to_ir_delta_max_20_200_db", None)),
            "post_to_ir_delta_max_hz_20_200": _safe_float(st.get("post_to_ir_delta_max_hz_20_200", None)),
            "post_to_ir_delta_offset_20_200": _safe_float(st.get("post_to_ir_delta_offset_20_200_db", None)),
            "post_to_ir_shape_delta_rms_20_200": _safe_float(st.get("post_to_ir_shape_delta_rms_20_200_db", None)),
            "post_to_ir_shape_delta_max_20_200": _safe_float(st.get("post_to_ir_shape_delta_max_20_200_db", None)),
            "post_to_ir_shape_delta_max_hz_20_200": _safe_float(st.get("post_to_ir_shape_delta_max_hz_20_200", None)),
            "post_to_ir_staged_delta_rms_20_200": _safe_float(st.get("post_to_ir_staged_delta_rms_20_200_db", None)),
            "post_to_ir_staged_delta_max_20_200": _safe_float(st.get("post_to_ir_staged_delta_max_20_200_db", None)),
            "post_to_ir_staged_delta_max_hz_20_200": _safe_float(st.get("post_to_ir_staged_delta_max_hz_20_200", None)),
            "post_to_ir_staged_delta_offset_20_200": _safe_float(st.get("post_to_ir_staged_delta_offset_20_200_db", None)),
            "post_to_ir_staged_shape_delta_rms_20_200": _safe_float(st.get("post_to_ir_staged_shape_delta_rms_20_200_db", None)),
            "post_to_ir_staged_shape_delta_max_20_200": _safe_float(st.get("post_to_ir_staged_shape_delta_max_20_200_db", None)),
            "post_to_ir_staged_shape_delta_max_hz_20_200": _safe_float(st.get("post_to_ir_staged_shape_delta_max_hz_20_200", None)),
            "ir_realized_level_match_enabled": bool(st.get("ir_realized_level_match_enabled", False)),
            "ir_realized_level_match_applied": bool(st.get("ir_realized_level_match_applied", False)),
            "ir_realized_level_match_reason": str(st.get("ir_realized_level_match_reason", "") or ""),
            "ir_realized_level_match_mid_lo_hz": _safe_float(st.get("ir_realized_level_match_mid_lo_hz", None)),
            "ir_realized_level_match_mid_hi_hz": _safe_float(st.get("ir_realized_level_match_mid_hi_hz", None)),
            "ir_realized_level_match_delta_db_raw": _safe_float(st.get("ir_realized_level_match_delta_db_raw", None)),
            "ir_realized_level_match_delta_db_applied": _safe_float(st.get("ir_realized_level_match_delta_db_applied", None)),
            "ir_realized_level_match_delta_db_after": _safe_float(st.get("ir_realized_level_match_delta_db_after", None)),
            "ir_realized_level_match_scale": _safe_float(st.get("ir_realized_level_match_scale", None)),
            "post_to_ir_delta_rms_magc": _safe_float(st.get("post_to_ir_delta_rms_magc_db", None)),
            "post_to_ir_delta_max_magc": _safe_float(st.get("post_to_ir_delta_max_magc_db", None)),
            "post_to_ir_delta_max_hz_magc": _safe_float(st.get("post_to_ir_delta_max_hz_magc", None)),
            "bass_boost_cap_enabled": bool(st.get("bass_boost_cap_enabled", False)),
            "bass_boost_cap_avg_extra_db_20_200": _safe_float(st.get("bass_boost_cap_avg_extra_db_20_200", None)),
            "bass_boost_cap_max_extra_db_20_200": _safe_float(st.get("bass_boost_cap_max_extra_db_20_200", None)),
            "bass_boost_post_restore_applied": bool(st.get("bass_boost_post_restore_applied", False)),
            "bass_boost_post_restore_strength": _safe_float(st.get("bass_boost_post_restore_strength", None)),
            "bass_boost_post_restore_bins": _safe_float(st.get("bass_boost_post_restore_bins", None)),
            "bass_boost_post_restore_delta_rms_20_200": _safe_float(st.get("bass_boost_post_restore_delta_rms_20_200", None)),
            "bass_boost_post_restore_delta_max_20_200": _safe_float(st.get("bass_boost_post_restore_delta_max_20_200", None)),
            "conf_pull_bass_boost_floor_hz": _safe_float(st.get("conf_pull_post_bass_boost_floor_hz", None)),
            "conf_pull_bass_boost_floor_min": _safe_float(st.get("conf_pull_post_bass_boost_floor_min", None)),
            "conf_pull_bass_boost_restore": _safe_float(st.get("conf_pull_post_bass_boost_restore", None)),
            "conf_pull_bass_boost_restore_mean_eff": _safe_float(st.get("conf_pull_post_bass_boost_restore_mean_eff", None)),
            "pre_ringing_db": None,
            "ir_pre_post_ratio": None,
            "pre_metric_suspect": False,
            "pre_metric_note": "",
        }
        f, m_corr, t, g_pred, g_real, mm, offset_db = _active_axes(st)
        if min(f.size, m_corr.size, t.size, g_pred.size) < 8:
            pre_db = _pre_ringing_db(st)
            out["pre_ringing_db"] = pre_db
            out["ir_pre_post_ratio"] = _pre_post_ratio(st, pre_db)
            out["pre_metric_suspect"], out["pre_metric_note"] = _pre_metric_info(st, debug_report=debug_report)
            return out

        cmin = _safe_float(st.get("mag_c_min", settings.get("mag_c_min", 20.0)))
        cmax = _safe_float(st.get("mag_c_max", settings.get("mag_c_max", 20000.0)))
        if cmin is None:
            cmin = 20.0
        if cmax is None or cmax <= cmin:
            cmax = float(np.max(f)) if f.size else (cmin + 1.0)

        def _compute_error_bundle(freqs, measured_aligned, target_db, gain_db, mag_mask, *, include_ripple=False):
            res = {
                "rms_magc": None,
                "max_magc": None,
                "max_hz_magc": None,
                "rms_20_200": None,
                "max_20_200": None,
                "rms_200_2000": None,
                "ripple_rms": None,
                "valid": None,
            }
            n_loc = int(min(freqs.size, measured_aligned.size, target_db.size, gain_db.size))
            if n_loc < 8:
                return res
            f_loc = np.asarray(freqs[:n_loc], dtype=float)
            m_loc = np.asarray(measured_aligned[:n_loc], dtype=float)
            t_loc = np.asarray(target_db[:n_loc], dtype=float)
            g_loc = np.asarray(gain_db[:n_loc], dtype=float)
            if mag_mask.size >= n_loc:
                mm_loc = np.asarray(mag_mask[:n_loc], dtype=float)
            else:
                mm_loc = np.asarray([], dtype=float)

            valid = (
                np.isfinite(f_loc)
                & np.isfinite(m_loc)
                & np.isfinite(t_loc)
                & np.isfinite(g_loc)
                & (f_loc > 0.0)
            )
            err = _mag_error_db(t_loc, m_loc, g_loc)

            def _band_stats(lo_hz: float, hi_hz: float):
                m_band = valid & (f_loc >= float(lo_hz)) & (f_loc <= float(hi_hz))
                if np.count_nonzero(m_band) < 8:
                    return None, None, None
                ev_loc = np.asarray(err[m_band], dtype=float)
                fv_loc = np.asarray(f_loc[m_band], dtype=float)
                rms_loc = _rms(ev_loc)
                if not np.isfinite(rms_loc):
                    rms_loc = None
                max_loc = None
                hz_loc = None
                if ev_loc.size:
                    idx_loc = int(np.argmax(np.abs(ev_loc)))
                    max_loc = float(np.abs(ev_loc[idx_loc]))
                    if fv_loc.size > idx_loc:
                        hz_loc = float(fv_loc[idx_loc])
                return rms_loc, max_loc, hz_loc

            rms_20_200, max_20_200, _ = _band_stats(20.0, 200.0)
            rms_200_2000, _, _ = _band_stats(200.0, 2000.0)
            res["rms_20_200"] = rms_20_200
            res["max_20_200"] = max_20_200
            res["rms_200_2000"] = rms_200_2000

            if mm_loc.size == n_loc:
                band = np.asarray(mm_loc, dtype=float) > 0.5
            else:
                band = (f_loc >= float(cmin)) & (f_loc <= float(cmax))
            mask = valid & band
            if np.count_nonzero(mask) < 8:
                mask = valid & (f_loc >= float(cmin)) & (f_loc <= float(cmax))
            if np.count_nonzero(mask) >= 8:
                ev = np.asarray(err[mask], dtype=float)
                fv = np.asarray(f_loc[mask], dtype=float)
                rms_magc = _rms(ev)
                res["rms_magc"] = rms_magc if np.isfinite(rms_magc) else None
                if ev.size:
                    idx = int(np.argmax(np.abs(ev)))
                    res["max_magc"] = float(np.abs(ev[idx]))
                    res["max_hz_magc"] = float(fv[idx]) if fv.size else None
                if include_ripple:
                    try:
                        sigma = max(1.0, float(ev.size) / 64.0)
                        ev_sm = scipy.ndimage.gaussian_filter1d(ev, sigma=sigma)
                        rp = ev - ev_sm
                        rp_rms = _rms(rp)
                        res["ripple_rms"] = rp_rms if np.isfinite(rp_rms) else None
                    except Exception:
                        res["ripple_rms"] = None

            res["valid"] = valid
            return res

        pred = _compute_error_bundle(f, m_corr, t, g_pred, mm, include_ripple=True)
        out["pred_mag_error_rms"] = pred["rms_magc"]
        out["pred_mag_error_max"] = pred["max_magc"]
        out["pred_mag_error_max_hz"] = pred["max_hz_magc"]
        out["pred_mag_error_rms_20_200"] = pred["rms_20_200"]
        out["pred_mag_error_max_20_200"] = pred["max_20_200"]
        out["pred_mag_error_rms_200_2000"] = pred["rms_200_2000"]
        out["ripple_rms"] = pred["ripple_rms"]

        real = _compute_error_bundle(f, m_corr, t, g_real, mm, include_ripple=False)
        out["real_mag_error_rms"] = real["rms_magc"]
        out["real_mag_error_max"] = real["max_magc"]
        out["real_mag_error_max_hz"] = real["max_hz_magc"]
        out["real_mag_error_rms_20_200"] = real["rms_20_200"]
        out["real_mag_error_max_20_200"] = real["max_20_200"]
        out["real_mag_error_rms_200_2000"] = real["rms_200_2000"]

        n_phase = int(min(f.size, g_real.size))
        if n_phase < 8:
            n_phase = int(min(f.size, g_pred.size))
            g_phase = np.asarray(g_pred[:n_phase], dtype=float)
        else:
            g_phase = np.asarray(g_real[:n_phase], dtype=float)
        f_phase = np.asarray(f[:n_phase], dtype=float)
        valid_phase = np.isfinite(f_phase) & np.isfinite(g_phase) & (f_phase > 0.0)
        phase_lim = _phase_limit_hz(st)
        pb_mdb, pb_hz = _phase_boundary_peak_mdb(f_phase, g_phase, valid_phase, phase_lim)
        out["phase_boundary_peak_mdb"] = pb_mdb
        out["phase_boundary_peak_hz"] = pb_hz

        pre_db = _pre_ringing_db(st)
        out["pre_ringing_db"] = pre_db
        out["ir_pre_post_ratio"] = _pre_post_ratio(st, pre_db)
        out["pre_metric_suspect"], out["pre_metric_note"] = _pre_metric_info(st, debug_report=debug_report)
        try:
            dmax = _safe_float(out.get("bass_adaptive_delta_max_20_200", None))
            emax = _safe_float(out.get("pred_mag_error_max_20_200", None))
            if dmax is not None and emax is not None and emax > 1e-9:
                out["bass_adaptive_effectiveness_pct"] = float((dmax / emax) * 100.0)
        except Exception:
            pass
        return out

    def _fmt(v, unit="", prec=2):
        if v is None:
            return "n/a"
        try:
            x = float(v)
            if not np.isfinite(x):
                return "n/a"
            return f"{x:.{int(prec)}f}{unit}"
        except Exception:
            return "n/a"

    def _fmt_ratio(v):
        if v is None:
            return "n/a"
        try:
            x = float(v)
            if not np.isfinite(x):
                return "n/a"
            return f"{x:.4g}"
        except Exception:
            return "n/a"

    lq = _collect(l_stats)
    rq = _collect(r_stats)
    def _fmt_onoff(v):
        return "ON" if bool(v) else "OFF"
    def _fmt_src(v):
        s = str(v or "").strip()
        return s if s else "n/a"
    lines = [
        "",
        "--- DSP Quality Report ---",
        f"Predicted mag error RMS within mag_c band:      L {_fmt(lq['pred_mag_error_rms'], ' dB')} | R {_fmt(rq['pred_mag_error_rms'], ' dB')}",
        f"Predicted mag error max within mag_c band:      L {_fmt(lq['pred_mag_error_max'], ' dB')} | R {_fmt(rq['pred_mag_error_max'], ' dB')}",
        f"Predicted mag error max within mag_c band @ Hz: L {_fmt(lq['pred_mag_error_max_hz'], '', 1)} | R {_fmt(rq['pred_mag_error_max_hz'], '', 1)}",
        f"Predicted mag error RMS @ 20-200 Hz:            L {_fmt(lq['pred_mag_error_rms_20_200'], ' dB')} | R {_fmt(rq['pred_mag_error_rms_20_200'], ' dB')}",
        f"Predicted mag error max @ 20-200 Hz:            L {_fmt(lq['pred_mag_error_max_20_200'], ' dB')} | R {_fmt(rq['pred_mag_error_max_20_200'], ' dB')}",
        f"Predicted mag error RMS @ 200-2000 Hz:          L {_fmt(lq['pred_mag_error_rms_200_2000'], ' dB')} | R {_fmt(rq['pred_mag_error_rms_200_2000'], ' dB')}",
        f"Realized mag error RMS within mag_c band:       L {_fmt(lq['real_mag_error_rms'], ' dB')} | R {_fmt(rq['real_mag_error_rms'], ' dB')}",
        f"Realized mag error max within mag_c band:       L {_fmt(lq['real_mag_error_max'], ' dB')} | R {_fmt(rq['real_mag_error_max'], ' dB')}",
        f"Realized mag error max within mag_c band @ Hz:  L {_fmt(lq['real_mag_error_max_hz'], '', 1)} | R {_fmt(rq['real_mag_error_max_hz'], '', 1)}",
        f"Realized mag error RMS @ 20-200 Hz:             L {_fmt(lq['real_mag_error_rms_20_200'], ' dB')} | R {_fmt(rq['real_mag_error_rms_20_200'], ' dB')}",
        f"Realized mag error RMS @ 200-2000 Hz:           L {_fmt(lq['real_mag_error_rms_200_2000'], ' dB')} | R {_fmt(rq['real_mag_error_rms_200_2000'], ' dB')}",
        f"Ripple RMS:              L {_fmt(lq['ripple_rms'], ' dB')} | R {_fmt(rq['ripple_rms'], ' dB')}",
        f"GD max (ms):             L {_fmt(lq['gd_max'], '', 2)} | R {_fmt(rq['gd_max'], '', 2)}",
        f"GD gradient max (ms/oct): L {_fmt(lq['gd_grad_max'], '', 2)} | R {_fmt(rq['gd_grad_max'], '', 2)}",
        f"GD gradient max @ Hz:    L {_fmt(lq['gd_grad_max_hz'], '', 1)} | R {_fmt(rq['gd_grad_max_hz'], '', 1)}",
        f"Phase boundary peak (mdB): L {_fmt(lq['phase_boundary_peak_mdb'], '', 2)} | R {_fmt(rq['phase_boundary_peak_mdb'], '', 2)}",
        f"Phase boundary peak @ Hz:  L {_fmt(lq['phase_boundary_peak_hz'], '', 1)} | R {_fmt(rq['phase_boundary_peak_hz'], '', 1)}",
        f"Pre-ringing dB:          L {_fmt(lq['pre_ringing_db'], ' dB')} | R {_fmt(rq['pre_ringing_db'], ' dB')}",
        f"IR pre/post energy ratio: L {_fmt_ratio(lq['ir_pre_post_ratio'])} | R {_fmt_ratio(rq['ir_pre_post_ratio'])}",
    ]
    if debug_report:
        lines += [
            f"Mid re-fit 200-2000: L {_fmt_onoff(lq['mid_refit_enabled'])} (k {_fmt(lq['mid_refit_k'], '', 2)}, conf {_fmt(lq['mid_refit_conf_avg_200_2000'], '', 2)}) | R {_fmt_onoff(rq['mid_refit_enabled'])} (k {_fmt(rq['mid_refit_k'], '', 2)}, conf {_fmt(rq['mid_refit_conf_avg_200_2000'], '', 2)})",
            f"Mid re-fit err RMS 200-2000 (dB): L {_fmt(lq['mid_refit_err_rms_before'], '', 2)} -> {_fmt(lq['mid_refit_err_rms_after'], '', 2)} (delta {_fmt(lq['mid_refit_delta_rms'], '', 2)}) | R {_fmt(rq['mid_refit_err_rms_before'], '', 2)} -> {_fmt(rq['mid_refit_err_rms_after'], '', 2)} (delta {_fmt(rq['mid_refit_delta_rms'], '', 2)})",
            f"Mid re-fit reason: L {_fmt_src(lq['mid_refit_reason'])} | R {_fmt_src(rq['mid_refit_reason'])}",
            f"Bass adaptive smoothing:  L {_fmt_onoff(lq['bass_adaptive_enabled'])} | R {_fmt_onoff(rq['bass_adaptive_enabled'])}",
            f"Bass adaptive conf source: L {_fmt_src(lq['bass_adaptive_conf_source'])} | R {_fmt_src(rq['bass_adaptive_conf_source'])}",
            f"Bass adaptive isolation mode: L {_fmt_onoff(lq['bass_adaptive_isolation_mode'])} | R {_fmt_onoff(rq['bass_adaptive_isolation_mode'])}",
            f"Bass adaptive params: L sigma {_fmt(lq['bass_adaptive_sigma_scale'], '', 2)} / conf {_fmt(lq['bass_adaptive_conf_floor'], '', 2)} / gamma {_fmt(lq['bass_adaptive_w_gamma'], '', 2)} / wmax {_fmt(lq['bass_adaptive_w_max'], '', 2)} | R sigma {_fmt(rq['bass_adaptive_sigma_scale'], '', 2)} / conf {_fmt(rq['bass_adaptive_conf_floor'], '', 2)} / gamma {_fmt(rq['bass_adaptive_w_gamma'], '', 2)} / wmax {_fmt(rq['bass_adaptive_w_max'], '', 2)}",
            f"Bass adaptive avg w 20-200: L {_fmt(lq['bass_adaptive_avg_w'], '', 3)} | R {_fmt(rq['bass_adaptive_avg_w'], '', 3)}",
            f"Bass adaptive delta RMS 20-200 (dB): L {_fmt(lq['bass_adaptive_delta_rms_20_200'], '', 2)} | R {_fmt(rq['bass_adaptive_delta_rms_20_200'], '', 2)}",
            f"Bass adaptive delta max 20-200 (dB): L {_fmt(lq['bass_adaptive_delta_max_20_200'], '', 2)} | R {_fmt(rq['bass_adaptive_delta_max_20_200'], '', 2)}",
            f"Bass adaptive delta max @ Hz (20-200): L {_fmt(lq['bass_adaptive_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['bass_adaptive_delta_max_hz_20_200'], '', 1)}",
            f"Bass adaptive effectiveness (% of max err 20-200): L {_fmt(lq['bass_adaptive_effectiveness_pct'], '%', 1)} | R {_fmt(rq['bass_adaptive_effectiveness_pct'], '%', 1)}",
            f"Bass adaptive delta basis: L {_fmt_src(lq['bass_adaptive_delta_basis'])} | R {_fmt_src(rq['bass_adaptive_delta_basis'])}",
            f"Post->IR delta RMS 20-200 (dB): L {_fmt(lq['post_to_ir_delta_rms_20_200'], '', 2)} | R {_fmt(rq['post_to_ir_delta_rms_20_200'], '', 2)}",
            f"Post->IR delta max @ Hz (20-200): L {_fmt(lq['post_to_ir_delta_max_20_200'], '', 2)} @ {_fmt(lq['post_to_ir_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['post_to_ir_delta_max_20_200'], '', 2)} @ {_fmt(rq['post_to_ir_delta_max_hz_20_200'], '', 1)}",
            f"Post->IR offset 20-200 (dB): L {_fmt(lq['post_to_ir_delta_offset_20_200'], '', 2)} | R {_fmt(rq['post_to_ir_delta_offset_20_200'], '', 2)}",
            f"Post->IR shape delta RMS/max 20-200 (dB): L {_fmt(lq['post_to_ir_shape_delta_rms_20_200'], '', 2)} / {_fmt(lq['post_to_ir_shape_delta_max_20_200'], '', 2)} @ {_fmt(lq['post_to_ir_shape_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['post_to_ir_shape_delta_rms_20_200'], '', 2)} / {_fmt(rq['post_to_ir_shape_delta_max_20_200'], '', 2)} @ {_fmt(rq['post_to_ir_shape_delta_max_hz_20_200'], '', 1)}",
            f"Post+staging->IR delta RMS 20-200 (dB): L {_fmt(lq['post_to_ir_staged_delta_rms_20_200'], '', 2)} | R {_fmt(rq['post_to_ir_staged_delta_rms_20_200'], '', 2)}",
            f"Post+staging->IR delta max @ Hz (20-200): L {_fmt(lq['post_to_ir_staged_delta_max_20_200'], '', 2)} @ {_fmt(lq['post_to_ir_staged_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['post_to_ir_staged_delta_max_20_200'], '', 2)} @ {_fmt(rq['post_to_ir_staged_delta_max_hz_20_200'], '', 1)}",
            f"Post+staging->IR offset 20-200 (dB): L {_fmt(lq['post_to_ir_staged_delta_offset_20_200'], '', 2)} | R {_fmt(rq['post_to_ir_staged_delta_offset_20_200'], '', 2)}",
            f"Post+staging->IR shape delta RMS/max 20-200 (dB): L {_fmt(lq['post_to_ir_staged_shape_delta_rms_20_200'], '', 2)} / {_fmt(lq['post_to_ir_staged_shape_delta_max_20_200'], '', 2)} @ {_fmt(lq['post_to_ir_staged_shape_delta_max_hz_20_200'], '', 1)} | R {_fmt(rq['post_to_ir_staged_shape_delta_rms_20_200'], '', 2)} / {_fmt(rq['post_to_ir_staged_shape_delta_max_20_200'], '', 2)} @ {_fmt(rq['post_to_ir_staged_shape_delta_max_hz_20_200'], '', 1)}",
            f"IR mid-band level match: L {_fmt_onoff(lq['ir_realized_level_match_applied'])} ({_fmt_src(lq['ir_realized_level_match_reason'])}) | R {_fmt_onoff(rq['ir_realized_level_match_applied'])} ({_fmt_src(rq['ir_realized_level_match_reason'])})",
            f"IR mid-band level match delta dB: L raw {_fmt(lq['ir_realized_level_match_delta_db_raw'], '', 2)} / applied {_fmt(lq['ir_realized_level_match_delta_db_applied'], '', 2)} / after {_fmt(lq['ir_realized_level_match_delta_db_after'], '', 2)} | R raw {_fmt(rq['ir_realized_level_match_delta_db_raw'], '', 2)} / applied {_fmt(rq['ir_realized_level_match_delta_db_applied'], '', 2)} / after {_fmt(rq['ir_realized_level_match_delta_db_after'], '', 2)}",
            f"IR mid-band level match scale/band: L x{_fmt(lq['ir_realized_level_match_scale'], '', 4)} @ {_fmt(lq['ir_realized_level_match_mid_lo_hz'], '', 0)}-{_fmt(lq['ir_realized_level_match_mid_hi_hz'], '', 0)} Hz | R x{_fmt(rq['ir_realized_level_match_scale'], '', 4)} @ {_fmt(rq['ir_realized_level_match_mid_lo_hz'], '', 0)}-{_fmt(rq['ir_realized_level_match_mid_hi_hz'], '', 0)} Hz",
            f"Bass boost cap: L {_fmt_onoff(lq['bass_boost_cap_enabled'])} | R {_fmt_onoff(rq['bass_boost_cap_enabled'])}",
            f"Bass boost cap extra 20-200 (dB): L avg {_fmt(lq['bass_boost_cap_avg_extra_db_20_200'], '', 2)} / max {_fmt(lq['bass_boost_cap_max_extra_db_20_200'], '', 2)} | R avg {_fmt(rq['bass_boost_cap_avg_extra_db_20_200'], '', 2)} / max {_fmt(rq['bass_boost_cap_max_extra_db_20_200'], '', 2)}",
            f"Bass boost post-restore: L {_fmt_onoff(lq['bass_boost_post_restore_applied'])} (set {_fmt(lq['bass_boost_post_restore_strength'], '', 2)}, bins {_fmt(lq['bass_boost_post_restore_bins'], '', 0)}) | R {_fmt_onoff(rq['bass_boost_post_restore_applied'])} (set {_fmt(rq['bass_boost_post_restore_strength'], '', 2)}, bins {_fmt(rq['bass_boost_post_restore_bins'], '', 0)})",
            f"Bass boost post-restore delta 20-200 (dB): L rms {_fmt(lq['bass_boost_post_restore_delta_rms_20_200'], '', 2)} / max {_fmt(lq['bass_boost_post_restore_delta_max_20_200'], '', 2)} | R rms {_fmt(rq['bass_boost_post_restore_delta_rms_20_200'], '', 2)} / max {_fmt(rq['bass_boost_post_restore_delta_max_20_200'], '', 2)}",
            f"Conf-pull bass boost floor: L {_fmt(lq['conf_pull_bass_boost_floor_min'], '', 2)} <= {_fmt(lq['conf_pull_bass_boost_floor_hz'], '', 0)} Hz | R {_fmt(rq['conf_pull_bass_boost_floor_min'], '', 2)} <= {_fmt(rq['conf_pull_bass_boost_floor_hz'], '', 0)} Hz",
            f"Conf-pull bass boost restore: L set {_fmt(lq['conf_pull_bass_boost_restore'], '', 2)} / eff {_fmt(lq['conf_pull_bass_boost_restore_mean_eff'], '', 2)} | R set {_fmt(rq['conf_pull_bass_boost_restore'], '', 2)} / eff {_fmt(rq['conf_pull_bass_boost_restore_mean_eff'], '', 2)}",
        ]
    if bool(lq.get("pre_metric_suspect", False)) or bool(rq.get("pre_metric_suspect", False)):
        l_note = str(lq.get("pre_metric_note", "") or "suspect").strip()
        r_note = str(rq.get("pre_metric_note", "") or "suspect").strip()
        lines.append(f"Pre-energy metric sanity: L {l_note} | R {r_note}")
    return lines


def _format_summary_content_legacy(settings, l_stats, r_stats):
    """Jasentaa tai muotoilee: format summary content."""
    from datetime import datetime
    import numpy as np

    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}
    program_version = str(settings.get("program_version", "") or "").strip()

    def _safe_float(v, default=0.0):
        try:
            x = float(v)
            if np.isfinite(x):
                return x
        except Exception:
            pass
        return float(default)

    def _fmt_score(v):
        return "n/a" if v is None else f"{float(v):.3f}/100"

    def _fmt_match(match_pct, rms_db):
        if match_pct is None or rms_db is None:
            return "n/a"
        return f"{float(match_pct):.1f}% (RMS {float(rms_db):.2f} dB)"

    def _fmt_range(rng):
        if not isinstance(rng, (list, tuple)) or len(rng) < 2:
            return "n/a"
        try:
            return f"{float(rng[0]):.0f}-{float(rng[1]):.0f} Hz"
        except Exception:
            return "n/a"

    def _report_offset_db(st: dict) -> float:
        try:
            if "cmp_offset_db" in st:
                return _safe_float(st.get("cmp_offset_db", 0.0), 0.0)
        except Exception:
            pass
        return _safe_float(st.get("offset_db", 0.0), 0.0)

    def _large_offset_warning_line() -> str | None:
        off_l = abs(_report_offset_db(l_stats))
        off_r = abs(_report_offset_db(r_stats))
        max_off = max(off_l, off_r)
        if max_off < 30.0:
            return None
        return (
            "Warning: Very large leveling offset detected "
            f"(L {off_l:.2f} dB | R {off_r:.2f} dB). "
            "Measurement level appears relative/uncalibrated, so absolute-level report lines should be interpreted cautiously."
        )

    def _phase_clamp_line(side: str, st: dict) -> str:
        lim = st.get("phase_corr_clamp_deg", None)
        bef = st.get("phase_corr_max_before_deg", None)
        if lim is None or bef is None:
            return f"{side}: n/a"
        return f"{side}: max {float(bef):.1f} deg -> clamp {float(lim):.1f} deg"

    def _gd_grad_max_value(st: dict):
        keys = (
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        )
        for k in keys:
            try:
                v = float(st.get(k, None))
                if np.isfinite(v):
                    return float(v)
            except Exception:
                continue
        return None

    def _fmt_gd_grad_max(st: dict) -> str:
        v = _gd_grad_max_value(st)
        return "n/a" if v is None else f"{float(v):.2f} ms/oct"

    def _gd_limiter_line(side: str, st: dict) -> str:
        try:
            enabled = bool(st.get("gd_limiter_enabled", st.get("gd_grad_limiter_enabled", False)))
            reason = str(st.get("gd_limiter_reason", st.get("gd_grad_limiter_reason", "unknown")) or "unknown")
            limit_v = st.get("gd_limiter_limit_ms_per_oct", st.get("gd_grad_limit_ms_per_oct", None))
            grad_before = st.get(
                "gd_limiter_max_grad_before_ms_per_oct",
                st.get("gd_grad_limiter_max_grad_before_ms_per_oct", None),
            )
            grad_after = st.get(
                "gd_limiter_max_grad_after_ms_per_oct",
                st.get("gd_grad_limiter_max_grad_after_ms_per_oct", _gd_grad_max_value(st)),
            )

            lim_txt = "n/a"
            if limit_v is not None:
                try:
                    lim_txt = f"{float(limit_v):.2f} ms/oct"
                except Exception:
                    lim_txt = "n/a"

            grad_txt = "n/a"
            try:
                gb = float(grad_before) if grad_before is not None else None
            except Exception:
                gb = None
            try:
                ga = float(grad_after) if grad_after is not None else None
            except Exception:
                ga = None
            if gb is not None and np.isfinite(gb) and ga is not None and np.isfinite(ga):
                grad_txt = f"{gb:.2f} -> {ga:.2f} ms/oct"
            elif ga is not None and np.isfinite(ga):
                grad_txt = f"{ga:.2f} ms/oct"

            return (
                f"{side}: {'ON' if enabled else 'OFF'} "
                f"(reason={reason}, limit={lim_txt}, GD-gradient max {grad_txt})"
            )
        except Exception:
            return f"{side}: n/a"

    def _afdw_line(side: str, st: dict) -> str:
        mode = str(st.get("fdw_mode", "") or "").strip().lower()
        if mode == "fixed":
            cyc = st.get("fdw_fixed_cycles", settings.get("fdw_cycles", None))
            bw = st.get("fdw_fixed_bw_oct", None)
            try:
                cyc_txt = f"{float(cyc):.2f}" if cyc is not None else "n/a"
            except Exception:
                cyc_txt = "n/a"
            try:
                bw_txt = f"{float(bw):.4f}" if bw is not None else "n/a"
            except Exception:
                bw_txt = "n/a"
            return f"{side}: FIXED | cycles={cyc_txt}, BW={bw_txt} oct (A-FDW OFF)"

        active = bool(st.get("afdw_active", False)) or bool(settings.get("enable_afdw", False))
        if not active:
            return f"{side}: OFF"
        mn = st.get("afdw_bw_min_oct", None)
        me = st.get("afdw_bw_mean_oct", None)
        mx = st.get("afdw_bw_max_oct", None)
        if mn is None or me is None or mx is None:
            return f"{side}: ON (effective bandwidth not available)"
        return f"{side}: ON | BW min/mean/max = {float(mn):.4f}/{float(me):.4f}/{float(mx):.4f} oct"

    def _fmt_bands(bands):
        return format_band_rt60_summary(bands)

    def _worst_event(st: dict) -> str:
        refs = st.get("reflections", []) or []
        if not refs:
            return "None"
        try:
            w = max(refs, key=lambda x: float(x.get("gd_error", 0.0) or 0.0))
        except Exception:
            return "None"
        freq = _safe_float(w.get("freq", 0.0), 0.0)
        gd_ms = _safe_float(w.get("gd_error", 0.0), 0.0)
        typ = str(w.get("type", "Event") or "Event")
        return f"{typ} at {freq:.0f} Hz ({gd_ms:.2f} ms)"

    def _calc_acoustic_score(conf_pct, match_pct, rt60_s=None, rt60_reliability=None):
        try:
            return globals()["calc_acoustic_score"](
                float(conf_pct),
                float(match_pct),
                rt60_s=rt60_s,
                rt60_rel=rt60_reliability,
            )
        except Exception:
            conf_pct = float(np.clip(float(conf_pct), 0.0, 100.0))
            match_pct = float(np.clip(float(match_pct), 0.0, 100.0))
            return float(np.clip(0.60 * match_pct + 0.40 * conf_pct, 0.0, 100.0))

    l_rt = _safe_float(l_stats.get("rt60_val", 0.0), 0.0)
    r_rt = _safe_float(r_stats.get("rt60_val", 0.0), 0.0)
    l_band_avg = _safe_float(l_stats.get("rt60_band_avg", 0.0), 0.0)
    r_band_avg = _safe_float(r_stats.get("rt60_band_avg", 0.0), 0.0)
    l_conf = _safe_float(l_stats.get("cmp_avg_confidence", l_stats.get("avg_confidence", 0.0)), 0.0)
    r_conf = _safe_float(r_stats.get("cmp_avg_confidence", r_stats.get("avg_confidence", 0.0)), 0.0)
    l_rms, l_match = _calc_target_match(l_stats)
    r_rms, r_match = _calc_target_match(r_stats)
    l_rms_raw, l_match_raw = _target_match_from_stats_ssot(
        l_stats or {},
        include_filter=False,
        use_confidence=True,
        use_smart_scan_range=True,
    )
    r_rms_raw, r_match_raw = _target_match_from_stats_ssot(
        r_stats or {},
        include_filter=False,
        use_confidence=True,
        use_smart_scan_range=True,
    )

    l_score = None
    if l_match is not None:
        l_score = _calc_acoustic_score(
            l_conf,
            l_match,
            l_stats.get("rt60_val", None),
            l_stats.get("rt60_reliability", None),
        )
    r_score = None
    if r_match is not None:
        r_score = _calc_acoustic_score(
            r_conf,
            r_match,
            r_stats.get("rt60_val", None),
            r_stats.get("rt60_reliability", None),
        )

    lines = [
        "=== CamillaFIR - Filter Generation Summary ===",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if program_version:
        lines.append(f"Version: {program_version}")
    lines += [
        "",
        "--- Executive Summary ---",
        f"Acoustic Score: L {_fmt_score(l_score)} | R {_fmt_score(r_score)}",
        f"Target Match:   L {_fmt_match(l_match, l_rms)} | R {_fmt_match(r_match, r_rms)}",
        f"Confidence:     L {l_conf:.1f}% | R {r_conf:.1f}%",
        f"RT60 Wideband:  L {l_rt:.2f}s | R {r_rt:.2f}s",
        f"Worst Event:    L {_worst_event(l_stats)} | R {_worst_event(r_stats)}",
        "",
        "--- Core Settings ---",
    ]

    keys = [
        "mode",
        "fs",
        "taps",
        "filter_type",
        "mixed_freq",
        "mag_c_min",
        "mag_c_max",
        "max_boost",
        "max_cut_db",
        "max_slope_db_per_oct",
        "hpf_enable",
        "hpf_freq",
        "hpf_slope",
        "enable_tdc",
        "tdc_strength",
        "enable_afdw",
        "comparison_mode",
        "stereo_link",
        "bass_first_ai",
    ]
    for k in keys:
        if k in settings:
            lines.append(f"{k}: {settings.get(k)}")

    lines.append("\n--- Analysis Mode ---")
    lines.append(
        f"Analysis mode: L {str(l_stats.get('analysis_mode', 'native'))} | "
        f"R {str(r_stats.get('analysis_mode', 'native'))}"
    )
    if str(l_stats.get("analysis_mode", "native")) == "comparison":
        lines.append(
            f"Comparison grid (L): fs={_safe_float(l_stats.get('cmp_ref_fs', 0), 0):.0f} "
            f"taps={_safe_float(l_stats.get('cmp_ref_taps', 0), 0):.0f}"
        )
    if str(r_stats.get("analysis_mode", "native")) == "comparison":
        lines.append(
            f"Comparison grid (R): fs={_safe_float(r_stats.get('cmp_ref_fs', 0), 0):.0f} "
            f"taps={_safe_float(r_stats.get('cmp_ref_taps', 0), 0):.0f}"
        )

    max_cut_db = _safe_float(settings.get("max_cut_db", 15.0), 15.0)
    max_slope = _safe_float(settings.get("max_slope_db_per_oct", 12.0), 12.0)
    max_slope_boost = _safe_float(settings.get("max_slope_boost_db_per_oct", 0.0), 0.0) or max_slope
    max_slope_cut = _safe_float(settings.get("max_slope_cut_db_per_oct", 0.0), 0.0) or max_slope
    low_bass_cut_hz = _float_allow_zero(settings.get("low_bass_cut_hz", 40.0), 40.0)

    lines.append("\n--- Correction Guards ---")
    lines.append(f"Max cut: -{max_cut_db:.1f} dB")
    if abs(max_slope_boost - max_slope_cut) > 1e-9:
        lines.append(f"Slope: boost {max_slope_boost:.1f} dB/oct | cut {max_slope_cut:.1f} dB/oct")
    else:
        lines.append(f"Max slope: {max_slope:.1f} dB/oct")
    lines.append(f"Low-bass cut policy: <{low_bass_cut_hz:.1f} Hz (cuts only)")

    lines.append("\n--- Temporal Decay Control (TDC) ---")
    tdc_enabled = bool(settings.get("enable_tdc", False))
    lines.append(f"TDC enabled: {'YES' if tdc_enabled else 'NO'}")
    if tdc_enabled:
        lines.append(f"TDC strength: {_safe_float(settings.get('tdc_strength', 0), 0):.0f} %")
        lines.append(f"TDC max reduction: {_safe_float(settings.get('tdc_max_reduction_db', 0), 0):.1f} dB")
        slope = _safe_float(settings.get("tdc_slope_db_per_oct", 0), 0)
        if slope > 0:
            lines.append(f"TDC slope limit: {slope:.1f} dB/oct")

    lines.append("\n--- A-FDW ---")
    lines.append(_afdw_line("Left", l_stats))
    lines.append(_afdw_line("Right", r_stats))

    lines.append("\n--- XO and Phase ---")
    lines.append(f"XO phase model: L {l_stats.get('xo_summary', '-')} | R {r_stats.get('xo_summary', '-')}")
    lines.append(_phase_clamp_line("L", l_stats))
    lines.append(_phase_clamp_line("R", r_stats))
    lines.append(_gd_limiter_line("L", l_stats))
    lines.append(_gd_limiter_line("R", r_stats))
    lines.append(f"A/B GD-gradient max: L {_fmt_gd_grad_max(l_stats)} | R {_fmt_gd_grad_max(r_stats)}")

    lines.append("\n--- RT60 and Confidence ---")
    lines.append(f"RT60 wideband: L {l_rt:.2f}s | R {r_rt:.2f}s")
    if (l_band_avg > 0.0) or (r_band_avg > 0.0):
        lines.append(f"RT60 band average (125-4kHz): L {l_band_avg:.2f}s | R {r_band_avg:.2f}s")
    l_bands = l_stats.get("rt60_bands", {}) or {}
    r_bands = r_stats.get("rt60_bands", {}) or {}
    if l_bands or r_bands:
        lines.append(f"Band RT60 L: {_fmt_bands(l_bands)}")
        lines.append(f"Band RT60 R: {_fmt_bands(r_bands)}")
    lines.append(f"Confidence: L {l_conf:.1f}% | R {r_conf:.1f}%")

    l_om = l_stats.get("cmp_offset_method", l_stats.get("offset_method", "")) or "-"
    r_om = r_stats.get("cmp_offset_method", r_stats.get("offset_method", "")) or "-"
    lines.append(f"Offset method: L {l_om} | R {r_om}")
    l_win = l_stats.get("cmp_smart_scan_range", l_stats.get("smart_scan_range", None))
    r_win = r_stats.get("cmp_smart_scan_range", r_stats.get("smart_scan_range", None))
    lines.append(f"Level window: L {_fmt_range(l_win)} | R {_fmt_range(r_win)}")

    lines.append("\n--- Target Curve Match ---")
    lines.append(f"Left:  {_fmt_match(l_match, l_rms)}")
    lines.append(f"Right: {_fmt_match(r_match, r_rms)}")
    lines.append(
        "Debug raw->pred: "
        f"L {_fmt_match(l_match_raw, l_rms_raw)} -> {_fmt_match(l_match, l_rms)} | "
        f"R {_fmt_match(r_match_raw, r_rms_raw)} -> {_fmt_match(r_match, r_rms)}"
    )

    lines += format_dsp_quality_report_block(settings, l_stats, r_stats)

    lines.append("\n--- Alignment and Peaks ---")
    lines.append(f"L peak (pre-norm): {_safe_float(l_stats.get('peak_before_norm', 0), 0):.2f} dB")
    lines.append(f"R peak (pre-norm): {_safe_float(r_stats.get('peak_before_norm', 0), 0):.2f} dB")
    lines.append(f"Global offset applied: {_safe_float(l_stats.get('offset_db', 0), 0):.2f} dB")
    offset_warn = _large_offset_warning_line()
    if offset_warn:
        lines.append(offset_warn)
    lines.append(f"Auto gain margin setting: {_safe_float(settings.get('gain', 0.0), 0.0):.2f} dB")
    lines.append(
        f"Applied auto gain: L {_safe_float(l_stats.get('auto_global_gain_db', 0.0), 0.0):.2f} dB | "
        f"R {_safe_float(r_stats.get('auto_global_gain_db', 0.0), 0.0):.2f} dB"
    )

    return "\n".join(lines)


def format_summary_content(settings, l_stats, r_stats):
    """Jasentaa tai muotoilee: format summary content."""
    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}

    if bool(settings.get("comparison_mode", False)):
        l_stats = _make_comparison_stats(l_stats, 44100, 65536)
        r_stats = _make_comparison_stats(r_stats, 44100, 65536)

    return _format_summary_content_legacy(settings, l_stats, r_stats)


_calc_acoustic_score = calc_acoustic_score

__all__ = [
    'format_band_rt60_summary',
    '_float_allow_zero',
    '_clamp',
    'calc_acoustic_score',
    '_calc_acoustic_score',
    'calc_ai_summary_from_stats',
    '_calc_target_match',
    'calc_target_match_from_stats',
    'format_dsp_quality_report_block',
    'format_summary_content',
]
