import json
import io
import logging
import os
import time
import zipfile
from typing import Any

import scipy.io.wavfile

from . import camillafir_plot as plots
from ..config.camillafir_convolver_configs import generate_hlc_config, generate_raspberry_yaml
from ..config.results import FilterResult
from ..dsp.smoothing import AFDW_BW_MAX_OCT, AFDW_BW_MIN_OCT
logger = logging.getLogger("CamillaFIR")

TEST_MODE = 1

TEST_MODE = os.environ.get("CAMILLAFIR_TEST", "0") == "1"

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        if x == x and abs(x) != float("inf"):
            return x
    except Exception:
        pass
    return float(default)

def _collect_reflections(st: dict | None) -> list:
    st = st or {}
    refs = st.get("cmp_reflections", st.get("reflections", []))
    if isinstance(refs, list):
        return refs
    return []

def _pick_metric(st: dict | None, keys: tuple[str, ...], *, abs_value: bool = False, nonneg: bool = False):
    st = st or {}
    for k in keys:
        v = _safe_float(st.get(k, float("nan")), float("nan"))
        if not (v == v) or abs(v) == float("inf"):
            continue
        if abs_value:
            v = abs(v)
        if nonneg and v < 0.0:
            continue
        return float(v)
    return None

def _dsp_quality_penalty(st: dict | None) -> float:
    st = st or {}
    penalty = 0.0

    real_rms = _pick_metric(
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

    ripple_rms = _pick_metric(
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

    gd_grad_max = _pick_metric(
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
        penalty += 0.60 * max(0.0, float(gd_grad_max) - 12.0)

    if not bool(st.get("pre_energy_metric_suspect", False)):
        pre_ringing_db = _pick_metric(
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
            penalty += 0.70 * max(0.0, float(pre_ringing_db) + 40.0)

        pre_post_ratio = _pick_metric(
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

    phase_boundary_mdb = _pick_metric(
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

    return float(max(0.0, penalty))

def _score_export_result(result: FilterResult) -> dict:
    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    l_ai = plots.calc_ai_summary_from_stats(l_st)
    r_ai = plots.calc_ai_summary_from_stats(r_st)

    l_score = l_ai.get("score")
    r_score = r_ai.get("score")
    l_score_f = _safe_float(l_score, 0.0)
    r_score_f = _safe_float(r_score, 0.0)
    avg_score = (l_score_f + r_score_f) / 2.0
    lr_delta = abs(l_score_f - r_score_f)

    net_boost_max = max(
        _safe_float(l_st.get("net_boost_peak_db", 0.0), 0.0),
        _safe_float(r_st.get("net_boost_peak_db", 0.0), 0.0),
    )
    reflections_n = len(_collect_reflections(l_st)) + len(_collect_reflections(r_st))
    dsp_penalty = 0.5 * (_dsp_quality_penalty(l_st) + _dsp_quality_penalty(r_st))

    # Rank score keeps Acoustic Score as the primary metric, then applies
    # small tie-break penalties for practical robustness.
    boost_pen = 1.5 * max(0.0, net_boost_max - 1.0)
    event_pen = 0.5 * float(reflections_n)
    lr_pen = 0.25 * lr_delta
    rank_score = max(0.0, min(100.0, avg_score - boost_pen - event_pen - lr_pen - float(dsp_penalty)))

    return {
        "fs": int(getattr(result, "fs", 0) or 0),
        "rank_score": float(rank_score),
        "avg_score": float(avg_score),
        "lr_delta_score": float(lr_delta),
        "max_net_boost_db": float(net_boost_max),
        "events_total": int(reflections_n),
        "dsp_penalty": float(dsp_penalty),
    }

def _build_export_ranking(results: list[FilterResult]) -> dict:
    entries = [_score_export_result(r) for r in list(results or [])]
    entries = sorted(
        entries,
        key=lambda e: (
            -_safe_float(e.get("rank_score"), 0.0),
            -_safe_float(e.get("avg_score"), 0.0),
            _safe_float(e.get("max_net_boost_db"), 0.0),
            _safe_float(e.get("events_total"), 0.0),
            _safe_float(e.get("lr_delta_score"), 0.0),
        ),
    )
    for idx, e in enumerate(entries, start=1):
        e["rank"] = int(idx)
    best = entries[0] if entries else None
    return {
        "entries": entries,
        "best_fs": None if best is None else int(best.get("fs", 0)),
    }

def _append_export_ranking(summary_content: str, fs_v: int, ranking_context: dict | None) -> str:
    ctx = ranking_context or {}
    entries = list(ctx.get("entries") or [])
    if not entries:
        return summary_content

    cur = None
    for e in entries:
        if int(e.get("fs", -1)) == int(fs_v):
            cur = e
            break
    if cur is None:
        return summary_content

    best_fs = ctx.get("best_fs", None)
    summary_content += "\n=== AUTO RANKING (RUN COMPARISON) ===\n"
    summary_content += (
        "Method: rank_score = avg_acoustic_score - boost_penalty - event_penalty - L/R_delta_penalty - dsp_penalty\n"
    )
    summary_content += (
        "Penalties: boost=1.5*dB over +1dB net boost max, event=0.5/event, L/R delta=0.25/score-point, dsp=quality-derived\n"
    )
    summary_content += (
        f"Current run: {int(fs_v)} Hz | rank #{int(cur.get('rank', 0))}/{len(entries)} | "
        f"rank_score={_safe_float(cur.get('rank_score'), 0.0):.3f}/100\n"
    )
    if best_fs is not None:
        summary_content += f"Recommended run: {int(best_fs)} Hz\n"
    summary_content += (
        f"{'Rank':<6}{'FS (Hz)':<10}{'RankScore':<12}{'AvgScore':<10}{'DSPpen':<9}"
        f"{'MaxNetBoost':<13}{'Events':<8}{'L/R Delta':<10}\n"
    )
    summary_content += "-" * 78 + "\n"
    for e in entries:
        summary_content += (
            f"#{int(e.get('rank', 0)):<5}"
            f"{int(e.get('fs', 0)):<10}"
            f"{_safe_float(e.get('rank_score'), 0.0):<12.3f}"
            f"{_safe_float(e.get('avg_score'), 0.0):<10.3f}"
            f"{_safe_float(e.get('dsp_penalty'), 0.0):<9.2f}"
            f"{_safe_float(e.get('max_net_boost_db'), 0.0):<13.2f}"
            f"{int(e.get('events_total', 0)):<8}"
            f"{_safe_float(e.get('lr_delta_score'), 0.0):<10.3f}\n"
        )
    return summary_content

def _append_dsp_effective_params(summary_content, data, fs_v):
    try:
        enable_afdw = bool(data.get('enable_afdw', False))
        enable_tdc  = bool(data.get('enable_tdc',  False))
        tdc_strength = float(data.get('tdc_strength', 0.0) or 0.0)
        fdw_cycles = float(data.get('fdw_cycles', 15.0) or 15.0)
        fdw_oct_width = (2.0 / fdw_cycles) if fdw_cycles > 0 else 0.0
        afdw_min = max(3.0, fdw_cycles / 3.0)
        afdw_min_oct_width = (2.0 / afdw_min) if afdw_min > 0 else 0.0
        fdw_oct_width = float(max(AFDW_BW_MIN_OCT, min(AFDW_BW_MAX_OCT, fdw_oct_width)))
        afdw_min_oct_width = float(max(AFDW_BW_MIN_OCT, min(AFDW_BW_MAX_OCT, afdw_min_oct_width)))

        df_on = bool(data.get('df_smoothing', False))
        df_ref = 44100.0 / 65536.0
        fsmooth = data.get('filter_smooth', data.get('smoothing_level', 12))
        base_sigma = 60 // (fsmooth / 12 if (fsmooth or 0) > 0 else 1)
        sigma_hz = float(base_sigma) * df_ref
        df_cur = (float(fs_v) / float(data.get('taps', 65536) or 65536))
        sigma_bins = (sigma_hz / df_cur) if (df_cur and df_cur > 0) else float(base_sigma)

        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += f"Sample rate: {int(fs_v)} Hz\n"
        try:
            psl = str(data.get("plot_smoothing_level", "Psychoacoustic") or "Psychoacoustic").strip()
            psl_display = "CamillaFIR Reference" if "psy" in psl.lower() else psl
            summary_content += f"Plot smoothing: {psl_display}\n"
        except Exception:
            pass


        if enable_afdw:
            summary_content += "FDW mode: Adaptive (A-FDW)\n"
            summary_content += f"FDW base cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"
            summary_content += f"FDW min cycles:  {afdw_min:.2f}  (oct width -> {afdw_min_oct_width:.3f})\n"
            summary_content += "Note: A-FDW adapts per frequency/confidence; values above are the configured baseline.\n"
        else:
            summary_content += "FDW mode: Fixed\n"
            summary_content += f"FDW cycles: {fdw_cycles:.2f}  (oct width -> {fdw_oct_width:.3f})\n"

        summary_content += f"TDC: {'ON' if enable_tdc else 'OFF'}\n"
        if enable_tdc:
            summary_content += f"TDC strength: {tdc_strength:.1f}% (base_strength = {tdc_strength/100.0:.3f})\n"

        summary_content += f"DF smoothing: {'ON' if df_on else 'OFF'}\n"
        if df_on:
            summary_content += f"DF smoothing sigma: {sigma_bins:.1f} bins -> {sigma_hz:.2f} Hz\n"

        try:
            auto_meta = data.get("_auto_mode_meta", None)
            if bool(data.get("camillafir_automatic_mode", False)) and isinstance(auto_meta, dict):
                bm = dict(auto_meta.get("best_metrics", {}) or {})
                bp = dict(auto_meta.get("best_preset", {}) or {})
                tc = dict(data.get("_auto_target_curve_meta", {}) or {})
                summary_content += "\n=== CAMILLAFIR AUTOMATIC MODE ===\n"
                summary_content += (
                    f"Trials: {int(auto_meta.get('trials_ok', 0))}/{int(auto_meta.get('trials_total', 0))} "
                    f"(search grid: {int(auto_meta.get('search_fs', 0))} Hz, {int(auto_meta.get('search_taps', 0))} taps)\n"
                )
                if tc:
                    tc_method = str(tc.get("selection_method", "fit_rms"))
                    summary_content += (
                        f"Selected target curve: {str(tc.get('selected_hc_mode', 'n/a'))} "
                        f"(fit_rms={float(tc.get('fit_rms_db', 0.0)):.3f} dB, method={tc_method})\n"
                    )
                    if tc_method == "top3x10_trials":
                        ev = list(tc.get("evaluated", []) or [])
                        seed = dict(tc.get("best_preset", {}) or {})
                        top_n = int(tc.get("top_n", 0) or 0)
                        tr_n = int(tc.get("trials_per_curve", 0) or 0)
                        if top_n > 0 and tr_n > 0:
                            summary_content += f"Target selection grid: top-{top_n} x {tr_n} trials\n"
                        if seed:
                            summary_content += (
                                "Target seed preset: "
                                + ", ".join([f"{k}={seed[k]}" for k in sorted(seed.keys())])
                                + "\n"
                            )
                        for i, row in enumerate(ev[:3], start=1):
                            bm_t = dict(row.get("best_metrics", {}) or {})
                            summary_content += (
                                f"Target #{i}: {str(row.get('hc_mode', 'n/a'))} "
                                f"(best_rank={float(bm_t.get('rank_score', 0.0)):.3f}, "
                                f"avg_rank={float(row.get('avg_rank_score', 0.0)):.3f}, "
                                f"ok={int(row.get('trials_ok', 0))}/{int(row.get('trials_total', 0))})\n"
                            )
                summary_content += (
                    f"Best rank score: {float(bm.get('rank_score', 0.0)):.3f}/100 "
                    f"(avg={float(bm.get('avg_score', 0.0)):.3f}, "
                    f"dsp_pen={float(bm.get('dsp_penalty', 0.0)):.2f}, "
                    f"exc_pen={float(bm.get('exc_penalty', 0.0)):.2f}, "
                    f"max_net_boost={float(bm.get('max_net_boost_db', 0.0)):.2f} dB, "
                    f"events={int(bm.get('events_total', 0))}, "
                    f"event_sev={float(bm.get('events_severity', 0.0)):.2f})\n"
                )
                if bp:
                    keys = [
                        "mixed_freq",
                        "phase_limit",
                        "fdw_cycles",
                        "tdc_strength",
                        "tdc_max_reduction_db",
                        "tdc_slope_db_per_oct",
                        "reg_strength",
                        "max_slope_db_per_oct",
                        "max_boost",
                        "mag_c_min",
                        "mag_c_max",
                        "trans_width",
                        "filter_smooth",
                        "bass_first_mode_max_hz",
                        "low_bass_cut_hz",
                    ]
                    picked = [f"{k}={bp[k]}" for k in keys if k in bp]
                    if picked:
                        summary_content += "Best preset: " + ", ".join(picked) + "\n"
        except Exception:
            pass
    except Exception as e:
        summary_content += "\n=== DSP EFFECTIVE PARAMS (THIS SAMPLE RATE) ===\n"
        summary_content += f"Could not compute effective params: {type(e).__name__}: {e}\n"

    return summary_content

def _append_realized_phase_limit(summary_content: str, data, l_st, r_st) -> str:
    try:
        summary_content += "\n=== PHASE CORRECTION LIMIT (REALIZED) ===\n"
        cfg_lim = _safe_float((data or {}).get("phase_limit", float("nan")), float("nan"))
        if cfg_lim == cfg_lim and abs(cfg_lim) != float("inf") and cfg_lim > 0.0:
            summary_content += f"Configured phase_limit: {float(cfg_lim):.1f} Hz\n"

        for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
            st = st if isinstance(st, dict) else {}
            realized_hz = _pick_metric(
                st,
                (
                    "mixed_phase_no_correction_hz",
                    "linear_phase_blend_end_hz",
                    "phase_limit_hz",
                ),
                nonneg=True,
            )
            source = "n/a"
            for key in ("mixed_phase_no_correction_hz", "linear_phase_blend_end_hz", "phase_limit_hz"):
                v = _pick_metric(st, (key,), nonneg=True)
                if v is not None:
                    source = key
                    break

            if realized_hz is not None and float(realized_hz) > 0.0:
                summary_content += f"[{side}] Realized upper limit: {float(realized_hz):.1f} Hz ({source})\n"
            else:
                summary_content += f"[{side}] Realized upper limit: n/a\n"
    except Exception:
        pass
    return summary_content

def _append_acoustic_events(summary_content, l_st, r_st):
    for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
        reflections = st.get('reflections') or []
        if reflections:
            summary_content += f"\n=== ACOUSTIC EVENTS ({side}) ===\n"
            summary_content += (
                "Note: 'Path delta' is an equivalent path-length from dt.\n"
                "Reflections: time-of-flight equivalent extra path.\n"
                "Resonances: not a physical distance.\n"
            )
            summary_content += f"{'Freq (Hz)':<10} {'Type':<12} {'dt (ms)':<12} {'Path delta (m)':<14}\n"
            summary_content += "-" * 50 + "\n"
            for rev in reflections:
                freq = float(rev.get('freq', 0) or 0)
                ev_type = str(rev.get('type', 'Event') or 'Event')
                gd_error = float(rev.get('gd_error', 0) or 0)
                dist = float(rev.get('dist', 0) or 0)
                try:
                    et = ev_type.strip().lower()
                except Exception:
                    et = ""
                if "reson" in et:
                    summary_content += f"{freq:<10} {ev_type:<12} {gd_error:<12} {'n/a':<14}\n"
                else:
                    summary_content += f"{freq:<10} {ev_type:<12} {gd_error:<12} {dist:<14}\n"
        summary_content += f"\n=== HEADROOM MANAGEMENT ({side}) ===\n"
        summary_content += f"Auto Gain Margin: {float(st.get('gain_margin_db', 0.0)):.2f} dB\n"
        summary_content += f"Applied Auto Gain: {float(st.get('auto_global_gain_db', 0.0)):.2f} dB\n"
        summary_content += f"Normalize: {'ON' if bool(st.get('do_normalize', False)) else 'OFF'}\n"
        summary_content += f"Peak Gain (pre-headroom): {float(st.get('peak_gain_db', 0.0)):.2f} dB\n"
        summary_content += f"Applied Headroom: {float(st.get('auto_headroom_db', 0.0)):.2f} dB\n"
        summary_content += f"Final Max (filter+auto_gain+headroom): {float(st.get('final_max_db', 0.0)):.2f} dB\n"
        summary_content += f"\n=== BOOST/CUT DIAGNOSTICS ({side}) ===\n"
        _mb_eff = float(st.get('max_boost_db_effective', st.get('max_boost_db', 0.0)) or 0.0)
        _mb_user = float(st.get('max_boost_db_user', st.get('max_boost_db', 0.0)) or 0.0)
        _mb_cap = float(st.get('max_safe_boost_db', 0.0) or 0.0)
        if _mb_cap > 0.0 and _mb_user > _mb_eff + 1e-9:
            summary_content += f"Config: max_boost_db={_mb_eff:.2f} dB (user={_mb_user:.2f}, cap={_mb_cap:.2f}), "
        else:
            summary_content += f"Config: max_boost_db={_mb_eff:.2f} dB, "
        summary_content += f"max_cut_db={float(st.get('max_cut_db', 0.0)):.2f} dB\n"
        summary_content += f"Config: low_bass_cut_hz={float(st.get('low_bass_cut_hz', 0.0)):.1f} Hz, "
        summary_content += f"exc_prot={'ON' if bool(st.get('exc_prot', False)) else 'OFF'}, "
        summary_content += f"exc_freq={float(st.get('exc_freq', 0.0)):.1f} Hz, "
        summary_content += f"max_slope_db_per_oct={float(st.get('max_slope_db_per_oct', 0.0)):.1f}\n"
        summary_content += f"Result (post-clamp): boost_peak={float(st.get('boost_peak_db', 0.0)):.2f} dB, "
        summary_content += f"cut_peak={float(st.get('cut_peak_db', 0.0)):.2f} dB, "
        summary_content += f"boost_bins={int(st.get('boost_bins', 0))}\n"
        summary_content += f"Net boost peak (post global/headroom): {float(st.get('net_boost_peak_db', 0.0)):.2f} dB\n"
        summary_content += f"Candidate (pre-softclip): boost_peak={float(st.get('boost_candidate_peak_db', 0.0)):.2f} dB, "
        summary_content += f"boost_bins={int(st.get('boost_candidate_bins', 0))}, "
        summary_content += f"lowbass_boost_bins={int(st.get('boost_candidate_bins_lowbass', 0))}, "
        summary_content += f"excprot_boost_bins={int(st.get('boost_candidate_bins_excprot', 0))}\n"
        summary_content += f"Boost blocked reason: {str(st.get('boost_blocked_reason', 'n/a'))}\n"
        summary_content += f"\n=== CLAMP DIAGNOSTICS ({side}) ===\n"
        summary_content += f"{str(st.get('clamp_summary', 'n/a'))}\n"
        summary_content += (
            f"soft_clip: boost_bins={int(st.get('softclip_boost_bins', 0))}, "
            f"cut_bins={int(st.get('softclip_cut_bins', 0))}, "
            f"worst_over_boost={float(st.get('softclip_worst_over_boost_db', 0.0)):.2f} dB, "
            f"worst_over_cut={float(st.get('softclip_worst_over_cut_db', 0.0)):.2f} dB\n"
        )
        summary_content += (
            f"hard_clamp: boost_bins={int(st.get('hardclamp_boost_bins', 0))}, "
            f"cut_bins={int(st.get('hardclamp_cut_bins', 0))}, "
            f"worst_over_boost={float(st.get('hardclamp_worst_over_boost_db', 0.0)):.2f} dB, "
            f"worst_over_cut={float(st.get('hardclamp_worst_over_cut_db', 0.0)):.2f} dB\n"
        )


        probes = st.get("stage_probes") or {}
        if isinstance(probes, dict) and probes:
            summary_content += f"\n=== STAGE CHECKPOINTS ({side}) ===\n"
            summary_content += f"{'Stage':<22} {'BoostPk':>8} {'CutPk':>8} {'BoostBins':>10} {'CutBins':>8} {'NetBoostPk':>11}\n"
            summary_content += "-" * 75 + "\n"
            order = [
                "after_gain_apply",
                "after_lowbass_policy",
                "after_slope",
                "after_fade",
                "pre_softclip",
                "post_softclip",
                "post_hardclamp",
            ]
            for key in order:
                p = probes.get(key)
                if not isinstance(p, dict):
                    continue
                stage = str(p.get("stage", key))
                bpk = float(p.get("boost_peak_db", 0.0) or 0.0)
                cpk = float(p.get("cut_peak_db", 0.0) or 0.0)
                bb  = int(p.get("boost_bins", 0) or 0)
                cb  = int(p.get("cut_bins", 0) or 0)
                nbp = float(p.get("net_boost_peak_db", 0.0) or 0.0)
                summary_content += f"{stage:<22} {bpk:>8.2f} {cpk:>8.2f} {bb:>10d} {cb:>8d} {nbp:>11.2f}\n"

            summary_content += f"\n=== BASS-FIRST AI ({side}) ===\n"
            summary_content += f"Bass-first AI active: {'YES' if bool(st.get('bass_first_ai', False)) else 'NO'}\n"

            pk_hz = st.get('bass_first_mode_peak_hz', None)
            pk_sc = st.get('bass_first_mode_peak_score', None)
            if (pk_hz is not None) and (pk_sc is not None):
                summary_content += f"Mode peak: {float(pk_hz):.1f} Hz (score {float(pk_sc):.2f})\n"
            else:
                summary_content += "Mode peak: n/a\n"

            summary_content += f"Smoothing conf floor applied: {'YES' if bool(st.get('bass_first_conf_floor_applied', False)) else 'NO'}\n"

            rm_max = st.get('bass_first_roommode_max_20_200', None)
            rel_mean = st.get('bass_first_rel_mean_20_200', None)
            rel_min = st.get('bass_first_rel_min_20_200', None)
            conf_eff_mean = st.get('bass_first_conf_eff_mean_20_200', None)
            conf_eff_min = st.get('bass_first_conf_eff_min_20_200', None)
            floor_applied = bool(st.get('bass_first_conf_floor_applied', False))
            if (rm_max is not None) or (rel_mean is not None) or (rel_min is not None) or (conf_eff_mean is not None):
                summary_content += (
                    f"BF masks (20-200): "
                    f"roommode_max={float(rm_max or 0.0):.3f}, "
                    f"rel_mean(raw)={float(rel_mean or 0.0):.3f}, "
                    f"rel_min(raw)={float(rel_min or 0.0):.3f}, "
                    f"conf_eff_mean={float(conf_eff_mean or 0.0):.3f}, "
                    f"conf_eff_min={float(conf_eff_min or 0.0):.3f}, "
                    f"conf_floor_applied={'YES' if floor_applied else 'NO'}\n"
                )

            src = st.get("bass_first_source", None)
            if isinstance(src, str) and src.strip():
                summary_content += f"BassFirst source: {src.strip()}\n"


 

    return summary_content


def _extract_result_payload(result: FilterResult) -> tuple[Any, ...]:
    meas = dict(getattr(result, "measurements", {}) or {})
    return (
        int(result.fs),
        meas.get("f_l"),
        meas.get("m_l"),
        meas.get("p_l"),
        result.l_ir,
        result.l_st,
        meas.get("f_r"),
        meas.get("m_r"),
        meas.get("p_r"),
        result.r_ir,
        result.r_st,
    )


def _write_fs_outputs(
    zf,
    data,
    fs_v,
    ft_short,
    file_ts,
    f_l,
    m_l,
    p_l,
    l_imp,
    l_st,
    f_r,
    m_r,
    p_r,
    r_imp,
    r_st,
    *,
    result: FilterResult | None = None,
    write_dashboards: bool = True,
    irw_tag: str = "auto",
    ui_dashboards: dict | None = None,
    ranking_context: dict | None = None,
):
    if result is not None:
        (
            fs_v,
            f_l,
            m_l,
            p_l,
            l_imp,
            l_st,
            f_r,
            m_r,
            p_r,
            r_imp,
            r_st,
        ) = _extract_result_payload(result)

    sum_name = f"Summary_{ft_short}_{fs_v}Hz_{file_ts}.txt"
    l_dash_name = f"L_Dashboard_{ft_short}_{fs_v}Hz.png"
    r_dash_name = f"R_Dashboard_{ft_short}_{fs_v}Hz.png"

    summary_content = plots.format_summary_content(data, l_st, r_st)
    summary_content = _append_export_ranking(summary_content, int(fs_v), ranking_context)
    try:
        hc_src = str(data.get('hc_source', '') or '').strip()
        if hc_src:
            summary_content = f"House curve: {hc_src}\n" + summary_content
    except Exception:
        pass
    summary_content = _append_dsp_effective_params(summary_content, data, fs_v)
    summary_content = _append_realized_phase_limit(summary_content, data, l_st, r_st)
    try:
        summary_content += "\n=== LEVELING ===\n"
        for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
            if not isinstance(st, dict):
                continue
            summary_content += f"[{side}]\n"
            summary_content += f"Method: {st.get('offset_method', 'n/a')}\n"
            win = st.get("smart_scan_range", None)
            if isinstance(win, (list, tuple)) and len(win) >= 2:
                try:
                    summary_content += f"Window: {float(win[0]):.0f}-{float(win[1]):.0f} Hz\n"
                except Exception:
                    pass
            try:
                summary_content += f"Offset to measurement: {float(st.get('offset_db', 0.0) or 0.0):+.2f} dB\n"
            except Exception:
                pass
            try:
                summary_content += f"Effective target level: {float(st.get('eff_target_db', 0.0) or 0.0):.2f} dB\n"
            except Exception:
                pass
            tilt = st.get("tilt_slope_db_per_oct", None)
            if tilt is not None:
                try:
                    tilt_f = float(tilt)
                    summary_content += f"Tilt slope: {tilt_f:+.2f} dB/oct\n"
                    if abs(tilt_f) > 1.5:
                        summary_content += (
                            "Warning: Large broadband tilt detected. "
                            "May indicate measurement/target mismatch or strong room tilt.\n"
                        )
                except Exception:
                    pass
            summary_content += "\n"
    except Exception:
        pass

    summary_content = _append_acoustic_events(summary_content, l_st, r_st)



    if 'auto_align' in l_st:
        res = l_st['auto_align']
        summary_content += "\n=== AUTO-ALIGN ===\n"
        summary_content += f"Delay: {res['delay_ms']} ms\n"
        summary_content += f"Distance Diff: {res['distance_cm']} cm\n"
        summary_content += f"Gain Diff: {res['gain_diff_db']} dB\n"

    if TEST_MODE:
        try:
            diag = _build_diagnostics_dict(data, fs_v, l_st, r_st)
            summary_content += "\n\n--- DIAGNOSTICS_JSON_BEGIN ---\n"
            summary_content += json.dumps(_json_safe(diag), indent=2)
            summary_content += "\n--- DIAGNOSTICS_JSON_END ---\n"
        except Exception as e:
            summary_content += "\n\n--- DIAGNOSTICS_JSON_BEGIN ---\n"
            summary_content += json.dumps({
                "schema_version": 1,
                "error": f"diagnostics_json_failed: {type(e).__name__}: {e}"
            }, indent=2)
            summary_content += "\n--- DIAGNOSTICS_JSON_END ---\n"



    zf.writestr(sum_name, summary_content)

    if bool(write_dashboards):
        
        psl = data.get("plot_smoothing_level", "Psychoacoustic")

        html_l = plots.generate_prediction_plot(
            f_l,
            m_l,
            p_l, l_imp, fs_v, "Left",
            None, l_st, data['mixed_freq'], "low",
            create_full_html=False,
            plot_smoothing_level=psl,
        )
        if isinstance(ui_dashboards, dict):
            ui_dashboards["fs_hz"] = int(fs_v)
            ui_dashboards["left_html"] = str(html_l)
        png_l = plots.generate_combined_plot_mpl(
            f_l, m_l, p_l, l_imp, fs_v, "Left", target_stats=l_st
        )
        if png_l:
            zf.writestr(l_dash_name, png_l)
        else:
            zf.writestr(l_dash_name.replace(".png", ".txt"), str(html_l))

        html_r = plots.generate_prediction_plot(
            f_r,
            m_r,
            p_r, r_imp, fs_v, "Right",
            None, r_st, data['mixed_freq'], "low",
            create_full_html=False,
            plot_smoothing_level=psl,
        )
        if isinstance(ui_dashboards, dict):
            ui_dashboards["right_html"] = str(html_r)
        png_r = plots.generate_combined_plot_mpl(
            f_r, m_r, p_r, r_imp, fs_v, "Right", target_stats=r_st
        )
        if png_r:
            zf.writestr(r_dash_name, png_r)
        else:
            zf.writestr(r_dash_name.replace(".png", ".txt"), str(html_r))

    target_curve_tag = str(data.get("target_curve_tag", "") or "").strip()

    hlc_cfg = generate_hlc_config(
        fs_v,
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
    )
    zf.writestr(f"Config_{ft_short}_{fs_v}Hz_{irw_tag}.cfg", hlc_cfg)

    if not bool(data.get("multi_rate_opt", False)):
        yaml_content = generate_raspberry_yaml(
            fs_v,
            ft_short,
            file_ts,
            master_gain_db=0.0,
            irw_tag=irw_tag,
            target_curve_tag=target_curve_tag,
        )
        zf.writestr(f"camilladsp_{ft_short}_{fs_v}Hz_{irw_tag}.yml", yaml_content)


def build_export_zip(
    *,
    data: dict,
    results: list[FilterResult],
    ft_short: str,
    file_ts: str,
    irw_tag: str = "auto",
    write_dashboards: bool = False,
    dash_fs: int | None = None,
) -> tuple[io.BytesIO, dict, dict]:
    """
    Build full export ZIP from pipeline results.

    Returns:
    - zip_buffer: in-memory ZIP payload
    - ui_dashboards: optional dashboard HTML payloads
    - perf: {"zip_png_s": float, "per_fs_stats": {fs: {"zip_png_s": float}}}
    """
    zip_buffer = io.BytesIO()
    ui_dashboards: dict[str, Any] = {}
    perf = {"zip_png_s": 0.0, "per_fs_stats": {}}
    target_curve_tag = str(data.get("target_curve_tag", "") or "").strip()
    multi_rate_on = bool(data.get("multi_rate_opt", False))
    ranking_context = _build_export_ranking(results)

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for result in list(results or []):
            fs_v = int(result.fs)
            t0 = time.perf_counter()

            wav_l = io.BytesIO()
            wav_r = io.BytesIO()
            scipy.io.wavfile.write(wav_l, fs_v, result.l_ir.astype("float32"))
            scipy.io.wavfile.write(wav_r, fs_v, result.r_ir.astype("float32"))

            zf.writestr(
                f"L_{ft_short}_{fs_v}Hz_{target_curve_tag}_{file_ts}_{irw_tag}.wav",
                wav_l.getvalue(),
            )
            zf.writestr(
                f"R_{ft_short}_{fs_v}Hz_{target_curve_tag}_{file_ts}_{irw_tag}.wav",
                wav_r.getvalue(),
            )

            write_dash = bool(write_dashboards and ((not multi_rate_on) or (dash_fs is not None and int(fs_v) == int(dash_fs))))
            _write_fs_outputs(
                zf,
                data,
                fs_v,
                ft_short,
                file_ts,
                None,
                None,
                None,
                None,
                {},
                None,
                None,
                None,
                None,
                {},
                result=result,
                write_dashboards=write_dash,
                irw_tag=irw_tag,
                ui_dashboards=ui_dashboards if (dash_fs is not None and int(fs_v) == int(dash_fs)) else None,
                ranking_context=ranking_context,
            )

            dt = max(0.0, float(time.perf_counter() - t0))
            perf["zip_png_s"] = float(perf.get("zip_png_s", 0.0)) + dt
            slot = perf["per_fs_stats"].setdefault(int(fs_v), {})
            slot["zip_png_s"] = float(slot.get("zip_png_s", 0.0)) + dt

        if multi_rate_on:
            yaml_content = generate_raspberry_yaml(
                int(data.get("fs") or 44100),
                ft_short,
                file_ts,
                master_gain_db=0.0,
                irw_tag=irw_tag,
                target_curve_tag=target_curve_tag,
            )
            zf.writestr(f"camilladsp_{ft_short}_{irw_tag}.yml", yaml_content)

    return zip_buffer, ui_dashboards, perf


def save_export_bundle(
    zip_buffer: io.BytesIO,
    *,
    ft_short: str,
    irw_tag: str,
    target_curve_tag: str,
    ts: str,
    output_dir: str | None = None,
) -> tuple[str, str, str]:
    filters_dir = output_dir or os.path.join(os.getcwd(), "filters")
    os.makedirs(filters_dir, exist_ok=True)
    fname = f"CamillaFIR_{ft_short}_{irw_tag}_{target_curve_tag}_{ts}.zip"
    out_path = os.path.join(filters_dir, fname)

    try:
        with open(out_path, "wb") as f:
            f.write(zip_buffer.getvalue())
        save_msg = f"Saved: {os.path.abspath(out_path)}"
    except Exception:
        save_msg = "Zip saving failed."
    return fname, os.path.abspath(filters_dir), save_msg
