import sys

from ..dsp.smoothing import AFDW_BW_MAX_OCT, AFDW_BW_MIN_OCT
from ..auto_mode.rank_score import attach_official_rank_score, official_rank_score
from .export_scoring import _pick_metric, _safe_float

_AUTO_ASYM_PHASE1_SEARCH_SPACE_EST = 1877500016615829065655090169509480


def _auto_search_space_summary(data: dict | None) -> str:
    try:
        ft = str((data or {}).get("filter_type", "") or "").strip().lower()
    except Exception:
        ft = ""

    if "asym" in ft:
        return (
            "Theoretical preset search space (discretized estimate, Asymmetric phase1): "
            f"~{_AUTO_ASYM_PHASE1_SEARCH_SPACE_EST:.3e} combinations. "
            "Automatic mode samples only a tiny subset and refines iteratively/cache-guided. "
            "Because the calculation count is still large, runtime depends strongly on computer performance."
        )
    if "linear" in ft:
        return (
            "Theoretical preset search space is extremely large. "
            "Automatic mode samples only a tiny subset and refines iteratively/cache-guided. "
            "Because the calculation count is still large, runtime depends strongly on computer performance."
        )
    if "mixed" in ft or "minimum" in ft or "min" in ft:
        return (
            "Theoretical preset search space is extremely large. "
            "Automatic mode samples only a tiny subset and refines iteratively/cache-guided. "
            "Because the calculation count is still large, runtime depends strongly on computer performance."
        )
    return (
        "Theoretical preset search space is extremely large. "
        "Automatic mode samples only a tiny subset and refines iteratively/cache-guided. "
        "Because the calculation count is still large, runtime depends strongly on computer performance."
    )


def _module_runtime_version(module_name: str) -> str:
    try:
        mod = __import__(str(module_name))
        ver = str(getattr(mod, "__version__", "") or "").strip()
        if ver:
            return ver
    except Exception:
        pass
    try:
        from importlib.metadata import version

        ver = str(version(str(module_name)) or "").strip()
        if ver:
            return ver
    except Exception:
        pass
    return "n/a"


def _runtime_versions_text() -> str:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    parts = [
        f"Python {py_ver}",
        f"numpy {_module_runtime_version('numpy')}",
        f"scipy {_module_runtime_version('scipy')}",
        f"optuna {_module_runtime_version('optuna')}",
    ]
    return "Runtime: " + " | ".join(parts)


def _append_dsp_effective_params(summary_content, data, fs_v):
    try:
        enable_afdw = bool(data.get("enable_afdw", False))
        enable_tdc = bool(data.get("enable_tdc", False))
        tdc_strength = float(data.get("tdc_strength", 0.0) or 0.0)
        fdw_cycles = float(data.get("fdw_cycles", 15.0) or 15.0)
        fdw_oct_width = (2.0 / fdw_cycles) if fdw_cycles > 0 else 0.0
        afdw_min = max(3.0, fdw_cycles / 3.0)
        afdw_min_oct_width = (2.0 / afdw_min) if afdw_min > 0 else 0.0
        fdw_oct_width = float(max(AFDW_BW_MIN_OCT, min(AFDW_BW_MAX_OCT, fdw_oct_width)))
        afdw_min_oct_width = float(max(AFDW_BW_MIN_OCT, min(AFDW_BW_MAX_OCT, afdw_min_oct_width)))

        df_on = bool(data.get("df_smoothing", False))
        df_ref = 44100.0 / 65536.0
        fsmooth = data.get("filter_smooth", data.get("smoothing_level", 12))
        base_sigma = 60 // (fsmooth / 12 if (fsmooth or 0) > 0 else 1)
        sigma_hz = float(base_sigma) * df_ref
        df_cur = float(fs_v) / float(data.get("taps", 65536) or 65536)
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
            summary_content += f"TDC strength: {tdc_strength:.1f}% (base_strength = {tdc_strength / 100.0:.3f})\n"

        summary_content += f"DF smoothing: {'ON' if df_on else 'OFF'}\n"
        if df_on:
            summary_content += f"DF smoothing sigma: {sigma_bins:.1f} bins -> {sigma_hz:.2f} Hz\n"

        try:
            auto_meta = data.get("_auto_mode_meta", None)
            if bool(data.get("camillafir_automatic_mode", False)) and isinstance(auto_meta, dict):
                bm = attach_official_rank_score(auto_meta.get("best_metrics", {}))
                bp = dict(auto_meta.get("best_preset", {}) or {})
                tc = dict(data.get("_auto_target_curve_meta", {}) or {})
                best_rank = official_rank_score(bm)
                summary_content += "\n=== CAMILLAFIR AUTOMATIC MODE ===\n"
                summary_content += (
                    f"Trials: {int(auto_meta.get('trials_ok', 0))}/{int(auto_meta.get('trials_total', 0))} "
                    f"(search grid: {int(auto_meta.get('search_fs', 0))} Hz, {int(auto_meta.get('search_taps', 0))} taps)\n"
                )
                optimizer_backend = str(auto_meta.get("optimizer_backend", "") or "").strip()
                if optimizer_backend:
                    summary_content += f"Optimizer backend: {optimizer_backend}\n"
                phase_limit_polish = dict(auto_meta.get("phase_limit_winner_polish", {}) or {})
                if bool(phase_limit_polish.get("applicable", False)):
                    polish_start = _safe_float(phase_limit_polish.get("start_phase_limit_hz", float("nan")), float("nan"))
                    polish_final = _safe_float(phase_limit_polish.get("final_phase_limit_hz", float("nan")), float("nan"))
                    polish_rank_before = _safe_float(phase_limit_polish.get("rank_before", float("nan")), float("nan"))
                    polish_rank_after = _safe_float(phase_limit_polish.get("rank_after", float("nan")), float("nan"))
                    polish_tested = [float(v) for v in list(phase_limit_polish.get("tested_phase_limits_hz", []) or [])]
                    polish_tested_txt = ", ".join([f"{float(v):.1f}" for v in polish_tested]) if polish_tested else "n/a"
                    if bool(phase_limit_polish.get("applied", False)):
                        summary_content += (
                            f"Phase-limit winner polish: applied "
                            f"({float(polish_start):.1f} -> {float(polish_final):.1f} Hz, "
                            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
                            f"tested [{polish_tested_txt}] Hz)\n"
                        )
                    else:
                        summary_content += (
                            f"Phase-limit winner polish: tested, no change "
                            f"(kept {float(polish_final):.1f} Hz, tested [{polish_tested_txt}] Hz)\n"
                        )
                mag_c_min_polish = dict(auto_meta.get("mag_c_min_winner_polish", {}) or {})
                if bool(mag_c_min_polish.get("applicable", False)):
                    polish_start = _safe_float(mag_c_min_polish.get("start_mag_c_min_hz", float("nan")), float("nan"))
                    polish_final = _safe_float(mag_c_min_polish.get("final_mag_c_min_hz", float("nan")), float("nan"))
                    polish_rank_before = _safe_float(mag_c_min_polish.get("rank_before", float("nan")), float("nan"))
                    polish_rank_after = _safe_float(mag_c_min_polish.get("rank_after", float("nan")), float("nan"))
                    polish_tested = [float(v) for v in list(mag_c_min_polish.get("tested_mag_c_min_hz", []) or [])]
                    polish_tested_txt = ", ".join([f"{float(v):.1f}" for v in polish_tested]) if polish_tested else "n/a"
                    if bool(mag_c_min_polish.get("applied", False)):
                        summary_content += (
                            f"Mag-c-min winner polish: applied "
                            f"({float(polish_start):.1f} -> {float(polish_final):.1f} Hz, "
                            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
                            f"tested [{polish_tested_txt}] Hz)\n"
                        )
                    else:
                        summary_content += (
                            f"Mag-c-min winner polish: tested, no change "
                            f"(kept {float(polish_final):.1f} Hz, tested [{polish_tested_txt}] Hz)\n"
                        )
                summary_content += _runtime_versions_text() + "\n"
                summary_content += _auto_search_space_summary(data) + "\n"
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
                            bm_t = attach_official_rank_score(row.get("best_metrics", {}))
                            summary_content += (
                                f"Target #{i}: {str(row.get('hc_mode', 'n/a'))} "
                                f"(best_rank={official_rank_score(bm_t):.3f}, "
                                f"avg_rank={float(row.get('avg_rank_score', 0.0)):.3f}, "
                                f"ok={int(row.get('trials_ok', 0))}/{int(row.get('trials_total', 0))})\n"
                            )
                summary_content += (
                    f"Best rank score: {best_rank:.3f}/100 "
                    f"(avg={float(bm.get('avg_score', 0.0)):.3f}, "
                    f"dsp_pen={float(bm.get('dsp_penalty', 0.0)):.2f}, "
                    f"exc_pen={float(bm.get('exc_penalty', 0.0)):.2f}, "
                    f"max_net_boost={float(bm.get('max_net_boost_db', 0.0)):.2f} dB, "
                    f"events={int(bm.get('events_total', 0))}, "
                    f"event_sev={float(bm.get('events_severity', 0.0)):.2f})\n"
                )
                exc_seed_hz = _safe_float(
                    auto_meta.get(
                        "auto_exc_seed_freq_hz",
                        data.get("_auto_exc_seed_freq_hz", data.get("_auto_exc_freq_hz", float("nan"))),
                    ),
                    float("nan"),
                )
                exc_final_hz = _safe_float(
                    auto_meta.get(
                        "best_auto_exc_freq_hz",
                        data.get("_auto_exc_freq_hz", data.get("exc_freq", float("nan"))),
                    ),
                    float("nan"),
                )
                seed_ok = (exc_seed_hz == exc_seed_hz) and (abs(exc_seed_hz) != float("inf"))
                final_ok = (exc_final_hz == exc_final_hz) and (abs(exc_final_hz) != float("inf"))
                if seed_ok and final_ok:
                    summary_content += (
                        f"Excursion protection (AUTO): seed {float(exc_seed_hz):.1f} Hz -> "
                        f"final {float(exc_final_hz):.1f} Hz\n"
                    )
                elif final_ok:
                    summary_content += f"Excursion protection (AUTO): final {float(exc_final_hz):.1f} Hz\n"
                elif seed_ok:
                    summary_content += f"Excursion protection (AUTO): seed {float(exc_seed_hz):.1f} Hz\n"
                if bp:
                    filter_type = str(bp.get("filter_type", data.get("filter_type", "")) or "").strip().lower()
                    keys = [
                        "fdw_cycles",
                        "tdc_strength",
                        "tdc_max_reduction_db",
                        "tdc_slope_db_per_oct",
                        "reg_strength",
                        "max_slope_db_per_oct",
                        "max_slope_boost_db_per_oct",
                        "max_slope_cut_db_per_oct",
                        "max_boost",
                        "mag_c_min",
                        "mag_c_max",
                        "trans_width",
                        "filter_smooth",
                        "bass_first_mode_max_hz",
                        "conf_pull_max_hz",
                        "low_bass_cut_hz",
                    ]
                    if filter_type == "mixed":
                        keys.insert(0, "mixed_freq")
                    elif filter_type in ("linear", "asym", "asymmetric"):
                        keys.insert(0, "phase_limit")
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
        reflections = st.get("reflections") or []
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
                freq = float(rev.get("freq", 0) or 0)
                ev_type = str(rev.get("type", "Event") or "Event")
                gd_error = float(rev.get("gd_error", 0) or 0)
                dist = float(rev.get("dist", 0) or 0)
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
        _mb_eff = float(st.get("max_boost_db_effective", st.get("max_boost_db", 0.0)) or 0.0)
        _mb_user = float(st.get("max_boost_db_user", st.get("max_boost_db", 0.0)) or 0.0)
        _mb_cap = float(st.get("max_safe_boost_db", 0.0) or 0.0)
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
                bb = int(p.get("boost_bins", 0) or 0)
                cb = int(p.get("cut_bins", 0) or 0)
                nbp = float(p.get("net_boost_peak_db", 0.0) or 0.0)
                summary_content += f"{stage:<22} {bpk:>8.2f} {cpk:>8.2f} {bb:>10d} {cb:>8d} {nbp:>11.2f}\n"

            summary_content += f"\n=== BASS-FIRST AI ({side}) ===\n"
            summary_content += f"Bass-first AI active: {'YES' if bool(st.get('bass_first_ai', False)) else 'NO'}\n"

            pk_hz = st.get("bass_first_mode_peak_hz", None)
            pk_sc = st.get("bass_first_mode_peak_score", None)
            if (pk_hz is not None) and (pk_sc is not None):
                summary_content += f"Mode peak: {float(pk_hz):.1f} Hz (score {float(pk_sc):.2f})\n"
            else:
                summary_content += "Mode peak: n/a\n"

            summary_content += f"Smoothing conf floor applied: {'YES' if bool(st.get('bass_first_conf_floor_applied', False)) else 'NO'}\n"

            rm_max = st.get("bass_first_roommode_max_20_200", None)
            rel_mean = st.get("bass_first_rel_mean_20_200", None)
            rel_min = st.get("bass_first_rel_min_20_200", None)
            conf_eff_mean = st.get("bass_first_conf_eff_mean_20_200", None)
            conf_eff_min = st.get("bass_first_conf_eff_min_20_200", None)
            floor_applied = bool(st.get("bass_first_conf_floor_applied", False))
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
