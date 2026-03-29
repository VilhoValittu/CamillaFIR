"""Target-curve selection orchestration for auto mode."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .cache_signature import (
    _auto_apply_seed,
    _auto_compat_version,
    _auto_seed_from_signature,
    _auto_signature,
)
from .candidate_generation import (
    _build_auto_mode_candidates,
    _build_auto_mode_candidates_local,
    _seed_auto_mode_candidate_local_optuna_params,
    _seed_auto_mode_candidate_optuna_params,
    _suggest_auto_mode_candidate_local_optuna,
)
from .scoring_ranking import (
    _auto_adaptive_shrink_factor,
    _auto_goal_uses_local_refine,
    _auto_rank_key,
    _auto_select_best_scored,
    _auto_target_result_mode_ripple,
    _auto_target_result_rank_key,
    _tc_score,
)
from .runtime_context import coerce_orchestrator_runtime
from .shared import (
    AutoModeConfig,
    AUTO_MODE_CACHE_ENABLED,
    AUTO_MODE_LOCAL_REFINE_ENABLED,
    AUTO_MODE_LOCAL_REFINE_TOP_K,
    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
    AUTO_MODE_LOCAL_REFINEMENT_SHRINK,
    AUTO_MODE_SYNTH_TARGET_NAME,
    AUTO_MODE_SYNTH_TARGET_BASS_COMP_FRAC,
    AUTO_MODE_SYNTH_TARGET_BASS_COMP_REF_DB,
    AUTO_MODE_SYNTH_TARGET_TILT_COMP_FRAC,
    AUTO_MODE_SYNTH_TARGET_HF_COMP_FRAC,
    AUTO_MODE_SYNTH_TARGET_SMOOTH_OCT,
    AUTO_MODE_TARGET_BEST_RANK_TIE_EPS,
    AUTO_MODE_TARGET_CACHE_AS_WILDCARD,
    AUTO_MODE_TARGET_MILDER_MAX_ASYM_ADD,
    AUTO_MODE_TARGET_MILDER_MAX_DIFFICULTY_ADD,
    AUTO_MODE_TARGET_MILDER_MAX_FIT_RMS_ADD_DB,
    AUTO_MODE_TARGET_PREFER_MILDER_STEP,
    AUTO_MODE_TARGET_TOP_N,
    AUTO_MODE_TARGET_TOP_N_SPREAD_DB,
    AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    MAX_SAFE_BOOST,
    _auto_builtin_target_name,
    _auto_filter_cache_key,
    _auto_goal,
    _auto_goal_basis_text,
    _auto_metric_text,
    _auto_optimizer_backend,
    _auto_phase_limit_center,
    _auto_phase_limit_clip,
    _auto_safe_bool,
    _auto_safe_float,
    _auto_trial_chunk_size,
    _clip,
)
from .target_preselection import (
    _auto_target_adaptive_shortlist,
    _auto_target_insert_cached_wildcard,
    _auto_target_one_step_milder,
)
from ..dsp.target_synthesis import synthesize_target_from_measurements

logger = logging.getLogger("CamillaFIR")
__all__ = ["select_target_curve_with_trials"]


def select_target_curve_with_trials(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb=None,
    top_n: int = AUTO_MODE_TARGET_TOP_N,
    trials_per_curve: int = AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    runtime=None,
) -> dict | None:
    return _select_target_curve_with_trials_impl(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        pin_obj=pin_obj,
        status_cb=status_cb,
        top_n=int(top_n),
        trials_per_curve=int(trials_per_curve),
        runtime=runtime,
    )


def _cache_target_valid(runtime, hc_name: str | None) -> bool:
    if str(hc_name or "").strip() == AUTO_MODE_SYNTH_TARGET_NAME:
        return False  # synthesized curves are measurement-specific, never cache
    hc = _auto_builtin_target_name(hc_name)
    if not hc:
        return False
    try:
        c_f, c_m = runtime.get_house_curve_by_name(hc)
        c_f = np.asarray(c_f, dtype=float).reshape(-1)
        c_m = np.asarray(c_m, dtype=float).reshape(-1)
        return bool(c_f.size >= 4 and c_m.size == c_f.size)
    except Exception:
        # Cache validation must stay resilient to malformed or missing house-curve data.
        return False


def _cached_target_fit(
    *,
    runtime,
    base_data: dict,
    measurements: dict,
    hc_name: str,
) -> tuple[float, float]:
    fit_rms_db = float("nan")
    offset_db = 0.0
    try:
        quick_fit = runtime.auto_select_builtin_target_curve(
            dict(base_data or {}),
            f_l=measurements.get("f_l"),
            m_l=measurements.get("m_l"),
            f_r=measurements.get("f_r"),
            m_r=measurements.get("m_r"),
        )
        candidates = list(
            (quick_fit or {}).get(
                "candidates_all",
                (quick_fit or {}).get("candidates", []),
            )
            or []
        )
        target_candidate = None
        for cand in candidates:
            if str((cand or {}).get("hc_mode", "") or "").strip() == str(hc_name):
                target_candidate = dict(cand or {})
                break
        if isinstance(target_candidate, dict):
            fit_rms_db = float(
                _auto_safe_float(
                    target_candidate.get("fit_rms_db", float("nan")),
                    float("nan"),
                )
            )
            offset_db = float(
                _auto_safe_float(target_candidate.get("offset_db", 0.0), 0.0)
            )
    except Exception:
        # Cached targets are best-effort metadata and should not abort target selection.
        pass
    return float(fit_rms_db), float(offset_db)


def _cached_target_return(
    *,
    runtime,
    cached_hc_mode: str | None,
    cached_preset: dict,
    selection_method: str,
    base_data: dict,
    measurements: dict,
    goal: str,
    rank_basis: str,
) -> dict | None:
    if not _cache_target_valid(runtime, cached_hc_mode):
        return None
    fit_rms_db, offset_db = _cached_target_fit(
        runtime=runtime,
        base_data=base_data,
        measurements=measurements,
        hc_name=str(cached_hc_mode),
    )
    return {
        "selected_hc_mode": str(cached_hc_mode),
        "fit_rms_db": float(fit_rms_db),
        "offset_db": float(offset_db),
        "selection_method": str(selection_method),
        "selection_basis": str(rank_basis),
        "auto_goal": str(goal),
        "top_n": 0,
        "trials_per_curve": 0,
        "candidates": [],
        "evaluated": [],
        "best_preset": dict(cached_preset or {}),
    }


def _target_eval_one(
    *,
    runtime,
    preset: dict,
    base_tc: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f_arr,
    hc_m_arr,
    pin_obj,
    filter_key: str,
) -> dict:
    trial_data = dict(base_tc)
    trial_data.update(dict(preset or {}))
    if str(filter_key) in ("linear", "asym"):
        trial_data["phase_limit"] = round(
            float(
                _auto_phase_limit_clip(
                    trial_data.get("phase_limit", base_tc.get("phase_limit", 400.0)),
                    default=400.0,
                )
            ),
            1,
        )
    trial_data["comparison_mode"] = True
    trial_measurements = dict(measurements or {})
    trial_measurements["ui_data"] = trial_data

    cfg = runtime.build_config(
        trial_data,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_f=hc_f_arr,
        hc_m=hc_m_arr,
        max_safe_boost=float(MAX_SAFE_BOOST),
    )
    try:
        setattr(
            cfg,
            "bass_smooth_w_gamma",
            float(trial_data.get("bass_smooth_w_gamma", 2.40)),
        )
        setattr(
            cfg,
            "bass_smooth_w_max",
            float(trial_data.get("bass_smooth_w_max", 0.45)),
        )
    except Exception:
        # Optional trial-only smoothing knobs may be absent on older config objects.
        pass

    result = runtime.run_pipeline(
        cfg,
        trial_measurements,
        include_response_arrays=False,
    )
    metrics = runtime.auto_score_result(
        result,
        auto_exc_freq_hz=_auto_safe_float(
            trial_data.get("_auto_exc_freq_hz", float("nan")),
            float("nan"),
        ),
        base_data=trial_data,
    )
    trial_preset = dict(preset or {})
    if str(filter_key) == "mixed":
        trial_preset["mixed_freq"] = round(
            _clip(
                trial_data.get("mixed_freq", base_tc.get("mixed_freq", 180.0)),
                80.0,
                320.0,
            ),
            1,
        )
    elif str(filter_key) in ("linear", "asym"):
        trial_preset["phase_limit"] = round(
            float(
                _auto_phase_limit_clip(
                    trial_data.get("phase_limit", base_tc.get("phase_limit", 400.0)),
                    default=400.0,
                )
            ),
            1,
        )
    return {
        "ok": True,
        "metrics": dict(metrics or {}),
        "preset": dict(trial_preset),
    }


def _run_target_trials(
    *,
    runtime,
    cfg,
    optimizer_backend: str,
    optuna_mod,
    target_study_sig: str,
    seed_target: int,
    cands: list[dict],
    base_tc: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f_arr,
    hc_m_arr,
    pin_obj,
    filter_key: str,
    phase_tag: str,
    target_name: str,
    phase_kind: str | None = None,
    n_total_override: int | None = None,
    seed_presets: list[dict] | None = None,
    optuna_builder=None,
    seed_to_params=None,
) -> list[dict]:
    use_optuna_trials = bool(
        str(optimizer_backend) == "optuna"
        and runtime.auto_optuna_module_ready(optuna_mod)
        and callable(optuna_builder)
    )
    n_total = int(n_total_override) if n_total_override is not None else int(len(cands))
    if n_total <= 0:
        return []
    workers = int(runtime.auto_trial_workers(base_tc, n_total))
    if workers > 1:
        logger.info(
            "Automatic mode target trials: target=%s, phase=%s, parallel workers=%d",
            str(target_name),
            str(phase_tag),
            int(workers),
        )

    if bool(use_optuna_trials):
        out_by_idx: dict[int, dict] = {}
        raw_scope = f"target-{str(target_name)}-{str(phase_tag)}"
        scope_eff = runtime.auto_optuna_effective_scope(
            base_tc,
            raw_scope,
            phase_kind=phase_kind,
        )
        study_name = runtime.auto_optuna_study_name(
            study_sig=target_study_sig,
            scope=scope_eff,
        )

        def _eval_one(idx: int, preset: dict) -> dict:
            out = _target_eval_one(
                runtime=runtime,
                preset=dict(preset or {}),
                base_tc=base_tc,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_f_arr=hc_f_arr,
                hc_m_arr=hc_m_arr,
                pin_obj=pin_obj,
                filter_key=filter_key,
            )
            out = dict(out or {})
            out["idx"] = int(idx)
            return out

        def _consume_one(idx: int, out: dict) -> bool:
            out_by_idx[int(idx)] = dict(out or {})
            return False

        runtime.auto_run_optuna_eval_loop(
            optuna_mod=optuna_mod,
            cfg=cfg,
            n_total=int(n_total),
            seed=int(
                seed_target
                + sum(ord(ch) for ch in str(target_name)) * 31
                + sum(ord(ch) for ch in str(phase_tag)) * 17
            ),
            base_data=dict(base_tc or {}),
            seed_presets=list(seed_presets or []),
            build_preset=optuna_builder,
            eval_one=_eval_one,
            consume_one=_consume_one,
            objective_value=lambda out: runtime.auto_optuna_objective_value(
                dict((out or {}).get("metrics", {}) or {}),
                use_refine_tiebreak=False,
            ),
            workers=int(workers),
            seed_to_params=seed_to_params,
            study_name=study_name,
            study_scope=raw_scope,
            phase_label=f"target {str(target_name)} {str(phase_tag)}",
            phase_kind=phase_kind,
        )
        return [
            dict(
                out_by_idx.get(
                    int(idx),
                    {"idx": int(idx), "ok": False, "error": "missing worker result"},
                )
                or {}
            )
            for idx in range(1, int(n_total) + 1)
        ]

    idx_presets = list(enumerate(list(cands or []), start=1))
    out_by_idx: dict[int, dict] = {}
    if workers <= 1 or n_total <= 1:
        for idx, preset in idx_presets:
            try:
                out = _target_eval_one(
                    runtime=runtime,
                    preset=dict(preset or {}),
                    base_tc=base_tc,
                    measurements=measurements,
                    fs_v=int(fs_v),
                    taps_v=int(taps_v),
                    xos=xos,
                    hpf=hpf,
                    hc_f_arr=hc_f_arr,
                    hc_m_arr=hc_m_arr,
                    pin_obj=pin_obj,
                    filter_key=filter_key,
                )
            except Exception as exc:
                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            out_by_idx[int(idx)] = dict(out or {})
    else:
        chunk_size = int(_auto_trial_chunk_size(workers))
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            for c0 in range(0, int(len(idx_presets)), int(chunk_size)):
                chunk = idx_presets[c0 : c0 + int(chunk_size)]
                future_map = {
                    executor.submit(
                        _target_eval_one,
                        runtime=runtime,
                        preset=dict(preset or {}),
                        base_tc=base_tc,
                        measurements=measurements,
                        fs_v=int(fs_v),
                        taps_v=int(taps_v),
                        xos=xos,
                        hpf=hpf,
                        hc_f_arr=hc_f_arr,
                        hc_m_arr=hc_m_arr,
                        pin_obj=pin_obj,
                        filter_key=filter_key,
                    ): int(idx)
                    for idx, preset in chunk
                }
                for future in as_completed(list(future_map.keys())):
                    idx = int(future_map.get(future, 0))
                    try:
                        out = future.result()
                        if not isinstance(out, dict):
                            out = {"ok": False, "error": "invalid worker result"}
                    except Exception as exc:
                        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    out_by_idx[int(idx)] = dict(out or {})

    return [
        dict(
            out_by_idx.get(
                int(idx),
                {"ok": False, "error": "missing worker result"},
            )
            or {}
        )
        for idx, _preset in idx_presets
    ]


def _evaluate_target_curve(
    *,
    runtime,
    cfg,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    goal: str,
    filter_key: str,
    optimizer_backend: str,
    optuna_mod,
    seed_target: int,
    target_study_sig: str,
    trials_eff: int,
    shortlisted: list[dict],
    status_cb,
    f6_txt: str,
    tc: dict,
    t_idx: int,
    emit_status: bool,
    curve_inner_workers: int | None,
) -> dict | None:
    hc_name = str(tc.get("hc_mode", "") or "").strip()
    if not hc_name:
        return None
    try:
        if "_synth_hc_f" in tc and "_synth_hc_m" in tc:
            hc_f = np.asarray(tc["_synth_hc_f"], dtype=float)
            hc_m = np.asarray(tc["_synth_hc_m"], dtype=float)
        else:
            hc_f, hc_m = runtime.get_house_curve_by_name(hc_name)
            hc_f = np.asarray(hc_f, dtype=float)
            hc_m = np.asarray(hc_m, dtype=float)
    except Exception:
        return None
    if hc_f.size < 4 or hc_m.size != hc_f.size:
        return None

    seed_tc = int(
        (
            _auto_seed_from_signature(
                base_data=base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=str(hc_name),
                include_hc_mode=True,
            )
            + sum(ord(ch) for ch in hc_name) * 13
        )
        & 0xFFFFFFFF
    )
    if bool(emit_status):
        _auto_apply_seed(seed_tc)
    base_tc = dict(base_data or {})
    base_tc["hc_mode"] = str(hc_name)
    if isinstance(curve_inner_workers, int) and int(curve_inner_workers) > 0:
        base_tc["auto_mode_workers"] = int(curve_inner_workers)
    if str(filter_key) in ("linear", "asym"):
        base_tc["phase_limit"] = round(
            float(_auto_phase_limit_center(base_tc.get("phase_limit", None))),
            1,
        )
    use_optuna_curve_trials = bool(
        str(optimizer_backend) == "optuna"
        and runtime.auto_optuna_module_ready(optuna_mod)
    )
    candidates = []
    phase1_seed_presets = _build_auto_mode_candidates(
        base_tc,
        n_trials=1,
        seed=seed_tc,
        optimize_mag_low=False,
    )
    if not bool(use_optuna_curve_trials):
        candidates = _build_auto_mode_candidates(
            base_tc,
            n_trials=trials_eff,
            seed=seed_tc,
            optimize_mag_low=False,
        )

    best_metrics = None
    best_preset = None
    ok_n = 0
    rank_sum = 0.0
    avg_score_sum = 0.0
    phase1_scored = []
    curve_scored = []
    phase1_trial_total = int(
        max(1, trials_eff if bool(use_optuna_curve_trials) else len(candidates))
    )
    trials_total_count = int(phase1_trial_total)
    cb = status_cb if bool(emit_status) else None

    phase1_out = _run_target_trials(
        runtime=runtime,
        cfg=cfg,
        optimizer_backend=optimizer_backend,
        optuna_mod=optuna_mod,
        target_study_sig=target_study_sig,
        seed_target=seed_target,
        cands=candidates,
        base_tc=base_tc,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_f_arr=hc_f,
        hc_m_arr=hc_m,
        pin_obj=pin_obj,
        filter_key=filter_key,
        phase_tag="phase1",
        target_name=hc_name,
        phase_kind="target",
        n_total_override=int(phase1_trial_total),
        seed_presets=list(phase1_seed_presets or []),
        optuna_builder=(
            (
                lambda tr, _base_tc=dict(base_tc): runtime.suggest_auto_mode_candidate_optuna(
                    _base_tc,
                    tr,
                    optimize_mag_low=False,
                )
            )
            if bool(use_optuna_curve_trials)
            else None
        ),
        seed_to_params=(
            (
                lambda preset, _base_tc=dict(base_tc): _seed_auto_mode_candidate_optuna_params(
                    _base_tc,
                    preset,
                    optimize_mag_low=False,
                )
            )
            if bool(use_optuna_curve_trials)
            else None
        ),
    )
    for c_idx, out in enumerate(phase1_out, start=1):
        improved = False
        if bool(out.get("ok", False)):
            met = dict(out.get("metrics", {}) or {})
            trial_preset = dict(out.get("preset", {}) or {})
            ok_n += 1
            rank_sum += _auto_safe_float(met.get("rank_score"), 0.0)
            avg_score_sum += _auto_safe_float(met.get("avg_score"), 0.0)
            phase1_scored.append({"metrics": dict(met), "preset": dict(trial_preset)})
            curve_scored.append({"metrics": dict(met), "preset": dict(trial_preset)})
            if best_metrics is None or _auto_rank_key(met) < _auto_rank_key(best_metrics):
                best_metrics = dict(met)
                best_preset = dict(trial_preset)
                improved = True
        else:
            logger.warning(
                "Automatic mode target trial failed: target=%s %d/%d (%s)",
                str(hc_name),
                int(c_idx),
                int(phase1_trial_total),
                str(out.get("error", "unknown error") or "unknown error"),
            )

        if callable(cb):
            rank_now = _auto_safe_float((best_metrics or {}).get("rank_score"), 0.0)
            if bool(improved):
                avg_now = _auto_safe_float((best_metrics or {}).get("avg_score"), 0.0)
                cb(
                    "CamillaFIR automatic mode: target trials best improved "
                    f"(target {t_idx}/{len(shortlisted)} {hc_name}, "
                    f"trial {c_idx}/{int(phase1_trial_total)}{f6_txt}, goal {goal}, "
                    f"rank {rank_now:.3f}, avg {avg_now:.3f}, "
                    f"fit {_auto_safe_float(tc.get('fit_rms_db', 0.0), 0.0):.3f}, "
                    f"pre {_auto_safe_float(tc.get('preselect_score', tc.get('fit_rms_db', 0.0)), 0.0):.3f})"
                )
            elif best_metrics is not None:
                cb(
                    f"CamillaFIR automatic mode: target trials "
                    f"(target {t_idx}/{len(shortlisted)} {hc_name}, "
                    f"trial {c_idx}/{int(phase1_trial_total)}, rank {rank_now:.3f})"
                )

    if (
        phase1_scored
        and bool(AUTO_MODE_LOCAL_REFINE_ENABLED)
        and _auto_goal_uses_local_refine(goal)
    ):
        from .scoring_ranking import _auto_build_refine_profile

        top_list = sorted(
            list(phase1_scored),
            key=lambda it: _auto_rank_key(dict(it.get("metrics", {}) or {})),
        )[: int(max(1, AUTO_MODE_LOCAL_REFINE_TOP_K))]
        ref_profile = _auto_build_refine_profile(
            base_data=base_tc,
            phase1_top=top_list,
        )
        phase1_best = dict(_auto_select_best_scored(top_list) or top_list[0])
        p1m = dict(phase1_best.get("metrics", {}) or {})
        p1p = dict(phase1_best.get("preset", {}) or {})
        p1_mixed = _auto_safe_float(
            p1p.get("mixed_freq", base_tc.get("mixed_freq", float("nan"))),
            float("nan"),
        )
        p1_phase = _auto_safe_float(
            p1p.get("phase_limit", base_tc.get("phase_limit", float("nan"))),
            float("nan"),
        )
        p1_tdc = _auto_safe_float(
            p1p.get("tdc_strength", base_tc.get("tdc_strength", float("nan"))),
            float("nan"),
        )
        p1_mode = _auto_safe_float(p1m.get("mode_ripple_db"), float("nan"))
        p1_boost = _auto_safe_float(p1m.get("max_net_boost_db"), float("nan"))
        p1_mode_txt = f"{p1_mode:.3f} dB" if np.isfinite(p1_mode) else "n/a"
        p1_boost_txt = f"{p1_boost:.2f} dB" if np.isfinite(p1_boost) else "n/a"
        if str(filter_key) == "mixed":
            p1_detail = f"mixed_freq={p1_mixed:.1f} Hz, tdc={p1_tdc:.1f}"
        elif str(filter_key) in ("linear", "asym"):
            p1_detail = (
                f"phase_limit={p1_phase:.1f} Hz, tdc={p1_tdc:.1f}"
                if np.isfinite(p1_phase)
                else f"phase_limit=n/a, tdc={p1_tdc:.1f}"
            )
        else:
            p1_detail = f"tdc={p1_tdc:.1f}"
        logger.info(
            "Automatic mode target Phase1 done: target=%s, avg_score=%.3f, %s",
            str(hc_name),
            _auto_safe_float(p1m.get("avg_score"), 0.0),
            str(p1_detail),
        )
        if callable(cb):
            cb(
                "CamillaFIR automatic mode: Phase1 done "
                f"target={hc_name}, rank={_auto_safe_float(p1m.get('rank_score'), 0.0):.3f}, "
                f"avg_score={_auto_safe_float(p1m.get('avg_score'), 0.0):.3f}, "
                f"mode_ripple={p1_mode_txt}, boost={p1_boost_txt}, {p1_detail}"
            )

        for li, item in enumerate(top_list, start=1):
            center = dict(item.get("preset", {}) or {})
            c_mixed = _auto_safe_float(
                center.get("mixed_freq", base_tc.get("mixed_freq", float("nan"))),
                float("nan"),
            )
            c_phase = _auto_safe_float(
                center.get("phase_limit", base_tc.get("phase_limit", float("nan"))),
                float("nan"),
            )
            local_detail = None
            if str(filter_key) == "mixed":
                local_detail = f"mixed_freq={c_mixed:.1f} Hz"
            elif str(filter_key) in ("linear", "asym"):
                local_detail = (
                    f"phase refine phase_limit={c_phase:.1f} Hz"
                    if np.isfinite(c_phase)
                    else "phase refine phase_limit=n/a"
                )
            if local_detail is not None:
                logger.info(
                    "Automatic mode target Local refine: target=%s, center #%d, %s",
                    str(hc_name),
                    int(li),
                    str(local_detail),
                )
                if callable(cb):
                    cb(
                        f"CamillaFIR automatic mode: Local refine target={hc_name} "
                        f"center #{li} {local_detail}"
                    )

            local_trial_total = int(AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP)
            local_shrink = float(
                _auto_adaptive_shrink_factor(
                    top_list,
                    base_shrink=float(AUTO_MODE_LOCAL_REFINEMENT_SHRINK),
                    plateau_hit=False,
                )
            )

            def _target_local_candidate_clip(cand_in: dict) -> dict:
                cand = dict(cand_in or {})
                mf = _auto_safe_float(cand.get("mixed_freq"), float("nan"))
                if np.isfinite(mf):
                    cand["mixed_freq"] = _clip(
                        mf,
                        ref_profile["mixed_center"] - ref_profile["mixed_span"],
                        ref_profile["mixed_center"] + ref_profile["mixed_span"],
                    )
                td = _auto_safe_float(cand.get("tdc_strength"), float("nan"))
                if np.isfinite(td):
                    cand["tdc_strength"] = _clip(
                        td,
                        ref_profile["tdc_lo"],
                        ref_profile["tdc_hi"],
                    )
                if str(filter_key) in ("linear", "asym"):
                    cand["phase_limit"] = round(
                        float(
                            _auto_phase_limit_clip(
                                cand.get("phase_limit", base_tc.get("phase_limit", 400.0)),
                                default=400.0,
                            )
                        ),
                        1,
                    )
                return dict(cand)

            local_seed_presets = [
                _target_local_candidate_clip(c)
                for c in _build_auto_mode_candidates_local(
                    base_tc,
                    center,
                    1,
                    int(seed_tc + li * 100003),
                    shrink=float(local_shrink),
                    optimize_mag_low=False,
                )
            ]
            local_candidates = []
            if not bool(use_optuna_curve_trials):
                local_candidates = [
                    _target_local_candidate_clip(c)
                    for c in _build_auto_mode_candidates_local(
                        base_tc,
                        center,
                        int(local_trial_total),
                        int(seed_tc + li * 100003),
                        shrink=float(local_shrink),
                        optimize_mag_low=False,
                    )
                ]
            trials_total_count += int(local_trial_total)
            local_out = _run_target_trials(
                runtime=runtime,
                cfg=cfg,
                optimizer_backend=optimizer_backend,
                optuna_mod=optuna_mod,
                target_study_sig=target_study_sig,
                seed_target=seed_target,
                cands=local_candidates,
                base_tc=base_tc,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_f_arr=hc_f,
                hc_m_arr=hc_m,
                pin_obj=pin_obj,
                filter_key=filter_key,
                phase_tag=runtime.auto_optuna_scope_with_context(
                    f"local_center_{li}_u1",
                    center=dict(center or {}),
                    shrink=float(local_shrink),
                    extra={
                        "filter_key": str(filter_key),
                        "target_name": str(hc_name),
                    },
                ),
                target_name=hc_name,
                phase_kind="local",
                n_total_override=int(local_trial_total),
                seed_presets=list(local_seed_presets or []),
                optuna_builder=(
                    (
                        lambda tr,
                        _base_tc=dict(base_tc),
                        _center=dict(center),
                        _shrink=float(local_shrink): _target_local_candidate_clip(
                            _suggest_auto_mode_candidate_local_optuna(
                                _base_tc,
                                _center,
                                tr,
                                shrink=float(_shrink),
                                optimize_mag_low=False,
                            )
                        )
                    )
                    if bool(use_optuna_curve_trials)
                    else None
                ),
                seed_to_params=(
                    (
                        lambda preset,
                        _base_tc=dict(base_tc),
                        _center=dict(center),
                        _shrink=float(local_shrink): _seed_auto_mode_candidate_local_optuna_params(
                            _base_tc,
                            _center,
                            preset,
                            shrink=float(_shrink),
                            optimize_mag_low=False,
                        )
                    )
                    if bool(use_optuna_curve_trials)
                    else None
                ),
            )
            for lc_idx, out in enumerate(local_out, start=1):
                if bool(out.get("ok", False)):
                    met = dict(out.get("metrics", {}) or {})
                    trial_preset = dict(out.get("preset", {}) or {})
                    ok_n += 1
                    rank_sum += _auto_safe_float(met.get("rank_score"), 0.0)
                    avg_score_sum += _auto_safe_float(met.get("avg_score"), 0.0)
                    curve_scored.append({"metrics": dict(met), "preset": dict(trial_preset)})
                    if best_metrics is None or _auto_rank_key(met) < _auto_rank_key(best_metrics):
                        prev = dict(best_metrics or {})
                        best_metrics = dict(met)
                        best_preset = dict(trial_preset)
                        logger.info(
                            "Automatic mode target Local refine winner improved: target=%s, avg_score=%.3f -> %.3f, rank_score=%.3f -> %.3f",
                            str(hc_name),
                            _auto_safe_float(prev.get("avg_score"), 0.0),
                            _auto_safe_float(met.get("avg_score"), 0.0),
                            _auto_safe_float(prev.get("rank_score"), 0.0),
                            _auto_safe_float(met.get("rank_score"), 0.0),
                        )
                else:
                    logger.warning(
                        "Automatic mode target local trial failed: target=%s center=%d %d/%d (%s)",
                        str(hc_name),
                        int(li),
                        int(lc_idx),
                        int(local_trial_total),
                        str(out.get("error", "unknown error") or "unknown error"),
                    )

    if ok_n <= 0 or not isinstance(best_metrics, dict):
        return None

    final_best = _auto_select_best_scored(curve_scored)
    if isinstance(final_best, dict):
        best_metrics = dict(final_best.get("metrics", {}) or {})
        best_preset = dict(final_best.get("preset", {}) or {})

    return {
        "hc_mode": str(hc_name),
        "fit_rms_db": _auto_safe_float(tc.get("fit_rms_db"), 0.0),
        "offset_db": _auto_safe_float(tc.get("offset_db"), 0.0),
        "preselect_score": _auto_safe_float(
            tc.get("preselect_score", tc.get("fit_rms_db", float("inf"))),
            float("inf"),
        ),
        "boost_penalty": _auto_safe_float(tc.get("boost_penalty", 0.0), 0.0),
        "asym_penalty_db": _auto_safe_float(tc.get("asym_penalty_db", 0.0), 0.0),
        "mode_fit_rms_db": _auto_safe_float(tc.get("mode_fit_rms_db", 0.0), 0.0),
        "from_cache_wildcard": bool(tc.get("from_cache_wildcard", False)),
        "trials_total": int(trials_total_count),
        "trials_ok": int(ok_n),
        "avg_rank_score": float(rank_sum / max(1, ok_n)),
        "avg_avg_score": float(avg_score_sum / max(1, ok_n)),
        "best_metrics": dict(best_metrics),
        "best_preset": dict(best_preset or {}),
    }


def _select_target_curve_with_trials_impl(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb=None,
    top_n: int = AUTO_MODE_TARGET_TOP_N,
    trials_per_curve: int = AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    runtime=None,
) -> dict | None:
    runtime = coerce_orchestrator_runtime(runtime)
    goal = _auto_goal(base_data)
    cfg = AutoModeConfig.from_base_data(base_data)
    compat_version = _auto_compat_version(base_data)
    filter_key = _auto_filter_cache_key(base_data)
    rank_basis = _auto_goal_basis_text(goal)
    optimizer_backend = _auto_optimizer_backend(
        base_data,
        default_optuna_enabled=bool(cfg.optuna_pilot_enabled),
    )
    optuna_mod = (
        runtime.auto_import_optuna() if str(optimizer_backend) == "optuna" else None
    )
    if str(optimizer_backend) == "optuna" and optuna_mod is None:
        logger.warning(
            "Automatic mode target select: optuna backend requested but unavailable; "
            "falling back to builtin sampler."
        )
        optimizer_backend = "builtin"
    logger.info(
        "Automatic mode target select: goal=%s, basis=%s",
        str(goal),
        str(rank_basis),
    )

    seed_target = _auto_seed_from_signature(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_mode=None,
        include_hc_mode=False,
    )
    _auto_apply_seed(seed_target)
    logger.info("Automatic mode target select: seed=%d", int(seed_target))
    target_study_sig = _auto_signature(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_mode=None,
        include_hc_mode=False,
    )

    # --- Adaptive fast path: synthesize target directly from measurements ---
    atm = str(base_data.get("auto_target_mode", "auto") or "auto").strip().lower()
    if atm == "adaptive":
        try:
            synth = synthesize_target_from_measurements(
                measurements.get("f_l"),
                measurements.get("m_l"),
                measurements.get("f_r"),
                measurements.get("m_r"),
                bass_comp_frac=float(AUTO_MODE_SYNTH_TARGET_BASS_COMP_FRAC),
                bass_comp_ref_db=float(AUTO_MODE_SYNTH_TARGET_BASS_COMP_REF_DB),
                tilt_comp_frac=float(AUTO_MODE_SYNTH_TARGET_TILT_COMP_FRAC),
                hf_comp_frac=float(AUTO_MODE_SYNTH_TARGET_HF_COMP_FRAC),
                smooth_oct=float(AUTO_MODE_SYNTH_TARGET_SMOOTH_OCT),
            )
            if synth is not None:
                synth_f, synth_m = synth
                logger.info("Automatic mode target select: adaptive fast path used")
                if callable(status_cb):
                    status_cb("CamillaFIR automatic mode: adaptive target synthesized from measurements")
                return {
                    "selected_hc_mode": str(AUTO_MODE_SYNTH_TARGET_NAME),
                    "fit_rms_db": float("nan"),
                    "offset_db": 0.0,
                    "selection_method": "adaptive",
                    "selection_basis": rank_basis,
                    "auto_goal": goal,
                    "top_n": 0,
                    "trials_per_curve": 0,
                    "candidates": [],
                    "evaluated": [],
                    "best_preset": {},
                    "_synth_hc_f": synth_f,
                    "_synth_hc_m": synth_m,
                }
        except Exception as _exc:
            logger.warning("Adaptive target synthesis failed: %s: %s", type(_exc).__name__, _exc)
        # Fall through to normal auto search if synthesis fails

    cached_target_hc = None
    cached_target_preset = {}
    cached_target_source = None

    cached_target_entry = None
    try:
        cached_target_entry = runtime.auto_cache_get_target_for_measurements(
            measurements,
            goal=goal,
            filter_key=filter_key,
            compat_version=compat_version,
        )
    except Exception:
        cached_target_entry = None
    if isinstance(cached_target_entry, dict):
        cached_hc = str(
            cached_target_entry.get(
                "best_target_curve",
                cached_target_entry.get("best_hc_mode", ""),
            )
            or ""
        ).strip()
        cached_hc = _auto_builtin_target_name(cached_hc)
        if _cache_target_valid(runtime, cached_hc):
            cached_target_hc = str(cached_hc)
            cached_target_preset = dict(cached_target_entry.get("best_preset", {}) or {})
            cached_target_source = "cache_measurement"
            logger.info(
                "Automatic mode target select: cache seed (measurement) target=%s",
                str(cached_target_hc),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: target preselect cache seed "
                    f"(measurement -> {str(cached_target_hc)})"
                )

    if bool(AUTO_MODE_CACHE_ENABLED):
        try:
            sig_target = _auto_signature(
                base_data=base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=None,
                include_hc_mode=False,
            )
            cached_hc = runtime.auto_cache_get_best_target(
                sig_target,
                filter_key=filter_key,
                compat_version=compat_version,
            )
            cached_hc = _auto_builtin_target_name(cached_hc)
            if _cache_target_valid(runtime, cached_hc):
                cached_target_hc = str(cached_hc)
                cached_target_preset = runtime.auto_cache_get_best(
                    sig_target,
                    filter_key=filter_key,
                    compat_version=compat_version,
                ) or {}
                cached_target_source = "cache_signature"
                logger.info(
                    "Automatic mode target select: cache seed (signature) target=%s",
                    str(cached_target_hc),
                )
                if callable(status_cb):
                    status_cb(
                        "CamillaFIR automatic mode: target preselect cache seed "
                        f"(signature -> {str(cached_target_hc)})"
                    )
        except Exception:
            pass

    if (
        str(cached_target_source) == "cache_signature"
        and _cache_target_valid(runtime, cached_target_hc)
        and isinstance(cached_target_preset, dict)
        and bool(cached_target_preset)
    ):
        fallback = _cached_target_return(
            runtime=runtime,
            cached_hc_mode=cached_target_hc,
            cached_preset=cached_target_preset,
            selection_method="cache_signature_hit",
            base_data=base_data,
            measurements=measurements,
            goal=goal,
            rank_basis=rank_basis,
        )
        if isinstance(fallback, dict):
            logger.info(
                "Automatic mode target select: exact cache hit for same measurements + settings, using cached target=%s and skipping all target comparison trials",
                str(cached_target_hc),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: target loaded directly from cache "
                    f"(same measurements + settings -> {str(cached_target_hc)}, "
                    "skipping target comparison trials)"
                )
            return dict(fallback)

    f6_hz = _auto_safe_float(
        base_data.get("_auto_mag_c_min_hz", base_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    f6_txt = f" (-6 dB {f6_hz:.1f} Hz)" if np.isfinite(f6_hz) else ""

    quick = runtime.auto_select_builtin_target_curve(
        base_data,
        f_l=measurements.get("f_l"),
        m_l=measurements.get("m_l"),
        f_r=measurements.get("f_r"),
        m_r=measurements.get("m_r"),
    )
    if not isinstance(quick, dict):
        fallback = _cached_target_return(
            runtime=runtime,
            cached_hc_mode=cached_target_hc,
            cached_preset=cached_target_preset,
            selection_method=str(cached_target_source or "cache"),
            base_data=base_data,
            measurements=measurements,
            goal=goal,
            rank_basis=rank_basis,
        )
        if isinstance(fallback, dict):
            logger.info(
                "Automatic mode target select: quick preselect unavailable, fallback to cached target=%s",
                str(cached_target_hc),
            )
            return dict(fallback)
        return None

    quick_candidates = list(
        quick.get("candidates_all", quick.get("candidates", []))
        or quick.get("candidates", [])
        or []
    )
    quick_candidates = [dict(tc or {}) for tc in quick_candidates if isinstance(tc, dict)]
    if not quick_candidates:
        fallback = _cached_target_return(
            runtime=runtime,
            cached_hc_mode=cached_target_hc,
            cached_preset=cached_target_preset,
            selection_method=str(cached_target_source or "cache"),
            base_data=base_data,
            measurements=measurements,
            goal=goal,
            rank_basis=rank_basis,
        )
        if isinstance(fallback, dict):
            logger.info(
                "Automatic mode target select: no quick candidates, fallback to cached target=%s",
                str(cached_target_hc),
            )
            return dict(fallback)
        return None
    quick_candidates = sorted(
        quick_candidates,
        key=lambda tc: (
            _tc_score(tc),
            _auto_safe_float(tc.get("fit_rms_db", float("inf")), float("inf")),
            str(tc.get("hc_mode", "") or "").strip(),
        ),
    )
    quick_rows = [
        f"{str(tc.get('hc_mode', 'n/a'))}: "
        f"fit={_auto_safe_float(tc.get('fit_rms_db', float('nan')), float('nan')):.3f} dB, "
        f"pre={_tc_score(tc):.3f}, "
        f"boost={_auto_safe_float(tc.get('boost_penalty', 0.0), 0.0):.3f}, "
        f"asym={_auto_safe_float(tc.get('asym_penalty_db', 0.0), 0.0):.3f}"
        for tc in quick_candidates
    ]
    logger.info(
        "Automatic mode target preselect candidates:\n%s",
        "\n".join(quick_rows),
    )
    if callable(status_cb):
        top3_txt = ", ".join(
            [
                (
                    f"{str(tc.get('hc_mode', 'n/a') or 'n/a')}"
                    f"(pre={_tc_score(tc):.3f}, fit={_auto_safe_float(tc.get('fit_rms_db', 0.0), 0.0):.3f}, "
                    f"boost={_auto_safe_float(tc.get('boost_penalty', 0.0), 0.0):.3f}, "
                    f"asym={_auto_safe_float(tc.get('asym_penalty_db', 0.0), 0.0):.3f})"
                )
                for tc in quick_candidates[:3]
            ]
        )
        status_cb(
            "CamillaFIR automatic mode: target preselect top-3 "
            f"(goal {goal}) {top3_txt}"
        )

    trials_eff = max(1, int(trials_per_curve))
    shortlisted, shortlist_meta = _auto_target_adaptive_shortlist(
        quick_candidates,
        top_n=int(top_n),
    )
    if not shortlisted:
        fallback = _cached_target_return(
            runtime=runtime,
            cached_hc_mode=cached_target_hc,
            cached_preset=cached_target_preset,
            selection_method=str(cached_target_source or "cache"),
            base_data=base_data,
            measurements=measurements,
            goal=goal,
            rank_basis=rank_basis,
        )
        if isinstance(fallback, dict):
            return dict(fallback)
        return None
    logger.info(
        "Automatic mode target shortlist: n=%d/%d (base_top_n=%d, spread_based_n=%d, spread=%.3f dB, best_score=%.3f)",
        int(shortlist_meta.get("shortlist_n", len(shortlisted))),
        int(shortlist_meta.get("candidate_total", len(quick_candidates))),
        int(shortlist_meta.get("top_n_eff", max(1, int(top_n)))),
        int(shortlist_meta.get("spread_based_n", len(shortlisted))),
        float(
            _auto_safe_float(
                shortlist_meta.get("spread_db", AUTO_MODE_TARGET_TOP_N_SPREAD_DB),
                AUTO_MODE_TARGET_TOP_N_SPREAD_DB,
            )
        ),
        float(
            _auto_safe_float(
                shortlist_meta.get("best_score", _tc_score(shortlisted[0])),
                _tc_score(shortlisted[0]),
            )
        ),
    )
    if callable(status_cb):
        status_cb(
            "CamillaFIR automatic mode: target shortlist "
            f"(selected {int(shortlist_meta.get('shortlist_n', len(shortlisted)))}/"
            f"{int(shortlist_meta.get('candidate_total', len(quick_candidates)))} "
            f"by spread {float(_auto_safe_float(shortlist_meta.get('spread_db', AUTO_MODE_TARGET_TOP_N_SPREAD_DB), AUTO_MODE_TARGET_TOP_N_SPREAD_DB)):.2f} dB)"
        )

    cache_wildcard_participated = False
    if _auto_safe_bool(AUTO_MODE_TARGET_CACHE_AS_WILDCARD, True) and _cache_target_valid(
        runtime,
        cached_target_hc,
    ):
        shortlisted, cache_meta = _auto_target_insert_cached_wildcard(
            shortlisted,
            quick_candidates,
            cached_hc_mode=str(cached_target_hc),
        )
        cache_wildcard_participated = bool(
            cache_meta.get("inserted", False)
            or cache_meta.get("already_present", False)
        )
        if bool(cache_meta.get("inserted", False)):
            logger.info(
                "Automatic mode target shortlist: inserted cache wildcard target=%s",
                str(cache_meta.get("hc_mode", cached_target_hc)),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: target shortlist cache wildcard inserted "
                    f"({str(cache_meta.get('hc_mode', cached_target_hc))})"
                )
        else:
            logger.info(
                "Automatic mode target shortlist: cache wildcard skipped target=%s (%s)",
                str(cache_meta.get("hc_mode", cached_target_hc)),
                str(cache_meta.get("reason", "unknown")),
            )
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: target shortlist cache wildcard "
                    f"{str(cache_meta.get('reason', 'skipped'))}"
                )

    prefer_milder = _auto_safe_bool(
        base_data.get(
            "auto_target_prefer_milder_step",
            AUTO_MODE_TARGET_PREFER_MILDER_STEP,
        ),
        AUTO_MODE_TARGET_PREFER_MILDER_STEP,
    )
    if prefer_milder and shortlisted:
        leader = dict(shortlisted[0] or {})
        lead_hc = str(leader.get("hc_mode", "") or "").strip()
        lead_milder = _auto_target_one_step_milder(lead_hc)
        if lead_milder:
            milder_tc = None
            for tc in quick_candidates:
                if str(tc.get("hc_mode", "") or "").strip() == str(lead_milder):
                    milder_tc = dict(tc)
                    break
            if isinstance(milder_tc, dict):
                already = {
                    str(tc.get("hc_mode", "") or "").strip()
                    for tc in shortlisted
                    if isinstance(tc, dict)
                }
                leader_fit = _auto_safe_float(leader.get("fit_rms_db", float("inf")), float("inf"))
                milder_fit = _auto_safe_float(milder_tc.get("fit_rms_db", float("inf")), float("inf"))
                leader_pre = _tc_score(leader)
                milder_pre = _tc_score(milder_tc)
                leader_asym = _auto_safe_float(leader.get("asym_penalty_db", 0.0), 0.0)
                milder_asym = _auto_safe_float(milder_tc.get("asym_penalty_db", 0.0), 0.0)
                leader_boost = _auto_safe_float(leader.get("boost_penalty", 0.0), 0.0)
                milder_boost = _auto_safe_float(milder_tc.get("boost_penalty", 0.0), 0.0)
                cond_not_dup = str(lead_milder) not in already
                cond_fit = bool(
                    float(milder_fit)
                    <= float(leader_fit) + float(AUTO_MODE_TARGET_MILDER_MAX_FIT_RMS_ADD_DB)
                )
                cond_diff = bool(
                    float(milder_pre)
                    <= float(leader_pre) + float(AUTO_MODE_TARGET_MILDER_MAX_DIFFICULTY_ADD)
                )
                cond_asym = bool(
                    float(milder_asym)
                    <= float(leader_asym) + float(AUTO_MODE_TARGET_MILDER_MAX_ASYM_ADD)
                )
                if cond_not_dup and cond_fit and cond_diff and cond_asym:
                    shortlisted = list(shortlisted) + [dict(milder_tc)]
                    logger.info(
                        "Automatic mode target shortlist: included milder target %s -> %s (fit %.3f->%.3f, pre %.3f->%.3f, boost %.3f->%.3f, asym %.3f->%.3f)",
                        str(lead_hc),
                        str(lead_milder),
                        float(leader_fit),
                        float(milder_fit),
                        float(leader_pre),
                        float(milder_pre),
                        float(leader_boost),
                        float(milder_boost),
                        float(leader_asym),
                        float(milder_asym),
                    )
                    if callable(status_cb):
                        status_cb(
                            "CamillaFIR automatic mode: target shortlist milder included "
                            f"({str(lead_hc)} -> {str(lead_milder)})"
                        )
                else:
                    logger.info(
                        "Automatic mode target shortlist: skipped milder target %s -> %s (not_dup=%s, fit_ok=%s, pre_ok=%s, asym_ok=%s, boost %.3f->%.3f)",
                        str(lead_hc),
                        str(lead_milder),
                        str(cond_not_dup),
                        str(cond_fit),
                        str(cond_diff),
                        str(cond_asym),
                        float(leader_boost),
                        float(milder_boost),
                    )
                    if callable(status_cb):
                        status_cb(
                            "CamillaFIR automatic mode: target shortlist milder skipped "
                            f"({str(lead_hc)} -> {str(lead_milder)}, "
                            f"not_dup={str(cond_not_dup)}, fit_ok={str(cond_fit)}, "
                            f"pre_ok={str(cond_diff)}, asym_ok={str(cond_asym)})"
                        )
            else:
                logger.info(
                    "Automatic mode target shortlist: milder target for %s not found in quick candidates",
                    str(lead_hc),
                )
        else:
            logger.info(
                "Automatic mode target shortlist: no one-step milder target for %s",
                str(lead_hc),
            )

    dedup_shortlisted = []
    seen_hc = set()
    for tc in shortlisted:
        if not isinstance(tc, dict):
            continue
        hc = str(tc.get("hc_mode", "") or "").strip()
        if not hc or hc in seen_hc:
            continue
        seen_hc.add(hc)
        dedup_shortlisted.append(dict(tc))
    shortlisted = list(dedup_shortlisted)
    if not shortlisted:
        fallback = _cached_target_return(
            runtime=runtime,
            cached_hc_mode=cached_target_hc,
            cached_preset=cached_target_preset,
            selection_method=str(cached_target_source or "cache"),
            base_data=base_data,
            measurements=measurements,
            goal=goal,
            rank_basis=rank_basis,
        )
        if isinstance(fallback, dict):
            return dict(fallback)
        return None
    for tc in shortlisted:
        tc.setdefault("preselect_score", _tc_score(tc))
        tc.setdefault("boost_penalty", _auto_safe_float(tc.get("boost_penalty", 0.0), 0.0))
        tc.setdefault("asym_penalty_db", _auto_safe_float(tc.get("asym_penalty_db", 0.0), 0.0))

    evaluated = []
    total_target_trial_load = int(max(1, len(shortlisted) * max(1, trials_eff)))
    curve_budget = int(runtime.auto_trial_workers(base_data, total_target_trial_load))
    curve_workers = int(max(1, min(len(shortlisted), curve_budget)))
    curve_inner_workers = int(max(1, curve_budget // max(1, curve_workers)))
    select_f6_txt = f", -6 dB point {f6_hz:.1f} Hz" if np.isfinite(f6_hz) else ""

    if curve_workers > 1:
        logger.info(
            "Automatic mode target select: curve-parallel enabled (curves=%d, workers=%d, inner_workers=%d)",
            int(len(shortlisted)),
            int(curve_workers),
            int(curve_inner_workers),
        )
        with ThreadPoolExecutor(max_workers=int(curve_workers)) as executor:
            future_map = {}
            best_done_item = None
            done_n = 0
            for t_idx, tc in enumerate(shortlisted, start=1):
                future = executor.submit(
                    _evaluate_target_curve,
                    runtime=runtime,
                    cfg=cfg,
                    base_data=base_data,
                    measurements=measurements,
                    fs_v=int(fs_v),
                    taps_v=int(taps_v),
                    xos=xos,
                    hpf=hpf,
                    pin_obj=pin_obj,
                    goal=goal,
                    filter_key=filter_key,
                    optimizer_backend=optimizer_backend,
                    optuna_mod=optuna_mod,
                    seed_target=int(seed_target),
                    target_study_sig=target_study_sig,
                    trials_eff=int(trials_eff),
                    shortlisted=shortlisted,
                    status_cb=status_cb,
                    f6_txt=f6_txt,
                    tc=dict(tc or {}),
                    t_idx=int(t_idx),
                    emit_status=False,
                    curve_inner_workers=int(curve_inner_workers),
                )
                future_map[future] = (
                    int(t_idx),
                    str((tc or {}).get("hc_mode", "") or "").strip(),
                )
            for future in as_completed(list(future_map.keys())):
                _t_idx, hc_name = future_map.get(future, (0, "n/a"))
                done_n += 1
                improved = False
                try:
                    item = future.result()
                except Exception as exc:
                    logger.warning(
                        "Automatic mode target curve failed: target=%s (%s)",
                        str(hc_name),
                        f"{type(exc).__name__}: {exc}",
                    )
                    item = None
                if isinstance(item, dict):
                    item_d = dict(item)
                    evaluated.append(dict(item_d))
                    if (
                        not isinstance(best_done_item, dict)
                        or _auto_target_result_rank_key(item_d)
                        < _auto_target_result_rank_key(best_done_item)
                    ):
                        best_done_item = dict(item_d)
                        improved = True
                if callable(status_cb) and bool(improved) and isinstance(best_done_item, dict):
                    bm_now = dict(best_done_item.get("best_metrics", {}) or {})
                    status_cb(
                        "CamillaFIR automatic mode: selecting target curve "
                        f"(best improved {int(done_n)}/{len(shortlisted)}, "
                        f"leader {str(best_done_item.get('hc_mode', 'n/a') or 'n/a')}, "
                        f"tested {str(hc_name or 'n/a')}, "
                        f"{int(trials_eff)} trials/curve{select_f6_txt}, goal {goal}, "
                        f"rank {_auto_safe_float(bm_now.get('rank_score'), 0.0):.3f}, "
                        f"avg {_auto_safe_float(bm_now.get('avg_score'), 0.0):.3f})"
                    )
    else:
        for t_idx, tc in enumerate(shortlisted, start=1):
            hc_name = str((tc or {}).get("hc_mode", "") or "").strip() or "n/a"
            if callable(status_cb):
                status_cb(
                    "CamillaFIR automatic mode: selecting target curve "
                    f"(testing {hc_name} {t_idx}/{len(shortlisted)}, "
                    f"{int(trials_eff)} trials/curve{select_f6_txt}, goal {goal})"
                )
            item = _evaluate_target_curve(
                runtime=runtime,
                cfg=cfg,
                base_data=base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                pin_obj=pin_obj,
                goal=goal,
                filter_key=filter_key,
                optimizer_backend=optimizer_backend,
                optuna_mod=optuna_mod,
                seed_target=int(seed_target),
                target_study_sig=target_study_sig,
                trials_eff=int(trials_eff),
                shortlisted=shortlisted,
                status_cb=status_cb,
                f6_txt=f6_txt,
                tc=dict(tc or {}),
                t_idx=int(t_idx),
                emit_status=True,
                curve_inner_workers=None,
            )
            if isinstance(item, dict):
                evaluated.append(dict(item))

    if not evaluated:
        quick_out = dict(quick or {})
        quick_out.setdefault("selection_method", "fit_rms")
        quick_out["selection_basis"] = str(rank_basis)
        quick_out["auto_goal"] = str(goal)
        return quick_out

    evaluated = sorted(evaluated, key=_auto_target_result_rank_key)
    rank_tie_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_TARGET_BEST_RANK_TIE_EPS, 0.05)))
    target_scored = [
        {
            **dict(it or {}),
            "_auto_select_kind": "target_curve",
            "_target_rank_tie_eps": float(rank_tie_eps),
        }
        for it in evaluated
    ]
    winner = dict(_auto_select_best_scored(target_scored) or evaluated[0])
    selection_method = str(winner.pop("_auto_selection_method", "top3x10_trials") or "top3x10_trials")
    if selection_method == "top3x10_trials_rank_tie_composite":
        old_winner = dict(evaluated[0])
        logger.info(
            "Automatic mode target select: rank tie-break by avg/mode/boost (eps=%.3f) %s -> %s, avg_rank=%.3f -> %.3f, mode_ripple=%.4f -> %.4f, boost_penalty=%.3f -> %.3f",
            float(rank_tie_eps),
            str(old_winner.get("hc_mode", "n/a")),
            str(winner.get("hc_mode", "n/a")),
            _auto_safe_float(old_winner.get("avg_rank_score"), 0.0),
            _auto_safe_float(winner.get("avg_rank_score"), 0.0),
            _auto_target_result_mode_ripple(old_winner),
            _auto_target_result_mode_ripple(winner),
            _auto_safe_float(old_winner.get("boost_penalty", 0.0), 0.0),
            _auto_safe_float(winner.get("boost_penalty", 0.0), 0.0),
        )
    if bool(cache_wildcard_participated) and bool(winner.get("from_cache_wildcard", False)):
        selection_method = "trial_with_cache_wildcard"

    winner_mode_ripple = _auto_target_result_mode_ripple(winner)
    logger.info(
        "Automatic mode target select: goal=%s, basis=%s, winner=%s, %s, avg_rank=%.3f, mode_ripple=%.4f, pre=%.3f, boost=%.3f, asym=%.3f, method=%s",
        str(goal),
        str(rank_basis),
        str(winner.get("hc_mode", "n/a")),
        _auto_metric_text(dict(winner.get("best_metrics", {}) or {}), goal),
        _auto_safe_float(winner.get("avg_rank_score", 0.0), 0.0),
        float(winner_mode_ripple),
        _auto_safe_float(winner.get("preselect_score", winner.get("fit_rms_db", 1e9)), 1e9),
        _auto_safe_float(winner.get("boost_penalty", 0.0), 0.0),
        _auto_safe_float(winner.get("asym_penalty_db", 0.0), 0.0),
        str(selection_method),
    )
    if callable(status_cb):
        status_cb(
            "CamillaFIR automatic mode: target finalize "
            f"(winner {str(winner.get('hc_mode', 'n/a'))}, method {selection_method}, "
            f"rank {_auto_safe_float(dict(winner.get('best_metrics', {}) or {}).get('rank_score', 0.0), 0.0):.3f}, "
            f"avg {_auto_safe_float(dict(winner.get('best_metrics', {}) or {}).get('avg_score', 0.0), 0.0):.3f}, "
            f"pre {_auto_safe_float(winner.get('preselect_score', winner.get('fit_rms_db', 1e9)), 1e9):.3f}, "
            f"fit {_auto_safe_float(winner.get('fit_rms_db', 0.0), 0.0):.3f} dB)"
        )
    return {
        "selected_hc_mode": str(winner.get("hc_mode", quick.get("selected_hc_mode", "Harman6"))),
        "fit_rms_db": float(winner.get("fit_rms_db", quick.get("fit_rms_db", 0.0))),
        "offset_db": float(winner.get("offset_db", quick.get("offset_db", 0.0))),
        "selection_method": str(selection_method),
        "selection_basis": str(rank_basis),
        "auto_goal": str(goal),
        "top_n": int(len(shortlisted)),
        "trials_per_curve": int(trials_eff),
        "candidates": list(shortlisted),
        "evaluated": list(evaluated),
        "best_preset": dict(winner.get("best_preset", {}) or {}),
    }
