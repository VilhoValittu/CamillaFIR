"""Winner-polish helpers for auto-mode finalize orchestration."""

from __future__ import annotations

import logging

import numpy as np

from .shared import (
    AUTO_MODE_MAG_C_MIN_MAX_HZ,
    AUTO_MODE_MAG_C_MIN_MIN_HZ,
    _auto_phase_limit_clip,
    _auto_safe_float,
)

logger = logging.getLogger("CamillaFIR")
__all__ = ["apply_mag_c_min_winner_polish", "apply_phase_limit_winner_polish"]


def apply_phase_limit_winner_polish(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    filter_key: str,
    enabled: bool,
    offsets_hz,
    status_cb,
    materialize_preset_result,
    cache_ready_preset,
    auto_is_better_refine,
) -> tuple[dict, dict, bool, dict]:
    cur_best_preset = dict(best_preset or {})
    cur_best_metrics = dict(best_metrics or {})
    phase_limit_meta = {
        "enabled": bool(enabled),
        "applicable": bool(str(filter_key) in ("linear", "asym")),
        "phase_label": str(phase_label),
        "tested_phase_limits_hz": [],
        "accepted_phase_limits_hz": [],
        "tested_count": 0,
        "applied": False,
        "start_phase_limit_hz": float("nan"),
        "final_phase_limit_hz": float("nan"),
        "rank_before": float("nan"),
        "rank_after": float("nan"),
        "avg_before": float("nan"),
        "avg_after": float("nan"),
    }
    if not bool(enabled):
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta
    if str(filter_key) not in ("linear", "asym"):
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta
    if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta

    phase_limit_base = _auto_safe_float(
        cur_best_preset.get(
            "phase_limit",
            dict(base_data_ref or {}).get("phase_limit", float("nan")),
        ),
        float("nan"),
    )
    if not np.isfinite(phase_limit_base):
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta
    initial_phase_limit = round(
        float(
            _auto_phase_limit_clip(
                phase_limit_base,
                default=400.0,
            )
        ),
        1,
    )
    phase_limit_meta["start_phase_limit_hz"] = float(initial_phase_limit)
    phase_limit_meta["rank_before"] = float(
        _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
    )
    phase_limit_meta["avg_before"] = float(
        _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
    )

    tested_phase_limits: set[float] = {float(initial_phase_limit)}
    candidate_phase_limits: list[float] = []
    for delta_hz in offsets_hz:
        cand_phase_limit = round(
            float(
                _auto_phase_limit_clip(
                    float(phase_limit_base) + float(_auto_safe_float(delta_hz, 0.0)),
                    default=400.0,
                )
            ),
            1,
        )
        if cand_phase_limit in tested_phase_limits:
            continue
        tested_phase_limits.add(cand_phase_limit)
        candidate_phase_limits.append(float(cand_phase_limit))

    phase_limit_meta["tested_phase_limits_hz"] = [float(v) for v in candidate_phase_limits]
    phase_limit_meta["tested_count"] = int(len(candidate_phase_limits))
    if not candidate_phase_limits:
        phase_limit_meta["final_phase_limit_hz"] = float(initial_phase_limit)
        phase_limit_meta["rank_after"] = float(
            _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
        )
        phase_limit_meta["avg_after"] = float(
            _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
        )
        return cur_best_preset, cur_best_metrics, False, phase_limit_meta

    improved = False
    logger.info(
        "Automatic mode %s: testing %d phase_limit winner-polish candidate(s) around %.1f Hz.",
        str(phase_label),
        int(len(candidate_phase_limits)),
        float(initial_phase_limit),
    )
    for idx, cand_phase_limit in enumerate(candidate_phase_limits, start=1):
        cand_test = dict(cur_best_preset or {})
        cand_test["phase_limit"] = float(cand_phase_limit)
        try:
            _phase_result, phase_metrics, _phase_data = materialize_preset_result(
                cand_test,
                include_response_arrays=False,
                summarize=False,
                base_data_override=base_data_ref,
            )
        except Exception as exc:
            logger.warning(
                "Automatic mode %s failed for candidate %d/%d (phase_limit=%.1f Hz): %s",
                str(phase_label),
                int(idx),
                int(len(candidate_phase_limits)),
                float(cand_phase_limit),
                f"{type(exc).__name__}: {exc}",
            )
            continue

        phase_metrics = dict(phase_metrics or {})
        better, reason = auto_is_better_refine(
            phase_metrics,
            cur_best_metrics,
            goal,
            return_reason=True,
        )
        # Guard: reject if rank_score doesn't genuinely improve (prevents tiebreak drift)
        if bool(better):
            prev_rank = _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
            new_rank = _auto_safe_float(phase_metrics.get("rank_score"), float("nan"))
            if np.isfinite(prev_rank) and np.isfinite(new_rank) and float(new_rank - prev_rank) < 0.01:
                better = False
                reason = "rank_score_guard"
        # Guard: reject if avg_score drops significantly
        if bool(better):
            prev_avg = _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
            new_avg = _auto_safe_float(phase_metrics.get("avg_score"), float("nan"))
            if np.isfinite(prev_avg) and np.isfinite(new_avg) and float(prev_avg - new_avg) > 3.0:
                better = False
                reason = "avg_score_guard"
        logger.info(
            "Automatic mode %s candidate %d/%d: phase_limit=%.1f Hz, rank=%.3f, decision=%s (%s)",
            str(phase_label),
            int(idx),
            int(len(candidate_phase_limits)),
            float(cand_phase_limit),
            _auto_safe_float(phase_metrics.get("rank_score"), 0.0),
            "accept" if bool(better) else "reject",
            str(reason),
        )
        if not bool(better):
            continue

        prev_best = dict(cur_best_metrics or {})
        cur_best_metrics = dict(phase_metrics or {})
        cur_best_preset = cache_ready_preset(cand_test, best_metrics=cur_best_metrics)
        improved = True
        accepted = list(phase_limit_meta.get("accepted_phase_limits_hz", []) or [])
        accepted.append(float(cand_phase_limit))
        phase_limit_meta["accepted_phase_limits_hz"] = accepted
        logger.info(
            "Automatic mode %s accepted candidate %d/%d: phase_limit %.1f -> %.1f Hz, rank %.3f -> %.3f, avg %.3f -> %.3f",
            str(phase_label),
            int(idx),
            int(len(candidate_phase_limits)),
            float(phase_limit_base),
            float(cand_phase_limit),
            _auto_safe_float(prev_best.get("rank_score"), 0.0),
            _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0),
            _auto_safe_float(prev_best.get("avg_score"), 0.0),
            _auto_safe_float(cur_best_metrics.get("avg_score"), 0.0),
        )
        phase_limit_base = float(cand_phase_limit)
        if callable(status_cb):
            status_cb(
                "CamillaFIR automatic mode: phase_limit winner polish improved "
                f"(phase_limit {float(cand_phase_limit):.1f} Hz, "
                f"rank {_auto_safe_float(cur_best_metrics.get('rank_score'), 0.0):.3f}, "
                f"avg {_auto_safe_float(cur_best_metrics.get('avg_score'), 0.0):.3f})"
            )

    phase_limit_meta["applied"] = bool(improved)
    phase_limit_meta["final_phase_limit_hz"] = float(
        round(
            float(
                _auto_phase_limit_clip(
                    cur_best_preset.get(
                        "phase_limit",
                        dict(base_data_ref or {}).get("phase_limit", 400.0),
                    ),
                    default=400.0,
                )
            ),
            1,
        )
    )
    phase_limit_meta["rank_after"] = float(
        _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
    )
    phase_limit_meta["avg_after"] = float(
        _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
    )
    return cur_best_preset, cur_best_metrics, bool(improved), dict(phase_limit_meta or {})


def apply_mag_c_min_winner_polish(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    enabled: bool,
    step_hz,
    max_down_hz,
    max_up_hz=None,
    status_cb,
    materialize_preset_result,
    cache_ready_preset,
    auto_is_better_refine,
) -> tuple[dict, dict, bool, dict]:
    cur_best_preset = dict(best_preset or {})
    cur_best_metrics = dict(best_metrics or {})
    mag_c_min_meta = {
        "enabled": bool(enabled),
        "applicable": True,
        "phase_label": str(phase_label),
        "tested_mag_c_min_hz": [],
        "accepted_mag_c_min_hz": [],
        "tested_count": 0,
        "applied": False,
        "start_mag_c_min_hz": float("nan"),
        "final_mag_c_min_hz": float("nan"),
        "rank_before": float("nan"),
        "rank_after": float("nan"),
        "avg_before": float("nan"),
        "avg_after": float("nan"),
    }
    if not bool(enabled):
        return cur_best_preset, cur_best_metrics, False, mag_c_min_meta
    if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
        return cur_best_preset, cur_best_metrics, False, mag_c_min_meta

    mag_c_min_base = _auto_safe_float(
        cur_best_preset.get(
            "mag_c_min",
            dict(base_data_ref or {}).get("mag_c_min", float("nan")),
        ),
        float("nan"),
    )
    if not np.isfinite(mag_c_min_base):
        return cur_best_preset, cur_best_metrics, False, mag_c_min_meta
    initial_mag_c_min = round(
        float(
            np.clip(
                float(mag_c_min_base),
                float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
            )
        ),
        1,
    )
    mag_c_min_meta["start_mag_c_min_hz"] = float(initial_mag_c_min)
    mag_c_min_meta["rank_before"] = float(
        _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
    )
    mag_c_min_meta["avg_before"] = float(
        _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
    )

    step_hz_eff = max(0.1, float(_auto_safe_float(step_hz, 1.0)))
    max_down_hz_eff = max(0.0, float(_auto_safe_float(max_down_hz, 15.0)))
    max_up_hz_eff = max(0.0, float(_auto_safe_float(max_up_hz, max_down_hz_eff)))
    min_mag_c_min = max(
        float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
        float(initial_mag_c_min - max_down_hz_eff),
    )
    max_mag_c_min = min(
        float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        float(initial_mag_c_min + max_up_hz_eff),
    )
    candidate_mag_c_mins: list[float] = []
    tested_mag_c_mins: set[float] = {float(initial_mag_c_min)}
    max_span_hz = max(float(max_down_hz_eff), float(max_up_hz_eff))
    max_steps = int(max(0, np.floor((float(max_span_hz) / float(step_hz_eff)) + 1e-9)))
    for step_idx in range(1, int(max_steps) + 1):
        down_value = float(initial_mag_c_min - float(step_idx) * float(step_hz_eff))
        if down_value >= (float(min_mag_c_min) - 1e-9):
            cand_mag_c_min = round(
                float(
                    np.clip(
                        down_value,
                        float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                        float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
                    )
                ),
                1,
            )
            if cand_mag_c_min not in tested_mag_c_mins:
                tested_mag_c_mins.add(cand_mag_c_min)
                candidate_mag_c_mins.append(float(cand_mag_c_min))

        up_value = float(initial_mag_c_min + float(step_idx) * float(step_hz_eff))
        if up_value <= (float(max_mag_c_min) + 1e-9):
            cand_mag_c_min = round(
                float(
                    np.clip(
                        up_value,
                        float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                        float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
                    )
                ),
                1,
            )
            if cand_mag_c_min not in tested_mag_c_mins:
                tested_mag_c_mins.add(cand_mag_c_min)
                candidate_mag_c_mins.append(float(cand_mag_c_min))

    mag_c_min_meta["tested_mag_c_min_hz"] = [float(v) for v in candidate_mag_c_mins]
    mag_c_min_meta["tested_count"] = int(len(candidate_mag_c_mins))
    if not candidate_mag_c_mins:
        mag_c_min_meta["final_mag_c_min_hz"] = float(initial_mag_c_min)
        mag_c_min_meta["rank_after"] = float(
            _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
        )
        mag_c_min_meta["avg_after"] = float(
            _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
        )
        return cur_best_preset, cur_best_metrics, False, mag_c_min_meta

    improved = False
    logger.info(
        "Automatic mode %s: testing %d mag_c_min winner-polish candidate(s) around %.1f Hz.",
        str(phase_label),
        int(len(candidate_mag_c_mins)),
        float(initial_mag_c_min),
    )
    for idx, cand_mag_c_min in enumerate(candidate_mag_c_mins, start=1):
        cand_test = dict(cur_best_preset or {})
        cand_test["mag_c_min"] = float(cand_mag_c_min)
        try:
            _mag_c_min_result, mag_c_min_metrics, _mag_c_min_data = materialize_preset_result(
                cand_test,
                include_response_arrays=False,
                summarize=False,
                base_data_override=base_data_ref,
            )
        except Exception as exc:
            logger.warning(
                "Automatic mode %s failed for candidate %d/%d (mag_c_min=%.1f Hz): %s",
                str(phase_label),
                int(idx),
                int(len(candidate_mag_c_mins)),
                float(cand_mag_c_min),
                f"{type(exc).__name__}: {exc}",
            )
            continue

        mag_c_min_metrics = dict(mag_c_min_metrics or {})
        better, reason = auto_is_better_refine(
            mag_c_min_metrics,
            cur_best_metrics,
            goal,
            return_reason=True,
        )
        # Guard: reject if rank_score doesn't genuinely improve (prevents tiebreak drift)
        if bool(better):
            prev_rank = _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
            new_rank = _auto_safe_float(mag_c_min_metrics.get("rank_score"), float("nan"))
            if np.isfinite(prev_rank) and np.isfinite(new_rank) and float(new_rank - prev_rank) < 0.01:
                better = False
                reason = "rank_score_guard"
        # Guard: reject if avg_score drops significantly
        if bool(better):
            prev_avg = _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
            new_avg = _auto_safe_float(mag_c_min_metrics.get("avg_score"), float("nan"))
            if np.isfinite(prev_avg) and np.isfinite(new_avg) and float(prev_avg - new_avg) > 3.0:
                better = False
                reason = "avg_score_guard"
        logger.info(
            "Automatic mode %s candidate %d/%d: mag_c_min=%.1f Hz, rank=%.3f, decision=%s (%s)",
            str(phase_label),
            int(idx),
            int(len(candidate_mag_c_mins)),
            float(cand_mag_c_min),
            _auto_safe_float(mag_c_min_metrics.get("rank_score"), 0.0),
            "accept" if bool(better) else "reject",
            str(reason),
        )
        if not bool(better):
            continue

        prev_best = dict(cur_best_metrics or {})
        prev_mag_c_min_base = float(mag_c_min_base)
        cur_best_metrics = dict(mag_c_min_metrics or {})
        cur_best_preset = cache_ready_preset(cand_test, best_metrics=cur_best_metrics)
        improved = True
        accepted = list(mag_c_min_meta.get("accepted_mag_c_min_hz", []) or [])
        accepted.append(float(cand_mag_c_min))
        mag_c_min_meta["accepted_mag_c_min_hz"] = accepted
        logger.info(
            "Automatic mode %s accepted candidate %d/%d: mag_c_min %.1f -> %.1f Hz, rank %.3f -> %.3f, avg %.3f -> %.3f",
            str(phase_label),
            int(idx),
            int(len(candidate_mag_c_mins)),
            float(mag_c_min_base),
            float(cand_mag_c_min),
            _auto_safe_float(prev_best.get("rank_score"), 0.0),
            _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0),
            _auto_safe_float(prev_best.get("avg_score"), 0.0),
            _auto_safe_float(cur_best_metrics.get("avg_score"), 0.0),
        )
        mag_c_min_base = float(cand_mag_c_min)
        if callable(status_cb):
            status_cb(
                "CamillaFIR automatic mode: mag_c_min winner polish improved "
                f"(mag_c_min {float(prev_mag_c_min_base):.1f} -> {float(cand_mag_c_min):.1f} Hz, "
                f"rank {_auto_safe_float(prev_best.get('rank_score'), 0.0):.3f} -> {_auto_safe_float(cur_best_metrics.get('rank_score'), 0.0):.3f}, "
                f"avg {_auto_safe_float(prev_best.get('avg_score'), 0.0):.3f} -> {_auto_safe_float(cur_best_metrics.get('avg_score'), 0.0):.3f})"
            )

    mag_c_min_meta["applied"] = bool(improved)
    mag_c_min_meta["final_mag_c_min_hz"] = float(
        round(
            float(
                np.clip(
                    cur_best_preset.get(
                        "mag_c_min",
                        dict(base_data_ref or {}).get("mag_c_min", 25.0),
                    ),
                    float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                    float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
                )
            ),
            1,
        )
    )
    mag_c_min_meta["rank_after"] = float(
        _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
    )
    mag_c_min_meta["avg_after"] = float(
        _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
    )
    return cur_best_preset, cur_best_metrics, bool(improved), dict(mag_c_min_meta or {})


__all__ = [
    "apply_mag_c_min_winner_polish",
    "apply_phase_limit_winner_polish",
]
