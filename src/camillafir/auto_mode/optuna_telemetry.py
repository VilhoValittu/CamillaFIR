"""Optuna telemetry formatting and logging helpers for automatic mode."""

from __future__ import annotations

import numpy as np


def _auto_metric_summary(values) -> dict:
    vals = []
    for v in list(values or []):
        try:
            fv = float(v)
            if np.isfinite(fv):
                vals.append(float(fv))
        except Exception:
            pass

    if not vals:
        return {"count": 0, "min": None, "median": None, "max": None}

    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }

def _auto_metric_summary_text(name: str, summary: dict | None, ndigits: int = 3) -> str:
    s = dict(summary or {})
    if int(s.get("count", 0) or 0) <= 0:
        return f"{str(name)} n/a"

    def _fmt(x):
        try:
            fx = float(x)
            if np.isfinite(fx):
                return f"{fx:.{int(ndigits)}f}"
        except Exception:
            pass
        return "n/a"

    return (
        f"{str(name)} min/med/max "
        f"{_fmt(s.get('min'))}/{_fmt(s.get('median'))}/{_fmt(s.get('max'))}"
    )

def _auto_optuna_log_run_telemetry(logger, *, phase_label: str, tel: dict | None) -> None:
    tel = dict(tel or {})
    if not tel:
        return

    msg = (
        "Automatic mode Optuna telemetry [%s]: requested=%d run=%d complete=%d fail=%d "
        "startup=%d model=%d dup=%d(replay=%d,reserved=%d)"
        % (
            str(phase_label),
            int(tel.get("requested_total", 0) or 0),
            int(tel.get("run_trials", 0) or 0),
            int(tel.get("complete_trials", 0) or 0),
            int(tel.get("failed_trials", 0) or 0),
            int(tel.get("startup_complete", 0) or 0),
            int(tel.get("model_complete", 0) or 0),
            int(tel.get("duplicate_skips", 0) or 0),
            int(tel.get("duplicate_replays", 0) or 0),
            int(tel.get("duplicate_reserved", 0) or 0),
        )
    )
    logger.info(msg)

    if bool(tel.get("constraints_active", False)):
        cflags = dict(tel.get("constraint_flags", {}) or {})
        use_events = bool(cflags.get("use_events", True))
        logger.info(
            "Automatic mode Optuna feasible [%s]: feasible=%d infeasible=%d "
            "best_raw=%s best_feasible=%s violations(r=%d,e=%d,b=%d)",
            str(phase_label),
            int(tel.get("feasible_trials", 0) or 0),
            int(tel.get("infeasible_trials", 0) or 0),
            "n/a" if tel.get("best_raw_value", None) is None else f"{float(tel['best_raw_value']):.6f}",
            "n/a" if tel.get("best_feasible_value", None) is None else f"{float(tel['best_feasible_value']):.6f}",
            int((tel.get("violation_counts", {}) or {}).get("ripple", 0) or 0),
            int((tel.get("violation_counts", {}) or {}).get("events", 0) or 0),
            int((tel.get("violation_counts", {}) or {}).get("boost", 0) or 0),
        )
        if not use_events:
            logger.info(
                "Automatic mode Optuna refine constraints [%s]: events constraint disabled for refine scope",
                str(phase_label),
            )
        if (
            use_events
            and
            int(tel.get("complete_trials", 0) or 0) > 0
            and int(tel.get("feasible_trials", 0) or 0) == 0
            and int(tel.get("infeasible_trials", 0) or 0) > 0
        ):
            ev_all_txt = _auto_metric_summary_text("events", tel.get("events_summary", {}), 3)
            ev_bad_txt = _auto_metric_summary_text("events_bad", tel.get("events_infeasible_summary", {}), 3)
            ev_thr = ((tel.get("constraint_thresholds", {}) or {}).get("max_events_severity", None))
            logger.warning(
                "Automatic mode Optuna zero-feasible [%s]: all complete trials violated constraints, "
                "events<=%s required, %s, %s",
                str(phase_label),
                "n/a" if ev_thr is None else f"{float(ev_thr):.3f}",
                str(ev_all_txt),
                str(ev_bad_txt),
            )

def _auto_optuna_fmt_value(v, ndigits: int = 3) -> str:
    try:
        fv = float(v)
        if np.isfinite(fv):
            return f"{fv:.{int(ndigits)}f}"
    except Exception:
        pass
    return "n/a"

def _auto_optuna_telemetry_text(tel: dict | None) -> str:
    return _auto_optuna_telemetry_text_ex(tel, include_phase_kind=False)

def _auto_optuna_telemetry_text_ex(tel: dict | None, *, include_phase_kind: bool = False) -> str:
    t = dict(tel or {})
    if not t:
        return ""

    run_n = int(t.get("run_trials", 0) or 0)
    complete_n = int(t.get("complete_trials", 0) or 0)
    startup_n = int(t.get("startup_complete", 0) or 0)
    model_n = int(t.get("model_complete", 0) or 0)
    dup_n = int(t.get("duplicate_skips", 0) or 0)

    parts = [
        f"optuna run={run_n}",
        f"ok={complete_n}",
        f"startup={startup_n}",
        f"model={model_n}",
    ]
    if bool(include_phase_kind):
        phase_kind = str(t.get("phase_kind", "") or "").strip()
        if phase_kind:
            parts.insert(0, f"phase={phase_kind}")
    if dup_n > 0:
        parts.append(f"dup={dup_n}")

    if bool(t.get("constraints_active", False)):
        cflags = dict(t.get("constraint_flags", {}) or {})
        feas_n = int(t.get("feasible_trials", 0) or 0)
        infeas_n = int(t.get("infeasible_trials", 0) or 0)
        parts.append(f"feas={feas_n}/{feas_n + infeas_n}")
        if not bool(cflags.get("use_events", True)):
            parts.append("events=off")
        best_raw = t.get("best_raw_value", None)
        best_feas = t.get("best_feasible_value", None)
        if best_raw is not None:
            parts.append(f"raw={_auto_optuna_fmt_value(best_raw, 3)}")
        if best_feas is not None:
            parts.append(f"best={_auto_optuna_fmt_value(best_feas, 3)}")

        vc = dict(t.get("violation_counts", {}) or {})
        vr = int(vc.get("ripple", 0) or 0)
        ve = int(vc.get("events", 0) or 0)
        vb = int(vc.get("boost", 0) or 0)
        if (vr + ve + vb) > 0:
            parts.append(f"viol r/e/b={vr}/{ve}/{vb}")
    else:
        best_raw = t.get("best_raw_value", None)
        if best_raw is not None:
            parts.append(f"best={_auto_optuna_fmt_value(best_raw, 3)}")

    return ", ".join(parts)

def _auto_optuna_events_debug_text(tel: dict | None, ndigits: int = 3) -> str:
    t = dict(tel or {})
    thr = dict(t.get("constraint_thresholds", {}) or {})
    cflags = dict(t.get("constraint_flags", {}) or {})
    use_events = bool(cflags.get("use_events", True))
    ev_thr = thr.get("max_events_severity", None)
    summ = dict(t.get("events_summary", {}) or {})

    def _fmt(x):
        try:
            fx = float(x)
            if np.isfinite(fx):
                return f"{fx:.{int(ndigits)}f}"
        except Exception:
            pass
        return "n/a"

    ev_body = "events n/a"
    if int(summ.get("count", 0) or 0) > 0:
        ev_body = (
            f"events min/med/max "
            f"{_fmt(summ.get('min'))}/{_fmt(summ.get('median'))}/{_fmt(summ.get('max'))}"
        )
    if not use_events:
        return f"events=off, {ev_body}"
    if ev_thr is None:
        return str(ev_body)
    return f"events<={_fmt(ev_thr)}, {ev_body}"

def _auto_optuna_fallback_summary_text(tel: dict | None) -> str:
    t = dict(tel or {})
    fallback_tel = dict(t.get("fallback_telemetry", {}) or {})
    constrained_txt = _auto_optuna_telemetry_text(t)
    fallback_txt = _auto_optuna_telemetry_text(fallback_tel)
    events_txt = _auto_optuna_events_debug_text(t, 3)

    parts = []
    if constrained_txt:
        parts.append(f"constrained {constrained_txt}")
    if fallback_txt:
        parts.append(f"fallback {fallback_txt}")
    if events_txt:
        parts.append(str(events_txt))
    return "; ".join(parts)

def _auto_optuna_telemetry_rollup(items: list[dict] | None) -> dict:
    arr = [dict(x or {}) for x in list(items or []) if isinstance(x, dict) and x]
    if not arr:
        return {}

    out = {
        "run_trials": 0,
        "complete_trials": 0,
        "failed_trials": 0,
        "startup_complete": 0,
        "model_complete": 0,
        "duplicate_skips": 0,
        "duplicate_replays": 0,
        "duplicate_reserved": 0,
        "constraints_active": False,
        "feasible_trials": 0,
        "infeasible_trials": 0,
        "best_raw_value": None,
        "best_feasible_value": None,
        "violation_counts": {"ripple": 0, "events": 0, "boost": 0},
    }

    for t in arr:
        out["run_trials"] += int(t.get("run_trials", 0) or 0)
        out["complete_trials"] += int(t.get("complete_trials", 0) or 0)
        out["failed_trials"] += int(t.get("failed_trials", 0) or 0)
        out["startup_complete"] += int(t.get("startup_complete", 0) or 0)
        out["model_complete"] += int(t.get("model_complete", 0) or 0)
        out["duplicate_skips"] += int(t.get("duplicate_skips", 0) or 0)
        out["duplicate_replays"] += int(t.get("duplicate_replays", 0) or 0)
        out["duplicate_reserved"] += int(t.get("duplicate_reserved", 0) or 0)

        if bool(t.get("constraints_active", False)):
            out["constraints_active"] = True
            out["feasible_trials"] += int(t.get("feasible_trials", 0) or 0)
            out["infeasible_trials"] += int(t.get("infeasible_trials", 0) or 0)

        br = t.get("best_raw_value", None)
        if br is not None:
            try:
                brf = float(br)
                if np.isfinite(brf) and (
                    out["best_raw_value"] is None or brf > float(out["best_raw_value"])
                ):
                    out["best_raw_value"] = float(brf)
            except Exception:
                pass

        bf = t.get("best_feasible_value", None)
        if bf is not None:
            try:
                bff = float(bf)
                if np.isfinite(bff) and (
                    out["best_feasible_value"] is None or bff > float(out["best_feasible_value"])
                ):
                    out["best_feasible_value"] = float(bff)
            except Exception:
                pass

        vc = dict(t.get("violation_counts", {}) or {})
        out["violation_counts"]["ripple"] += int(vc.get("ripple", 0) or 0)
        out["violation_counts"]["events"] += int(vc.get("events", 0) or 0)
        out["violation_counts"]["boost"] += int(vc.get("boost", 0) or 0)

    return out
