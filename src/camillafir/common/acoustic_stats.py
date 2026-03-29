from __future__ import annotations

from ..dsp.target_match import target_match_from_stats as _target_match_ssot


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return float(max(lo, min(hi, float(x))))
    except Exception:
        return float(lo)


def calc_acoustic_score(conf_pct: float, match_pct: float, rt60_s: float | None = None, rt60_rel: float | None = None) -> float:
    conf = _clamp(conf_pct, 0.0, 100.0)
    match = _clamp(match_pct, 0.0, 100.0)

    base = 0.55 * match + 0.35 * conf

    rt_bonus_eff = 0.0
    try:
        if rt60_s is not None:
            rt60 = float(rt60_s)
            if rt60 > 0:
                rel = 1.0 if rt60_rel is None else _clamp(float(rt60_rel), 0.0, 1.0)
                rt_bonus = 15.0 * _clamp((0.35 - rt60) / 0.25, 0.0, 1.0)
                rt_bonus_eff = rt_bonus * rel
    except Exception:
        rt_bonus_eff = 0.0

    return _clamp(base + rt_bonus_eff, 0.0, 100.0)


def calc_ai_summary_from_stats(stats: dict) -> dict:
    stats = stats or {}
    conf = float(stats.get('cmp_avg_confidence', stats.get('avg_confidence', 0.0)) or 0.0)
    try:
        rms, match = _target_match_ssot(
            stats,
            include_filter=True,
            use_confidence=True,
            use_smart_scan_range=True,
        )
    except Exception:
        rms, match = None, None
    if match is None:
        return {"conf": conf, "rms": None, "match": None, "score": None}
    rt60 = stats.get("rt60_val", None)
    rt_rel = stats.get("rt60_reliability", None)
    score = calc_acoustic_score(conf, float(match), rt60_s=rt60, rt60_rel=rt_rel)
    return {
        "conf": conf,
        "rms": float(rms) if rms is not None else None,
        "match": float(match),
        "score": score,
        "rt60": float(rt60) if rt60 is not None else None,
        "rt60_rel": float(rt_rel) if rt_rel is not None else None,
    }


__all__ = ["_clamp", "calc_acoustic_score", "calc_ai_summary_from_stats"]
