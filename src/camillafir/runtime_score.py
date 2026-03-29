import numpy as np

from .dsp.target_match import target_match_from_stats


def calculate_score(st, is_predicted=False):
    if not st:
        return 0.0

    conf = float(st.get("cmp_avg_confidence", st.get("avg_confidence", 0.0)) or 0.0)
    conf = float(np.clip(conf, 0.0, 100.0))

    _rms, match_pct = target_match_from_stats(
        st or {},
        include_filter=bool(is_predicted),
        use_confidence=True,
        use_smart_scan_range=True,
    )
    if match_pct is None:
        return float(np.clip(conf, 0.0, 99.0))
    match_pct = float(match_pct)

    base = 0.55 * match_pct + 0.35 * conf

    rt_bonus = 0.0
    try:
        rt = float(st.get("rt60_val", None)) if st.get("rt60_val", None) is not None else None
    except Exception:
        rt = None
    try:
        rel = float(st.get("rt60_reliability", 0.0) or 0.0)
    except Exception:
        rel = 0.0
    rel = float(np.clip(rel, 0.0, 1.0))

    if rt is not None and rt > 0:
        if rt <= 0.35:
            rt_bonus = ((0.35 - rt) / 0.25) * 15.0
        elif rt >= 0.55:
            rt_bonus = -min(15.0, ((rt - 0.55) / 0.35) * 15.0)
        rt_bonus *= rel

    events = st.get("cmp_reflections", st.get("reflections", [])) or []
    penalty_mult = 0.5 if is_predicted else 1.0
    event_penalty = min(8.0, float(len(events)) * 1.0) * penalty_mult

    score = base + rt_bonus - event_penalty
    return float(np.clip(score, 0.0, 99.0))
