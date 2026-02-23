import json
from pathlib import Path

BEGIN = "--- DIAGNOSTICS_JSON_BEGIN ---"
END = "--- DIAGNOSTICS_JSON_END ---"

ROOT = Path(__file__).resolve().parent
SUMMARY_DIR = ROOT / "data" / "summary"

TOL = 1e-6


def _extract_diag(summary_path: Path) -> dict:
    txt = summary_path.read_text(encoding="utf-8", errors="replace")
    i = txt.find(BEGIN)
    j = txt.find(END)
    assert i != -1 and j != -1 and j > i, "Diagnostics JSON block markers not found"
    payload = txt[i + len(BEGIN): j].strip()
    return json.loads(payload)


def _find_stage_probes_in_stats(stats: dict):
    """
    Try to locate stage probes in common shapes:
    - stats["stage_probes"] as list[dict(stage=..., boost_peak_db=...)]
    - stats["stage_probes"] as dict[name -> dict(metrics)]
    - stats["stage_probe"] or stats["stage_checkpoints"] etc.
    Returns (ordered_list) where each item = {"name": str, "metrics": dict}
    """
    if not isinstance(stats, dict):
        return []

    candidates = []
    for k in ("stage_probes", "stage_probe", "stage_checkpoints", "stage_probing", "probes"):
        if k in stats:
            candidates.append(stats[k])

    # If nothing obvious, try fuzzy match
    if not candidates:
        for k, v in stats.items():
            if isinstance(k, str) and "stage" in k.lower() and isinstance(v, (list, dict)):
                candidates.append(v)

    if not candidates:
        return []

    sp = candidates[0]

    # Case A: list of dicts with "stage"/"name"
    if isinstance(sp, list):
        out = []
        for item in sp:
            if not isinstance(item, dict):
                continue
            name = item.get("stage") or item.get("name") or item.get("stage_name") or "unknown"
            out.append({"name": str(name), "metrics": item})
        return out

    # Case B: dict keyed by stage name -> metrics dict
    if isinstance(sp, dict):
        out = []
        # Preserve insertion order if present (Python 3.7+ keeps it)
        for name, metrics in sp.items():
            if isinstance(metrics, dict):
                out.append({"name": str(name), "metrics": metrics})
        return out

    return []


def _get_boost_peak(metrics: dict):
    """
    Find boost peak db in a probe metrics dict using common key names.
    """
    if not isinstance(metrics, dict):
        return None
    for k in ("boost_peak_db", "boostpk_db", "boost_pk_db", "boostPk", "BoostPk", "boost_pk"):
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                pass
    # Some probe formats store generic peaks
    for k in ("peak_boost_db", "max_boost_db"):
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                pass
    return None


def _is_clamp_stage(stage_name: str) -> bool:
    s = (stage_name or "").lower()
    # Treat any stage containing these as "clamp barrier"
    return any(token in s for token in ("clamp", "hard", "soft", "headroom", "normalize", "limit"))


def _assert_boost_not_increase_after_clamp(stage_list):
    """
    Find first clamp stage. From that stage onward, boost_peak must not increase.
    """
    # Build a clean sequence of (name, boost_peak)
    seq = []
    for it in stage_list:
        name = it["name"]
        b = _get_boost_peak(it["metrics"])
        if b is not None:
            seq.append((name, b))

    if len(seq) < 2:
        # Nothing to assert
        return

    # Find the first clamp barrier
    clamp_idx = None
    for idx, (name, _) in enumerate(seq):
        if _is_clamp_stage(name):
            clamp_idx = idx
            break

    if clamp_idx is None:
        # No clamp stages recorded => don't fail; just skip
        return

    barrier = seq[clamp_idx][1]
    for name, b in seq[clamp_idx + 1:]:
        assert b <= barrier + TOL, (
            f"Boost increased after clamp barrier.\n"
            f"Clamp stage: {seq[clamp_idx][0]} boost={barrier:.6f} dB\n"
            f"Later stage: {name} boost={b:.6f} dB"
        )
        barrier = min(barrier, b)  # once clamped, track tightest bound


def test_stage_smoke_boost_not_increase_after_clamp_linear_44100():
    p = SUMMARY_DIR / "Summary_Linear_44100Hz.txt"
    diag = _extract_diag(p)

    for side_key in ("left", "right"):
        stats = diag.get(side_key)
        if not isinstance(stats, dict):
            continue

        probes = _find_stage_probes_in_stats(stats)
        _assert_boost_not_increase_after_clamp(probes)
