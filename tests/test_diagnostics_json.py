import json
import math
from pathlib import Path

BEGIN = "--- DIAGNOSTICS_JSON_BEGIN ---"
END = "--- DIAGNOSTICS_JSON_END ---"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
GOLDEN = DATA / "golden"
SUMMARY = DATA / "summary"


def _extract_diagnostics_json(summary_text: str) -> dict:
    i = summary_text.find(BEGIN)
    j = summary_text.find(END)
    assert i != -1 and j != -1 and j > i, "Diagnostics JSON block markers not found"
    payload = summary_text[i + len(BEGIN) : j].strip()
    return json.loads(payload)


def _remove_keys(d: dict, keys: set[str]) -> dict:
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            if k in keys:
                continue
            out[k] = _remove_keys(v, keys)
        return out
    if isinstance(d, list):
        return [_remove_keys(x, keys) for x in d]
    return d


def _round_floats(obj, ndigits=6):
    # Reduce tiny float noise (numpy->python conversions etc.)
    if isinstance(obj, float):
        if math.isfinite(obj):
            return round(obj, ndigits)
        return obj
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, ndigits) for v in obj]
    return obj


def _normalize(diag: dict) -> dict:
    # Drop obviously volatile fields (add more as needed)
    drop = {
        # meta
        "version",
        "program",
        # anything path-like if you later add them
        "output_dir",
        "input_path",
    }
    diag2 = _remove_keys(diag, drop)
    # Float rounding
    return _round_floats(diag2, ndigits=6)


def test_diagnostics_json_schema_version():
    p = SUMMARY / "Summary_Linear_44100Hz.txt"
    txt = p.read_text(encoding="utf-8", errors="replace")
    diag = _extract_diagnostics_json(txt)

    assert "schema_version" in diag
    assert diag["schema_version"] == 1

    # Must-have top-level blocks
    for k in ("meta", "settings", "leveling", "left", "right"):
        assert k in diag, f"Missing key: {k}"


def test_diagnostics_json_matches_golden():
    p = SUMMARY / "Summary_Linear_44100Hz.txt"
    txt = p.read_text(encoding="utf-8", errors="replace")
    diag = _normalize(_extract_diagnostics_json(txt))

    g = json.loads((GOLDEN / "diag_linear_44100.json").read_text(encoding="utf-8"))
    g = _normalize(g)

    assert diag == g
