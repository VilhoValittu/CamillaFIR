"""Helper utilities for process_run() wiring in camillafir.py.

These were previously defined inline in camillafir.py alongside
_process_run_legacy(). They belong to orchestration, not UI.
"""
from __future__ import annotations

import os
import re
import typing
import unicodedata


def auto_target_mode_norm(mode: typing.Any) -> str:
    """
    Normalisoi AUTO-tilan target-kayran valintatavan.

    Palauttaa joko `auto` (etsi paras target-kayra) tai
    `selected` (kayta kayttajan valitsemaa target-kayraa).
    """
    try:
        m = str(mode or "auto").strip().lower()
    except Exception:
        m = "auto"
    if m in ("selected", "manual", "fixed", "user"):
        return "selected"
    if m == "adaptive":
        return "adaptive"
    return "auto"


def auto_target_selection_method_text(method: typing.Any) -> str:
    """
    Muuntaa auto-target-valinnan metodit luettaviksi loki- ja UI-teksteiksi.
    """
    try:
        key = str(method or "").strip().lower()
    except Exception:
        key = ""

    mapping = {
        "adaptive": "adaptive (synthesized from measurements)",
        "cache_signature_hit": "cache hit (same measurements + settings, no target re-evaluation)",
        "cache_measurement": "measurement cache seed",
        "cache_signature": "signature cache seed",
        "trial_with_cache_wildcard": "trial winner with cache wildcard",
        "top3x10_trials": "trial comparison",
        "top3x10_trials_rank_tie_composite": "trial comparison with rank tie-break",
        "fit_rms": "quick fit preselect",
    }
    return str(mapping.get(key, key or "unknown"))


def slugify_filename_token(value: typing.Any, *, default: str = "target", max_len: int = 48) -> str:
    """Muuntaa tekstin turvalliseksi tiedostonimiosaksi."""
    try:
        raw = str(value or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return default

    try:
        txt = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    except Exception:
        txt = raw
    txt = re.sub(r"[^A-Za-z0-9]+", "-", txt).strip("-").lower()
    if not txt:
        return default
    if len(txt) > int(max_len):
        txt = txt[: int(max_len)].rstrip("-")
    return txt or default


def pick_target_curve_label(data: dict) -> str:
    """Valitsee target curven nimen vientitiedostonimiin."""
    try:
        up = data.get("hc_custom_file")
        if isinstance(up, dict):
            for k in ("filename", "name", "file_name"):
                v = up.get(k)
                if isinstance(v, str) and v.strip():
                    return os.path.splitext(os.path.basename(v.strip()))[0]
    except Exception:
        pass

    try:
        p = str(data.get("local_path_house") or "").strip()
    except Exception:
        p = ""
    if p:
        return os.path.splitext(os.path.basename(p))[0]

    try:
        hc_mode = str(data.get("hc_mode") or "").strip()
    except Exception:
        hc_mode = ""
    if hc_mode:
        return hc_mode

    try:
        src = str(data.get("hc_source") or "").strip()
    except Exception:
        src = ""
    if src:
        return src
    return "Target"


def has_uploaded_target_file(data: dict) -> bool:
    """Checks if UI data contains an uploaded custom target file."""
    try:
        up = data.get("hc_custom_file", None)
    except Exception:
        up = None

    def _is_uploaded_file_obj(v: typing.Any) -> bool:
        if not isinstance(v, dict):
            return False
        try:
            content = v.get("content", None)
            if isinstance(content, (bytes, bytearray)) and len(content) > 0:
                return True
        except Exception:
            pass
        for k in ("filename", "name", "file_name"):
            try:
                s = str(v.get(k, "") or "").strip()
            except Exception:
                s = ""
            if s:
                return True
        return False

    if _is_uploaded_file_obj(up):
        return True
    if isinstance(up, list):
        for item in up:
            if _is_uploaded_file_obj(item):
                return True
    return False


def resolve_ui_stats_fs(ui_stats_fs: typing.Any, selected_fs: typing.Any) -> int:
    """
    Valitsee UI-statistiikalle oikean sample raten.

    Priorisoi dashboardille valitun analyysinopeuden (`ui_stats_fs`) ja
    kaatuu turvallisesti UI-valintaan (`selected_fs`) tai 44100 Hz:iin.
    """
    for cand in (ui_stats_fs, selected_fs, 44100):
        try:
            fs_i = int(cand)
        except Exception:
            continue
        if fs_i > 0:
            return fs_i
    return 44100
