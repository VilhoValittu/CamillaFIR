from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger("CamillaFIR")

_STATUS_BASE_MSG = ""
_STATUS_DOM_READY = False
_STATUS_LAST_TEXT = ""
_STATUS_SUMMARY_TEXT = ""
_STATUS_INFO_TEXT = ""
_AUTO_SELECTED_BAR_MSG = ""
_AUTO_STATUS_DETAILS: list[str] = []
_AUTO_STATUS_DETAIL_MAX = 80
_AUTO_STATUS_LAST_DETAIL = ""
_RUN_WALL_CLOCK_TEXT = ""
_STATUS_RENDERER: Callable[..., None] | None = None


def set_status_renderer(renderer: Callable[..., None] | None) -> None:
    global _STATUS_RENDERER
    _STATUS_RENDERER = renderer if callable(renderer) else None


def mark_status_dom_ready(is_ready: bool = True) -> None:
    global _STATUS_DOM_READY
    _STATUS_DOM_READY = bool(is_ready)


def is_status_dom_ready() -> bool:
    return bool(_STATUS_DOM_READY)


def _status_split_elapsed_suffix(msg: str) -> tuple[str, str]:
    try:
        s = str(msg or "").strip()
    except Exception:
        s = ""
    if not s:
        return "", ""
    try:
        match = re.match(r"^(.*?)(\|\s*\d+(?:\.\d+)?\s*s)\s*$", s, flags=re.IGNORECASE)
    except Exception:
        match = None
    if not match:
        return str(s), ""
    return str(match.group(1) or "").strip(), str(match.group(2) or "").strip()


def _compact_auto_status_core(core: str) -> str:
    try:
        s = str(core or "").strip()
    except Exception:
        s = ""
    prefix = "CamillaFIR automatic mode:"
    if not s.startswith(prefix):
        return s
    after = str(s[len(prefix):] or "").strip()
    low = after.lower()
    if low.startswith("target shortlist"):
        phase = "target shortlist"
    elif low.startswith("target preselect"):
        phase = "target preselect"
    elif low.startswith("selecting target curve"):
        phase = "selecting target curve"
    elif low.startswith("target trials"):
        phase = "target trials"
    elif low.startswith("phase1 done"):
        phase = "phase1 done"
    elif low.startswith("local refine"):
        phase = "local refine"
    elif low.startswith("target finalize"):
        phase = "target finalize"
    elif low.startswith("preset search"):
        phase = "preset search"
    elif low.startswith("protection model"):
        phase = "protection model"
    elif low.startswith("hpf auto-fit"):
        phase = "hpf auto-fit"
    elif low.startswith("finalize"):
        phase = "finalize"
    elif low.startswith("target curve mode="):
        phase = "target curve mode"
    elif low.startswith("target preselect winner"):
        phase = "target preselect winner"
    elif low.startswith("adaptive target"):
        phase = "adaptive target"
    elif low.startswith("init"):
        phase = "init"
    else:
        try:
            phase = re.sub(r"\s*\(.*\)\s*$", "", after).strip()
        except Exception:
            phase = after
        if not phase:
            phase = "running"
    return f"{prefix} {phase}".strip()


def _status_compact_with_detail(msg) -> tuple[str, str | None]:
    try:
        raw = str(msg or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return "CamillaFIR running", None
    core, elapsed = _status_split_elapsed_suffix(raw)
    compact_core = _compact_auto_status_core(core)
    out = compact_core
    if elapsed:
        out = f"{compact_core} {elapsed}"
    detail = None
    if (
        isinstance(core, str)
        and core.startswith("CamillaFIR automatic mode:")
        and str(core).strip() != str(compact_core).strip()
    ):
        detail = str(core).strip()
    return out, detail


def _status_base_from_text(msg) -> str:
    text, _detail = _status_compact_with_detail(msg)
    core, _elapsed = _status_split_elapsed_suffix(text)
    core = str(core or "").strip()
    return core or "CamillaFIR running"


def _normalize_auto_selected_text(msg) -> str:
    try:
        txt = str(msg or "").strip()
    except Exception:
        txt = ""
    return txt


def _normalize_status_notice_text(msg) -> str:
    try:
        txt = str(msg or "").strip()
    except Exception:
        txt = ""
    return txt


def get_status_base_message(default: str = "CamillaFIR running") -> str:
    try:
        value = str(_STATUS_BASE_MSG or "").strip()
    except Exception:
        value = ""
    return value or str(default)


def set_run_wall_clock_text(value) -> None:
    global _RUN_WALL_CLOCK_TEXT
    try:
        _RUN_WALL_CLOCK_TEXT = str(value or "").strip()
    except Exception:
        _RUN_WALL_CLOCK_TEXT = ""


def get_run_wall_clock_text(default: str = "") -> str:
    try:
        value = str(_RUN_WALL_CLOCK_TEXT or "").strip()
    except Exception:
        value = ""
    return value or str(default or "")


def get_status_snapshot() -> dict:
    details = [str(item or "") for item in list(_AUTO_STATUS_DETAILS or [])]
    return {
        "status_base_message": str(_STATUS_BASE_MSG or ""),
        "status_dom_ready": bool(_STATUS_DOM_READY),
        "status_last_text": str(_STATUS_LAST_TEXT or ""),
        "status_summary_text": _normalize_status_notice_text(_STATUS_SUMMARY_TEXT),
        "status_info_text": _normalize_status_notice_text(_STATUS_INFO_TEXT),
        "auto_selected_bar_text": _normalize_auto_selected_text(_AUTO_SELECTED_BAR_MSG),
        "auto_status_details": details,
        "auto_status_detail_body": "\n".join(details),
        "run_wall_clock_text": get_run_wall_clock_text(""),
    }


def _notify_renderer(event: str) -> None:
    renderer = _STATUS_RENDERER
    if not callable(renderer):
        return
    try:
        renderer(event=str(event or ""), snapshot=get_status_snapshot())
    except Exception:
        logger.debug("UI status renderer update failed", exc_info=True)


def update_status(msg) -> None:
    global _STATUS_BASE_MSG, _STATUS_LAST_TEXT, _AUTO_STATUS_DETAILS, _AUTO_STATUS_LAST_DETAIL
    text, detail = _status_compact_with_detail(msg)
    _STATUS_BASE_MSG = _status_base_from_text(text)
    _STATUS_LAST_TEXT = str(text or "")
    if isinstance(detail, str) and detail.strip():
        detail_txt = str(detail).strip()
        if detail_txt != str(_AUTO_STATUS_LAST_DETAIL or ""):
            _AUTO_STATUS_LAST_DETAIL = detail_txt
            _AUTO_STATUS_DETAILS = list(_AUTO_STATUS_DETAILS or []) + [detail_txt]
            max_n = int(max(10, _AUTO_STATUS_DETAIL_MAX))
            if len(_AUTO_STATUS_DETAILS) > max_n:
                _AUTO_STATUS_DETAILS = list(_AUTO_STATUS_DETAILS[-max_n:])
    _notify_renderer("status")


def update_auto_selected_bar(msg) -> None:
    global _AUTO_SELECTED_BAR_MSG
    _AUTO_SELECTED_BAR_MSG = _normalize_auto_selected_text(msg)
    _notify_renderer("auto_selected_bar")


def update_status_notices(*, summary_text=None, info_text=None) -> None:
    global _STATUS_SUMMARY_TEXT, _STATUS_INFO_TEXT
    if summary_text is not None:
        _STATUS_SUMMARY_TEXT = _normalize_status_notice_text(summary_text)
    if info_text is not None:
        _STATUS_INFO_TEXT = _normalize_status_notice_text(info_text)
    _notify_renderer("status_notices")


def reset_auto_status_details() -> None:
    global _AUTO_STATUS_DETAILS, _AUTO_STATUS_LAST_DETAIL
    _AUTO_STATUS_DETAILS = []
    _AUTO_STATUS_LAST_DETAIL = ""
    _notify_renderer("reset_auto_status_details")


__all__ = [
    "get_run_wall_clock_text",
    "get_status_base_message",
    "get_status_snapshot",
    "is_status_dom_ready",
    "mark_status_dom_ready",
    "reset_auto_status_details",
    "set_run_wall_clock_text",
    "set_status_renderer",
    "update_auto_selected_bar",
    "update_status",
    "update_status_notices",
]
