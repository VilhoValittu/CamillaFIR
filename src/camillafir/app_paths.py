import os
import re
import sys
from pathlib import Path

PROGRAM_NAME = "CamillaFIR"


def camillafir_data_dir() -> Path:
    """Return platform-correct user data directory for CamillaFIR."""
    home = Path.home()
    try:
        if os.name == "nt":
            appdata = str(os.environ.get("APPDATA", "") or "").strip()
            if appdata:
                return Path(appdata) / PROGRAM_NAME
            return home / "AppData" / "Roaming" / PROGRAM_NAME
        if sys.platform == "darwin":
            return home / "Library" / "Application Support" / PROGRAM_NAME
        xdg_data_home = str(os.environ.get("XDG_DATA_HOME", "") or "").strip()
        if xdg_data_home:
            return Path(xdg_data_home) / PROGRAM_NAME
        return home / ".local" / "share" / PROGRAM_NAME
    except Exception:
        return home / PROGRAM_NAME


def default_filters_export_base_dir() -> Path:
    """
    Return default base directory for exported filters.

    Prefer Documents on all platforms for discoverability.
    """
    home = Path.home()
    try:
        if os.name == "nt":
            userprofile = str(os.environ.get("USERPROFILE", "") or "").strip()
            base_home = Path(userprofile) if userprofile else home
        else:
            base_home = home
        return base_home / "Documents" / PROGRAM_NAME / "filters"
    except Exception:
        return home / "Documents" / PROGRAM_NAME / "filters"


def program_version_token(version: str | None, *, default: str = "v0") -> str:
    """Create filesystem-safe version token for folder/file names."""
    try:
        raw = str(version or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return str(default)
    if raw.lower().startswith("v."):
        raw = "v" + raw[2:]
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-_")
    return str(raw or default)


def safe_filters_dir(path: str | None, *, program_version: str | None = None) -> str:
    """
    Resolve a writable export filters directory with robust fallbacks.

    Guards:
    - avoid top-level/root paths such as `/filters`
    - fall back to platform user data directory if target is not writable
    """
    version_tag = program_version_token(program_version, default="v0")
    fallback_base = default_filters_export_base_dir()
    fallback = fallback_base / version_tag
    try:
        p_base = Path(path).expanduser() if path else fallback_base
    except Exception:
        p_base = fallback_base

    try:
        if p_base.is_absolute() and len(p_base.parts) <= 2:
            p_base = fallback_base
    except Exception:
        p_base = fallback_base

    try:
        if str(getattr(p_base, "name", "")).strip() != str(version_tag):
            p = p_base / version_tag
        else:
            p = p_base
    except Exception:
        p = fallback

    try:
        p.mkdir(parents=True, exist_ok=True)
        return str(p.resolve())
    except OSError:
        p = fallback
    except Exception:
        p = fallback

    try:
        p.mkdir(parents=True, exist_ok=True)
        return str(p.resolve())
    except Exception:
        last = Path.cwd() / "filters" / version_tag
        try:
            last.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(last)
