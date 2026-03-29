import os
import re

DEFAULT_VERSION = "v.4.0.0"


def normalize_version(value: str | None, *, default: str = DEFAULT_VERSION) -> str:
    """Normalize runtime/build version text to the UI/export format `v.X.Y.Z`."""
    try:
        raw = str(value or "").strip()
    except Exception:
        raw = ""
    if not raw:
        return str(default)

    if raw.lower().startswith("v."):
        body = raw[2:]
    elif raw.lower().startswith("v"):
        body = raw[1:]
    else:
        body = raw
    body = re.sub(r"[^0-9A-Za-z._-]+", "-", str(body or "")).strip(".-_")
    if not body:
        return str(default)
    return f"v.{body}"


def resolve_version(*, default: str = DEFAULT_VERSION) -> str:
    env_version = str(os.environ.get("CAMILLAFIR_VERSION", "") or "").strip()
    if env_version:
        return normalize_version(env_version, default=default)

    try:
        from .build_version import VERSION as build_version

        return normalize_version(build_version, default=default)
    except Exception:
        return normalize_version(default, default=default)


VERSION = resolve_version()

