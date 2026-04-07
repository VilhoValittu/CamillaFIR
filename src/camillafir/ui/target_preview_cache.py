"""Cached measurement parsing helpers for the target preview."""
from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np

from ..io import measurements_txt, measurements_wav
from ..io.measurement_bundle import TransferData

_CURVE_CACHE: dict[tuple[Any, ...], tuple[np.ndarray | None, np.ndarray | None]] = {}
_TRANSFER_CACHE: dict[tuple[Any, ...], TransferData | None] = {}


def _coerce_content_bytes(content: Any) -> bytes:
    if isinstance(content, bytes):
        return content
    if isinstance(content, bytearray):
        return bytes(content)
    if content is None:
        return b""
    return str(content).encode("utf-8", errors="ignore")


def _normalize_curve(freqs, mags):
    try:
        ff = np.asarray(freqs, dtype=float)
        mm = np.asarray(mags, dtype=float)
        if ff.size < 8 or mm.size != ff.size:
            return None, None
        mask = np.isfinite(ff) & np.isfinite(mm) & (ff > 0)
        ff, mm = ff[mask], mm[mask]
        order = np.argsort(ff)
        ff, mm = ff[order], mm[order]
        uniq, idx = np.unique(ff, return_index=True)
        if uniq.size < 8:
            return None, None
        return uniq, mm[idx]
    except Exception:
        return None, None


def _clean_local_path(path_raw: Any) -> str:
    try:
        path = str(path_raw or "").strip().strip('"').strip("'")
    except Exception:
        path = ""
    if not path:
        return ""
    try:
        return os.path.abspath(path)
    except Exception:
        return path


def _upload_cache_key(upload: dict[str, Any] | None, *, pre_ms: float, post_ms: float, smoothing_level: int) -> tuple[Any, ...] | None:
    if not isinstance(upload, dict):
        return None
    content = _coerce_content_bytes(upload.get("content", b""))
    if not content:
        return None
    digest = str(upload.get("content_sha256", "") or "")
    if not digest:
        digest = hashlib.sha256(content).hexdigest()
    return (
        "upload",
        str(upload.get("filename", "") or ""),
        digest,
        float(pre_ms),
        float(post_ms),
        int(smoothing_level),
    )


def _path_cache_key(path_raw: Any, *, pre_ms: float, post_ms: float, smoothing_level: int) -> tuple[str, tuple[Any, ...] | None]:
    path = _clean_local_path(path_raw)
    if not path:
        return "", None
    try:
        stat = os.stat(path)
    except OSError:
        return path, None
    return (
        path,
        (
            "path",
            os.path.normcase(os.path.normpath(path)),
            int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
            int(stat.st_size),
            float(pre_ms),
            float(post_ms),
            int(smoothing_level),
        ),
    )


def _parse_upload_curve(upload: dict[str, Any], *, pre_ms: float, post_ms: float, smoothing_level: int):
    content = _coerce_content_bytes(upload.get("content", b""))
    if not content:
        return None, None

    name = str(upload.get("filename", "") or "").strip().lower()
    is_wav = name.endswith(".wav") or (len(content) >= 4 and content[:4] == b"RIFF")
    try:
        if is_wav:
            ff, mm, _ = measurements_wav.parse_measurements_from_wav_bytes(
                content,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
                logger=None,
            )
        else:
            ff, mm, _ = measurements_txt.parse_measurements_from_bytes(content)
        return _normalize_curve(ff, mm)
    except Exception:
        return None, None


def _parse_upload_transfer(
    upload: dict[str, Any],
    *,
    pre_ms: float,
    post_ms: float,
    smoothing_level: int,
    label: str,
) -> TransferData | None:
    content = _coerce_content_bytes(upload.get("content", b""))
    if not content:
        return None
    name = str(upload.get("filename", "") or "").strip().lower()
    is_wav = name.endswith(".wav") or (len(content) >= 4 and content[:4] == b"RIFF")
    if not is_wav:
        return None
    try:
        return measurements_wav.parse_coherent_transfer_from_wav_bytes(
            content,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
            label=label,
            logger=None,
        )
    except Exception:
        return None


def _parse_path_curve(path: str, *, pre_ms: float, post_ms: float, smoothing_level: int):
    try:
        if path.lower().endswith(".wav"):
            ff, mm, _ = measurements_wav.parse_measurements_from_wav_path(
                path,
                pre_ms=pre_ms,
                post_ms=post_ms,
                smoothing_level=smoothing_level,
                logger=None,
            )
        else:
            ff, mm, _ = measurements_txt.parse_measurements_from_path(path, logger=None)
        return _normalize_curve(ff, mm)
    except Exception:
        return None, None


def _parse_path_transfer(
    path: str,
    *,
    pre_ms: float,
    post_ms: float,
    smoothing_level: int,
    label: str,
) -> TransferData | None:
    try:
        if not str(path or "").strip().lower().endswith(".wav"):
            return None
        return measurements_wav.parse_coherent_transfer_from_wav_path(
            path,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
            label=label,
            logger=None,
        )
    except Exception:
        return None


def load_upload_measurement_curve(upload: dict[str, Any] | None, *, pre_ms: float, post_ms: float, smoothing_level: int):
    key = _upload_cache_key(upload, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=smoothing_level)
    if key is None or not isinstance(upload, dict):
        return None, None
    if key not in _CURVE_CACHE:
        _CURVE_CACHE[key] = _parse_upload_curve(
            upload,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
        )
    return _CURVE_CACHE[key]


def load_upload_measurement_transfer(
    upload: dict[str, Any] | None,
    *,
    pre_ms: float,
    post_ms: float,
    smoothing_level: int,
    label: str,
) -> TransferData | None:
    key = _upload_cache_key(upload, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=smoothing_level)
    if key is None or not isinstance(upload, dict):
        return None
    transfer_key = ("transfer",) + tuple(key) + (str(label or ""),)
    if transfer_key not in _TRANSFER_CACHE:
        _TRANSFER_CACHE[transfer_key] = _parse_upload_transfer(
            upload,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
            label=label,
        )
    return _TRANSFER_CACHE[transfer_key]


def load_path_measurement_curve(path_raw: Any, *, pre_ms: float, post_ms: float, smoothing_level: int):
    path, key = _path_cache_key(path_raw, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=smoothing_level)
    if not path:
        return None, None
    if key is None:
        return _parse_path_curve(
            path,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
        )
    if key not in _CURVE_CACHE:
        _CURVE_CACHE[key] = _parse_path_curve(
            path,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
        )
    return _CURVE_CACHE[key]


def load_path_measurement_transfer(
    path_raw: Any,
    *,
    pre_ms: float,
    post_ms: float,
    smoothing_level: int,
    label: str,
) -> TransferData | None:
    path, key = _path_cache_key(path_raw, pre_ms=pre_ms, post_ms=post_ms, smoothing_level=smoothing_level)
    if not path:
        return None
    transfer_key = None
    if key is not None:
        transfer_key = ("transfer",) + tuple(key) + (str(label or ""),)
    if transfer_key is None:
        return _parse_path_transfer(
            path,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
            label=label,
        )
    if transfer_key not in _TRANSFER_CACHE:
        _TRANSFER_CACHE[transfer_key] = _parse_path_transfer(
            path,
            pre_ms=pre_ms,
            post_ms=post_ms,
            smoothing_level=smoothing_level,
            label=label,
        )
    return _TRANSFER_CACHE[transfer_key]
