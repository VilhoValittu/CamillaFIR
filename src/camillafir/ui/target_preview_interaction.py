from __future__ import annotations

import math
import re
import statistics

from ..application.house_curve_service import MANUAL_TARGET_TILT_PIVOT_HZ


_PATH_TOKEN_RE = re.compile(r"[ML]|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_PATH_VALIDATE_RE = re.compile(
    r"\s*(?:[ML]\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*,\s*|\s+)"
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*)+"
)
_LAST_BASE_POINTS_ID: int | None = None
_LAST_BASE_MANUAL_DB = 0.0
_LAST_TILT_HANDLE_POINTS_ID: int | None = None
_LAST_BASE_MANUAL_TILT = 0.0

_TILT_HANDLE_CENTER_FREQ_HZ = 16000.0
_TILT_HANDLE_SPAN_FACTOR = 1.18
_PREVIEW_LEVEL_WINDOW_FILL = "rgba(148,163,184,0.18)"
_PREVIEW_MARKER_LINE = "rgba(15,23,42,0.35)"


def round_manual_target_tilt_db_per_oct(value: float) -> float:
    try:
        return round(float(value) * 10.0) / 10.0
    except Exception:
        return 0.0


def round_manual_target_db(value: float) -> float:
    try:
        return round(float(value) * 10.0) / 10.0
    except Exception:
        return 0.0


def clamp_manual_target_db(value: float, y_min: float = -10.0, y_max: float = 20.0) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.0
    lo = float(y_min)
    hi = float(y_max)
    if lo > hi:
        lo, hi = hi, lo
    return min(max(v, lo), hi)


def build_target_curve_path(freq_axis, target_curve_display) -> str:
    points = [
        (float(x), float(y))
        for x, y in zip(freq_axis, target_curve_display)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if not points:
        return ""
    path_parts = [f"M {_format_svg_number(points[0][0])} {_format_svg_number(points[0][1])}"]
    for x, y in points[1:]:
        path_parts.append(f"L {_format_svg_number(x)} {_format_svg_number(y)}")
    return " ".join(path_parts)


def build_draggable_target_shape(freq_axis, target_curve_display) -> dict:
    return {
        "type": "path",
        "path": build_target_curve_path(freq_axis, target_curve_display),
        "xref": "x",
        "yref": "y",
        "layer": "above",
        "editable": True,
        "fillcolor": "rgba(0,0,0,0)",
        "line": {
            "color": "#4caf50",
            "width": 2.0,
        },
    }


def build_tilt_handle_path(handle_y_db: float) -> str:
    center_freq = float(_TILT_HANDLE_CENTER_FREQ_HZ)
    span = float(_TILT_HANDLE_SPAN_FACTOR)
    x0 = max(1.0, center_freq / span)
    x1 = min(20000.0, center_freq * span)
    return build_target_curve_path([x0, x1], [float(handle_y_db), float(handle_y_db)])


def build_draggable_tilt_handle_shape(handle_y_db: float) -> dict:
    return {
        "type": "path",
        "path": build_tilt_handle_path(handle_y_db),
        "xref": "x",
        "yref": "y",
        "layer": "above",
        "editable": True,
        "fillcolor": "rgba(0,0,0,0)",
        "line": {
            "color": "#f4a261",
            "width": 9.0,
        },
    }


def build_level_window_trace(
    x_min: float,
    x_max: float,
    *,
    y_min: float = -10.0,
    y_max: float = 20.0,
) -> dict:
    x0 = max(1.0, float(x_min))
    x1 = max(1.0, float(x_max))
    if x0 > x1:
        x0, x1 = x1, x0
    return {
        "type": "scatter",
        "mode": "lines",
        "x": [x0, x0, x1, x1],
        "y": [float(y_min), float(y_max), float(y_max), float(y_min)],
        "fill": "toself",
        "fillcolor": _PREVIEW_LEVEL_WINDOW_FILL,
        "line": {"color": "rgba(148,163,184,0.0)", "width": 0},
        "hoverinfo": "skip",
        "hovertemplate": None,
        "showlegend": False,
        "name": "",
    }


def build_vertical_marker_trace(
    x_value: float,
    *,
    y_min: float = -10.0,
    y_max: float = 20.0,
) -> dict:
    x = max(1.0, float(x_value))
    return {
        "type": "scatter",
        "mode": "lines",
        "x": [x, x],
        "y": [float(y_min), float(y_max)],
        "line": {"color": _PREVIEW_MARKER_LINE, "width": 1},
        "hoverinfo": "skip",
        "hovertemplate": None,
        "showlegend": False,
        "name": "",
    }


def extract_vertical_shift_from_shape_relayout(
    payload: dict,
    base_path_points: list[tuple[float, float]],
) -> float | None:
    global _LAST_BASE_MANUAL_DB, _LAST_BASE_POINTS_ID

    path = _extract_shape_path(payload, shape_index=0)
    if not path or not base_path_points:
        return None

    updated_points = parse_svg_path_points(path)
    if not updated_points:
        return None
    if _path_points_match(base_path_points, updated_points):
        return None

    base_points_id = id(base_path_points)
    if base_points_id != _LAST_BASE_POINTS_ID:
        _LAST_BASE_POINTS_ID = base_points_id
        _LAST_BASE_MANUAL_DB = _current_manual_target_db()

    point_count = min(len(base_path_points), len(updated_points))
    if point_count <= 0:
        return None

    deltas = []
    for index in range(point_count):
        _, base_y = base_path_points[index]
        _, updated_y = updated_points[index]
        if math.isfinite(base_y) and math.isfinite(updated_y):
            deltas.append(float(updated_y) - float(base_y))
    if not deltas:
        return None

    sample = deltas[: min(len(deltas), 16)]
    delta = statistics.median(sample)
    new_manual = _LAST_BASE_MANUAL_DB + float(delta)
    return clamp_manual_target_db(round_manual_target_db(new_manual))


def extract_target_tilt_from_shape_relayout(
    payload: dict,
    base_handle_points: list[tuple[float, float]],
    *,
    pivot_hz: float = MANUAL_TARGET_TILT_PIVOT_HZ,
) -> float | None:
    global _LAST_TILT_HANDLE_POINTS_ID, _LAST_BASE_MANUAL_TILT

    path = _extract_shape_path(payload, shape_index=1)
    if not path or not base_handle_points:
        return None

    updated_points = parse_svg_path_points(path)
    if not updated_points:
        return None
    if _path_points_match(base_handle_points, updated_points):
        return None

    handle_points_id = id(base_handle_points)
    if handle_points_id != _LAST_TILT_HANDLE_POINTS_ID:
        _LAST_TILT_HANDLE_POINTS_ID = handle_points_id
        _LAST_BASE_MANUAL_TILT = _current_manual_target_tilt_db_per_oct()

    try:
        pivot = float(pivot_hz)
    except Exception:
        pivot = MANUAL_TARGET_TILT_PIVOT_HZ
    if not math.isfinite(pivot) or pivot <= 0.0:
        pivot = MANUAL_TARGET_TILT_PIVOT_HZ

    handle_freq_hz = _path_center_frequency_hz(base_handle_points)
    if handle_freq_hz is None:
        return None

    log_oct = math.log2(float(pivot) / float(handle_freq_hz))
    if not math.isfinite(log_oct) or abs(log_oct) <= 1e-9:
        return None

    base_y = _path_center_y_db(base_handle_points)
    updated_y = _path_center_y_db(updated_points)
    if base_y is None or updated_y is None:
        return None

    delta_y = float(updated_y) - float(base_y)
    new_tilt = _LAST_BASE_MANUAL_TILT + (float(delta_y) / float(log_oct))
    return round_manual_target_tilt_db_per_oct(new_tilt)


def parse_svg_path_points(path: str) -> list[tuple[float, float]]:
    if not isinstance(path, str) or not _PATH_VALIDATE_RE.fullmatch(path.strip()):
        return []

    tokens = _PATH_TOKEN_RE.findall(path)
    points: list[tuple[float, float]] = []
    idx = 0
    while idx < len(tokens):
        command = tokens[idx]
        idx += 1
        if command not in {"M", "L"} or idx + 1 >= len(tokens):
            return []
        try:
            x = float(tokens[idx])
            y = float(tokens[idx + 1])
        except Exception:
            return []
        points.append((x, y))
        idx += 2
    return points


def _extract_shape_path(payload: dict, *, shape_index: int) -> str | None:
    if not isinstance(payload, dict):
        return None

    direct_path = payload.get(f"shapes[{int(shape_index)}].path")
    if isinstance(direct_path, str):
        return direct_path

    indexed_shape = payload.get(f"shapes[{int(shape_index)}]")
    if isinstance(indexed_shape, dict):
        path = indexed_shape.get("path")
        if isinstance(path, str):
            return path

    shapes = payload.get("shapes")
    if isinstance(shapes, list) and len(shapes) > int(shape_index):
        shape = shapes[int(shape_index)]
        if isinstance(shape, dict):
            path = shape.get("path")
            if isinstance(path, str):
                return path

    return None


def _current_manual_target_db() -> float:
    from . import ng_controls as ctrl

    try:
        value = float(ctrl.value("lvl_manual_db", 0.0) or 0.0)
    except Exception:
        value = 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def _current_manual_target_tilt_db_per_oct() -> float:
    from . import ng_controls as ctrl

    try:
        value = float(ctrl.value("manual_target_tilt_db_per_oct", 0.0) or 0.0)
    except Exception:
        value = 0.0
    if not math.isfinite(value):
        return 0.0
    return value


def _path_points_match(
    base_points: list[tuple[float, float]],
    updated_points: list[tuple[float, float]],
    *,
    tol: float = 1e-9,
) -> bool:
    if len(base_points) != len(updated_points):
        return False
    for (base_x, base_y), (updated_x, updated_y) in zip(base_points, updated_points):
        if abs(float(base_x) - float(updated_x)) > float(tol):
            return False
        if abs(float(base_y) - float(updated_y)) > float(tol):
            return False
    return True


def _path_center_y_db(points: list[tuple[float, float]]) -> float | None:
    y_values = [float(y) for _, y in points if math.isfinite(float(y))]
    if not y_values:
        return None
    return float(statistics.median(y_values))


def _path_center_frequency_hz(points: list[tuple[float, float]]) -> float | None:
    x_values = [float(x) for x, _ in points if math.isfinite(float(x)) and float(x) > 0.0]
    if not x_values:
        return None
    try:
        return float(math.exp(statistics.mean(math.log(x) for x in x_values)))
    except Exception:
        return None


def _format_svg_number(value: float) -> str:
    return f"{float(value):.12g}"
