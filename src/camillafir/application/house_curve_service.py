"""Application-level house-curve loading helpers."""

import numpy as np
from ..common.house_curves import _normalize_hc_mode_key, get_house_curve_by_name

MANUAL_TARGET_TILT_PIVOT_HZ = 1000.0


def apply_manual_target_curve_tilt(
    freq_axis,
    target_curve,
    tilt_db_per_oct: float,
    *,
    pivot_hz: float = MANUAL_TARGET_TILT_PIVOT_HZ,
):
    """Apply a broadband manual target tilt around a fixed pivot frequency."""
    target_arr = np.asarray(target_curve, dtype=float)
    try:
        freq_arr = np.asarray(freq_axis, dtype=float)
    except Exception:
        return target_arr

    if target_arr.shape != freq_arr.shape:
        return target_arr

    try:
        slope = float(tilt_db_per_oct)
    except Exception:
        slope = 0.0
    if not np.isfinite(slope) or abs(slope) <= 1e-9:
        return target_arr

    try:
        pivot = float(pivot_hz)
    except Exception:
        pivot = MANUAL_TARGET_TILT_PIVOT_HZ
    if not np.isfinite(pivot) or pivot <= 0.0:
        pivot = MANUAL_TARGET_TILT_PIVOT_HZ

    positive_freqs = freq_arr[np.isfinite(freq_arr) & (freq_arr > 0.0)]
    if positive_freqs.size > 0:
        floor_hz = float(np.min(positive_freqs))
    else:
        floor_hz = float(pivot)
    safe_freqs = np.where(np.isfinite(freq_arr) & (freq_arr > 0.0), freq_arr, floor_hz)
    return target_arr + (float(slope) * np.log2(float(pivot) / safe_freqs))


def load_target_curve(file_content: bytes):
    """Lataa tai lukee: load target curve."""
    try:
        content_str = file_content.decode("utf-8")
        lines = content_str.split("\n")
        freqs, mags = [], []
        for line in lines:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.replace(",", ".").split()
            if len(parts) >= 2:
                try:
                    f = float(parts[0])
                    m = float(parts[1])
                    if f > 0:
                        freqs.append(f)
                        mags.append(m)
                except ValueError:
                    continue

        if len(freqs) < 2:
            return None, None

        freqs = np.array(freqs)
        mags = np.array(mags)
        if np.mean(mags) > 30:
            mags -= np.mean(mags)

        sort_idx = np.argsort(freqs)
        return freqs[sort_idx], mags[sort_idx]
    except Exception:
        return None, None


def load_house_curve(data: dict, *, parse_measurements_from_path=None):
    """Lataa tai lukee: load house curve."""
    hc_f, hc_m = None, None
    hc_source = "Preset"
    mode_key = _normalize_hc_mode_key(data.get("hc_mode"))

    # Synthesized adaptive target: arrays are stored in data by the pipeline
    if mode_key == "Adaptive" and data.get("_synth_hc_f") is not None:
        try:
            synth_f = np.asarray(data["_synth_hc_f"], dtype=float)
            synth_m = np.asarray(data["_synth_hc_m"], dtype=float)
            if synth_f.size >= 4 and synth_m.size == synth_f.size:
                hc_f = synth_f
                hc_m = synth_m
                hc_source = "Adaptive"
        except Exception:
            pass

    want_upload = (mode_key == "Upload")


    try:
        up = data.get("hc_custom_file", None) if isinstance(data, dict) else None
        if want_upload and up and isinstance(up, dict) and up.get("content"):
            hc_f, hc_m = load_target_curve(up["content"])
            if hc_f is not None and hc_m is not None:
                hc_source = "Upload"
    except Exception:
        pass

    if hc_f is None and data.get("local_path_house"):
        if callable(parse_measurements_from_path):
            try:
                hc_f, hc_m, _ = parse_measurements_from_path(data["local_path_house"])
                if hc_f is not None:
                    s_idx = np.argsort(hc_f)
                    hc_f, hc_m = hc_f[s_idx], hc_m[s_idx]
                    hc_source = "LocalFile"
            except Exception:
                hc_f, hc_m = None, None

    if hc_f is None:
        preset_key = mode_key
        if preset_key == "Upload":
            preset_key = "Flat"
            hc_source = "Upload (no file)"

        if preset_key in ("Custom", "Upload"):
            hc_f, hc_m = get_house_curve_by_name("Flat")
            hc_source = "Upload (no file)"
        else:
            hc_f, hc_m = get_house_curve_by_name(preset_key)
        if hc_source == "Preset":
            hc_source = f"Preset ({preset_key})"


    try:
        lvl_mode = str(data.get("lvl_mode", "Auto") or "Auto").strip().lower()
    except Exception:
        lvl_mode = "auto"
    if hc_f is not None and hc_m is not None and "manual" in lvl_mode:
        hc_m = apply_manual_target_curve_tilt(
            hc_f,
            hc_m,
            data.get("manual_target_tilt_db_per_oct", 0.0),
        )

    return hc_f, hc_m, hc_source
