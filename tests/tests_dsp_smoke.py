#!/usr/bin/env python3
"""
tests_dsp_smoke.py

Lightweight DSP smoke tests for CamillaFIR.

Goals:
- Catch import errors / circular deps
- Run the end-to-end DSP pipeline on synthetic data
- Assert invariants (finite, sane limits, alignment-ish metrics)
- Provide a short, actionable console summary

Usage:
  python tests_dsp_smoke.py
Optional:
  python tests_dsp_smoke.py --seed 123 --fs 48000 --n 1024 --verbose

Notes:
- This assumes you run it from the project root (so `camillafir` is importable).
- It tries to instantiate `FilterConfig()` with defaults. If your FilterConfig requires args,
  adapt `make_cfg()` accordingly.
- Manual helper only: this file is meant for direct script use, not normal pytest collection.
  Keep smoke checks here as plain helper routines unless they are split into dedicated `test_*.py` tests.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _ensure_src_on_path() -> None:
    """Allow running this script directly without external PYTHONPATH setup."""
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    src_path = str(src_dir)
    if src_dir.is_dir() and src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()


def _set_if_has(obj: Any, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _get(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _finite(name: str, arr: np.ndarray) -> None:
    if not np.all(np.isfinite(arr)):
        bad = np.count_nonzero(~np.isfinite(arr))
        raise AssertionError(f"{name}: contains {bad} non-finite values")


def _assert_within(name: str, arr: np.ndarray, lo: float, hi: float, eps: float = 1e-6) -> None:
    a = np.asarray(arr, float)
    if a.size == 0:
        return
    mn = float(np.nanmin(a))
    mx = float(np.nanmax(a))
    if mn < lo - eps or mx > hi + eps:
        raise AssertionError(f"{name}: out of bounds [{lo}, {hi}] (min={mn:.4f}, max={mx:.4f})")


def _oct_slope_db_per_oct(freq: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
    """
    Approx slope in dB/oct between adjacent points on log2 freq axis.
    """
    f = np.asarray(freq, float)
    g = np.asarray(gain_db, float)
    # Avoid invalid/zero frequencies
    m = (f > 0) & np.isfinite(f) & np.isfinite(g)
    f = f[m]
    g = g[m]
    if f.size < 3:
        return np.array([], dtype=float)
    x = np.log2(f)
    dx = np.diff(x)
    dg = np.diff(g)
    dx = np.where(dx == 0, 1e-12, dx)
    return dg / dx


def _summary_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": float("nan"), "max": float("nan"), "rms": float("nan")}
    return {
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
    }


def _resolve_lf_guard_hz(stats: Dict[str, Any]) -> float:
    try:
        low_hz = float(stats.get("low_bass_cut_hz", 0.0) or 0.0)
    except Exception:
        low_hz = 0.0
    try:
        exc_on = bool(stats.get("exc_prot", False))
    except Exception:
        exc_on = False
    try:
        exc_f = float(stats.get("exc_freq", 0.0) or 0.0)
    except Exception:
        exc_f = 0.0
    hz = 0.0
    if np.isfinite(low_hz) and low_hz > 0.0:
        hz = max(hz, float(low_hz))
    if exc_on and np.isfinite(exc_f) and exc_f > 0.0:
        hz = max(hz, float(exc_f * 1.41))
    return float(hz)


def _lf_boost_max(freq_axis: np.ndarray, filt_db: np.ndarray, guard_hz: float) -> float:
    f = np.asarray(freq_axis, float)
    g = np.asarray(filt_db, float)
    if f.size == 0 or g.size == 0 or f.shape != g.shape:
        return 0.0
    if not np.isfinite(guard_hz) or guard_hz <= 0.0:
        return 0.0
    m = (f > 0.0) & (f <= float(guard_hz))
    if not np.any(m):
        return 0.0
    return float(np.nanmax(g[m]))


def make_synth_measurement(fs: int, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      freqs (Hz), meas_mags (dB), raw_phases (rad)
    Shapes: (n,)
    """
    rng = np.random.default_rng(seed)

    # Build a plausible frequency axis: 10 Hz .. Nyquist-ish, log spaced
    fmin = 10.0
    fmax = max(200.0, fs * 0.49)
    freqs = np.geomspace(fmin, fmax, num=n).astype(float)

    # Magnitude: gentle tilt + a couple of smooth bumps (in dB)
    tilt = -1.5 * np.log10(np.maximum(freqs, 1.0) / 1000.0)
    bump1 = 3.0 * np.exp(-0.5 * ((np.log(freqs) - np.log(80.0)) / 0.30) ** 2)
    bump2 = -4.0 * np.exp(-0.5 * ((np.log(freqs) - np.log(250.0)) / 0.35) ** 2)
    noise = 0.25 * rng.normal(size=freqs.size)

    meas_mags = (tilt + bump1 + bump2 + noise).astype(float)

    # Phase: smooth-ish random walk (radians)
    dphi = 0.03 * rng.normal(size=freqs.size)
    raw_phases = np.cumsum(dphi).astype(float)

    return freqs, meas_mags, raw_phases


def make_cfg(fs: int, filter_type: str, residual_on: bool, normalize_on: bool) -> Any:
    """
    Instantiate FilterConfig with defaults and set a few knobs safely.

    If your FilterConfig requires arguments, edit this function accordingly.
    """
    from camillafir.config.models import FilterConfig  # noqa: WPS433

    cfg = FilterConfig()

    # Core sampling
    _set_if_has(cfg, "fs", fs)

    # Filter type / mode string (your code uses cfg.filter_type_str)
    _set_if_has(cfg, "filter_type_str", filter_type)

    # Sensible defaults for smoke tests (only set if fields exist)
    _set_if_has(cfg, "enable_mag_correction", True)
    _set_if_has(cfg, "enable_residual_pass", residual_on)
    _set_if_has(cfg, "do_normalize", normalize_on)

    # Typical safe bounds
    _set_if_has(cfg, "max_boost_db", _get(cfg, "max_boost_db", 6.0) or 6.0)
    _set_if_has(cfg, "max_cut_db", _get(cfg, "max_cut_db", 12.0) or 12.0)

    # Keep correction band reasonable if present
    if hasattr(cfg, "mag_c_min"):
        _set_if_has(cfg, "mag_c_min", float(_get(cfg, "mag_c_min", 20.0) or 20.0))
    if hasattr(cfg, "mag_c_max"):
        _set_if_has(cfg, "mag_c_max", float(_get(cfg, "mag_c_max", 8000.0) or 8000.0))
    if hasattr(cfg, "trans_width"):
        _set_if_has(cfg, "trans_width", float(_get(cfg, "trans_width", 200.0) or 200.0))

    # Mixed defaults if fields exist
    _set_if_has(cfg, "mixed_split_freq", float(_get(cfg, "mixed_split_freq", 300.0) or 300.0))

    # Phase limit off by default unless already set
    _set_if_has(cfg, "phase_limit", float(_get(cfg, "phase_limit", 0.0) or 0.0))

    # HPF off by default unless already enabled
    if hasattr(cfg, "hpf_settings") and getattr(cfg, "hpf_settings") is None:
        _set_if_has(cfg, "hpf_settings", {"enabled": False})

    # Keep pre-energy guard visible in A/B diagnostics.
    try:
        setattr(cfg, "enable_ir_pre_energy_guard", True)
    except Exception:
        pass

    return cfg


def run_one_case(
    *,
    fs: int,
    n: int,
    seed: int,
    filter_type: str,
    residual_on: bool,
    normalize_on: bool,
    verbose: bool,
) -> Dict[str, Any]:
    """
    Runs generate_filter() and returns a summary dict.
    """
    from camillafir.dsp.camillafir_dsp import generate_filter  # noqa: WPS433

    freqs, meas_mags, raw_phases = make_synth_measurement(fs=fs, n=n, seed=seed)
    cfg = make_cfg(fs=fs, filter_type=filter_type, residual_on=residual_on, normalize_on=normalize_on)

    impulse, stats = generate_filter(freqs, meas_mags, raw_phases, cfg)

    impulse = np.asarray(impulse, float)
    _finite("impulse", impulse)

    # Basic IR checks
    peak_idx = int(np.argmax(np.abs(impulse))) if impulse.size else -1
    peak_val = float(np.max(np.abs(impulse))) if impulse.size else 0.0

    # Stats arrays (stored as lists)
    freq_axis = np.asarray(stats.get("freq_axis", []), float)
    filt_db = np.asarray(stats.get("filter_mags", []), float)
    conf = np.asarray(stats.get("confidence_mask", []), float)

    if freq_axis.size:
        _finite("freq_axis", freq_axis)
        if not np.all(np.diff(freq_axis) > 0):
            raise AssertionError("freq_axis: not strictly increasing")

    if filt_db.size:
        _finite("filter_mags", filt_db)

        max_boost = float(_get(cfg, "max_boost_db", 0.0) or 0.0)
        max_cut = abs(float(_get(cfg, "max_cut_db", 0.0) or 0.0))
        # Some code paths may soft-clip then hard clamp; allow tiny epsilon.
        _assert_within("filter_mags", filt_db, lo=-max_cut, hi=max_boost, eps=1e-3)

        # Slope sanity (only if max_slope exists)
        max_slope = float(_get(cfg, "max_slope_db_per_oct", 0.0) or 0.0)
        if max_slope > 0 and freq_axis.size == filt_db.size:
            slope = _oct_slope_db_per_oct(freq_axis, filt_db)
            _finite("slope_db_per_oct", slope)
            # allow some epsilon because slope limiting might not be exact at boundaries
            if slope.size:
                if float(np.nanmax(np.abs(slope))) > max_slope + 3.0:
                    raise AssertionError(
                        f"slope: exceeded max_slope_db_per_oct "
                        f"(max|slope|={float(np.nanmax(np.abs(slope))):.2f}, limit={max_slope:.2f})"
                    )

    if conf.size:
        _finite("confidence_mask", conf)
        if np.nanmin(conf) < -1e-3 or np.nanmax(conf) > 1.0 + 1e-3:
            raise AssertionError(
                f"confidence_mask: outside [0..1] (min={float(np.nanmin(conf))}, max={float(np.nanmax(conf))})"
            )

    guard_hz = _resolve_lf_guard_hz(stats)
    lf_boost_max_db = float(stats.get("lf_boost_max_db", _lf_boost_max(freq_axis, filt_db, guard_hz)) or 0.0)
    pre_guard_triggered = bool(stats.get("ir_pre_energy_guard_triggered", False))
    pre_guard_scale = float(stats.get("ir_pre_energy_guard_scale", 1.0) or 1.0)
    if (not pre_guard_triggered) and np.isfinite(pre_guard_scale):
        pre_guard_triggered = bool(pre_guard_scale < 0.999999)

    # Summary
    out = {
        "filter_type": filter_type,
        "residual_on": residual_on,
        "normalize_on": normalize_on,
        "fs": fs,
        "n": n,
        "impulse_len": int(impulse.size),
        "impulse_peak_idx": peak_idx,
        "impulse_peak_abs": peak_val,
        "filter_db_stats": _summary_stats(filt_db) if filt_db.size else None,
        "conf_stats": _summary_stats(conf) if conf.size else None,
        "lf_guard_hz": float(guard_hz),
        "lf_boost_max_db": float(lf_boost_max_db),
        "pre_energy_guard_triggered": bool(pre_guard_triggered),
        "pre_energy_guard_scale": float(pre_guard_scale),
        "stats_keys": sorted(list(stats.keys())),
    }

    if verbose:
        print("  stats keys:", ", ".join(out["stats_keys"]))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--fs", type=int, default=48000)
    ap.add_argument("--n", type=int, default=1024, help="number of measurement points")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = [
        # Base linear-ish
        dict(filter_type="Linear", residual_on=False, normalize_on=False),
        # Residual pass toggled
        dict(filter_type="Linear", residual_on=True, normalize_on=False),
        # Mixed (if supported by your code via 'Mixed' substring)
        dict(filter_type="Mixed", residual_on=False, normalize_on=False),
        dict(filter_type="Mixed", residual_on=True, normalize_on=False),
        # Normalize ON
        dict(filter_type="Linear", residual_on=False, normalize_on=True),
    ]

    results = []
    print("DSP smoke tests starting...\n")

    # Import smoke: ensures the post-split modules are importable
    try:
        from camillafir.dsp.dsp_phase_ir import run_phase_ir_stage  # noqa: F401
        from camillafir.dsp.phase_ir_theoretical import compute_theoretical_phase_and_store_stats  # noqa: F401
        from camillafir.dsp.phase_ir_residual import apply_residual_pass_if_enabled  # noqa: F401
        from camillafir.dsp.phase_ir_autogain import compute_auto_gain_and_headroom  # noqa: F401
        from camillafir.dsp.phase_ir_build import build_phase_and_ir  # noqa: F401

        print("Imports: OK")
    except Exception as e:
        print("Imports: FAILED")
        print("Error:", e)
        return 2

    for i, c in enumerate(cases, start=1):
        print(
            f"\nCase {i}/{len(cases)}: "
            f"type={c['filter_type']}, residual={c['residual_on']}, normalize={c['normalize_on']}"
        )
        try:
            r = run_one_case(
                fs=args.fs,
                n=args.n,
                seed=args.seed,
                filter_type=c["filter_type"],
                residual_on=c["residual_on"],
                normalize_on=c["normalize_on"],
                verbose=args.verbose,
            )
            results.append(r)
            fstats = r["filter_db_stats"]
            if fstats:
                print(
                    f"  filter_mags: min={fstats['min']:.2f} dB, "
                    f"max={fstats['max']:.2f} dB, rms={fstats['rms']:.2f} dB"
                )
            print(
                f"  impulse: len={r['impulse_len']}, peak_idx={r['impulse_peak_idx']}, "
                f"peak_abs={r['impulse_peak_abs']:.6f}"
            )
        except Exception as e:
            print("  FAILED:", repr(e))
            return 1

    # Quick comparative check: residual should change the correction somewhat (not mandatory, but useful signal)
    try:
        lin0 = next(x for x in results if x["filter_type"] == "Linear" and not x["residual_on"] and not x["normalize_on"])
        lin1 = next(x for x in results if x["filter_type"] == "Linear" and x["residual_on"] and not x["normalize_on"])
        if lin0["filter_db_stats"] and lin1["filter_db_stats"]:
            dmax = abs(lin1["filter_db_stats"]["max"] - lin0["filter_db_stats"]["max"])
            # If residual pass is enabled but has zero effect, that might be fine for some configs,
            # but it's worth flagging.
            if dmax < 1e-3:
                print("\nNote: residual pass appears to have ~no effect on max gain in this synthetic case.")
    except Exception:
        pass

    try:
        lin0 = next(x for x in results if x["filter_type"] == "Linear" and not x["residual_on"] and not x["normalize_on"])
        lin1 = next(x for x in results if x["filter_type"] == "Linear" and x["residual_on"] and not x["normalize_on"])
        hz = max(float(lin0.get("lf_guard_hz", 0.0) or 0.0), float(lin1.get("lf_guard_hz", 0.0) or 0.0))
        print(
            "\nA/B LF max boost: "
            f"residual OFF={float(lin0.get('lf_boost_max_db', 0.0) or 0.0):.2f} dB, "
            f"ON={float(lin1.get('lf_boost_max_db', 0.0) or 0.0):.2f} dB "
            f"(LF guard <= {hz:.1f} Hz)"
        )
    except Exception:
        pass

    try:
        off_cases = [x for x in results if not x["residual_on"]]
        on_cases = [x for x in results if x["residual_on"]]
        off_triggers = sum(1 for x in off_cases if bool(x.get("pre_energy_guard_triggered", False)))
        on_triggers = sum(1 for x in on_cases if bool(x.get("pre_energy_guard_triggered", False)))
        print(
            "A/B pre-energy guard triggers: "
            f"residual OFF={off_triggers}/{len(off_cases)}, "
            f"ON={on_triggers}/{len(on_cases)}"
        )
    except Exception:
        pass

    print("\nDSP smoke tests: ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
