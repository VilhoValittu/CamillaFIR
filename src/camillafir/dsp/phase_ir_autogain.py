from __future__ import annotations

import numpy as np


def _cfg_float(cfg, name: str, default: float) -> float:
    try:
        v = float(getattr(cfg, name, default))
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _gain_values_for_peak(gain_db: np.ndarray, mask_c: np.ndarray) -> np.ndarray:
    g = np.asarray(gain_db, dtype=float)
    try:
        m = np.asarray(mask_c, dtype=bool)
        if m.shape == g.shape and np.any(m):
            g = g[m]
    except Exception:
        pass
    g = g[np.isfinite(g)]
    return np.asarray(g, dtype=float)


def compute_auto_gain_and_headroom(
    *,
    cfg,
    gain_db: np.ndarray,
    mask_c: np.ndarray,
    logger,
) -> dict:
    """
    Contract:
      - This stage is gain-only.
      - It must not mutate IR/phase arrays.
      - It may compute gain terms and emit logs.

    Returns dict with:
      current_peak_gain: float
      gain_margin_db: float
      auto_global_gain_db: float
      auto_headroom_db: float
      final_gain_total: np.ndarray
    Side effects: logs StageStats and Auto Level exactly like before.
    """
    peak_max_db = 0.0
    peak_percentile_db = 0.0
    peak_effective_db = 0.0
    spike_detected = False
    spike_delta_db = 0.0
    try:
        _g_peak = _gain_values_for_peak(gain_db, mask_c)
        peak_max_db = float(np.max(_g_peak)) if _g_peak.size else 0.0

        q = _cfg_float(cfg, "auto_gain_peak_percentile", 99.5)
        q = float(np.clip(q, 50.0, 100.0))
        if _g_peak.size:
            if q >= 100.0:
                peak_percentile_db = float(peak_max_db)
            else:
                peak_percentile_db = float(np.percentile(_g_peak, q))
        else:
            peak_percentile_db = 0.0

        spike_delta_db = float(peak_max_db - peak_percentile_db)
        spike_threshold_db = max(0.0, _cfg_float(cfg, "auto_gain_spike_threshold_db", 1.0))
        spike_guard_db = max(0.0, _cfg_float(cfg, "auto_gain_spike_guard_db", 0.5))

        peak_effective_db = float(peak_max_db)
        spike_detected = bool(spike_delta_db >= spike_threshold_db)
        if spike_detected:
            peak_effective_db = float(min(peak_max_db, peak_percentile_db + spike_guard_db))
        if not np.isfinite(peak_effective_db):
            peak_effective_db = float(peak_max_db)
    except Exception:
        peak_max_db = 0.0
        peak_percentile_db = 0.0
        peak_effective_db = 0.0
        spike_detected = False
        spike_delta_db = 0.0
    current_peak_gain = float(peak_max_db)

    try:
        gain_margin_db = float(
            getattr(cfg, "auto_gain_margin_db", getattr(cfg, "global_gain_db", 0.0)) or 0.0
        )
    except Exception:
        gain_margin_db = 0.0
    if (not np.isfinite(gain_margin_db)) or (gain_margin_db < 0.0):
        gain_margin_db = 0.0

    auto_headroom_db = 0.0
    auto_global_gain_db = 0.0
    try:
        if bool(getattr(cfg, "debug_stage_stats", True)):
            _v = np.asarray(gain_db, dtype=float)
            _m = np.asarray(mask_c, dtype=bool)
            _vv = _v[_m] if (_m.shape == _v.shape and np.any(_m)) else _v
            _vv = _vv[np.isfinite(_vv)]
            if _vv.size >= 4:
                logger.info(
                    "StageStats: gain_db_pre_headroom: "
                    f"max={float(np.max(_vv)):.3f} dB, "
                    f"min={float(np.min(_vv)):.3f} dB, "
                    f"rms={float(np.sqrt(np.mean(_vv * _vv))):.3f} dB"
                )
    except Exception:
        pass

    try:
        _override = getattr(cfg, "auto_gain_db_override", None)
        if _override is None:
            raise ValueError("no override")
        auto_global_gain_db = float(_override)
        if not np.isfinite(auto_global_gain_db):
            raise ValueError("non-finite override")
        logger.info(
            f"Auto Level: using shared override {auto_global_gain_db:.2f} dB "
            f"(peak_max={peak_max_db:.2f} dB, peak_eff={peak_effective_db:.2f} dB, margin={gain_margin_db:.2f} dB)"
        )
    except Exception:
        auto_global_gain_db = -max(0.0, float(peak_effective_db)) - float(gain_margin_db)
        logger.info(
            f"Auto Level: peak_max={peak_max_db:.2f} dB, "
            f"peak_p={peak_percentile_db:.2f} dB, "
            f"peak_eff={peak_effective_db:.2f} dB "
            f"(spike={'yes' if spike_detected else 'no'}, delta={spike_delta_db:.2f} dB) "
            f"+ margin={gain_margin_db:.2f} dB "
            f"-> auto_global_gain={auto_global_gain_db:.2f} dB"
        )

    if bool(getattr(cfg, "do_normalize", False)):
        peak_after_auto = float(peak_effective_db + auto_global_gain_db)
        if peak_after_auto > -0.1:
            auto_headroom_db = -peak_after_auto - 0.1
            logger.info(
                f"Clip Prevention (Normalize ON): post-auto peak={peak_after_auto:.2f} dB "
                f"-> extra headroom={auto_headroom_db:.2f} dB"
            )
        else:
            logger.info(
                f"Clip Prevention (Normalize ON): no extra headroom needed "
                f"(post-auto peak={peak_after_auto:.2f} dB)"
            )

    final_gain_total = gain_db + auto_global_gain_db + auto_headroom_db

    return {
        "current_peak_gain": float(current_peak_gain),
        "gain_margin_db": float(gain_margin_db),
        "auto_global_gain_db": float(auto_global_gain_db),
        "auto_headroom_db": float(auto_headroom_db),
        "final_gain_total": np.asarray(final_gain_total, dtype=float),
    }
