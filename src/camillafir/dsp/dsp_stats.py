from __future__ import annotations

import numpy as np


def safe_stats_update(stats: dict, extra) -> None:
    if isinstance(extra, dict) and extra:
        stats.update(extra)


def arr_if_valid_for_stats(value, *, expected_size: int | None = None):
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size < 8:
        return None
    if expected_size is not None and expected_size >= 8 and arr.size != expected_size:
        return None
    return np.asarray(arr, dtype=float)


def apply_measured_mag_stats(stats: dict, *, target_mags, freq_axis, m_anal, calc_offset_db: float) -> None:
    stats["target_mags"] = np.asarray(target_mags, dtype=float).tolist()
    f_ref = np.asarray(stats.get("freq_axis", freq_axis), dtype=float).reshape(-1)
    n_ref = int(f_ref.size)

    m_corr_st = arr_if_valid_for_stats(stats.get("measured_mags", None), expected_size=n_ref)
    m_corr = np.asarray(m_anal, dtype=float) - float(calc_offset_db) if m_corr_st is None else np.asarray(m_corr_st, dtype=float)

    m_raw_st = arr_if_valid_for_stats(stats.get("measured_mags_raw", None), expected_size=n_ref)
    m_raw = np.asarray(m_corr, dtype=float) + float(calc_offset_db) if m_raw_st is None else np.asarray(m_raw_st, dtype=float)

    stats["measured_mags"] = np.asarray(m_corr, dtype=float).tolist()
    stats["measured_mags_raw"] = np.asarray(m_raw, dtype=float).tolist()


def apply_afdw_stats(
    stats: dict,
    *,
    afdw_on: bool,
    afdw_bw_oct,
    afdw_bw_min_oct,
    afdw_bw_mean_oct,
    afdw_bw_max_oct,
    afdw_bw_min_hz,
    afdw_bw_max_hz,
) -> None:
    if not bool(afdw_on) or afdw_bw_oct is None:
        return
    stats["afdw_bw_oct"] = np.asarray(afdw_bw_oct, dtype=float).tolist()
    stats["afdw_bw_min_oct"] = float(afdw_bw_min_oct) if afdw_bw_min_oct is not None else None
    stats["afdw_bw_mean_oct"] = float(afdw_bw_mean_oct) if afdw_bw_mean_oct is not None else None
    stats["afdw_bw_max_oct"] = float(afdw_bw_max_oct) if afdw_bw_max_oct is not None else None
    stats["afdw_bw_min_hz"] = float(afdw_bw_min_hz) if afdw_bw_min_hz is not None else None
    stats["afdw_bw_max_hz"] = float(afdw_bw_max_hz) if afdw_bw_max_hz is not None else None


def safe_stage_probes(stage_probes) -> dict:
    if not isinstance(stage_probes, dict):
        return {}
    out: dict = {}
    for key, value in stage_probes.items():
        if isinstance(value, dict):
            out[key] = dict(value)
    return out


def apply_lf_guard_stats(stats: dict, *, cfg, freq_axis, gain_db) -> None:
    low_hz_cfg = float(stats.get("low_bass_cut_hz", getattr(cfg, "low_bass_cut_hz", 0.0)) or 0.0)
    exc_on_cfg = bool(stats.get("exc_prot", getattr(cfg, "exc_prot", False)))
    exc_f_cfg = float(stats.get("exc_freq", getattr(cfg, "exc_freq", 0.0)) or 0.0)
    lf_guard_hz = 0.0
    if np.isfinite(low_hz_cfg) and low_hz_cfg > 0.0:
        lf_guard_hz = max(lf_guard_hz, float(low_hz_cfg))
    if exc_on_cfg and np.isfinite(exc_f_cfg) and exc_f_cfg > 0.0:
        lf_guard_hz = max(lf_guard_hz, float(exc_f_cfg * 1.41))
    lf_mask = (freq_axis > 0.0) & (freq_axis <= lf_guard_hz) if lf_guard_hz > 0.0 else np.zeros_like(freq_axis, dtype=bool)
    lf_boost_max_db = float(np.max(np.asarray(gain_db, dtype=float)[lf_mask])) if np.any(lf_mask) else 0.0
    stats["lf_guard_hz"] = float(lf_guard_hz)
    stats["lf_guard_bins"] = int(np.count_nonzero(lf_mask))
    stats["lf_boost_max_db"] = float(lf_boost_max_db)


def apply_clamp_stats(
    stats: dict,
    *,
    softclip_boost_bins: int,
    softclip_cut_bins: int,
    over_boost: float,
    over_cut: float,
    hardclamp_boost_bins: int,
    hardclamp_cut_bins: int,
    hard_over_boost: float,
    hard_over_cut: float,
) -> None:
    stats["softclip_boost_bins"] = int(softclip_boost_bins)
    stats["softclip_cut_bins"] = int(softclip_cut_bins)
    stats["softclip_worst_over_boost_db"] = float(over_boost)
    stats["softclip_worst_over_cut_db"] = float(over_cut)
    stats["hardclamp_boost_bins"] = int(hardclamp_boost_bins)
    stats["hardclamp_cut_bins"] = int(hardclamp_cut_bins)
    stats["hardclamp_worst_over_boost_db"] = float(hard_over_boost)
    stats["hardclamp_worst_over_cut_db"] = float(hard_over_cut)
    stats["clamp_summary"] = (
        f"soft_clip: boost={stats['softclip_boost_bins']} cut={stats['softclip_cut_bins']} "
        f"(worst_over_boost={stats['softclip_worst_over_boost_db']:.2f} dB, worst_over_cut={stats['softclip_worst_over_cut_db']:.2f} dB); "
        f"hard_clamp: boost={stats['hardclamp_boost_bins']} cut={stats['hardclamp_cut_bins']} "
        f"(worst_over_boost={stats['hardclamp_worst_over_boost_db']:.2f} dB, worst_over_cut={stats['hardclamp_worst_over_cut_db']:.2f} dB)"
    )


def apply_boost_blocked_reason(stats: dict, *, cfg) -> None:
    max_boost_db_cfg = float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
    low_hz_cfg = float(getattr(cfg, "low_bass_cut_hz", 40.0) or 40.0)
    exc_on = bool(getattr(cfg, "exc_prot", False))
    exc_f_cfg = float(getattr(cfg, "exc_freq", 0.0) or 0.0)
    do_norm = bool(getattr(cfg, "do_normalize", False))
    g_global = float(stats.get("auto_global_gain_db", getattr(cfg, "global_gain_db", 0.0)) or 0.0)

    boost_bins_post = int(stats.get("boost_bins", 0) or 0)
    boost_bins_cand = int(stats.get("boost_candidate_bins", 0) or 0)
    boost_bins_cand_low = int(stats.get("boost_candidate_bins_lowbass", 0) or 0)
    boost_bins_cand_exc = int(stats.get("boost_candidate_bins_excprot", 0) or 0)
    boost_peak_post = float(stats.get("boost_peak_db", 0.0) or 0.0)
    net_boost_peak = boost_peak_post + g_global + float(stats.get("auto_headroom_db", 0.0) or 0.0)
    stats["net_boost_peak_db"] = float(net_boost_peak)

    reasons = []
    if max_boost_db_cfg <= 0.0:
        reasons.append("max_boost_db <= 0 (boost disabled)")
    if boost_bins_cand == 0 and boost_bins_post == 0:
        reasons.append("no boost candidates (algorithm produced only cuts in correction band)")
    if boost_bins_cand > 0 and boost_bins_post == 0:
        if boost_bins_cand_low == boost_bins_cand and low_hz_cfg > 0:
            reasons.append(f"all boost candidates were <= low_bass_cut_hz ({low_hz_cfg:.1f} Hz) where cuts-only policy applies")
        if exc_on and exc_f_cfg > 0 and boost_bins_cand_exc == boost_bins_cand:
            reasons.append(f"all boost candidates were < exc_freq ({exc_f_cfg:.1f} Hz) while exc_prot is ON")
        if not reasons:
            reasons.append("boost candidates existed but were removed by limits/safety clamp (check max_boost_db, slope limits, exc/low-bass policies)")
    if boost_bins_cand > 0 and boost_bins_post > 0 and boost_bins_post < boost_bins_cand:
        if boost_bins_cand_low > 0:
            reasons.append(f"some boost candidates were in low-bass restricted region (<= {low_hz_cfg:.1f} Hz)")
        if exc_on and boost_bins_cand_exc > 0 and exc_f_cfg > 0:
            reasons.append(f"some boost candidates were in exc_prot region (< {exc_f_cfg:.1f} Hz)")
        reasons.append("some boost candidates were reduced by limits/safety clamp")
    if boost_bins_post > 0 and net_boost_peak <= 0.0:
        reasons.append(f"net boost peak <= 0.00 dB after global gain/headroom (net_peak={net_boost_peak:.2f} dB, normalize={'ON' if do_norm else 'OFF'})")
    stats["boost_blocked_reason"] = "; ".join(dict.fromkeys(reasons)) if reasons else "no blocking detected"
