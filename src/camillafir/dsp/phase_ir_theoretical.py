from __future__ import annotations

import math

import numpy as np

from .phase import calculate_theoretical_phase


def compute_theoretical_phase_and_store_stats(
    *,
    cfg,
    freq_axis: np.ndarray,
    p_rad_interp: np.ndarray,
    st: dict,
    logger,
) -> dict:
    """
    Contract:
      - This stage must not touch gain-domain data.
      - Allowed signal-domain mutation is only p_rad_interp phase-limit blending.
      - st updates and logging are allowed.

    Returns dict with:
      theo_xo: np.ndarray
      theo_on_raw: np.ndarray
      theo_off_raw: np.ndarray
      xo_txt: str
      hpf_txt: str
      hpf_freq: float|None
      hpf_slope: float|None
      p_rad_interp: np.ndarray
    Side effects:
      updates st[...] keys exactly like before
      logs exactly like before
    """
    hpf_freq = None
    hpf_slope = None
    hs = cfg.hpf_settings

    if isinstance(hs, dict) and hs.get('enabled'):
        hpf_freq = float(hs.get('freq', 0.0) or 0.0)
        hpf_order = int(hs.get('order', 0) or 0)
        if hpf_freq > 0 and hpf_order > 0:
            hpf_slope = float(hpf_order * 6)

    try:
        xo_list = getattr(cfg, "crossovers", None) or []
        if xo_list:
            xo_txt = ", ".join([
                f"{float(x.get('freq',0.0)):.1f}Hz/{int(x.get('slope', int(x.get('order',1))*6))}dB/oct"
                for x in xo_list if x.get("freq", None) is not None
            ])
        else:
            xo_txt = "off"
        if hpf_freq and hpf_slope and float(hpf_freq) > 0 and float(hpf_slope) > 0:
            hpf_txt = f"{float(hpf_freq):.1f}Hz/{int(round(float(hpf_slope)))}dB/oct"
        else:
            hpf_txt = "off"
        logger.info(f"Phase model: XO={xo_txt} | HPF={hpf_txt}")
    except Exception:
        xo_txt = "n/a"
        hpf_txt = "n/a"

    theo_on_raw = calculate_theoretical_phase(
        freq_axis,
        cfg.crossovers,
        hpf_freq=hpf_freq,
        hpf_slope=hpf_slope,
        max_phase_deg=None
    )
    theo_off_raw = calculate_theoretical_phase(
        freq_axis,
        [],
        hpf_freq=None,
        hpf_slope=None,
        max_phase_deg=None
    )

    theo_xo = calculate_theoretical_phase(
        freq_axis,
        cfg.crossovers,
        hpf_freq=hpf_freq,
        hpf_slope=hpf_slope,
        max_phase_deg=None
    )
    try:
        xo_list = getattr(cfg, "crossovers", None) or []
        hpf_on = (hpf_freq and hpf_slope and float(hpf_freq) > 0 and float(hpf_slope) > 0)

        if xo_list or hpf_on:
            f = np.asarray(freq_axis, float)

            theo_on_raw = calculate_theoretical_phase(
                f,
                xo_list,
                hpf_freq=hpf_freq,
                hpf_slope=hpf_slope,
                max_phase_deg=None
            )
            theo_off_raw = calculate_theoretical_phase(
                f,
                [],
                hpf_freq=None,
                hpf_slope=None,
                max_phase_deg=None
            )

            dphi_raw = np.unwrap(np.asarray(theo_on_raw, float) - np.asarray(theo_off_raw, float))
            dphi_raw_deg = np.rad2deg(dphi_raw)

            def _wrap_deg(x_deg: np.ndarray) -> np.ndarray:
                return (x_deg + 180.0) % 360.0 - 180.0

            dphi_wrapped_deg = _wrap_deg(dphi_raw_deg)

            xo_mask = np.zeros_like(f, dtype=bool)
            xo_band_masks = []
            for xo in xo_list:
                try:
                    fc = float(xo.get("freq", 0.0) or 0.0)
                except Exception:
                    continue
                if fc > 0:
                    m = (f >= fc / 4.0) & (f <= fc * 4.0)
                    xo_mask |= m
                    xo_band_masks.append((fc, m))

            hpf_mask = np.zeros_like(f, dtype=bool)
            if hpf_on:
                fc = float(hpf_freq)
                hpf_mask |= (f >= max(fc / 8.0, 10.0)) & (f <= fc * 2.0)

            wide_mask = np.isfinite(f) & (f >= 20.0) & (f <= float(np.nanmax(f)))
            if np.count_nonzero(xo_mask) < 8:
                xo_mask = wide_mask
            if np.count_nonzero(hpf_mask) < 8:
                hpf_mask = wide_mask if hpf_on else np.zeros_like(f, dtype=bool)

            def _max_abs_in_mask(arr, mask):
                a = np.asarray(arr, float)
                m = np.asarray(mask, bool)
                if np.count_nonzero(m) < 8:
                    return None, None
                aa = np.where(m, np.abs(a), -1.0)
                idx = int(np.argmax(aa))
                if aa[idx] < 0:
                    return None, None
                return float(np.abs(a[idx])), float(f[idx])

            xo_phi, xo_phi_hz = _max_abs_in_mask(dphi_wrapped_deg, xo_mask)
            if xo_phi is not None:
                st["xo_diff_raw_max_phase_deg"] = xo_phi
                st["xo_diff_raw_max_phase_hz"] = xo_phi_hz

                best_fc = None
                if xo_band_masks:
                    candidates = []
                    for fc, m in xo_band_masks:
                        try:
                            idx_win = int(np.argmin(np.abs(f - xo_phi_hz)))
                        except Exception:
                            continue
                        if bool(m[idx_win]):
                            candidates.append(fc)
                    if candidates:
                        best_fc = float(min(candidates, key=lambda c: abs(float(c) - float(xo_phi_hz))))
                    else:
                        best_fc = float(min([fc for fc, _ in xo_band_masks], key=lambda c: abs(float(c) - float(xo_phi_hz))))
                if best_fc is not None:
                    st["xo_diff_raw_max_phase_xo_fc_hz"] = float(best_fc)

            for i, xo in enumerate(xo_list, start=1):
                try:
                    fc = float(xo.get("freq", 0.0) or 0.0)
                except Exception:
                    continue
                if fc <= 0:
                    continue
                idx_fc = int(np.argmin(np.abs(f - fc)))
                st[f"xo{i}_dphi_wrapped_deg@fc"] = float(dphi_wrapped_deg[idx_fc])

            hpf_phi, hpf_phi_hz = _max_abs_in_mask(dphi_raw_deg, hpf_mask)
            if hpf_on and hpf_phi is not None:
                st["hpf_diff_raw_max_phase_deg"] = hpf_phi
                st["hpf_diff_raw_max_phase_hz"] = hpf_phi_hz

            w = 2.0 * math.pi * f
            dw = np.gradient(w) + 1e-30
            ph_on = np.unwrap(np.asarray(theo_on_raw, float))
            ph_off = np.unwrap(np.asarray(theo_off_raw, float))
            gd_on = (-np.gradient(ph_on) / dw) * 1000.0
            gd_off = (-np.gradient(ph_off) / dw) * 1000.0
            dgd = gd_on - gd_off

            for i, xo in enumerate(xo_list, start=1):
                try:
                    fc = float(xo.get("freq", 0.0) or 0.0)
                except Exception:
                    continue
                if fc <= 0:
                    continue
                idx_fc = int(np.argmin(np.abs(f - fc)))
                try:
                    st[f"xo{i}_dgd_ms@fc"] = float(dgd[idx_fc])
                except Exception:
                    pass

            xo_gd, xo_gd_hz = _max_abs_in_mask(dgd, xo_mask)
            if xo_gd is not None:
                st["xo_diff_raw_max_gd_ms"] = xo_gd
                st["xo_diff_raw_max_gd_hz"] = xo_gd_hz
                try:
                    best_fc_gd = None
                    if xo_band_masks:
                        idx_win = int(np.argmin(np.abs(f - xo_gd_hz)))
                        candidates = [fc for fc, m in xo_band_masks if bool(m[idx_win])]
                        if candidates:
                            best_fc_gd = float(min(candidates, key=lambda c: abs(float(c) - float(xo_gd_hz))))
                        else:
                            best_fc_gd = float(min([fc for fc, _ in xo_band_masks], key=lambda c: abs(float(c) - float(xo_gd_hz))))
                    if best_fc_gd is not None:
                        st["xo_diff_raw_max_gd_xo_fc_hz"] = float(best_fc_gd)
                except Exception:
                    pass

            hpf_gd, hpf_gd_hz = _max_abs_in_mask(dgd, hpf_mask)
            if hpf_on and hpf_gd is not None:
                st["hpf_diff_raw_max_gd_ms"] = hpf_gd
                st["hpf_diff_raw_max_gd_hz"] = hpf_gd_hz

    except Exception as e:
        logger.debug("XO raw diff indicator failed: %s", e)

    try:
        st["xo_summary"] = xo_txt
        st["hpf_summary"] = hpf_txt
        for fchk in (20.0, 80.0, 200.0, 1000.0, 5000.0):
            idx = int(np.argmin(np.abs(freq_axis - fchk)))
            st[f"theo_xo_deg@{int(fchk)}Hz"] = float(np.rad2deg(theo_xo[idx]))
    except Exception:
        pass

    try:
        phase_lim = float(getattr(cfg, "phase_limit", 0.0) or 0.0)
        if phase_lim > 0.0 and freq_axis.size == theo_xo.size:
            f0 = float(phase_lim)
            f1 = float(phase_lim) * 1.41

            f_min = float(freq_axis[0]) if freq_axis.size else 0.0
            f_max = float(freq_axis[-1]) if freq_axis.size else 0.0
            f0 = max(f0, max(f_min, 1e-6))
            f1 = min(f1, f_max)

            if f1 <= f0:
                mask = freq_axis <= f0
                p_rad_interp = np.where(mask, p_rad_interp, theo_xo)
            else:
                w = (freq_axis - f0) / (f1 - f0)
                w = np.clip(w, 0.0, 1.0)

                p_rad_interp = (1.0 - w) * p_rad_interp + w * theo_xo
    except Exception:
        pass

    if hpf_freq and hpf_slope:
        logger.info(
            f"Theoretical phase includes HPF: f={hpf_freq:.1f} Hz, "
            f"slope={hpf_slope:.0f} dB/oct (order={int(hpf_slope/6)})"
        )
    else:
        logger.info("Theoretical phase: HPF not included")

    if getattr(cfg, "phase_safe_2058", False):
        logger.info("Phase mode: 2058-safe (no room phase correction)")
    else:
        logger.info("Phase mode: modern (excess-phase + confidence + FDW)")

    return {
        "theo_xo": np.asarray(theo_xo, dtype=float),
        "theo_on_raw": np.asarray(theo_on_raw, dtype=float),
        "theo_off_raw": np.asarray(theo_off_raw, dtype=float),
        "xo_txt": xo_txt,
        "hpf_txt": hpf_txt,
        "hpf_freq": hpf_freq,
        "hpf_slope": hpf_slope,
        "p_rad_interp": np.asarray(p_rad_interp, dtype=float),
    }
