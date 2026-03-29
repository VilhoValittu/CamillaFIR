import logging
import math

from pywebio.output import put_html, use_scope
from pywebio.pin import pin

from ..io.auto_mode.shared import _auto_goal_forced_level_window
from ..resources.i8n.camillafir_i18n import t

logger = logging.getLogger("CamillaFIR")

try:
    from .camillafir_housecurve import (
        _normalize_hc_mode_key,
        get_house_curve_by_name,
        load_house_curve,
        load_target_curve,
    )
except Exception:
    _normalize_hc_mode_key = None
    get_house_curve_by_name = None
    load_house_curve = None
    load_target_curve = None


def _pin_get(name, default=None):
    try:
        return pin[name]
    except Exception:
        pass

    try:
        getter = getattr(pin, "get", None)
        if callable(getter):
            value = getter(name, None)
            return default if value is None else value
    except Exception:
        pass

    return default


def update_target_preview_ui(_=None):
    """Soveltaa tai paivittaa: update target preview ui."""

    def _norm_key(x: str) -> str:
        value = str(x or "").strip()
        if not value:
            return ""
        try:
            if callable(_normalize_hc_mode_key):
                return str(_normalize_hc_mode_key(value))
        except Exception:
            pass
        return value

    try:
        import numpy as np
        import plotly.graph_objects as go
        import plotly.io as pio

        from ..dsp.smoothing import psychoacoustic_smoothing as _psycho_smooth
        from ..io.measurements_txt import (
            parse_measurements_from_bytes as _parse_txt_bytes,
            parse_measurements_from_path as _parse_txt_path,
        )
        from ..io.measurements_wav import (
            parse_measurements_from_wav_bytes as _parse_wav_bytes,
            parse_measurements_from_wav_path as _parse_wav_path,
        )

        def _to_float(v, default):
            try:
                x = float(v)
                if np.isfinite(x):
                    return x
            except Exception:
                pass
            return float(default)

        def _normalize_curve(freqs, mags):
            try:
                ff = np.asarray(freqs, dtype=float)
                mm = np.asarray(mags, dtype=float)
                if ff.size < 8 or mm.size != ff.size:
                    return None, None
                mask = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.0)
                ff = ff[mask]
                mm = mm[mask]
                if ff.size < 8:
                    return None, None
                order = np.argsort(ff)
                ff = ff[order]
                mm = mm[order]
                uniq, idx = np.unique(ff, return_index=True)
                ff = uniq
                mm = mm[idx]
                if ff.size < 8:
                    return None, None
                return ff, mm
            except Exception:
                return None, None

        def _pick_upload(name):
            value = _pin_get(name, None)
            if isinstance(value, list) and len(value) > 0:
                value = value[0]
            if isinstance(value, dict) and (value.get("content") is not None):
                return value
            return None

        def _parse_uploaded_measurement(up):
            if not isinstance(up, dict):
                return None, None
            content = up.get("content", None)
            if content is None:
                return None, None

            name = str(up.get("filename", "") or "").strip().lower()
            ext = "." + name.rsplit(".", 1)[1] if "." in name else ""
            pre_ms = _to_float(_pin_get("ir_window_left", 85.0), 85.0)
            post_raw = _pin_get("ir_window_right", None)
            if post_raw in (None, ""):
                post_raw = _pin_get("ir_window", 500.0)
            post_ms = _to_float(post_raw, 500.0)
            try:
                smoothing_level = int(float(_pin_get("filter_smooth", _pin_get("smoothing_level", 0)) or 0))
            except Exception:
                smoothing_level = 0

            try:
                is_wav = (ext == ".wav") or (
                    isinstance(content, (bytes, bytearray)) and len(content) >= 4 and content[:4] == b"RIFF"
                )
                if is_wav:
                    ff, mm, _ = _parse_wav_bytes(
                        content,
                        pre_ms=pre_ms,
                        post_ms=post_ms,
                        smoothing_level=smoothing_level,
                        logger=None,
                    )
                else:
                    ff, mm, _ = _parse_txt_bytes(content)
            except Exception:
                return None, None
            return _normalize_curve(ff, mm)

        def _parse_local_measurement(path_raw):
            path = str(path_raw or "").strip().strip('"').strip("'")
            if not path:
                return None, None
            path_l = path.lower()
            pre_ms = _to_float(_pin_get("ir_window_left", 85.0), 85.0)
            post_raw = _pin_get("ir_window_right", None)
            if post_raw in (None, ""):
                post_raw = _pin_get("ir_window", 500.0)
            post_ms = _to_float(post_raw, 500.0)
            try:
                smoothing_level = int(float(_pin_get("filter_smooth", _pin_get("smoothing_level", 0)) or 0))
            except Exception:
                smoothing_level = 0
            try:
                if path_l.endswith(".wav"):
                    ff, mm, _ = _parse_wav_path(
                        path,
                        pre_ms=pre_ms,
                        post_ms=post_ms,
                        smoothing_level=smoothing_level,
                        logger=None,
                    )
                else:
                    ff, mm, _ = _parse_txt_path(path, logger=None)
            except Exception:
                return None, None
            return _normalize_curve(ff, mm)

        def _align_to_target_window(m_curve, t_curve, freq_axis, fmin, fmax):
            try:
                m_curve = np.asarray(m_curve, dtype=float)
                t_curve = np.asarray(t_curve, dtype=float)
                freq_axis = np.asarray(freq_axis, dtype=float)
                if m_curve.size != freq_axis.size or t_curve.size != freq_axis.size:
                    return m_curve
                mask = (freq_axis >= float(fmin)) & (freq_axis <= float(fmax)) & np.isfinite(m_curve) & np.isfinite(t_curve)
                if np.count_nonzero(mask) < 16:
                    return m_curve
                off = float(np.median(m_curve[mask] - t_curve[mask]))
                if not np.isfinite(off):
                    return m_curve
                return m_curve - off
            except Exception:
                return np.asarray(m_curve, dtype=float)

        def _smooth_for_preview(freq_axis, mags_curve):
            try:
                return np.asarray(_psycho_smooth(freq_axis, mags_curve), dtype=float)
            except Exception:
                return np.asarray(mags_curve, dtype=float)

        hc_mode_raw = str(_pin_get("hc_mode", "Harman6") or "Harman6")
        hc_mode = _norm_key(hc_mode_raw)

        mag_c_min = _to_float(_pin_get("mag_c_min", 10.0), 10.0)
        mag_c_max = _to_float(_pin_get("mag_c_max", 200.0), 200.0)

        lvl_min = _to_float(_pin_get("lvl_min", 500.0), 500.0)
        lvl_max = _to_float(_pin_get("lvl_max", 2000.0), 2000.0)
        if not (lvl_min > 0.0 and lvl_max > lvl_min):
            lvl_min, lvl_max = 500.0, 2000.0
        try:
            app_mode = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            app_mode = "BASIC"
        auto_goal = str(_pin_get("auto_goal", "balanced") or "balanced").strip().lower()
        forced_level_window = _auto_goal_forced_level_window(auto_goal) if app_mode == "AUTO" else None

        lvl_mode = str(_pin_get("lvl_mode", "Auto") or "Auto")
        if app_mode in ("BASIC", "AUTO"):
            lvl_mode = "Auto"
        if forced_level_window is not None:
            lvl_min, lvl_max = float(forced_level_window[0]), float(forced_level_window[1])

        is_manual_level = "manual" in lvl_mode.strip().lower()
        lvl_mode_label = t("lvl_mode_manual") if is_manual_level else t("lvl_mode_auto")
        lvl_manual_db = _to_float(_pin_get("lvl_manual_db", 0.0), 0.0)
        preview_level_shift_db = lvl_manual_db if is_manual_level else 0.0

        freq_axis = np.logspace(np.log10(10.0), np.log10(20000.0), 600)

        target_curve = None
        src = t("target_preview_source_builtin")

        key_l = str(hc_mode).strip().lower()
        is_upload = key_l in ("upload", "custom", "hc_mode_upload") or ("upload" in key_l)

        def _parse_target_bytes_fallback(b: bytes):
            try:
                s = b.decode("utf-8", errors="ignore")
            except Exception:
                s = str(b)

            freqs = []
            mags = []
            for line in s.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#") or line.startswith(";") or line.startswith("//"):
                    continue
                parts = line.replace(",", ".").split()
                if len(parts) < 2:
                    continue
                try:
                    freqs.append(float(parts[0]))
                    mags.append(float(parts[1]))
                except Exception:
                    continue

            if len(freqs) < 2:
                raise ValueError("No valid (freq, mag) pairs found")

            ff = np.asarray(freqs, dtype=float)
            yy = np.asarray(mags, dtype=float)
            mask = np.isfinite(ff) & np.isfinite(yy) & (ff > 0)
            ff = ff[mask]
            yy = yy[mask]
            if ff.size < 2:
                raise ValueError("Not enough valid samples after filtering")
            idx = np.argsort(ff)
            return ff[idx], yy[idx]

        if is_upload:
            up = _pin_get("hc_custom_file", None)
            if isinstance(up, list) and len(up) > 0:
                up = up[0]

            if isinstance(up, dict) and (up.get("content") is not None):
                try:
                    if callable(load_target_curve):
                        tf_f, tf_y = load_target_curve(up["content"])
                    else:
                        raise RuntimeError("load_target_curve not available")

                    if tf_f is None or tf_y is None:
                        raise ValueError("load_target_curve() returned no data")

                    tf_f = np.asarray(tf_f, dtype=float)
                    tf_y = np.asarray(tf_y, dtype=float)
                    if tf_f.size >= 2 and tf_y.size == tf_f.size:
                        target_curve = np.interp(freq_axis, tf_f, tf_y, left=tf_y[0], right=tf_y[-1])
                        src = t("target_preview_source_upload")
                    else:
                        raise ValueError("Target data malformed (size mismatch)")
                except Exception as e1:
                    try:
                        tf_f, tf_y = _parse_target_bytes_fallback(up.get("content", b""))
                        target_curve = np.interp(freq_axis, tf_f, tf_y, left=tf_y[0], right=tf_y[-1])
                        src = t("target_preview_source_upload")
                    except Exception as e2:
                        with use_scope("target_preview_scope", clear=True):
                            put_html(
                                "<div style='opacity:0.85; font-size:13px; padding:8px 0;'>"
                                "Custom target could not be parsed.<br>"
                                f"<span style='opacity:0.75'>Loader error: {str(e1)}</span><br>"
                                f"<span style='opacity:0.75'>Fallback error: {str(e2)}</span>"
                                "</div>"
                            )
                        return

            if not (isinstance(up, dict) and up.get("content") is not None):
                with use_scope("target_preview_scope", clear=True):
                    put_html(
                        "<div style='opacity:0.8; font-size:13px; padding:8px 0;'>"
                        "Custom target selected, but no file loaded yet."
                        "</div>"
                    )
                return

        if target_curve is None:
            hc = None
            try:
                if callable(get_house_curve_by_name):
                    hc = get_house_curve_by_name(hc_mode)
            except Exception:
                hc = None

            if hc is None:
                try:
                    if callable(load_house_curve):
                        hc = load_house_curve(hc_mode)
                except Exception:
                    hc = None
            if callable(hc):
                target_curve = np.asarray(hc(freq_axis), dtype=float)
            elif isinstance(hc, (tuple, list)) and len(hc) >= 2:
                hf = np.asarray(hc[0], dtype=float)
                hy = np.asarray(hc[1], dtype=float)
                if hf.size >= 2 and hy.size == hf.size:
                    target_curve = np.interp(freq_axis, hf, hy, left=hy[0], right=hy[-1])
            elif isinstance(hc, dict):
                for fk, mk in (("freqs", "mags"), ("f", "y"), ("freq", "mag"), ("hz", "db")):
                    if (fk in hc) and (mk in hc):
                        hf = np.asarray(hc[fk], dtype=float)
                        hy = np.asarray(hc[mk], dtype=float)
                        if hf.size >= 2 and hy.size == hf.size:
                            target_curve = np.interp(freq_axis, hf, hy, left=hy[0], right=hy[-1])
                            break

        if target_curve is None:
            with use_scope("target_preview_scope", clear=True):
                put_html(
                    "<div style='opacity:0.8; font-size:13px; padding:8px 0;'>"
                    "Target preview could not be generated (unknown curve format). "
                    "Try switching the target curve or re-uploading the custom file."
                    "</div>"
                )
            return

        speaker_curves = {}
        for ch, up_key, path_key in (
            ("L", "file_l", "local_path_l"),
            ("R", "file_r", "local_path_r"),
        ):
            up = _pick_upload(up_key)
            ff, mm = (None, None)
            if up is not None:
                ff, mm = _parse_uploaded_measurement(up)
            if ff is None or mm is None:
                ff, mm = _parse_local_measurement(_pin_get(path_key, ""))
            if ff is not None and mm is not None:
                speaker_curves[ch] = (ff, mm)

        speaker_interp = {}
        for ch, (ff, mm) in speaker_curves.items():
            m_raw = np.interp(freq_axis, ff, mm, left=mm[0], right=mm[-1])
            m_aligned = _align_to_target_window(m_raw, target_curve, freq_axis, lvl_min, lvl_max)
            speaker_interp[ch] = _smooth_for_preview(freq_axis, m_aligned)

        if abs(preview_level_shift_db) > 1e-9:
            for key in list(speaker_interp.keys()):
                speaker_interp[key] = np.asarray(speaker_interp[key], dtype=float) + float(preview_level_shift_db)

        speaker_avg = None
        if len(speaker_interp) > 0:
            speaker_avg = np.mean(np.vstack([speaker_interp[k] for k in sorted(speaker_interp.keys())]), axis=0)

        speaker_label = "No speaker data loaded"
        if "L" in speaker_interp and "R" in speaker_interp:
            speaker_label = f"L + R (aligned {lvl_min:.0f}-{lvl_max:.0f} Hz)"
        elif "L" in speaker_interp:
            speaker_label = f"L only (aligned {lvl_min:.0f}-{lvl_max:.0f} Hz)"
        elif "R" in speaker_interp:
            speaker_label = f"R only (aligned {lvl_min:.0f}-{lvl_max:.0f} Hz)"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=freq_axis,
                y=target_curve,
                mode="lines",
                name=f"Target ({hc_mode_raw})",
                line=dict(color="#4caf50", width=2.0),
            )
        )

        if "L" in speaker_interp:
            fig.add_trace(
                go.Scatter(
                    x=freq_axis,
                    y=speaker_interp["L"],
                    mode="lines",
                    name="Speaker L",
                    line=dict(color="rgba(102, 187, 255, 0.55)", width=1.2),
                )
            )
        if "R" in speaker_interp:
            fig.add_trace(
                go.Scatter(
                    x=freq_axis,
                    y=speaker_interp["R"],
                    mode="lines",
                    name="Speaker R",
                    line=dict(color="rgba(255, 167, 102, 0.55)", width=1.2),
                )
            )
        if speaker_avg is not None:
            fig.add_trace(
                go.Scatter(
                    x=freq_axis,
                    y=speaker_avg,
                    mode="lines",
                    name="Speaker avg",
                    line=dict(color="#ffd166", width=2.0),
                )
            )

        fig.add_vrect(
            x0=max(1.0, lvl_min),
            x1=max(1.0, lvl_max),
            fillcolor="rgba(180, 180, 180, 0.16)",
            line_width=0,
            layer="below",
        )

        fig.add_vline(x=max(1.0, mag_c_min), line_width=1, opacity=0.35)
        fig.add_vline(x=max(1.0, mag_c_max), line_width=1, opacity=0.35)

        fig.update_xaxes(
            type="log",
            title_text="Hz",
            range=[math.log10(10.0), math.log10(20000.0)],
            fixedrange=True,
        )
        fig.update_yaxes(
            title_text="dB",
            range=[-10.0, 20.0],
            fixedrange=True,
        )
        fig.update_layout(
            height=320,
            width=1800,
            margin=dict(l=40, r=20, t=30, b=35),
            showlegend=True,
            template="plotly_dark",
            uirevision="target_preview_lock",
        )

        html = pio.to_html(fig, include_plotlyjs=True, full_html=False)

        with use_scope("target_preview_scope", clear=True):
            put_html(
                f"<div style='opacity:0.85; font-size:12.5px; margin:6px 0 8px 0;'>"
                f"{t('target_preview_source_label')}: <b>{src}</b> &nbsp;|&nbsp; {t('target_preview_correction_band_label')}: "
                f"<b>{mag_c_min:.0f}</b>-<b>{mag_c_max:.0f}</b> Hz &nbsp;|&nbsp; Level window: <b>{lvl_min:.0f}-{lvl_max:.0f} Hz</b> "
                f"&nbsp;|&nbsp; Level mode: <b>{lvl_mode_label}</b>"
                f"{f' (target {lvl_manual_db:.1f} dB, speaker shift {preview_level_shift_db:+.1f} dB)' if is_manual_level else ''}"
                f" &nbsp;|&nbsp; Speaker data: <b>{speaker_label}</b>"
                f"</div>"
            )
            put_html(html)

    except Exception as exc:
        with use_scope("target_preview_scope", clear=True):
            put_html(
                "<div style='opacity:0.8; font-size:13px; padding:8px 0;'>"
                "Target preview failed. See console/log for details."
                "</div>"
            )
        try:
            logger.warning("update_target_preview_ui error: %r", exc)
        except Exception:
            pass


__all__ = [
    "update_target_preview_ui",
]
