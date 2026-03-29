import json
import logging
import os
from typing import Any

from . import camillafir_plot as plots
from .export_scoring import _append_export_ranking
from .export_summary_text import (
    _append_acoustic_events,
    _append_dsp_effective_params,
    _append_realized_phase_limit,
)
from ..app_paths import program_version_token
from ..common.result_postprocess import _irwin_tag
from ..config.camillafir_convolver_configs import generate_hlc_config, generate_raspberry_yaml
from ..config.results import FilterResult
from ..auto_mode.rank_score import attach_official_rank_score, official_rank_score

logger = logging.getLogger("CamillaFIR")

_PROGRAM_NAME = "CamillaFIR"


def _json_safe(obj, *, _depth=0, _max_depth=12):
    try:
        if _depth > _max_depth:
            return str(obj)
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        try:
            import numpy as _np
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    ks = str(k)
                except Exception:
                    ks = "key"
                out[ks] = _json_safe(v, _depth=_depth + 1, _max_depth=_max_depth)
            return out
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8", errors="replace")
            except Exception:
                return str(obj)
        return str(obj)
    except Exception:
        return str(obj)


def _build_diagnostics_dict(data, fs_v, l_st, r_st):
    from ..version import VERSION as _version  # noqa: PLC0415

    def _leveling_block(st):
        if not isinstance(st, dict):
            return {}
        win = st.get("smart_scan_range", None)
        try:
            if isinstance(win, (list, tuple)) and len(win) >= 2:
                win = [float(win[0]), float(win[1])]
            else:
                win = None
        except Exception:
            win = None
        return {
            "method": st.get("offset_method", None),
            "window_hz": win,
            "offset_db": st.get("offset_db", None),
            "eff_target_db": st.get("eff_target_db", None),
            "tilt_slope_db_per_oct": st.get("tilt_slope_db_per_oct", None),
            "avg_confidence_pct": st.get("avg_confidence", None),
        }

    return {
        "schema_version": 1,
        "meta": {
            "program": _PROGRAM_NAME,
            "version": str(data.get("program_version", _version) or _version),
            "fs_hz": int(fs_v),
            "taps": int(float(data.get("taps", 0) or 0)),
            "filter_type": str(data.get("filter_type", "") or ""),
            "layout": str(data.get("layout", "Mono") or "Mono"),
            "multi_rate": bool(data.get("multi_rate_opt", False)),
            "ir_export_window_mode": str(data.get("ir_export_window_mode", "") or ""),
            "ir_export_window_tag": str(_irwin_tag(data.get("ir_export_window_mode"))),
        },
        "settings": _json_safe(data),
        "leveling": {
            "stereo_link": bool(data.get("stereo_link", False)),
            "left": _leveling_block(l_st),
            "right": _leveling_block(r_st),
        },
        "left": _json_safe(l_st),
        "right": _json_safe(r_st),
    }

TEST_MODE = os.environ.get("CAMILLAFIR_TEST", "0") == "1"


def _export_version_tag(data: dict | None, *, program_version: str | None = None) -> str:
    raw_version = program_version
    if raw_version is None:
        try:
            raw_version = str((data or {}).get("program_version", "") or "").strip()
        except Exception:
            raw_version = ""
    return str(program_version_token(raw_version, default="v0"))


def _export_winner_rank_score(data: dict | None) -> float:
    try:
        auto_used = bool((data or {}).get("camillafir_automatic_mode", False))
    except Exception:
        auto_used = False
    if not auto_used:
        return float("nan")

    try:
        auto_meta = dict((data or {}).get("_auto_mode_meta", {}) or {})
        best_metrics = attach_official_rank_score(auto_meta.get("best_metrics", {}))
    except Exception:
        return float("nan")

    return float(official_rank_score(best_metrics))


def _export_winner_rank_tag(data: dict | None) -> str:
    rank_score = _export_winner_rank_score(data)
    if not (rank_score == rank_score) or abs(rank_score) == float("inf"):
        return ""
    return f"rank{rank_score:.3f}"


def _camilladsp_yaml_name(
    *,
    data: dict | None,
    ft_short: str,
    irw_tag: str,
    fs_v: int | None = None,
) -> str:
    parts = ["camilladsp", str(ft_short)]
    if fs_v is not None:
        parts.append(f"{int(fs_v)}Hz")
    parts.append(str(irw_tag))
    parts.append(_export_version_tag(data))
    rank_tag = _export_winner_rank_tag(data)
    if rank_tag:
        parts.append(rank_tag)
    return "_".join(parts) + ".yml"


def _extract_result_payload(result: FilterResult) -> tuple[Any, ...]:
    meas = dict(getattr(result, "measurements", {}) or {})
    return (
        int(result.fs),
        meas.get("f_l"),
        meas.get("m_l"),
        meas.get("p_l"),
        result.l_ir,
        result.l_st,
        meas.get("f_r"),
        meas.get("m_r"),
        meas.get("p_r"),
        result.r_ir,
        result.r_st,
    )


def _write_fs_outputs(
    zf,
    data,
    fs_v,
    ft_short,
    file_ts,
    f_l,
    m_l,
    p_l,
    l_imp,
    l_st,
    f_r,
    m_r,
    p_r,
    r_imp,
    r_st,
    *,
    result: FilterResult | None = None,
    write_dashboards: bool = True,
    irw_tag: str = "auto",
    ui_dashboards: dict | None = None,
    ranking_context: dict | None = None,
):
    if result is not None:
        (
            fs_v,
            f_l,
            m_l,
            p_l,
            l_imp,
            l_st,
            f_r,
            m_r,
            p_r,
            r_imp,
            r_st,
        ) = _extract_result_payload(result)

    sum_name = f"Summary_{ft_short}_{fs_v}Hz_{file_ts}.txt"
    l_dash_name = f"L_Dashboard_{ft_short}_{fs_v}Hz.png"
    r_dash_name = f"R_Dashboard_{ft_short}_{fs_v}Hz.png"

    summary_content = plots.format_summary_content(data, l_st, r_st)
    summary_content = _append_export_ranking(summary_content, int(fs_v), ranking_context)
    try:
        hc_src = str(data.get("hc_source", "") or "").strip()
        if hc_src:
            summary_content = f"House curve: {hc_src}\n" + summary_content
    except Exception:
        pass
    summary_content = _append_dsp_effective_params(summary_content, data, fs_v)
    summary_content = _append_realized_phase_limit(summary_content, data, l_st, r_st)
    try:
        summary_content += "\n=== LEVELING ===\n"
        for side, st in [("LEFT", l_st), ("RIGHT", r_st)]:
            if not isinstance(st, dict):
                continue
            summary_content += f"[{side}]\n"
            summary_content += f"Method: {st.get('offset_method', 'n/a')}\n"
            win = st.get("smart_scan_range", None)
            if isinstance(win, (list, tuple)) and len(win) >= 2:
                try:
                    summary_content += f"Window: {float(win[0]):.0f}-{float(win[1]):.0f} Hz\n"
                except Exception:
                    pass
            try:
                summary_content += f"Offset to measurement: {float(st.get('offset_db', 0.0) or 0.0):+.2f} dB\n"
            except Exception:
                pass
            try:
                summary_content += f"Effective target level: {float(st.get('eff_target_db', 0.0) or 0.0):.2f} dB\n"
            except Exception:
                pass
            tilt = st.get("tilt_slope_db_per_oct", None)
            if tilt is not None:
                try:
                    tilt_f = float(tilt)
                    summary_content += f"Tilt slope: {tilt_f:+.2f} dB/oct\n"
                    if abs(tilt_f) > 1.5:
                        summary_content += (
                            "Warning: Large broadband tilt detected. "
                            "May indicate measurement/target mismatch or strong room tilt.\n"
                        )
                except Exception:
                    pass
            summary_content += "\n"
    except Exception:
        pass

    summary_content = _append_acoustic_events(summary_content, l_st, r_st)

    if "auto_align" in l_st:
        res = l_st["auto_align"]
        summary_content += "\n=== AUTO-ALIGN ===\n"
        summary_content += f"Delay: {res['delay_ms']} ms\n"
        summary_content += f"Distance Diff: {res['distance_cm']} cm\n"
        summary_content += f"Gain Diff: {res['gain_diff_db']} dB\n"

    if TEST_MODE:
        try:
            diag = _build_diagnostics_dict(data, fs_v, l_st, r_st)
            summary_content += "\n\n--- DIAGNOSTICS_JSON_BEGIN ---\n"
            summary_content += json.dumps(_json_safe(diag), indent=2)
            summary_content += "\n--- DIAGNOSTICS_JSON_END ---\n"
        except Exception as e:
            summary_content += "\n\n--- DIAGNOSTICS_JSON_BEGIN ---\n"
            summary_content += json.dumps(
                {
                    "schema_version": 1,
                    "error": f"diagnostics_json_failed: {type(e).__name__}: {e}",
                },
                indent=2,
            )
            summary_content += "\n--- DIAGNOSTICS_JSON_END ---\n"

    zf.writestr(sum_name, summary_content)

    if bool(write_dashboards):
        psl = data.get("plot_smoothing_level", "Psychoacoustic")

        html_l = plots.generate_prediction_plot(
            f_l,
            m_l,
            p_l,
            l_imp,
            fs_v,
            "Left",
            None,
            l_st,
            data["mixed_freq"],
            "low",
            create_full_html=False,
            plot_smoothing_level=psl,
        )
        if isinstance(ui_dashboards, dict):
            ui_dashboards["fs_hz"] = int(fs_v)
            ui_dashboards["left_html"] = str(html_l)
        png_l = plots.generate_combined_plot_mpl(
            f_l,
            m_l,
            p_l,
            l_imp,
            fs_v,
            "Left",
            target_stats=l_st,
        )
        if png_l:
            zf.writestr(l_dash_name, png_l)
        else:
            zf.writestr(l_dash_name.replace(".png", ".txt"), str(html_l))

        html_r = plots.generate_prediction_plot(
            f_r,
            m_r,
            p_r,
            r_imp,
            fs_v,
            "Right",
            None,
            r_st,
            data["mixed_freq"],
            "low",
            create_full_html=False,
            plot_smoothing_level=psl,
        )
        if isinstance(ui_dashboards, dict):
            ui_dashboards["right_html"] = str(html_r)
        png_r = plots.generate_combined_plot_mpl(
            f_r,
            m_r,
            p_r,
            r_imp,
            fs_v,
            "Right",
            target_stats=r_st,
        )
        if png_r:
            zf.writestr(r_dash_name, png_r)
        else:
            zf.writestr(r_dash_name.replace(".png", ".txt"), str(html_r))

    target_curve_tag = str(data.get("target_curve_tag", "") or "").strip()

    hlc_cfg = generate_hlc_config(
        fs_v,
        ft_short,
        file_ts,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        layout=data.get("layout", "Mono"),
    )
    zf.writestr(f"Config_{ft_short}_{fs_v}Hz_{irw_tag}.cfg", hlc_cfg)

    if not bool(data.get("multi_rate_opt", False)):
        yaml_content = generate_raspberry_yaml(
            fs_v,
            ft_short,
            file_ts,
            master_gain_db=0.0,
            irw_tag=irw_tag,
            target_curve_tag=target_curve_tag,
            layout=data.get("layout", "Mono"),
            program_version=str(data.get("program_version", "") or "").strip(),
            winner_rank_score=_export_winner_rank_score(data),
        )
        zf.writestr(
            _camilladsp_yaml_name(data=data, ft_short=ft_short, irw_tag=irw_tag, fs_v=int(fs_v)),
            yaml_content,
        )
