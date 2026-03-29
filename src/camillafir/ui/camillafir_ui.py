import logging

from pywebio.output import put_error, set_processbar, use_scope
from pywebio.pin import pin

from ..common.result_postprocess import _irwin_tag
from ..resources.i8n.camillafir_i18n import t
from . import app as _app
from .results_formatters import plot_smoothing_label
from .results_sections import (
    render_auto_mode_diagnostics_section,
    render_dsp_quality_metrics_section,
    render_plot_export_section,
    render_raw_debug_section,
    render_run_overview_section,
)

logger = logging.getLogger("CamillaFIR")
PROGRAM_NAME = "CamillaFIR"
VERSION = ""
MAX_SAFE_BOOST = 8.0


def build_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float):
    g = globals()
    g["PROGRAM_NAME"] = PROGRAM_NAME
    g["VERSION"] = VERSION
    g["MAX_SAFE_BOOST"] = float(MAX_SAFE_BOOST)
    return _app.build_app(
        process_run=process_run,
        PROGRAM_NAME=PROGRAM_NAME,
        VERSION=VERSION,
        MAX_SAFE_BOOST=MAX_SAFE_BOOST,
    )


main = _app.main
update_status = _app.update_status
update_status_notices = _app.update_status_notices
update_auto_selected_bar = _app.update_auto_selected_bar


def _log_df_smoothing_for_fs(cfg, fs_v, df_on):
    if df_on:
        try:
            fsmooth = float(getattr(cfg, "filter_smooth", getattr(cfg, "smoothing_level", 12)) or 12)
            if fsmooth <= 0:
                fsmooth = 12
            base_sigma = 60 // (fsmooth / 12 if fsmooth > 0 else 1)
            df_ref = 44100.0 / 65536.0
            sigma_hz = base_sigma * df_ref
            df_cur = fs_v / cfg.num_taps
            sigma_bins = sigma_hz / df_cur if df_cur > 0 else base_sigma
            logger.info(
                f"{fs_v//1000} kHz -> DF smoothing ON "
                f"(sigma = {sigma_bins:.1f} bins -> {sigma_hz:.1f} Hz)"
            )
        except Exception:
            logger.info(f"{fs_v//1000} kHz -> DF smoothing ON")
    else:
        logger.info(f"{fs_v//1000} kHz -> DF smoothing OFF")


def _pin_get(key, default=None):
    """Sisainen apufunktio: pin get."""
    try:
        v = pin.get(key, None)
        if v is None:
            return default
        return v
    except Exception:
        try:
            return pin[key]
        except Exception:
            return default


def _json_safe(obj, *, _depth=0, _max_depth=12):
    """Sisainen apufunktio: json safe."""
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
    """Sisainen apufunktio: build diagnostics dict."""

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

    diag = {
        "schema_version": 1,
        "meta": {
            "program": PROGRAM_NAME,
            "version": VERSION,
            "fs_hz": int(fs_v),
            "taps": int(float(data.get("taps", 0) or 0)),
            "filter_type": str(data.get("filter_type", "") or ""),
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
    return diag


def _render_results(
    data,
    f_l,
    m_l,
    p_l,
    f_r,
    m_r,
    p_r,
    l_imp_f,
    r_imp_f,
    l_st_f,
    r_st_f,
    fname,
    zip_buffer,
    *,
    dash_html_l=None,
    dash_html_r=None,
    run_started_at=None,
    perf_stats=None,
    per_fs_stats=None,
    saved_filters_dir=None,
    auto_cache_path=None,
    optuna_storage_path=None,
):
    import time

    render_started_at = time.perf_counter()
    if run_started_at is not None:
        try:
            elapsed_now = max(0.0, float(time.perf_counter() - float(run_started_at)))
            update_status(f"{t('stat_plot')} | {elapsed_now:.1f} s")
        except Exception:
            update_status(t("stat_plot"))
    else:
        update_status(t("stat_plot"))

    time.sleep(0.05)
    set_processbar("bar", 0.8)
    psl_str = plot_smoothing_label(data.get("plot_smoothing_level", "Psychoacoustic"))

    with use_scope("results", clear=True):
        if l_st_f is None or r_st_f is None:
            put_error("Error: No results captured.")
            return

        render_run_overview_section(
            data=data,
            l_st_f=l_st_f,
            r_st_f=r_st_f,
        )
        render_auto_mode_diagnostics_section(data=data)
        render_dsp_quality_metrics_section(
            data=data,
            l_st_f=l_st_f,
            r_st_f=r_st_f,
            psl_str=psl_str,
        )
        render_plot_export_section(
            data=data,
            f_l=f_l,
            m_l=m_l,
            p_l=p_l,
            f_r=f_r,
            m_r=m_r,
            p_r=p_r,
            l_imp_f=l_imp_f,
            r_imp_f=r_imp_f,
            l_st_f=l_st_f,
            r_st_f=r_st_f,
            fname=fname,
            zip_buffer=zip_buffer,
            dash_html_l=dash_html_l,
            dash_html_r=dash_html_r,
            saved_filters_dir=saved_filters_dir,
            auto_cache_path=auto_cache_path,
            optuna_storage_path=optuna_storage_path,
        )
        render_raw_debug_section(
            render_started_at=render_started_at,
            run_started_at=run_started_at,
            perf_stats=perf_stats,
            per_fs_stats=per_fs_stats,
        )

        done_status = t("stat_done")
        if saved_filters_dir:
            try:
                done_status = done_status.format(path=str(saved_filters_dir))
            except Exception:
                done_status = f"{done_status} {saved_filters_dir}"

        done_msg = t("done_msg")
        if run_started_at is not None:
            try:
                total_s = max(0.0, float(time.perf_counter() - float(run_started_at)))
                update_status_notices(summary_text=done_msg, info_text="")
                update_status(f"{done_status} | {total_s:.1f} s")
            except Exception:
                update_status_notices(summary_text=done_msg, info_text="")
                update_status(done_status)
        else:
            update_status_notices(summary_text=done_msg, info_text="")
            update_status(done_status)
        set_processbar("bar", 1.0)
    return main
