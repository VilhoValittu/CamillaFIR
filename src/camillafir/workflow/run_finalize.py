from __future__ import annotations

import logging
import time
import typing

from ..auto_mode.api import AUTO_MODE_COMPAT_VERSION, _auto_optuna_storage_path, get_auto_mode_cache_path

if typing.TYPE_CHECKING:
    from .process_run_flow import ProcessRunSupport
    from ..ui.ng_bridge import ProcessRunCallbacks

logger = logging.getLogger("CamillaFIR")


def _finalize_run_outputs(ctx: dict, *, callbacks: ProcessRunCallbacks, support: ProcessRunSupport):
    data = ctx["data"]
    results_by_fs = ctx["results_by_fs"]
    perf_stats = ctx["perf_stats"]
    per_fs_stats = ctx["per_fs_stats"]
    ft_short = ctx["ft_short"]
    file_ts = ctx["file_ts"]
    irw_tag = ctx["irw_tag"]
    dash_fs = int(ctx["dash_fs"])
    ts = ctx["ts"]
    target_curve_tag = ctx["target_curve_tag"]

    zip_started_at = time.perf_counter()
    zip_buffer, ui_dashboards, zip_perf = support.ui_bridge.build_export_zip(
        data=data,
        results=results_by_fs,
        ft_short=ft_short,
        file_ts=file_ts,
        irw_tag=irw_tag,
        write_dashboards=bool(ctx["zip_dashboards_on"]),
        dash_fs=dash_fs,
    )
    ctx["ui_dashboards"] = ui_dashboards
    zip_elapsed = max(0.0, float(time.perf_counter() - zip_started_at))
    perf_stats["zip_png_s"] += max(float(zip_perf.get("zip_png_s", 0.0) or 0.0), zip_elapsed)
    for fs_key, st in (zip_perf.get("per_fs_stats", {}) or {}).items():
        slot = per_fs_stats.setdefault(int(fs_key), {})
        slot["zip_png_s"] = float(slot.get("zip_png_s", 0.0)) + float(st.get("zip_png_s", 0.0) or 0.0)

    fname, saved_filters_dir, _save_msg = support.ui_bridge.save_export_bundle(
        zip_buffer,
        data=data,
        ft_short=ft_short,
        irw_tag=irw_tag,
        target_curve_tag=target_curve_tag,
        ts=ts,
        program_version=str(data.get("program_version", support.version) or support.version),
    )
    auto_cache_path = None
    optuna_storage_path = None
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    if bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False)):
        try:
            auto_cache_path = str(
                get_auto_mode_cache_path(
                    compat_version=str(
                        data.get("auto_mode_compat_version", AUTO_MODE_COMPAT_VERSION)
                        or AUTO_MODE_COMPAT_VERSION
                    ),
                )
            )
        except Exception:
            auto_cache_path = None
        try:
            if bool(data.get("auto_mode_optuna_persistent_study", True)):
                optuna_storage_path = str(
                    _auto_optuna_storage_path(
                        compat_version=str(
                            data.get("auto_mode_compat_version", AUTO_MODE_COMPAT_VERSION)
                            or AUTO_MODE_COMPAT_VERSION
                        ),
                    )
                )
            else:
                optuna_storage_path = None
        except Exception:
            optuna_storage_path = None

    l_st_f = ctx["l_st_f"]
    r_st_f = ctx["r_st_f"]
    l_imp_f = ctx["l_imp_f"]
    r_imp_f = ctx["r_imp_f"]
    if l_st_f is None or r_st_f is None or l_imp_f is None or r_imp_f is None:
        fallback = results_by_fs[-1]
        l_st_f, r_st_f = fallback.l_st, fallback.r_st
        l_imp_f, r_imp_f = fallback.l_ir, fallback.r_ir

    sub_ir_f = ctx.get("sub_ir_f")
    sub_st_f = ctx.get("sub_st_f")
    sub_meas_f = dict(ctx.get("sub_meas_f") or {})
    if (sub_ir_f is None or sub_st_f is None) and results_by_fs:
        fallback_r = results_by_fs[-1]
        if sub_ir_f is None:
            sub_ir_f = fallback_r.sub_ir
        if sub_st_f is None:
            sub_st_f = fallback_r.sub_st
        if not sub_meas_f:
            sub_meas_f = {k: fallback_r.measurements[k] for k in ("f_sub", "m_sub", "p_sub") if k in fallback_r.measurements}
    bi_meta = data.get("_bass_integration_meta")
    if isinstance(bi_meta, dict) and isinstance(sub_st_f, dict):
        bi_meta["sub_filter_stats"] = sub_st_f

    logger.info(
        f"UI stats mode L/R: {l_st_f.get('analysis_mode')}/{r_st_f.get('analysis_mode')} | "
        f"len cmp f/m/t = {len(l_st_f.get('cmp_freq_axis', []))}/{len(l_st_f.get('cmp_measured_mags', []))}/{len(l_st_f.get('cmp_target_mags', []))}"
    )

    support.ui_bridge.render_results(
        data,
        ctx["f_l"],
        ctx["m_l"],
        ctx["p_l"],
        ctx["f_r"],
        ctx["m_r"],
        ctx["p_r"],
        l_imp_f,
        r_imp_f,
        l_st_f,
        r_st_f,
        fname,
        zip_buffer,
        dash_html_l=ui_dashboards.get("left_html"),
        dash_html_r=ui_dashboards.get("right_html"),
        run_started_at=ctx["run_started_at"],
        perf_stats=perf_stats,
        per_fs_stats=per_fs_stats,
        saved_filters_dir=saved_filters_dir,
        auto_cache_path=auto_cache_path,
        optuna_storage_path=optuna_storage_path,
        sub_imp_f=sub_ir_f,
        sub_meas_f=sub_meas_f,
        sub_st_f=sub_st_f,
    )
