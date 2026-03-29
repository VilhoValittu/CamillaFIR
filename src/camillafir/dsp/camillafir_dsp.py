import numpy as np
import scipy.ndimage
import copy
import logging
logger = logging.getLogger("CamillaFIR.dsp")
from camillafir.config.models import FilterConfig
from .dsp_correction import run_correction_stage
from ._pruning import get_pruning_hook as _get_pruning_hook
from .dsp_ops import (
    _limit_gd_gradient_ms_per_oct as _limit_gd_gradient_ms_per_oct_impl,
    _stage_probe as _stage_probe_impl,
    apply_confidence_weighted_target_pull as _apply_confidence_weighted_target_pull_impl,
    apply_hpf_to_mags as _apply_hpf_to_mags_impl,
    interpolate_response as _interpolate_response_impl,
)
from .dsp_phase_ir import run_phase_ir_stage
from .dsp_preprocess import run_preprocess
from .dsp_stats import (
    apply_afdw_stats as _apply_afdw_stats_impl,
    apply_boost_blocked_reason as _apply_boost_blocked_reason_impl,
    apply_clamp_stats as _apply_clamp_stats_impl,
    apply_lf_guard_stats as _apply_lf_guard_stats_impl,
    apply_measured_mag_stats as _apply_measured_mag_stats_impl,
    arr_if_valid_for_stats as _arr_if_valid_for_stats_impl,
    safe_stage_probes as _safe_stage_probes_impl,
    safe_stats_update as _safe_stats_update_impl,
)
from .camillafir_leveling import StereoLinkContext, find_shared_stereo_level_window
from .dsp_utils import cfg_float_allow_zero as _cfg_float_allow_zero
from .dsp_utils import safe_range as _safe_range
from .filter_pipeline import _run_generate_filter_pipeline
from .filter_result import _assemble_generate_filter_result


def _safe_stats_update(stats: dict, extra) -> None:
    return _safe_stats_update_impl(stats, extra)


def _arr_if_valid_for_stats(value, *, expected_size: int | None = None):
    return _arr_if_valid_for_stats_impl(value, expected_size=expected_size)


def _apply_measured_mag_stats(
    stats: dict,
    *,
    target_mags,
    freq_axis,
    m_anal,
    calc_offset_db: float,
) -> None:
    return _apply_measured_mag_stats_impl(
        stats,
        target_mags=target_mags,
        freq_axis=freq_axis,
        m_anal=m_anal,
        calc_offset_db=calc_offset_db,
    )


def _apply_afdw_stats(
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
    return _apply_afdw_stats_impl(
        stats,
        afdw_on=afdw_on,
        afdw_bw_oct=afdw_bw_oct,
        afdw_bw_min_oct=afdw_bw_min_oct,
        afdw_bw_mean_oct=afdw_bw_mean_oct,
        afdw_bw_max_oct=afdw_bw_max_oct,
        afdw_bw_min_hz=afdw_bw_min_hz,
        afdw_bw_max_hz=afdw_bw_max_hz,
    )


def _safe_stage_probes(stage_probes) -> dict:
    return _safe_stage_probes_impl(stage_probes)


def _apply_lf_guard_stats(stats: dict, *, cfg, freq_axis, gain_db) -> None:
    return _apply_lf_guard_stats_impl(stats, cfg=cfg, freq_axis=freq_axis, gain_db=gain_db)


def _apply_clamp_stats(
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
    return _apply_clamp_stats_impl(
        stats,
        softclip_boost_bins=softclip_boost_bins,
        softclip_cut_bins=softclip_cut_bins,
        over_boost=over_boost,
        over_cut=over_cut,
        hardclamp_boost_bins=hardclamp_boost_bins,
        hardclamp_cut_bins=hardclamp_cut_bins,
        hard_over_boost=hard_over_boost,
        hard_over_cut=hard_over_cut,
    )


def _apply_boost_blocked_reason(stats: dict, *, cfg) -> None:
    return _apply_boost_blocked_reason_impl(stats, cfg=cfg)


def apply_confidence_weighted_target_pull(
    target_db,
    measured_db,
    confidence_mask,
    *,
    conf_floor: float = 0.07,
    conf_ceil: float = 0.95,
    freq_axis=None,
    freq_limit_hz: float | None = 400.0,
    gamma_cut: float = 0.70,
    gamma_boost: float = 1.20,
    return_telemetry: bool = False,
):
    return _apply_confidence_weighted_target_pull_impl(
        target_db,
        measured_db,
        confidence_mask,
        conf_floor=conf_floor,
        conf_ceil=conf_ceil,
        freq_axis=freq_axis,
        freq_limit_hz=freq_limit_hz,
        gamma_cut=gamma_cut,
        gamma_boost=gamma_boost,
        return_telemetry=return_telemetry,
    )

def _limit_gd_gradient_ms_per_oct(
    freq_axis,
    phase_rad,
    *,
    mask=None,
    max_grad_ms_per_oct=30.0,
    f_min=20.0,
    f_max=250.0,
    grad_smooth_sigma=0.8,
    soft_limit=True,
):
    return _limit_gd_gradient_ms_per_oct_impl(
        freq_axis,
        phase_rad,
        mask=mask,
        max_grad_ms_per_oct=max_grad_ms_per_oct,
        f_min=f_min,
        f_max=f_max,
        grad_smooth_sigma=grad_smooth_sigma,
        soft_limit=soft_limit,
    )



def _stage_probe(stage_name, freq_axis, arr_db, mask_c, global_gain_db=0.0, auto_headroom_db=0.0, logger_obj=None):
    return _stage_probe_impl(
        stage_name,
        freq_axis,
        arr_db,
        mask_c,
        global_gain_db=global_gain_db,
        auto_headroom_db=auto_headroom_db,
        logger_obj=logger_obj,
    )


def apply_hpf_to_mags(freqs, mags, cutoff, order):
    return _apply_hpf_to_mags_impl(freqs, mags, cutoff, order)


def interpolate_response(input_freqs, input_values, target_freqs):
    return _interpolate_response_impl(input_freqs, input_values, target_freqs)




def generate_filter(freqs, meas_mags, raw_phases, cfg: FilterConfig, *, stereo_link_ctx: StereoLinkContext | None = None):
    pipeline = _run_generate_filter_pipeline(
        freqs,
        meas_mags,
        raw_phases,
        cfg,
        stereo_link_ctx=stereo_link_ctx,
    )
    freq_axis = pipeline["freq_axis"]
    st = pipeline["st"]
    reflections = pipeline["reflections"]
    target_mags = pipeline["target_mags"]
    m_anal = pipeline["m_anal"]
    conf_mask = pipeline["conf_mask"]
    cmp = pipeline["cmp"]
    analysis_mode = pipeline["analysis_mode"]
    delay_slope = pipeline["delay_slope"]
    current_rt60 = pipeline["current_rt60"]
    rt60_bands = pipeline["rt60_bands"]
    band_avg = pipeline["band_avg"]
    target_level_db = pipeline["target_level_db"]
    calc_offset_db = pipeline["calc_offset_db"]
    meas_level_db_window = pipeline["meas_level_db_window"]
    target_level_db_window = pipeline["target_level_db_window"]
    offset_method = pipeline["offset_method"]
    s_min = pipeline["s_min"]
    s_max = pipeline["s_max"]
    target_shift_db = pipeline["target_shift_db"]
    gain_db = pipeline["gain_db"]
    afdw_on = pipeline["afdw_on"]
    mask_c = pipeline["mask_c"]
    stage_probes = pipeline["stage_probes"]
    use_bassfirst = pipeline["use_bassfirst"]
    bf_room_mode = pipeline["bf_room_mode"]
    bf_rel = pipeline["bf_rel"]
    bf_conf_for_smoothing = pipeline["bf_conf_for_smoothing"]
    boost_peak_db = pipeline["boost_peak_db"]
    cut_peak_db = pipeline["cut_peak_db"]
    n_boost = pipeline["n_boost"]
    boost_cand_peak = pipeline["boost_cand_peak"]
    boost_cand_min_hz = pipeline["boost_cand_min_hz"]
    n_boost_cand = pipeline["n_boost_cand"]
    n_boost_cand_low = pipeline["n_boost_cand_low"]
    n_boost_cand_exc = pipeline["n_boost_cand_exc"]
    softclip_boost_bins = pipeline["softclip_boost_bins"]
    softclip_cut_bins = pipeline["softclip_cut_bins"]
    over_boost = pipeline["over_boost"]
    over_cut = pipeline["over_cut"]
    hardclamp_boost_bins = pipeline["hardclamp_boost_bins"]
    hardclamp_cut_bins = pipeline["hardclamp_cut_bins"]
    hard_over_boost = pipeline["hard_over_boost"]
    hard_over_cut = pipeline["hard_over_cut"]
    impulse = pipeline["impulse"]
    auto_global_gain_db = pipeline["auto_global_gain_db"]
    gain_margin_db = pipeline["gain_margin_db"]
    auto_headroom_db = pipeline["auto_headroom_db"]
    current_peak_gain = pipeline["current_peak_gain"]
    final_gain_total = pipeline["final_gain_total"]

    max_peak = np.max(np.abs(impulse))
    normalize_gain_db_applied = 0.0
    if cfg.do_normalize and max_peak > 0:
        _norm_scale = float(0.89 / max_peak)
        impulse *= _norm_scale
        try:
            normalize_gain_db_applied = float(20.0 * np.log10(max(_norm_scale, 1e-12)))
        except (TypeError, ValueError, FloatingPointError):
            normalize_gain_db_applied = 0.0

    stats = {

        'analysis_mode': analysis_mode,
        'freq_axis': freq_axis.tolist(),
        'mag_c_min': float(getattr(cfg, 'mag_c_min', 0.0) or 0.0),
        'mag_c_max': float(getattr(cfg, 'mag_c_max', 0.0) or 0.0),
        'target_mags': target_mags.tolist(),
        'measured_mags_raw': m_anal.tolist(),
        'measured_mags': (m_anal - calc_offset_db).tolist(),
        'predicted_filter_mags': gain_db.tolist(),
        'predicted_filter_mags_source': "mag_post_limits_pre_ir",
        'filter_mags': gain_db.tolist(),
        'filter_mags_source': "mag_post_limits_pre_ir",
        'mag_mask': np.asarray(mask_c, dtype=float).tolist(),
        'confidence_mask': conf_mask.tolist(),
        'afdw_active': bool(afdw_on),
        'reflections': reflections,
        'smart_scan_range': [float(s_min), float(s_max)],
        'eff_target_db': float(target_level_db),
        'offset_db': float(calc_offset_db),
        'meas_level_db_window': float(meas_level_db_window),
        'target_level_db_window': float(target_level_db_window),
        'offset_method': str(offset_method),
            'tilt_slope_db_per_oct': (
                float(getattr(cfg, "_lvl_tilt_slope_db_per_oct"))
                if getattr(cfg, "_lvl_tilt_slope_db_per_oct", None) is not None
                else None
            ),
        'rt60_val': float(current_rt60),
        'rt60_band_avg': float(band_avg),
        'rt60_bands': rt60_bands,        
        'avg_confidence': float(np.mean(conf_mask)*100),
        'delay_samples': float((delay_slope * cfg.fs) / (2 * np.pi)) if 'delay_slope' in locals() else 0.0,
        'peak_before_norm': float(20*np.log10(max_peak + 1e-12)),
        'do_normalize': bool(getattr(cfg, 'do_normalize', False)),
        'gain_margin_db': float(gain_margin_db),
        'auto_global_gain_db': float(auto_global_gain_db),
        'auto_headroom_db': float(auto_headroom_db),
        'normalize_gain_db_applied': float(normalize_gain_db_applied),
        'peak_gain_db': float(current_peak_gain),
        'final_max_db': float(np.max(final_gain_total)),

        'max_boost_db': float(getattr(cfg, 'max_boost_db', 0.0) or 0.0),
        'max_boost_db_effective': float(getattr(cfg, 'max_boost_db', 0.0) or 0.0),
        'max_boost_db_user': float(getattr(cfg, 'max_boost_db_user', getattr(cfg, 'max_boost_db', 0.0)) or 0.0),
        'max_safe_boost_db': float(getattr(cfg, 'max_safe_boost_db', 0.0) or 0.0),
        'max_cut_db': float(abs(float(getattr(cfg, 'max_cut_db', 15.0) or 15.0))),
        'low_bass_cut_hz': _cfg_float_allow_zero(cfg, "low_bass_cut_hz", 40.0),
        'exc_prot': bool(getattr(cfg, 'exc_prot', False)),
        'exc_freq': float(getattr(cfg, 'exc_freq', 0.0) or 0.0),
        'max_slope_db_per_oct': float(getattr(cfg, 'max_slope_db_per_oct', 0.0) or 0.0),
        'max_slope_boost_db_per_oct': float(getattr(cfg, 'max_slope_boost_db_per_oct', 0.0) or 0.0),
        'max_slope_cut_db_per_oct': float(getattr(cfg, 'max_slope_cut_db_per_oct', 0.0) or 0.0),
        

        'boost_peak_db': float(locals().get('boost_peak_db', 0.0)),
        'cut_peak_db': float(locals().get('cut_peak_db', 0.0)),
        'boost_bins': int(locals().get('n_boost', 0)),
        'boost_candidate_peak_db': float(locals().get('boost_cand_peak', 0.0)),
        'boost_candidate_min_hz': float(locals().get('boost_cand_min_hz', float("nan"))),
        'boost_candidate_bins': int(locals().get('n_boost_cand', 0)),
        'boost_candidate_bins_lowbass': int(locals().get('n_boost_cand_low', 0)),
        'boost_candidate_bins_excprot': int(locals().get('n_boost_cand_exc', 0)),


        'bass_first_ai': bool(locals().get('use_bassfirst', False)),

        'bass_first_mode_peak_hz': (
            float(freq_axis[int(np.argmax(np.asarray(bf_room_mode)))])
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_room_mode', None) is not None)
                and len(np.asarray(bf_room_mode)) > 0
            )
            else None
        ),

        'bass_first_mode_peak_score': (
            float(np.max(np.asarray(bf_room_mode)))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_room_mode', None) is not None)
                and len(np.asarray(bf_room_mode)) > 0
            )
            else None
        ),

        'bass_first_conf_floor_applied': (
            bool(
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_conf_for_smoothing', None) is not None)
                and (locals().get('bf_rel', None) is not None)
                and np.any(
                    np.asarray(bf_conf_for_smoothing) > (np.asarray(bf_rel) + 1e-6)
                )
            )
            if (
                locals().get('bf_conf_for_smoothing', None) is not None
                and locals().get('bf_rel', None) is not None
            )
            else False
        ),


        'bass_first_rel_mean_20_200': (
            float(np.mean(np.asarray(bf_rel)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_rel', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_rel_min_20_200': (
            float(np.min(np.asarray(bf_rel)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_rel', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_conf_eff_mean_20_200': (
            float(np.mean(np.asarray(bf_conf_for_smoothing)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_conf_for_smoothing', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_conf_eff_min_20_200': (
            float(np.min(np.asarray(bf_conf_for_smoothing)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_conf_for_smoothing', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_roommode_max_20_200': (
            float(np.max(np.asarray(bf_room_mode)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(locals().get('use_bassfirst', False))
                and (locals().get('bf_room_mode', None) is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

    }


    _safe_stats_update(stats, st)
    # Keep target on a shared absolute reference level for scoring/quality.
    # Preserve measured arrays from `st` when available to keep UI view
    # behavior stable; only fill missing measured fields.
    _apply_measured_mag_stats(
        stats,
        target_mags=target_mags,
        freq_axis=freq_axis,
        m_anal=m_anal,
        calc_offset_db=float(calc_offset_db),
    )
    _apply_afdw_stats(
        stats,
        afdw_on=bool(afdw_on),
        afdw_bw_oct=locals().get("afdw_bw_oct", None),
        afdw_bw_min_oct=locals().get("afdw_bw_min_oct", None),
        afdw_bw_mean_oct=locals().get("afdw_bw_mean_oct", None),
        afdw_bw_max_oct=locals().get("afdw_bw_max_oct", None),
        afdw_bw_min_hz=locals().get("afdw_bw_min_hz", None),
        afdw_bw_max_hz=locals().get("afdw_bw_max_hz", None),
    )
    stats["stage_probes"] = _safe_stage_probes(stage_probes)
    _apply_lf_guard_stats(stats, cfg=cfg, freq_axis=freq_axis, gain_db=gain_db)
    _apply_clamp_stats(
        stats,
        softclip_boost_bins=int(softclip_boost_bins),
        softclip_cut_bins=int(softclip_cut_bins),
        over_boost=float(over_boost),
        over_cut=float(over_cut),
        hardclamp_boost_bins=int(hardclamp_boost_bins),
        hardclamp_cut_bins=int(hardclamp_cut_bins),
        hard_over_boost=float(hard_over_boost),
        hard_over_cut=float(hard_over_cut),
    )
    _apply_boost_blocked_reason(stats, cfg=cfg)


    if isinstance(cmp, dict) and cmp:
        stats.update(cmp)
        if stats.get('analysis_mode') != "comparison":
            stats['analysis_mode'] = "native"
    try:
        cmp_g_pred = np.asarray(stats.get("cmp_predicted_filter_mags", []), dtype=float).reshape(-1)
        cmp_g_cur = np.asarray(stats.get("cmp_filter_mags", []), dtype=float).reshape(-1)
        if cmp_g_pred.size < 8 and cmp_g_cur.size >= 8:
            stats["cmp_predicted_filter_mags"] = cmp_g_cur.tolist()
            stats["cmp_predicted_filter_mags_source"] = str(
                stats.get("cmp_filter_mags_source", "mag_post_limits_pre_ir") or "mag_post_limits_pre_ir"
            )
    except (TypeError, ValueError):
        pass
    try:
        cmp_m_raw = np.asarray(stats.get("cmp_measured_mags_raw", []), dtype=float).reshape(-1)
        cmp_m_cur = np.asarray(stats.get("cmp_measured_mags", []), dtype=float).reshape(-1)
        if cmp_m_raw.size < 8 and cmp_m_cur.size >= 8:
            cmp_off = float(stats.get("cmp_offset_db", 0.0) or 0.0)
            stats["cmp_measured_mags_raw"] = (cmp_m_cur + cmp_off).tolist()
    except (TypeError, ValueError):
        pass

    # Canonical filter magnitude for reporting: always derive from final IR.
    # This keeps DSP Quality Report aligned with the exported/used filter.
    try:
        ir = np.asarray(impulse, dtype=float).flatten()
        fs_i = int(getattr(cfg, "fs", 0) or 0)
        f_native = np.asarray(stats.get("freq_axis", freq_axis), dtype=float).reshape(-1)
        if ir.size >= 8 and fs_i > 0 and f_native.size >= 4:
            h = np.fft.rfft(ir)
            f_fft = np.fft.rfftfreq(ir.size, d=1.0 / float(fs_i))
            g_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
            f_q = np.clip(f_native, float(np.min(f_fft)), float(np.max(f_fft)))
            g_pred_native = np.asarray(stats.get("predicted_filter_mags", []), dtype=float).reshape(-1)
            if g_pred_native.size < 8:
                g_cur_native = np.asarray(stats.get("filter_mags", []), dtype=float).reshape(-1)
                if g_cur_native.size >= 8:
                    stats["predicted_filter_mags"] = g_cur_native.tolist()
                    stats["predicted_filter_mags_source"] = str(
                        stats.get("filter_mags_source", "mag_post_limits_pre_ir") or "mag_post_limits_pre_ir"
                    )

            g_real_native = np.interp(f_q, f_fft, g_db)
            stats["realized_filter_mags"] = g_real_native.tolist()
            stats["realized_filter_mags_source"] = "ir_fft_final"
            stats["filter_mags"] = g_real_native.tolist()
            stats["filter_mags_source"] = "ir_fft_final"

            f_cmp = np.asarray(stats.get("cmp_freq_axis", []), dtype=float).reshape(-1)
            if f_cmp.size >= 4:
                f_cmp_q = np.clip(f_cmp, float(np.min(f_fft)), float(np.max(f_fft)))
                g_pred_cmp = np.asarray(stats.get("cmp_predicted_filter_mags", []), dtype=float).reshape(-1)
                if g_pred_cmp.size < 4:
                    g_cur_cmp = np.asarray(stats.get("cmp_filter_mags", []), dtype=float).reshape(-1)
                    if g_cur_cmp.size >= 4:
                        stats["cmp_predicted_filter_mags"] = g_cur_cmp.tolist()
                        stats["cmp_predicted_filter_mags_source"] = str(
                            stats.get("cmp_filter_mags_source", "mag_post_limits_pre_ir") or "mag_post_limits_pre_ir"
                        )
                g_real_cmp = np.interp(f_cmp_q, f_fft, g_db)
                stats["cmp_realized_filter_mags"] = g_real_cmp.tolist()
                stats["cmp_realized_filter_mags_source"] = "ir_fft_final"
                stats["cmp_filter_mags"] = g_real_cmp.tolist()
                stats["cmp_filter_mags_source"] = "ir_fft_final"

            # Realization delta diagnostics:
            # compare post-limits mag gain_db against final IR-derived filter_mags.
            try:
                g_post = np.asarray(gain_db, dtype=float).reshape(-1)
                g_ir = np.asarray(stats.get("filter_mags", []), dtype=float).reshape(-1)
                n = int(min(f_q.size, g_post.size, g_ir.size))
                if n >= 8:
                    f_eval = np.asarray(f_q[:n], dtype=float)
                    d_eval = np.asarray(g_ir[:n], dtype=float) - np.asarray(g_post[:n], dtype=float)
                    valid = np.isfinite(f_eval) & np.isfinite(d_eval) & (f_eval > 0.0)

                    def _band_delta_on(d_arr: np.ndarray, lo_hz: float, hi_hz: float):
                        m = valid & (f_eval >= float(lo_hz)) & (f_eval <= float(hi_hz))
                        if int(np.count_nonzero(m)) < 8:
                            return None, None, None
                        dv = np.asarray(d_arr[m], dtype=float)
                        fv = f_eval[m]
                        idx = int(np.argmax(np.abs(dv)))
                        return float(np.sqrt(np.mean(dv * dv))), float(np.abs(dv[idx])), float(fv[idx])

                    rms_b, max_b, hz_b = _band_delta_on(d_eval, 20.0, 200.0)
                    stats["post_to_ir_delta_rms_20_200_db"] = rms_b
                    stats["post_to_ir_delta_max_20_200_db"] = max_b
                    stats["post_to_ir_delta_max_hz_20_200"] = hz_b

                    m20 = valid & (f_eval >= 20.0) & (f_eval <= 200.0)
                    if int(np.count_nonzero(m20)) >= 8:
                        off20 = float(np.median(np.asarray(d_eval[m20], dtype=float)))
                        d_shape_20 = np.asarray(d_eval, dtype=float) - float(off20)
                        srms20, smax20, shz20 = _band_delta_on(d_shape_20, 20.0, 200.0)
                        stats["post_to_ir_delta_offset_20_200_db"] = float(off20)
                        stats["post_to_ir_shape_delta_rms_20_200_db"] = srms20
                        stats["post_to_ir_shape_delta_max_20_200_db"] = smax20
                        stats["post_to_ir_shape_delta_max_hz_20_200"] = shz20

                        # Same diagnostic, but baseline includes gain staging
                        # (auto gain/headroom + possible final normalize scale).
                        g_stage = (
                            np.asarray(g_post, dtype=float)
                            + float(auto_global_gain_db)
                            + float(auto_headroom_db)
                            + float(normalize_gain_db_applied)
                        )
                        d_stage = np.asarray(g_ir[:n], dtype=float) - np.asarray(g_stage[:n], dtype=float)
                        srms_abs, smax_abs, shz_abs = _band_delta_on(d_stage, 20.0, 200.0)
                        stats["post_to_ir_staged_delta_rms_20_200_db"] = srms_abs
                        stats["post_to_ir_staged_delta_max_20_200_db"] = smax_abs
                        stats["post_to_ir_staged_delta_max_hz_20_200"] = shz_abs

                        off_stage = float(np.median(np.asarray(d_stage[m20], dtype=float)))
                        d_stage_shape = np.asarray(d_stage, dtype=float) - float(off_stage)
                        srms_shape, smax_shape, shz_shape = _band_delta_on(d_stage_shape, 20.0, 200.0)
                        stats["post_to_ir_staged_delta_offset_20_200_db"] = float(off_stage)
                        stats["post_to_ir_staged_shape_delta_rms_20_200_db"] = srms_shape
                        stats["post_to_ir_staged_shape_delta_max_20_200_db"] = smax_shape
                        stats["post_to_ir_staged_shape_delta_max_hz_20_200"] = shz_shape

                    cmin = float(stats.get("mag_c_min", getattr(cfg, "mag_c_min", 20.0)) or 20.0)
                    cmax = float(stats.get("mag_c_max", getattr(cfg, "mag_c_max", 20000.0)) or 20000.0)
                    if (not np.isfinite(cmin)) or (cmin < 0.0):
                        cmin = 20.0
                    if (not np.isfinite(cmax)) or (cmax <= cmin):
                        cmax = max(200.0, cmin + 1.0)
                    rms_c, max_c, hz_c = _band_delta_on(d_eval, cmin, cmax)
                    stats["post_to_ir_delta_rms_magc_db"] = rms_c
                    stats["post_to_ir_delta_max_magc_db"] = max_c
                    stats["post_to_ir_delta_max_hz_magc"] = hz_c
                    m_c = valid & (f_eval >= float(cmin)) & (f_eval <= float(cmax))
                    if int(np.count_nonzero(m_c)) >= 8:
                        off_c = float(np.median(np.asarray(d_eval[m_c], dtype=float)))
                        d_shape_c = np.asarray(d_eval, dtype=float) - float(off_c)
                        srms_c, smax_c, shz_c = _band_delta_on(d_shape_c, cmin, cmax)
                        stats["post_to_ir_delta_offset_magc_db"] = float(off_c)
                        stats["post_to_ir_shape_delta_rms_magc_db"] = srms_c
                        stats["post_to_ir_shape_delta_max_magc_db"] = smax_c
                        stats["post_to_ir_shape_delta_max_hz_magc"] = shz_c
            except (TypeError, ValueError, FloatingPointError, IndexError):
                pass
    except (TypeError, ValueError, FloatingPointError, IndexError, KeyError):
        pass

    return _assemble_generate_filter_result(impulse, stats)

def generate_filter_pair(f_l, m_l, p_l, f_r, m_r, p_r, cfg: FilterConfig):
    """
    Generoi vasemman ja oikean kanavan FIR-suodattimet.

    Jos `stereo_link` ei ole paalla, kanavat lasketaan itsenaisesti.
    Jos `stereo_link` on paalla, toteutus tekee kaksivaiheisen ajon:
    1) alustava ajo molemmille kanaville (ikkuna + offset-arviot)
    2) yhteisen offsetin ja mahdollisen auto-gain-overriden laskenta
    3) uusi ajo strategian mukaan:
       - shared: sama ikkuna + sama offset molemmille
       - hybrid: kanavakohtainen ikkuna + sama offset molemmille
       - auto: guard-valinta shared/hybrid

    Palauttaa `(l_imp, l_stats, r_imp, r_stats)`.
    """
    if not bool(getattr(cfg, "stereo_link", False)):
        l_imp, l_st = generate_filter(f_l, m_l, p_l, cfg)
        r_imp, r_st = generate_filter(f_r, m_r, p_r, cfg)
        return l_imp, l_st, r_imp, r_st

    l_imp1, l_st1 = generate_filter(f_l, m_l, p_l, cfg)
    r_imp1, r_st1 = generate_filter(f_r, m_r, p_r, cfg)

    def _as_stat_float(st: dict | None, key: str, default=np.nan) -> float:
        try:
            if isinstance(st, dict):
                v = float(st.get(key, default))
                return v if np.isfinite(v) else float(default)
        except (TypeError, ValueError):
            pass
        return float(default)

    def _shared_window_from_stats(st_l: dict | None, st_r: dict | None):
        try:
            if not isinstance(st_l, dict) or not isinstance(st_r, dict):
                return None
            freq_l = np.asarray(st_l.get("freq_axis", []) or [], dtype=float)
            meas_l = np.asarray(st_l.get("measured_mags", []) or [], dtype=float)
            targ_l = np.asarray(st_l.get("target_mags", []) or [], dtype=float)
            freq_r = np.asarray(st_r.get("freq_axis", []) or [], dtype=float)
            meas_r = np.asarray(st_r.get("measured_mags", []) or [], dtype=float)
            targ_r = np.asarray(st_r.get("target_mags", []) or [], dtype=float)
            if (
                freq_l.size < 50
                or freq_r.size < 50
                or meas_l.size != freq_l.size
                or targ_l.size != freq_l.size
                or meas_r.size != freq_r.size
                or targ_r.size != freq_r.size
            ):
                return None
            try:
                hpf_settings = getattr(cfg, "hpf_settings", None)
                hpf_freq = float(hpf_settings.get("freq", 0.0)) if hpf_settings else 0.0
            except (AttributeError, TypeError, ValueError):
                hpf_freq = 0.0
            win = find_shared_stereo_level_window(
                freq_l,
                meas_l,
                targ_l,
                freq_r,
                meas_r,
                targ_r,
                float(lvl_min),
                float(lvl_max),
                window_size_octaves=1.0,
                hpf_freq=float(hpf_freq),
                tilt_comp=bool(getattr(cfg, "lvl_tilt_comp", True)),
                tilt_max_db_per_oct=float(getattr(cfg, "lvl_tilt_max_db_per_oct", 2.0) or 2.0),
                perceptual_weighting=bool(getattr(cfg, "lvl_perceptual_weighting", False)),
                perceptual_strength=float(getattr(cfg, "lvl_perceptual_strength", 0.12) or 0.12),
                perceptual_min_hz=float(getattr(cfg, "lvl_perceptual_min_hz", 250.0) or 250.0),
                perceptual_max_hz=float(getattr(cfg, "lvl_perceptual_max_hz", 4000.0) or 4000.0),
                perceptual_tie_only=bool(getattr(cfg, "lvl_perceptual_tie_only", True)),
            )
            return _safe_range(win, lvl_min, lvl_max)
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
            return None

    def _pick_quieter_anchor():
        left = {
            "channel": "left",
            "offset_db": _as_stat_float(l_st1, "offset_db", np.nan),
            "target_level_db": _as_stat_float(l_st1, "eff_target_db", np.nan),
            "target_shift_db": _as_stat_float(l_st1, "target_shift_db", np.nan),
            "meas_level_db_window": _as_stat_float(l_st1, "meas_level_db_window", np.nan),
        }
        right = {
            "channel": "right",
            "offset_db": _as_stat_float(r_st1, "offset_db", np.nan),
            "target_level_db": _as_stat_float(r_st1, "eff_target_db", np.nan),
            "target_shift_db": _as_stat_float(r_st1, "target_shift_db", np.nan),
            "meas_level_db_window": _as_stat_float(r_st1, "meas_level_db_window", np.nan),
        }

        def _anchor_key(candidate):
            shift = candidate["target_shift_db"]
            meas = candidate["meas_level_db_window"]
            target = candidate["target_level_db"]
            if np.isfinite(shift):
                return (0, float(shift))
            if np.isfinite(meas):
                return (1, float(meas))
            if np.isfinite(target):
                return (2, float(target))
            return (3, float("inf"))

        candidates = [c for c in (left, right) if _anchor_key(c)[0] < 3]
        if not candidates:
            return None
        return min(candidates, key=_anchor_key)

    lvl_min = float(getattr(cfg, "lvl_min", 200.0) or 200.0)
    lvl_max = float(getattr(cfg, "lvl_max", 3000.0) or 3000.0)

    mode = str(getattr(cfg, "lvl_mode", "Auto") or "Auto")
    if "Manual" in mode:
        win_l = [lvl_min, lvl_max]
        win_r = [lvl_min, lvl_max]
    else:
        win_l = _safe_range((l_st1 or {}).get("smart_scan_range"), lvl_min, lvl_max)
        win_r = _safe_range((r_st1 or {}).get("smart_scan_range"), lvl_min, lvl_max)

    shared_win_from_scan = _shared_window_from_stats(l_st1, r_st1) if "Manual" not in mode else None
    win_shared = list(shared_win_from_scan) if shared_win_from_scan is not None else list(win_l)
    if not (win_shared[1] > win_shared[0]):
        win_shared = list(win_l)
    if not (win_shared[1] > win_shared[0]):
        win_shared = list(win_r)
    if not (win_shared[1] > win_shared[0]):
        win_shared = [lvl_min, lvl_max]

    anchor = _pick_quieter_anchor()
    off_l = float((l_st1 or {}).get("offset_db", 0.0) or 0.0)
    off_r = float((r_st1 or {}).get("offset_db", 0.0) or 0.0)
    if anchor is not None and np.isfinite(float(anchor["offset_db"])):
        off_shared = float(anchor["offset_db"])
    else:
        off_shared = min(float(off_l), float(off_r))
    tgt_l = _as_stat_float(l_st1, "eff_target_db", np.nan)
    tgt_r = _as_stat_float(r_st1, "eff_target_db", np.nan)
    if anchor is not None and np.isfinite(float(anchor["target_level_db"])):
        target_shared = float(anchor["target_level_db"])
    elif np.isfinite(tgt_l) and np.isfinite(tgt_r):
        target_shared = min(float(tgt_l), float(tgt_r))
    elif np.isfinite(tgt_l):
        target_shared = float(tgt_l)
    elif np.isfinite(tgt_r):
        target_shared = float(tgt_r)
    else:
        target_shared = None
    tshift_l = _as_stat_float(l_st1, "target_shift_db", np.nan)
    tshift_r = _as_stat_float(r_st1, "target_shift_db", np.nan)
    if anchor is not None and np.isfinite(float(anchor["target_shift_db"])):
        target_shift_shared = float(anchor["target_shift_db"])
    elif np.isfinite(tshift_l) and np.isfinite(tshift_r):
        target_shift_shared = min(float(tshift_l), float(tshift_r))
    elif np.isfinite(tshift_l):
        target_shift_shared = float(tshift_l)
    elif np.isfinite(tshift_r):
        target_shift_shared = float(tshift_r)
    else:
        target_shift_shared = None

    try:
        strategy_req = str(getattr(cfg, "stereo_link_strategy", "shared") or "shared").strip().lower()
    except (AttributeError, TypeError, ValueError):
        strategy_req = "shared"
    if strategy_req not in ("shared", "hybrid", "auto"):
        strategy_req = "shared"

    tilt_l = _as_stat_float(l_st1, "tilt_slope_db_per_oct", np.nan)
    tilt_r = _as_stat_float(r_st1, "tilt_slope_db_per_oct", np.nan)
    off_diff = abs(float(off_l) - float(off_r))
    tilt_diff = abs(float(tilt_l) - float(tilt_r)) if (np.isfinite(tilt_l) and np.isfinite(tilt_r)) else 0.0
    tilt_abs_max = max(abs(float(tilt_l)) if np.isfinite(tilt_l) else 0.0, abs(float(tilt_r)) if np.isfinite(tilt_r) else 0.0)

    guard_triggered = bool(
        (off_diff > 1.5)
        or (tilt_diff > 0.7)
        or (tilt_abs_max > 1.2)
    )
    strategy_resolved = "hybrid" if (strategy_req == "auto" and guard_triggered) else ("shared" if strategy_req == "auto" else strategy_req)

    shared_auto_gain_db = None
    try:
        margin_db = float(getattr(cfg, "auto_gain_margin_db", getattr(cfg, "global_gain_db", 0.0)) or 0.0)
    except (AttributeError, TypeError, ValueError):
        margin_db = 0.0
    if (not np.isfinite(margin_db)) or (margin_db < 0.0):
        margin_db = 0.0
    try:
        ag_l = float((l_st1 or {}).get("auto_global_gain_db", np.nan))
        ag_r = float((r_st1 or {}).get("auto_global_gain_db", np.nan))
        if np.isfinite(ag_l) and np.isfinite(ag_r):
            # More negative dB value is the safer shared attenuation.
            shared_auto_gain_db = min(ag_l, ag_r)
        else:
            peak_l = float((l_st1 or {}).get("peak_gain_db", 0.0) or 0.0)
            peak_r = float((r_st1 or {}).get("peak_gain_db", 0.0) or 0.0)
            peak_shared = max(0.0, peak_l, peak_r)
            shared_auto_gain_db = -(peak_shared + margin_db)
    except (TypeError, ValueError):
        shared_auto_gain_db = None

    cfg2 = copy.deepcopy(cfg)
    try:
        cfg2.stereo_link = False
        if shared_auto_gain_db is not None and np.isfinite(shared_auto_gain_db):
            cfg2.auto_gain_db_override = float(shared_auto_gain_db)
    except (AttributeError, TypeError, ValueError):
        pass

    if strategy_resolved == "hybrid":
        stereo_ctx_l = StereoLinkContext(
            forced_window_hz=(float(win_l[0]), float(win_l[1])),
            forced_offset_db=float(off_shared),
            shared_target_level_db=(float(target_shared) if target_shared is not None else None),
            shared_target_shift_db=(float(target_shift_shared) if target_shift_shared is not None else None),
        )
        stereo_ctx_r = StereoLinkContext(
            forced_window_hz=(float(win_r[0]), float(win_r[1])),
            forced_offset_db=float(off_shared),
            shared_target_level_db=(float(target_shared) if target_shared is not None else None),
            shared_target_shift_db=(float(target_shift_shared) if target_shift_shared is not None else None),
        )
    else:
        stereo_ctx = StereoLinkContext(
            forced_window_hz=(float(win_shared[0]), float(win_shared[1])),
            forced_offset_db=float(off_shared),
            shared_target_level_db=(float(target_shared) if target_shared is not None else None),
            shared_target_shift_db=(float(target_shift_shared) if target_shift_shared is not None else None),
        )
        stereo_ctx_l = stereo_ctx
        stereo_ctx_r = stereo_ctx

    l_imp2, l_st2 = generate_filter(f_l, m_l, p_l, cfg2, stereo_link_ctx=stereo_ctx_l)
    r_imp2, r_st2 = generate_filter(f_r, m_r, p_r, cfg2, stereo_link_ctx=stereo_ctx_r)

    try:
        if isinstance(l_st2, dict):
            mode_tag = "StereoLinkHybrid" if strategy_resolved == "hybrid" else "StereoLinkShared"
            l_st2["offset_method"] = str(l_st2.get("offset_method", "")) + f" ({mode_tag})"
            l_st2["stereo_link_mode"] = str(strategy_resolved)
            l_st2["stereo_link_requested_mode"] = str(strategy_req)
            l_st2["stereo_link_guard_triggered"] = bool(strategy_req == "auto" and guard_triggered)
            l_st2["stereo_link_guard_off_diff_db"] = float(off_diff)
            l_st2["stereo_link_guard_tilt_diff_db_per_oct"] = float(tilt_diff)
            l_st2["stereo_link_guard_tilt_abs_max_db_per_oct"] = float(tilt_abs_max)
            l_st2["stereo_link_shared_offset_db"] = float(off_shared)
            if anchor is not None:
                l_st2["stereo_link_level_anchor_channel"] = str(anchor["channel"])
            if target_shared is not None and np.isfinite(float(target_shared)):
                l_st2["stereo_link_shared_target_level_db"] = float(target_shared)
            if target_shift_shared is not None and np.isfinite(float(target_shift_shared)):
                l_st2["stereo_link_shared_target_shift_db"] = float(target_shift_shared)
            l_st2["stereo_link_window_used"] = [float(win_l[0]), float(win_l[1])] if strategy_resolved == "hybrid" else [float(win_shared[0]), float(win_shared[1])]
            if strategy_resolved != "hybrid":
                l_st2["stereo_link_shared_window"] = [float(win_shared[0]), float(win_shared[1])]
            if shared_auto_gain_db is not None and np.isfinite(shared_auto_gain_db):
                l_st2["stereo_link_shared_auto_gain_db"] = float(shared_auto_gain_db)
        if isinstance(r_st2, dict):
            mode_tag = "StereoLinkHybrid" if strategy_resolved == "hybrid" else "StereoLinkShared"
            r_st2["offset_method"] = str(r_st2.get("offset_method", "")) + f" ({mode_tag})"
            r_st2["stereo_link_mode"] = str(strategy_resolved)
            r_st2["stereo_link_requested_mode"] = str(strategy_req)
            r_st2["stereo_link_guard_triggered"] = bool(strategy_req == "auto" and guard_triggered)
            r_st2["stereo_link_guard_off_diff_db"] = float(off_diff)
            r_st2["stereo_link_guard_tilt_diff_db_per_oct"] = float(tilt_diff)
            r_st2["stereo_link_guard_tilt_abs_max_db_per_oct"] = float(tilt_abs_max)
            r_st2["stereo_link_shared_offset_db"] = float(off_shared)
            if anchor is not None:
                r_st2["stereo_link_level_anchor_channel"] = str(anchor["channel"])
            if target_shared is not None and np.isfinite(float(target_shared)):
                r_st2["stereo_link_shared_target_level_db"] = float(target_shared)
            if target_shift_shared is not None and np.isfinite(float(target_shift_shared)):
                r_st2["stereo_link_shared_target_shift_db"] = float(target_shift_shared)
            r_st2["stereo_link_window_used"] = [float(win_r[0]), float(win_r[1])] if strategy_resolved == "hybrid" else [float(win_shared[0]), float(win_shared[1])]
            if strategy_resolved != "hybrid":
                r_st2["stereo_link_shared_window"] = [float(win_shared[0]), float(win_shared[1])]
            if shared_auto_gain_db is not None and np.isfinite(shared_auto_gain_db):
                r_st2["stereo_link_shared_auto_gain_db"] = float(shared_auto_gain_db)
    except (TypeError, ValueError):
        pass

    return l_imp2, l_st2, r_imp2, r_st2
