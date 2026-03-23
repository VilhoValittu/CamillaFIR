import json
import os

CONFIG_FILE = "config.json"


def _normalize_filter_type(value) -> str:
    """Normalize persisted filter type names to the UI/program canonical labels."""
    try:
        ft = str(value or "").strip()
    except Exception:
        ft = ""
    ft_l = ft.lower()
    if "asym" in ft_l:
        return "Asymmetric"
    if "mixed" in ft_l:
        return "Mixed"
    if "minimum" in ft_l or "minphase" in ft_l or ft_l == "min":
        return "Minimum"
    if "linear" in ft_l:
        return "Linear"
    return "Asymmetric"


def load_config() -> dict:
    default_conf = {
        "fmt": "WAV",
        "layout": "Mono",
        "fs": 44100,
        "taps": 65536,
        "mode": "AUTO",
        "auto_goal": "balanced",
        "auto_target_mode": "auto",
        "auto_mode_workers": 0,
        "auto_mode_optuna_multivariate": True,
        "auto_mode_optuna_group": False,
        "auto_mode_optuna_constant_liar": True,
        "auto_mode_optuna_persistent_study": True,
        "auto_mode_optuna_avoid_duplicates": True,
        "filter_type": "Asymmetric",
        "gain": 0.0,
        "hc_mode": "Harman6",
        "mag_correct": True,
        "unsafe_raw_dsp": False,
        "camillafir_automatic_mode": True,
        "plot_smoothing_level": "Psychoacoustic",
        "filter_smooth": 12,
        "bass_smooth_adaptive": True,
        "bass_smooth_hz": 200.0,
        "bass_smooth_sigma_scale": 1.4,
        "bass_smooth_conf_floor": 0.3,
        "bass_adaptive_isolation_mode": True,
        "mid_refit_enable": True,
        "mid_refit_hz_lo": 200.0,
        "mid_refit_hz_hi": 2000.0,
        "mid_refit_k": 0.45,
        "mid_refit_smooth_oct": 0.60,
        "mid_refit_conf_min_avg": 0.20,
        "bass_boost_cap_enable": True,
        "bass_boost_cap_hz": 200.0,
        "bass_boost_cap_extra_db": 2.0,
        "bass_boost_cap_conf_min": 0.55,
        "bass_boost_post_restore_enable": True,
        "bass_boost_post_restore_strength": 0.60,
        "fdw_cycles": 10.0,
        "mag_c_min": 10.0,
        "mag_c_max": 200.0,
        "max_boost": 5.0,
        "lvl_mode": "Auto",
        "lvl_algo": "Median",
        "lvl_manual_db": 0.0,
        "lvl_min": 300.0,
        "lvl_max": 3000.0,
        "normalize_opt": False,
        "align_opt": True,
        "multi_rate_opt": False,
        "reg_strength": 30.0,
        "stereo_link": True,
        "stereo_link_strategy": "auto",
        "exc_prot": True,
        "exc_freq": 20.0,
        "low_bass_cut_hz": 40.0,
        "hpf_enable": False,
        "hpf_freq": 20.0,
        "hpf_slope": 24,
        "local_path_l": "",
        "local_path_r": "",
        "xo1_f": None,
        "xo1_s": 12,
        "xo2_f": None,
        "xo2_s": 12,
        "xo3_f": None,
        "xo3_s": 12,
        "xo4_f": None,
        "xo4_s": 12,
        "xo5_f": None,
        "xo5_s": 12,
        "mixed_freq": 180.0,
        "phase_limit": 400.0,
        "phase_safe_2058": False,
        "excess_phase_strength": 0.9,
        "low_freq_full_correction_hz": 140.0,
        "high_freq_no_correction_hz": 900.0,
        "phase_boundary_smooth_sigma_bins": 1.2,
        "phase_tail_monotonic_enable": True,
        "phase_tail_start_ratio": 0.72,
        "phase_tail_abs_smooth_sigma_bins": 2.5,
        "phase_tail_cosine_strength": 0.85,
        "linear_phase_blend_start_ratio": 0.65,
        "enable_ir_pre_energy_guard": True,
        "pre_energy_ratio_max": 0.25,
        "pre_energy_guard_strength": 0.8,
        "max_pre_ringing_db": -35.0,
        "max_excess_delay_ms": 2.5,
        "gd_grad_limit_ms_per_oct": 20.0,
        "ir_anchor_mode": "min_causal",
        "min_causal_ms": 80.0,
        "auto_asym_left_ratio": 0.35,
        "auto_asym_left_max_ms": 25.0,
        "ir_window_right": 500.0,
        "ir_window_left": 85.0,
        "ir_export_window_mode": "auto",
        "ir_export_window_shape": "hann",
        "ir_export_tukey_alpha": 0.25,
        "enable_tdc": True,
        "tdc_strength": 50.0,
        "enable_afdw": True,
        "max_cut_db": 30.0,
        "max_slope_db_per_oct": 24.0,
        "max_slope_boost_db_per_oct": 0.0,
        "max_slope_cut_db_per_oct": 0.0,
        "df_smoothing": False,
        "comparison_mode": True,
        "tdc_max_reduction_db": 9.0,
        "tdc_slope_db_per_oct": 6.0,
        "bass_first_ai": True,
        "bass_first_mode_max_hz": 200.0,
        "conf_pull_bass_boost_floor_hz": 200.0,
        "conf_pull_bass_boost_floor_min": 0.45,
        "conf_pull_bass_boost_restore": 0.55,
        "debug_stage_stats": True,
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)


            for k in [
                "mag_correct",
                "normalize_opt",
                "align_opt",
                "multi_rate_opt",
                "stereo_link",
                "exc_prot",
                "hpf_enable",
                "df_smoothing",
                "bass_smooth_adaptive",
                "bass_adaptive_isolation_mode",
                "mid_refit_enable",
                "bass_boost_cap_enable",
                "bass_boost_post_restore_enable",
                "comparison_mode",
                "phase_safe_2058",
                "enable_ir_pre_energy_guard",
                "phase_tail_monotonic_enable",
                "unsafe_raw_dsp",
                "camillafir_automatic_mode",
            ]:
                if k in saved and isinstance(saved[k], list):
                    saved[k] = bool(saved[k])

            try:
                if "lvl_manual_db" in saved:
                    _v = float(saved.get("lvl_manual_db"))
                    if 40.0 <= _v <= 110.0:
                        saved["lvl_manual_db"] = float(_v - 75.0)
            except Exception:
                pass

            try:
                saved["filter_type"] = _normalize_filter_type(
                    saved.get("filter_type", default_conf.get("filter_type"))
                )
            except Exception:
                saved["filter_type"] = str(default_conf.get("filter_type", "Asymmetric"))

            default_conf.update(saved)
        except Exception:
            pass

    try:
        mode_u = str(default_conf.get("mode", "AUTO") or "AUTO").strip().upper()
    except Exception:
        mode_u = "AUTO"

    try:
        legacy_auto = bool(default_conf.get("camillafir_automatic_mode", False))
    except Exception:
        legacy_auto = False

    if legacy_auto:
        mode_u = "AUTO"
    if mode_u not in ("AUTO", "BASIC", "ADVANCED"):
        mode_u = "AUTO"

    default_conf["mode"] = mode_u
    default_conf["camillafir_automatic_mode"] = bool(mode_u == "AUTO")

    return default_conf


def save_config(data: dict) -> None:
    try:
        clean_data = {
            k: v for k, v in (data or {}).items()
            if (
                not str(k).startswith("file_")
                and str(k) != "auto_mode_compat_version"
                and v is not None
            )
        }
        clean_data["filter_type"] = _normalize_filter_type(
            clean_data.get("filter_type", "Asymmetric")
        )
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=4)
    except Exception:
        pass
