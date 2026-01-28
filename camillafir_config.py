# camillafir_config.py
import json
import os

CONFIG_FILE = "config.json"


def load_config() -> dict:
    default_conf = {
        "fmt": "WAV",
        "layout": "Mono",
        "fs": 44100,
        "taps": 65536,
        "mode": "BASIC",
        "filter_type": "Linear Phase",
        "gain": 0.0,
        "hc_mode": "Harman6",
        "mag_correct": True,
        "smoothing_type": "smooth_psy",
        "fdw_cycles": 10.0,
        "mag_c_min": 10.0,
        "mag_c_max": 200.0,
        "max_boost": 5.0,
        "lvl_mode": "Auto",
        "lvl_algo": "Median",
        "lvl_manual_db": 75.0,
        "lvl_min": 300.0,
        "lvl_max": 3000.0,
        "normalize_opt": False,
        "align_opt": True,
        "multi_rate_opt": False,
        "reg_strength": 30.0,
        "stereo_link": True,
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
        "mixed_freq": 300.0,
        "phase_limit": 600.0,
        "phase_safe_2058": False,
        "ir_window": 500.0,
        "ir_window_left": 50.0,
        "enable_tdc": True,
        "tdc_strength": 50.0,
        "enable_afdw": True,
        "max_cut_db": 30.0,
        "max_slope_db_per_oct": 24.0,
        "max_slope_boost_db_per_oct": 0.0,
        "max_slope_cut_db_per_oct": 0.0,
        "df_smoothing": False,
        "comparison_mode": False,
        "tdc_max_reduction_db": 9.0,
        "tdc_slope_db_per_oct": 6.0,
        "bass_first_ai": True,
        "bass_first_mode_max_hz": 200.0,
    }

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)

            # Legacy: PyWebIO checkbox pins saved as []/[True]
            for k in [
                "mag_correct",
                "normalize_opt",
                "align_opt",
                "multi_rate_opt",
                "stereo_link",
                "exc_prot",
                "hpf_enable",
                "df_smoothing",
                "comparison_mode",
                "phase_safe_2058",
            ]:
                if k in saved and isinstance(saved[k], list):
                    saved[k] = bool(saved[k])

            default_conf.update(saved)
        except Exception:
            pass

    return default_conf


def save_config(data: dict) -> None:
    try:
        clean_data = {k: v for k, v in (data or {}).items() if not str(k).startswith("file_")}
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=4)
    except Exception:
        pass
