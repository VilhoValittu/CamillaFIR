from camillafir.config.camillafir_config import load_config
from camillafir.config.camillafir_pipeline import build_filter_config
from camillafir.config.models import FilterConfig


def test_bass_adaptive_default_stays_enabled_when_pin_none():
    class _Pin:
        def __init__(self):
            self._d = {"bass_smooth_adaptive": None}

        def get(self, key, default=None):
            return self._d.get(key, default)

        def __getitem__(self, key):
            if key in self._d:
                return self._d[key]
            raise KeyError(key)

    data = load_config()
    data.update(
        {
            "filter_type": "Linear Phase",
            "mixed_freq": 180.0,
            "mag_c_min": 10.0,
            "mag_c_max": 200.0,
            "max_boost": 5.0,
            "phase_limit": 600.0,
            "mag_correct": True,
            "reg_strength": 30.0,
            "normalize_opt": False,
            "exc_prot": False,
            "exc_freq": 20.0,
            "low_bass_cut_hz": 40.0,
            "ir_window_right": 500.0,
            "ir_window_left": 85.0,
            "lvl_manual_db": 0.0,
            "lvl_min": 200.0,
            "lvl_max": 3000.0,
            "lvl_algo": "Median",
            "trans_width": 100.0,
            "bass_smooth_adaptive": None,
        }
    )

    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=44100,
        taps_v=65536,
        data=data,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin=_Pin(),
    )

    assert bool(getattr(cfg, "bass_smooth_adaptive", False)) is True
