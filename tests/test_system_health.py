from camillafir.resources.i8n.camillafir_i18n import t
from camillafir.ui.system_health import compute_health


def _base_data() -> dict:
    return {
        "hc_mode": "Flat",
        "mag_correct": True,
        "mag_c_min": 20.0,
        "mag_c_max": 250.0,
        "fs": 44100,
        "taps": 131072,
        "filter_type": "Linear",
        "exc_prot": True,
        "max_boost": 5.0,
    }


def _has_taps_high_warning(hr) -> bool:
    title = t("health_taps_count_high")
    return any(i.level == "warn" and i.title == title for i in hr.issues)


def test_taps_warning_is_suppressed_when_left_window_under_120ms():
    data = _base_data()
    data["ir_window_left"] = 100.0
    hr = compute_health(data, mode="BASIC")
    assert not _has_taps_high_warning(hr)


def test_taps_warning_remains_when_left_window_is_120ms_or_more():
    data = _base_data()
    data["ir_window_left"] = 120.0
    hr = compute_health(data, mode="BASIC")
    assert _has_taps_high_warning(hr)
