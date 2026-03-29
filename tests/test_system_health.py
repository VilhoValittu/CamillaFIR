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


def _measurement_issue(hr):
    title = t("health_measurements")
    return next(i for i in hr.issues if i.title == title)


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


def test_measurements_are_warned_when_local_files_do_not_exist(tmp_path):
    data = _base_data()
    data["local_path_l"] = str(tmp_path / "missing_l.txt")
    data["local_path_r"] = str(tmp_path / "missing_r.txt")

    hr = compute_health(data, mode="BASIC")

    assert _measurement_issue(hr).level == "warn"


def test_measurements_are_ok_when_local_files_exist(tmp_path):
    left = tmp_path / "L.txt"
    right = tmp_path / "R.txt"
    left.write_text("20 -3\n100 -1\n1000 0\n", encoding="utf-8")
    right.write_text("20 -3\n100 -1\n1000 0\n", encoding="utf-8")

    data = _base_data()
    data["local_path_l"] = str(left)
    data["local_path_r"] = str(right)

    hr = compute_health(data, mode="BASIC")

    assert _measurement_issue(hr).level == "ok"


def test_measurements_are_warned_when_uploaded_files_have_no_content():
    data = _base_data()
    data["file_l"] = {"filename": "L.txt"}
    data["file_r"] = {"filename": "R.txt"}

    hr = compute_health(data, mode="BASIC")

    assert _measurement_issue(hr).level == "warn"
