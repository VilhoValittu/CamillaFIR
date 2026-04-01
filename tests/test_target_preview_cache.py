import os

import numpy as np

from camillafir.ui import target_preview_cache


def _curve(offset: float = 0.0):
    freqs = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0], dtype=float)
    mags = np.array([offset + step for step in range(freqs.size)], dtype=float)
    return freqs, mags, None


def test_load_upload_measurement_curve_reuses_cached_parse(monkeypatch):
    monkeypatch.setattr(target_preview_cache, "_CURVE_CACHE", {})
    calls = {"n": 0}

    def _fake_parse_txt_bytes(content):
        calls["n"] += 1
        return _curve()

    monkeypatch.setattr(target_preview_cache.measurements_txt, "parse_measurements_from_bytes", _fake_parse_txt_bytes)
    upload = {"filename": "left.txt", "content": b"same-content"}

    first = target_preview_cache.load_upload_measurement_curve(
        upload,
        pre_ms=85.0,
        post_ms=500.0,
        smoothing_level=0,
    )
    second = target_preview_cache.load_upload_measurement_curve(
        upload,
        pre_ms=85.0,
        post_ms=500.0,
        smoothing_level=0,
    )

    assert calls["n"] == 1
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_load_path_measurement_curve_reparses_when_file_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(target_preview_cache, "_CURVE_CACHE", {})
    calls = {"n": 0}
    path = tmp_path / "left.txt"
    path.write_text("baseline", encoding="utf-8")

    def _fake_parse_txt_path(raw_path, logger=None):
        calls["n"] += 1
        return _curve(offset=float(calls["n"]))

    monkeypatch.setattr(target_preview_cache.measurements_txt, "parse_measurements_from_path", _fake_parse_txt_path)

    first = target_preview_cache.load_path_measurement_curve(
        str(path),
        pre_ms=85.0,
        post_ms=500.0,
        smoothing_level=0,
    )
    second = target_preview_cache.load_path_measurement_curve(
        str(path),
        pre_ms=85.0,
        post_ms=500.0,
        smoothing_level=0,
    )

    path.write_text("baseline-updated", encoding="utf-8")
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))

    third = target_preview_cache.load_path_measurement_curve(
        str(path),
        pre_ms=85.0,
        post_ms=500.0,
        smoothing_level=0,
    )

    assert calls["n"] == 2
    assert np.array_equal(first[0], second[0])
    assert not np.array_equal(first[1], third[1])
