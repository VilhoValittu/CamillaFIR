from camillafir.ui.ng_tab_files import (
    _build_upload_payload,
    _describe_local_path,
    _format_upload_size,
    _guess_upload_format,
    _normalize_local_path_value,
    _normalize_layout_value,
)


def test_normalize_layout_value_accepts_legacy_and_stable_values():
    assert _normalize_layout_value("Mono") == "mono"
    assert _normalize_layout_value("mono") == "mono"
    assert _normalize_layout_value("Stereo") == "stereo"
    assert _normalize_layout_value("stereo") == "stereo"


def test_guess_upload_format_detects_txt_and_wav():
    assert _guess_upload_format({"filename": "left.txt", "content": b"20 0\n40 0\n"}) == "TXT"
    assert _guess_upload_format({"filename": "left.wav", "content": b"RIFF...."}) == "WAV"
    assert _guess_upload_format({"filename": "left.bin", "content": b"??"}) == "Unknown"


def test_normalize_local_path_value_strips_quotes():
    assert _normalize_local_path_value('"C:\\Temp\\left.txt"') == "C:\\Temp\\left.txt"


def test_describe_local_path_reports_existing_file(tmp_path):
    path = tmp_path / "left.txt"
    path.write_text("test", encoding="utf-8")

    info = _describe_local_path(str(path))

    assert info["entered"] is True
    assert info["exists"] is True
    assert info["filename"] == "left.txt"
    assert info["format"] == "TXT"
    assert info["size_bytes"] == 4


def test_describe_local_path_reports_missing_file():
    info = _describe_local_path(r"C:\missing\left.wav")

    assert info["entered"] is True
    assert info["exists"] is False
    assert info["format"] == "WAV"


def test_build_upload_payload_includes_digest_and_size():
    payload = _build_upload_payload(filename="left.txt", content=b"hello", mime_type="text/plain")

    assert payload["filename"] == "left.txt"
    assert payload["mime_type"] == "text/plain"
    assert payload["size_bytes"] == 5
    assert len(str(payload["content_sha256"])) == 64


def test_format_upload_size_uses_human_readable_units():
    assert _format_upload_size(1024) == "1.0 KB"
    assert _format_upload_size(2 * 1024 * 1024) == "2.00 MB"
