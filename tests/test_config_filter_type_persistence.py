import json

import camillafir.config.camillafir_config as camillafir_config


def test_save_config_normalizes_asymmetric_filter_type(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(camillafir_config, "CONFIG_FILE", str(cfg_path))

    camillafir_config.save_config({"filter_type": "asym", "mode": "AUTO"})

    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["filter_type"] == "Asymmetric"


def test_load_config_normalizes_legacy_asymmetric_filter_type(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(
        json.dumps({"filter_type": "asym", "mode": "AUTO"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(camillafir_config, "CONFIG_FILE", str(cfg_path))

    loaded = camillafir_config.load_config()
    assert loaded["filter_type"] == "Asymmetric"
