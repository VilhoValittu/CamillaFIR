from camillafir.config.camillafir_pipeline import collect_ui_data


def test_collect_ui_data_normalizes_auto_target_mode_to_auto_by_default():
    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_target_mode": "invalid-value",
    }
    data = collect_ui_data(pin)
    assert str(data.get("auto_target_mode")) == "auto"


def test_collect_ui_data_accepts_selected_aliases_for_auto_target_mode():
    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_target_mode": "manual",
    }
    data = collect_ui_data(pin)
    assert str(data.get("auto_target_mode")) == "selected"
