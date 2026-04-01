import numpy as np

from camillafir.io import measurements_txt
from camillafir.ui import ng_tab_target
from camillafir.ui.plot_prediction import generate_prediction_plot


def test_target_preview_uses_light_theme(monkeypatch):
    values = {
        "hc_mode": "Flat",
        "lvl_min": 500.0,
        "lvl_max": 2000.0,
        "mag_c_min": 20.0,
        "mag_c_max": 200.0,
        "auto_goal": "balanced",
        "mode": "ADVANCED",
        "lvl_mode": "Manual",
        "lvl_manual_db": 1.5,
        "manual_target_tilt_db_per_oct": 0.0,
        "ir_window_left": 85.0,
        "ir_window_right": 500.0,
        "filter_smooth": 0,
    }

    monkeypatch.setattr(
        ng_tab_target.ctrl,
        "value",
        lambda name, default=None: values.get(name, default),
    )

    fig_dict, _drag_points, _tilt_points = ng_tab_target._build_target_preview_fig()

    assert fig_dict is not None
    layout = fig_dict["layout"]
    assert layout["paper_bgcolor"] == "#ffffff"
    assert layout["plot_bgcolor"] == "#ffffff"
    assert layout["font"]["color"] == "#1f2937"
    assert fig_dict["data"][1]["line"]["color"] == "rgba(15,23,42,0.35)"


def test_target_preview_speaker_avg_is_red(monkeypatch):
    values = {
        "hc_mode": "Flat",
        "lvl_min": 500.0,
        "lvl_max": 2000.0,
        "mag_c_min": 20.0,
        "mag_c_max": 200.0,
        "auto_goal": "balanced",
        "mode": "ADVANCED",
        "lvl_mode": "Auto",
        "ir_window_left": 85.0,
        "ir_window_right": 500.0,
        "filter_smooth": 0,
        "local_path_l": "left.txt",
        "local_path_r": "right.txt",
    }

    monkeypatch.setattr(
        ng_tab_target.ctrl,
        "value",
        lambda name, default=None: values.get(name, default),
    )

    def _fake_parse_txt_path(path, logger=None):
        freqs = np.array([20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0], dtype=float)
        if "left" in str(path):
            mags = np.array([0.0, 0.8, 1.1, 0.7, 0.2, -0.3, -0.8, -1.1, -1.5, -1.8], dtype=float)
        else:
            mags = np.array([0.5, 0.9, 0.6, 0.1, -0.4, -0.8, -1.2, -1.6, -1.9, -2.1], dtype=float)
        return freqs, mags, None

    monkeypatch.setattr(measurements_txt, "parse_measurements_from_path", _fake_parse_txt_path)

    fig_dict, _drag_points, _tilt_points = ng_tab_target._build_target_preview_fig()

    assert fig_dict is not None
    avg_trace = next(trace for trace in fig_dict["data"] if trace.get("name") == "Speaker avg")
    assert avg_trace["line"]["color"] == "#dc2626"


def test_prediction_plot_uses_light_theme():
    freqs = np.array([20.0, 100.0, 1000.0, 10000.0, 20000.0], dtype=float)
    mags = np.zeros_like(freqs)
    phases = np.zeros_like(freqs)
    filt_ir = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    _html, fig = generate_prediction_plot(
        freqs,
        mags,
        phases,
        filt_ir,
        48000,
        "Test",
        create_full_html=False,
        return_fig=True,
    )

    assert fig is not None
    assert fig.layout.paper_bgcolor == "#ffffff"
    assert fig.layout.plot_bgcolor == "#ffffff"
    assert fig.layout.font.color == "#1f2937"
