import sys
import types

from camillafir.ui import ng_tab_target


class _DummyPreviewContainer:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class _DummyPlot:
    def __init__(self, figure) -> None:
        self.figure = figure
        self.update_figure_calls = []
        self.events = []
        self.class_names = []

    def classes(self, class_name: str):
        self.class_names.append(class_name)
        return self

    def on(self, event_name: str, handler) -> None:
        self.events.append((event_name, handler))

    def update_figure(self, figure) -> None:
        self.figure = figure
        self.update_figure_calls.append(figure)


class _DummyUI:
    def __init__(self) -> None:
        self.created_plots = []

    def plotly(self, figure):
        plot = _DummyPlot(figure)
        self.created_plots.append(plot)
        return plot


def test_refresh_target_preview_reuses_existing_plot(monkeypatch):
    dummy_ui = _DummyUI()
    preview_col = _DummyPreviewContainer()
    figures = iter(
        [
            ({"data": [{"name": "first"}], "layout": {}, "config": {}}, [(10.0, 0.0)], []),
            ({"data": [{"name": "second"}], "layout": {}, "config": {}}, [(20.0, 1.0)], []),
        ]
    )

    monkeypatch.setitem(sys.modules, "nicegui", types.SimpleNamespace(ui=dummy_ui))
    monkeypatch.setattr(ng_tab_target.ctrl, "get_container", lambda name: preview_col)
    monkeypatch.setattr(ng_tab_target, "_build_target_preview_fig", lambda: next(figures))
    monkeypatch.setattr(ng_tab_target, "_TARGET_PREVIEW_PLOT", None)

    ng_tab_target.refresh_target_preview()
    ng_tab_target.refresh_target_preview()

    assert preview_col.clear_calls == 1
    assert len(dummy_ui.created_plots) == 1
    assert dummy_ui.created_plots[0].update_figure_calls == [
        {"data": [{"name": "second"}], "layout": {}, "config": {}}
    ]


def test_refresh_target_preview_clears_existing_plot_when_figure_is_missing(monkeypatch):
    preview_col = _DummyPreviewContainer()
    existing_plot = _DummyPlot({"data": [], "layout": {}, "config": {}})

    monkeypatch.setattr(ng_tab_target.ctrl, "get_container", lambda name: preview_col)
    monkeypatch.setattr(ng_tab_target, "_build_target_preview_fig", lambda: (None, [], []))
    monkeypatch.setattr(ng_tab_target, "_TARGET_PREVIEW_PLOT", existing_plot)

    ng_tab_target.refresh_target_preview()

    assert preview_col.clear_calls == 1
    assert ng_tab_target._TARGET_PREVIEW_PLOT is None
