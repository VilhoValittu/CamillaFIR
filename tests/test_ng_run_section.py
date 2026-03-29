import sys
import types

from camillafir.resources.i8n.camillafir_i18n import t
from camillafir.ui import ng_run_section, ui_state


class _DummyButton:
    def __init__(self) -> None:
        self.disable_calls = 0
        self.enable_calls = 0

    def disable(self) -> None:
        self.disable_calls += 1

    def enable(self) -> None:
        self.enable_calls += 1


class _DummyContainer:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1


class _DummyProgress:
    def __init__(self) -> None:
        self.value = None
        self.text_color = None
        self.visible = None

    def set_value(self, value) -> None:
        self.value = value

    def set_text_color(self, color) -> None:
        self.text_color = color

    def set_visibility(self, visible: bool) -> None:
        self.visible = bool(visible)


class _ImmediateThread:
    def __init__(self, *args, target=None, daemon=None, **kwargs) -> None:
        self._target = target
        self.daemon = daemon

    def start(self) -> None:
        if callable(self._target):
            self._target()


def test_handle_start_clears_previous_results_and_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "nicegui", types.SimpleNamespace(ui=object()))
    monkeypatch.setattr(ng_run_section.threading, "Thread", _ImmediateThread)

    container = _DummyContainer()
    progress = _DummyProgress()
    button = _DummyButton()
    run_clock = {"started_at": None, "active": False, "elapsed_s": None}
    run_calls: list[str] = []

    monkeypatch.setattr(ng_run_section, "_results_container_ref", container)
    monkeypatch.setattr(ng_run_section, "_progress_ref", progress)

    ui_state.set_last_run_info({"score": 97.5, "match": 88.0, "conf": 77.0})
    ui_state.update_status("CamillaFIR automatic mode: target shortlist (old)")
    ui_state.update_status("Done previously")
    ui_state.update_status_notices(summary_text="old summary", info_text="old info")
    ui_state.update_auto_selected_bar("old auto")

    def _on_start_click() -> None:
        run_calls.append("started")

    ng_run_section._handle_start(_on_start_click, button, run_clock)

    snap = ui_state.get_status_snapshot()

    assert container.clear_calls == 1
    assert ui_state.get_last_run_info() == {}
    assert ui_state.get_status_base_message() == t("stat_reading")
    assert snap["status_summary_text"] == ""
    assert snap["status_info_text"] == ""
    assert snap["auto_selected_bar_text"] == ""
    assert snap["auto_status_detail_body"] == ""
    assert progress.value == 0.0
    assert progress.text_color == "primary"
    assert progress.visible is True
    assert button.disable_calls == 1
    assert button.enable_calls == 1
    assert run_calls == ["started"]
    assert run_clock["started_at"] is not None
    assert run_clock["active"] is False
    assert run_clock["elapsed_s"] is not None
