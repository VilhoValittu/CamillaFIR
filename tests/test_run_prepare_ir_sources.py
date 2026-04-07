from __future__ import annotations

import numpy as np

from camillafir.application.run_request import RunRequest
from camillafir.ui.ng_bridge import ProcessRunCallbacks
from camillafir.workflow.run_prepare import _prepare_ui_and_measurements


class _DummyUiBridge:
    def ensure_progress_bar(self) -> None:
        return None

    def toast_health_gate_result(self, hr, mode) -> bool:
        return False


class _DummySupport:
    def __init__(self) -> None:
        self.ui_bridge = _DummyUiBridge()


def _callbacks() -> ProcessRunCallbacks:
    return ProcessRunCallbacks(
        status=lambda msg: None,
        set_auto_selected_bar=lambda msg="": None,
    )


def _measurement_tuple():
    f = np.asarray([20.0, 100.0], dtype=float)
    m = np.asarray([0.0, -3.0], dtype=float)
    p = np.asarray([0.0, 0.0], dtype=float)
    return f, m, p, f, m, p


def _patch_common(monkeypatch) -> None:
    monkeypatch.setattr("camillafir.workflow.run_prepare.compute_health", lambda data, mode: object())
    monkeypatch.setattr("camillafir.workflow.run_prepare.save_config", lambda data: None)


def test_prepare_ui_reads_lr_raw_ir_from_regular_slots(monkeypatch) -> None:
    _patch_common(monkeypatch)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "camillafir.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: _measurement_tuple(),
    )
    monkeypatch.setattr(
        "camillafir.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("unexpected bass integration load")),
    )

    def _fake_load_raw_irs_lr(data, *, logger=None, **kwargs):
        calls["lr_kwargs"] = dict(kwargs)
        return np.asarray([1.0], dtype=float), 48000, np.asarray([2.0], dtype=float), 48000

    monkeypatch.setattr("camillafir.workflow.run_prepare.load_raw_irs_lr", _fake_load_raw_irs_lr)
    monkeypatch.setattr(
        "camillafir.workflow.run_prepare.load_raw_ir_sub",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("sub IR should stay inactive")),
    )

    ctx = _prepare_ui_and_measurements(
        request=RunRequest(raw_ui_data={"mode": "AUTO", "bass_integration_enable": False}),
        callbacks=_callbacks(),
        support=_DummySupport(),
    )

    assert ctx is not None
    assert calls["lr_kwargs"] == {}
    assert np.array_equal(ctx["raw_ir_l"], np.asarray([1.0], dtype=float))
    assert np.array_equal(ctx["raw_ir_r"], np.asarray([2.0], dtype=float))
    assert ctx["raw_ir_sub"] is None
    assert ctx["raw_ir_fs_sub"] == 0


def test_prepare_ui_reads_bass_integration_raw_ir_from_main_and_sub_slots(monkeypatch) -> None:
    _patch_common(monkeypatch)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "camillafir.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("unexpected regular measurement load")),
    )
    monkeypatch.setattr(
        "camillafir.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: ("bundle", *_measurement_tuple()),
    )

    def _fake_load_raw_irs_lr(data, *, logger=None, **kwargs):
        calls["lr_kwargs"] = dict(kwargs)
        return np.asarray([3.0], dtype=float), 48000, np.asarray([4.0], dtype=float), 48000

    def _fake_load_raw_ir_sub(data, *, logger=None):
        calls["sub_called"] = True
        return np.asarray([5.0], dtype=float), 48000

    monkeypatch.setattr("camillafir.workflow.run_prepare.load_raw_irs_lr", _fake_load_raw_irs_lr)
    monkeypatch.setattr("camillafir.workflow.run_prepare.load_raw_ir_sub", _fake_load_raw_ir_sub)

    ctx = _prepare_ui_and_measurements(
        request=RunRequest(raw_ui_data={"mode": "AUTO", "bass_integration_enable": True}),
        callbacks=_callbacks(),
        support=_DummySupport(),
    )

    assert ctx is not None
    assert calls["lr_kwargs"] == {
        "file_key_l": "file_l_main",
        "path_key_l": "local_path_l_main",
        "file_key_r": "file_r_main",
        "path_key_r": "local_path_r_main",
    }
    assert calls["sub_called"] is True
    assert ctx["bass_integration_bundle"] == "bundle"
    assert np.array_equal(ctx["raw_ir_l"], np.asarray([3.0], dtype=float))
    assert np.array_equal(ctx["raw_ir_r"], np.asarray([4.0], dtype=float))
    assert np.array_equal(ctx["raw_ir_sub"], np.asarray([5.0], dtype=float))
    assert ctx["raw_ir_fs_sub"] == 48000
