from camillafir.ui import ng_controls as ctrl


class _Event:
    def __init__(self, value) -> None:
        self.value = value


class _DummyControl:
    def __init__(self, value=None) -> None:
        self.value = value
        self._handler = None

    def set_value(self, value) -> None:
        self.value = value
        if self._handler is not None:
            self._handler(_Event(value))

    def on_value_change(self, handler) -> None:
        self._handler = handler


class _DummyContainer:
    def __init__(self) -> None:
        self.visible = None

    def set_visibility(self, visible: bool) -> None:
        self.visible = bool(visible)


def test_set_visibility_supports_registered_containers():
    ctrl.reset()
    scope = _DummyContainer()
    ctrl.register_container("lvl_manual_scope", scope)

    ctrl.set_visibility("lvl_manual_scope", True)

    assert scope.visible is True


def test_set_value_can_suppress_registered_change_callback():
    ctrl.reset()
    changes = []
    element = ctrl.register("lvl_manual_db", _DummyControl(0.0))

    ctrl.on_change("lvl_manual_db", lambda value: changes.append(value))

    ctrl.set_value("lvl_manual_db", 1.5, emit=False)

    assert element.value == 1.5
    assert changes == []

    ctrl.set_value("lvl_manual_db", 2.0)

    assert changes == [2.0]


def test_value_holder_emits_change_callbacks() -> None:
    holder = ctrl._ValueHolder("before")
    seen: list[str] = []

    holder.on_value_change(lambda e: seen.append(str(e.value)))
    holder.set_value("after")

    assert holder.value == "after"
    assert seen == ["after"]


def test_ctrl_set_value_uses_value_holder_callbacks() -> None:
    ctrl.reset()
    holder = ctrl._ValueHolder("before")
    seen: list[str] = []
    holder.on_value_change(lambda e: seen.append(str(e.value)))
    ctrl.register("logical_path", holder)

    ctrl.set_value("logical_path", "after")

    assert ctrl.value("logical_path") == "after"
    assert seen == ["after"]
