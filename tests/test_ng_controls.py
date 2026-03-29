from camillafir.ui import ng_controls as ctrl


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
