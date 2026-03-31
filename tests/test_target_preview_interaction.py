from camillafir.ui import ng_controls as ctrl
from camillafir.ui.target_preview_interaction import (
    build_tilt_handle_path,
    build_target_curve_path,
    clamp_manual_target_db,
    extract_target_tilt_from_shape_relayout,
    extract_vertical_shift_from_shape_relayout,
    parse_svg_path_points,
    round_manual_target_tilt_db_per_oct,
    round_manual_target_db,
)


class _DummyControl:
    def __init__(self, value) -> None:
        self.value = value


def test_round_and_clamp_manual_target_db():
    assert round_manual_target_db(1.26) == 1.3
    assert round_manual_target_tilt_db_per_oct(-0.74) == -0.7
    assert clamp_manual_target_db(25.0) == 20.0
    assert clamp_manual_target_db(-12.0) == -10.0


def test_build_target_curve_path_round_trips_through_parser():
    path = build_target_curve_path([10.0, 20.0, 40.0], [-1.5, 0.0, 1.5])

    assert path.startswith("M ")
    assert " L " in path
    assert parse_svg_path_points(path) == [(10.0, -1.5), (20.0, 0.0), (40.0, 1.5)]


def test_extract_vertical_shift_from_shapes_path_payload_uses_current_manual_offset():
    ctrl.reset()
    ctrl.register("lvl_manual_db", _DummyControl(1.0))
    base_points = [(10.0, -2.0), (20.0, -1.0), (40.0, 0.5)]
    shifted_path = build_target_curve_path([10.0, 20.0, 40.0], [0.0, 1.0, 2.5])

    updated = extract_vertical_shift_from_shape_relayout(
        {"shapes[0].path": shifted_path},
        base_points,
    )

    assert updated == 3.0


def test_extract_vertical_shift_from_shapes_list_payload_ignores_x_drag_but_keeps_rebuild_value():
    ctrl.reset()
    ctrl.register("lvl_manual_db", _DummyControl(-1.5))
    base_points = [(10.0, -2.0), (20.0, -1.0), (40.0, 0.5)]
    x_shift_only_path = build_target_curve_path([15.0, 25.0, 45.0], [-2.0, -1.0, 0.5])

    updated = extract_vertical_shift_from_shape_relayout(
        {"shapes": [{"path": x_shift_only_path}]},
        base_points,
    )

    assert updated == -1.5


def test_extract_vertical_shift_anchors_repeated_drag_events_to_same_rendered_base_curve():
    ctrl.reset()
    manual = ctrl.register("lvl_manual_db", _DummyControl(0.0))
    base_points = [(10.0, -2.0), (20.0, -1.0), (40.0, 0.5)]

    first = extract_vertical_shift_from_shape_relayout(
        {"shapes[0].path": build_target_curve_path([10.0, 20.0, 40.0], [-1.0, 0.0, 1.5])},
        base_points,
    )
    manual.value = first
    second = extract_vertical_shift_from_shape_relayout(
        {"shapes[0].path": build_target_curve_path([10.0, 20.0, 40.0], [-0.5, 0.5, 2.0])},
        base_points,
    )

    assert first == 1.0
    assert second == 1.5


def test_extract_vertical_shift_returns_none_when_only_tilt_handle_shape_changed():
    ctrl.reset()
    ctrl.register("lvl_manual_db", _DummyControl(0.0))
    base_curve_points = [(10.0, -2.0), (20.0, -1.0), (40.0, 0.5)]
    moved_tilt_handle = build_tilt_handle_path(-3.0)

    updated = extract_vertical_shift_from_shape_relayout(
        {
            "shapes": [
                {"path": build_target_curve_path([10.0, 20.0, 40.0], [-2.0, -1.0, 0.5])},
                {"path": moved_tilt_handle},
            ]
        },
        base_curve_points,
    )

    assert updated is None


def test_extract_target_tilt_from_shape_relayout_uses_right_side_handle_y_drag():
    ctrl.reset()
    ctrl.register("manual_target_tilt_db_per_oct", _DummyControl(0.0))
    base_handle_points = parse_svg_path_points(build_tilt_handle_path(-1.0))
    moved_handle_path = build_tilt_handle_path(-3.0)

    updated = extract_target_tilt_from_shape_relayout(
        {"shapes[1].path": moved_handle_path},
        base_handle_points,
    )

    assert updated == 0.5


def test_extract_target_tilt_from_shape_relayout_ignores_x_drag_but_keeps_rebuild_value():
    ctrl.reset()
    ctrl.register("manual_target_tilt_db_per_oct", _DummyControl(-0.5))
    base_handle_points = parse_svg_path_points(build_tilt_handle_path(-1.0))
    x_shift_only_path = build_target_curve_path([14000.0, 19000.0], [-1.0, -1.0])

    updated = extract_target_tilt_from_shape_relayout(
        {"shapes": [{}, {"path": x_shift_only_path}]},
        base_handle_points,
    )

    assert updated == -0.5
