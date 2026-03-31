from camillafir.auto_mode.auto_mode_profile import (
    AutoModeProfiler,
    active_profiler_scope,
    profiled_section,
)


def test_profiled_section_records_nested_sections_under_wrapped_call():
    profiler = AutoModeProfiler()

    def _work():
        with profiled_section("inner"):
            return "ok"

    wrapped = profiler.wrap(_work, "outer")

    assert wrapped() == "ok"
    assert "outer" in profiler._sections
    assert "inner" in profiler._sections
    assert profiler._sections["outer"][1] == 1
    assert profiler._sections["inner"][1] == 1


def test_profiled_section_records_nested_sections_under_explicit_active_scope():
    profiler = AutoModeProfiler()

    with active_profiler_scope(profiler):
        with profiler.section("outer"):
            with profiled_section("inner"):
                pass

    assert "outer" in profiler._sections
    assert "inner" in profiler._sections
    assert profiler._sections["outer"][1] == 1
    assert profiler._sections["inner"][1] == 1
