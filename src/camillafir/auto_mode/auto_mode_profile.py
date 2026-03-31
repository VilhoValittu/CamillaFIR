"""Lightweight profiling helpers for automatic mode.

Enable via base_data["auto_mode_profile"] = True
or environment variable CAMILLAFIR_AUTO_PROFILE=1.

Results are logged at INFO level at the end of the search.
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
from contextlib import contextmanager


def auto_mode_profile_enabled(base_data: dict | None) -> bool:
    env = os.environ.get("CAMILLAFIR_AUTO_PROFILE", "").strip()
    if env in ("1", "true", "yes"):
        return True
    return bool((base_data or {}).get("auto_mode_profile", False))


class AutoModeProfiler:
    """Accumulates wall-clock timings for named sections, thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sections: dict[str, list] = {}  # name -> [total_s, count]

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                if name not in self._sections:
                    self._sections[name] = [0.0, 0]
                self._sections[name][0] += elapsed
                self._sections[name][1] += 1

    def wrap(self, fn, name: str):
        """Return a wrapped callable that times each call under *name*."""
        profiler = self

        def _timed(*args, **kwargs):
            with active_profiler_scope(profiler):
                with profiler.section(name):
                    return fn(*args, **kwargs)

        return _timed

    # Top-level stage names whose times should not be double-counted in total.
    _TOP_LEVEL_SECTIONS = frozenset({"cache_micro_refine", "search_refine_stages", "finalize"})

    def log_summary(self, logger=None, *, label: str = "auto-mode") -> None:
        with self._lock:
            sections = dict(self._sections)
        if not sections:
            return

        # Wall time = sum of top-level (non-nested) sections only.
        top_s = {k: v for k, v in sections.items() if k in self._TOP_LEVEL_SECTIONS}
        wall_s = sum(v[0] for v in top_s.values()) if top_s else sum(v[0] for v in sections.values())
        lines = [f"[PROFILE] [{label}] wall={wall_s:.3f}s"]
        for name, (s, n) in sorted(sections.items(), key=lambda x: -x[1][0]):
            pct = 100.0 * s / wall_s if wall_s > 0 else 0.0
            per = s / n if n > 0 else 0.0
            if n == 1:
                lines.append(f"  {name}: {s:.3f}s ({pct:.1f}%)")
            else:
                lines.append(
                    f"  {name}: {s:.3f}s ({pct:.1f}%) n={n} avg={per*1000:.1f}ms"
                )
        print("\n".join(lines), flush=True)


_NOOP_PROFILER: AutoModeProfiler | None = None
_ACTIVE_PROFILER: contextvars.ContextVar[AutoModeProfiler | None] = contextvars.ContextVar(
    "camillafir_auto_mode_profiler",
    default=None,
)


def current_active_profiler() -> AutoModeProfiler | None:
    return _ACTIVE_PROFILER.get()


@contextmanager
def active_profiler_scope(profiler: AutoModeProfiler | None):
    if profiler is None:
        yield
        return
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield
    finally:
        _ACTIVE_PROFILER.reset(token)


@contextmanager
def profiled_section(name: str):
    profiler = current_active_profiler()
    if profiler is None:
        yield
        return
    with profiler.section(name):
        yield


def _noop_profiler() -> AutoModeProfiler:
    """Shared instance used when profiling is off – wrapping has zero cost
    because wrap() is never called on it."""
    global _NOOP_PROFILER
    if _NOOP_PROFILER is None:
        _NOOP_PROFILER = AutoModeProfiler()
    return _NOOP_PROFILER
