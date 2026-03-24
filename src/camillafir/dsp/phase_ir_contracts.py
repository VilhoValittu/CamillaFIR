from __future__ import annotations

import numpy as np


def arrays_equal_strict(lhs, rhs) -> bool:
    a = np.asarray(lhs)
    b = np.asarray(rhs)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in ("f", "c") or b.dtype.kind in ("f", "c"):
        return bool(np.allclose(a, b, atol=0.0, rtol=0.0, equal_nan=True))
    return bool(np.array_equal(a, b))


def require_unchanged(stage: str, value_name: str, before, after) -> None:
    if arrays_equal_strict(before, after):
        return
    raise RuntimeError(
        f"Phase-IR contract breach in {stage}: '{value_name}' was modified, "
        "but this stage is not allowed to mutate it."
    )


def require_scalar_unchanged(stage: str, value_name: str, before, after) -> None:
    if bool(np.isclose(float(before), float(after), atol=0.0, rtol=0.0, equal_nan=True)):
        return
    raise RuntimeError(
        f"Phase-IR contract breach in {stage}: scalar '{value_name}' changed "
        "but must remain immutable in this stage."
    )


def require_allowed_keys(stage: str, obj, allowed: frozenset[str]) -> None:
    if not isinstance(obj, dict):
        raise RuntimeError(
            f"Phase-IR contract breach in {stage}: expected dict output, got {type(obj).__name__}."
        )
    extra = set(obj.keys()) - set(allowed)
    if extra:
        keys_txt = ", ".join(sorted(str(k) for k in extra))
        raise RuntimeError(
            f"Phase-IR contract breach in {stage}: unexpected output keys: {keys_txt}."
        )
