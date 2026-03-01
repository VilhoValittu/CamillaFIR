from __future__ import annotations

import numpy as np
import scipy.fft


def _build_complex_spectrum(corr_mag, phase_final) -> np.ndarray:
    return np.asarray(corr_mag, dtype=float) * np.exp(1j * np.asarray(phase_final, dtype=float))


def _ifft_to_ir(H, n: int | None = None) -> np.ndarray:
    if n is None:
        n = max(2, int((len(H) - 1) * 2))
    return scipy.fft.irfft(np.asarray(H), n=int(n))


def _normalize_ir(ir, cfg) -> np.ndarray:
    if not bool(getattr(cfg, "phase_ir_stage_normalize", False)):
        return np.asarray(ir, dtype=float)
    out = np.asarray(ir, dtype=float).copy()
    mx = float(np.max(np.abs(out))) if out.size else 0.0
    if mx > 0.0:
        out *= (0.89 / mx)
    return out
