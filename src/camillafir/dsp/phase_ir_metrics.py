from __future__ import annotations

from typing import Any

import numpy as np

from .phase_ir_utils import _pre_post_energy_ratio, _pre_ringing_db, _resolve_ir_energy_split


def _summarize_ir_metrics(ir_final, cfg, st) -> dict[str, Any]:
    x = np.asarray(ir_final, dtype=float)
    split_hint = None
    try:
        if isinstance(st, dict):
            split_hint = st.get("ir_energy_split_samples", None)
    except Exception:
        split_hint = None
    if x.size == 0:
        out = {
            "ir_len": 0,
            "ir_peak_samples": 0,
            "ir_peak_db": float("-inf"),
            "ir_rms": 0.0,
            "ir_pre_ringing_db": float("nan"),
            "ir_pre_post_ratio": float("nan"),
            "ir_pre_ringing_db_raw": float("nan"),
            "ir_pre_post_ratio_raw": float("nan"),
            "ir_pre_energy_split_samples": 0,
            "pre_energy_metric_suspect": False,
            "pre_energy_metric_note": "empty_ir",
        }
    else:
        peak_idx = int(np.argmax(np.abs(x)))
        peak = float(np.max(np.abs(x)))
        split_idx = int(_resolve_ir_energy_split(x, split_hint if split_hint is not None else peak_idx))
        ratio = float(_pre_post_energy_ratio(x, split=split_idx))
        pre_db = float(_pre_ringing_db(x, split=split_idx))
        suspect = bool(np.isfinite(ratio) and (ratio < 1e-10))
        note = "ok"
        if suspect:
            note = "pre/post < 1e-10 (likely zeroed or split issue)"
        elif not np.isfinite(ratio):
            note = "pre/post unavailable (n<32 or invalid split)"
        display_db = float("nan") if suspect else float(pre_db)
        display_ratio = float("nan") if suspect else float(ratio)
        out = {
            "ir_len": int(x.size),
            "ir_peak_samples": int(peak_idx),
            "ir_peak_db": float(20.0 * np.log10(peak + 1e-30)),
            "ir_rms": float(np.sqrt(np.mean(x * x))),
            "ir_pre_ringing_db": float(display_db),
            "ir_pre_post_ratio": float(display_ratio),
            "ir_pre_ringing_db_raw": float(pre_db),
            "ir_pre_post_ratio_raw": float(ratio),
            "ir_pre_energy_split_samples": int(split_idx),
            "pre_energy_metric_suspect": bool(suspect),
            "pre_energy_metric_note": str(note),
        }
    try:
        if isinstance(st, dict):
            for k, v in out.items():
                st[k] = v
    except Exception:
        pass
    return out
