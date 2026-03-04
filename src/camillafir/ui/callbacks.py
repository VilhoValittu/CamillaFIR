import numpy as np
from pywebio.output import put_html, use_scope
from pywebio.pin import pin, pin_on_change, pin_update

from .camillafir_housecurve import _normalize_hc_mode_key
from .camillafir_modes import MODE_DEFAULTS
from .camillafir_ui_helpers import (
    _warn_max_boost_if_over_cap,
    _warn_taps_if_over_cap,
    apply_afdw_preset,
    apply_mode_defaults_to_ui,
    apply_tdc_preset,
    update_afdw_cycles_ui,
    update_basic_clamp_hints_ui,
    update_confidence_pull_ui,
    update_ir_export_window_mode_ui,
    update_ir_lr_window_ui,
    update_ir_tukey_ui,
    update_ir_window_shape_ui,
    update_low_bass_cut_ui,
    update_lvl_ui,
    update_mixed_freq_ui,
    update_mode_desc,
    update_taps_auto_info,
    update_target_preview_ui,
    update_tdc_controls_ui,
    update_unsafe_raw_dsp_ui,
)

_PROCESS_RUN_HOOK = None


def configure_engine_hooks(*, process_run=None):
    global _PROCESS_RUN_HOOK
    _PROCESS_RUN_HOOK = process_run


def on_start_click():
    if callable(_PROCESS_RUN_HOOK):
        import threading
        import time

        stop_evt = threading.Event()
        run_started_at = time.perf_counter()

        def _timer_tick():
            try:
                from . import app as _app
            except Exception:
                return
            while not stop_evt.wait(1.0):
                try:
                    base = _app.get_status_base_message(default="CamillaFIR running")
                    elapsed = max(0.0, float(time.perf_counter() - run_started_at))
                    _app.update_status(f"{base} | {elapsed:.1f} s")
                except Exception:
                    pass

        try:
            from pywebio.session import register_thread
        except Exception:
            register_thread = None

        timer_thread = threading.Thread(
            target=_timer_tick,
            name="camillafir_status_timer",
            daemon=True,
        )
        try:
            if callable(register_thread):
                register_thread(timer_thread)
        except Exception:
            pass
        timer_thread.start()

        try:
            try:
                from . import app as _app
                _app.update_status("CamillaFIR running | 0.0 s")
            except Exception:
                pass
            return _PROCESS_RUN_HOOK()
        finally:
            stop_evt.set()
    return None


def on_mode_apply_defaults():
    apply_mode_defaults_to_ui()


def on_afdw_preset(preset):
    apply_afdw_preset(preset)


def _pin_get(key, default=None):
    try:
        v = pin.get(key, None)
        if v is None:
            return default
        return v
    except Exception:
        try:
            return pin[key]
        except Exception:
            return default


def update_engine_metrics_ui(*, pin=pin, pin_update=pin_update):
    """Soveltaa tai paivittaa: update engine metrics ui."""
    try:
        try:
            fs_v = pin["fs"]
            taps_v = pin["taps"]
        except Exception:
            return

        if fs_v in (None, "") or taps_v in (None, ""):
            return

        fs_f = float(fs_v)
        taps_i = int(float(taps_v))
        if fs_f <= 0 or taps_i <= 0:
            return

        latency_ms = (taps_i / 2.0) / fs_f * 1000.0
        bin_hz = fs_f / float(taps_i)

        lat_txt = f"{latency_ms:.0f} ms"
        res_txt = f"{bin_hz:.3f} Hz/bin"

        with use_scope("engine_metrics_scope", clear=True):
            put_html(
                f"""
                <div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:10px;">
                  <div style="padding:6px 10px; border:1px solid rgba(255,255,255,0.14); border-radius:999px;
                              background: rgba(255,255,255,0.04); font-size:12px; color:#cbd5e1;">
                    ⏱ Latency (Linear): <b style="color:#fff">{lat_txt}</b>
                  </div>
                  <div style="padding:6px 10px; border:1px solid rgba(255,255,255,0.14); border-radius:999px;
                              background: rgba(255,255,255,0.04); font-size:12px; color:#cbd5e1;">
                    🔎 Resolution: <b style="color:#fff">{res_txt}</b>
                  </div>
                </div>
                """
            )
    except Exception:
        return


def register_callbacks(*, t, get_val, pin=pin, pin_update=pin_update, pin_on_change=pin_on_change):
    def _on_hc_mode_change(v):
        if _normalize_hc_mode_key(v) != "Upload":
            pin_update("hc_custom_file", value=None)

    pin_on_change("hc_mode", _on_hc_mode_change)

    update_target_preview_ui(None)

    def _render_engine_metrics_later():
        try:
            update_engine_metrics_ui(pin=pin, pin_update=pin_update)
        except Exception:
            pass

    _render_engine_metrics_later()

    def _on_lvl_range_change(_=None):
        try:
            a = pin.get("lvl_min", None)
            b = pin.get("lvl_max", None)
            if a is None or b is None:
                update_target_preview_ui()
                return
            a = float(a)
            b = float(b)
            if not np.isfinite(a) or not np.isfinite(b):
                update_target_preview_ui()
                return
            if a > b:
                pin_update("lvl_min", value=b)
                pin_update("lvl_max", value=a)
        except Exception:
            pass
        update_target_preview_ui()

    update_lvl_ui()
    update_ir_tukey_ui()
    update_ir_export_window_mode_ui()
    update_mixed_freq_ui()
    update_ir_window_shape_ui()
    update_ir_lr_window_ui()
    update_low_bass_cut_ui(
        pin=pin,
        pin_update=pin_update,
        get_val=get_val,
        t=t,
    )
    update_basic_clamp_hints_ui(pin=pin, pin_update=pin_update, t=t)
    update_confidence_pull_ui(pin=pin, get_val=get_val, t=t)
    update_unsafe_raw_dsp_ui(pin=pin, get_val=get_val, t=t)
    update_target_preview_ui()

    def _refresh_ir_window_controls(_=None):
        try:
            m = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
            ft = str(_pin_get("filter_type", "") or "").strip().lower()
            is_asym_filter = ("asym" in ft)
            if m in ("BASIC", "AUTO") and (not is_asym_filter):
                if str(pin.get("ir_export_window_mode", "") or "").lower() != "auto":
                    pin_update("ir_export_window_mode", value="auto")
        except Exception:
            pass

        update_ir_export_window_mode_ui()
        update_ir_window_shape_ui()
        update_ir_tukey_ui()
        update_ir_lr_window_ui()
        update_afdw_cycles_ui(pin=pin, get_val=get_val, t=t)
        update_tdc_controls_ui(pin=pin, get_val=get_val, t=t, apply_tdc_preset=apply_tdc_preset)
        update_basic_clamp_hints_ui(pin=pin, pin_update=pin_update, t=t)

    def _on_filter_type_change(_=None):
        update_mixed_freq_ui()
        _refresh_ir_window_controls()

    pin_on_change("hc_mode", onchange=lambda _: update_target_preview_ui())
    pin_on_change("hc_custom_file", onchange=lambda _: update_target_preview_ui())
    pin_on_change("mag_c_min", onchange=lambda _: update_target_preview_ui())
    pin_on_change("mag_c_max", onchange=lambda _: update_target_preview_ui())
    for _preview_pin in (
        "file_l",
        "file_r",
        "local_path_l",
        "local_path_r",
        "lvl_min",
        "lvl_max",
        "ir_window_left",
        "ir_window_right",
        "ir_window",
        "filter_smooth",
        "smoothing_level",
    ):
        pin_on_change(_preview_pin, onchange=lambda _: update_target_preview_ui())

    pin_on_change("ir_export_window_mode", onchange=_refresh_ir_window_controls)
    pin_on_change("filter_type", onchange=_on_filter_type_change)
    pin_on_change(
        "low_bass_cut_enable",
        onchange=lambda _: (
            update_low_bass_cut_ui(
                pin=pin,
                pin_update=pin_update,
                get_val=get_val,
                t=t,
            ),
            update_basic_clamp_hints_ui(pin=pin, pin_update=pin_update, t=t),
        ),
    )
    pin_on_change(
        "enable_tdc",
        onchange=lambda _: (
            update_tdc_controls_ui(pin=pin, get_val=get_val, t=t, apply_tdc_preset=apply_tdc_preset),
            update_basic_clamp_hints_ui(pin=pin, pin_update=pin_update, t=t),
        ),
    )

    pin_on_change("lvl_mode", onchange=lambda _: (update_lvl_ui(), update_target_preview_ui()))
    pin_on_change("ir_export_window_shape", onchange=update_ir_tukey_ui)
    pin_on_change("lvl_min", onchange=_on_lvl_range_change)
    pin_on_change("lvl_max", onchange=_on_lvl_range_change)
    pin_on_change("lvl_manual_db", onchange=_on_lvl_range_change)
    pin_on_change("taps", onchange=_warn_taps_if_over_cap)
    pin_on_change(
        "enable_afdw",
        onchange=lambda _: (
            update_afdw_cycles_ui(pin=pin, get_val=get_val, t=t),
            update_basic_clamp_hints_ui(pin=pin, pin_update=pin_update, t=t),
        ),
    )

    _warn_taps_if_over_cap()

    def _on_mode_change(_=None):
        try:
            m = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
        except Exception:
            m = "BASIC"

        try:
            if m in ("BASIC", "AUTO"):
                pin_update(
                    "lvl_mode",
                    options=[{"label": t("lvl_mode_auto"), "value": "Auto"}],
                    value="Auto",
                )
            else:
                cur_mode = str(_pin_get("lvl_mode", "Auto") or "Auto")
                if cur_mode not in ("Auto", "Manual"):
                    cur_mode = "Auto"
                pin_update(
                    "lvl_mode",
                    options=[
                        {"label": t("lvl_mode_auto"), "value": "Auto"},
                        {"label": t("lvl_mode_manual"), "value": "Manual"},
                    ],
                    value=cur_mode,
                )
        except Exception:
            pass

        try:
            update_confidence_pull_ui(pin=pin, get_val=get_val, t=t)
        except Exception:
            pass
        try:
            update_unsafe_raw_dsp_ui(pin=pin, get_val=get_val, t=t)
        except Exception:
            pass

        update_mode_desc()
        try:
            update_lvl_ui()
            update_target_preview_ui()
            update_basic_clamp_hints_ui(pin=pin, pin_update=pin_update, t=t)
        except Exception:
            pass
        try:
            m = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
            mode_key = "BASIC" if m == "AUTO" else m
            v = (MODE_DEFAULTS.get(mode_key, {}) or {}).get("ir_export_window_mode", None)
            if isinstance(v, str) and v.strip():
                pin_update("ir_export_window_mode", value=v.strip())
                _refresh_ir_window_controls()
        except Exception:
            pass

    pin_on_change("mode", onchange=_on_mode_change)
    _on_mode_change()

    pin_on_change("multi_rate_opt", onchange=update_taps_auto_info)
    pin_on_change("fs", onchange=update_taps_auto_info)
    pin_on_change("taps", onchange=update_taps_auto_info)
    pin_on_change("fs", onchange=lambda _: update_engine_metrics_ui(pin=pin, pin_update=pin_update))
    pin_on_change("taps", onchange=lambda _: update_engine_metrics_ui(pin=pin, pin_update=pin_update))
    update_taps_auto_info()

    pin_on_change("max_boost", onchange=_warn_max_boost_if_over_cap)
    _warn_max_boost_if_over_cap()
