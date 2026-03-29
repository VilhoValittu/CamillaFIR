import logging
import sys
from datetime import datetime

import numpy as np
from pywebio.output import put_html, use_scope
from pywebio.pin import pin, pin_on_change, pin_update

from .camillafir_housecurve import _normalize_hc_mode_key
from ..config.mode_policy import MODE_DEFAULTS
from . import ui_state
from .camillafir_ui_helpers import (
    _warn_max_boost_if_over_cap,
    _warn_taps_if_over_cap,
)
from .controls_ir_window import (
    update_afdw_cycles_ui,
    update_bass_first_ui,
    update_ir_export_window_mode_ui,
    update_ir_lr_window_ui,
    update_ir_tukey_ui,
    update_ir_window_shape_ui,
    update_low_bass_cut_ui,
    update_lvl_ui,
    update_mixed_freq_ui,
    update_tdc_controls_ui,
)
from .controls_mode import (
    apply_afdw_preset,
    apply_mode_defaults_to_ui,
    apply_tdc_preset,
    update_auto_mode_controls_ui,
    update_basic_clamp_hints_ui,
    update_confidence_pull_ui,
    update_mode_desc,
    update_taps_auto_info,
    update_unsafe_raw_dsp_ui,
)
from .controls_target_preview import update_target_preview_ui

logger = logging.getLogger("CamillaFIR")
_PROCESS_RUN_HOOK = None


def _terminal_run_event(label: str, *, clock_text: str, elapsed_s: float | None = None, started_clock: str | None = None):
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        parts = [f"{ts} - INFO - {str(label or '').strip()} {str(clock_text or '').strip()}"]
        if started_clock:
            parts.append(f"(started {str(started_clock).strip()})")
        if elapsed_s is not None:
            parts.append(f"| {float(elapsed_s):.1f} s")
        sys.stdout.write(" ".join(parts).strip() + "\n")
        sys.stdout.flush()
    except (OSError, TypeError, ValueError):
        pass


def configure_engine_hooks(*, process_run=None):
    global _PROCESS_RUN_HOOK
    _PROCESS_RUN_HOOK = process_run


def on_start_click():
    if callable(_PROCESS_RUN_HOOK):
        import threading
        import time

        try:
            from . import app as _app
            _app.activate_run_tab()
        except Exception:
            # UI-only convenience; keep the DSP run reachable even if tab activation fails.
            logger.debug("Run-tab activation unavailable in current UI context", exc_info=True)
            pass

        try:
            with use_scope("results", clear=True):
                pass
        except Exception:
            # UI-only cleanup must not block starting the DSP run if the scope is not mounted yet.
            logger.debug("Unable to clear previous results scope before run start", exc_info=True)
            pass

        stop_evt = threading.Event()
        run_started_at = time.perf_counter()
        run_started_clock = datetime.now().strftime("%H:%M:%S")
        _terminal_run_event("START", clock_text=run_started_clock)
        timer_error_logged = False

        def _timer_tick():
            nonlocal timer_error_logged
            while not stop_evt.wait(1.0):
                try:
                    base = ui_state.get_status_base_message(default="CamillaFIR running")
                    elapsed = max(0.0, float(time.perf_counter() - run_started_at))
                    ui_state.update_status(f"{base} | {elapsed:.1f} s")
                except Exception:
                    # Best-effort UI refresh only; keep the processing thread alive if the session/DOM changes.
                    if not timer_error_logged:
                        logger.debug("Status timer update failed; suppressing further timer errors", exc_info=True)
                        timer_error_logged = True
                    pass

        try:
            from pywebio.session import register_thread
        except ImportError:
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
            # Thread registration is optional; the timer still works as a daemon fallback.
            logger.debug("Failed to register status timer thread with PyWebIO session", exc_info=True)
            pass
        timer_thread.start()

        try:
            try:
                ui_state.set_run_wall_clock_text(run_started_clock)
                ui_state.update_status_notices(summary_text="", info_text="")
                ui_state.update_auto_selected_bar("")
                ui_state.reset_auto_status_details()
                ui_state.update_status("CamillaFIR running | 0.0 s")
            except Exception:
                # Status reset is cosmetic; starting the run must remain possible even if UI state sync fails.
                logger.debug("Failed to reset UI status state before run start", exc_info=True)
                pass
            result = _PROCESS_RUN_HOOK()
            elapsed_done = max(0.0, float(time.perf_counter() - run_started_at))
            _terminal_run_event(
                "READY",
                clock_text=datetime.now().strftime("%H:%M:%S"),
                elapsed_s=elapsed_done,
                started_clock=run_started_clock,
            )
            return result
        except Exception:
            logger.exception("CamillaFIR run failed in UI start handler")
            elapsed_fail = max(0.0, float(time.perf_counter() - run_started_at))
            _terminal_run_event(
                "FAILED",
                clock_text=datetime.now().strftime("%H:%M:%S"),
                elapsed_s=elapsed_fail,
                started_clock=run_started_clock,
            )
            raise
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
        except (KeyError, TypeError):
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
        logger.debug("Engine metrics UI refresh failed", exc_info=True)
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
            logger.debug("Deferred engine metrics UI refresh failed", exc_info=True)
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
            logger.debug("Level-range normalization failed; continuing with preview refresh", exc_info=True)
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
    update_bass_first_ui(pin=pin, get_val=get_val, t=t)
    update_afdw_cycles_ui(pin=pin, get_val=get_val, t=t)
    update_tdc_controls_ui(pin=pin, get_val=get_val, t=t, apply_tdc_preset=apply_tdc_preset)
    update_basic_clamp_hints_ui(pin=pin, pin_update=pin_update, t=t)
    update_confidence_pull_ui(pin=pin, get_val=get_val, t=t)
    update_unsafe_raw_dsp_ui(pin=pin, get_val=get_val, t=t)
    update_auto_mode_controls_ui()
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
            logger.debug("IR-window mode normalization failed; proceeding with UI refresh", exc_info=True)
            pass

        update_ir_export_window_mode_ui()
        update_ir_window_shape_ui()
        update_ir_tukey_ui()
        update_ir_lr_window_ui()
        update_afdw_cycles_ui(pin=pin, get_val=get_val, t=t)
        update_tdc_controls_ui(pin=pin, get_val=get_val, t=t, apply_tdc_preset=apply_tdc_preset)
        update_basic_clamp_hints_ui(pin=pin, pin_update=pin_update, t=t)
        update_auto_mode_controls_ui()

    def _on_filter_type_change(_=None):
        update_mixed_freq_ui()
        _refresh_ir_window_controls()

    pin_on_change("hc_mode", onchange=lambda _: update_target_preview_ui())
    pin_on_change("auto_goal", onchange=lambda _: update_target_preview_ui())
    pin_on_change("auto_target_mode", onchange=lambda _: (update_target_preview_ui(), update_auto_mode_controls_ui()))
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
            update_auto_mode_controls_ui(),
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
            update_auto_mode_controls_ui(),
        ),
    )
    pin_on_change(
        "bass_first_ai",
        onchange=lambda _: update_bass_first_ui(pin=pin, get_val=get_val, t=t),
    )

    def _update_stereo_link_strategy_state(_=None):
        try:
            from pywebio.session import run_js
            mode_u = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
            is_auto = (mode_u == "AUTO")
            enabled_raw = pin.get("stereo_link", None)
            if isinstance(enabled_raw, (list, tuple, set)):
                link_on = len(enabled_raw) > 0
            else:
                link_on = bool(enabled_raw) if enabled_raw is not None else False
            disable = (not link_on) or is_auto
            js_dis = "true" if disable else "false"
            run_js(
                "(function(){"
                "var els=document.querySelectorAll('[name=\"stereo_link_strategy\"]');"
                "if(!els.length)return;"
                f"var dis={js_dis};"
                "els.forEach(function(el){el.disabled=dis;});"
                "var wrap=els[0].closest('.form-group')||els[0].closest('div');"
                "if(wrap){"
                "wrap.style.opacity=dis?'0.55':'';"
                "wrap.style.pointerEvents=dis?'none':'';"
                "wrap.style.filter=dis?'grayscale(1)':'';"
                "}"
                "})();"
            )
        except Exception:
            pass

    pin_on_change("stereo_link", onchange=_update_stereo_link_strategy_state)

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
            logger.debug("Failed to update lvl_mode selector for current mode", exc_info=True)
            pass

        try:
            update_confidence_pull_ui(pin=pin, get_val=get_val, t=t)
        except Exception:
            logger.debug("Confidence-pull UI refresh failed", exc_info=True)
            pass
        try:
            update_unsafe_raw_dsp_ui(pin=pin, get_val=get_val, t=t)
        except Exception:
            logger.debug("Unsafe-raw-DSP UI refresh failed", exc_info=True)
            pass

        update_mode_desc()
        update_auto_mode_controls_ui()
        try:
            update_lvl_ui()
            update_target_preview_ui()
            update_basic_clamp_hints_ui(pin=pin, pin_update=pin_update, t=t)
        except Exception:
            logger.debug("Mode-change UI refresh failed", exc_info=True)
            pass
        try:
            m = str(_pin_get("mode", "BASIC") or "BASIC").strip().upper()
            mode_key = "BASIC" if m == "AUTO" else m
            v = (MODE_DEFAULTS.get(mode_key, {}) or {}).get("ir_export_window_mode", None)
            if isinstance(v, str) and v.strip():
                pin_update("ir_export_window_mode", value=v.strip())
                _refresh_ir_window_controls()
        except Exception:
            logger.debug("Mode default IR-window sync failed", exc_info=True)
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
