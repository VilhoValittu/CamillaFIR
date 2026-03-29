"""NiceGUI dark theme for CamillaFIR.

Call apply_theme() once during page setup (before any components are created).
"""
from __future__ import annotations

_CF_CSS = """
:root {
    --cf-bg:        #0b0f14;
    --cf-surface:   rgba(255,255,255,0.04);
    --cf-surface-2: rgba(255,255,255,0.07);
    --cf-surface-3: rgba(255,255,255,0.11);
    --cf-border:    rgba(255,255,255,0.10);
    --cf-border-2:  rgba(255,255,255,0.18);
    --cf-text:      #e8eaed;
    --cf-muted:     #8a929e;
    --cf-faint:     #5a6070;
    --cf-accent:    #4f8ef7;
    --cf-focus:     rgba(79,142,247,0.35);
    --cf-radius:    8px;
    --cf-radius-sm: 4px;
    --cf-shadow:    0 2px 12px rgba(0,0,0,0.45);
}

body { background: var(--cf-bg) !important; color: var(--cf-text) !important; }

/* Quasar card/surface overrides */
.q-card       { background: var(--cf-surface-2) !important; border: 1px solid var(--cf-border) !important; }
.q-expansion-item { border: 1px solid var(--cf-border) !important; margin-bottom: 4px; border-radius: var(--cf-radius-sm); }
.q-expansion-item__header { background: var(--cf-surface-2) !important; }
.q-expansion-item__content { background: var(--cf-bg) !important; }

/* Tabs */
.q-tabs      { background: var(--cf-surface-2) !important; }
.q-tab       { color: var(--cf-muted) !important; }
.q-tab--active { color: var(--cf-accent) !important; }
.q-tabs__content .q-tab__indicator { background: var(--cf-accent) !important; }

/* Sticky top area */
.cf-top-shell {
    position: sticky;
    top: 0;
    z-index: 1000;
    background: var(--cf-bg);
    box-shadow: 0 8px 18px rgba(0,0,0,0.28);
}
.cf-top-shell .q-separator { background: var(--cf-border) !important; }

/* Inputs */
.q-field__control { background: var(--cf-surface-2) !important; border-radius: var(--cf-radius-sm) !important; }
.q-field__label   { color: var(--cf-muted) !important; }
.q-field__native, .q-field__input { color: var(--cf-text) !important; }
.q-select__dropdown-icon { color: var(--cf-muted) !important; }

/* Buttons */
.q-btn:not(.bg-primary):not(.bg-positive):not(.bg-negative):not(.bg-warning) {
    border: 1px solid var(--cf-border-2) !important;
}

/* Table */
.q-table thead tr th { background: var(--cf-surface-3) !important; color: var(--cf-muted) !important; }
.q-table tbody tr:nth-child(even) { background: var(--cf-surface) !important; }
.q-table tbody tr:hover           { background: var(--cf-surface-2) !important; }

/* Scrollbar */
::-webkit-scrollbar            { width: 8px; height: 8px; }
::-webkit-scrollbar-track      { background: var(--cf-bg); }
::-webkit-scrollbar-thumb      { background: var(--cf-border-2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cf-muted); }

/* Brand header */
.cf-brand-logo-wrap { display: flex; align-items: center; gap: 12px; padding: 8px 0; }
.cf-brand-logo      { font-size: 1.6rem; font-weight: 700; letter-spacing: 2px;
                      color: var(--cf-text); user-select: none; }
.cf-brand-version   { font-size: 0.75rem; color: var(--cf-muted); align-self: flex-end; padding-bottom: 3px; }

/* Status area */
.cf-status-summary { background: rgba(40,120,60,0.18); border-left: 3px solid #3a9a5c;
                     padding: 8px 12px; border-radius: var(--cf-radius-sm); margin-bottom: 6px; }
.cf-status-info    { background: rgba(40,90,160,0.18); border-left: 3px solid #3a72c8;
                     padding: 8px 12px; border-radius: var(--cf-radius-sm); margin-bottom: 6px; }
.cf-status-text    { color: var(--cf-muted); font-size: 0.85rem; }
.cf-auto-bar       { background: rgba(40,120,60,0.22); border-left: 3px solid #3a9a5c;
                     padding: 6px 12px; border-radius: var(--cf-radius-sm); font-size: 0.85rem; }

/* Dialogs */
.q-dialog__backdrop {
    background: rgba(0,0,0,0.78) !important;
}
.cf-modal-card {
    background: #151a22 !important;
    border: 1px solid var(--cf-border-2) !important;
    box-shadow: 0 18px 48px rgba(0,0,0,0.62) !important;
}
.cf-modal-card .nicegui-markdown {
    color: var(--cf-text);
}

/* Info panel (top-right header summary) */
.cf-info-panel {
    background: var(--cf-surface-2);
    border: 1px solid var(--cf-border);
    border-radius: var(--cf-radius-sm);
    padding: 6px 12px;
    min-width: 220px;
    max-width: 340px;
    text-align: right;
    font-size: 0.78rem;
    line-height: 1.6;
    color: var(--cf-text);
    white-space: nowrap;
    overflow: hidden;
    flex-shrink: 0;
}
.cf-info-line-dim   { color: var(--cf-muted); }
.cf-info-line-score { color: var(--cf-accent); font-weight: 600; }
"""


def apply_theme() -> None:
    """Inject dark theme CSS and enable NiceGUI dark mode.

    Must be called inside a @ui.page handler, before other components.
    """
    from nicegui import ui  # noqa: PLC0415
    ui.dark_mode().enable()
    ui.add_css(_CF_CSS)
