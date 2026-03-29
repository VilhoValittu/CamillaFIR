from pywebio.output import put_html


def _inject_dark_css():
    put_html(
        """
<style>
  /* ===== CamillaFIR Matte Dark - PyWebIO specific ===== */

  :root{
    color-scheme: dark;

    --cf-bg: #0b0f14;
    --cf-surface: rgba(255,255,255,0.035);
    --cf-surface-2: rgba(255,255,255,0.055);
    --cf-surface-3: rgba(255,255,255,0.075);
    --cf-border: rgba(255,255,255,0.10);
    --cf-border-2: rgba(255,255,255,0.16);

    --cf-text: rgba(255,255,255,0.92);
    --cf-muted: rgba(255,255,255,0.70);
    --cf-faint: rgba(255,255,255,0.52);

    --cf-accent: rgba(160,210,255,0.80);
    --cf-focus: rgba(160,210,255,0.22);

    --cf-radius: 14px;
    --cf-radius-sm: 10px;
    --cf-shadow: 0 8px 22px rgba(0,0,0,0.35);
  }

  /* ===== Root containers ===== */
  body.webio-theme-dark {
    background: var(--cf-bg) !important;
    color: var(--cf-text) !important;
  }

  .pywebio,
  #output-container,
  #input-container {
    background: var(--cf-bg) !important;
    color: var(--cf-text) !important;
  }

  /* ===== Markdown output ===== */
  .markdown-body {
    color: var(--cf-text) !important;
  }
  .markdown-body h1 { font-size: 28px; }
  .markdown-body h2 { font-size: 20px; margin-top: 18px; }
  .markdown-body h3 { font-size: 16px; opacity: 0.95; }
  .markdown-body p,
  .markdown-body li { color: var(--cf-text); }
  .markdown-body em,
  .markdown-body small { color: var(--cf-muted); }
  .markdown-body hr { border-color: rgba(255,255,255,0.08); }

  /* ===== Cards / collapses ===== */
  .card,
  .collapse,
  .panel,
  .well {
    background: var(--cf-surface) !important;
    border: 1px solid var(--cf-border) !important;
    border-radius: var(--cf-radius) !important;
    box-shadow: var(--cf-shadow) !important;
  }

  .card-header,
  .collapse > .title {
    color: var(--cf-text) !important;
    font-weight: 700;
  }

  /* ===== Tabs (Bootstrap via PyWebIO) ===== */
  .nav-tabs {
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
  }
  .nav-tabs .nav-link {
    color: var(--cf-muted) !important;
    background: transparent !important;
    border: 0 !important;
    border-radius: 12px 12px 0 0 !important;
    padding: 10px 14px !important;
  }
  .nav-tabs .nav-link.active {
    color: var(--cf-text) !important;
    background: var(--cf-surface-2) !important;
    border: 1px solid var(--cf-border) !important;
    border-bottom: 0 !important;
  }

  .tab-content {
    background: var(--cf-surface) !important;
    border: 1px solid var(--cf-border) !important;
    border-top: 0 !important;
    border-radius: 0 0 var(--cf-radius) var(--cf-radius) !important;
    padding: 14px 14px 6px !important;
  }

  /* ===== Inputs ===== */
  input,
  .form-control,
  select,
  .custom-select,
  textarea {
    background: #000000 !important;
    color: var(--cf-text) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
  }
  input:hover,
  .form-control:hover,
  select:hover,
  .custom-select:hover,
  textarea:hover {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.18) !important;
  }
  input:focus,
  .form-control:focus,
  select:focus,
  .custom-select:focus,
  textarea:focus {
    border-color: rgba(160,210,255,0.45) !important;
    box-shadow: 0 0 0 3px var(--cf-focus) !important;
  }

  input[readonly],
  textarea[readonly],
  .form-control[readonly] {
    background: #000000 !important;
    color: var(--cf-text) !important;
  }

  .input-group,
  .custom-file {
    background: transparent !important;
  }

  .input-group-text,
  .custom-file-label,
  .custom-file-label::after {
    background: #000000 !important;
    color: var(--cf-text) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
  }

  .custom-file-label::after {
    border-left: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 0 12px 12px 0 !important;
  }

  .custom-file-input:focus ~ .custom-file-label {
    border-color: rgba(160,210,255,0.45) !important;
    box-shadow: 0 0 0 3px var(--cf-focus) !important;
  }

  label { color: var(--cf-text) !important; }
  .help-block,
  .form-text,
  .input-help {
    color: var(--cf-faint) !important;
    font-size: 13px;
  }

  /* ===== Native select dropdown (critical fix) ===== */
  select { color-scheme: dark; }
  select option,
  select optgroup {
    color: #0b0f14 !important;
    background: #ffffff !important;
  }
  select option:checked {
    background: #dbeafe !important;
    color: #0b0f14 !important;
  }

  /* ===== Buttons ===== */
  button,
  .btn,
  .pywebio-button {
    background: var(--cf-surface-2) !important;
    color: var(--cf-text) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 14px !important;
  }
  button:hover,
  .btn:hover,
  .pywebio-button:hover {
    background: var(--cf-surface-3) !important;
    border-color: rgba(255,255,255,0.22) !important;
  }

  /* ===== Tables ===== */
  table,
  .table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    background: transparent !important;
    color: var(--cf-text) !important;
  }
  thead,
  tbody,
  tr {
    background: transparent !important;
  }
  th {
    background: rgba(255,255,255,0.05) !important;
    color: var(--cf-text) !important;
    font-weight: 700;
    border: 1px solid var(--cf-border) !important;
    padding: 8px 10px !important;
  }
  td {
    background: #000000 !important;
    color: var(--cf-text) !important;
    border: 1px solid var(--cf-border) !important;
    padding: 8px 10px !important;
  }
  tr:hover td { background: rgba(255,255,255,0.04) !important; }

  /* ===== Header brand ===== */
  .cf-brand-logo-wrap {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 8px;
    background: #000000 !important;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
  }

  .cf-brand-logo {
    display: block;
    background: #000000 !important;
  }

  /* ===== Scrollbars (Chromium) ===== */
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.16); border-radius: 999px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }
  ::-webkit-scrollbar-track { background: rgba(255,255,255,0.04); }

  /* ===== Alerts (PyWebIO put_success/put_info/put_warning/put_error) ===== */
  .alert{
    border-radius: var(--cf-radius) !important;
    border: 1px solid var(--cf-border) !important;
    color: var(--cf-text) !important;
    box-shadow: none !important;
  }
  .alert-success{
    background: rgba(34, 197, 94, 0.10) !important;
    border-color: rgba(34, 197, 94, 0.25) !important;
    color: rgba(235, 255, 245, 0.92) !important;
    font-weight: 700 !important;
  }
  .alert-info{
    background: rgba(59, 130, 246, 0.10) !important;
    border-color: rgba(59, 130, 246, 0.25) !important;
    color: rgba(235, 245, 255, 0.92) !important;
    font-weight: 650 !important;
  }
  .alert-warning{
    background: rgba(234, 179, 8, 0.12) !important;
    border-color: rgba(234, 179, 8, 0.28) !important;
    color: rgba(255, 250, 235, 0.92) !important;
    font-weight: 650 !important;
  }
  .alert-danger{
    background: rgba(239, 68, 68, 0.12) !important;
    border-color: rgba(239, 68, 68, 0.28) !important;
    color: rgba(255, 235, 235, 0.92) !important;
    font-weight: 650 !important;
  }
</style>
"""
    )


__all__ = ["_inject_dark_css"]
