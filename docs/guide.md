---
layout: default
title: "How to Create FIR Filters from REW Measurements - Current Guide"
description: "Up-to-date step-by-step CamillaFIR guide for creating FIR convolution filters from REW measurements."
permalink: /guide/
---

# CamillaFIR Guide (Current Workflow)

This guide reflects the current CamillaFIR workflow (v3.3.0 generation): from REW measurement data to convolution filters for CamillaDSP, Roon, and Equalizer APO.

## Quick workflow

1. Measure Left and Right in REW.
2. Export REW data (TXT with magnitude+phase, or WAV/IR workflow).
3. Open CamillaFIR and choose mode (`BASIC` recommended, `ADVANCED` for expert control).
4. Apply mode defaults, then set correction band and safety limits.
5. Generate filters and export ZIP package.
6. Load WAV filters into your convolution engine.
7. Re-measure and validate.

---

## 1. Prepare measurements in REW

- Load the correct microphone calibration file in REW before measuring.
- Measure Left and Right channels separately.
- Keep measurement procedure and timing reference consistent between channels.
- Avoid clipping and bad SNR.
- For TXT export, include Frequency + Magnitude + Phase (do not omit phase).
- If you use WAV/IR import workflow, use REW export settings: `Mono`, `float32`, `Normalise`, `Place t=0 (256)`.
- Header/comment lines are supported (`*`, `#`, `;` are ignored).

Tip: good input data matters more than aggressive correction.

---

## 2. Import into CamillaFIR and select mode

CamillaFIR has two modes:

- `BASIC` (recommended): strong safety rails and runtime clamps.
- `ADVANCED`: fewer policy limits for expert workflows.

Important:

- Selecting mode alone does not rewrite all values.
- Use `Apply mode defaults` to load mode-specific defaults into UI.

BASIC defaults (high level):
- Correction band: `25-250 Hz`
- Max boost/cut: `+3 dB / -15 dB`
- TDC: ON
- A-FDW: ON
- Stereo link: ON

ADVANCED defaults (high level):
- Correction band: `10-200 Hz`
- Max boost/cut: `+3 dB / -30 dB`
- More manual freedom

---

## 3. Choose filter type and correction range

Filter types:

- `Linear Phase`: best phase linearity, highest latency.
- `Minimum Phase`: lower latency, no linear-phase target.
- `Mixed Phase`: blend of linear and minimum behavior.
- `Asymmetric Linear`: low-latency linear-phase strategy with asymmetric windowing.

Practical starting point:
- Keep correction mostly in bass/lower mids unless measurement confidence is very high.
- Use conservative max boost (often `+3 dB` is enough).
- CamillaFIR also applies a global safety cap (`8 dB` maximum effective boost).

---

## 4. Use safety features on purpose

Key protections:

- Max boost/cut limits
- Slope limits (global and optional split boost/cut)
- Excursion protection
- Low-bass boost lock
- Optional HPF (true FIR magnitude HPF in correction path)
- TDC (Temporal Decay Control) for ringing/decay management
- A-FDW for confidence-aware behavior in difficult regions

Do not try to fix deep nulls with heavy boost. Placement, crossover work, or room treatment is usually more effective.

---

## 5. Generate and export filters

Export creates a ZIP package in:

`filters/` (project root)

Typical package contents:
- L/R FIR WAV files (`32-bit float`)
- `Summary_...txt` report
- CamillaDSP config snippet (`.cfg`)
- CamillaDSP `.yml` (single-rate export, or multi-rate variant)
- Dashboard PNG files (or TXT fallback if PNG is unavailable)

Multi-rate export targets:
- `44.1 / 48 / 88.2 / 96 / 176.4 / 192 kHz`

---

## 6. Load filters into your DSP

- CamillaDSP: load L/R WAV files into convolution pipeline (optionally use generated YAML).
- Roon: load FIR WAV in Convolution settings.
- Equalizer APO: use Convolution module and set safe preamp/headroom.

---

## 7. Verify after applying filters

Always re-measure with filter active:

- Confirm target match improved.
- Check headroom and clipping risk.
- Review `Summary.txt` for confidence, decay, and clamp diagnostics.
- Adjust correction range/boost if result sounds or measures over-corrected.

---

## Common mistakes to avoid

- Overboosting bass/nulls.
- Correcting too wide a band with low-confidence data.
- Ignoring latency implications of filter type.
- Skipping post-filter verification measurement.

---

## Related docs

- `docs/Official_Manual.md`
- `docs/Modes.md`
- `docs/CamillaFIR_Reading_Output_Guide.md`

## Download

https://github.com/VilhoValittu/CamillaFIR/releases

### Disclaimer
AI was used to translate this document from Finnish to English.
