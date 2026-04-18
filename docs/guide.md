---
layout: default
title: "How to Create FIR Filters from Measurements - Recommended AUTO Workflow"
description: "Up-to-date step-by-step CamillaFIR guide for creating FIR convolution filters from REW exports, built-in sweep measurements, or WAV/IR captures with the recommended automatic workflow."
permalink: /guide/
---

# CamillaFIR Guide (Recommended AUTO Workflow)

This guide reflects the current CamillaFIR workflow in v4.2.1: measure in REW or with the built-in measurement tool, keep `AUTO` mode, let CamillaFIR search for a good preset, export FIR filters, and verify with a new measurement.

## Quick workflow

1. Measure Left and Right in REW, or use the built-in measurement tool.
2. Export REW data as TXT with magnitude + phase, use the WAV/IR workflow, or load the saved built-in measurement IR files.
3. Open CamillaFIR and keep `AUTO` mode.
4. Leave the filter type at `Asymmetric` unless you have a specific reason to use another type.
5. Choose an AUTO goal (`balanced` for most rooms, `subwoofers` for sub-only work), then set optional target and HPF preferences.
6. Press `START`, review the winning preset, and export the ZIP package.
7. Load the WAV filters into your convolution engine.
8. Re-measure and validate.

---

## 1. Prepare measurements

If you use the built-in measurement tool, save the generated IR files first and then continue from the import step below. The detailed points in this section are mainly for the REW path.

- Load the correct microphone calibration file in REW before measuring.
- Measure Left and Right channels separately.
- Keep measurement procedure and timing reference consistent between channels.
- Avoid clipping and bad SNR.
- For TXT export, include Frequency + Magnitude + Phase.
- If you use the WAV/IR import workflow, use REW export settings: `Mono`, `float32`, `Normalise`, `Place t=0 (256)`.
- Header/comment lines are supported (`*`, `#`, `;` are ignored).

Tip: good input data matters more than aggressive correction.

---

## 2. Import into CamillaFIR and keep AUTO mode

CamillaFIR has three operating modes:

- `AUTO`: recommended for most users; automatic target selection and preset search.
- `BASIC`: guarded manual workflow.
- `ADVANCED`: expert manual workflow with fewer policy limits.

Why `AUTO` is the recommended starting point:

- It can auto-select a suitable built-in target curve if you do not lock in your own target.
- It searches multiple candidate presets and applies the best-ranked winner before export.
- In the current v4.2.1 workflow it can also use harmonic curves and IACC-aware ranking to avoid overly aggressive or overly symmetric winners.
- It reuses cache hits when the same measurements and relevant settings are seen again.
- It exports summary metadata about the winning preset, target choice, and run details.

Important:

- Selecting a mode alone does not rewrite all visible values.
- Use `Apply mode defaults` only when you want to reset the current UI values to that mode's baseline.
- Fresh configs start in `AUTO`.
- Fresh configs also start with `Asymmetric` as the default filter type, which is a good general-purpose choice.

Current AUTO goals:

- `balanced`: recommended default for most rooms.
- `room-safe`: more conservative in difficult rooms.
- `low-ripple`: prioritizes smoother bass behavior around room modes.
- `flat`: prioritizes the flattest measured result.
- `subwoofers`: subwoofer-focused AUTO goal that forces Smart Scan leveling to `20-200 Hz`.

---

## 3. Set the search constraints, not every value manually

In `AUTO`, the visible UI values act as the search baseline and constraints. You usually do not need to tune every field by hand before running.

Practical starting point:

- Filter type: `Asymmetric` for most systems.
- Goal: `balanced` unless you are doing sub-only work or solving a clearly difficult room.
- Max boost: keep it conservative, usually `+3 dB` to `+5 dB`.
- Correction range: keep correction mostly in bass/lower mids unless your measurement quality is very high.
- Target curve: let AUTO choose from built-in targets, or explicitly select your own preferred target.
 - Max boost: In `AUTO` the visible max-boost control does not directly determine the applied boost; `AUTO` enforces its own safe boost limits during the search and aims to keep boosts conservative.
 - Correction range: keep correction mostly in bass/lower mids unless your measurement quality is very high.
 - Target curve: let AUTO choose from built-in targets, or explicitly select your own preferred target.
 - HPF: optional in `AUTO`. If you enable HPF in `AUTO`, the program will select optimal HPF frequency and slope automatically; the UI enable/disable flag is respected but frequency/slope are determined by `AUTO`.

Filter type guidance:

- `Asymmetric`: recommended default and best general balance of correction quality, stability, and latency.
- `Linear Phase`: use if maximum linear-phase behavior matters more than latency.
- `Minimum Phase`: use for lower-latency causal correction.
- `Mixed Phase`: use if you specifically want mixed-phase behavior.

---

## 4. Use safety features on purpose

CamillaFIR still relies on safety limits even when `AUTO` is doing the preset search.

Key protections:

- Max boost/cut limits
- Global effective boost cap (`8 dB` maximum safe boost)
- Slope limits
- Excursion protection
- Low-bass cut
- Optional HPF
- TDC (Temporal Decay Control)
- A-FDW
- Stereo link

During automatic runs, CamillaFIR may manage or lock some controls to keep the search valid and safe.

Do not try to fix deep nulls with heavy boost. Placement, crossover work, or room treatment is usually more effective.

---

## 5. Run AUTO, then export filters

Press `START` to begin the automatic workflow.

What happens in practice:

- CamillaFIR may evaluate built-in target curves first if you have not locked a target.
- It runs automatic preset search and refinement passes.
- It applies the best-ranked winner to the final result view and export bundle.
- Repeated runs with the same measurements and key settings may reuse cache data and finish faster.

Export creates a ZIP package in the default export folder:

`Documents/CamillaFIR/filters/<version>/`

If that location is not writable, CamillaFIR uses a safe fallback path and shows the final path in the Results view.

Typical package contents:

- L/R FIR WAV files (`32-bit float`)
- `Summary_...txt` report with AUTO-mode metadata
- CamillaDSP config snippet (`.cfg`)
- CamillaDSP `.yml` (single-rate export, or multi-rate variant)
- Dashboard PNG files (or TXT fallback if PNG is unavailable)

Multi-rate export targets:

- `44.1 / 48 / 88.2 / 96 / 176.4 / 192 kHz`

---

## 6. Load filters into your DSP

- CamillaDSP: load L/R WAV files into the convolution pipeline, or use the generated YAML.
- Roon: load FIR WAV in Convolution settings.
- Equalizer APO: use the Convolution module and keep enough preamp/headroom.

---

## 7. Verify after applying filters

Always re-measure with the filter active:

- Confirm target match improved.
- Check headroom and clipping risk.
- Review `Summary.txt` for winner details, confidence, decay, and clamp diagnostics.
- Adjust goal, correction range, or boost limit if the result sounds or measures over-corrected.

---

## Common mistakes to avoid

- Switching to `BASIC` by habit when `AUTO` would be the better starting point.
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
