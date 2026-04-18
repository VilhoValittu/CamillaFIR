# CamillaFIR Modes: AUTO vs BASIC vs ADVANCED (v4.2.1)

CamillaFIR has three operating modes:

- `AUTO`: automatic preset search and target selection
- `BASIC`: guarded manual workflow
- `ADVANCED`: expert manual workflow

The DSP engine is shared across all three. What changes is how much the program searches for you, which defaults are applied, and whether policy clamps are enforced at runtime.

## How modes are applied

- Startup defaults load `AUTO`.
- Selecting a mode does not immediately rewrite every field.
- The button `Apply mode defaults` writes that mode's defaults into the UI.
- BASIC clamps are re-applied at runtime for safety.
- AUTO and ADVANCED do not use BASIC-style hard mode clamps.

---

## AUTO (default startup mode)

Goal: let CamillaFIR search for a good preset automatically.

### What AUTO does

- Runs automatic preset search before final export.
- Can auto-select the target curve, or respect the selected target if `auto_target_mode=selected`.
- Reuses cache hits when the same measurements and relevant settings are seen again.
- Uses the current UI values as the search baseline and constraints, then reports the winning preset.

### AUTO defaults when applying mode defaults

- Filter type: `Mixed Phase`
- Correction band: `18-230 Hz`
- Max boost / cut: `+5 dB / -24 dB`
- Phase limit: `320 Hz`
- Filter smoothing: `1/48 oct` (`filter_smooth=48`)
- FDW cycles: `10`
- Regularization: `18 dB`
- TDC: ON (`15%`, max reduction `6 dB`, slope `12 dB/oct`)
- A-FDW: ON
- Bass-first AI: ON (`max 200 Hz`)
- Stereo link: ON by default
- Excursion protection: ON
- Low-bass cut: ON, `40 Hz`

Notes:
- Fresh config files currently start with `Asymmetric` as the default filter type before any mode-defaults button is applied.
- In AUTO, the final winner may differ from the visible seed values because the search is allowed to move around that baseline.

---

## BASIC

Goal: predictable manual results with hard guard rails.

### BASIC policy behavior

- `lvl_mode` is forced to `Auto` in pipeline/runtime.
- `stereo_link` is forced ON.
- `enable_tdc` is forced ON.
- `enable_afdw` is forced ON.
- `low_bass_cut_enable` is forced ON.
- `ir_export_window_mode` is forced to `auto`.
- Critical System Health issues block the run in BASIC.

### BASIC defaults

- Filter type: `Linear Phase`
- Correction band: `25-250 Hz`
- Max boost / cut: `+3 dB / -15 dB`
- Phase limit: `400 Hz`
- Filter smoothing: `1/12 oct` (`filter_smooth=12`)
- FDW cycles: `10`
- Regularization: `30 dB`
- Slope limits (global/boost/cut): `12 / 6 / 24 dB/oct`
- DF smoothing: ON
- TDC: ON (`50%`, max reduction `9 dB`, slope `6 dB/oct`)
- Bass-first AI: ON (`max 180 Hz`)
- Leveling: `Auto + Median`, window `500-2000 Hz`
- Excursion protection: ON
- Low-bass cut: ON, `50 Hz`
- IR windowing mode: `auto`
- IR windows (L/R): `80 ms / 500 ms`

### BASIC hard clamps

- `max_boost_db`: `0.0..4.0`
- `max_cut_db`: `0.0..15.0`
- `filter_smooth`: `1..24`
- `reg_strength`: `10.0..60.0`
- `fdw_cycles`: `10.0..15.0`
- `phase_limit`: `200.0..450.0 Hz`
- `mag_c_min`, `mag_c_max`: `18.0..300.0 Hz`
- `tdc_strength`: `0.0..70.0`
- `tdc_max_reduction_db`: `0.0..12.0`
- `tdc_slope_db_per_oct`: `0.0..12.0`
- `mixed_split_freq`: `100.0..200.0 Hz`
- `low_bass_cut_hz`: `20.0..100.0 Hz`
- Forced booleans: `enable_tdc=True`, `enable_afdw=True`, `low_bass_cut_enable=True`, `stereo_link=True`
- Forced mode: `ir_export_window_mode='auto'`

---

## ADVANCED

Goal: manual expert workflow with fewer policy constraints.

### ADVANCED policy behavior

- No BASIC-style mode clamps are applied by policy.
- Critical System Health issues are warned, but not blocked by mode policy.
- Advanced-only controls remain available.

### ADVANCED defaults

- Filter type: `Mixed Phase`
- Correction band: `18-230 Hz`
- Max boost / cut: `+5 dB / -24 dB`
- Phase limit: `320 Hz`
- Filter smoothing: `1/48 oct` (`filter_smooth=48`)
- FDW cycles: `10`
- Regularization: `18 dB`
- Slope limits (global/boost/cut): `24 / 36 / 0 dB/oct`
- DF smoothing: OFF
- TDC: ON (`15%`, max reduction `6 dB`, slope `12 dB/oct`)
- A-FDW: ON
- Bass-first AI: ON (`max 200 Hz`)
- Leveling: `Auto + Median`, window `200-3000 Hz`
- Stereo link: ON by default
- Excursion protection: ON
- Low-bass cut: ON, `40 Hz`
- IR windowing mode: `auto`

---

## Additional runtime notes

- Auto-align is always forced ON by pipeline policy.
- Max boost is globally safety-capped to `MAX_SAFE_BOOST` (currently `8.0 dB`) in all modes.
- In the UI, IR windowing choices are limited to `auto` and `rew_asym`; `rew_asym` is available only when allowed by filter type and mode rules.

## Implementation reference

- `src/camillafir/ui/camillafir_modes.py`
- `src/camillafir/config/camillafir_pipeline.py`
- `src/camillafir/ui/system_health.py`
- `src/camillafir/camillafir.py`

### Disclaimer
AI was used to translate this document from Finnish to English.
