# CamillaFIR Modes: BASIC vs ADVANCED (v3.2.0)

CamillaFIR has two operating modes.
The DSP engine is the same in both modes. Modes change:
- recommended defaults
- runtime policy constraints (BASIC only)
- some UI availability and lock behavior

## How modes are applied

- Selecting a mode does not automatically rewrite all fields.
- The button `Apply mode defaults` writes that mode's defaults into UI values.
- During processing, BASIC clamps are re-applied at runtime for safety.
- If mode is missing or invalid, fallback is `BASIC`.

---

## BASIC (recommended)

Goal: predictable results with hard guard rails.

### BASIC policy behavior
- `lvl_mode` is forced to `Auto` in pipeline/runtime.
- `stereo_link` is forced ON.
- `enable_tdc` is forced ON.
- `enable_afdw` is forced ON.
- `low_bass_cut_enable` is forced ON.
- `ir_export_window_mode` is forced to `auto`.
- Critical System Health issues block run in BASIC.

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
- Low-bass boost lock: ON, `50 Hz`
- IR windowing mode: `auto`
- IR windows (L/R): `85 ms / 500 ms`

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
- `low_bass_cut_hz`: `20.0..100.0 Hz`
- Forced booleans: `enable_tdc=True`, `enable_afdw=True`, `low_bass_cut_enable=True`, `stereo_link=True`
- Forced mode: `ir_export_window_mode='auto'`

---

## ADVANCED (expert)

Goal: fewer policy constraints.

### ADVANCED policy behavior
- No mode clamps are applied by policy.
- Critical System Health issues are warned, but not blocked by mode policy.
- Advanced-only controls (for example Confidence Pull section) are available.

### ADVANCED defaults
- Filter type: `Linear Phase`
- Correction band: `10-200 Hz`
- Max boost / cut: `+3 dB / -30 dB`
- Phase limit: `400 Hz`
- Filter smoothing: `1/24 oct` (`filter_smooth=24`)
- FDW cycles: `10`
- Regularization: `30 dB`
- Slope limits (global/boost/cut): `24 / 0 / 0 dB/oct` (`0` = off for split limits)
- DF smoothing: OFF
- TDC: ON (`50%`, max reduction `12 dB`, slope `6 dB/oct`)
- Bass-first AI: ON (`max 200 Hz`)
- Leveling default: `Auto + Median`, window `200-3000 Hz`
- Excursion protection: OFF
- Low-bass boost lock: ON, `20 Hz`
- Stereo link: ON by default (not forced)
- IR windowing mode: `auto` default (not forced by mode policy)

---

## Additional runtime notes

- Auto-align is always forced ON by pipeline policy.
- Max boost is globally safety-capped to `MAX_SAFE_BOOST` (currently `8.0 dB`) in all modes.
- In UI, IR windowing choices are limited to `auto` and `rew_asym`; `rew_asym` is available only when allowed by filter type/mode rules.

## Implementation reference

- `src/camillafir/ui/camillafir_modes.py` (mode defaults and clamps)
- `src/camillafir/config/camillafir_pipeline.py` (BASIC leveling and pipeline policy)
- `src/camillafir/ui/system_health.py` (BASIC health-gate blocking behavior)
- `src/camillafir/camillafir.py` (runtime clamp reapply + global max boost safety cap)

### Disclaimer
AI was used to translate this document from Finnish to English.
