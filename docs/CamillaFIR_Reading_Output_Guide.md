# Reading CamillaFIR Output (Current Behavior)

This guide reflects the current code path in CamillaFIR (v3.1.x line).

## 1. What you get after a run

CamillaFIR produces:

1. Interactive dashboards in the UI (Left/Right analysis view).
2. A ZIP package saved to `filters/`:
   - FIR WAV files (`L_...wav`, `R_...wav`)
   - Summary report (`Summary_<type>_<fs>Hz.txt`)
   - Convolver configs (`Config_...cfg`, `camilladsp_...yml`)

Important current detail:
- ZIP dashboard PNG export is currently disabled by default.
- TXT FIR export is not part of the current ZIP export path.

## 2. Reading the dashboard (UI)

The UI dashboard has 5 panels:

### 2.1 Magnitude and Alignment

Shows:
- Measured (blue)
- Target (green dashed)
- Predicted (orange)
- Confidence line (magenta)
- Level reference line and smart scan window shading
- Correction range shading (`mag_c_min` to `mag_c_max`)

Read it like this:
- Predicted should track target without narrow high-Q spikes.
- Red-tinted bands indicate low confidence regions.
- Strong LF lift means headroom risk; check Summary headroom section.

### 2.2 Phase

Shows wrapped phase of the predicted response.
Use it mainly to see whether behavior looks smooth around crossover and correction limits.

### 2.3 Group Delay

Highlights delay structure over frequency.
- Broad LF structures often relate to room modes.
- Sharp local spikes can indicate reflection-related behavior.

### 2.4 Filter (dB)

This is the FIR gain curve after level-compensated plotting.
Use it to confirm boost/cut aggressiveness and low-frequency policy behavior.

### 2.5 A-FDW Effective BW (oct)

Shows effective adaptive bandwidth when available.
If unavailable, the panel displays "No A-FDW BW data (...)". This is expected in some modes.

## 3. Reading `Summary_...txt`

The summary is layered. Main blocks:

### 3.1 Executive Summary

Includes:
- Acoustic Score (L/R)
- Target Match (percentage + RMS dB)
- Confidence (L/R)
- RT60 wideband (L/R)
- Worst detected event (L/R)

### 3.2 Core Settings and Analysis Mode

Includes:
- Key settings used for generation
- Analysis mode (`native` or `comparison`)
- Comparison grid metadata when comparison mode is active

### 3.3 Guards and control blocks

Includes:
- Correction Guards (max cut, slope limits, low-bass cut policy)
- Temporal Decay Control (TDC) status and limits
- A-FDW status and effective bandwidth stats
- XO and phase clamp lines

### 3.4 RT60 and matching blocks

Includes:
- RT60 wideband and optional per-band values
- Confidence and leveling window info
- Target curve match lines for left/right

### 3.5 Alignment and gain staging

Includes:
- Peak (pre-norm), global offset, auto gain margin, applied auto gain
- Additional sections appended during export:
  - `DSP EFFECTIVE PARAMS`
  - `LEVELING`
  - `HEADROOM MANAGEMENT`
  - `BOOST/CUT DIAGNOSTICS`
  - `CLAMP DIAGNOSTICS`
  - `STAGE CHECKPOINTS`
  - `BASS-FIRST AI`
  - `AUTO-ALIGN`
  - `ACOUSTIC EVENTS` (when detected)

## 4. Practical pass/fail checks

Use these first:

1. `Target Match` improves clearly versus uncorrected.
2. `Net boost peak (post global/headroom)` is not positive or only slightly positive.
3. `hard_clamp`/`soft_clip` counters are not excessive.
4. No unexpected large tilt in `LEVELING`.
5. `Final Max (filter+auto_gain+headroom)` stays sensible for your playback headroom.

## 5. If the result sounds wrong

- Too little bass:
  - Check `low_bass_cut_hz`
  - Check effective max boost and blocked boost reason
  - Check correction range upper/lower bounds
- Too bright or sharp:
  - Reduce correction range in upper mids/highs
  - Increase smoothing / reduce aggressive slope behavior
- Unstable imaging or odd timing:
  - Check `AUTO-ALIGN` delay and gain difference
  - Re-verify measurement consistency

Use re-measurement with filter enabled as the final validation.
