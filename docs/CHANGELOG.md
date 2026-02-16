# Changelog

All notable changes to **CamillaFIR** are documented in this file.

---

## [3.0.3] - 2026-02-16

### UI
- Added warnings if correction range is too wide

### macOS
- Switched to universal2 build (Intel + Apple Silicon support)
- Fixes "Bad CPU type in executable" on older Intel Macs

---

## [3.0.2] - 2026-02-15

### UI
- Simplified and reorganized UI structure for clearer workflow
- Added real-time house curve preview
- Added project logo to the interface

---

## [3.0.1] - 2026-02-14

### DSP
- GD-gradient limiter redesigned
  - Bass-focused (20–250 Hz)
  - Soft limiting (tanh) replaces hard clipping
  - Relaxed slope limit: 8 → 30 ms/oct
  - Conditionally enabled (bypassed when A-FDW + Bass-first stabilization are active, except in high-risk windowing modes)

### UI
- Fixed A-FDW plot rendering issue with measurement sample rates >48 kHz

---

## [3.0.0] - 2026-02-11

Major DSP engine update and smoothing redesign.

### UI
- Renamed *Psychoacoustic* smoothing to **CamillaFIR Reference**
  - Clarifies that this smoothing is a CamillaFIR-specific reference/safety view
  - Not equivalent to REW-style psychoacoustic smoothing
  - View-only (does not affect DSP calculation)

- Added user-configurable parameters for **Confidence Pull**
  - Available in Advanced mode only
  - Floor threshold
  - Max frequency limit
  - Cut aggressiveness
  - Boost conservativeness
  - Recommended defaults included in help text

### DSP
- Replaced legacy fixed 1/24 octave smoothing in filter calculation
  - New adaptive smoothing:
    - 0–230 Hz: 1/48 octave
    - 230–500 Hz: gradual transition
    - >500 Hz: 1/3 octave
  - Improves LF precision while stabilizing mid/high frequency correction

- Improved Bass-first behavior
  - Better alignment with confidence weighting
  - More predictable low-frequency correction shaping

- Refined confidence pull handling
  - More stable behavior in low-confidence regions
  - Reduced over-aggressive boosts in uncertain bands

### Build
- Linux build switched to `onedir` distribution
  - Improves portability and runtime reliability
  - Avoids common single-file extraction issues

### Behavior changes
- Adjusted default Confidence Floor (0.15 → 0.07)
  - Correction operates more freely before safety pull engages

---

## [2.9.5] - 2026-02-08

### UI
- **Low-bass cut is now toggleable (ON/OFF)**
  - Added an enable checkbox for `low_bass_cut_hz`.
  - When disabled, the Hz input remains visible but is **greyed out (locked)** and cannot be edited.
  - Disabled state uses an **empty value** for the cutoff field to represent “off” (instead of `None`),
    improving UI → pipeline compatibility.
  - UI logic lives in a dedicated helper (`update_low_bass_cut_ui`) and renders inside a scope for clean updates.

---

## [2.9.4] - 2026-02-08

### CFG
-  Fixed low_bass_cut_hz value not saving correctly in config.

---

## [2.9.3] - 2026-02-08

### UI
-  Fixed typo at psychoacoustic plot smoothing code (1/48 / 1/3 ---> 1/6 / 1/3)

---

## [2.9.2] - 2026-02-07

### UI
- **IR windowing hard restrictions clarified and enforced**
  - In **Basic mode**, IR export windowing is now **always forced to Auto**.
    - The windowing mode selector is locked and cannot be changed.
    - A persistent warning message is shown to explain the restriction.
  - When **Filter type = Asymmetric**, IR export windowing is also **locked to Auto**
    in both Basic and Advanced modes.
  - Prevents confusing transient UI states where windowing controls appeared
    editable even though the value was internally forced back to Auto.
  - Ensures UI behavior always matches DSP policy and exported configuration.

---

## [2.9.1] - 2026-02-06

### UI
- **Mixed Phase crossover frequency is now state-dependent**
  - The “Mixed Phase crossover frequency (Hz)” field is active **only** when **Filter type = Mixed Phase**.
  - For other filter types, the field remains visible but is **greyed out (locked)** to prevent invalid configurations.
- **REW Asymmetric IR windowing restricted to Linear Phase filters**
  - The “Asymmetric” IR windowing option (REW Asymmetric export) is selectable **only** when **Filter type = Linear Phase**.
  - For other filter types, the option is shown but **greyed out**, and the UI displays a warning message:
    - `WORKS ONLY WITH LINEAR PHASE FILTERS`

---

## [2.9.0] - 2026-02-06

### UI
- **Windowing mode simplification**
  - Removed **“Symmetric”** and **“Off”** windowing modes.
  - Windowing now offers only:
    - **Auto** – REW-based, adaptive windowing selected automatically from the impulse response.
    - **Asymmetric** – REW-based asymmetric windowing with optional latency reduction.
  - Simplifies the UI and focuses on the most effective and reliable windowing strategies.

### DSP
- **High-pass filter (HPF) magnitude fix**
  - Fixed HPF handling so it is applied as a **true magnitude filter** in the FIR path.
  - HPF is now applied directly to the correction curve (**`gain_db += hpf_db`**),
    instead of being baked into the target response.
  - Ensures **magnitude and phase consistency**.
  - Prevents double-HPF behavior, incorrect low-frequency response,
    and artificial group delay artifacts when HPF is enabled.


---

## [2.8.9] - 2026-02-05

### DSP
- **REW Asymmetric low-latency bass safety**
  - Added automatic safety guards for ultra–low-latency REW Asymmetric mode.
  - When latency target (Left window) is **below 15 ms**:
    - Bass-first (A-FDW confidence shaping) is automatically limited to low frequencies.
  - When latency target is **below 10 ms**:
    - Low-frequency **boosts are disabled** (cuts remain allowed).
  - Prevents unstable bass behavior, excessive ripple, and aggressive FIR boosts
    when time-domain constraints become too tight.
  - Safeguards are automatic, non-configurable, and only active in REW Asymmetric mode.


---

## [2.8.8] - 2026-02-04

### DSP
- **Phase correction safety clamp (±45°)**
  - Room / excess-phase correction is now internally limited to ±45 degrees.
  - The clamp is applied **only to the correction component** (measured − target phase),
    never to loudspeaker minimum-phase or theoretical crossover phase.
  - Prevents excessive phase rotations, pre-ringing, and unstable group delay behavior,
    especially in low-confidence or sparsely measured regions.
  - Improves robustness, repeatability, and subjective transient clarity.
  - No user-facing control; this is a fixed safety default.

### Analysis & Reporting
- Phase correction clamp status is now always reported:
  - Logged during processing (e.g. `max=54.5° -> 45.0°`).
  - Included in `summary.txt` per channel.
  - Shown in **DSP info** section in the UI.

### UI
- Removed slope-limit envelope visualization from magnitude plots.
  - Eliminates confusing shaded artifacts without affecting DSP behavior.

### Notes
- No changes to magnitude targets, A-FDW, TDC, leveling, or IR export behavior.
- Existing presets and workflows remain fully compatible.

---

## [2.8.7] - 2026-02-04

### Fixed
- Psychoacoustic smoothing corrected.
- IR windowing no longer affects filter level.

### UI
- Updated UI text strings and help descriptions.
- Little update to interface look
---

## [2.8.6] - 2026-02-01
 
 ### Changed
- Refactored DSP-related code into a clearer, more modular file structure.
- Separated DSP logic from UI and orchestration layers to improve maintainability.
- Clarified responsibility boundaries between filter generation, windowing, and export logic.
- No functional changes to DSP algorithms or generated filters.

### Internal
- DSP modules are now organized to allow easier future extensions.
- Reduced implicit cross-dependencies between DSP and UI code.
- Improved long-term stability by making DSP behavior less sensitive to UI-side changes.

---

## [2.8.5] - 2026-02-01

### DSP / IR export
- **IR export window edge shape selection (Hann / Tukey)**
  - Added support for selecting IR window edge shape during FIR export.
  - Tukey window includes adjustable alpha parameter (0–1, default 0.25).
  - Window shape is applied only at IR export stage (WAV generation).

- **REW-style asymmetric export placement fix**
  - When `rew_asym` is selected, FIR impulse peak is placed causally
    (early in the impulse) instead of remaining centered.
  - Reduces effective playback latency compared to symmetric placement.

### UI / Pipeline
- **IR window shape & alpha preserved through UI → pipeline → DSP**
  - Prevents silent fallback to legacy Hann window.
  - Selected window shape and alpha are logged immediately after UI collection
    for traceability.

### Notes
- No changes to FIR magnitude targets, phase correction algorithms,
  A-FDW, TDC, or auto-leveling behavior.
- Differences between Hann and Tukey exports are sample-accurate
  and produce non-identical FIR WAV files.

---

## [2.8.4] - 2026-02-01

### Misc
- Github actions now makes running files.

---

## [2.8.3] - 2026-01-31

### DSP update
- **REW-style IR windowing enabled in DSP export path**
  - Adds support for REW-compatible symmetric and asymmetric IR windowing during FIR export.
  - Windowing is applied **only at IR export stage** (WAV generation), not during correction, target fitting, leveling, or scoring.
  - Supported modes:
    - `auto` – automatic window selection (default)
    - `off` – no IR windowing
    - `rew_sym` – REW-style symmetric window
    - `rew_asym` – REW-style asymmetric (causal) window

### IO update
- **IR windowing type is now included in exported filenames**
  - ZIP and FIR WAV filenames include a short windowing tag for traceability and A/B comparisons.
  - Tags: `auto`, `off`, `sym`, `asym`
  - Example: `CamillaFIR_<type>_sym_<timestamp>.zip`
  - Example: `L_<type>_<fs>Hz_<timestamp>_sym.wav`

### Notes
- No change to correction targets, phase modes, FDW, TDC, or leveling behavior.
- This update improves reproducibility and comparability between different IR export strategies.

---

## [2.8.2.3] - 2026-01-30

### Io-update
- Fixed ZIP output when multi-rate is enabled. Generate a single CamillaDSP .yml using $samplerate$

### DSP-update
- More precise leveling tilt used in magnitude calculation

---

## [2.8.2.2] - 2026-01-29

### Ui-update
- updated translations and phase plot to be more clear

---

## [2.8.2.1] - 2026-01-28

## Nothing changed DSP or UI
- changed file structure to more debug-friendly format
- filters go now to ./filters directory

---

## [2.8.2] - 2026-01-27

### Ui-update
- improved robustness of file upload parsing from browser & added xo_help translation

---

## [v2.8.1.2] - 2026-01-27   

### Fixed
- Bug fix for modes selection, that was not saving ui state correctly

---

## [2.8.1.1] - 2026-01-27

### Ui-update
- Added modes selection (Basic & Advanced)

---

## [2.8.1] – 2026-01-25

### Fixed
- **A-FDW bandwidth limits**
  - Corrected incorrect or overly permissive A-FDW bandwidth constraints.
  - Prevents misleading smoothing widths and improves consistency between analysis and visualization.

### Improved
- **A-FDW & TDC guides**
  - Plot guides and annotations now reflect the *effective* (clamped) A-FDW bandwidth.
  - Improves interpretability of confidence masking and decay-based correction limits.

### Notes
- No changes to FIR magnitude targets, phase correction modes, or leveling behavior.
- Maintenance update focused on analysis clarity and safety transparency.

---

## [2.8.0] - 2026-01-24

- **Plot export robustness (ZIP outputs)**
  - Fixed a broken Plotly PNG export path caused by an invalid `try/except` structure.
  - ZIP exports now store dashboard plots as static PNG images generated via Plotly’s native Kaleido backend.
  - Ensures exported plots exactly match the HTML dashboard “Download plot as PNG” output.
  - Eliminates dependency on `.html` files and local `plotly.min.js` for offline ZIP viewing.

---

## [2.7.9] – 2026-01-24

### Fixed
- **Custom house curve upload**
  - Fixed an issue where user-uploaded house curves could fail to load or apply correctly.
  - Improves validation and consistency between UI preview and DSP processing.

### Notes
- No changes to FIR magnitude, phase, or leveling behavior.
- Safe update focused on UI → DSP data integrity.

---


## [2.7.8] – 2026-01-23

### Added
- **Stereo-linked auto-leveling (TXT-compatible default)**
  - SmartScan level window and gain are computed from a shared L/R reference and applied identically to both channels.
  - Eliminates channel-dependent gain drift while preserving automatic delay alignment.

- **Correction-band visualization**
  - Active magnitude correction range (`mag_c_min … mag_c_max`) is now explicitly carried through DSP stats and visualized in plots.
  - Makes it immediately clear where correction is applied and where it is intentionally inactive.

- **Reliability / confidence visualization**
  - Low-confidence frequency regions are visually shaded in plots.
  - Helps explain why certain bands are protected or only lightly corrected (measurement reliability, A-FDW behavior).

### Changed
- **Auto-leveling behavior (default)**
  - Stereo leveling now uses a single shared window and offset instead of independent per-channel SmartScan decisions.
  - Results are deterministic and TXT-compatible by default.

- **Summary.txt clarity**
  - Level window and offset method explicitly indicate stereo-linked operation
    (e.g. `ForcedOffset (StereoLink)`).

### Fixed
- **Auto-align gain drift**
  - Fixed cases where left/right channels could diverge by several dB due to independent leveling window selection.

### Notes
- Auto-align delay estimation is unchanged and remains fully automatic.
- FIR magnitude and phase are unaffected by alignment-only time shifts.

---

## [2.7.7] – 2026-01-20

### Added
- **2058-safe phase mode**
  - Disables room phase correction (confidence/FDW/excess-phase) and uses only theoretical crossover phase and minimum-phase where applicable.

- **Independent slope limits for boost vs cut**
  - Separate dB/oct limits prevent gentle boosts from being flattened while still constraining aggressive cuts.

- **TDC safety brakes**
  - Hard cap on total Temporal Decay Control reduction.
  - Optional slope limit for predictable, stable decay shaping.

- **DF smoothing (experimental)**
  - Gaussian smoothing with approximately constant Hz width across different sample rates and tap counts.

- **Comparison mode**
  - Locks scoring and plots to a fixed analysis grid (fs/taps) for meaningful A/B comparisons.

- **Multi-rate auto-taps mapping**
  - Maintains constant FIR time length across sample rates (44.1 kHz reference).

### Changed
- Refactored leveling logic into a dedicated module for robustness and testability.
- Improved guard logic against unstable phase and excessive corrections.

---

## [2.7.6] and earlier

- Initial public releases and iterative improvements to TDC, confidence masking,
  and the FIR generation pipeline.
- See commit history for detailed technical changes.
