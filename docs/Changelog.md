# Changelog

All notable changes to **CamillaFIR** are documented in this file.

The format loosely follows *Keep a Changelog*, with a focus on user-visible DSP behavior and reproducibility.

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
