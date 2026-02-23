# Stability and Reproducibility (v3.1.1.2)

CamillaFIR is designed to avoid the classic failure mode:

**small measurement or setting changes -> large unstable filter changes**

The system stays predictable because it is deterministic, bounded, and observable.

## 1) Deterministic pipeline and versioned behavior

Given the same:
- CamillaFIR version
- input files
- settings
- base fs/taps or comparison-grid settings

CamillaFIR produces repeatable outputs.

The pipeline does not use stochastic optimization or random restarts.
Release `v3.1.1.2` (2026-02-22) keeps this deterministic model and adds stronger startup validation in CI.

## 2) Input robustness: TXT and WAV/IR

Modern workflows mix REW text exports and WAV/IR measurements.

Stability improvements include:
- deterministic WAV parsing policy aligned to TXT baseline behavior
- path sanitation and local WAV existence validation
- higher minimum FFT robustness for WAV IR -> FR conversion
- alignment guard between delay estimate and impulse-peak alignment

This reduces source-format-dependent surprises.

## 3) Bounded correction prevents inverse-filter runaway

CamillaFIR keeps each stage bounded:

Magnitude bounds:
- `max_boost_db`, `max_cut_db`
- low-bass safety controls
- smoothing and regularization (`filter_smooth`, `reg_strength`)

Shape bounds:
- slope limits (`max_slope_db_per_oct`)
- optional asymmetric slope limits for boost vs cut

Decay bounds:
- TDC strength and max reduction limits
- optional TDC slope limit

When limits are active, behavior remains controlled instead of extreme.

## 4) Confidence-guided correction improves repeatability

Room data confidence is frequency-dependent.
CamillaFIR uses confidence-aware behavior to avoid overreacting to artifacts:
- Confidence Pull logic
- A-FDW support
- smoothing that favors robust trends over narrow features

Practical effect: small mic-position changes usually alter details, not the whole correction profile.

## 5) Phase-domain safeguards

Phase processing is constrained by design:
- `phase_limit` bounds correction bandwidth
- Mixed-phase excess correction is faded from LF to HF
- Mixed-only limits cap excess delay and pre-ringing risk
- adaptive excess-phase clamping and conditional GD spike guards improve robustness

These controls prevent unstable phase behavior and reduce ringing risk.

## 6) Headroom reproducibility and stereo consistency

Level handling is deterministic and tied to realized correction:
- auto-headroom model computes attenuation from realized max boost + margin
- shared auto-gain behavior is available for stereo-linked runs
- plot-only compensation keeps analysis visuals readable while preserving output safety

This keeps output gain decisions consistent across repeated runs.

## 7) Comparable evaluation across fs/taps

Changing fs/taps can change analysis grids even when audible behavior is similar.
CamillaFIR addresses this with:
- comparison mode (fixed reference grid for scoring/match)
- multi-rate export with time-length-aware tap mapping

So you can compare settings more fairly without grid-induced score drift.

## 8) Observability and audit trail

Reproducibility needs traceability.
CamillaFIR provides:
- Summary version stamp and key effective parameters
- run timing breakdown (Read, DSP, ZIP/PNG, Render, Total)
- expanded System Health warnings for source completeness and risky settings

If two runs differ, diagnostics help explain why.

---

## Reproducible-run checklist

1. Use the same app version (`v3.1.1.2` or newer pinned release).
2. Use identical measurement sources and source type (TXT vs WAV) per comparison.
3. Keep base fs/taps fixed, or enable comparison mode.
4. Keep correction guardrails fixed (boost/cut/slope/regularization/phase limit).
5. Keep phase strategy fixed.
6. Keep headroom margin and stereo-link policy fixed.
7. Verify System Health is clean before exporting.
8. Compare Summary outputs and timing data when validating repeated runs.

**Bottom line:** CamillaFIR is reproducible because correction strength is bounded, phase is safety-limited, and the full run is observable end-to-end.

### Disclaimer
AI was used to translate this document from Finnish to English.
