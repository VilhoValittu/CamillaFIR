# Why CamillaFIR Works (v3.2.0)

CamillaFIR is not "magic EQ". It is a bounded DSP workflow built for real room data.

Core rule:

> Correct what is physically plausible and perceptually useful, and avoid inverting unreliable measurement detail.

This page explains why that approach stays stable and audible in practice.

## 1) Time-domain first, then frequency-domain shaping

In-room measurements combine multiple phenomena:
- propagation delay (time-of-flight)
- loudspeaker/crossover behavior
- room reflections and comb filtering
- modal energy storage (ringing)

A pure inverse-EQ approach mixes these into one problem and tends to overcorrect.
CamillaFIR separates them:
- TOF is aligned before phase analysis
- magnitude correction is bounded
- phase is reconstructed with explicit strategy and limits
- decay is treated separately via TDC

## 2) Confidence-weighted correction instead of blind inversion

Not every dip is correctable. Reflection cancellations and low-confidence bins can move with tiny mic-position changes.

CamillaFIR reduces overfit using:
- confidence-aware shaping
- smoothing and regularization
- optional Adaptive FDW (A-FDW)
- Confidence Pull behavior in uncertain regions

Result: corrections track robust trends instead of chasing fragile artifacts.

## 3) Explicit guardrails keep filters physically sane

Most failures in room correction come from unbounded boosts and steep local corrections.
CamillaFIR uses explicit limits, including:
- `max_boost_db`, `max_cut_db`
- `max_slope_db_per_oct` and optional split slope limits for boost vs cut
- `low_bass_cut_hz` / excursion-oriented low-bass safety
- `reg_strength` to avoid null-filling behavior

These are not cosmetic settings. They prevent unstable inverse-filter behavior.

## 4) Phase reconstruction with layered safety

Phase correction can improve transients, but only if it is constrained.

CamillaFIR provides multiple safety layers:
- `phase_limit` to bound correction bandwidth
- FDW / A-FDW to reduce reflection-driven phase noise
- Mixed-phase excess correction fade (LF full correction -> HF no correction)
- Mixed-only guards: `max_excess_delay_ms` and `max_pre_ringing_db`
- adaptive excess-phase clamp behavior for robust operation
- conditional GD spike guarding in bass-focused high-risk cases

This keeps phase work useful without forcing textbook-flat but fragile phase curves.

## 5) TDC targets ringing that EQ alone cannot fix

Room modes are not only amplitude peaks; they are time-domain energy storage.

Temporal Decay Control (TDC):
- shapes decay behavior directly
- is independent from static magnitude EQ
- includes bounded controls (`tdc_strength`, `tdc_max_reduction_db`, optional `tdc_slope_db_per_oct`)

This is why bass can become tighter without simply reducing bass level.

## 6) Headroom and channel consistency are part of DSP safety

A "good" filter that clips is still a bad filter.

CamillaFIR uses an auto-headroom model:
- output attenuation follows realized max boost plus margin
- normalization remains optional as extra safety
- stereo-link can force shared attenuation behavior between channels

So output level handling is integrated into correction quality, not left as an afterthought.

## 7) Reproducibility across fs/taps and input formats

CamillaFIR includes features for apples-to-apples evaluation:
- comparison mode with fixed analysis grid
- multi-rate generation with auto-taps time-length mapping
- deterministic WAV policy aligned with TXT baseline behavior

The goal is stable interpretation when you compare settings, sample rates, or tap counts.

## 8) Operational transparency

The workflow is observable, not opaque:
- System Health checks highlight risky settings and missing sources
- Summary includes version stamp and key effective parameters
- run timing breakdown exposes read/DSP/export/render stages

This makes tuning and troubleshooting traceable.

## 9) Practical takeaway

CamillaFIR works because it combines:
- explicit separation of delay, magnitude, phase, and decay
- confidence-aware correction strength
- hard safety bounds
- deterministic processing and reproducible comparison tools

If you want a robust workflow:
1. Start with bounded magnitude correction and sane headroom.
2. Add phase only within clean confidence and frequency limits.
3. Apply TDC carefully for modal decay problems.
4. Use comparison mode when doing A/B decisions.
5. Re-check output diagnostics and Summary before final deployment.

See also image reference: `docs/pics/tdc_impulse_example.png`

### Disclaimer
AI was used to translate this document from Finnish to English.
