# IR Export Windowing vs DSP Correction

In CamillaFIR, **IR export windowing** and **DSP correction** are intentionally treated as two separate stages.
This separation is fundamental for clarity, reproducibility, and meaningful A/B comparisons.

---

## DSP Correction (what is corrected)

The actual correction logic operates during analysis and optimization, before any FIR is written to disk.
This stage defines **what is corrected and by how much**, including:

- target curve fitting
- magnitude correction
- phase / excess-phase reconstruction
- FDW / Adaptive FDW (A-FDW)
- Temporal Decay Control (TDC)
- auto-leveling and safety limits

These processes determine the final frequency- and phase-domain behavior of the correction.
They are **not affected** by IR export windowing.

---

## IR Export Windowing (how the FIR is written)

IR export windowing affects **only the time-domain representation of the exported FIR impulse**.

REW-style windowing:
- reshapes the impulse response in the time domain
- affects onset delay, symmetry, and tail behavior
- does **not** change the underlying frequency or phase response of the FIR

Supported modes (UI):

- `auto` - automatic window selection (default)
- `rew_asym` - REW-style asymmetric (causal) window

Legacy config values (`off`, `rew_sym`) are still accepted when set directly in config files,
but are no longer exposed in the UI.

---

## Why this distinction matters

The same DSP correction can legitimately produce **multiple FIR files** that share:
- identical frequency response
- identical phase response

but differ in:
- time alignment
- impulse symmetry
- practical latency characteristics

This is useful for:
- matching different convolver requirements
- minimizing audible latency
- comparing results directly with REW
- controlled listening tests and A/B evaluation

For this reason, CamillaFIR:
- keeps DSP correction deterministic and invariant
- allows IR export windowing as a separate, explicit choice
- includes the windowing type in exported filenames for traceability

---

In short:  
**DSP correction defines the filter behavior.  
IR export windowing defines how that behavior is packaged in time.**

### Disclaimer
AI was used to translate this document from Finnish to English.
