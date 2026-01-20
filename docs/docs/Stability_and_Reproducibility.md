# Stability and Reproducibility (v2.7.7)

CamillaFIR is built to avoid the classic room-correction failure mode: **tiny measurement differences -> huge, audible filter differences**.

The core idea is simple:
- treat measurements as *partly unreliable* (especially in-room)
- only correct what is likely correctable
- keep every stage bounded so nothing can run away

## 1) Deterministic pipeline
CamillaFIR does not use stochastic optimization, random seeds, ML estimators, or iterative "fit until convergence" loops.

If you run the same version with:
- the same input measurement exports
- the same settings
- the same base sample rate / taps (or the same multi-rate selection)

then the FIR output is reproducible (the math is deterministic).

## 2) Confidence-guided processing (stability against mic moves)
Room measurements contain combing, reflection dips, and windowing artifacts. A naive inverse filter turns those into narrow, high-Q boosts or phase swings.

CamillaFIR reduces that risk by using a *confidence mask* and windowing strategies:
- unreliable regions are smoothed rather than aggressively inverted
- phase work is limited to where it is likely meaningful
- Adaptive FDW (A-FDW) can automatically shorten the window when confidence is low

Practical result: small mic placement changes tend to change only small details, not the entire correction personality.

## 3) Explicit guardrails (bounded correction)
Every major stage has a hard limit so the filter cannot become extreme:

Magnitude and targets
- **max_boost_db**: caps positive correction (soft-limited)
- **max_cut_db**: caps attenuation depth
- **low_bass_cut_hz**: below this frequency only cuts are allowed (prevents risky sub-bass boosts)

Slope / smoothness
- **max_slope_db_per_oct**: legacy max slope limit
- **max_slope_boost_db_per_oct** and **max_slope_cut_db_per_oct** (v2.7.7): optional asymmetric slope limits so boosts and cuts can be constrained differently

Regularization and smoothing
- **reg_strength**: prevents deep nulls from turning into huge boosts
- **df_smoothing** (experimental): keeps smoothing width more constant in Hz across different fs/taps so results stay comparable when you change sample rate or taps

These constraints are not cosmetic. They are the difference between "correction" and "unstable inverse filter".

## 4) Phase safety options
Phase correction is powerful but easy to misuse in-room. v2.7.7 gives you multiple safety levers:
- **phase_limit (Hz)**: do not correct room phase above a chosen frequency
- **FDW cycles** and optional **A-FDW**: reduce phase noise by windowing
- **2058-safe phase mode**: disables room phase correction (confidence/FDW/excess-phase) and uses only theoretical crossover phase and minimum-phase where applicable

If step response or group delay looks spiky, 2058-safe mode is the fast way to return to predictable behavior.

## 5) Separation of problems (less interaction, more stability)
CamillaFIR treats different physical effects as different DSP steps:
- propagation delay / alignment
- crossover linearization (theoretical phase)
- magnitude correction
- decay shaping (TDC)

Because these are separated and bounded, you avoid feedback-like interactions where fixing one problem breaks another.

## 6) Multi-rate output and reproducible evaluation
Multi-rate generation creates filters for multiple sample rates. That is good for deployment, but it can make "scores" look different because FFT binning changes with fs/taps.

To keep evaluation comparable, v2.7.7 supports **comparison mode**:
- scoring and match metrics are locked to a reference grid (for example 44.1 kHz / 65536 taps)
- you can still generate multi-rate filters, but the analysis stays apples-to-apples

## 7) Numerical robustness and edge-case handling
v2.7.7 includes guard code specifically for real-world edge cases:
- empty masks (no valid points in a window)
- NaN/inf values from division or log operations
- out-of-range indexing and "no data" conditions

Level matching was moved to a dedicated module so it is easier to test and less likely to regress. The key promise: the applied offset value is always defined and finite.

## 8) Auditing: logs and Summary
For reproducibility you need traceability:
- the UI/settings are recorded in the generated report/summary
- stage probes can log peak boost/cut evolution through the pipeline

If two runs sound different, these logs tell you *why* (changed settings, different masks, different bounds), not just that they differ.

---

## Reproducible-run checklist
Use this when you want repeatable results across machines or over time.

1. Use the same CamillaFIR version (tag releases).
2. Use the same measurement export format and resolution.
3. Keep base fs/taps the same, or enable comparison mode for stable scoring.
4. Keep smoothing type + smoothing resolution fixed.
5. Keep all guardrails fixed: max boost/cut, slope limits, reg_strength, phase_limit.
6. If testing phase behavior, note whether 2058-safe is ON/OFF.

**Bottom line:** CamillaFIR is predictable because it is bounded, confidence-aware, and deterministic.
