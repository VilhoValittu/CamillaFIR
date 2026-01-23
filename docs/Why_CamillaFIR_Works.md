# Why CamillaFIR Works (v2.7.7)

CamillaFIR is not "magic EQ". It is a set of DSP decisions that follow a simple rule:

> Correct what is physically plausible and perceptually relevant, and refuse to overfit the measurement.

This page explains the main ideas in practical terms.

## 1) Time first, then frequency
In-room measurements mix multiple phenomena:
- propagation delay (distance)
- loudspeaker/crossover behavior
- room reflections
- true resonances (energy storage)

If you treat all of that as one frequency response and invert it, you get unstable filters.
CamillaFIR instead separates the problems:
- **time alignment** handles arrival-time differences
- **crossover linearization** uses theoretical crossover phase where you want it
- **magnitude correction** is applied only within user limits
- **TDC** (Temporal Decay Control) targets ringing (decay) rather than amplitude alone

## 2) Confidence-guided DSP (do not trust every dip)
A deep dip in an in-room response is often a cancellation from reflections. Boosting it can:
- waste headroom
- increase distortion
- change drastically with small mic moves

CamillaFIR uses a confidence concept to reduce over-correction:
- unreliable regions are **smoothed**, not aggressively inverted
- optional **Adaptive FDW (A-FDW)** shortens the time window when confidence is low

The result is a filter that changes less when the measurement changes a little.

## 3) Guardrails that prevent "inverse filter" behavior
Most audible failures come from unbounded correction. v2.7.7 makes the limits explicit:

- **max_boost_db**: cap boosts (soft-limited)
- **max_cut_db**: cap attenuation depth
- **low_bass_cut_hz**: below this frequency allow only cuts (protects from risky sub-bass boosts)
- **max_slope_db_per_oct**: limit how fast the correction curve can change with frequency
- **max_slope_boost_db_per_oct / max_slope_cut_db_per_oct**: optional asymmetric slope limits so boosts and cuts can be constrained differently
- **reg_strength**: reduces the urge to "fill" deep nulls with huge boosts

These limits are what make the correction listenable and repeatable.

## 4) Phase handling that you can trust
Phase correction is useful when it is based on reliable information, and harmful when it chases noise.

CamillaFIR provides three safety layers:
1. **phase_limit (Hz)**: only do phase work up to a chosen frequency.
2. **FDW (cycles)** and optional **A-FDW**: windowing reduces reflection-driven phase noise.
3. **2058-safe phase mode**: disables room phase correction (confidence/FDW/excess-phase) and uses only theoretical crossover phase and minimum-phase where applicable.

If group delay looks spiky or step response rings, 2058-safe mode is the fast path back to predictable behavior.

## 5) Mixed-phase done safely
The Linear/Minimum/Mixed strategies exist because "perfect" is not the goal; *useful* is.

- **Linear phase** keeps timing consistent but can pre-ring.
- **Minimum phase** avoids pre-ringing but does not preserve absolute timing.
- **Mixed phase** splits behavior: linear where it matters (typically low frequencies) and minimum where it reduces artifacts (typically higher frequencies).

The point is not a textbook ideal, but a controlled tradeoff that stays stable under measurement variance.

## 6) TDC: fixing what EQ cannot
Room modes are not just amplitude peaks. They are **energy storage** problems.

EQ changes the steady-state amplitude. It does not directly shorten the decay tail.
**Temporal Decay Control (TDC)** shapes the target so resonances stop faster.

v2.7.7 adds two important safety brakes:
- **tdc_max_reduction_db**: caps total TDC reduction per frequency bin (prevents stacked deep notches)
- **tdc_slope_db_per_oct**: optional slope limit for the TDC reduction curve (keeps it smooth and predictable)

Conceptually, this is why you can get "tighter bass" without simply turning bass down.

## 7) Consistent behavior across sample rates and taps
Changing fs/taps changes FFT binning and smoothing behavior. That can make analysis and scores look different even if the audible result is similar.

v2.7.7 includes tools to keep comparisons fair:
- **multi-rate generation** for deployment across 44.1/48/88.2/96/176.4/192 kHz
- **comparison mode** to lock scoring and match metrics to a reference grid (apples-to-apples)
- **DF smoothing (experimental)** to keep smoothing width more constant in Hz across fs/taps

## 8) The practical takeaway
CamillaFIR works because it:
- separates different physical causes (delay, crossover, magnitude, decay)
- distrusts low-confidence measurement detail
- enforces hard bounds so the filter cannot become extreme
- gives you safe phase options, including a "no room phase correction" mode
- targets decay (TDC) where amplitude EQ is insufficient

If you want a simple workflow:
1. Get magnitude stable (smoothing + guardrails + regularization).
2. Add phase only where it stays clean (phase_limit + FDW/A-FDW).
3. Add TDC carefully (cap + slope) for tighter decay.
4. Use comparison mode when you are doing A/B tests across settings.

## 9) Seeing the effect
A resonance problem often looks like a long ringing tail in the impulse response. TDC aims to reduce that tail.

If you include the example plot in your docs, reference it like this:
- `tdc_impulse_example.png` (Before vs After TDC)

CamillaFIR also writes a human-readable summary (and optional debug probes) so you can connect what you hear to what changed in the DSP.
