# Why CamillaFIR Works

## 1. Time First, Frequency Second
Most room correction systems treat phase as secondary.
CamillaFIR removes propagation delay *before* phase analysis,
so excess phase truly represents distortion, not distance.

## 2. Confidence-Guided DSP
The engine does not assume all data is correct.
Unreliable regions are smoothed, not equalized.

This avoids:
- Treble harshness
- Over-tight bass
- Phase noise correction

## 3. FIR-Accurate Mixed Phase
The Linear/Minimum transition is done using FIR-domain summation.
This guarantees:
- No dips
- No phase rotation
- Unity gain through crossover

## 4. TDC Targets Energy Storage
Room modes are energy storage problems.
EQ only changes amplitude.
TDC changes **decay behavior**, which is what you actually hear.

## 5. Human-Perceptual Metrics
Final scoring uses:
- Confidence-weighted RMS
- Sigmoid mapping to percentages

This correlates far better with listening results
than raw dB deviation.

---

In short:
CamillaFIR corrects what is *physically correctable*,
ignores what is not,
and never lies to the listener.
