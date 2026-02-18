# Reading CamillaFIR Output

*A practical guide to understanding generated filters, dashboards and
summaries*

CamillaFIR generates three main types of output:

1.  Dashboard plot (PNG)
2.  Filter files (WAV/TXT)
3.  Generation Summary (.txt report)

This document explains how to interpret them correctly.

------------------------------------------------------------------------

# 1. Dashboard Plot Explained

## 1.1 Magnitude & Alignment

This graph shows: - Measured response - Target curve - Predicted
corrected response - Confidence weighting - Level reference window

What to look for: - Predicted should follow Target smoothly - No extreme
narrow boosts - Bass region (20--200 Hz) controlled but not
over-equalized - Large boost areas = potential headroom risk

------------------------------------------------------------------------

## 1.2 Phase

Shows: - Wrapped phase - Mixed-phase behaviour - Crossover influence -
Phase correction clamping

If phase correction is clamped (e.g. 54° -\> 45°), protection is active
to maintain time-domain stability.

------------------------------------------------------------------------

## 1.3 Group Delay (GD)

-   Large peaks in bass → room modes
-   Narrow spikes → reflections
-   Small bumps near crossover are normal

Healthy crossover GD impact should remain low (\<1 ms).

------------------------------------------------------------------------

## 1.4 Filter (dB)

This is the actual FIR gain curve.

Recommended: - Keep max boost ≤ +3 dB - Deep cuts are acceptable - Avoid
large LF boost for headroom safety

------------------------------------------------------------------------

## 1.5 A-FDW Effective Bandwidth

Shows adaptive smoothing bandwidth per frequency.

Typical: - \~1/5 octave average - Wider smoothing in bass - Narrower in
midrange

Prevents overfitting noise.

------------------------------------------------------------------------

# 2. Understanding the Generation Summary

## 2.1 Target Curve Match

-   Above 90% = excellent
-   RMS error \< 1 dB = very accurate correction

------------------------------------------------------------------------

## 2.2 Acoustic Score

Combines: - Match accuracy - Temporal behavior - RT60 balance - Boost
safety

90 ≈ very controlled room\
70--80 ≈ average untreated room\
\<60 = significant acoustic issues

------------------------------------------------------------------------

## 2.3 RT60

-   0.2--0.4s typical domestic room
-   Bass RT60 usually higher
-   If RT60 \> 0.6s → EQ alone is insufficient

Resonances show decay time, not distance.\
Reflections show equivalent path-length distance.

------------------------------------------------------------------------

## 2.4 Boost / Cut Diagnostics

Check: - boost_peak - hard_clamp activity

If boost_peak \> 6 dB → reduce boost limit.

------------------------------------------------------------------------

## 2.5 Bass-First AI

When active: - 20--200 Hz gets correction priority - Protection prevents
unnatural HF shaping

------------------------------------------------------------------------

## 2.6 Headroom Management

If Normalize is ON: - FIR peak reduced to prevent clipping

Recommendation: Keep max boost ≤ +3 dB unless system headroom allows
more.

------------------------------------------------------------------------

# 3. How To Judge If A Filter Is Good

A healthy filter typically shows: - Match \> 90% - RMS error \< 1 dB -
Boost ≤ 3 dB - No excessive clamp warnings - Smooth GD below 500 Hz

If these are met, the filter is technically sound.

------------------------------------------------------------------------

# 4. When Things Look Wrong

Too little bass: - Check low_bass_cut_hz - Check max_boost - Check
correction range

Too harsh: - Over-correction above 2--4 kHz - Too narrow smoothing

Remember: Room treatment always beats excessive EQ.
