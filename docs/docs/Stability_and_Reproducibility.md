# Stability and Reproducibility

## Why CamillaFIR Produces Stable Results

Room correction often suffers from instability: small measurement changes
produce large audible differences. CamillaFIR avoids this through explicit
DSP constraints.

### 1. Deterministic Pipeline
- No stochastic optimization
- No iterative fitting loops
- No ML-based estimators

Given identical input data and settings, CamillaFIR always produces
bit-identical FIR outputs.

### 2. Confidence-Guided Processing
Low-confidence regions are smoothed instead of inverted.
This prevents unstable high-Q corrections caused by reflections
or microphone placement variance.

### 3. Bounded Correction
All correction stages are explicitly bounded:
- Maximum boost (soft-limited)
- Maximum cut
- Maximum slope (dB/oct)
- Phase correction frequency limit

These bounds guarantee numerical and perceptual stability.

### 4. Time-Domain Separation
Propagation delay, excess phase, and decay are treated independently.
This avoids feedback-like interactions between DSP stages.

---

**Result:**  
CamillaFIR behaves predictably across measurements,
rooms, and loudspeaker systems.
