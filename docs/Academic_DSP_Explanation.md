# CamillaFIR: Time-Domain-First FIR Room Correction
## Academic DSP Rationale and Mathematical Foundations (v3.1.1.2)

### Abstract
CamillaFIR is a FIR room-correction framework that separates:
- propagation delay (time of flight),
- excess phase distortion,
- magnitude-domain target tracking,
- temporal decay behavior (TDC).

The current engine (v3.1.1.2) uses confidence-aware and safety-bounded processing in both magnitude and phase paths, with optional fixed-grid comparison analysis for reproducible scoring.

---

## 1. Measurement model

Let the measured response be

\[
H_m(f)=|H_m(f)|e^{j\phi_m(f)}.
\]

CamillaFIR does not directly invert \(H_m\). It first removes linear delay and then applies bounded correction to magnitude and excess phase.

---

## 2. TOF removal (implemented)

Unwrapped phase is modeled as

\[
\phi_m(f)\approx a f+\phi_e(f),
\]

where \(af\) is the linear delay term and \(\phi_e(f)\) is excess phase. The slope \(a\) is estimated by linear regression over 1-10 kHz and removed:

\[
\phi_c(f)=\phi_m(f)-af.
\]

This ensures phase correction acts on excess phase, not acoustic distance.

---

## 3. Acoustic confidence and reflection extraction

From complex analysis data, CamillaFIR computes group delay:

\[
GD(f)=-\frac{1}{2\pi}\frac{d\phi(f)}{df}.
\]

A smoothed baseline \(GD_s(f)\) is subtracted and confidence is formed from deviation:

\[
\Delta GD(f)=|GD(f)-GD_s(f)|,
\]
\[
C(f)=\frac{1}{1+\exp\left(1.5(\Delta GD(f)-2.5\text{ ms})\right)}.
\]

Reflection/resonance nodes are detected as peaks of \(\Delta GD\) and later reused by TDC.

---

## 4. Target formation, leveling, and comparison grid

Target \(T(f)\) is either flat (0 dB) or interpolated from a loaded house curve. Before correction, target/measurement alignment is solved by the leveling stage (auto/manual windowing and offset estimation).

When comparison mode is enabled, analysis metrics are additionally projected to a fixed grid (44.1 kHz, 65536 taps) for stable score reporting across run settings.

---

## 5. Magnitude correction pipeline (current implementation)

### 5.1 Raw gain
\[
G_{raw}(f)=T(f)-\left(M(f)-\Delta L\right)+B_{manual},
\]
where \(M(f)\) is analyzed magnitude, \(\Delta L\) is computed level offset, and \(B_{manual}\) is optional manual bias.

### 5.2 Regularization and smoothing
CamillaFIR computes a smoothed reference \(G_{sm}\) and blends it using regularization ratio \(r\in[0,1]\):

\[
G_{reg}(f)=(1-r)G_{raw}(f)+rG_{sm}(f),
\]
\[
r=\text{reg\_strength}/100.
\]

Smoothing mode is either fractional-octave or DF (constant-Hz equivalent kernel width across sample-rate/tap changes).

### 5.3 Adaptive FDW (A-FDW)
If enabled, smoothing bandwidth is confidence-weighted via cycles:

\[
N(f)=N_{min}+C(f)(N_{base}-N_{min}),\quad BW_{oct}(f)\propto \frac{1}{N(f)}.
\]

### 5.4 Bass-first modulation
Optional Bass-first AI derives reliability and room-mode masks from GD behavior, spectral prominence/Q, roughness, and phase-jerk cues. It strengthens cuts and suppresses boosts in modal regions.

### 5.5 Safety stack (order in code)
Inside correction band:
1. Low-bass cuts-only policy (optionally strength-weighted)
2. Soft clip:
\[
G_{+}=B_{max}\tanh(G/B_{max}),\quad
G_{-}=-C_{max}\tanh((-G)/C_{max})
\]
3. Slope limit (global or asymmetric boost/cut dB/oct)
4. Confidence-weighted pull toward safe reference (post-slope stage):
\[
G_{out}=w(f)G+(1-w(f))G_{ref}
\]
with \(w\) derived from confidence floor/ceil and separate boost/cut gamma (in the current pipeline this is run in the slope-limited correction branch)
5. Transition fade near \(mag\_c\_max\)
6. Excursion protection constraints
7. Hard clamp to max boost/cut
8. Clamp-aware post-smoothing and final re-clamp
9. WAV-source ripple polish (transition-zone stabilization)

---

## 6. Temporal Decay Control (TDC)

TDC modifies the target (not final phase directly) using detected reflection/mode nodes and RT60 context:

\[
D(f)=\sum_i k_i\exp\left(-\frac{(f-f_i)^2}{2\sigma_i^2}\right),
\]
\[
0\le D(f)\le D_{max}.
\]

Then:
\[
T_{tdc}(f)=T(f)-D(f).
\]

Optional TDC slope limiting constrains \(D(f)\) in dB/oct to avoid narrow stacked notches.

---

## 7. Phase modeling and guards

### 7.1 Theoretical phase baseline
Theoretical phase \(\phi_{theo}(f)\) is built from configured XO + HPF analog Butterworth models.

### 7.2 Excess phase correction
\[
\phi_{ex}(f)=\phi_c(f)-\phi_{theo}(f).
\]
Correction term:
\[
\phi_{add}(f)=-\phi_{ex}(f)\,W_{phase}(f)\,W_{conf}(f),
\]
where weights depend on filter type, phase-limit region, and confidence.

### 7.3 Adaptive phase clamp
\(\phi_{add}\) is clamped by a frequency/confidence-adaptive budget. Default mixed-phase budgets are tighter at HF and wider at LF.

### 7.4 Conditional GD-gradient limiter
A GD-gradient limiter on phase correction is enabled in conservative cases (notably tight REW asymmetric latency mode), to prevent unstable bass spikes.

### 7.5 Mixed-phase extra guards
Mixed mode includes:
- excess-delay guard (scales correction if max group delay exceeds configured limit),
- pre-ringing guard (iterative scaling to respect max pre-ringing dB target),
- time-domain mixed fusion of linear/minimum components around split + transition.

---

## 8. FIR synthesis, gain staging, and export IR windowing

Final magnitude:
\[
|H_{corr}(f)|=10^{G_{final}(f)/20}.
\]

Phase depends on mode (Linear, Minimum, Mixed, Asymmetric), then IFFT gives \(h[n]\).

Automatic gain staging applies:
- auto global gain from peak estimate and requested margin,
- optional extra headroom when normalization is enabled.

IR export windowing (auto/off/rew_sym/rew_asym; Hann/Tukey) is applied after DSP correction to shape impulse time spread. This affects IR realization, not the target definition logic.

---

## 9. Stability summary (current)

CamillaFIR stability is based on layered constraints:
- TOF removed before phase correction,
- confidence-aware smoothing and correction pull,
- bounded boost/cut and slope (soft + hard clamp),
- excursion/low-bass safety policies,
- bounded phase correction with adaptive clamp and mixed guards,
- optional fixed comparison grid for repeatable scoring.
