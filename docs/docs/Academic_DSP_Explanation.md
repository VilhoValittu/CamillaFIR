
# CamillaFIR: A Time-Domain–First FIR Room Correction Framework
## Academic DSP Rationale and Mathematical Foundations

### Abstract
CamillaFIR is a room-correction DSP framework that prioritizes **time-domain correctness**
before frequency-domain equalization. Unlike traditional equalization-centric systems,
CamillaFIR explicitly separates *propagation delay*, *excess phase*, and *room-induced energy
storage*, applying different mathematical treatments to each. This document formalizes
the DSP principles, equations, and justifications behind the system.

---

## 1. Signal Model

Let the measured frequency response of a loudspeaker-room system be:

\[
H_m(f) = H_s(f)\,H_r(f)\,e^{-j 2\pi f \tau}
\]

where:
- \(H_s(f)\) is the loudspeaker + crossover response,
- \(H_r(f)\) is the room response,
- \(\tau\) is the acoustic Time of Flight (TOF).

Conventional systems attempt to invert \(H_m(f)\) directly.
CamillaFIR instead **explicitly removes** the linear phase term
associated with \(\tau\).

---

## 2. Time-of-Flight (TOF) Removal

The unwrapped phase \(\phi(f)\) of \(H_m(f)\) is approximated by:

\[
\phi(f) \approx -2\pi f \tau + \phi_e(f)
\]

where \(\phi_e(f)\) is the excess phase.

A linear regression is performed over a stable frequency band
(typically 1–10 kHz):

\[
\tau = -\frac{1}{2\pi} \frac{d\phi(f)}{df}
\]

The corrected response becomes:

\[
H_c(f) = H_m(f)\,e^{j2\pi f \tau}
\]

This ensures that subsequent phase processing targets **distortion**,
not distance.

---

## 3. Confidence Mask Estimation

Each frequency bin is assigned a confidence value \(C(f) \in [0,1]\),
derived from:
- phase variance,
- reflection dominance,
- decay stability.

Conceptually:

\[
C(f) = \exp\left(-\alpha \sigma_\phi^2(f) - \beta E_r(f)\right)
\]

where \(\sigma_\phi^2\) is local phase variance and \(E_r\) reflected energy.
This mask governs all downstream DSP aggressiveness.

---

## 4. Frequency-Dependent Windowing (FDW)

The effective time window length is:

\[
N(f) = N_{\min} + C(f)\,(N_{\max}-N_{\min})
\]

Higher confidence yields longer windows (higher resolution),
lower confidence yields shorter windows (smoothing).

This prevents inversion of stochastic interference patterns.

---

## 5. Magnitude Correction with Regularization

The target magnitude is \(T(f)\).
Raw correction gain:

\[
G_{\text{raw}}(f) = T(f) - |H_c(f)|
\]

Regularization limits correction depth:

\[
G(f) = \frac{G_{\text{raw}}(f)}{1 + \left(\frac{|G_{\text{raw}}(f)|}{R(f)}\right)^2}
\]

where \(R(f)\) increases with frequency to avoid high-Q treble correction.

---

## 6. Slope Limiting

To avoid non-physical correction shapes, gain slope is constrained:

\[
\left| \frac{dG(f)}{d\log_2 f} \right| \le S_{\max}
\]

This is enforced by forward and backward passes in log-frequency space.

---

## 7. Phase Reconstruction Strategies

### 7.1 Minimum Phase
Given magnitude spectrum \(M(f)\), minimum phase is computed via the
Hilbert transform:

\[
\phi_{\min}(f) = -\mathcal{H}\{\ln M(f)\}
\]

This guarantees causality and zero pre-ringing.

### 7.2 Linear Phase
Linear phase correction directly inverts \(\phi_e(f)\):

\[
H_{\text{lin}}(f) = e^{-j \phi_e(f)}
\]

### 7.3 Mixed Phase
A linear-phase FIR crossover \(W(f)\) blends both:

\[
H(f) = W(f)\,H_{\text{lin}}(f) + [1-W(f)]\,H_{\min}(f)
\]

Magnitude unity is preserved by construction.

### 7.4 Asymmetric Linear Phase
The impulse response is windowed asymmetrically:

\[
w(n) =
\begin{cases}
w_L(n), & n < 0 \\
w_R(n), & n \ge 0
\end{cases}
\]

with \(w_L \ll w_R\), suppressing audible pre-ringing.

---

## 8. Temporal Decay Control (TDC)

Room modes represent excessive energy storage.
Let the modal decay envelope be:

\[
h(t) = e^{-t/\tau_r}
\]

TDC injects an inverse-decay kernel:

\[
h_{\text{tdc}}(t) = -k\,e^{-t/\tau_r}
\]

Such that the combined impulse shortens decay time
without reducing steady-state SPL.

This is fundamentally **time-domain control**, not EQ.

---

## 9. FIR Synthesis

The final complex correction spectrum is:

\[
H_{\text{corr}}(f) = 10^{G(f)/20} \cdot H_{\text{phase}}(f)
\]

IFFT yields the FIR impulse response, followed by
normalization and optional multi-rate scaling.

---

## 10. Perceptual Scoring

Deviation is measured as confidence-weighted RMS:

\[
\text{RMS} = \sqrt{\frac{\sum C(f)\,\Delta^2(f)}{\sum C(f)}}
\]

Mapped to perceptual percentage via a sigmoid function.

---

## 11. Design Philosophy Summary

CamillaFIR:
- Corrects **only what is physically correctable**
- Avoids inversion of noise and reflections
- Treats room modes as time problems, not EQ problems
- Uses FIR mathematics consistently across all domains

This is why the system remains stable, natural,
and perceptually transparent.
