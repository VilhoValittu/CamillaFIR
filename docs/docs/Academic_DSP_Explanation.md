# CamillaFIR: A Time-Domain–First FIR Room Correction Framework
## Academic DSP Rationale and Mathematical Foundations (v2.7.7)

### Abstract
CamillaFIR is a room-correction DSP framework that prioritizes **time-domain correctness**
before frequency-domain equalization. Unlike traditional EQ-centric systems,
CamillaFIR explicitly separates *propagation delay*, *excess phase*, and *room-induced energy
storage*, applying different mathematical treatments to each.

---

## 1. Signal model

Let the measured loudspeaker-room frequency response be

\[
H_m(f) = H_s(f)\,H_r(f)\,e^{-j 2\pi f \tau}
\]

where:
- \(H_s(f)\) is the loudspeaker + crossover response,
- \(H_r(f)\) is the room response,
- \(\tau\) is the acoustic time of flight (TOF).

A direct inversion of \(H_m\) implicitly mixes distance, excess phase, and room decay.
CamillaFIR instead removes TOF first and treats the remaining terms with bounded, physically
meaningful operations.

---

## 2. Time-of-Flight (TOF) removal

The unwrapped phase \(\phi(f)\) of \(H_m(f)\) is approximated by

\[
\phi(f) \approx -2\pi f\,\tau + \phi_e(f)
\]

where \(\phi_e(f)\) is the excess phase. Over a stable band, \(\tau\) can be estimated by a
linear fit of phase slope:

\[
\tau = -\frac{1}{2\pi}\,\frac{d\phi(f)}{df}
\]

The corrected response is

\[
H_c(f) = H_m(f)\,e^{j 2\pi f\tau}
\]

This ensures subsequent phase processing targets **distortion**, not distance.

---

## 3. Confidence masking

Each frequency bin gets a reliability score \(C(f)\in[0,1]\), derived from phase stability,
reflection dominance, and decay behavior. A conceptual form is

\[
C(f) = \exp\left(-\alpha\,\sigma_\phi^2(f) - \beta\,E_r(f)\right)
\]

where \(\sigma_\phi^2\) is local phase variance and \(E_r\) is reflected energy.
This mask controls downstream aggressiveness (windowing, smoothing, and correction strength).

---

## 4. Frequency-dependent windowing (FDW) and adaptive FDW

FDW controls the time window length as a function of frequency. A practical parameterization is

\[
N(f) = N_{\min} + C(f)\,(N_{\max}-N_{\min})
\]

High confidence yields longer windows (higher resolution); low confidence yields shorter windows
(more smoothing). This prevents inverting stochastic interference patterns.

CamillaFIR’s adaptive FDW reports the **effective bandwidth** range (min/mean/max) that results
from confidence-weighted windowing.

---

## 5. Magnitude correction with regularization

Let \(T(f)\) be the target magnitude (in dB) and \(|H_c(f)|\) the corrected measurement magnitude.
Raw correction gain (dB):

\[
G_{raw}(f) = T(f) - |H_c(f)|
\]

Regularization limits correction depth (especially for deep nulls). One stable form is a
soft saturation:

\[
G(f) = \frac{G_{raw}(f)}{1 + \left(\frac{|G_{raw}(f)|}{R(f)}\right)^2}
\]

where \(R(f)\) is the regularization strength in dB.

---

## 6. Slope limiting (symmetric and asymmetric)

To avoid non-physical correction shapes, gain slope is constrained in log-frequency space:

\[
\left|\frac{dG(f)}{d\log_2 f}\right| \le S_{\max}
\]

Implementation uses forward and backward passes over \(\log_2 f\) to enforce the bound.

### 6.1 Independent boost/cut slope limits

A key stability improvement is using different bounds for rising vs falling segments:

\[
\frac{dG}{d\log_2 f} \le S_{\max}^{(+)} \quad (\text{boost / rising})
\]
\[
\frac{dG}{d\log_2 f} \ge -S_{\max}^{(-)} \quad (\text{cut / falling})
\]

This prevents small boosts from being flattened by an overly strict symmetric limiter, while
keeping cuts controlled.

---

## 7. Phase reconstruction strategies

### 7.1 Minimum phase
Given a magnitude spectrum \(M(f)\), minimum phase can be computed via the Hilbert transform:

\[
\phi_{min}(f) = -\mathcal{H}\{\ln M(f)\}
\]

### 7.2 Linear phase
Linear-phase correction inverts the estimated excess phase:

\[
H_{lin}(f) = e^{-j\,\phi_e(f)}
\]

### 7.3 Mixed phase
A frequency-domain weighting \(W(f)\) blends linear and minimum phase:

\[
H_{phase}(f) = W(f)\,H_{lin}(f) + [1-W(f)]\,H_{min}(f)
\]

### 7.4 Asymmetric linear phase
Asymmetric linear phase applies an asymmetric time window to reduce audible pre-ringing:

\[
w(n)=\begin{cases}w_L(n), & n<0\\ w_R(n), & n\ge 0\end{cases}\quad\text{with } w_L\ll w_R
\]

---

## 8. Temporal Decay Control (TDC)

Room modes are energy-storage phenomena. A simplified modal impulse envelope is

\[
h(t)=e^{-t/\tau_r}
\]

TDC injects an inverse-decay component to shorten the tail:

\[
h_{tdc}(t)=-k\,e^{-t/\tau_r}
\]

### 8.1 Safety brakes for predictability
TDC in CamillaFIR is bounded by:
- a **hard cap** on total reduction per frequency bin, \(R_{tdc,max}\) (dB)
- an **optional slope limit** on the reduction curve (dB/oct) to avoid narrow, stacked notches

Conceptually, the accumulated reduction curve \(D(f)\) is constrained by

\[
0 \le D(f) \le R_{tdc,max}
\]

and optionally

\[
\left|\frac{dD(f)}{d\log_2 f}\right| \le S_{tdc}
\]

---

## 9. DF smoothing (constant-Hz smoothing across sample rates)

When analysis grids (fs/taps) change, octave-based smoothing changes its effective width in Hz.
DF smoothing targets a roughly constant width in **Hz** by setting the smoothing kernel width
in bins proportional to the analysis frequency resolution.

If FFT bin spacing is \(\Delta f = \frac{f_s}{N}\), a constant-Hz kernel \(\sigma_{Hz}\) maps to

\[
\sigma_{bins} \approx \frac{\sigma_{Hz}}{\Delta f} = \sigma_{Hz}\,\frac{N}{f_s}
\]

This makes “detail level” comparable across sample rates.

---

## 10. FIR synthesis

The final complex correction spectrum is

\[
H_{corr}(f) = 10^{G(f)/20}\,H_{phase}(f)
\]

IFFT yields the FIR impulse response, followed by optional normalization and export.

---

## 11. Stability summary

CamillaFIR is stable because it:
- avoids inverting low-confidence data
- bounds gain (boost/cut), slope, and phase correction bandwidth
- treats TOF, excess phase, and decay as separate problems
- provides reproducible scoring via an optional fixed analysis grid
