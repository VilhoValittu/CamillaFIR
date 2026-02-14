# CamillaFIR – Official Manual (v3.0.1)

## 1. Overview
CamillaFIR generates **FIR room-correction filters** from REW exports (magnitude + phase).
It prioritizes **time-domain correctness** before frequency-domain equalization.

CamillaFIR explicitly separates:
- **Propagation delay (Time-of-Flight / TOF)** → removed before phase analysis
- **Excess phase distortion** → handled by FIR phase reconstruction (Linear / Minimum / Mixed / Asymmetric)

## 2A. Detailed DSP Signal Flow

The diagram below describes the internal signal-processing architecture
at a more technical level than the simplified pipeline overview.

```
                ┌──────────────────────────────┐
                │   REW Measurement Input      │
                │   (Magnitude + Phase)        │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Robust Parsing & Normalization │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Time-of-Flight Detection    │
                │  & Phase Reference Alignment │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │  Confidence & Reflection     │
                │  Analysis (GD, slope, etc.)  │
                └──────────────┬───────────────┘
                               │
                ├──────────────┴───────────────┐
                ▼                              ▼
     ┌────────────────────┐        ┌────────────────────────┐
     │ Magnitude Path     │        │ Phase Path             │
     └─────────┬──────────┘        └──────────┬─────────────┘
               │                                │
               ▼                                ▼
   ┌──────────────────────────┐     ┌────────────────────────────┐
   │ Target Construction      │     │ Phase Mode Selection       │
   │ (house curve, tilt, XO)  │     │ (Linear/Min/Mixed/Asym)    │
   └─────────┬────────────────┘     └──────────┬─────────────────┘
             │                                  │
             ▼                                  ▼
   ┌──────────────────────────┐     ┌────────────────────────────┐
   │ Level Matching           │     │ Excess-Phase Reconstruction │
   │ (Smart / Manual)         │     │ & Min-Phase Separation      │
   └─────────┬────────────────┘     └──────────┬─────────────────┘
             │                                  │
             ▼                                  ▼
   ┌──────────────────────────┐     ┌────────────────────────────┐
   │ Magnitude Correction     │     │ Conditional GD Stabilization│
   │ - Boost/Cut limits       │     │ (Bass-focused, soft-limited)│
   │ - Slope limits           │     └──────────┬─────────────────┘
   │ - Confidence Pull        │                │
   │ - A-FDW                  │                ▼
   └─────────┬────────────────┘     ┌────────────────────────────┐
             │                      │ Phase Safety Clamp (±45°)  │
             │                      └──────────┬─────────────────┘
             │                                 │
             └──────────────┬──────────────────┘
                            ▼
                ┌──────────────────────────────┐
                │ Optional Temporal Decay      │
                │ Control (TDC)                │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │ FIR Synthesis (IFFT)        │
                │ + Normalization              │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │ Multi-rate Export            │
                └──────────────────────────────┘
```

### Architectural principles

- Magnitude-domain safety (boost/cut/slope/confidence) prevents
  physically unsafe or measurement-driven overcorrection.

- Phase-domain reconstruction is explicitly separated from magnitude logic.

- Group-delay stabilization operates only as a **conditional spike guard**
  and does not act as a wideband phase shaper.

- Temporal Decay Control (TDC) modifies time-domain energy storage
  independently from steady-state magnitude equalization.


---

## 2b. Processing pipeline (high level)
1. Import REW magnitude + phase
2. Robust parsing and unit normalization
3. Optional smoothing (Standard / Psychoacoustic / Adaptive FDW)
4. TOF detection & removal
5. Confidence analysis & reflection detection
6. Target curve construction
7. Level matching (Smart Scan or Manual window)
8. Magnitude correction with safety guards
9. Phase reconstruction (Linear / Minimum / Mixed / Asymmetric)
10. Optional TDC (decay control)
11. FIR synthesis, optional normalization
12. Multi-rate export (optional)

---

## 3. Installation

### Prerequisites
- Python 3.10+ (3.11 recommended)
- Git (optional, but recommended)

### Option A: Standalone EXE (Windows)
**[Download Standalone EXE](https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI)**

### Option B: Run from source

```bash
git clone https://github.com/VilhoValittu/CamillaFIR.git
cd CamillaFIR
python -m venv venv
# Windows:
#   .\venv\Scripts\activate
# Linux/macOS:
#   source venv/bin/activate
pip install -r requirements.txt
# Linux:
#   pip install -r requirements-linux.txt

# Linux
sudo apt update
sudo apt install -y chromium-browser

python src/camillafir/camillafir.py
# Linux
#   python3 src/camillafir/camillafir.py

```

The UI opens in your browser (default: `http://localhost:8080`).

---

## 4. Input data (REW export)

CamillaFIR expects text exports with columns:

- Frequency (Hz)
- Magnitude (dB)
- Phase (deg)

Headers are optional. Comment lines starting with `*`, `#`, or `;` are ignored.

Tips:
- Export both Left and Right separately from REW.
- Use a consistent time reference in REW (same measurement procedure per channel).

---

## 5. Core controls and what they do

### 5.1 Base sample rate and taps
- **Base Sample Rate (fs):** the sample rate used for FIR design.
- **Taps:** FIR length. Higher taps → better low-frequency resolution but more latency.

**Multi-rate generation:** exports multiple sample rates (44.1/48/88.2/96/176.4/192 kHz).

**Auto-taps mapping (multi-rate):** keeps FIR time-length roughly constant across sample rates using a 44.1 kHz reference.

### 5.2 Filter type
- **Linear Phase:** best timing precision, can create audible pre-ringing at high frequencies.
- **Minimum Phase:** no pre-ringing; magnitude correction only, phase derived via minimum-phase reconstruction.
- **Mixed Phase:** linear phase below a split frequency, minimum phase above.
- **Asymmetric Linear:** linear phase, but with an asymmetric time window to suppress audible pre-ringing while preserving the leading edge.

#### Asymmetric Linear (REW Asym)

Asymmetric Linear is a **low-latency linear-phase mode** that reduces audible pre-ringing
by shifting the impulse peak earlier in time.

The **Left window (ms)** parameter defines the **latency target**:
Only **Auto** and **Asymmetric** windowing modes are available.
Legacy **Symmetric** and **Off** modes have been removed to simplify the UI
and focus on the most effective REW-based strategies.

**Practical guidance:**
- **10 ms (default):** best balance between low latency and stable bass correction
- **5–15 ms:** safe operating range
- **< 5 ms:** extreme low-latency mode, expert use only

##### Automatic safety behavior (important)

To prevent unstable bass behavior at very low latency, CamillaFIR applies
automatic safeguards in REW Asymmetric mode:

- When **Left < 15 ms**  
  → bass-first (A-FDW confidence shaping) is automatically limited to low frequencies  
- When **Left < 10 ms**  
  → low-frequency **boosts are disabled** (cuts are still allowed)

These safeguards do **not** reduce correction quality at mid and high frequencies,
but prevent excessive ripple and instability in the bass region.


### 5.3 Smoothing
- **Standard smoothing:** classic fractional-octave smoothing.
- **Psychoacoustic smoothing:** heavier smoothing where the ear is less sensitive (useful for robust targets).
- **Adaptive FDW (A-FDW):** dynamically adjusts the effective window based on confidence. Low confidence → heavier smoothing.

### 5.4 Safety limits (highly recommended)
- **Max boost (dB):** hard safety ceiling for positive gain.
- **Max cut (dB):** maximum allowed attenuation depth.
- **Max slope (dB/oct):** limits how fast correction can change over frequency.
- **Independent slope limits for boost/cut:** optional, prevents small boosts from being flattened while keeping cuts controlled.
- **Excursion protection:** blocks bass boost below a chosen frequency.
- **HPF (subsonic):** protects woofers from ultra-low content.

**HPF behavior (important):**
- HPF is applied as a **true magnitude high-pass filter** in the FIR path.
- The HPF response is added directly to the correction curve
  (equivalent to applying a Butterworth HPF to the final FIR magnitude).
- This ensures **magnitude and phase consistency**.
- Prevents double-HPF behavior, incorrect low-frequency response,
  and artificial group-delay artifacts.

### 5.5 Level matching
CamillaFIR aligns measurement and target levels before synthesizing the filter.

Modes:
- **Smart Scan (Automatic Optimization):** searches for a stable frequency window where measurement follows target shape best, then computes offset using Median or Average.
- **Manual Window:** you choose the lower/upper frequency limits and the target level.

Recommended:
- Use **Median** for room measurements (immune to narrow peaks/dips).
- Use **Average** mainly for nearfield or very smooth data.

---

## 6. Temporal Decay Control (TDC)
TDC is **not EQ**. It targets resonant energy storage (ringing) rather than steady-state amplitude.

Controls:
- **TDC Strength (0–100%)**: how strongly decay is shortened.
- **TDC Max Reduction (dB)**: hard cap for the total reduction applied per frequency bin.
- **TDC Slope Limit (dB/oct)**: optional smoothing of the TDC reduction curve (predictable, avoids narrow notches).

When to enable:
- Room modes dominate the bass (slow decay, boomy notes).

When to reduce or disable:
- Very dry rooms or nearfield measurements where decay is already short.

---

## 7. 2058-safe phase mode
**2058-safe** disables room phase correction (confidence/FDW/excess-phase).
It uses only:
- theoretical crossover phase (if crossover linearization is used)
- minimum-phase where applicable

Use 2058-safe when:
- phase or group delay plots look “spiky”
- step response rings more after phase correction
- you want magnitude correction plus the most conservative phase behavior

---

## 8. Outputs
Typical output package contains:
- FIR filters (`.wav` 32-bit float or text)
- Summary report (`Summary.txt`)
- Plots (magnitude/phase/GD/filter response)
- Optional CamillaDSP YAML snippet

The Summary report typically includes:
- correction range, smoothing, FDW/A-FDW info
- max boost/cut/slope limits applied
- RT60 estimate and confidence summary
- match score and (optionally) comparison-mode grid info

### Output directory
All generated filter packages (`.zip`) are written to the **`filters/`** directory
in the CamillaFIR project root.  
The directory is created automatically during export.

### IR export windowing vs DSP correction

IR windowing applied during FIR export is intentionally separated from the
actual DSP correction logic.
This distinction is important for understanding why FIR files may differ
in time-domain appearance without changing the audible correction.

See:
- [IR Export Windowing vs DSP Correction](IR_Export_Windowing.md)

---

## 9. MiniDSP / limited-taps workflow (practical)
Many MiniDSP devices have limited FIR taps per channel.
A reliable approach is:

1. Use IIR/PEQ on subs (and delay) to get subs reasonably flat and aligned.
2. Measure mains alone.
3. Generate a CamillaFIR filter for mains.
4. Keep correction minimum frequency above the sub crossover (example: 80 Hz).
5. If an IIR crossover exists on the device, CamillaFIR can “unwrap” the crossover phase in the measurement with FIR.
6. Finally align subs/mains timing with delay around the crossover point.

---

## 10. Troubleshooting

### “Spiky” phase / odd step response
- Enable **2058-safe** and retest.
- Reduce phase correction limit.
- Increase smoothing (or enable A-FDW).

### Too aggressive treble
- Use heavier smoothing.
- Lower max slope.
- Limit correction max frequency.

### Bass boost feels unsafe
- Set excursion protection frequency.
- Enable HPF.
- Reduce max boost.

### HPF does not seem to affect bass
- Verify that HPF is enabled and frequency/order are non-zero.
- Check the filter magnitude plot: a proper roll-off should be visible below HPF frequency.
- HPF is applied in the FIR magnitude path, not by disabling correction below cutoff.

---

## 11. DSP Design Rationale

CamillaFIR is built around a small set of explicit design principles.
This section summarizes the reasoning behind the architecture.

### 11.1 Separation of physical phenomena

Room measurements contain multiple independent effects:

1. Propagation delay (Time-of-Flight)
2. Loudspeaker minimum-phase behavior
3. Excess phase distortion
4. Room-induced modal energy storage

Treating these as a single “EQ problem” leads to overcorrection
and unstable filters.

CamillaFIR separates these domains explicitly:

- TOF is removed before phase analysis.
- Minimum-phase and excess-phase components are handled separately.
- Room decay is treated in the time domain (TDC), not as static amplitude EQ.

This separation reduces unintended cross-coupling between magnitude,
phase, and decay shaping.

---

### 11.2 Confidence-weighted correction

Measured data is not equally reliable across frequency.
Reflection density, windowing, and signal-to-noise ratio
all influence trustworthiness.

Instead of applying uniform correction strength,
CamillaFIR uses confidence-aware logic:

- Adaptive FDW (A-FDW)
- Confidence Pull
- Bass-first masking logic

Low-confidence regions are smoothed or gently pulled toward a safe target,
preventing aggressive corrections driven by measurement artefacts.

---

### 11.3 Phase reconstruction philosophy

Phase correction is applied to excess-phase only.
Loudspeaker minimum-phase and theoretical crossover phase
are preserved unless explicitly modified.

Additional safeguards:

- Phase correction clamp (±45°)
- Conditional group-delay gradient stabilization

The GD stabilization stage is intentionally limited:
it acts only as a spike guard and does not reshape
wideband phase trends.

The objective is transient integrity,
not visual flatness of group delay.

---

### 11.4 Time-domain priority

Many correction systems optimize magnitude first
and treat time-domain behaviour as secondary.

CamillaFIR reverses this priority:

- TOF is corrected before phase modelling.
- Phase reconstruction precedes decay shaping.
- Temporal Decay Control modifies energy storage directly.

This ordering minimizes pre-ringing,
reduces modal ringing,
and preserves leading-edge clarity.

---

### 11.5 Determinism and reproducibility

Given identical inputs and configuration,
CamillaFIR produces deterministic outputs.

Safety limits and internal clamps are:

- explicitly documented,
- reported in Summary.txt,
- and visible in DSP info.

The system avoids hidden heuristics that alter behaviour silently.

The result is a correction workflow that is
transparent, repeatable, and technically defensible.

---

### 11.6 Group-delay gradient limiter (mathematical definition)

CamillaFIR includes an optional group-delay (GD) gradient limiter used as a **conditional spike guard**
to prevent artificial phase “kinks” (typically from unwrap/interpolation artefacts)
without reshaping wideband phase trends.

In version 3.0.1 and later, the limiter is:

- **Bass-focused (20–250 Hz)**
- **Soft-limited (tanh)**
- **Conditionally enabled**

It acts strictly as a spike guard, not as a wideband phase shaper.

**Group delay from phase**

Let the unwrapped phase be \( \phi(f) \) in radians, frequency \( f \) in Hz.
Group delay in seconds:

\[
\tau_g(f) = -\frac{1}{2\pi}\frac{d\phi(f)}{df}
\]

In milliseconds:

\[
\mathrm{GD}_{ms}(f) = 1000 \cdot \tau_g(f)
                 = -\frac{1000}{2\pi}\frac{d\phi(f)}{df}
\]

**Gradient per octave**

The limiter operates on the GD slope with respect to the log-frequency axis (octaves):

\[
g(f) = \frac{d\,\mathrm{GD}_{ms}(f)}{d(\log_2 f)}
\quad [\mathrm{ms}/\mathrm{oct}]
\]

**Soft limiting**

Instead of hard clipping, a soft limiter is used to preserve natural trends while compressing extremes:

\[
g_{lim}(f) = L \cdot \tanh\!\Big(\frac{g(f)}{L}\Big)
\]

where \(L\) is the configured limit in \(\mathrm{ms}/\mathrm{oct}\) (e.g. 30 ms/oct when enabled).

**Reconstruction**

The limited GD curve is reconstructed by integrating \(g_{lim}(f)\) over \(\log_2 f\),
anchored at the band center for stability. The limited phase is then obtained by integrating:

\[
\frac{d\phi(f)}{df} = -2\pi \frac{\mathrm{GD}_{ms}(f)}{1000}
\]

In practice:

- The limiter operates only within the **bass-focused band (20–250 Hz)**.
- It is **conditionally enabled** (e.g. bypassed when A-FDW and Bass-first
  stabilization are active, except in high-risk windowing modes).
- The soft-limiting function ensures continuity and avoids sharp clipping artefacts.

This guarantees that group-delay stabilization does not reduce transient
liveliness or alter broadband phase behaviour

---

### 11.7 FIR length vs time / frequency resolution (practical tradeoff)

FIR design always trades time-domain behaviour against frequency-domain resolution.
For a filter with \(N\) taps at sample rate \(f_s\):

- **Time length** (impulse duration):
  \[
  T \approx \frac{N}{f_s}
  \]

- **Frequency-bin spacing / resolution** (typical FFT grid intuition):
  \[
  \Delta f \approx \frac{f_s}{N}
  \]

- **Linear-phase latency** (group delay of a symmetric FIR):
  \[
  \tau \approx \frac{N-1}{2f_s}
  \]

Implications:

- More taps (higher \(N\)) improve low-frequency precision and reduce ripple sensitivity,
  but increase latency and can make time-domain constraints (e.g. low-latency asymmetric exports)
  harder to satisfy.

- Higher sample rate (higher \(f_s\)) reduces time length and latency for the same \(N\),
  but also increases \(\Delta f\). This is why multi-rate export commonly scales taps to keep
  the **time length** approximately constant across sample rates.

Practical guidance:

- Use more taps when you need finer low-frequency control (room modes / long decay).
- Use shorter time length when low latency is required (live monitoring / AV sync),
  accepting reduced LF resolution and relying more on conservative phase behaviour and safety guards.