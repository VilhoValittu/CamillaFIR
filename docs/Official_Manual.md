# CamillaFIR – Official Manual (v2.7.7)

## 1. Overview
CamillaFIR generates **FIR room-correction filters** from REW exports (magnitude + phase).
It prioritizes **time-domain correctness** before frequency-domain equalization.

CamillaFIR explicitly separates:
- **Propagation delay (Time-of-Flight / TOF)** → removed before phase analysis
- **Excess phase distortion** → handled by FIR phase reconstruction (Linear / Minimum / Mixed / Asymmetric)
- **Room-induced energy storage (room modes)** → handled by **Temporal Decay Control (TDC)**

---

## 2. Processing pipeline (high level)
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
pip install numpy scipy pywebio matplotlib plotly
python camillafir.py
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

