

---

# CamillaFIR

**Automated Universal FIR Filter Generator & DSP Pipeline**

CamillaFIR is a Python-based DSP utility designed to automate the generation of high-precision Finite Impulse Response (FIR) correction kernels based on acoustic measurements. It leverages **PyWebIO** for a browser-based interface and **SciPy/NumPy** for complex domain signal processing.

While originally architected for **CamillaDSP** (including automated `.yml` configuration generation), the tool generates standard IEEE 32-bit float impulse response files (`.wav` and `.txt`). These kernels are **universally compatible** with any convolution engine or hardware DSP that supports FIR filtering, including:

* **Hardware DSPs:** Analog Devices SigmaDSP (ADAU1701 via SigmaStudio), MiniDSP (OpenDRC, Flex, SHD etc.) etc.
* **Software DSPs:** Roon (Convolution), Equalizer APO, Volumio, Moode, FusionDSP, JRiver Media Center etc.

## 💡 Project Philosophy
CamillaFIR was originally developed as a personal project to perfect my own listening environment. The workflow is primarily optimized for **2-channel stereo systems** (2.0, 2.1, or 2.2 setups), focusing on absolute phase linearity and transient accuracy.

While the interface is streamlined for stereo use, the underlying DSP engine is channel-agnostic. With a bit of imagination and creative routing, it can absolutely be adapted for multi-channel home theater applications.

---
### ✨ New in v2.6.2 Stable

#### ⏳ Temporal Decay Control (TDC)
* **Function:** Actively reduces the ringing time ($T_{60}$) of room resonances.
* **Mechanism:** Identifies high-Q modes and injects inverse-phase decay kernels to "stop" the bass note faster, rather than just lowering its volume.
* **Benefit:** Significantly tighter bass response and improved transient clarity in resonant rooms.

#### 🧠 Adaptive Frequency Dependent Windowing (A-FDW)
* **Function:** Dynamically adjusts the windowing length based on signal quality.
* **Logic:** $N_{cycles}(f) = N_{min} + C(f) \cdot (N_{base} - N_{min})$
* **Benefit:** Uses long windows for high-confidence data (high resolution) and short windows for chaotic reflections (smoothing), preventing over-correction of acoustic noise.

#### 🚀 Asymmetric Linear Phase
* **Topology:** A specialized low-latency Linear Phase mode.
* **Strategy:** Decoupled adjustdable windowing (basic settings : 100ms pre-peak / 500ms post-peak) maintains perfect phase linearity while eliminating audible pre-ringing artifacts common in standard linear phase filters.

#### 🔍 Acoustic Intelligence Engine
* **Reflection Tracking:** Automatically detects boundary reflections and calculates their physical distance (meters).
* **Confidence Masking:** Assigns a reliability score (%) to the measurement data, guiding the DSP engine on how aggressively to correct specific frequency bands.

---


## 🚀 DSP Engine Architecture (v2.5.5)

The core processing engine has been significantly upgraded in version 2.5.5 to ensure mathematical transparency and safety. It performs complex domain signal processing to derive correction impulses, utilizing advanced strategies to separate excess phase from propagation delay.

### 1. Advanced Phase Reconstruction
* **Transparent Mixed Phase Split (New in v2.5.5):**
  The transition between Linear Phase (Bass) and Minimum Phase (Treble) is now handled by a **Linear Phase FIR Crossover** instead of analog IIR emulation.
  * **Benefit:** This ensures perfect magnitude reconstruction and zero phase distortion at the crossover point (typically 300Hz). The Linear and Minimum phase components sum to unity without "dips" or phase rotation.
* **Excess Phase Safety Masks (New in v2.5.5):**
  The engine now applies hard constraints to phase correction at extreme frequencies (<15 Hz and >19 kHz).
  * **Benefit:** Prevents the algorithm from trying to correct measurement noise or infrasonic anomalies, eliminating potential pre-ringing artifacts and protecting transducers.

### 2. Signal Processing Pipeline
* **Time-of-Flight (TOF) Correction:**
  The engine implements an algorithm to automatically detect and remove acoustic propagation delay. It performs linear regression analysis on the unwrapped phase response within the stable passband (1kHz–10kHz) to determine the slope (group delay). This linear phase component is mathematically subtracted before Minimum Phase calculation, ensuring that "Excess Phase" extraction targets only group delay distortions derived from the transducer and crossover, not distance.
* **Frequency-Dependent Regularization:**
  To prevent high-Q correction of non-minimum phase nulls (room modes), the engine applies variable regularization strength:
    * **< 200 Hz:** Low regularization (50% of base) allows for aggressive modal linearization.
    * **200 Hz - 2 kHz:** Linear ramp interpolation.
    * **> 2 kHz:** High regularization is applied to preserve the transducer's natural power response and prevent "phasiness" in the stochastic region.
* **Excursion Protection:**
  Applies a hard-knee high-pass constraint to the target curve below user-defined frequencies (e.g., < 40Hz) to prevent mechanical over-excursion of woofers.

---

### 2. Phase Reconstruction Strategies

* **Linear Phase:** Inverts both magnitude and phase (Group Delay) across the entire bandwidth. Uses TOF-corrected excess phase inversion to achieve a theoretically flat group delay ().
* **Minimum Phase:** Corrects magnitude response only. The phase response is mathematically derived via the **Discrete Hilbert Transform** of the magnitude cepstrum, ensuring strict causality, zero pre-ringing, and minimal latency.
* **Mixed Phase (Hybrid):** Performs sub-band processing using complex-domain filtering:
* **LF Band (< 300Hz):** Linear Phase correction for time-alignment of low-frequency transients and group delay linearization.
* **HF Band (> 300Hz):** Minimum Phase correction to ensure natural decay and eliminate pre-ringing on high-frequency transients.
* **Reconstruction:** Recombination is performed using 4th-order Linkwitz-Riley filters in the complex domain.



### 3. Hardware DSP Support (SigmaStudio / ADAU1701)

Version 2.5.x introduces specific optimizations for resource-constrained fixed-point DSPs:

* **Low Tap-Count Support:** Generation of 512 and 1024 tap kernels.
* **Raw Coefficient Export:** Direct export of impulse response coefficients to `.txt` format for import into SigmaStudio **FIR Filter** or **Table** blocks.
* **Resolution Logic:** The engine automatically advises on phase strategy validity based on tap count (e.g., forcing Minimum Phase for low-resolution kernels where low-frequency wave lengths exceed filter length).


### 4. New Filename Convention: 

Exported filters are now named using a highly descriptive format for better organization: L/R_[Type]_[Samplerate]_[Time]_[Date].wav (e.g., L_Mixed_48000Hz_1730_060126.wav).

---

## 🛠️ Installation

### Prerequisites

* **Python 3.10** or newer.
* **Git** (for cloning).

### Option A: Binary (Windows Only)

For users who do not wish to manage Python environments:
**[Download Standalone EXE](https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI)**

### Option B: Source (Cross-Platform)

**1. Clone the Repository**

```bash
git clone https://github.com/VilhoValittu/CamillaFIR.git
cd CamillaFIR

```

**2. Install Dependencies**
It is recommended to run this in a virtual environment.

*Windows (PowerShell):*

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install numpy scipy pywebio matplotlib requests

```

*Linux / macOS:*

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scipy pywebio matplotlib requests

```

---

## 🖥️ Usage

1. Start the application server:
```bash
python CamillaFIR.py

```


2. The GUI will automatically launch in your default web browser (default: `http://localhost:8080`).
3. **Workflow:**
* **I/O:** Upload measurement exports (TXT format from REW) or define local paths.
* **Target:** Select a target curve (Harman, Toole, etc.) or upload a custom CSV.
* **DSP Settings:** Select filter topology (Linear/Minimum/Mixed) and Tap count.
* **Execute:** Click "Run Analysis".


4. **Output:** The tool generates a `.zip` archive containing:
* **FIR Kernels:** `.wav` (32-bit float) or `.txt` coefficients for multiple sample rates.
* **Configuration:** `camilladsp.yml` snippet.
* **Analysis:** Plots for Magnitude, Phase, Group Delay, and Filter Response.



---

## 📊 Input Data Format

CamillaFIR parses standard frequency response text exports (e.g., from Room EQ Wizard):

* Format: `Frequency (Hz) | Magnitude (dB) | Phase (deg)`
* Headers are optional; comments marked with `*` or `#` are ignored.

---

## Acknowledgements

Development inspired by the methodologies of **OCA** (Obsessive Compulsive Audiophile): [https://www.youtube.com/@ocaudiophile](https://www.youtube.com/@ocaudiophile)

---

