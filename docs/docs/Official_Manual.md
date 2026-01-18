# CamillaFIR – Official Manual (v2.7.3)

## 1. Overview
CamillaFIR is an AI-assisted DSP engine for generating high-resolution FIR correction filters
from acoustic measurements. It focuses on **time-domain correctness first**, then magnitude.

Key design goals:
- Preserve transients
- Avoid over-correction
- Produce physically meaningful acoustic metrics

---

## 2. Processing Pipeline (High Level)
1. Import REW magnitude + phase
2. Robust parsing and unit normalization
3. Psychoacoustic or standard smoothing
4. Time-of-Flight (TOF) detection & removal
5. Confidence analysis & reflection detection
6. Target curve construction
7. Smart leveling (Median / Average)
8. Magnitude correction with safety guards
9. Phase reconstruction (Linear / Min / Mixed / Asymmetric)
10. TDC (optional)
11. FIR synthesis + normalization
12. Multi-rate export

---

## 3. Filter Types (What Actually Happens)

### Linear Phase
- Full-spectrum excess phase correction
- Maximum transient precision
- Risk of pre-ringing above ~2 kHz

### Minimum Phase
- Magnitude-only correction
- Phase reconstructed via Hilbert transform
- No pre-ringing

### Mixed Phase
- Linear phase below split frequency
- Minimum phase above
- Unity magnitude sum at crossover

### Asymmetric Linear
- Linear phase with asymmetric time window
- Preserves impulse leading edge
- Greatly reduced audible pre-ringing

---

## 4. Acoustic Intelligence

### Confidence Mask
Each frequency bin is assigned a reliability score based on:
- Phase stability
- Reflection dominance
- Energy decay

This mask directly controls:
- FDW window length
- Smoothing strength
- Correction aggressiveness

### Reflection Analysis
Detected reflections report:
- Frequency
- Group delay error (ms)
- Physical distance (meters)

Distances are real because TOF is removed before analysis.

---

## 5. Temporal Decay Control (TDC)
TDC is **not EQ**.

It:
- Identifies excessive decay (RT60)
- Injects inverse decay impulses
- Reduces ringing time instead of SPL

Result:
Bass notes stop faster instead of just getting quieter.

---

## 6. Safety Systems
- Soft-clip boost limiting
- Independent max cut limit
- dB/oct slope limiting
- HPF & excursion protection
- Phase correction frequency limit

---

## 7. Outputs
- FIR WAV / TXT (IEEE float)
- Summary report (RT60, confidence, match %)
- Optional CamillaDSP YAML

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

For MiniDSP etc. users 

1. The Hardware Limitation (Taps)
The MiniDSP Flex typically supports a limited number of taps per output channel (e.g., 1024 or 2048 taps at 48kHz/96kHz).

* For Bass/Subwoofers: To linearize phase in the sub-bass region (e.g., 30Hz–80Hz), you typically need huge tap counts (e.g., 16,000+). With ~1024 taps, the frequency resolution is too low to effectively linearize sub-bass phase without causing significant artifacts or simply not working.
* Using FIR to linearize the subwoofers themselves on a Flex is likely not feasible due to the tap count constraint. Stick to IIR (PEQ) for the subwoofers.

2. The Solution: Linearizing the Crossover (Mains)
Since you cannot place FIR on the Inputs of the Flex, you have to treat the outputs. Here is the recommended workflow with CamillaFIR for your setup:

* Step A (Subs): Use standard IIR filters (PEQ) and Delay on the Flex to flatten the subwoofer response and time-align them to each other.
* Step B (Mains): Measure your Mains without the subwoofers playing.
* Step C (CamillaFIR): Generate a filter for the Mains.
    - Phase Correction: Use the tool to linearize the phase of the mains down to the crossover point.
    - Excursion Protection / Min Frequency: Set the optimization range (e.g., 'lvl_min' or 'hc_min') to start above your sub crossover (e.g., 80Hz).
    - Crossover Unwrapping: If you use a standard IIR crossover (e.g., LR24) on the Flex to cut the bass from the mains, that crossover introduces a phase shift. CamillaFIR will "see" this phase shift in the measurement and generate a FIR kernel to flatten it (unwrapping the phase).
* Step D (Integration): Once the Mains are phase-linear (thanks to FIR) and the Subs are standard IIR, use the Delay setting on the Mains (or Subs) in the MiniDSP to align the phase at the crossover point.

3. "Crossover Linearization" Tool
CamillaFIR also has a specific feature called Crossover Linearization. If you know you are using a specific filter type (e.g., Linkwitz-Riley 4th Order) on the Flex for your subs/mains split, you can generate a specific FIR kernel just to counteract that phase shift. This is very tap-efficient and works well on MiniDSP devices.

Summary:
Don't try to linearize the subwoofers with FIR on the Flex (lack of taps). Instead, use CamillaFIR on the Mains outputs to linearize the mains' phase response and "unwrap" the phase rotation caused by the high-pass crossover. This will give you a perfect impulse response from 80Hz upwards!
