
---

# CamillaFIR

**Automated Universal FIR Filter Generator**

CamillaFIR is a Python-based DSP utility designed to automate the creation of high-precision Finite Impulse Response (FIR) correction filters based on acoustic measurements. It leverages **PyWebIO** for a browser-based interface and **SciPy/NumPy** for heavy mathematical lifting.

While originally designed for **CamillaDSP** (including automated `.yml` config generation), the tool generates standard IEEE 32-bit float `.wav` impulse response files. These filters are **universally compatible** with any DSP hardware or software that supports convolution, including:

* **MiniDSP** (OpenDRC, Flex, SHD, etc.)
* **Equalizer APO** (Windows System-wide EQ)
* **Roon** (DSP Engine / Convolution)
* **Volumio / Moode / FusionDSP**
* **JRiver Media Center**
* **CamillaDSP**

---

## 🚀 Technical Capabilities

### DSP Engine (v2.5.0)

The core processing engine performs complex domain signal processing to derive correction impulses.

* **Time-of-Flight (TOF) Correction:** Automatically detects and removes acoustic delay (distance-based phase slope) using linear regression analysis on the unwrapped phase between 1kHz and 10kHz. This ensures "Excess Phase" calculations target only group delay distortions, not propagation delay.
* **Frequency-Dependent Regularization:** Applies variable regularization strength across the frequency spectrum to prevent over-correction of high-Q nulls and measurement anomalies:
* **< 200 Hz:** Low regularization (50% of base) for aggressive modal correction.
* **200 Hz - 2 kHz:** Linear ramp.
* **> 2 kHz:** High regularization to preserve natural "air" and prevent phase hash.


* **Psychoacoustic Smoothing:** Implements a hybrid smoothing algorithm that simulates human hearing integration windows (variable octave smoothing with peak hold).
* **Excursion Protection:** Hard-knee limiting of gain below user-defined frequencies (e.g., < 40Hz) to protect woofer mechanics.

### Phase Strategies

1. **Linear Phase:** Corrects both magnitude and phase (Group Delay) across the entire bandwidth. Uses TOF-corrected excess phase inversion.
2. **Minimum Phase:** Corrects magnitude only. Phase is mathematically derived via the **Hilbert Transform**, ensuring zero pre-ringing and minimal latency (causal).
3. **Mixed Phase (Hybrid):** Splits the correction at a transition frequency (default 300Hz).
* **Low Frequency:** Linear Phase for tight time-alignment of bass transients.
* **High Frequency:** Minimum Phase for natural decay and zero pre-ringing on treble transients.
* *Reconstruction:* Sub-band recombination is performed using 4th order Linkwitz-Riley filters in the complex domain.



---

## 🛠️ Installation

### Prerequisites

To run this tool, you need **Python 3.10** or newer installed on your system.

#### How to Install Python

* **Windows:**
1. Download the installer from [python.org](https://www.python.org/downloads/).
2. Run the installer and **check the box "Add Python to PATH"** at the bottom of the first screen.
3. Click "Install Now".


* **macOS:**
1. Download the installer from [python.org](https://www.python.org/downloads/macos/).
2. Follow the installation prompts.
3. Alternatively, if you have Homebrew: `brew install python`


* **Linux:**
* Python is usually pre-installed. Verify with `python3 --version`.
* If missing: `sudo apt update && sudo apt install python3 python3-pip python3-venv`



### 1. Clone the Repository

You can download the code as a ZIP file from GitHub or use Git:

```bash
git clone https://github.com/VilhoValittu/CamillaFIR.git
cd CamillaFIR

```

### 2. Install Dependencies

It is highly recommended to run this tool in a virtual environment to keep your system clean.

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install numpy scipy pywebio matplotlib requests

```

**Linux / macOS:**

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


2. The GUI will automatically open in your default web browser (usually at `http://localhost:8080`).
3. **Workflow:**
* **Files:** Upload your Left/Right measurement exports (TXT format from REW).
* **Settings:** Select your target curve (Harman, Toole, etc.) and filter type (Mixed/Linear/Min).
* **Run:** Click "Run Analysis".


4. **Output:** The tool generates a `.zip` archive containing:
* **Universal Stereo FIR filters** (`.wav` 32-bit float) for multiple sample rates (44.1k - 192k). These files can be loaded into any convolution engine.
* `camilladsp.yml` configuration snippet (specifically for CamillaDSP users).
* Analysis plots (Magnitude, Phase, Group Delay, Step Response).



---

## 📊 Input Data Format

CamillaFIR expects standard frequency response text exports (e.g., from Room EQ Wizard):

* Format: `Frequency (Hz) | Magnitude (dB) | Phase (deg)`
* No headers required, comments marked with `*` or `#` are ignored.

---

## Acknowledgements

Inspired by OCA [https://www.youtube.com/@ocaudiophile](https://www.youtube.com/@ocaudiophile)

---

## 📥 Binary Download

For Windows users who do not wish to install Python environments:

**EXE-file available [https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI**](https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI)
