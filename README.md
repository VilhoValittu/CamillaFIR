# CamillaFIR
Automated FIR filter generator for REW measurements. Creates phase-linear correction files (WAV/CSV) for Equalizer APO, Roon, and CamillaDSP. Features crossover linearization, smart room correction, and safe subsonic filtering. By VilhoValittu &amp; GeminiPro. Inspired by OCA https://www.youtube.com/@ocaudiophile

Hi everyone,

I wanted to share a new tool developed by "VilhoValittu & GeminiPro". It's a Python-based utility designed to automate the creation of FIR correction filters from REW measurements.

Creating phase-linear FIR filters manually (e.g., in rePhase) acts as a bottleneck for many. This tool aims to streamline the process, creating convolution files ready for Equalizer APO, Roon, CamillaDSP, etc., in seconds.

Key Features:

Automated Phase Linearization: Corrects the phase shift of your existing IIR crossovers (just input freq & slope).

Smart Room Correction: Applies frequency response correction based on a target House Curve with adjustable max boost limits.

Configurable Taps: Choose from 2048 up to 131,072 taps (balancing latency vs. bass resolution).

Safe Subsonic Filter: Optional High Pass Filter (10-60Hz) implemented as Minimum Phase to prevent pre-ringing artifacts in the bass.

Auto-Leveling: Matches the House Curve to your speaker's natural response to avoid drastic gain jumps.

Output Formats:

WAV: 32-bit float (Mono or Stereo files).

CSV: Ready-to-use coefficients for Equalizer APO.

Deep Dive: How the Phase Correction Works Unlike simple auto-EQs that try to flatten the measured phase blindly (often causing severe pre-ringing artifacts), this tool uses a Hybrid Approach:

Theoretical Linearization: First, it calculates the mathematical inverse of the IIR crossovers you specify (e.g., LR4 @ 2000Hz). This unwraps the phase shift caused by your existing crossovers purely based on math, guaranteeing zero artifacts for this part of the correction.

Measured Fine-Tuning: It then looks at your actual measurement data (Excess Phase). It applies a secondary correction to align the driver's natural phase deviations.

Safety Clamping: This fine-tuning is strictly clamped (max ±45 degrees). This ensures the tool never tries to correct room reflections or measurement noise, which is the #1 cause of bad-sounding FIR filters.

The result is a step response that looks like a single coherent spike, improving transient attack and soundstage depth without the "processed" sound of aggressive room correction.

Workflow:

Measure in REW.

Export measurements as text files (L.txt & R.txt) into the same folder as the tool.

Note: If using a House Curve, place that .txt file in the same folder as well.

Run the generator.

Select your preferences (Sample rate, House curve, Output format).

Load the generated file into your convolution engine.


Feedback is welcome!

EXE file available https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI?usp=sharing

CamillaFIR v1.0 - The GUI & Analysis Update

This major release transitions CamillaFIR from a CLI utility to a full-featured graphical application with built-in analysis tools.

Key Changes:

Web-Based GUI: Now uses a browser-based interface for easy configuration of all parameters (Crossovers, House Curves, Taps, etc.).

Prediction Plots: Generates visual feedback after processing, showing both Magnitude and Phase responses (Original vs. Predicted).

Smart Phase Analysis: The phase plot automatically calculates and removes IR delay (centers the peak), allowing for a readable, unwrapped view of phase linearization.

Performance Optimization: Switched plotting calculations to FFT-based methods for instant rendering even at high tap counts (e.g., 131k).

Enhanced UX: Dedicated input fields for up to 5 crossovers, toggleable magnitude correction, and built-in default house curves.



```markdown
# CamillaFIR

**CamillaFIR** is a Python-based tool for generating FIR filters for speaker correction. It features a graphical user interface (GUI), phase linearization, house curve application, and predicted response analysis.

## 🚀 Installation & Running

### Prerequisites
* **Python 3.8** or newer.

---

### 1. Install Python

If you don't have Python installed, follow these steps:

#### 🪟 Windows
1. Download the installer from [python.org](https://www.python.org/downloads/).
2. Run the installer.
3. **Important:** Check the box **"Add Python to PATH"** at the bottom of the installer before clicking "Install Now".

#### 🍎 macOS
The easiest way is using Homebrew. Open your Terminal and run:
```bash
brew install python

```

Alternatively, download the installer from [python.org](https://www.python.org/downloads/macos/).

#### 🐧 Linux

Run the following in your terminal (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install python3 python3-pip

```

---

### 2. Install Dependencies

Open your terminal or command prompt in the project folder and run:

```bash
pip install -r requirements.txt

```

*(Note: On Linux/macOS, you might need to use `pip3` instead of `pip`).*

---

### 3. Run the Application

Start the program by running the script. It will launch a local web server and automatically open the interface in your default browser.

```bash
python CamillaFIR.py

```

*(Note: On Linux/macOS, you might need to use `python3 CamillaFIR.py`).*

```

