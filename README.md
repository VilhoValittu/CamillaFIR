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
# Installation & Running

## 1. Install Python
CamillaFIR requires **Python 3.8** or newer. If you don't have Python installed, follow the instructions for your operating system:

### <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Windows_10_Logo.svg" width="20"/> Windows
1. Download the latest Python installer from [python.org](https://www.python.org/downloads/).
2. Run the installer.
3. **Important:** Ensure you check the box **"Add Python to PATH"** at the bottom of the installer window before clicking "Install Now".

### <img src="https://upload.wikimedia.org/wikipedia/commons/2/21/MacOS_wordmark_%282017%29.svg" width="35"/> macOS
The easiest way is using Homebrew. Open your Terminal and run:
```bash
brew install python

```

Alternatively, download the macOS installer from [python.org](https://www.python.org/downloads/macos/).

### <img src="https://www.google.com/search?q=https://upload.wikimedia.org/wikipedia/commons/3/35/Tux.svg" width="20"/> Linux

Python 3 is usually pre-installed on most Linux distributions. You can ensure you have Python and `pip` (package manager) installed by running:

* **Debian/Ubuntu:**
```bash
sudo apt update
sudo apt install python3 python3-pip

```


* **Fedora:**
```bash
sudo dnf install python3

```



---

## 2. Install Dependencies

Open your terminal or command prompt (CMD/PowerShell) in the project folder and run:

```bash
pip install -r requirements.txt

```

*(Note: On Linux/macOS, you might need to use `pip3` instead of `pip`).*

---

## 3. Run the Application

Start the program by running the script. This will launch a local web server and automatically open the interface in your default browser.

```bash
python CamillaFIR.py

```

*(Note: On Linux/macOS, you might need to use `python3 CamillaFIR.py`).*

```

```
