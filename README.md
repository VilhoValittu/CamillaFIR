# CamillaFIR

**CamillaFIR** is a powerful, Python-based GUI tool designed to generate FIR (Finite Impulse Response) correction filters specifically for **CamillaDSP**. It streamlines the process of converting Room EQ Wizard (REW) measurements into convolution files, handling complex DSP tasks like minimum phase calculation, frequency-dependent windowing, and multi-rate resampling automatically.

## 🚀 Key Features

* **Browser-Based GUI:** Built with `PyWebIO` for a responsive, modern interface that runs locally.
* **Multi-Rate Generation:** Automatically generates FIR filters for all standard sample rates (44.1kHz to 192kHz) from a single measurement, scaling tap count to maintain frequency resolution.
* **Advanced Phase Handling:**
* **Linear Phase:** Corrects magnitude while maintaining phase linearity (at the cost of latency).
* **Minimum Phase:** Calculates a causal minimum phase response using the **Hilbert Transform** method, eliminating pre-ringing and providing zero-latency filtering (ideal for gaming/video).


* **Psychoacoustic Smoothing:** Applies variable smoothing that mimics human hearing perception (less smoothing at low frequencies, more at high frequencies).
* **Regularization (Dip Limiting):** Intelligent algorithm to prevent over-correction of sharp spectral dips (room nulls), reducing ringing artifacts and amplifier strain.
* **Frequency Dependent Windowing (FDW):** Reduces the influence of room reflections in the measurement by narrowing the time window as frequency increases.
* **Interactive Visualization:** Fully interactive **Plotly** charts for analyzing Magnitude, Phase, and Filter responses directly in the browser.

## 🛠 Technical Details

### Minimum Phase Calculation via Hilbert Transform

Unlike simple magnitude inversion, CamillaFIR derives a mathematically correct minimum phase response to ensure causality and zero latency.

1. Compute the natural logarithm of the magnitude spectrum: .
2. Construct the full symmetric spectrum.
3. Apply the **Hilbert Transform** to the log-magnitude.
4. The minimum phase (in radians) is extracted from the negative imaginary part of the analytic signal: .

### Regularization Algorithm

To avoid "over-cooking" the filter by trying to boost deep nulls (which are often physical cancellations that cannot be EQ'd), CamillaFIR uses a comparative smoothing approach:

1. Calculates the raw required gain (Target - Measured).
2. Calculates a heavily smoothed version of the gain curve (Gaussian filter).
3. If the raw gain exceeds the smoothed gain significantly (indicating a sharp, narrow dip), the boost is attenuated based on the user-defined `Regularization Strength %`.

### Multi-Rate Scaling

When generating filters for higher sample rates (e.g., 192kHz) based on a 48kHz measurement:

* The tap count is automatically scaled (e.g., 65k taps @ 48kHz  262k taps @ 192kHz).
* This ensures that the frequency resolution (bin width) remains constant across all sample rates, preserving bass correction accuracy.

## 📦 Installation & Usage

### Prerequisites

* Python 3.8+
* The following Python packages:

```bash
pip install numpy scipy matplotlib plotly pywebio

```

### Running the Application

1. Clone the repository or download `CamillaFIR.py`.
2. Run the script:
```bash
python CamillaFIR.py

```


3. The GUI will automatically open in your default web browser (usually at `http://localhost:8080`).

## ⚙️ Configuration Guide

### Input Files

* **Measurements:** Accepts exported text files (`.txt`) from REW (Room EQ Wizard). Format: `Frequency, SPL, Phase`.
* **House Curve:** Built-in targets (Harman, B&K, Flat, Cinema) or upload a custom `.txt` target curve.

### DSP Settings

* **Taps:** The length of the FIR filter. Standard is **65536** (for 44.1/48kHz). Higher taps = better bass resolution but higher latency (in Linear Phase).
* **Filter Type:**
* *Linear:* Best for music playback.
* *Minimum:* Best for real-time monitoring, gaming, or lip-sync critical video.


* **FDW Cycles:** Controls the window width.
* *15:* Standard for room correction.
* *5:* Forced automatically for Minimum Phase to ensure stable low-end.


* **Regularization:** **30-50%** is recommended to keep the sound natural and avoid ringing in the bass region.

## 📂 Output

The program generates a ZIP archive in the script's directory containing:

1. **Convolution Files (.wav):** Stereo or Mono impulse response files named by sample rate (e.g., `Stereo_corr_Harman_Linear_96000Hz.wav`).
2. **HTML Plots:** Interactive Plotly visualizations for detailed analysis.
3. **Summary.txt:** A report of the applied gain, delays, and peak levels.

## 📄 License

[MIT License](https://www.google.com/search?q=LICENSE) (or your preferred license).

---

*Created by VilhoValittu & GeminiPro.*

---
EXE file : https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI
---
