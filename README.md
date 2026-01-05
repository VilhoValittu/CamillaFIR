

---

# 🎛️ CamillaFIR

**Advanced FIR Filter Generator for CamillaDSP**

CamillaFIR is a Python-based tool designed to generate studio-grade Finite Impulse Response (FIR) correction filters from acoustic measurements (e.g., REW exports). It provides a user-friendly Web GUI to analyze impulse responses, apply psychoacoustic smoothing, and generate correction filters with precise phase handling.

> Inspired by OCA [https://www.youtube.com/@ocaudiophile](https://www.youtube.com/@ocaudiophile)

## 🚀 Key Features

* **Browser-Based GUI:** Built with `PyWebIO`, offering an interactive interface for file uploads and parameter tuning.
* **Multi-Rate Export:** Automatically generates FIR filters for all standard sample rates (44.1kHz to 192kHz) from a single measurement.
* **Target Curve Integration:** Built-in support for Harman, B&K, and Toole curves, plus custom text file support.
* **Advanced Smoothing:** Supports standard fractional octave and variable **Psychoacoustic Smoothing**.
* **Protection Mechanisms:**
* **Excursion Protection:** Prevents boosting frequencies below the port tuning/roll-off point.
* **Soft-Knee Limiting:** Smoothly limits max gain to prevent digital clipping.
* **Regularization:** Frequency-dependent regularization to prevent over-correcting deep room nulls (anti-ringing).


* **CamillaDSP Ready:** Exports a ready-to-use `.yml` configuration snippet and `.wav` impulse files.

## 🧠 Technical Deep Dive: Phase Handling Strategies

CamillaFIR v2.1.0 implements state-of-the-art Digital Signal Processing (DSP) techniques to handle phase response, offering three distinct modes:

### 1. 📉 Linear Phase (The Purist)

* **Algorithm:** Corrects both Magnitude and Phase response across the entire spectrum.
* **Technique:** Calculates the **Excess Phase** of the room (measured phase minus minimum phase) and inverts it. This aligns the Group Delay perfectly, resulting in extremely tight bass response.
* **Fade-Out Logic:** Applies a frequency-dependent fade-out to the phase correction vector to prevent high-frequency "ringing" artifacts often associated with aggressive phase correction.
* **Trade-off:** High latency (typically >100ms) and potential pre-ringing on sharp transients (snare drums, claps).

### 2. ⚡ Minimum Phase (The Gamer / Cinema)

* **Algorithm:** REW-style reconstruction. The tool calculates the ideal magnitude response (Target - Measurement) and then mathematically generates the corresponding Minimum Phase response using the **Hilbert Transform**.
* **Impulse Centering:** Features an intelligent "Impulse Centering" algorithm that detects the true peak of the measurement—regardless of time-of-flight delays—ensuring the analysis window never clips the signal data.
* **Result:** **Zero Latency** and **Zero Pre-ringing**. Perfect for gaming, lip-sync video, and monitoring.

### 3. 🧬 Mixed Phase (The Audiophile Choice)

* **Algorithm:** A hybrid convolution approach that combines the best of both worlds.
* **Low Frequencies (< 300Hz):** Uses **Linear Phase with Excess Phase Correction**. This corrects the time-domain errors caused by crossovers and port resonances, resulting in "fast" and impactful bass.
* **High Frequencies (> 300Hz):** Switches to **Minimum Phase**. This ensures that high-frequency transients remain pristine and free from pre-ringing artifacts.
* **Implementation:** The software generates two separate filters internally and stitches them together in the frequency domain, creating a single coherent impulse response file.

## 🛠️ Signal Processing Chain

1. **Ingestion:** Parses raw `.txt` measurement data (Frequency, Magnitude, Phase).
2. **Preprocessing:** Applies impulse centering to handle acoustic delay.
3. **Smoothing:** Applies Frequency Dependent Windowing (FDW) or Psychoacoustic smoothing to separate direct sound from room reflections.
4. **Target Matching:** Aligns the measurement to the target curve using Median or Mean energy analysis.
5. **Gain Calculation:** Calculates `Target - Measured`. Applies soft-clipping and regularization.
6. **Phase Generation:** Computes the complex conjugate based on the selected mode (Linear/Min/Mixed).
7. **IFFT & Windowing:** Performs Inverse Fast Fourier Transform to convert data back to the time domain and applies a Tuckey or Hann window to ensure the filter decays to absolute zero.

## 📦 Installation & Usage

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/CamillaFIR.git
cd CamillaFIR

```


2. **Install dependencies:**
```bash
pip install numpy scipy matplotlib plotly pywebio

```


3. **Run the tool:**
```bash
python CamillaFIR.py

```


The GUI will automatically open in your default web browser (usually http://localhost:8080).

## 📊 Visualizations

The tool generates detailed reports including:

* **Magnitude & Phase Plots:** Comparing Original, Predicted, and Target responses.
* **Filter Response:** Visualizing exactly what the FIR filter is doing.
* **Step Response / Group Delay:** (In summary logs).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---
EXE file : https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI
---
