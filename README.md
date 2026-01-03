Tässä on kattava ja ammattimainen **README.md** -pohja GitHub-repositoriollesi. Se korostaa ohjelman uusia "Pro-tason" ominaisuuksia (v1.3.0), kuten FDW:tä ja psykoakustista silotusta.

Voit kopioida tämän tekstin suoraan GitHubiin.

---

# 🎛️ CamillaFIR

**CamillaFIR** is a Python-based tool designed to generate high-quality Finite Impulse Response (FIR) correction filters for active loudspeakers. It bridges the gap between raw measurements (from REW) and convolution engines like **CamillaDSP**, **Equalizer APO**, or **MiniDSP**.

Unlike simple "invert the curve" tools, CamillaFIR uses advanced DSP techniques—such as **Frequency Dependent Windowing (FDW)**, **Hilbert Transforms**, and **Psychoacoustic Smoothing**—to create filters that sound natural, phase-linear, and transient-perfect.

---

## ✨ Key Features

### 1. Advanced Phase Linearization 🧠

CamillaFIR doesn't just flatten the phase; it understands it.

* **Hilbert Transform Analysis:** Automatically separates the measured phase into **Minimum Phase** (system/driver behavior) and **Excess Phase** (time delays/crossovers).
* **Frequency Dependent Windowing (FDW):** Applies variable windowing to the Excess Phase. It corrects phase anomalies in the bass (room modes) while leaving the treble "air" and natural reflections untouched to avoid pre-ringing artifacts.
* **Crossover Reversal:** Mathematically "unwinds" the phase shift caused by your analog or IIR crossovers (Linkwitz-Riley, Butterworth, etc.), resulting in a perfect step response.

### 2. Psychoacoustic Magnitude Correction 👂

* **Smart Smoothing:** Uses a **Psychoacoustic Smoothing** algorithm that preserves audible peaks (which need cutting) but fills in narrow dips (which shouldn't be boosted). This saves amplifier headroom and prevents unnatural sound.
* **House Curves:** Built-in "Harman-style" house curve or upload your own custom target curve.
* **Safety Limits:** Hard-coded **8dB safety limit** on boosts to protect your drivers and amplifiers, regardless of user input.

### 3. Modern Web GUI 🖥️

* **Browser-Based Interface:** No command-line arguments needed. Just launch and configure in your web browser.
* **Auto-Save:** Remembers your crossovers, gain settings, and preferences automatically (`config.json`).
* **Instant Analysis:** Displays predicted **Frequency Response** and **Phase Response** graphs immediately after generation.
* **Smart Plotting:** Automatically estimates and removes IR delay for readable, unwrapped phase plots.

### 4. Flexible Output 💾

* **Dual Channel:** Processes Left and Right channels independently in one go.
* **Formats:** Exports to `.wav` (Stereo/Mono) or `.csv` text files.
* **Universal:** Works with any sample rate (44.1k, 48k, 96k...) and tap count (up to 262k+).

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or newer
* Measurements exported from **REW (Room EQ Wizard)** as text files (`.txt`).

### Installation

1. Clone the repository or download the source code.
2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


*(Dependencies: `numpy`, `scipy`, `matplotlib`, `pywebio`)*

### Usage

1. Run the application:
```bash
python CamillaFIR.py

```


2. Your browser will open automatically.
3. **Upload** your Left and Right measurement files.
4. **Configure** your target settings (House curve, Crossovers, FDW cycles).
5. **Generate** and download your FIR filters.

---

## ⚙️ DSP Pipeline Explained

How CamillaFIR processes your sound:

1. **Input:** Reads raw magnitude and phase data from measurement files.
2. **Smoothing:** Applies Psychoacoustic smoothing to the magnitude response.
3. **Normalization:** Aligns the Target Curve (House Curve) to match the measurement level at the correction limit.
4. **Phase Extraction:** Calculates the theoretical **Minimum Phase** from the smoothed magnitude using the Hilbert Transform (Cepstral method).
5. **Excess Phase Calculation:** Subtracts Minimum Phase from the Measured Phase to isolate time-domain errors.
6. **FDW Processing:** Applies Frequency Dependent Windowing to the Excess Phase to distinguish direct sound from room reflections.
7. **Filter Generation:** Combines the EQ curve (Magnitude) and the Corrected Phase (FDW + Crossover reversal) into a complex signal.
8. **IFFT:** Performs an Inverse FFT to create the time-domain Impulse Response (the FIR filter).

---

## 📦 Building a Standalone EXE (Windows)

If you want to distribute the tool as a single executable file without requiring Python installed:

```bash
python -m PyInstaller --noconfirm --onefile --console --name "CamillaFIR_1.3.0" --hidden-import=pywebio.platform.tornado --hidden-import=matplotlib --hidden-import=matplotlib.backends.backend_agg CamillaFIR.py

```

---

## License

MIT License. Feel free to fork, modify, and contribute!

## EXE
Exe-file : https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI
