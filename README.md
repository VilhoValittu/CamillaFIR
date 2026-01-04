# 🎛️ CamillaFIR

**Automated Audiophile-Grade FIR Filter Generator**

**CamillaFIR** is a powerful Python tool designed to bridge the gap between acoustic measurements (e.g., from REW) and convolution engines like **CamillaDSP**, **Equalizer APO**, **Volumio**, or **Roon**.

Unlike complex manual tools (such as rePhase) that require manual amplitude and phase adjustments, CamillaFIR **automates the heavy DSP math**. It analyzes your room measurements and generates phase-accurate FIR filters in seconds using a modern, browser-based interface.

Inspired by OCA https://www.youtube.com/@ocaudiophile

## 🚀 Key Features

* ** automated DSP Pipeline:** Transforms raw REW text exports into ready-to-use `.wav` or `.csv` convolution filters.
* **Frequency Dependent Windowing (FDW):** Intelligently separates direct sound from room reflections. It corrects steady-state bass problems while preserving the natural "airiness" and transient response of the treble.
* **Two Filter Modes:**
* **Linear Phase:** Corrects phase timing errors and unwinds crossover phase shifts. (Best for critical music listening).
* **Minimum Phase:** Zero-latency correction without pre-ringing. (Best for **Gaming, TV/Lip-sync, and Live monitoring**).


* **Smart Level Matching:** automatically aligns the target curve to your measurement using robust algorithms (Median or Average) to prevent excessive cuts or dangerous digital boosts.
* **Psychoacoustic Analysis:** Plots use VAR-smoothing (Variable Audio Resolution) to show how human hearing perceives the response, rather than raw messy data.
* **High-Res Support:** Supports sample rates from **44.1 kHz** up to **384 kHz** (perfect for HQPlayer / DSD upsampling pipelines).
* **Protection:** Built-in High-Pass Filter (HPF) and hard-coded gain limits to protect your equipment.
* **Modern GUI:** Runs locally in your web browser using `PyWebIO`.

## 🛠️ Installation

### Prerequisites

You need **Python 3.x** installed on your system.

### 1. Clone the repository

```bash
git clone https://github.com/VilhoValittu/CamillaFIR.git
cd CamillaFIR

```

### 2. Install dependencies

Install the required Python libraries:

```bash
pip install numpy scipy matplotlib pywebio

```

## 📖 How to Use

1. **Export Measurements:**
* Measure your speakers using **REW (Room EQ Wizard)**.
* Go to `File` -> `Export` -> `Export measurement as text`.
* Save Left and Right channels as separate `.txt` files.


2. **Run the Tool:**
```bash
python CamillaFIR.py

```


* The tool will automatically launch in your default web browser (usually `http://localhost:8080`).


3. **Generate Filters:**
* **Upload** your measurement files.
* Select your target **Sample Rate** (e.g., 44100 Hz, 192000 Hz).
* **Taps:** Choose filter length (Rule of thumb: Double sample rate = Double taps).
* **Filter Type:** Choose *Linear Phase* (Music) or *Minimum Phase* (Low Latency).
* **Target Curve:** Use the built-in Harman-like curve or upload your own.
* Click **Submit**.


4. **Result:**
* View the predicted **Frequency** and **Phase** response graphs.
* The tool saves the FIR filters (e.g., `Stereo_corr_48000Hz_....wav`) in the project folder.
* Load these files into your convolution engine.



## ⚙️ Advanced DSP Explained

* **FDW (Cycles):** Controls the "window" of time used for correction.
* *Low (3-6):* Very aggressive, "dry" sound. Removes almost all room reverb from the correction.
* *Standard (15):* Balanced. Corrects bass modes but respects the room's natural decay in highs.


* **Crossover Linearization:** If you input your speaker's existing passive crossover points (e.g., 2000 Hz 4th order), CamillaFIR creates a reverse-phase curve to mathematically "unwind" the phase shift, resulting in a near-perfect step response.
* **Level Match Algo:**
* *Median:* (Recommended) Ignores extreme peaks/nulls when calculating volume.
* *Average:* Traditional RMS average.



## 🌍 Language Support

The interface automatically adapts to your system language:

* 🇫🇮 **Finnish** (Detected automatically)
* 🇬🇧 **English** (Default)

## 🤝 Contributing

Pull requests are welcome! If you have ideas for new DSP features or GUI improvements, feel free to open an issue.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)

---

**Created by:** VilhoValittu & GeminiPro

---
EXE file : https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI
---
