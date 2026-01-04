# 🎛️ CamillaFIR

**Automated Audiophile-Grade FIR Filter Generator**

**CamillaFIR** is a Python-based tool designed to bridge the gap between acoustic measurements (REW) and convolution engines like **CamillaDSP**, **Equalizer APO**, **Volumio**, or **Roon**.

Unlike complex manual tools (like rePhase), CamillaFIR automates the heavy DSP math, offering a modern, browser-based GUI to generate phase-accurate FIR filters in seconds.



## 🚀 Key Features (v1.5.0)

* **Frequency Dependent Windowing (FDW):** intelligently separates direct sound from room reflections. Corrects bass modes heavily while leaving high-frequency airiness natural.
* **Two Filter Modes:**
* **Linear Phase:** Corrects phase timing errors and unwinds crossover phase shifts (Audiofile quality, high latency).
* **Minimum Phase:** Zero latency correction suitable for **Gaming, TV, and Live monitoring**.


* **Psychoacoustic Smoothing:** mimics human hearing to prevent over-correction of narrow dips, saving amplifier headroom.
* **Advanced DSP:** Uses Hilbert Transforms to separate Minimum Phase from Excess Phase.
* **Safety First:** Hard-coded limits prevent dangerous boosts that could damage tweeters.
* **High-Res Support:** Supports sample rates from **44.1 kHz** up to **384 kHz** (great for DSD upsampling).
* **Web-Based GUI:** Runs locally in your browser using `PyWebIO`.
* **Auto-Save:** Remembers your measurement file paths and settings automatically.

## 🛠️ Installation

### Prerequisites

You need **Python 3.x** installed on your system.

### 1. Clone the repository

```bash
git clone https://github.com/YourUsername/CamillaFIR.git
cd CamillaFIR

```

### 2. Install dependencies

Run the following command to install the required libraries:

```bash
pip install numpy scipy matplotlib pywebio

```

## 📖 How to Use

1. **Export Measurements:**
* Measure your room using **REW (Room EQ Wizard)**.
* Export measurements as text files: `File` -> `Export` -> `Export measurement as text`.
* Save Left and Right channels separately (e.g., `L.txt`, `R.txt`).


2. **Run the Tool:**
```bash
python CamillaFIR.py

```


* The tool will automatically open in your default web browser (usually at `http://localhost:8080`).


3. **Generate Filters:**
* **Upload** your `.txt` measurement files (or paste the local file path).
* Choose your target **Sample Rate** and **Taps** (Use the built-in guide if unsure).
* Select **Filter Type** (Linear Phase for music, Minimum Phase for video/gaming).
* Click **Submit**.


4. **Result:**
* The tool calculates the filters and displays the predicted **Frequency** and **Phase** response.
* The FIR filters are saved as `.wav` (or `.csv`) files in the project folder.
* Load these files into your convolution engine (e.g., CamillaDSP pipeline).



## ⚙️ Advanced Settings

* **FDW Cycles:** Controls the windowing "aggressiveness".
* *15 (Default):* Balanced correction.
* *3-6:* Very dry, removes almost all room reverb (Nearfield).
* *30+:* Gentle, includes room character.


* **Crossover Linearization:** If you know your speaker's passive crossover points (e.g., 2000Hz 4th order), enter them to "unwind" the phase shift mathematically.
* **House Curve:** Use the built-in target curve (Harman-like) or upload your own text file.

## 🌍 Language Support

The tool automatically detects your system language.

* 🇫🇮 **Finnish** (Detected if OS locale is Finland)
* 🇬🇧 **English** (Default for everyone else)

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

[MIT](https://choosealicense.com/licenses/mit/)

---

**Created by:** VilhoValittu & GeminiPro

---
EXE-file : https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI

---
