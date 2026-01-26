# CamillaFIR by Vilho Valittu

## v2.8.0 (current / upcoming)

### Plot export & reproducibility improvements

- **Dashboard export as static PNG**
  - Filter dashboards are now exported to ZIP files as static **PNG images** instead of interactive HTML.
  - The PNG output is generated using Plotly’s native Kaleido backend and **matches exactly** the HTML dashboard
    “Download plot as PNG” output.
  - Exported ZIP files no longer depend on `plotly.min.js`, browser security policies, or local file handling.

- **Improved offline & cross-platform compatibility**
  - Exported results open identically on Windows, macOS, and Linux without a browser.
  - Eliminates issues with missing JS assets, blocked local scripts, or CDN availability.

- **More robust packaging (Windows exe)**
  - Kaleido and its runtime dependencies are now explicitly included in the PyInstaller build.
  - Ensures PNG export works reliably in standalone executables.

### Notes
- No changes to FIR magnitude, phase, leveling, or correction logic.
- Visualization changes only affect exported artifacts, not DSP behavior.

---

 ## v2.7.9

### Stability & export fixes

- **House curve upload fix**
  - Fixed an issue where custom (user-uploaded) house curves could fail to load or apply correctly.
  - Improves consistency between UI preview, DSP processing, and saved configurations.

- **Plot export bugfix**
  - Fixed a broken Plotly PNG export path caused by an invalid `try/except` structure.
  - Ensures Plotly image export works correctly when generating offline artifacts.

### Notes
- Safe maintenance release.
- No changes to filter generation, phase handling, or auto-leveling behavior.

## What’s new in v2.7.8

- **Stereo-linked auto-leveling (TXT-compatible default)**  
  SmartScan level window and gain are computed from a shared L/R reference and applied identically to both channels.  
  This removes channel-dependent gain drift while preserving automatic delay alignment.

- **Correction-band visualization**  
  The active magnitude correction range (`mag_c_min … mag_c_max`) is shaded in plots, making it immediately obvious where correction is applied.

- **Reliability / confidence shading**  
  Low-confidence frequency regions are visually highlighted in plots to explain why some areas are protected or only lightly corrected.


**Time-domain–first FIR room correction (CamillaDSP-focused)**

CamillaFIR generates high-resolution **FIR correction filters** from REW exports (magnitude + phase).
Instead of treating everything as “EQ”, it separates three physical phenomena and corrects each with the right DSP method:

- **Propagation delay (Time-of-Flight / TOF):** removed explicitly before phase analysis
- **Excess phase distortion:** corrected with FIR phase reconstruction (Linear / Minimum / Mixed / Asymmetric)
- **Room-induced energy storage (room modes):** reduced with **Temporal Decay Control (TDC)** (time-domain, not amplitude EQ)


---

## Acknowledgements

Development inspired by the methodologies of **OCA** (Obsessive Compulsive Audiophile): [https://www.youtube.com/@ocaudiophile](https://www.youtube.com/@ocaudiophile)

---

**[Download Standalone EXE](https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI)**

---


## What you get

- FIR filters exported as **WAV (32-bit float)** or text
- Optional **CamillaDSP YAML** snippet
- Plots and a **Summary.txt** report (confidence, RT60, match score, effective A-FDW bandwidth, safety limits)
- Multi-rate export for common sample rates (44.1/48/88.2/96/176.4/192 kHz)

---

## Quickstart (source)

```bash
git clone https://github.com/VilhoValittu/CamillaFIR.git
cd CamillaFIR
python -m venv venv
# Windows:
#   .\venv\Scripts\activate
# Linux/macOS:
#   source venv/bin/activate
pip install -r requirements.txt
python camillafir.py
```

The UI opens in your browser (default: `http://localhost:8080`).

---

## Key features (v2.7.7)

- **2058-safe phase mode**: disables room phase correction (confidence/FDW/excess-phase) and uses only theoretical XO phase and minimum-phase where applicable. Good when phase/GD looks “spiky”.
- **Independent slope limits for boost vs cut** (dB/oct): keeps gentle boosts from being flattened while still preventing wild cut shapes.
- **TDC safety brakes**: hard cap for total TDC reduction plus optional slope limit for a predictable reduction curve.
- **DF smoothing (experimental)**: keeps smoothing width roughly constant in Hz across different fs/taps (useful for comparable “detail level” across sample rates).
- **Comparison mode**: locks scoring/plots to a fixed analysis grid (fs/taps) so A/B comparisons remain meaningful.
- **Multi-rate auto-taps mapping**: keeps FIR time-length constant across sample rates (44.1 kHz reference).

---

## Documentation

- 📘 User & technical manual → `docs/Official_Manual.md`
- 🧠 Why this works → `docs/Why_CamillaFIR_Works.md`
- 📐 Academic DSP rationale (math) → `docs/Academic_DSP_Explanation.md`
- 🔁 Stability & reproducibility → `docs/Stability_and_Reproducibility.md`
- ⚖️ Comparison vs conventional EQ → `docs/Comparison_vs_EQ.md`

---

## Screenshot

![Effect of Temporal Decay Control](tdc_impulse_example.png)
