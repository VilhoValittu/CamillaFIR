# CamillaFIR 

# Please update to version 2.7.7 (20.01.2026). Fixed many comma mistakes = now everything really works!

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
