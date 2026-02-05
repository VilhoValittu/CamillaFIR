# CamillaFIR by Vilho Valittu

## v2.8.9

### Phase correction safety (v2.8.8)

- **Internal phase correction clamp (±45°)**
  - Room-induced *excess phase* correction is now internally limited to **±45 degrees**.
  - The clamp is applied **only to the correction component**
    (measured phase minus target phase),
    and **never** to:
    - loudspeaker minimum-phase
    - theoretical crossover phase
    - user-defined target behavior
  - Prevents extreme phase rotations, excessive pre-ringing,
    and unstable group delay in low-confidence regions.
  - Improves subjective transient clarity and overall robustness,
    especially with sparse or noisy measurements.

- **Always reported**
  - Phase correction clamp status is:
    - logged during processing
    - included in `Summary.txt` per channel
    - shown in the **DSP info** section in the UI
  - Example:
    ```
    Phase Correction Clamp: max=54.5° -> 45.0°
    ```

- **No user control by design**
  - This is a fixed safety default.
  - Advanced users still retain full control over:
    - phase mode (Linear / Minimum / Mixed / Asymmetric)
    - confidence masking
    - A-FDW behavior
    - correction bandwidth

### REW-style IR windowing (DSP & export) (v.2.8.6)

- **REW-compatible IR windowing added to DSP export path**
  - Enables REW-style **symmetric** and **asymmetric (causal)** IR windowing during FIR export.
  - Windowing is applied **only at IR export stage** (WAV generation), not during correction, target fitting, leveling, or scoring.

- **Supported window modes**
  - `auto` – automatic window selection (default)
  - `off` – no IR windowing
  - `rew_sym` – REW-style symmetric window
  - `rew_asym` – REW-style asymmetric (causal) window

- **IR windowing type included in exported filenames**
  - ZIP and FIR WAV filenames include a short window tag for traceability and A/B comparison.
  - Examples:
    - `CamillaFIR_<type>_sym_<timestamp>.zip`
    - `L_<type>_<fs>Hz_<timestamp>_asym.wav`

- **Selectable IR window edge shape (Hann / Tukey)**
  - IR export window edge can be shaped using either:
    - **Hann** (legacy behavior)
    - **Tukey** with adjustable alpha (default 0.25)
  - Tukey window can reduce perceived sharpness and edge ripple compared to Hann,
    especially with asymmetric (causal) exports.
  - This affects **exported FIR impulse shape only** and does not modify DSP correction,
    targets, phase reconstruction, or leveling.


### Notes
- IR windowing affects **exported FIR impulse shape only**.
- No change to correction targets, phase reconstruction, FDW, TDC, or leveling logic.

### A-FDW & TDC guidance improvements

- **Refined A-FDW bandwidth limits**
  - Fixed incorrect A-FDW bandwidth constraints that could produce overly wide or misleading smoothing regions.
  - Ensures A-FDW operates strictly within intended psychoacoustic and confidence-based limits.

- **Clearer A-FDW & TDC visualization**
  - Updated plot annotations and guides to better reflect the *effective* A-FDW bandwidth actually used.
  - Reduces confusion between configured values and internally clamped / safety-limited behavior.

- **Improved documentation & UI hints**
  - A-FDW and TDC descriptions updated to better explain *why* certain regions are protected or limited.
  - Helps advanced users interpret confidence masks, decay control, and correction safety logic.

### Notes
- DSP behavior is unchanged except for corrected A-FDW limit handling.
- No changes to FIR magnitude targets, phase algorithms, or auto-leveling logic.



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

## Download

- **Windows (standalone EXE, recommended):**  
  https://github.com/VilhoValittu/CamillaFIR/releases/latest  
  → Download `CamillaFIR_<version>_windows.zip` and run `CamillaFIR.exe`

- **macOS (Intel & Apple Silicon):**  
  https://github.com/VilhoValittu/CamillaFIR/releases/latest  
  → Download `CamillaFIR_<version>_macos.tar.gz`, extract, then run `CamillaFIR`

- **Linux (Ubuntu / Debian compatible):**  
  https://github.com/VilhoValittu/CamillaFIR/releases/latest  
  → Download `CamillaFIR_<version>_linux.tar.gz`, extract, then run `CamillaFIR`

- **All releases:**  
  https://github.com/VilhoValittu/CamillaFIR/releases

> **Windows note:**  
> If Windows SmartScreen warns on first run, choose **More info → Run anyway**  
> (this is normal for unsigned open-source binaries).

> **macOS note:**  
> On first launch, macOS may block the app.  
> Open **System Settings → Privacy & Security → Open Anyway**.

> **Linux note:**  
> If needed, make the binary executable:  
> `chmod +x CamillaFIR`


---


## What you get

- FIR filters exported as **WAV (32-bit float)** or text
- Optional **CamillaDSP YAML** snippet
- Plots and a **Summary.txt** report (confidence, RT60, match score, effective A-FDW bandwidth, safety limits)
- Multi-rate export for common sample rates (44.1/48/88.2/96/176.4/192 kHz)

### Output location
Generated filter packages (`.zip`) are automatically saved to the **`filters/`** directory
in the project root. The directory is created automatically if it does not exist.

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

#Linux
sudo apt update
sudo apt install -y chromium-browser

python src/camillafir/camillafir.py
```

The UI opens in your browser (default: `http://localhost:8080`).

## System dependencies (PNG export)

Plotly PNG export uses Kaleido, which requires a Chromium-based browser
installed on the machine where CamillaFIR runs.

Ubuntu:
    sudo apt install chromium-browser


---

## Key features (v2.8.1)

- **2058-safe phase mode**  
  Disables room phase correction (confidence / FDW / excess-phase) and uses only theoretical XO phase and minimum-phase where applicable. Recommended when phase or group delay looks unstable or “spiky”.

- **Independent slope limits for boost vs cut (dB/oct)**  
  Separate slope constraints for boosts and cuts prevent gentle boosts from being flattened while still keeping aggressive cuts under control.

- **Temporal Decay Control (TDC) with safety brakes**  
  Time-domain reduction of room-induced energy storage with:
  - hard cap on total reduction per frequency bin  
  - optional slope limit (dB/oct) for predictable and stable decay shaping

- **Adaptive FDW (A-FDW)**  
  Frequency-dependent smoothing driven by measurement confidence:
  - high confidence → sharper correction  
  - low confidence → heavier smoothing  
  Includes corrected bandwidth limits and clear visualization of the *effective* A-FDW range.

- **DF smoothing (experimental)**  
  Optional Gaussian smoothing with approximately constant bandwidth in Hz across different sample rates and tap lengths, keeping perceived detail comparable.

- **Comparison mode (locked analysis grid)**  
  Locks scoring and plots to a fixed reference (fs/taps) so A/B comparisons remain deterministic and meaningful.

- **Multi-rate auto-taps mapping**  
  Automatically scales FIR taps to keep filter **time length constant** across sample rates (44.1 kHz reference).

- **WAV/IR-aware analysis path**  
  When using IR-derived WAV inputs, phase and reliability heuristics are adapted to avoid false “unreliable” regions compared to REW TXT-based responses.


---

## Documentation

- 📘 User & technical manual → `docs/Official_Manual.md`
- 📘 Modes → `docs/Modes.md`
- 🧠 Why this works → `docs/Why_CamillaFIR_Works.md`
- 📐 Academic DSP rationale (math) → `docs/Academic_DSP_Explanation.md`
- 🔁 Stability & reproducibility → `docs/Stability_and_Reproducibility.md`
- ⚖️ Comparison vs conventional EQ → `docs/Comparison_vs_EQ.md`

---

## Screenshot

![Effect of Temporal Decay Control](tdc_impulse_example.png)
