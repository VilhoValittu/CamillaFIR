# CamillaFIR by Vilho Valittu

## v3.1.1.1
Stable release - feedback welcome: camillafir.py@gmail.com

CamillaFIR generates high-resolution FIR room-correction filters from REW exports
(magnitude + phase) and WAV/IR measurements.

## v3.1.1 Highlights

- WAV input parsing is aligned to a deterministic TXT-baseline policy for more consistent WAV vs TXT behavior.
- Added WAV-only ripple cleanup near the correction upper edge and final FIR post-polish.
- Added stricter System Health checks for missing/incomplete L/R measurement sources.
- Added run timing visibility in UI (read, DSP, ZIP/PNG, render, total).
- Summary export now includes program version (`Version: v3.1.1`).

## Download

- Windows: https://github.com/VilhoValittu/CamillaFIR/releases/latest
- macOS (Intel + Apple Silicon): https://github.com/VilhoValittu/CamillaFIR/releases/latest
- Linux: https://github.com/VilhoValittu/CamillaFIR/releases/latest
- All releases: https://github.com/VilhoValittu/CamillaFIR/releases

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
# Linux:
#   pip install -r requirements-linux.txt

python src/camillafir/camillafir.py
# Linux:
#   python3 src/camillafir/camillafir.py
```

UI opens in browser at `http://localhost:8080`.

## What You Get

- FIR filters exported as WAV (32-bit float)
- Optional CamillaDSP YAML
- Summary report (`Summary.txt`)
- Multi-rate export (44.1/48/88.2/96/176.4/192 kHz)

Output ZIP files are saved to `filters/` in the project root.

## Browser And PNG Notes

- CamillaFIR UI is browser-based.
- Interactive graphs can be saved from the graph download button in UI.
- ZIP export is focused on filter artifacts and summary data; dashboard image inclusion can be disabled in perf mode.

## Known Issue (Windows, Vivaldi)

In some Windows setups, using Vivaldi can trigger NumPy `MemoryError` under browser memory pressure.

Workarounds:

- Use Chrome, Edge, or Firefox
- Close extra Vivaldi tabs/extensions
- Re-run process in another browser if needed

## Documentation

- User and technical manual: `docs/Official_Manual.md`
- Modes: `docs/Modes.md`
- Why this works: `docs/Why_CamillaFIR_Works.md`
- Academic DSP rationale: `docs/Academic_DSP_Explanation.md`
- Stability and reproducibility: `docs/Stability_and_Reproducibility.md`
- Comparison vs conventional EQ: `docs/Comparison_vs_EQ.md`

## UI Overview

### 1. Files
![Files view](pics/ui_1.png)

### 2. Basic
![Basic mode](pics/ui_2.png)

### 3. Target
![Target settings](pics/ui_3.png)

### 4. Advanced
![Advanced settings](pics/ui_4.png)

### 5. Windowing and TDC
![Windowing and TDC](pics/ui_5.png)

### 6. XO
![Crossover (XO)](pics/ui_6.png)

### TDC
![Effect of Temporal Decay Control](pics/tdc_impulse_example.png)