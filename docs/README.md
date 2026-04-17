# CamillaFIR

## v4.2.0

CamillaFIR is an automatic FIR filter generator and room-correction tool for REW exports, built-in sweep measurements, and WAV/IR captures.
It is designed for CamillaDSP and other FIR-capable playback systems.

CamillaFIR is also listed in the official CamillaDSP README under **Measurement and filter generation tools**.

## v4.2.0 highlights

- Built-in guided measurement is now part of the normal CamillaFIR workflow, so you can capture sine sweeps, save IR WAV results, and continue directly into filter generation without a separate REW session.
- Automatic mode now uses extracted harmonic curves (`H2` / `H3`) as a distortion-aware guard, reducing the preference for candidates that add boost in already nonlinear bands. Requires CamillaFIRs built-in measurements.

## Quick links

- [Download latest release](https://github.com/VilhoValittu/CamillaFIR/releases/latest)
- [Quick Start Guide](Quick_Start_Guide.md)
- [User Manual](Official_Manual.md)
- [Installation Guide](Installation.md)
- [Changelog](CHANGELOG.md)
- [All documentation](docs/)
- [CamillaDSP reference](https://github.com/HEnquist/camilladsp?tab=readme-ov-file#measurement-and-filter-generation-tools)

## What CamillaFIR does

- Generates FIR room-correction filters from REW exports, built-in sweep measurements, and WAV/IR captures
- Includes an integrated measurement workflow with repeat handling, outlier rejection, and IR export ready for the filter pipeline
- Supports **Automatic**, **Basic**, and **Advanced** workflows
- Exports ready-to-use FIR filters for CamillaDSP and other FIR-capable DSP systems
- Supports **Asymmetric**, **Linear**, **Minimum**, and **Mixed Phase** filter types
- Includes automatic target selection, room-safe optimization, and summary exports
- Automatic mode can use harmonic curves and IACC-aware ranking when comparing candidate filters
- Supports multi-rate export up to 192 kHz

## Why use CamillaFIR

CamillaFIR is built for practical loudspeaker and room correction, not just static EQ matching.
It combines target matching, phase handling, correction safety, and export workflow into one tool.

Key strengths:

- strong automatic mode for final-use filters with distortion-aware and stereo-aware ranking
- practical low-frequency and room-mode handling
- built-in measurement workflow in addition to REW import
- multiple FIR filter strategies for different latency/phase goals
- browser-based graphical UI
- direct export workflow for real playback systems

## Quick start

1. Download the latest release from the [Releases page](https://github.com/VilhoValittu/CamillaFIR/releases/latest).
2. Open the [Quick Start Guide](Quick_Start_Guide.md).
3. For full usage instructions, open the [User Manual](Official_Manual.md).
4. Load your measurements, or create them with the built-in measurement tool, choose a workflow, and generate filters.

## Documentation

### Start here

- [Quick Start Guide](Quick_Start_Guide.md)
- [User Manual](Official_Manual.md)
- [Installation Guide](Installation.md)

### Core documentation

- [Modes](Modes.md)
- [Why CamillaFIR Works](Why_CamillaFIR_Works.md)
- [Comparison vs EQ](Comparison_vs_EQ.md)
- [Stability and Reproducibility](Stability_and_Reproducibility.md)
- [Academic DSP Explanation](Academic_DSP_Explanation.md)
- [Reading Output Guide](CamillaFIR_Reading_Output_Guide.md)
- [IR Export Windowing](IR_Export_Windowing.md)
- [DSP Guards](CamillaFIR_dsp_guards.md)
- [FAQ](faq.md)

### Reference material

- [Changelog](CHANGELOG.md)
- [Case Study PDF](CamillaFIR_Full_Case_Study_v3.pdf)
- [Extended Output Guide PDF](CamillaFIR_Reading_Output_Guide_Extended.pdf)

## Download

- [Latest release](https://github.com/VilhoValittu/CamillaFIR/releases/latest)
- [All releases](https://github.com/VilhoValittu/CamillaFIR/releases)

For platform-specific setup instructions, see the [Installation Guide](Installation.md).

## Typical workflow

1. Measure your system with the built-in measurement tool, REW, or coherent WAV/IR captures.
2. Load the measurements into CamillaFIR.
3. Choose **Automatic**, **Basic**, or **Advanced** mode.
4. Select the filter type that fits your use case.
5. Generate filters and review the summary.
6. Export WAV filters and optional CamillaDSP configuration assets.

## Output

CamillaFIR can export:

- FIR filters as WAV
- optional CamillaDSP YAML/config assets
- summary report files such as `Summary.txt`
- multi-rate filter sets

## Filter types

- **Asymmetric**: recommended default for most users; strong correction with practical latency
- **Linear**: use when maximum linear-phase behavior matters more than latency
- **Minimum**: practical low-latency causal correction
- **Mixed Phase**: useful when mixed-phase behavior and smoother GD shaping are preferred

## UI overview

### Files
![Files view](pics/ui_1.png)

### Basic
![Basic mode](pics/ui_2.png)

### Target
![Target settings](pics/ui_3.png)

### Advanced
![Advanced settings](pics/ui_4.png)

### XO
![Crossover (XO)](pics/ui_6.png)

### Results
![Results 1](pics/ui_7.png)

## Notes

- CamillaFIR uses a browser-based UI.
- Interactive graphs can be saved directly from the UI.
- PNG/dashboard export behavior may depend on host browser support.
- For known setup issues and platform-specific notes, see the [Installation Guide](Installation.md).

## Contact

Feedback: camillafir.py@gmail.com

## Source Code Status

CamillaFIR is currently developed as a closed-source / proprietary project.

The source code is not available under an open-source license.
Copying, redistribution, commercial reuse, removal of attribution, and modification
for redistribution are not allowed without prior written permission.

Public documentation, changelogs, screenshots, and release information may still be
shared publicly, but that does not grant permission to reuse the source code.

Licensing inquiries:
camillafir.py@gmail.com
