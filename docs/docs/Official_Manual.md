# CamillaFIR – Official Manual (v2.7.3)

## 1. Overview
CamillaFIR is an AI-assisted DSP engine for generating high-resolution FIR correction filters
from acoustic measurements. It focuses on **time-domain correctness first**, then magnitude.

Key design goals:
- Preserve transients
- Avoid over-correction
- Produce physically meaningful acoustic metrics

---

## 2. Processing Pipeline (High Level)
1. Import REW magnitude + phase
2. Robust parsing and unit normalization
3. Psychoacoustic or standard smoothing
4. Time-of-Flight (TOF) detection & removal
5. Confidence analysis & reflection detection
6. Target curve construction
7. Smart leveling (Median / Average)
8. Magnitude correction with safety guards
9. Phase reconstruction (Linear / Min / Mixed / Asymmetric)
10. TDC (optional)
11. FIR synthesis + normalization
12. Multi-rate export

---

## 3. Filter Types (What Actually Happens)

### Linear Phase
- Full-spectrum excess phase correction
- Maximum transient precision
- Risk of pre-ringing above ~2 kHz

### Minimum Phase
- Magnitude-only correction
- Phase reconstructed via Hilbert transform
- No pre-ringing

### Mixed Phase
- Linear phase below split frequency
- Minimum phase above
- Unity magnitude sum at crossover

### Asymmetric Linear
- Linear phase with asymmetric time window
- Preserves impulse leading edge
- Greatly reduced audible pre-ringing

---

## 4. Acoustic Intelligence

### Confidence Mask
Each frequency bin is assigned a reliability score based on:
- Phase stability
- Reflection dominance
- Energy decay

This mask directly controls:
- FDW window length
- Smoothing strength
- Correction aggressiveness

### Reflection Analysis
Detected reflections report:
- Frequency
- Group delay error (ms)
- Physical distance (meters)

Distances are real because TOF is removed before analysis.

---

## 5. Temporal Decay Control (TDC)
TDC is **not EQ**.

It:
- Identifies excessive decay (RT60)
- Injects inverse decay impulses
- Reduces ringing time instead of SPL

Result:
Bass notes stop faster instead of just getting quieter.

---

## 6. Safety Systems
- Soft-clip boost limiting
- Independent max cut limit
- dB/oct slope limiting
- HPF & excursion protection
- Phase correction frequency limit

---

## 7. Outputs
- FIR WAV / TXT (IEEE float)
- Summary report (RT60, confidence, match %)
- Optional CamillaDSP YAML

---
