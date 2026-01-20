# CamillaFIR vs Conventional EQ-Based Room Correction (v2.7.7)

## Conventional approach (typical IIR / minimum-phase EQ)
Most room correction systems:
- treat deviations as magnitude errors
- apply IIR/PEQ or minimum-phase EQ
- ignore (or cannot control) time-domain decay
- implicitly mix distance, phase, and room effects

### Common consequences
- bass gets quieter but not tighter (decay unchanged)
- harsh or “glassy” treble if small combing dips are inverted
- results vary strongly with microphone position

---

## CamillaFIR approach

| Aspect | Conventional EQ | CamillaFIR |
|---|---|---|
| Propagation delay (TOF) | ignored | explicitly removed before phase analysis |
| Excess phase | limited / implicit | selectable FIR phase strategies |
| Room modes | amplitude EQ | **Temporal Decay Control (TDC)** targets decay |
| Reflections / combing | often inverted | confidence-masked + smoothing / A-FDW |
| Correction bounds | varies | hard limits: boost, cut, slope, phase bandwidth |
| Slope control | usually none | dB/oct limiting, optionally **separate boost vs cut** |
| Reproducible A/B | hard (grid changes) | optional **comparison mode** with fixed analysis grid |
| Multi-rate export | not typical | native multi-rate generation |

---

## Audible result (typical)
- bass notes stop faster (less overhang)
- transients stay intact
- treble stays natural because low-confidence interference is not inverted
- tuning is repeatable and predictable

CamillaFIR corrects **less**, but corrects **the right things**.
