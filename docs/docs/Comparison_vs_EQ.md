# CamillaFIR vs Conventional EQ-Based Room Correction

## Conventional Approach
Typical room correction systems:
- Treat all deviations as magnitude errors
- Apply minimum-phase or IIR EQ
- Ignore decay behavior
- Implicitly mix distance, phase, and room effects

### Consequences
- Bass becomes quieter but not tighter
- Treble may become harsh
- Results vary strongly with microphone position

---

## CamillaFIR Approach

| Aspect | Conventional EQ | CamillaFIR |
|------|----------------|------------|
| Propagation Delay | Ignored | Explicitly removed |
| Phase Correction | Limited / implicit | Explicit excess-phase handling |
| Room Modes | EQ amplitude | Reduce decay time (TDC) |
| Reflections | Often inverted | Confidence-masked |
| DSP Domain | Mostly frequency | Time + frequency |
| Stability | Measurement-sensitive | Bounded & deterministic |

---

## Audible Result
- Faster bass decay
- Preserved transients
- Natural treble
- Repeatable tuning results

CamillaFIR corrects *less*, but corrects *the right things*.
