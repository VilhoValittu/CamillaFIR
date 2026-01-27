CamillaFIR Modes: BASIC vs ADVANCED

This project has two operating modes that affect **FilterConfig defaults** and (in BASIC) apply **hard clamps**.
The DSP engine is the same in both modes — the difference is guard rails.

## BASIC (recommended)

Designed for “good results without surprises”.

In BASIC mode:
- Boost/cut ranges are hard-limited
- TDC + A-FDW are ON with sensible guard rails
- Low-bass boost is limited via `low_bass_cut_hz` (default 30 Hz)
- Slope limiting is enabled with asymmetric defaults (boost gentler than cut)
- Stereo-link leveling is ON by default (consistent L/R)

Typical use cases:
- most rooms and systems
- quick, repeatable corrections
- users who prefer safety and predictability

## ADVANCED (expert)

Designed for users who want full control.

In ADVANCED mode:
- No hard clamps are applied by mode policy
- Defaults allow larger boost/cut ranges
- Slope limiting is OFF by default (user decides)
- Low-bass “cuts-only” default is disabled (`low_bass_cut_hz = 0`)

Typical use cases:
- experimental tuning
- VBA-style workflows
- advanced multi-sub alignment work

## Notes

- Mode is applied in the config builder (after UI values are read), so it works even if the UI changes.
- If mode is missing/unknown, the program falls back to **BASIC**.

## Where is this implemented?

- `camillafir_modes.py`: mode defaults + clamps and `apply_mode_to_cfg()`
- `camillafir.py`: UI exposes `mode` selection and passes it into FilterConfig build pipeline
