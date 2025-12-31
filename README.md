# CamilleFIR
Automated FIR filter generator for REW measurements. Creates phase-linear correction files (WAV/CSV) for Equalizer APO, Roon, and CamillaDSP. Features crossover linearization, smart room correction, and safe subsonic filtering. By VilhoValittu &amp; GeminiPro. Inspired by OCA https://www.youtube.com/@ocaudiophile

Hi everyone,

I wanted to share a new tool developed by "VilhoValittu & GeminiPro". It's a Python-based utility designed to automate the creation of FIR correction filters from REW measurements.

Creating phase-linear FIR filters manually (e.g., in rePhase) acts as a bottleneck for many. This tool aims to streamline the process, creating convolution files ready for Equalizer APO, Roon, CamillaDSP, etc., in seconds.

Key Features:

Automated Phase Linearization: Corrects the phase shift of your existing IIR crossovers (just input freq & slope).

Smart Room Correction: Applies frequency response correction based on a target House Curve with adjustable max boost limits.

Configurable Taps: Choose from 2048 up to 131,072 taps (balancing latency vs. bass resolution).

Safe Subsonic Filter: Optional High Pass Filter (10-60Hz) implemented as Minimum Phase to prevent pre-ringing artifacts in the bass.

Auto-Leveling: Matches the House Curve to your speaker's natural response to avoid drastic gain jumps.

Output Formats:

WAV: 32-bit float (Mono or Stereo files).

CSV: Ready-to-use coefficients for Equalizer APO.

Deep Dive: How the Phase Correction Works Unlike simple auto-EQs that try to flatten the measured phase blindly (often causing severe pre-ringing artifacts), this tool uses a Hybrid Approach:

Theoretical Linearization: First, it calculates the mathematical inverse of the IIR crossovers you specify (e.g., LR4 @ 2000Hz). This unwraps the phase shift caused by your existing crossovers purely based on math, guaranteeing zero artifacts for this part of the correction.

Measured Fine-Tuning: It then looks at your actual measurement data (Excess Phase). It applies a secondary correction to align the driver's natural phase deviations.

Safety Clamping: This fine-tuning is strictly clamped (max ±45 degrees). This ensures the tool never tries to correct room reflections or measurement noise, which is the #1 cause of bad-sounding FIR filters.

The result is a step response that looks like a single coherent spike, improving transient attack and soundstage depth without the "processed" sound of aggressive room correction.

Workflow:

Measure in REW.

Export measurements as text files (L.txt & R.txt) into the same folder as the tool.

Note: If using a House Curve, place that .txt file in the same folder as well.

Run the generator.

Select your preferences (Sample rate, House curve, Output format).

Load the generated file into your convolution engine.


Feedback is welcome!

EXE file available https://drive.google.com/drive/folders/1AkESLDo-UhPqxDCdaZuXE6u8-H4EDuOI?usp=sharing
