# CamillaFIR (v2.6.5 Stable) – Official Documentation

CamillaFIR is an AI-assisted DSP engine designed for creating high-resolution FIR filters. The program analyzes room acoustics to correct not only the frequency response but also time-domain errors, phase shifts, and room resonances (modes).

---

## 1. Basic Settings (Input & Technical)
Define your source material and the technical framework for the filter.

* **Local Path L / R**: Select the measurement files exported from REW (Room EQ Wizard) as `.txt`. Ensure the file includes both **Magnitude (dB)** and **Phase**.
* **Sample Rate (Hz)**: Matches your CamillaDSP system rate (usually 44100 or 48000).
* **Number of Taps**: The length/resolution of the filter.
    * **65536**: Standard high resolution for most systems.
    * **131072**: Extreme resolution for ultra-precise low-bass correction.
* **Output Format**: `WAV` (standard for CamillaDSP) or `txt`.



---

## 2. Filter Design (Phase Handling)
Determines how the engine manages the timing and phase of the audio.

* **Filter Type**:
    * **Linear Phase**: Perfect phase correction. Offers sharp impact but can cause "pre-ringing" on high frequencies.
    * **Minimum Phase**: No phase correction, only magnitude. Most natural treble, zero pre-echo.
    * **Asymmetric Linear**: The modern standard. Controlled pre-ringing, keeping transients sharp while fixing bass timing.
    * **Mixed Phase**: Applies Linear Phase to the bass (for impact) and Minimum Phase to the treble (for natural air).
* **Mixed Phase Split (Hz)**: The frequency where the logic switches from Linear to Minimum (Recommended: 300–500 Hz).
* **Global Gain (dB)**: Overall volume offset. Usually kept at 0 dB due to automatic normalization.



---

## 3. Target & Magnitude (The Sound Signature)
Defines the final tonal balance of your system.

* **House Curve Mode**:
    * **Flat**: Theoretically perfect, but can sound "thin" in home environments.
    * **Harman (+6dB / +8dB)**: Most popular choice. Provides a warm bass boost and a gentle treble roll-off.
* **Correction Range (Min/Max)**: The frequency limits for correction (e.g., 20 Hz – 500 Hz for bass-only, or full range).
* **Max Boost (dB)**: Limits how much the engine is allowed to "fill" acoustic dips. Recommended: 6–8 dB.



---

## 4. Leveling & Balancing
Ensures the filter does not significantly change the perceived loudness.

* **Level Mode**: `Automatic` (recommended) or `Manual`.
* **Match Range (Min/Max)**: The frequency area used to calculate average SPL (Recommended: 500–2000 Hz).
* **Algorithm**: `Median` is better at ignoring extreme acoustic peaks/dips than `Average`.

---

## 5. DSP Logic & Smoothing
Determines the "tightness" of the correction.

* **Regulation Strength**:
    * **1–10**: "Surgical". Extremely tight tracking of the target (use for sharp resonances). *Note: In v2.6.3, a value of 30 corresponds to 1 in older versions.*
    * **50+**: Softer correction that preserves the speaker's natural character.
* **Smoothing Type**: `Psychoacoustic` (recommended) as it mimics human hearing.
* **FDW Cycles**: Determines the timing window. Recommended: 15.

---

## 6. Smart Correction (Acoustic Intelligence)

* **Adaptive FDW (A-FDW)**: Dynamically adjusts windowing based on measurement reliability.
* **Temporal Decay Control (TDC)**:
    * **Function**: Reduces the ringing time (decay) of room modes.
    * **Strength**: 50–80% effectively removes "boomy" bass tails.



---

## 7. Crossovers & Protection

* **HPF (High Pass Filter)**: Protects drivers from subsonic frequencies (e.g., 20 Hz / 24dB slope).
* **Crossovers (1-5)**: Allows building up to 5-way systems with automatic phase alignment.
* **Excursion Protection**: Prevents electrical boost below a set frequency to protect the woofer's physical limits.

---

## 8. Dashboard Interpretation (Visual Analytics)

The Dashboard is your window into the room's behavior and the filter's performance.

### A. Magnitude & Confidence (Top Panel)
* **Blue Dotted Line**: Your raw room measurement.
* **Green Line**: Your chosen House Curve.
* **Orange Line**: **Predicted** – The expected result after filtering.
* **Magenta Line (Confidence)**: Reliability of the measurement (0-100%).
    * *Above 85%*: Excellent reliability.
    * *Below 50%*: High acoustic noise or interference; correction is automatically scaled back here.

### B. Phase
* Timing across frequencies. Goal: A flat line up to your **Phase Limit**. Fixed phase improves imaging and "3D" soundstage.

### C. Group Delay
* Reveals room resonances. High peaks indicate "ringing" bass. CamillaFIR aims to flatten these peaks.

### D. RT60 (Reverberation Time)
* The decay rate of your room. CamillaFIR uses this data for the **Smart TDC** logic to ensure resonances stop as fast as the rest of the room.

---

## 9. Troubleshooting & Acoustic Insight

| Observation | Likely Cause | Solution |
| :--- | :--- | :--- |
| **Predicted (Orange) misses Target** | Max Boost is too low | Increase Max Boost (e.g., to 8 dB) |
| **Bass feels "slow" or boomy** | Long decay / resonances | Increase TDC Strength or decrease Regulation Strength |
| **Audible "echo" or pre-ringing** | High-freq phase correction | Lower Phase Limit (e.g., to 400 Hz) or use Mixed Phase |
| **Sharp dips in Magnitude** | Acoustic Null (Cancellations) | Do not boost; try moving speakers 10-20 cm |
| **Messy Phase at high freqs** | Comb Filtering / Reflections | Use Psychoacoustic smoothing and lower FDW cycles |

---


# 11. Troubleshooting Guide: Interpreting Dashboard Acoustics

This section helps you identify common acoustic problems from the Dashboard graphs and explains how to address them using CamillaFIR settings.

---

## 1. Problem: Uncontrolled Bass Resonance (Room Mode)

**What does it look like on the Dashboard?**
* **Magnitude**: A high and narrow peak in the bass region (e.g., 30–100 Hz).
* **Group Delay**: A sharp, high "tower" at the same frequency.
* **RT60**: The value rises significantly higher at that specific frequency compared to the rest of the room.



**Solution in CamillaFIR:**
1. **Regulation Strength**: Lower the value to 10–30 (on the v2.6.3 scale). This forces the filter to cut the peak with high precision.
2. **Temporal Decay Control (TDC)**: Increase the strength (70–90%). TDC is specifically designed to stop resonance ringing in the time domain.

---

## 2. Problem: Acoustic Null (Cancellation)

**What does it look like on the Dashboard?**
* **Magnitude**: A deep, narrow "trench" or "dip" that the orange predicted line cannot fill, despite trying to.
* **Confidence**: The line often drops significantly at the location of the null.



**Analysis:**
This is usually caused by a reflection where the sound waves cancel each other out (destructive interference). This cannot be fixed by adding more power (DSP boost), as more power simply results in a stronger reflection and further cancellation.

**Solution in CamillaFIR:**
1. **Do not force it**: Keep **Max Boost** moderate (max 6-8 dB).
2. **Placement**: If the null is in a critical frequency range, try moving the speakers or the listening position by 10–20 cm. DSP cannot overcome the laws of physics in this scenario.

---

## 3. Problem: Pre-ringing

**What does it look like on the Dashboard?**
* **Auditory**: You hear a "hissing" or "metallic" echo immediately before a sharp transient (e.g., a snare drum hit).
* **Impulse Response**: You see oscillations occurring *before* the main impulse peak.



**Solution in CamillaFIR:**
1. **Filter Type**: Switch to `Asymmetric Linear` or `Mixed Phase` filter types.
2. **Phase Limit**: Lower the phase correction limit (e.g., to 400 Hz). The higher you attempt to correct the phase, the greater the risk of audible pre-ringing.

---

## 4. Problem: Comb Filtering

**What does it look like on the Dashboard?**
* **Magnitude**: The response zig-zags densely up and down across the mid and high frequencies.
* **Phase**: The phase graph spins or wraps wildly.



**Analysis:**
This is typically caused by a strong reflection from a nearby surface, such as a desk, floor, or side wall, arriving shortly after the direct sound.

**Solution in CamillaFIR:**
1. **Smoothing**: Ensure you are using `Psychoacoustic` smoothing so the software does not attempt to correct these microscopic, location-dependent errors too aggressively.
2. **FDW Cycles**: Reduce the value (e.g., from 15 -> 10). A shorter time window ignores late-arriving reflections.

---

## 5. Problem: Inaccurate Center Image (Phase Mismatch)

**What does it look like on the Dashboard?**
* **Left vs. Right**: When comparing the Phase panels of the L and R channels, they look completely different in the 200–1000 Hz range.

**Solution in CamillaFIR:**
1. **Stereo Link**: Ensure that magnitude correction is linked if the room and speaker setup are intended to be symmetrical.
2. **Phase Limit**: Ensure both channels use the exact same phase limit so that timing is corrected identically for both speakers.

---

## Summary: The Ideal Dashboard

1. **Magnitude**: The orange predicted line follows the green target line within +/- 2 dB accuracy.
2. **Group Delay**: The peaks in the bass region have been significantly flattened compared to the original blue dotted line.
3. **Confidence**: Stays above 80% across almost the entire frequency range.
4. **RT60**: The graph is relatively smooth and consistent without individual "peaks" at specific frequencies.
