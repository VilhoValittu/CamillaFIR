---
layout: default
title: "Quick Start: How to Use the CamillaFIR Filter Maker"
description: "Step-by-step guide to generating convolution filters using CamillaFIR, REW measurements, and the recommended Automatic Asymmetric mode."
permalink: /quick-start/
---

# Quick Start Guide: Generating FIR Filters

This guide shows you how to use **CamillaFIR**, an automated **FIR filter maker**, to create high-quality room correction filters for your audio system.

---

## Step 1: Measure with REW (Room EQ Wizard)
Before using the **FIR filter generator**, you need accurate measurement data.

1.  **Microphone Setup:** Connect your calibrated measurement microphone (e.g., UMIK-1).
2.  [cite_start]**Calibration:** Ensure your microphone calibration file is loaded in REW[cite: 1].
3.  **Sweep:** Run a frequency response sweep (typically 0 Hz – 24 kHz).
4.  **Export:** * Go to `File` > `Export` > `Measurement as Text`.
    * [cite_start]**Crucial:** Select "Include Phase" in the export settings[cite: 1].
    * [cite_start]Alternatively, for WAV/IR workflow, use `Mono`, `float32`, `Normalise`, and `Place t=0 (256)`[cite: 1].

---

## Step 2: Recommended Mode – Automatic Asymmetric
For the best balance between sonic accuracy and system stability, we recommend using the **Automatic Asymmetric mode**.

* **Why Asymmetric?** Unlike standard linear-phase filters, the **Asymmetric FIR filter maker** strategy provides high-precision correction with lower pre-ringing risk and optimized latency.
* **Automatic Workflow:**
    1.  Set the filter mode to **Asymmetric**.
    2.  Enable **CamillaFIR Automatic mode** to align your target curve and measurements automatically.
    3.  The engine handles **TOF (Time of Flight) removal** automatically to ensure phase correction acts only on excess phase.
    4.  It utilizes **Adaptive FDW (Frequency Dependent Windowing)** to weight correction based on acoustic confidence.

---

## Step 3: Generate and Export
Now, let the **FIR filter generator** process your data.

1.  **Launch CamillaFIR:** Open the application.
2.  [cite_start]**Load Measurement:** Select the TXT or WAV file you exported from REW[cite: 1].
3.  **Set Target Curve:** Choose a flat target or load a custom house curve.
4.  [cite_start]**Safety Guards:** * **Max Boost:** We recommend a safe limit of **+3 dB** to avoid clipping[cite: 1].
    * **Low-Bass:** Enable safe bass correction policies.
5.  **Process:** Click generate. [cite_start]CamillaFIR produces a ZIP package containing your **convolution-ready** output files[cite: 1].

---

## Step 4: Apply to Your DSP
Load the resulting **WAV FIR filter** into your preferred engine:

* [cite_start]**CamillaDSP:** Add the WAV to your convolution block[cite: 1].
* [cite_start]**Roon:** Upload the WAV (or ZIP) into Roon's Convolution settings[cite: 1].
* [cite_start]**Equalizer APO:** Load the WAV filter and ensure sufficient preamp headroom[cite: 1].

## Step 5: Verify
[cite_start]**Always re-measure** your speakers with the FIR filter active to confirm the correction behaves as expected[cite: 1].

---

## Why use CamillaFIR?
Unlike basic EQ, this **FIR filter maker** handles **excess phase** and **temporal decay (TDC)**, providing tighter bass and more repeatable tuning.

[← Back to Home]({{ site.baseurl }}/) | [Read the FAQ]({{ site.baseurl }}/faq)