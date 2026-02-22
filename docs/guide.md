---
layout: default
title: "How to Create FIR Filters from REW Measurements – Complete Guide"
description: "Step-by-step guide for creating FIR filters using REW measurements. Learn how to generate convolution filters for CamillaDSP, Roon and Equalizer APO using CamillaFIR."
permalink: /guide/
---

# How to Create FIR Filters from REW Measurements (Step-by-Step Guide)

This complete guide explains how to create **FIR filters** from **REW measurements** using **CamillaFIR**.

If you searched for:

- FIR filter maker  
- FIR filter generator  
- REW FIR convolution  
- How to create FIR filters  
- Room correction FIR  

You are in the right place.

---

# 1. What Is a FIR Filter?

A **FIR (Finite Impulse Response) filter** is a digital filter used in audio DSP for:

- Room correction
- Speaker correction
- Crossover linearization
- Phase alignment
- Subwoofer integration

Unlike IIR filters, FIR filters can provide:

- Linear-phase response
- Precise magnitude shaping
- Time-domain control
- Mixed-phase correction strategies

This makes FIR filters ideal for convolution engines like:

- CamillaDSP
- Roon
- Equalizer APO

---

# 2. Why Use FIR Instead of REW EQ Filters?

REW can export IIR parametric EQ filters. Those are useful but limited:

| Feature | IIR (PEQ) | FIR |
|----------|-----------|-----|
| Linear phase | ❌ | ✅ |
| Mixed-phase correction | ❌ | ✅ |
| Precise magnitude shaping | Limited | High |
| Convolution support | No | Yes |

If you want full control and better bass integration, a **FIR filter maker** like CamillaFIR gives more flexibility.

---

# 3. Step 1 – Measure Your Speakers in REW

1. Measure left and right speakers separately.
2. Use proper mic calibration.
3. Avoid clipping.
4. Use appropriate smoothing for analysis (not baked into export).

For best results:
- Keep boost targets conservative.
- Avoid trying to fix deep nulls.
- Focus correction on realistic frequency ranges.

---

# 4. Step 2 – Export Measurement Data from REW

Export frequency response as text file.

Recommended:
- No excessive smoothing
- Full resolution export
- Verify measurement integrity

You will use this file inside CamillaFIR.

---

# 5. Step 3 – Generate FIR Filters with CamillaFIR

Open CamillaFIR and:

1. Load measurement file(s)
2. Select correction range
3. Choose target curve
4. Set maximum boost (recommended: +3 dB)
5. Select filter type:
   - Linear phase
   - Mixed phase
   - Fast minimum-phase workflow

Avoid extreme boost values like +8 dB unless you are sure your amplifier and speakers can handle it.

Safer boosts = more reliable system.

---

# 6. Linear Phase vs Mixed Phase

## Linear Phase

- Symmetrical impulse response
- Pure phase linearity
- Higher latency

Best for:
- Offline correction
- Roon convolution
- DSP systems with delay compensation

## Mixed Phase

- More natural time response
- Lower latency
- Controlled excess phase handling

Often preferred for:
- Real-world listening systems
- Subwoofer integration
- Practical DSP workflows

---

# 7. Step 4 – Export Convolution Files

CamillaFIR exports:

- WAV FIR filters
- TXT FIR filters

These work with:

- CamillaDSP
- Roon convolution
- Equalizer APO
- Any convolution-capable DSP

---

# 8. Step 5 – Load FIR Filters into Your DSP

## CamillaDSP
Load WAV file into convolution block.

## Roon
Use convolution engine → upload FIR WAV.

## Equalizer APO
Use convolution module → load WAV filter.

Always verify gain staging.

---

# 9. Safe Correction Strategy (Important)

Do NOT:

- Boost deep nulls aggressively
- Overcorrect above practical frequency range
- Use excessive boost (+8 dB or more) without reason

Recommended:
- Keep boost around +3 dB
- Limit correction range logically
- Verify with re-measurement

Good DSP is controlled DSP.

---

# 10. Common Mistakes

### ❌ Overboosting bass  
Causes distortion and amplifier stress.

### ❌ Fixing nulls  
Deep nulls are often position-related.

### ❌ Overcorrecting full bandwidth  
Correction should be intentional.

### ❌ Ignoring verification measurement  
Always re-measure after applying FIR filter.

---

# 11. FAQ

## What is a FIR filter maker?
A tool that generates convolution-ready FIR filters from measurement data.

## Can CamillaFIR create linear-phase filters?
Yes.

## Does it support mixed-phase correction?
Yes.

## Is FIR better than IIR?
It depends on the goal. FIR offers more control and precision.

## Does it work with CamillaDSP?
Yes.

---

# 12. When Should You Use CamillaFIR?

Use it when you want:

- Better bass integration
- Controlled room correction
- Convolution-ready filters
- Advanced DSP workflow
- More precision than standard REW EQ

---

# 13. Final Thoughts

A good FIR filter is not about extreme correction.

It is about:

- Control
- Stability
- Safe boost
- Logical correction range
- Verification

CamillaFIR is designed as a practical FIR filter maker for real-world systems.

---

## Download CamillaFIR

👉 https://github.com/VilhoValittu/CamillaFIR/releases

---
