---
layout: default
title: "CamillaFIR - FIR Filter Maker & Room Correction Filter Generator"
description: "Automatic FIR filter maker / generator for REW measurements. Exports WAV FIR filters and companion config files for CamillaDSP, Roon, Equalizer APO and more."
permalink: /
---

<meta name="keywords" content="FIR filter maker,FIR filter generator,FIR designer,room correction FIR,REW FIR,camilladsp FIR,Equalizer APO FIR,Roon convolution,convolution filter generator,mixed-phase FIR,linear-phase FIR,digital filter design,audio DSP" />
<link rel="canonical" href="{{ site.url }}{{ site.baseurl }}/" />

<meta property="og:type" content="website" />
<meta property="og:title" content="CamillaFIR - FIR Filter Maker & Room Correction Filter Generator" />
<meta property="og:description" content="Automatic FIR filter maker / generator for REW measurements. Exports WAV FIR filters and companion config files for CamillaDSP, Roon, Equalizer APO and more." />
<meta property="og:url" content="{{ site.url }}{{ site.baseurl }}/" />

<meta name="twitter:card" content="summary" />
<meta name="twitter:title" content="CamillaFIR - FIR Filter Maker & Room Correction Filter Generator" />
<meta name="twitter:description" content="Automatic FIR filter maker / generator for REW measurements. Exports WAV FIR filters and companion config files for CamillaDSP, Roon, Equalizer APO and more." />

<script type="application/ld+json">
{
  "@context":"https://schema.org",
  "@type":"SoftwareApplication",
  "name":"CamillaFIR",
  "applicationCategory":"MultimediaApplication",
  "operatingSystem":"Windows, Linux, macOS",
  "description":"Automatic FIR filter maker / generator for REW measurements. Exports WAV FIR filters and companion config files for CamillaDSP, Roon, Equalizer APO and more. Supports mixed-phase and linear-phase workflows.",
  "featureList": "Automatic FIR filter maker, Room correction filter generator, Mixed-phase FIR design, REW measurement integration",
  "url":"{{ site.url }}{{ site.baseurl }}/",
  "downloadUrl":"https://github.com/VilhoValittu/CamillaFIR/releases",
  "codeRepository":"https://github.com/VilhoValittu/CamillaFIR",
  "author":{
    "@type":"Person",
    "name":"VilhoValittu"
  }
}
</script>

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {
      "@type": "Question",
      "name": "What is a FIR filter maker used for?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "A FIR filter maker like CamillaFIR is used to create digital filters for room correction and speaker optimization. It processes measurement data to generate convolution filters that improve audio fidelity."
      }
    },
    {
      "@type": "Question",
      "name": "Can I use CamillaFIR with REW?",
      "acceptedAnswer": {
        "@type": "Answer",
        "text": "Yes, CamillaFIR is specifically designed to work with REW (Room EQ Wizard) exports, converting them into WAV FIR filters for CamillaDSP, Roon, and Equalizer APO."
      }
    }
  ]
}
</script>

# CamillaFIR - FIR Filter Maker & Room Correction Filter Generator

CamillaFIR is an **automatic FIR filter maker / generator** that turns **REW measurements** into **convolution FIR filters** you can load into:

- **CamillaDSP**
- **Roon (Convolution)**
- **Equalizer APO** (and similar convolution engines)

CamillaFIR is listed in the official CamillaDSP README under
[Measurement and filter generation tools](https://github.com/HEnquist/camilladsp?tab=readme-ov-file#measurement-and-filter-generation-tools), as a recommended tool for measurement-based filter generation.

---

## Technical Documentation & Guides

Learn more about how CamillaFIR works and how it compares to other DSP methods:

* **[Why CamillaFIR Works]({{ site.baseurl }}/Why_CamillaFIR_Works)** – The logic behind the correction.
* **[Comparison: FIR vs. IIR EQ]({{ site.baseurl }}/Comparison_vs_EQ)** – Why use a FIR filter maker instead of traditional EQ.
* **[Academic DSP Explanation]({{ site.baseurl }}/Academic_DSP_Explanation)** – Deep dive into the signal processing.
* **[FAQ]({{ site.baseurl }}/faq)** – Frequently asked questions about FIR generation.

---

## What CamillaFIR does

- Converts measurement data (for example REW exports) into **FIR correction filters**
- Produces **convolution-ready** output files (WAV FIR + companion config files)
- Helps you avoid common pitfalls like excessive boost and unstable corrections
- Includes tools/plots to inspect results (magnitude, phase, IR)

---

## Quick links

- **Releases (Downloads):** https://github.com/VilhoValittu/CamillaFIR/releases
- **Repository:** https://github.com/VilhoValittu/CamillaFIR
- **Support / Ko-fi:** https://ko-fi.com/camillafir

---

## Typical workflow

1. Measure your speakers / room with **REW** (load your microphone calibration file first)
2. Export measurements (for TXT include magnitude + phase; for WAV use `Mono`, `float32`, `Normalise`, `Place t=0 (256)`)
3. Generate correction filters with **CamillaFIR**
4. Load the resulting FIR WAV into **CamillaDSP / Roon / Equalizer APO**
5. Verify the result (measurement + listening)

---

## Get started

Go to the **latest release** and download the build for your OS:  
https://github.com/VilhoValittu/CamillaFIR/releases

### Disclaimer
AI was used to translate this document from Finnish to English.
