---
layout: default
title: "CamillaFIR - FIR Filter Maker & Room Correction Filter Generator"
description: "Automatic FIR filter maker / generator for REW measurements. Exports WAV FIR filters and companion config files for CamillaDSP, Roon, Equalizer APO and more."
permalink: /
---

<!-- SEO: OpenGraph/Twitter/JSON-LD -->
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
  "url":"{{ site.url }}{{ site.baseurl }}/",
  "downloadUrl":"https://github.com/VilhoValittu/CamillaFIR/releases",
  "codeRepository":"https://github.com/VilhoValittu/CamillaFIR",
  "author":{
    "@type":"Person",
    "name":"VilhoValittu"
  }
}
</script>

# CamillaFIR - FIR Filter Maker & Room Correction Filter Generator

CamillaFIR is an **automatic FIR filter maker / generator** that turns **REW measurements** into **convolution FIR filters** you can load into:

- **CamillaDSP**
- **Roon (Convolution)**
- **Equalizer APO** (and similar convolution engines)

It exports **WAV FIR filters** plus companion summary/config files and provides a practical workflow for **room correction**, **safe bass correction**, and optional **mixed-phase / linear-phase** strategies.

---

## What CamillaFIR does

- Converts measurement data (for example REW exports) into **FIR correction filters**
- Produces **convolution-ready** output files (WAV FIR + companion config files)
- Helps you avoid common pitfalls like excessive boost and unstable corrections
- Includes tools/plots to inspect results (magnitude, phase, IR)

If you search for *"FIR filter maker"*, *"FIR filter generator"*, or *"REW FIR convolution"*, this is exactly that.

---

## Quick links

- **Releases (Downloads):** https://github.com/VilhoValittu/CamillaFIR/releases
- **Repository:** https://github.com/VilhoValittu/CamillaFIR
- **Documentation:** https://github.com/VilhoValittu/CamillaFIR#readme
- **Support / Ko-fi:** https://ko-fi.com/camillafir
- **FAQ:** {{ site.baseurl }}/faq/

---

## Typical workflow

1. Measure your speakers / room with **REW** (load your microphone calibration file first)
2. Export measurements (for TXT include magnitude + phase; for WAV use `Mono`, `float32`, `Normalise`, `Place t=0 (256)`)
3. Generate correction filters with **CamillaFIR**
4. Load the resulting FIR WAV into **CamillaDSP / Roon / Equalizer APO**
5. Verify the result (measurement + listening)

---

## Keywords (so Google understands the page)

**FIR filter maker**, **FIR filter generator**, **FIR designer**, **REW convolution**, **room correction FIR**, **CamillaDSP FIR filters**, **Roon convolution filter**, **Equalizer APO FIR**, **mixed-phase FIR**, **linear-phase FIR**, **audio DSP**.

---

## Get started

Go to the **latest release** and download the build for your OS:  
https://github.com/VilhoValittu/CamillaFIR/releases

### Disclaimer
AI was used to translate this document from Finnish to English.
