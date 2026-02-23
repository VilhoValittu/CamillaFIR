---
layout: default
title: "CamillaFIR FAQ – FIR Filter Maker (REW → Convolution Filters)"
description: "Frequently asked questions about CamillaFIR: FIR filter maker / generator for REW measurements. Convolution filters for CamillaDSP, Roon, Equalizer APO."
permalink: /faq/
---

<!-- Basic SEO -->
<meta name="keywords" content="CamillaFIR FAQ,FIR filter maker,FIR filter generator,REW FIR,convolution filters,CamillaDSP,Roon convolution,Equalizer APO" />
<link rel="canonical" href="{{ site.url }}{{ site.baseurl }}/faq/" />

<!-- FAQPage structured data (JSON-LD) -->
<script type="application/ld+json">
{
  "@context":"https://schema.org",
  "@type":"FAQPage",
  "mainEntity":[
    {
      "@type":"Question",
      "name":"What is CamillaFIR?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"CamillaFIR is an automatic FIR filter maker / generator that converts measurement data (e.g., REW exports) into convolution-ready FIR filters (WAV/TXT) for DSP engines such as CamillaDSP, Roon Convolution, and Equalizer APO."
      }
    },
    {
      "@type":"Question",
      "name":"What is a FIR filter maker / generator?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"A FIR filter maker (or generator) is a tool that creates FIR convolution filters from measurement data or a target response. The resulting FIR files can be loaded into convolution engines to apply room correction, speaker correction, or crossover/phase alignment."
      }
    },
    {
      "@type":"Question",
      "name":"Does CamillaFIR work with REW measurements?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"Yes. CamillaFIR is designed to use measurement data exported from Room EQ Wizard (REW) and generate FIR correction filters based on that data."
      }
    },
    {
      "@type":"Question",
      "name":"Where can I download CamillaFIR?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"You can download the latest version from the GitHub Releases page: https://github.com/VilhoValittu/CamillaFIR/releases"
      }
    },
    {
      "@type":"Question",
      "name":"What output formats does CamillaFIR export?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"CamillaFIR exports convolution-ready FIR filters, typically as WAV files and/or TXT files depending on your workflow and target DSP engine."
      }
    },
    {
      "@type":"Question",
      "name":"Can I use CamillaFIR with CamillaDSP?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"Yes. CamillaFIR output FIR WAV/TXT files can be used in CamillaDSP’s convolution block. Always verify gain staging to avoid clipping."
      }
    },
    {
      "@type":"Question",
      "name":"Can I use CamillaFIR with Roon Convolution?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"Yes. Roon supports convolution filters. Export the convolution-ready FIR WAV from CamillaFIR and load it into Roon’s Convolution settings."
      }
    },
    {
      "@type":"Question",
      "name":"Can I use CamillaFIR with Equalizer APO?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"Yes, if you use Equalizer APO’s convolution capability. Load the exported FIR WAV filter and ensure preamp/headroom settings prevent clipping."
      }
    },
    {
      "@type":"Question",
      "name":"Is FIR better than IIR (PEQ)?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"It depends on the goal. FIR offers more precise magnitude shaping and can support linear-phase or mixed-phase strategies via convolution. IIR/PEQ is lighter and often sufficient for simple magnitude correction. FIR is typically preferred when you want more control over time/phase behavior."
      }
    },
    {
      "@type":"Question",
      "name":"Does CamillaFIR support linear-phase filters?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"Yes. CamillaFIR can generate linear-phase style FIR filters (higher latency, symmetrical impulse response) when that fits your use case."
      }
    },
    {
      "@type":"Question",
      "name":"Does CamillaFIR support mixed-phase correction?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"Yes. Mixed-phase workflows aim to balance time-domain behavior and correction effectiveness, often providing a more practical solution in real systems where latency and impulse behavior matter."
      }
    },
    {
      "@type":"Question",
      "name":"What is a safe maximum boost for correction filters?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"As a general safe starting point, keep maximum boost around +3 dB. Higher boost (e.g., +8 dB) can demand much more from amplifiers and speakers and increases the risk of distortion or clipping, especially in the bass."
      }
    },
    {
      "@type":"Question",
      "name":"Should I try to fix deep nulls with FIR correction?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"Usually no. Deep nulls are often caused by room/speaker-listener geometry and reflections. Boosting them can waste headroom without improving the real in-room response. It’s typically better to address nulls with placement, multiple subs, or room treatment."
      }
    },
    {
      "@type":"Question",
      "name":"Do I need to re-measure after applying the FIR filter?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"Yes. Always verify by re-measuring with the filter active. This confirms that the correction behaves as expected and that gain staging/headroom is safe."
      }
    },
    {
      "@type":"Question",
      "name":"How should I export measurements from REW for CamillaFIR?",
      "acceptedAnswer":{
        "@type":"Answer",
        "text":"Before measuring, load your microphone calibration file in REW. For TXT export, include Frequency, Magnitude, and Phase. For WAV/IR workflow, use Mono, float32, Normalise, and Place t=0 (256)."
      }
    }
  ]
}
</script>

# CamillaFIR FAQ

This page answers common questions about **CamillaFIR**, an automatic **FIR filter maker / generator** for **REW measurements**, producing **convolution filters** for **CamillaDSP**, **Roon**, and **Equalizer APO**.

## General

### What is CamillaFIR?
CamillaFIR is an automatic FIR filter maker / generator that converts measurement data (e.g., REW exports) into convolution-ready FIR filters (WAV/TXT).

### What is a FIR filter maker / generator?
A FIR filter maker generates convolution FIR filters from measurement data and a target response.

### Where can I download CamillaFIR?
Download the latest version from GitHub Releases:  
https://github.com/VilhoValittu/CamillaFIR/releases

### How should I export measurements from REW for CamillaFIR?
Load your microphone calibration file in REW before measuring.
For TXT export, include Frequency + Magnitude + Phase.
For WAV/IR workflow, use `Mono`, `float32`, `Normalise`, `Place t=0 (256)`.

## Compatibility

### Does CamillaFIR work with CamillaDSP?
Yes. Use the exported FIR filter in CamillaDSP’s convolution block.

### Does CamillaFIR work with Roon Convolution?
Yes. Load the exported convolution-ready FIR filter in Roon’s Convolution settings.

### Does CamillaFIR work with Equalizer APO?
Yes, via Equalizer APO’s convolution support. Ensure correct preamp/headroom.

## FIR vs IIR

### Is FIR better than IIR (PEQ)?
It depends. FIR provides more control and can support linear-phase or mixed-phase strategies via convolution. IIR is simpler and lighter.

### Does CamillaFIR support linear-phase filters?
Yes.

### Does CamillaFIR support mixed-phase correction?
Yes.

## Safety and best practices

### What is a safe maximum boost?
A good starting point is **+3 dB max boost**. Larger boosts can stress hardware and increase distortion/clipping risk.

### Should I fix deep nulls?
Usually not. Deep nulls are typically position/room related, and boosting them often wastes headroom.

### Do I need to re-measure?
Yes—always verify with a measurement after applying the FIR filter.

### Disclaimer
AI was used to translate this document from Finnish to English.
