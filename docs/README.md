# CamillaFIR

**Time-domain–first FIR room correction**

CamillaFIR is a DSP engine for generating high-resolution FIR correction filters
from acoustic measurements.

Unlike conventional EQ-based room correction, CamillaFIR explicitly separates:
- propagation delay (Time of Flight),
- excess phase distortion,
- room-induced energy storage (room modes).

Each phenomenon is corrected using the physically correct DSP method.

---

## Acknowledgements

Development inspired by the methodologies of **OCA** (Obsessive Compulsive Audiophile): [https://www.youtube.com/@ocaudiophile](https://www.youtube.com/@ocaudiophile)

---


## Documentation

- 📘 User & Technical Manual  
  → [docs/Official_Manual.md](docs/Official_Manual.md)

- 🧠 Why this works  
  → [docs/Why_CamillaFIR_Works.md](docs/Why_CamillaFIR_Works.md)

- 📐 Academic DSP rationale  
  → [docs/Academic_DSP_Explanation.md](docs/Academic_DSP_Explanation.md)

- 🔁 Stability & reproducibility  
  → [docs/Stability_and_Reproducibility.md](docs/Stability_and_Reproducibility.md)

- ⚖️ Comparison vs EQ-based correction  
  → [docs/Comparison_vs_EQ.md](docs/Comparison_vs_EQ.md)

---

![Effect of Temporal Decay Control](tdc_impulse_example.png)
