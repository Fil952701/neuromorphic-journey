# Neuromorphic Journey 🚀🧠

Documenting my path into **Neuromorphic AI**, with a focus on **Spiking Neural Networks (SNNs)**, **STDP**, and brain-inspired hardware/software frameworks.

---

## 🔭 Goals

- Build solid theoretical foundations (neuronal models, spike coding, plasticity rules).
- Implement SNNs from scratch in Brian2 and train them with surrogate gradients (Norse/SpikingJelly).
- Experiment with STDP variants and ANN→SNN conversion.
- Explore neuromorphic frameworks/hardware (Intel Lava/Loihi, SpiNNaker, BrainScaleS – when accessible).
- Publish code, notes, and results openly (GitHub, blog posts, maybe short workshop papers).

---

## 🗺️ Macro Roadmap

**Weeks 1–2**  
Fundamentals: LIF neuron, spike encodings, Brian2 setup.

**Weeks 3–5**  
STDP (pair-based, triplet, homeostasis) in Brian2. Toy pattern-recognition task.

**Weeks 6–8**  
Training SNNs with surrogate gradients (Norse/SpikingJelly) on NMNIST/SHD or DVS Gesture. Compare direct training vs ANN→SNN conversion. Building some SNNs and STDP projects.

**Weeks 9–10**  
Touch hardware/SDK side: Intel Lava (Loihi simulator), SpiNNaker toolchain (if accessible). Write a tutorial-style demo.

**Weeks 11–12**  
Packaging & visibility: blog post, README polishing, small video demos. Start contacting labs/companies for internships or contracts.

## Struttura repo
- `notebooks/`: experiments, demo e visualizations
- `src/`: ensamble models
- `data/`: dataset spike-based (NMNIST, SHD, ecc.)
- `docs/`: notes, roadmap, paper list

## Requisiti
View `requirements.txt`.

## Licenza
MIT
