# Neuromorphic Journey 🚀🧠

Obiettivo: documentare il mio percorso di studio e sviluppo su **Spiking Neural Networks (SNN)**, **STDP** e **ingegneria neuromorfica**.

## Roadmap (macro)
- Week 1–2: Fondamentali SNN (LIF, codifiche spike) + Brian2 setup
- Week 3–5: STDP pair-based & varianti (Brian2)
- Week 6–8: Surrogate gradients / ANN→SNN (Norse/SpikingJelly)
- Week 9–10: Framework/Hardware (Lava/Loihi simulator, SpiNNaker toolchain)
- Week 11–12: Packaging (blog post, video demo), networking/internship hunt

## Log settimanale
### Week 1
- **Day 1**: setup environment, repo, LIF neuron with Poisson input (Brian2)
- **Day 2**: 2 neurons + synapse, parameter sweep
- **Day 3**: plan STDP implementation, start coding pair-based rule

_(Si aggiorna ogni settimana)_

## Struttura repo
- `notebooks/`: esperimenti, demo e visualizzazioni
- `src/`: modelli riutilizzabili (facoltativo all’inizio)
- `data/`: dataset spike-based (NMNIST, SHD, ecc.)
- `docs/`: note, roadmap dettagliata, paper list

## Requisiti
Vedi `requirements.txt`.

## Licenza
MIT
