import matplotlib.pyplot as plt
import brian2 as b

b.prefs.codegen.target = "numpy"

# Parametri simulazione
duration = 100*b.ms
tau = 10*b.ms
v_th = 1
v_reset = 0

# Equazioni del modello LIF (Leaky Integrate-and-Fire)
eqs = '''
dv/dt = (1 - v)/tau : 1
'''

# Crea 3 neuroni LIF
G = b.NeuronGroup(3, eqs, threshold='v > v_th', reset='v = v_reset', method='exact')

# Stato iniziale: solo il primo neurone è carico
G.v = [1.1, 0, 0]

# Sinapsi: Neurone 0 → 1 e 1 → 2, ogni spike aggiunge +1.2 per dare impulso al successivo
S = b.Synapses(G, G, on_pre='v_post += 1.2')
S.connect(i=[0, 1], j=[1, 2])  # Neurone 0 stimola 1, 1 stimola 2

# Monitoraggio dei potenziali e spike
M = b.StateMonitor(G, 'v', record=True)
spikemon = b.SpikeMonitor(G)

# Simulazione
b.run(duration)

# Plot dei potenziali
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(M.t/b.ms, M.v[i], label=f'Neurone {i}')
plt.axhline(y=v_th, color='r', linestyle='--', label='Soglia di spike')
plt.xlabel('Tempo (ms)')
plt.ylabel('Potenziale di membrana')
plt.title('Attivazione a catena in una SNN con 3 neuroni')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Stampiamo anche il momento esatto degli spike
print("Spikes della Spike Neural Network (SNN) registrati:")
for i in range(len(G)):
    print(f"Neurone {i} ha 'sparato' ai tempi:", spikemon.t[spikemon.i == i])
