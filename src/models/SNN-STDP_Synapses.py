# SNN with STDP Synapses

import matplotlib.pyplot as plt
import brian2 as b

# Simulation parameters
tau_pre = 20*b.ms
tau_post = 20*b.ms
A_pre = 0.01
A_post = -A_pre * 1.05

# Create a group of 10 neurons
G = b.NeuronGroup(10, model='dv/dt = -v / (10*ms) : 1',
                threshold='v > 1', reset='v = 0', method='exact')

# Synapses with STDP
S = b.Synapses(G, G,
             model='''w : 1
                      dapre/dt = -apre / tau_pre : 1 (event-driven)
                      dapost/dt = -apost / tau_post : 1 (event-driven)''',
             on_pre='''v_post += w
                       apre += A_pre
                       w = clip(w + apost, 0, 1)''',
             on_post='''apost += A_post
                        w = clip(w + apre, 0, 1)''')

# Connect randomly with probability 20%
S.connect(p=0.2)
S.w = 0.5

# Define input to the first few neurons (simulate stimulation from outside)
input = b.TimedArray([3, 0, 0, 3, 0, 0, 3], dt=100*b.ms) # stimoli meno frequenti ma più potenti
#input = b.TimedArray([1]*10, dt=50*b.ms)  # più stimoli, più ravvicinati
stimulated_neurons = [0, 1, 2]
for i in stimulated_neurons: # stimulating i neuron
    G[i].run_regularly('v += input(t)', dt=100*b.ms, when='start')

# Monitors
weight_mon = b.StateMonitor(S, 'w', record=True) # monitor weight evolution
spikemon = b.SpikeMonitor(G) # record spikes
vmon = b.StateMonitor(G, 'v', record=True) # andamento del potenziale v

# Run simulation
b.run(700*b.ms)

# Plot synaptic weights from some synapses
plt.figure()
plt.subplot(121)
for i in range(min(5, len(weight_mon.w))):  # plot max 5 weight evolutions
    plt.plot(weight_mon.t/b.ms, weight_mon.w[i], label=f'Synapse {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic weight')
plt.title('STDP Weight Changes')
plt.legend()
plt.grid()

# Plot potential v
plt.figure()
for i in stimulated_neurons:
    plt.plot(vmon.t/b.ms, vmon.v[i], label=f'Neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v')
plt.title('v(t) per stimulated neurons')
plt.legend()
plt.grid()
plt.show()

# Raster plot of spikes
plt.subplot(122)
plt.plot(spikemon.t/b.ms, spikemon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spike Raster Plot')

plt.tight_layout()
plt.show()

# Print spike times for each neuron
print("Spike times per neuron:")
for i in range(len(G)):
    spikes = spikemon.t[spikemon.i == i]
    if len(spikes) > 0:
        print(f"Neuron {i} spiked at: {spikes}")
