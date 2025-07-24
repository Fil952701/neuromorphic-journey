# SNN with STDP Synapses - Gaussian Noise and connections between synapses of every input neuron with designed output neuron

import matplotlib.pyplot as plt
import brian2 as b

# Simulation parameters
tau_pre = 20*b.ms
tau_post = 20*b.ms
A_pre = 0.01 # pre-synaptic neuron activation
A_post = -A_pre * 1.05 # post-synaptic neuron activation

# Create a group of 10 neurons with gaussian noise
G = b.NeuronGroup(10, model='''dv/dt = -v / (10*ms) + sigma * xi * (1/second)**0.5 : 1
                               sigma : 1''',
                threshold='v > 1', reset='v = 0', method='euler')

# Gaussian sigma noise constant
G.sigma = 0.2

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
S.connect(p=0.2) # there is only 20% probability of connections among neurons
# Forcing some synapse connections between input and output
forced_inputs = [(0,9), (1,9), (2,9)]
for i, j in forced_inputs:
    if not ((S.i[:] == i) & (S.j[:] == j)).any():
        S.connect(i=i, j=j)
S.w = '0.4 + 0.2*rand()' # weight variabilities

# Synapses for every neuron from 0, 1, 2 to 9
syn_0_9 = []
syn_1_9 = []
syn_2_9 = []
for k in range(len(S.i)):
    if S.i[k] in [0,1,2] and S.j[k] == 9:
        if S.i[k] == 0: syn_0_9.append(k)
        elif S.i[k] == 1: syn_1_9.append(k)
        elif S.i[k] == 2: syn_2_9.append(k)

# Define input to the first few neurons (simulate stimulation from outside)
#input = b.TimedArray([3, 0, 0, 3, 0, 0, 3], dt=100*b.ms) # stronger stimulation but less frequent
input = b.TimedArray([3]*18, dt=50*b.ms) # continuous stimulation frequency with same strong stimulation of less frequent version
stimulated_neurons = [0, 1, 2]
target_neuron = 9  # output neuron
for i in stimulated_neurons: # stimulating i neuron
    G[i].run_regularly('v += input(t)', dt=100*b.ms, when='start')

# Monitors
weight_mon = b.StateMonitor(S, 'w', record=True) # monitor weight evolution
spikemon = b.SpikeMonitor(G) # record spikes
vmon = b.StateMonitor(G, 'v', record=True) # potential v

# Run simulation
#b.run_regularly(900*b.ms)
b.run(1500*b.ms)

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

# Plot potential v + output neuron
plt.figure()
for i in stimulated_neurons + [target_neuron]:
    plt.plot(vmon.t/b.ms, vmon.v[i], label=f'Neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential v')
plt.title('v(t) - Stimulated and Output Neurons')
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

# Print spike times for output neuron
print("\nOutput neuron (9) analysis:")
output_spikes = spikemon.t[spikemon.i == target_neuron]
if len(output_spikes) > 0:
    print(f"Neuron 9 spiked at: {output_spikes}")
    if any(output_spikes > 600*b.ms):
        print("Neuron 9 fired in the late phase → possible learning detected.")
    else:
        print("Neuron 9 fired early or not at all → no clear learning yet.")
else:
    print("Neuron 9 did not fire → no learning.")

# Print synapse weights from input neurons to output neuron
print("\nTracking synapse weights from 0,1,2 to 9:")
for label, syn_list in zip([0,1,2], [syn_0_9, syn_1_9, syn_2_9]):
    if syn_list:
        last_weights = [weight_mon.w[k][-1] for k in syn_list]
        print(f"From neuron {label} → 9: final weight(s) = {[round(w,3) for w in last_weights]}")
    else:
        print(f"No synapse found from neuron {label} to neuron 9.")

# Plot them
plt.figure(figsize=(6,4))
for label, syn_list in zip([0,1,2], [syn_0_9, syn_1_9, syn_2_9]):
    for k in syn_list:
        plt.plot(weight_mon.t/b.ms, weight_mon.w[k], label=f'{label}→9')
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic weight')
plt.title('Weights from Neurons 0,1,2 → Neuron 9')
plt.legend()
plt.grid()
plt.show()

# Finding all the synapses on output neuron 9
syn_to_9 = [k for k in range(len(S.i)) if S.j[k] == target_neuron]

# Print final weight
print(f"\nSynaptic weights towards Neuron {target_neuron}:")
for k in syn_to_9:
    source = int(S.i[k])
    dest = int(S.j[k])  # always 9 because there's only one output neuron: 9
    final_weight = weight_mon.w[k][-1]
    print(f"Synapse from Neuron {source} → Neuron {dest}: Final weight = {final_weight:.3f}")

# Plot temporal variation of weights
plt.figure()
for k in syn_to_9:
    plt.plot(weight_mon.t / b.ms, weight_mon.w[k], label=f'{int(S.i[k])} → {int(S.j[k])}')
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic weight')
plt.title('Synaptic Weights to Neuron 9 Over Time')
plt.legend()
plt.grid()
plt.show()
