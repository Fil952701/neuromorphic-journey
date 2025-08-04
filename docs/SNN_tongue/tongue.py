# SNN with STDP to simulate an artificial tongue that feel some different tastes which stimulate different neurons

import brian2 as b
import matplotlib.pyplot as plt

# 0. Initialize simulation
b.start_scope()
duration = 5 * b.second

# 1. Initialize all 7 possible tastes + 1 unknown taste
# SWEET, BITTER, SALTY, SOUR, UMAMI, FATTY, SPICY + UNKNOWN
num_tastes = 8
input_rate = 20 * b.Hz # all tastes stimulated at the same time
taste_pattern = [100, 1, 1, 1, 1, 1, 1, 1] # Hz per taste -> there are only two strong stimolous

# The sensitivity is stochastic, so we initialize PoissonGroup receptors
poisson_inputs = [b.PoissonGroup(1, rates=rate*b.Hz) for rate in taste_pattern]

# 2. Creating 1 output neuron per taste -> the same number as inputs
taste_neurons = b.NeuronGroup(num_tastes,
    model='dv/dt = (-v) / (10*ms) : 1',
    threshold='v > 1',
    reset='v = 0',
    method='exact'
)

# 3. STDP synapses between each receptor and its corrisponding taste neuron
synapses = []
tau = 20*b.ms
for i in range(num_tastes):
    S = b.Synapses(poisson_inputs[i], taste_neurons[i:i+1], '''
        w : 1
        dApre/dt = -Apre / tau : 1 (event-driven)
        dApost/dt = -Apost / tau : 1 (event-driven)
    ''',
    on_pre='''
        v_post += w
        Apre += 0.01
        w = clip(w + Apost, 0, 1)
    ''',
    on_post='''
        Apost += 0.01
        w = clip(w + Apre, 0, 1)
    ''')

    S.connect()
    S.w = '0.8 + rand() * 0.2' # initialize synaptic weights randomly
    synapses.append(S)

# 4. Monitor neurons
# Spike monitor for taste neurons
spike_mon = b.SpikeMonitor(taste_neurons)
# Optionally monitor membrane potential
state_mon = b.StateMonitor(taste_neurons, 'v', record=True)
# Track the evolution of the weights
weight_monitors = [b.StateMonitor(S, 'w', record=True) for S in synapses]

# 5. Building SNN and start the simulation
# Include all objects in my new created custom network
net = b.Network()
net.add(poisson_inputs)
net.add(taste_neurons)
net.add(synapses)
net.add(spike_mon)
net.add(state_mon)
net.add(weight_monitors)

duration_int = int(duration)
# Start the simulation
print("Simulation started...")
net.run(duration)
for i in range(duration_int):
    print(i+1, "ms")
print("Simulation finished.")

# 6. Print the output
# Print total spike counts per neuron
print("Spike counts per taste neuron:")
print(spike_mon.count)  # array with counts for each neuron

# Print total number of spikes recorded
print("Total spikes:", spike_mon.num_spikes)

# Print individual spike times for each neuron
spike_trains = spike_mon.spike_trains()
print("Spike times per neuron:")
for neuron_idx, times in spike_trains.items():
    print(f"Neuron {neuron_idx}: {len(times)} spikes at times {times}")

# Print final weight for every stimulated taste
for i, S in enumerate(synapses):
    print(f"Final weight for taste {i}:", S.w[:])

# 7. Plot the output
plt.figure(figsize=(10, 4))
plt.plot(spike_mon.t / b.ms, spike_mon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Spikes of taste neurons')
plt.show()

# Plot the weights
plt.figure(figsize=(12, 6))
for i, wm in enumerate(weight_monitors):
    plt.plot(wm.t / b.ms, wm.w[0], label=f'Taste {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic weight')
plt.title('STDP weight changes per taste')
plt.legend()
plt.show()

# Plot membrane potential of first neuron (taste 0)
plt.figure(figsize=(10, 4))
plt.plot(state_mon.t / b.ms, state_mon.v[0])
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (v)')
plt.title('Membrane potential of neuron 0 (SWEET)')
plt.show()


