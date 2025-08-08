# SNN with STDP to simulate an artificial tongue that feel some different tastes which stimulate different neurons

import brian2 as b
b.prefs.codegen.target = 'numpy'
import matplotlib.pyplot as plt
import sys
import numpy as np

# Initialize the simulation
b.start_scope()
b.defaultclock.dt = 0.1 * b.ms  # more precision

# 1. Initialize all 7 possible tastes + 1 unknown taste
# SWEET, BITTER, SALTY, SOUR, UMAMI, FATTY, SPICY + UNKNOWN
num_tastes = 8 # numeric counter for tastes
taste_labels = ["SWEET", "BITTER", "SALTY", "SOUR", "UMAMI", "FATTY", "SPICY", "UNKNOWN"]
taste_reactions = {
    0: "Ouh... yummy!         ",
    1: "So acid!              ",
    2: "Need water... now!    ",
    3: "Mehhh!                ",
    4: "So delicious!         ",
    5: "Oh, I'm a big fat boy!",
    6: "I'm a blazing dragon! ",
    7: "WTF!                  "
}
# tastes dictionary -> { num_taste : taste_label }
taste_map = {idx: label for idx, label in enumerate(taste_labels)}
print("All tastes available:")
print(taste_map)

# It is possible to manage Lateral Inhibition for only one neuron at time, or it is also possible to identify several tastes at time
mode = 'multi' # 'single' per WTA, 'multi' per thresholding multi-label
tau = 20 * b.ms
training_duration = 500 * b.ms
pause_duration = 100 * b.ms
n_repeats = 10

# 2. Creating 1 output neuron per taste -> the same number as inputs
taste_neurons = b.NeuronGroup(num_tastes,
    model='dv/dt = (-v) / (10*ms) : 1',
    threshold='v > 1.0',
    reset='v = 0',
    method='exact'
)

# Monitors
spike_mon = b.SpikeMonitor(taste_neurons)
state_mon = b.StateMonitor(taste_neurons, 'v', record=True)

# SNN components
synapses = []
weight_monitors = []
training_net = b.Network()
training_net.add(taste_neurons, spike_mon, state_mon)

# 3. Main training loop: one taste at a time
print("Training started...")
step = 0
total_steps = num_tastes * n_repeats
for taste_id in range(num_tastes):
    for repeat in range(n_repeats):
        step += 1
        bar_len = 30
        frac = step/total_steps
        filled = int(frac*bar_len)
        reaction = taste_reactions.get(taste_id, "")
        bar = '█'*filled + '░'*(bar_len-filled)
        msg = (f"\r[{bar}] {int(frac*100)}% "
               f"| Step {step}/{total_steps} "
               f"| Taste {taste_id} - {taste_map[taste_id]} | {reaction}")
        sys.stdout.write(msg); 
        sys.stdout.flush()

        # input noise added
        noise = np.clip(np.random.normal(10,5,num_tastes),0,None)
        noise[taste_id] = 100
        pg = b.PoissonGroup(num_tastes, rates=noise*b.Hz)

        # STDP synapses 
        S = b.Synapses(
            pg, taste_neurons,
            model='''
                w : 1
                dApre/dt  = -Apre/tau : 1 (event-driven)
                dApost/dt = -Apost/tau: 1 (event-driven)
            ''',
            on_pre='''
                v_post += w
                Apre += 0.01
                w = clip(w + Apost, 0, 1)
            ''',
            on_post='''
                Apost += 0.01
                w = clip(w + Apre, 0, 1)
            ''',
            namespace={'tau': tau}
        )
        S.connect(condition='True') # fully-connected
        S.w = '0.2 + rand()*0.8'

        if repeat == 0:
            wm = b.StateMonitor(S, 'w', record=True)
            training_net.add(wm)
            weight_monitors.append((wm, S))
        if repeat == n_repeats-1:
            synapses.append(S)

        training_net.add(pg, S)
        # execution
        training_net.run(training_duration)

        # Reinforcement with dopamine during training: only on the correct synapse
        idxs = np.where((S.i==taste_id)&(S.j==taste_id))[0]
        if len(idxs)==1:
            si = idxs[0]
            # bump +0.05 and clip
            new_w = float(np.clip(S.w[si] + 0.05, 0, 1))
            S.w[si] = new_w

        # pause for eligibility
        training_net.run(pause_duration)

        # cleanup
        training_net.remove(pg, S)
        if repeat == 0:
            training_net.remove(wm)

print("\nTraining finished!\n")
training_net.run(50*b.ms)
taste_neurons.v = 0
print("Total spikes during training:", spike_mon.num_spikes)

# LATERAL INHIBITION: for the neurons competition -> Winner-Takes-All (WTA)
inhibitory_S = b.Synapses(taste_neurons, taste_neurons, on_pre='v_post -= 0.3', delay=1*b.ms)
inhibitory_S.connect(condition='i != j')
training_net.add(inhibitory_S)
if mode == 'multi':
    training_net.remove(inhibitory_S)

# 5. Plots
plt.figure(figsize=(10, 4))
plt.plot(spike_mon.t / b.ms, spike_mon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('Taste neuron spikes during supervised training')
plt.show()

plt.figure(figsize=(14, 6))
for idx, (wm, _) in enumerate(weight_monitors):
    if wm.w.shape[0] > 0:
        plt.plot(wm.t / b.ms, wm.w[0], label=f'{taste_map[idx]}')
plt.xlabel('Time (ms)')
plt.ylabel('Synaptic weight')
plt.title('STDP weight changes for each taste (first synapse)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
for idx in range(num_tastes):
    plt.plot(state_mon.t / b.ms, state_mon.v[idx], label=f'{taste_map[idx]}')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane potential (v)')
plt.title('Membrane potentials of all taste neurons during training')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# 6. Testing phase - multiple tastes are stimulated
print("\nTESTING PHASE - Multiple tastes")
test_taste_id = [0, 3]  # SWEET and SOUR
print(f"Stimulating test tastes: {test_taste_id} ({', '.join([taste_map[k] for k in test_taste_id])})")

test_rates = [0] * num_tastes
for t_id in test_taste_id:
    test_rates[t_id] = 300 # stimulation rate

# Synapses with reinforcement learning dopaminergic reward -> the clip line manage this
test_pg = b.PoissonGroup(num_tastes, rates=[r*b.Hz for r in test_rates])
test_S = b.Synapses(test_pg, taste_neurons, 
'''
    w : 1
    reward : 1
''',
on_pre='''
    v_post += w
    w = clip(w + 0.05 * reward - 0.03 * (1 - reward), 0, 1)
''')
test_S.connect(condition='True')
# initialize the reward variable
test_S.reward = 0

# saving just the specific weights of fired neurons
for tid in test_taste_id:
    relevant_weights = []
    for s in synapses:
        try:
            weight = s.w[tid, tid]
            relevant_weights.append(weight)
        except IndexError:
            continue
    if relevant_weights:
        test_S.w[tid, tid] = np.mean(relevant_weights)
    else:
        print(f"[WARNING] No valid synapses found for taste {tid} ({taste_map[tid]})")
print(f"Assigned mean weight from training to test synapse [{tid},{tid}]: {np.mean(relevant_weights):.3f}")

test_spike_mon = b.SpikeMonitor(taste_neurons)

# first test for multiple tastes
training_net.add(test_pg, test_S, test_spike_mon)
training_net.run(300 * b.ms)
counts = test_spike_mon.count
print("Spike counts (testing - multiple tastes):")
for idx, count in enumerate(counts):
    print(f"Taste neuron {idx} ({taste_map[idx]}): {count} spikes")

# Automatic classification
spike_counts = np.array(test_spike_mon.count)
min_spikes = 1
if mode == 'single':
    winners = [int(np.argmax(spike_counts))]  # unico vincitore in single‐label
elif mode == 'multi':
    winners = [i for i, c in enumerate(spike_counts) if c >= min_spikes]
else:
    raise ValueError(f"Unknown mode {mode!r}")
print(f"\nWinners: {[taste_map[i] for i in winners]}")

# 7. REINFORCEMENT LEARNING: Dopaminergic feedback
print("\nReinforcement learning phase with dopaminergic feedback...")
if mode == 'single':
    # in single-label -> only one winner
    correct_classification = winners[0] in test_taste_id
else:
    # in multi-label -> several winners
    correct_classification = all(t in winners for t in test_taste_id)

# initializing reward
test_S.reward = 0

# reward = 1 for winners
for win in winners:
    syn_idx = np.where((test_S.i == win) & (test_S.j == win))[0]
    if len(syn_idx) == 1:
        test_S.reward[syn_idx[0]] = 1

if correct_classification:
    print("Correct classification! Reinforcement learning with reward applied.\n")
else:
    print("Incorrect classification! No reinforcement learning with reward applied.\n")
training_net.run(200 * b.ms)
print("Reinforcement learning complete!")

# 8. Showing updated weights
print("\nSynaptic weights after reinforcement learning:")
for win in winners:
    syn_idx = np.where((test_S.i == win) & (test_S.j == win))[0]
    if len(syn_idx) != 1:
        print(f"[ERROR] synapse for ({win}→{win}) not found or duplicated: {syn_idx}")
        continue
    si = syn_idx[0]
    prev_w = float(test_S.w[si])
    if correct_classification:
        new_w = np.clip(prev_w + 0.05, 0, 1)
    else:
        new_w = np.clip(prev_w - 0.03, 0, 1)
    test_S.w[si] = float(new_w)
    print(f"Weight for {taste_map[win]}→{taste_map[win]}: {new_w:.3f}")

# Print the top activated neurons
top_neurons = sorted(enumerate(spike_counts), key=lambda x: x[1], reverse=True)[:3]
print("\nTop activated neurons:")
for idx, count in top_neurons:
    print(f"Neuron {idx} ({taste_map[idx]}): {count} spikes")

# Buffer net flush
training_net.remove(test_pg, test_S, test_spike_mon)