# SNN with STDP + eligibility trace to simulate an artificial tongue 
# that continuously learns to recognize multiple tastes (always-on).

import brian2 as b
b.prefs.codegen.target = 'numpy'
import matplotlib.pyplot as plt
import sys
import random
import numpy as np

# seed for the random reproducibility
np.random.seed(0)
b.seed(0)

# Rates vector helper with normalization
def set_stimulus_vect_norm(rate_vec, total_rate=None):
    r = np.asarray(rate_vec, dtype=float).copy()
    r[unknown_id] = 0.0
    if total_rate is not None and r.sum() > 0:
        r *= float(total_rate) / r.sum()
    pg.rates = r * b.Hz

# Rates vector helper without normalization
def set_stimulus_vector(rate_vec):
    r = np.asarray(rate_vec, dtype=float).copy()
    r[unknown_id] = 0.0
    pg.rates = r * b.Hz

# 1. Initialize the simulation
b.start_scope()
b.defaultclock.dt = 0.1 * b.ms  # high temporal precision
print("\n- ARTIFICIAL TONGUE's SNN with STDP, ELIGIBILITY TRACE and LATERAL INHIBITION: WTA (Winner-Take-All) -")

# 2. Define tastes
num_tastes = 8  # SWEET, BITTER, SALTY, SOUR, UMAMI, FATTY, SPICY, UNKNOWN

taste_labels = ["SWEET", "BITTER", "SALTY", "SOUR", "UMAMI",
                 "FATTY", "SPICY", "UNKNOWN"]
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
# map from index to label
taste_map = {idx: lbl for idx, lbl in enumerate(taste_labels)}

# Separate normal and special taste
normal_tastes = {idx: lbl for idx, lbl in taste_map.items() if lbl != "UNKNOWN"} # without UNKNOWN
special_taste = {idx: lbl for idx, lbl in taste_map.items() if lbl == "UNKNOWN"} # only UNKNOWN

# Print all tastes
print("\nAll tastes available:")
for idx, lbl in normal_tastes.items():
    print(f"{idx}: {lbl}")

print("\nSpecial taste:")
for idx, lbl in special_taste.items():
    print(f"{idx}: {lbl}")
unknown_id = num_tastes-1

# 3. Simulation global parameters
tau                = 30 * b.ms        # STDP time constant
Te                 = 50 * b.ms        # eligibility trace decay time constant
alpha              = 0.1              # learning rate for positive reward
beta               = 0.02             # learning rate for negative reward
noise_mu           = 5                # noise mu constant
noise_sigma        = 0.8              # noise sigma constant
inhib_amp          = 0.1              # lateral inhibition constant
training_duration  = 1000 * b.ms      # stimulus duration
pause_duration     = 100 * b.ms       # pause for eligibility decay
n_repeats          = 10               # repetitions per taste
progress_bar_len   = 30               # characters
weight_monitors    = []               # list for weights to monitor
threshold_ratio    = 0.7              # threshold for winner spiking neurons
min_spikes_for_known = 5              # minimum number of spikes for neuron, otherwise UNKNOWN
weight_decay      = 1e-4              # weight decay for trial
verbose_rewards   = False             # dopamine reward logs


# 4. Create taste neurons
taste_neurons = b.NeuronGroup(
    num_tastes,
    model='dv/dt = -v/(10*ms) : 1',
    threshold='v>1.0',
    reset='v=0',
    refractory=2*b.ms, # refractory period to stabilize WTA
    method='exact'
)

# 5. Monitors
spike_mon = b.SpikeMonitor(taste_neurons)
state_mon = b.StateMonitor(taste_neurons, 'v', record=True)

# 6. Poisson inputs and STDP + eligibility trace synapses
# 1) Labelled stimulus (yes plasticity)
pg = b.PoissonGroup(num_tastes, rates=np.zeros(num_tastes)*b.Hz)

# 2) Neutral noise (no plasticity)
baseline_hz = 0.5  # 0.5–1 Hz
pg_noise = b.PoissonGroup(num_tastes, rates=baseline_hz*np.ones(num_tastes)*b.Hz)

# STDP Synapses with eligibility trace and lateral inhibition WTA
S = b.Synapses(
    pg, taste_neurons,
    model='''
        w             : 1
        dApre/dt      = -Apre/tau   : 1 (event-driven)
        dApost/dt     = -Apost/tau  : 1 (event-driven)
        delig/dt      = -elig/Te    : 1 (clock-driven)
    ''',
    on_pre='''
        v_post += w
        Apre    += A_plus
        elig    += Apost
    ''',
    on_post='''
        Apost   += A_minus
        elig    += Apre
    ''',
    method='exact',
    namespace={'tau': tau, 'Te': Te,
               'A_plus': 0.01, 'A_minus': -0.012}
)

# Fully-connected except for UNKNOWN
S.connect('i == j and i != unknown_id')
S.w = '0.2 + 0.8*rand()'
# Diagonal synapses index
diag_idx = {k: int(np.where((S.i==k) & (S.j==k))[0][0]) for k in range(num_tastes-1)}
# noise added
S_noise = b.Synapses(pg_noise, taste_neurons,
                     model='w_noise : 1',
                     on_pre='v_post += w_noise')
S_noise.connect('i==j and i != unknown_id')
S_noise.w_noise = 0.02

w_mon = b.StateMonitor(S, 'w', record=True)
weight_monitors.append((w_mon, S))

# 7. Lateral inhibition for WTA (Winner-Take-All)
inhibitory_S = b.Synapses(
    taste_neurons, taste_neurons,
    on_pre=f'v_post -= {inhib_amp}', delay=1*b.ms
)
inhibitory_S.connect(condition='i!=j')

# 8. Building the SNN network and adding levels
net = b.Network(
    taste_neurons, 
    pg,
    pg_noise, # imput neurons noise introducted
    S,
    S_noise,
    inhibitory_S,
    spike_mon, 
    state_mon
)
net.add(w_mon)

# 9. Prepare stimuli list
# 9A: pure‐taste training
pure_train = []
for taste_id in range(num_tastes-1):  # 0..6
    for _ in range(n_repeats):
        noise = np.clip(np.random.normal(noise_mu, noise_sigma, num_tastes), 0, None)
        noise[taste_id] = 100
        pure_train.append((noise, [taste_id],
                           f"TASTE: {taste_id} - '{taste_map[taste_id]}'"))

# 9B: mixture training
mixture_train = []
for _ in range(n_repeats):
    # SWEET+SOUR
    noise = np.clip(np.random.normal(noise_mu, noise_sigma, num_tastes), 0, None)
    noise[0], noise[3] = 300, 300
    mixture_train.append((noise, [0,3], "TASTE: 'SWEET' + 'SOUR' (train)"))
    # SWEET+SALTY
    noise = np.clip(np.random.normal(noise_mu, noise_sigma, num_tastes), 0, None)
    noise[0], noise[2] = 250, 250
    mixture_train.append((noise, [0,2], "TASTE: 'SWEET' + 'SALTY' (train)"))

# 9C: test set
test_stimuli = [
    (np.array([300,0,0,300,0,0,0,0]), [0,3], "TASTE: 'SWEET' + 'SOUR'"),
    (np.array([250,0,250,0,0,0,0,0]), [0,2], "TASTE: 'SWEET' + 'SALTY'")
]
# total stimuli
training_stimuli = pure_train + mixture_train
random.shuffle(training_stimuli) # continually randomize the stimuli without adapting patterns

# 10. Main "always-on" loop
print("\nStarting TRAINING phase...")
S.Apre[:]  = 0
S.Apost[:] = 0
S.elig[:]  = 0
taste_neurons.v[:] = 0

step = 0
total_steps = len(training_stimuli) # pure + mixture

for input_rates, true_ids, label in training_stimuli:
    step += 1
    frac   = step / total_steps
    filled = int(frac * progress_bar_len)
    bar    = '█'*filled + '░'*(progress_bar_len - filled)
    if len(true_ids) == 1:
        reaction = taste_reactions[true_ids[0]]
        msg = (f"\r[{bar}] {int(frac*100)}% | Step {step}/{total_steps} | {label} | {reaction}")
    else:
        msg = (f"\r[{bar}] {int(frac*100)}% | Step {step}/{total_steps} | {label} (mixture)")
    sys.stdout.write(msg); 
    sys.stdout.flush()

    # 1) training stimulus with masking on no target neurons
    masked = np.zeros_like(input_rates)
    masked[true_ids] = input_rates[true_ids]
    set_stimulus_vect_norm(masked, total_rate=500)

    # 2) spikes counting during trial
    prev_counts = spike_mon.count[:].copy()
    net.run(training_duration)
    diff_counts = spike_mon.count[:] - prev_counts
    if diff_counts.max() <= 0:
        print("There's no computed spike, skipping rewarding phase...")
        S.elig[:] = 0
        net.run(pause_duration)
        continue

    # 3) "over-threshold" winners except for UNKNOWN
    scores = diff_counts.astype(float)
    scores[unknown_id] = -1e9
    mx = scores.max()

    # selecting all the spiking neurons >= threshold_ratio
    winners = np.where(scores >= threshold_ratio * mx)[0].tolist()
    if not winners:
        winners = [int(np.argmax(scores))]  # fallback: the best one
    
    # total scores printing
    order = np.argsort(scores)
    dbg = [(taste_map[i], int(scores[i])) for i in order[::-1]]

    # 4) training reinforcement learning dopamine rewards for the winner neurons
    for wid in winners:
        si = diag_idx[wid]
        r = alpha if wid in true_ids else -beta
        old_w = float(S.w[si])
        delta = r * float(S.elig[si])
        S.w[si] = float(np.clip(S.w[si] + delta, 0, 1))
        # visualizing who is rewarding
        if verbose_rewards:
            print(f"  reward on {taste_map[wid]}: r={r:+.3f}, elig={float(S.elig[si]):.4f}, "
                f"Δw={delta:+.4f}, w: {old_w:.3f}→{float(S.w[si]):.3f}")
        S.elig[si] = 0

    # light weight decay for all the weights to avoid constant saturation to w=1
    if weight_decay > 0:
        S.w[:] = np.clip(S.w[:] * (1 - weight_decay), 0, 1)

    # 5) eligibility trace decay among trials
    net.run(pause_duration)
    S.elig[:] = 0

print("\nEnded TRAINING phase!")
print(f"Target weights after training:")
for idx in range(num_tastes-1):
    si = np.where((S.i==idx)&(S.j==idx))[0][0]
    print(f"  {taste_map[idx]}→{taste_map[idx]} = {S.w[si]:.3f}")

# weights copying before TEST
print("\n— Unsupervised TEST phase with STDP frozen —")
w_before_test = S.w[:].copy()
test_w_mon = b.StateMonitor(S, 'w', record=True)
net.add(test_w_mon)

# 11. Freezing STDP for testing phase
print("Freezing STDP for TEST phase…")
S.pre.code  = 'v_post += w'
S.post.code = '' # no weights updating during test
taste_neurons.v[:] = 0
S.Apre[:] = 0; S.Apost[:] = 0; S.elig[:] = 0

# 12. TEST PHASE
print("\nStarting TEST phase...")
results = []
test_duration = 300 * b.ms
recovery_between_trials = 100 * b.ms  # refractory recovery

exact_hits = 0
total_test = len(test_stimuli)

for step, (_rates_vec, true_ids, label) in enumerate(test_stimuli, start=1):
    # progress bar
    frac   = step / total_test
    filled = int(frac * progress_bar_len)
    bar    = '█'*filled + '░'*(progress_bar_len - filled)
    sys.stdout.write(f"\r[{bar}] {int(frac*100)}% | Step {step}/{total_test} | Testing → {label}")
    sys.stdout.flush()

    # 1) stimulus on target classes
    set_stimulus_vect_norm(_rates_vec, total_rate=500)

    # 2) spikes counting during trial
    prev_counts = spike_mon.count[:].copy()
    net.run(test_duration)
    diff_counts = spike_mon.count[:] - prev_counts

    # 3) take the winners
    scores = diff_counts.astype(float)
    scores[unknown_id] = -1e9
    mx = scores.max()

    # if spikes threshold is weak => WINNERS, otherwise => UNKNOWN
    if mx < min_spikes_for_known:
        winners = [unknown_id]
    else:
        winners = np.where(scores >= threshold_ratio * mx)[0].tolist()
        if not winners:
            winners = [int(np.argmax(scores))]

    order = np.argsort(scores)
    dbg = [(taste_map[i], int(scores[i])) for i in order[::-1]]
    print("\nTest scores:", dbg)

    # to make a confrontation: expected vs predicted values
    expected  = [taste_map[i] for i in true_ids]
    predicted = [taste_map[w] for w in winners]
    hit = set(winners) == set(true_ids)

    # output visualization
    print(f"\n{label}\n  expected:   {expected}\n  predicted: {predicted}\n  exact:   {hit}")
    results.append((label, expected, predicted, hit))

    # 4) final valutation
    if set(winners) == set(true_ids):
        exact_hits += 1

    # 5) final refractory period after trial
    net.run(recovery_between_trials)

# Metrics classification report with Jaccard class and confusion matrix
# a. Test accuracy
ok = 0
for label, exp, pred, hit in results:
    status = "OK" if hit else "MISS"
    print(f"{label:26s} | expected={exp} | predicted={pred} | {status}")
    ok += int(hit)
print(f"\nTest accuracy (exact-set match): {ok}/{len(results)} = {ok/len(results):.2%}")

# b. Jaccard class, recall, precision, f1-score
label_to_id = {lbl: idx for idx, lbl in taste_map.items()}
classes = [i for i in range(num_tastes) if i != unknown_id]
# Class counters
stats = {i: {'tp': 0, 'fp': 0, 'fn': 0} for i in classes}
jaccard_per_case = []
for _, exp_labels, pred_labels, _ in results:
    T = {label_to_id[lbl] for lbl in exp_labels if label_to_id[lbl] != unknown_id}
    P = {label_to_id[lbl] for lbl in pred_labels if label_to_id[lbl] != unknown_id}

    inter = T & P
    union = T | P
    jaccard_per_case.append(len(inter) / len(union) if len(union) > 0 else 1.0)

    for c in classes:
        if c in P and c in T:
            stats[c]['tp'] += 1
        elif c in P and c not in T:
            stats[c]['fp'] += 1
        elif c not in P and c in T:
            stats[c]['fn'] += 1

# Metric helpers
def prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1   = (2*prec*rec/(prec+rec)) if np.isfinite(prec) and np.isfinite(rec) and (prec+rec) > 0 else float('nan')
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')  # Jaccard per class
    return prec, rec, f1, iou

def fmt_pct(x):
    return "—" if not np.isfinite(x) else f"{x*100:5.1f}%"

# Confusion matrix
print("\nMulti-label metrics for every taste:")
for c in classes:
    tp, fp, fn = stats[c]['tp'], stats[c]['fp'], stats[c]['fn']
    prec, rec, f1, iou = prf(tp, fp, fn)
    print(f"{taste_map[c]:>6s}: TP={tp:2d} FP={fp:2d} FN={fn:2d} | "
          f"P={fmt_pct(prec)} R={fmt_pct(rec)} F1={fmt_pct(f1)} IoU={fmt_pct(iou)}")
    
# Micro / Macro
sum_tp = sum(d['tp'] for d in stats.values())
sum_fp = sum(d['fp'] for d in stats.values())
sum_fn = sum(d['fn'] for d in stats.values())

micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else float('nan')
micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else float('nan')
micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)) if np.isfinite(micro_p) and np.isfinite(micro_r) and (micro_p+micro_r) > 0 else float('nan')

per_class_prec = [prf(stats[c]['tp'], stats[c]['fp'], stats[c]['fn'])[0] for c in classes]
per_class_rec  = [prf(stats[c]['tp'], stats[c]['fp'], stats[c]['fn'])[1] for c in classes]
per_class_f1   = [prf(stats[c]['tp'], stats[c]['fp'], stats[c]['fn'])[2] for c in classes]
per_class_iou  = [prf(stats[c]['tp'], stats[c]['fp'], stats[c]['fn'])[3] for c in classes]

macro_p = float(np.nanmean(per_class_prec)) if len(per_class_prec) else float('nan')
macro_r = float(np.nanmean(per_class_rec))  if len(per_class_rec)  else float('nan')
macro_f1 = float(np.nanmean(per_class_f1))  if len(per_class_f1)  else float('nan')
mean_iou = float(np.nanmean(per_class_iou)) if len(per_class_iou) else float('nan')

print("\n— Micro/Macro —")
print(f"Micro  -> P={fmt_pct(micro_p)} R={fmt_pct(micro_r)} F1={fmt_pct(micro_f1)}")
print(f"Macro  -> P={fmt_pct(macro_p)} R={fmt_pct(macro_r)} F1={fmt_pct(macro_f1)}")
print(f"Mean IoU (per-class): {fmt_pct(mean_iou)}")

# Jaccard per test-case (expected vs predicted)
if jaccard_per_case:
    mean_jaccard_cases = float(np.mean(jaccard_per_case))
    print("\nJaccard per test-case:", [f"{j:.2f}" for j in jaccard_per_case])
    print(f"Average Jaccard (set vs set): {mean_jaccard_cases:.2f}")

# weight changes during test confrontation
print("\nWeight changes during unsupervised test:")
for idx in range(num_tastes-1):
    si = np.where((S.i==idx)&(S.j==idx))[0][0]
    delta = float(S.w[si] - w_before_test[si])
    print(f"  {taste_map[idx]}→{taste_map[idx]}: Δw = {delta:+.3f}")

print("\nEnded TEST phase successfully!")

# 13. Plots
# a) Spikes over time
plt.figure(figsize=(10,4))
plt.plot(spike_mon.t/b.ms, spike_mon.i, '.k')
plt.xlabel("Time (ms)")
plt.ylabel("Neuron index")
plt.title("All spikes (input neurons silent, only taste_neurons)")
plt.show()

# b) Weight trajectories for diagonal synapses (i→i)
plt.figure(figsize=(14,6))
wm, syn = weight_monitors[0]  # monitor + associated synapse object
has_labels = False  # track if we actually added any visible label

for record_index, syn_index in enumerate(wm.record):
    pre = int(syn.i[syn_index])
    post = int(syn.j[syn_index])
    if pre == post and pre != unknown_id:
        plt.plot(wm.t/b.ms, wm.w[record_index], label=taste_map[pre])
        has_labels = True

plt.xlabel("Time (ms)")
plt.ylabel("Weight w")
plt.title("STDP + eligibility: diagonal synapses over time")

# Only show legend if we actually added any labeled line
if has_labels:
    plt.legend(loc='upper right')
else:
    print("[WARNING] No diagonal synapse (i→i) found among monitored synapses. Legend skipped.")

plt.tight_layout()
plt.show()

# c) Membrane potentials for all neurons
plt.figure(figsize=(14,6))
for idx in range(num_tastes):
    if idx == unknown_id:
        continue
    plt.plot(state_mon.t/b.ms, state_mon.v[idx], label=taste_map[idx])
plt.xlabel("Time (ms)")
plt.ylabel("v")
plt.title("Membrane potentials during always-on loop")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()