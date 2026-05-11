import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import kstest, expon
import random
from numba import njit
import pandas as pd
import seaborn as sns
from seaborn import set_theme
set_theme()

# Import trials

trial_indexes = range(1, 11)

trials = dict()
n_times = np.zeros(250)
for trial in trial_indexes:
    df = pd.read_csv(f'neuronal_data/{trial}.csv', index_col=0)
    path = []
    for mark, id in enumerate(df.index):
        times = df.loc[id][df.loc[id] > 0].values
        path.extend(list(zip(times, np.ones_like(times, dtype=int)*(mark+1))))
        n_times[mark] += len(times)

    path = sorted(path, key=lambda x: x[0])
    path = [(t-path[0][0], m) for t, m in path]
    trials[trial] = path

print(f'Keep trials with less that 10 non-active neurons')
n_jumps = dict()
sub_trials = dict()
for key, values in trials.items():
    n_jumps[key] = np.ones(250)
    for j in range(1, 251):  # Marks
        times = [t for t, m in values if m==j]
        n_jumps[key][j-1] = len(times)

    if np.sum(n_jumps[key]==0) < 10:
        sub_trials[key] = trials[key]

min_jumps = np.ones(250) * 1e8
max_jumps = np.zeros(250)
for key, values in sub_trials.items():
    for j in range(1, 251):  # Marks
        times = [t for t, m in values if m==j]
        min_jumps[j-1] = np.fmin(min_jumps[j-1], len(times))
        max_jumps[j-1] = np.fmax(max_jumps[j-1], len(times))


n_min = 50
print('\n')
print(f'Keep neurons with at least {n_min} jumps over all trials')
for key, values in sub_trials.items():
    sub_trials[key] = [(t, m) for t, m in values if min_jumps[m-1] >= n_min]
keeped_neurons = np.unique([m for t, m in sub_trials[key]])

print('\n')
print(f'Kept neurons: {keeped_neurons} ({len(keeped_neurons)})')

new_trial_indexes = [key for key, values in sub_trials.items()]

print('\n')
print(f'Kept trials:', new_trial_indexes)

time_cut = 10
print('\n')
print(f'Cut time at {time_cut}')
for key, values in sub_trials.items():
    sub_trials[key] = [(t, m) for t, m in values if t <= time_cut]

# Rewrite with correct indexes

new_trials = {}
for trial in new_trial_indexes:
    filtered_data = sub_trials[trial]  

    path = []
    for mark, neuron_id in enumerate(keeped_neurons):
        times = [time for time, n_id in filtered_data if n_id == neuron_id]
        path.extend([(time, mark ) for time in times])

    path = sorted(path, key=lambda x: x[0])
    #path = [(t - path[0][0], m ) for t, m in path]

    if path and path[0][0] == 0:
        path = path[1:]

    new_trials[trial] = path

# Resample 

nb_samples = 25
nb_processes = 3

resampled_trial_indexes = np.arange(nb_samples)
resampled_trials = [ [] for _ in range(nb_samples)]

for k in range(nb_samples):
    indexes = random.sample(new_trial_indexes, nb_processes)
    s = 0
    for j in indexes:
        resampled_trials[k] += [(t+s,m) for t,m in new_trials[j]]
        s = resampled_trials[k][-1][0]

# Represent mean number of spikes for each resampled trial

def cumulative_rate_for_process(trial, total_time, num_time_points=100):
    time_points = np.linspace(0.01, total_time, num_time_points)  # Avoid t=0
    spike_times = sorted(spike_time for spike_time, _ in trial)

    cumulative_counts = np.zeros(len(time_points))
    idx = 0  # Index to track spikes

    # Efficiently count spikes up to each time point
    for i, t in enumerate(time_points):
        while idx < len(spike_times) and spike_times[idx] <= t:
            idx += 1
        cumulative_counts[i] = idx

    cumulative_rate = cumulative_counts / time_points  # N(t) / t
    return time_points, cumulative_rate

fig,ax=plt.subplots(figsize=(10,5))

for index in resampled_trial_indexes:
    trial_data = resampled_trials[index]
    max_time = trial_data[-1][0]
    time_points_total, cumulative_rate_total = cumulative_rate_for_process(trial_data, max_time)
    ax.plot(time_points_total, cumulative_rate_total, label=f'Trial {index}')
#plt.legend()
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)

plt.show()