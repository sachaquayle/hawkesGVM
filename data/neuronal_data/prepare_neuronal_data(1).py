import numpy as np
import pickle as pkl
import pandas as pd


neurons = [48, 58, 72, 99, 115, 116, 119, 126, 140, 152, 153, 192, 194, 210, 230, 232, 240]  # 250
trial_indexes = [1, 2, 4, 5, 6, 7, 9, 10]

trials = dict()
for trial in trial_indexes:
    print(f'Trial {trial}')

    df = pd.read_csv(f'neuronal_data/{trial}.csv', index_col=0)
    df = df.loc[neurons]

    path = []
    for mark, id in enumerate(df.index):
        times = df.loc[id][df.loc[id] > 0].values
        print(f' Neuron {id} ({len(times)} times)')
        path.extend(list(zip(times, np.ones_like(times, dtype=int)*(mark+1))))

    path = sorted(path, key=lambda x: x[0])
    path = [(t-path[0][0], m) for t, m in path]
    trials[trial] = path

print('Summary')
for j in range(1, len(neurons)+1):
    print(f'Suprocess {j} (Neuron {neurons[j-1]}): {sum([len([t for t, k in path if k==j]) for _, path in trials.items()])} times')

for ipath in range(10):
    print(f'=== Path {ipath}')

    print('Training permutation')
    perm_train = np.random.permutation(list(trials.keys()))
    path = []
    start = 0.
    for id in perm_train:
        path.extend(sorted([(t+start, m) for t, m in trials[id]], key=lambda x: x[0]))
        print(f' {id} (start at {start:0.2f})')
        start = path[-1][0]

    print('Eval permutation')
    test_train = np.random.permutation(list(trials.keys()))
    path_eval = []
    start = 0.
    for id in perm_train[:len(perm_train)//2]:
        path_eval.extend(sorted([(t+start, m) for t, m in trials[id]], key=lambda x: x[0]))
        print(f' {id} (start at {start:0.2f})')
        start = path_eval[-1][0]

    print('Test permutation')
    path_test = []
    start = 0.
    for id in perm_train[len(perm_train)//2:]:
        path_test.extend(sorted([(t+start, m) for t, m in trials[id]], key=lambda x: x[0]))
        print(f' {id} (start at {start:0.2f})')
        start = path_test[-1][0]

    with open(f'neuronal_data/paths_{ipath}.pkl', 'wb') as file:
        pkl.dump(dict(train=path, test=path_test, eval=path_eval), file)
