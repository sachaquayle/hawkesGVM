import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import kstest, expon
import time
from numba import njit
import pandas as pd
import pickle as pkl
import copy
import random
import importlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from collections import defaultdict
np.set_printoptions(threshold=np.inf)
import pickle

import ClassGVM
from ClassGVM import negloglikelihood_with_grad_single_jit
from ClassGVM import ExponentialHawkesGVM

# import all neurons

with open("data_61.txt", "r") as file:
    file_content = file.read()
exec(file_content)

with open("7_trials.txt", "r") as file:
    file_content = file.read()
exec(file_content)

d = 61

with open("first_estimation_61_hp.txt", "r") as file:
    file_content = file.read()
exec(file_content)

hawkes_hp = ExponentialHawkesGVM(model='hp')

hawkes_hp.d = d
hawkes_hp.multiple_estimations = hp_estimations
hawkes_hp.nb_realisations = len(resampled_trials)
hawkes_hp.alpha_zero_coefficients = interaction_matrix_cfst


# final estimation

start = time.time()

hawkes_hp.fit(trials, each_realisation=False)

hp_estimation = hawkes_hp.estimation
hp_pvalues = hawkes_hp.pvalues(nb_iterations=50, av_estimation=False, distribution='expon', method='cramervonmises')

end = time.time()

hp_execution_time_cfst = end-start


# print results

print('UNDER HP:')
print('Estimation time estimation:', end-start)


output_filename = "61_hp_cfst_nonaverage.txt"
with open(output_filename, "w") as f:
    f.write("hp_execution_time_cfst = ")
    f.write(f"    {hp_execution_time_cfst}")
    f.write("\n\n")

    f.write("hp_estimation = (\n")
    mu, alpha, beta = hp_estimation
    f.write(f"    np.array({repr(mu.tolist())}),\n")
    f.write(f"    np.array({repr(alpha.tolist())}),\n")
    f.write(f"    np.array({repr(beta.tolist())})\n")
    f.write(")\n\n")

    f.write("hp_pvalues = np.array(\n")
    f.write(f"{repr(hp_pvalues.tolist())}\n")
    f.write(")\n\n")

print(f"Estimation results with estimation for each trial saved to {output_filename}")



