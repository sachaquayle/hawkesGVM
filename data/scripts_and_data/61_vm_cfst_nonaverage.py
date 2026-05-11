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

with open("first_estimation_61_vm.txt", "r") as file:
    file_content = file.read()
exec(file_content)

hawkes_vm = ExponentialHawkesGVM(model='vm')

hawkes_vm.d = d
hawkes_vm.multiple_estimations = vm_estimations
hawkes_vm.nb_realisations = len(resampled_trials)
hawkes_vm.alpha_zero_coefficients = interaction_matrix_cfst


# final estimation

start = time.time()

hawkes_vm.fit(trials, each_realisation=False)

vm_estimation = hawkes_vm.estimation
vm_pvalues = hawkes_vm.pvalues(nb_iterations=50, av_estimation=False, distribution='expon', method='cramervonmises')


end = time.time()

vm_execution_time_cfst = end-start


# print results

print('UNDER vm:')
print('Estimation time estimation:', end-start)


output_filename = "61_vm_cfst_nonaverage.txt"
with open(output_filename, "w") as f:
    f.write("vm_execution_time_cfst = ")
    f.write(f"    {vm_execution_time_cfst}")
    f.write("\n\n")

    f.write("vm_estimation = (\n")
    mu, alpha, beta = vm_estimation
    f.write(f"    np.array({repr(mu.tolist())}),\n")
    f.write(f"    np.array({repr(alpha.tolist())}),\n")
    f.write(f"    np.array({repr(beta.tolist())})\n")
    f.write(")\n\n")

    f.write("vm_pvalues = np.array(\n")
    f.write(f"{repr(vm_pvalues.tolist())}\n")
    f.write(")\n\n")


print(f"Estimation results with estimation for each trial saved to {output_filename}")



