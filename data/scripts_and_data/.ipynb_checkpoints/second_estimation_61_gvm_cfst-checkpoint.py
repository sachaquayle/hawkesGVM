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


with open("first_estimation_61_gvm.txt", "r") as file:
    file_content = file.read()
exec(file_content)



hawkes = ExponentialHawkesGVM(model='gvm')


hawkes.alpha_zero_coefficients = interaction_matrix_cfst
hawkes.alpha_tilde_zero_coefficients = interaction_matrix_cfst


# second estimation

start = time.time()

hawkes.fit(resampled_trials)

gvm_estimations = hawkes.multiple_estimations
gvm_average_estimation = hawkes.average_estimation

hawkes.tests_hp_vm(asymptotic=True)

gvm_hp_matrix_cfst = hawkes.equal_coefficients
gvm_vm_matrix_cfst = hawkes.alpha_tilde_zero_coefficients



end = time.time()

gvm_execution_time_cfst = end-start


# print results

print('UNDER GVM:')
print('Estimation time estimation:', end-start)


output_filename = "second_estimation_61_gvm_cfst.txt"
with open(output_filename, "w") as f:
    f.write("gvm_execution_time_cfst = ")
    f.write(f"    {gvm_execution_time_cfst}")
    f.write("\n\n")

    f.write("gvm_estimations = [\n")
    for mu, alpha, beta, alpha_tilde in gvm_estimations:
        f.write(f"    (np.array({repr(mu.tolist())}),\n")
        f.write(f"     np.array({repr(alpha.tolist())}),\n")
        f.write(f"     np.array({repr(beta.tolist())}),\n")
        f.write(f"     np.array({repr(alpha_tilde.tolist())})),\n")
    f.write("]\n\n")

    f.write("gvm_average_estimation = (\n")
    mu, alpha, beta, alpha_tilde = gvm_average_estimation
    f.write(f"    np.array({repr(mu.tolist())}),\n")
    f.write(f"    np.array({repr(alpha.tolist())}),\n")
    f.write(f"    np.array({repr(beta.tolist())}),\n")
    f.write(f"    np.array({repr(alpha_tilde.tolist())})\n")
    f.write(")\n\n")

    f.write('gvm_hp_matrix_cfst =')
    f.write(f'   np.array({repr(gvm_hp_matrix_cfst.tolist())})\n') 
    f.write("\n")
    
    f.write('gvm_vm_matrix_cfst =')
    f.write(f'   np.array({repr(gvm_vm_matrix_cfst.tolist())})\n') 
    f.write("\n")

print(f"Estimation results with estimation for each trial saved to {output_filename}")



