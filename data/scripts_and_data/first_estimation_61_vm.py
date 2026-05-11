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


####### MIGUEL CODE #######

with open("data_61.txt", "r") as file:
    file_content = file.read()
exec(file_content)



####### estimation under vm ######


# first estimation

hawkes_vm = ExponentialHawkesGVM(model='vm')

start = time.time()
hawkes_vm.fit(resampled_trials)

vm_estimations = hawkes_vm.multiple_estimations
vm_average_estimation = hawkes_vm.average_estimation

hawkes_vm.test_sparsity_alpha(asymptotic=False)
interaction_matrix_cfe = hawkes_vm.alpha_zero_coefficients

hawkes_vm.alpha_zero_coefficients = np.zeros((hawkes_vm.d, hawkes_vm.d), dtype=bool)
hawkes_vm.test_sparsity_alpha(asymptotic=True)
interaction_matrix_cfst = hawkes_vm.alpha_zero_coefficients

end = time.time()



# print results

print('UNDER vm:')
print('Estimation time estimation for each trial:', end-start)


output_filename = "first_estimation_61_vm.txt"
with open(output_filename, "w") as f:
    f.write("vm_estimations = [\n")
    for mu, alpha, beta in vm_estimations:
        f.write(f"    (np.array({repr(mu.tolist())}),\n")
        f.write(f"     np.array({repr(alpha.tolist())}),\n")
        f.write(f"     np.array({repr(beta.tolist())})),\n")
    f.write("]\n\n")

    f.write("vm_average_estimation = (\n")
    mu, alpha, beta = vm_average_estimation
    f.write(f"    np.array({repr(mu.tolist())}),\n")
    f.write(f"    np.array({repr(alpha.tolist())}),\n")
    f.write(f"    np.array({repr(beta.tolist())})\n")
    f.write(")\n\n")

    f.write('interaction_matrix_cfe =')
    f.write(f'   np.array({repr(interaction_matrix_cfe.tolist())})\n') 
    f.write("\n")

    f.write('interaction_matrix_cfst =')
    f.write(f'   np.array({repr(interaction_matrix_cfst.tolist())})\n') 
    f.write("\n")

print(f"Estimation results with estimation for each trial saved to {output_filename}")




