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



####### estimation under GVM ######


# first estimation

hawkes = ExponentialHawkesGVM(model='gvm')

start = time.time()
hawkes.fit(resampled_trials)

gvm_estimations = hawkes.multiple_estimations
gvm_average_estimation = hawkes.average_estimation

hawkes.test_absence_interactions(asymptotic=False)
interaction_matrix_cfe = hawkes.alpha_zero_coefficients

hawkes.alpha_zero_coefficients = np.zeros((hawkes.d, hawkes.d), dtype=bool)
hawkes.alpha_tilde_zero_coefficients = np.zeros((hawkes.d, hawkes.d), dtype=bool)

hawkes.test_absence_interactions(asymptotic=True)
interaction_matrix_cfst = hawkes.alpha_zero_coefficients

end = time.time()



# print results

print('UNDER GVM:')
print('Estimation time estimation for each trial:', end-start)


output_filename = "first_estimation_61_gvm.txt"
with open(output_filename, "w") as f:
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

    f.write('interaction_matrix_cfe =')
    f.write(f'   np.array({repr(interaction_matrix_cfe.tolist())})\n') 
    f.write("\n")

    f.write('interaction_matrix_cfst =')
    f.write(f'   np.array({repr(interaction_matrix_cfst.tolist())})\n') 
    f.write("\n")

print(f"Estimation results with estimation for each trial saved to {output_filename}")




