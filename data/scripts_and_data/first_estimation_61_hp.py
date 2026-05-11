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



####### estimation under hp ######


# first estimation

hawkes_hp = ExponentialHawkesGVM(model='hp')

start = time.time()
hawkes_hp.fit(resampled_trials)

hp_estimations = hawkes_hp.multiple_estimations
hp_average_estimation = hawkes_hp.average_estimation

hawkes_hp.test_sparsity_alpha(asymptotic=False)
interaction_matrix_cfe = hawkes_hp.alpha_zero_coefficients

hawkes_hp.alpha_zero_coefficients = np.zeros((hawkes_hp.d, hawkes_hp.d), dtype=bool)
hawkes_hp.test_sparsity_alpha(asymptotic=True)
interaction_matrix_cfst = hawkes_hp.alpha_zero_coefficients

end = time.time()



# print results

print('UNDER HP:')
print('Estimation time estimation for each trial:', end-start)


output_filename = "first_estimation_61_hp.txt"
with open(output_filename, "w") as f:
    f.write("hp_estimations = [\n")
    for mu, alpha, beta in hp_estimations:
        f.write(f"    (np.array({repr(mu.tolist())}),\n")
        f.write(f"     np.array({repr(alpha.tolist())}),\n")
        f.write(f"     np.array({repr(beta.tolist())})),\n")
    f.write("]\n\n")

    f.write("hp_average_estimation = (\n")
    mu, alpha, beta = hp_average_estimation
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




