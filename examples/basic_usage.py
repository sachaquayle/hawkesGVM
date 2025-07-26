import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import kstest, expon
from numba import njit

import ClassGVM
from ClassGVM import logit, der_logit, negloglikelihood_with_grad_single_jit
from ClassGVM import ExponentialHawkesGVM

alpha = np.array([[0.2, 0.0],[-0.6,1.2]])
beta = np.array([3.5,2.0])
mu = np.array([0.7,1.0])

# Simulate times 

size = 5000
nb_realisations = 25

hawkes_simulate = ExponentialHawkesGVM(model='gvm', mu=mu, alpha=alpha, beta=beta, alpha_tilde=alpha)
times = hawkes_simulate.simulate(size=size, nb_realisations=nb_realisations)

# Estimation

hawkes_est = ExponentialHawkesGVM()

## Step 1 

hawkes_est.fit(times)

print(' After Step 1:', '\n\n',
      'mu estimation:', hawkes_est.average_estimation[0], '\n\n',
     'alpha estimation:', hawkes_est.average_estimation[1], '\n\n',
     'beta estimation:', hawkes_est.average_estimation[2], '\n\n',
     'alpha_tilde estimation:', hawkes_est.average_estimation[3], '\n\n')

## Step 2 

hawkes_est.test_absence_interactions()

## Step 3 

hawkes_est.fit(times)

print(' After Step 3:', '\n\n',
      'mu estimation:', hawkes_est.average_estimation[0], '\n\n',
     'alpha estimation:', hawkes_est.average_estimation[1], '\n\n',
     'beta estimation:', hawkes_est.average_estimation[2], '\n\n',
     'alpha_tilde estimation:', hawkes_est.average_estimation[3], '\n\n')

## Step 4 

hawkes_est.tests_hp_vm()

## Step 5 

hawkes_est.fit(times)

print(' After Step 5:', '\n\n',
      'mu estimation:', hawkes_est.average_estimation[0], '\n\n',
     'alpha estimation:', hawkes_est.average_estimation[1], '\n\n',
     'beta estimation:', hawkes_est.average_estimation[2], '\n\n',
     'alpha_tilde estimation:', hawkes_est.average_estimation[3], '\n\n')

print('Average p-value with final estimation:', np.mean(hawkes_est.pvalues()))
