import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.stats import kstest, expon, cramervonmises
import time
from numba import njit
import statsmodels

# Auxiliary softplus functions

@njit
def logit(x, t):
    # res = np.log(1+np.exp(t*x))/t
    # No overflow version
    res = 0.0
    mask = x > 0
    if mask:
        res = x + np.log(1 + np.exp(-t*x)) / t
    else:
        res = np.log(1 + np.exp(t*x)) / t
    return res + 1e-16

@njit
def der_logit(x, t):
    # res = 1 / (1+np.exp(-t*x))
    # No overflow version
    mask = x < 0
    res=0.0
    if mask:
        res = np.exp(t*x) / (1+np.exp(t*x))
    else : 
        res = 1 / (1+np.exp(-t*x))
    return res

# Log likelihood function for one realisation using njit
    
@njit
def negloglikelihood_with_grad_single_jit(parameters, realisation, model, d, mask_alpha, mask_alpha_tilde, mask_equal, t):
    """
    Compute the log-likelihood and its gradient for a single realisation of a Hawkes process.
    
    Parameters
    ----------
    parameters : array
        Model parameters. 
        - For 'hp' and 'vm': 
          [mu_1, ..., mu_d, alpha_11, alpha_12, ..., alpha_dd, beta_1, ..., beta_d].
        - For 'gvm': 
          [mu_1, ..., mu_d, alpha_11, alpha_12, ..., alpha_dd, beta_1, ..., beta_d, alpha_tilde_11, alpha_tilde_12, ..., alpha_tilde_dd].
    realisation : array-like of tuple (float, int)
        Sequence of event times and their corresponding dimensions.
    model : int
        Specifies the model to be used: 0 for 'hp', 1 for 'vm', 2 for 'gvm'.
    d : int
        The number of dimensions of the point process.
    mask_alpha : array-like of bool, size (d,d)
        Mask indicating whether each alpha parameter is fixed to zero (True) or not (False).
    mask_alpha_tilde : array-like of bool, size (d,d)
        Mask indicating whether each alpha_tilde parameter is fixed to zero (True) or not (False).
    mask_equal : array-like of bool, size (d,d)
        Mask indicating whether each alpha parameter is equal to its corresponding alpha_tilde (True) or not (False).
    t : float
        Softplus function parameter.
        
    Returns
    -------
    log_likelihood : float
        The computed log-likelihood for the given realisation.
    gradient : array-like
        The gradient of the log-likelihood with respect to the model parameters.
        The shape of this array matches the input 'parameters' array.
    
    Notes
    -----
    - The function assumes the input 'parameters' array is correctly formatted according to the chosen model, dimension, and masks.
    - The structure of 'realisation' should match the dimensions ('d') and represent event times properly.
    """

    nb_alpha_nonzero = (~mask_alpha).sum()
    nb_alpha_tilde_nonzero = (np.logical_and(~mask_alpha_tilde, ~mask_equal)).sum()
    
    N = len(realisation)
    restart_times = np.zeros(d)
    mu = parameters[:d]

    first_jump = realisation[0,0]
    first_index = int(realisation[0,1])

    ############# Gradient
    grad_mu = first_jump*np.ones(d)
    grad_alpha = np.zeros((d, d))
    grad_beta = np.zeros(d)
    grad_alpha_tilde = np.zeros((d,d))

    grad_mu[first_index] -= der_logit(mu[first_index],t) / logit(mu[first_index],t)  # Grad from log intensity

    log_likelihood = 0.0
    if model in [0, 1]:  # Models 'hp' and 'vm'
        beta = parameters[-d:]
        beta_1 = 1/beta
        alpha_masked = parameters[d:d+nb_alpha_nonzero]

        alpha = np.zeros((d,d)) # Reconstruct alpha
        k = 0  # Index for alpha_masked
        for i in range(d):
            for j in range(d):
                if not mask_alpha[i, j]:  # Check the mask
                    alpha[i, j] = alpha_masked[k]
                    k += 1
        
        dA = np.zeros((d, d))
        dB = np.zeros(d)

        log_likelihood = np.sum(mu) * realisation[0, 0] - np.log( logit(mu[int(realisation[0, 1])], t))
        intensities = np.zeros(d)

        for i in range(1, N):
            jump_time = realisation[i, 0]
            last_jump_time = realisation[i - 1, 0]

            current_index = int(realisation[i, 1])
            last_index = int(realisation[i - 1, 1])

            decay_factor = np.exp(-beta * (jump_time - last_jump_time))

            inside_log = np.where(mu + intensities + alpha[:, last_index] < 0,
                                    (-intensities - alpha[:, last_index]) / mu,
                                    1)

            vector_aux = np.log(inside_log)
            decay_diff = np.zeros(d)
            j_i = np.zeros(d)

            for j in range(d):
                restart_times[j] = np.minimum(jump_time, last_jump_time + vector_aux[j]/beta[j])
                decay_diff[j] = np.exp(-beta[j] * (restart_times[j] - last_jump_time)) -np.exp(-beta[j] * (jump_time - last_jump_time))
                j_i[j] = mu[j] * (jump_time-restart_times[j]) + ((intensities[j] + alpha[j, last_index]) / beta[j] )* (decay_diff[j])

            ############ Gradient with compensator term
            for j in range(d):
                grad_mu[j] += jump_time - restart_times[j] 
                for k in range(d):
                    grad_alpha[j,k] += beta_1[j] * dA[j,k] * decay_diff[j]
                grad_alpha[j,last_index] += beta_1[j] * (decay_diff[j])
                grad_beta[j] += beta_1[j] * (dB[j] - beta_1[j] * (intensities[j]+alpha[j,last_index])) * decay_diff[j] + beta_1[j] * (intensities[j]+alpha[j,last_index]) * ( (jump_time - last_jump_time) * decay_factor[j] - (restart_times[j]-last_jump_time)* np.exp(-beta[j] * (restart_times[j] - last_jump_time)))
            
            ############ Update log-likelihood
            new_intensities = decay_factor * (intensities + alpha[:, last_index])
            
            int_before_jump = mu[current_index] + new_intensities[current_index]
            
            psi = logit(int_before_jump, t)
            int_term = der_logit(int_before_jump, t) / psi

            log_likelihood += np.sum(j_i) - np.log(psi)

            ######## Gradient with log intensity term (difference with classic negloglik)
            grad_mu[current_index] -= int_term
            for k in range(d):
                grad_alpha[current_index,k] -= dA[current_index,k] * decay_factor[current_index] * int_term
            grad_alpha[current_index,last_index] -= decay_factor[current_index] * int_term
            grad_beta[current_index] -= ((dB[current_index] - (jump_time-last_jump_time) * (intensities[current_index] + alpha[current_index,last_index])) * decay_factor[current_index]) * int_term

            for j in range(d):
                for k in range(d):
                    dA[j,k] *= decay_factor[j]
                dA[j,last_index] += decay_factor[j]
                dB[j] = (dB[j] - (jump_time-last_jump_time) * (intensities[j]+alpha[j,last_index]) ) * decay_factor[j]

            intensities = new_intensities
            
            if model == 1:
                intensities[current_index] = 0  # For 'vm' model, memory reset  
                dA[current_index,:] = np.zeros(d)
                dB[current_index] = 0
            ########
        idx = 0
        new_grad_alpha = np.zeros(nb_alpha_nonzero)
        for i in range(d):
            for j in range(d):
                if not mask_alpha[i, j]:  # If the mask is False
                    new_grad_alpha[idx] = grad_alpha[i, j] 
                    idx += 1
        
        grad_comp = np.concatenate((grad_mu,new_grad_alpha,grad_beta))

        return log_likelihood, grad_comp

    else:  # Model 'gvm'
        alpha_masked = parameters[d:d+nb_alpha_nonzero]
        beta = parameters[d+nb_alpha_nonzero:2*d+nb_alpha_nonzero]
        beta_1 = 1/beta
        alpha_tilde_masked = parameters[2*d+nb_alpha_nonzero:]
        
        alpha = np.zeros((d,d)) # Reconstruct alpha
        k = 0  # Index for alpha_masked
        for i in range(d):
            for j in range(d):
                if not mask_alpha[i, j]:  # Check the mask
                    alpha[i, j] = alpha_masked[k]
                    k += 1
                    
        alpha_tilde = np.zeros((d,d)) # Reconstruct alpha_tilde
        k = 0  # Index for alpha_tilde_masked
        for i in range(d):
            for j in range(d):
                if not mask_alpha_tilde[i, j] and not mask_equal[i,j]:  # Check the mask
                    alpha_tilde[i, j] = alpha_tilde_masked[k]
                    k += 1

        for i in range(d):
            for j in range(d):
                if mask_equal[i, j]:
                    alpha_tilde[i, j] = alpha[i, j]

        log_likelihood = np.sum(mu) * realisation[0, 0] - np.log(logit(mu[int(realisation[0, 1])], t))
        intensities = np.zeros(3 * d)

        dA_eta = np.zeros((d, d))
        dA_aux = np.zeros((d, d))
        dA_eta_tilde = np.zeros((d, d))

        dB_eta = np.zeros(d)
        dB_aux = np.zeros(d)
        dB_eta_tilde = np.zeros(d)

        dAt_eta = np.zeros((d,d))
        dAt_aux = np.zeros((d,d))
        dAt_eta_tilde = np.zeros((d,d))

        for i in range(1, N):
            jump_time = realisation[i, 0]
            last_jump_time = realisation[i - 1, 0]

            current_index = int(realisation[i, 1])
            last_index = int(realisation[i - 1, 1])

            decay_factor = np.exp(-beta * (jump_time - last_jump_time))

            inside_log = np.where(mu + intensities[:d] + intensities[2*d:] + alpha[:, last_index] < 0,
                                    (-intensities[:d] - intensities[2*d:] - alpha[:, last_index]) / mu,
                                    1)

            vector_aux = np.log(inside_log)
            decay_diff = np.zeros(d)
            j_i = np.zeros(d)
            
            for j in range(d):
                restart_times[j]= np.minimum(jump_time, last_jump_time + vector_aux[j]/beta[j])
                decay_diff[j] = np.exp(-beta[j] * (restart_times[j] - last_jump_time)) - np.exp(-beta[j] * (jump_time - last_jump_time))
                j_i[j] =  mu[j] * (jump_time - restart_times[j]) + beta_1[j]*(intensities[j] + intensities[2*d+j] + alpha[j, last_index]) *(decay_diff[j]) 

            ############ Gradient comp term
            for j in range(d):
                grad_mu[j] += jump_time - restart_times[j] 
                for k in range(d):
                    grad_alpha[j,k] += beta_1[j] * (dA_eta[j,k]+dA_eta_tilde[j,k]) * decay_diff[j]
                    grad_alpha_tilde[j,k] += beta_1[j] * (dAt_eta[j,k]+dAt_eta_tilde[j,k]) * decay_diff[j]
                grad_alpha[j,last_index] += beta_1[j] * decay_diff[j]

                if mask_equal[j,last_index]:
                    avg_gradient = (grad_alpha[j,last_index] + grad_alpha_tilde[j,last_index]) / 2
                    grad_alpha[j,last_index] = avg_gradient
                    grad_alpha_tilde[j,last_index] = avg_gradient
                
                grad_beta[j] += beta_1[j] * (dB_eta[j]+dB_eta_tilde[j] - beta_1[j] * (intensities[j]+intensities[2*d+j]+alpha[j,last_index])) * decay_diff[j] + beta_1[j] * (intensities[j]+intensities[2*d+j]+alpha[j,last_index]) * ( (jump_time - last_jump_time) * decay_factor[j] - (restart_times[j]-last_jump_time)* np.exp(-beta[j] * (restart_times[j] - last_jump_time)))
            
            ############ Update log-likelihood
            new_eta = decay_factor * (intensities[:d] + alpha[:, last_index])
            new_eta_aux = decay_factor * (intensities[d:2*d] + alpha_tilde[:, last_index])
            new_eta_tilde = decay_factor * (intensities[2*d:])
            
            int_before_jump = mu[current_index] + new_eta[current_index] + new_eta_tilde[current_index]
                
            psi = logit(int_before_jump, t)
            int_term = der_logit(int_before_jump, t) / psi
            log_likelihood += np.sum(j_i) - np.log(psi)

            ######## Gradient log intensity term (difference with classic negloglik)
            grad_mu[current_index] -= 1 * int_term
            for k in range(d):
                grad_alpha[current_index,k] -= ((dA_eta[current_index,k]+dA_eta_tilde[current_index,k]) * decay_factor[current_index] )* int_term
                grad_alpha_tilde[current_index,k] -= ((dAt_eta[current_index,k]+dAt_eta_tilde[current_index,k]) * decay_factor[current_index]) * int_term    
            grad_alpha[current_index,last_index] -= decay_factor[current_index] * int_term
            
            if mask_equal[current_index,last_index]:
                avg_gradient = (grad_alpha[current_index,last_index] + grad_alpha_tilde[current_index,last_index]) / 2
                grad_alpha[current_index,last_index] = avg_gradient
                grad_alpha_tilde[current_index,last_index] = avg_gradient
            
            grad_beta[current_index] -= ((dB_eta[current_index]+dB_eta_tilde[current_index] - (jump_time-last_jump_time) * (intensities[current_index] + intensities[2*d+current_index] + alpha[current_index,last_index])) * decay_factor[current_index]) * int_term
            for j in range(d):
                if j != current_index:
                    for k in range(d):
                        dA_eta[j,k] *= decay_factor[j]
                        dA_aux[j,k] *= decay_factor[j]
                        dA_eta_tilde[j,k] *= decay_factor[j]

                        dAt_eta[j,k] *= decay_factor[j]
                        dAt_aux[j,k] *= decay_factor[j]
                        dAt_eta_tilde[j,k] *= decay_factor[j]

                    dA_eta[j,last_index] += decay_factor[j]
                    dAt_aux[j,last_index] += decay_factor[j]

                    if mask_equal[j,last_index]:
                        avg_value = (dA_eta[j,last_index] + dAt_eta[j,last_index]) / 2
                        dA_eta[j,last_index] = avg_value
                        dAt_eta[j,last_index] = avg_value

                        avg_value = (dA_aux[j,last_index] + dAt_aux[j,last_index]) / 2
                        dA_aux[j,last_index] = avg_value
                        dAt_aux[j,last_index] = avg_value
                        
                    dB_eta[j] = (dB_eta[j] - (jump_time-last_jump_time)*(intensities[j]+alpha[j,last_index]))*decay_factor[j]
                    dB_aux[j] = (dB_aux[j] - (jump_time-last_jump_time)*(intensities[d+j]+alpha_tilde[j,last_index]))*decay_factor[j]
                    dB_eta_tilde[j] = (dB_eta_tilde[j] - (jump_time-last_jump_time) * (intensities[2*d+j]) ) * decay_factor[j]
                else :
                    for k in range(d):
                        dA_eta_tilde[current_index,k] = (dA_eta_tilde[current_index,k] + dA_aux[current_index,k])*decay_factor[j]
                        dAt_eta_tilde[current_index,k] = (dAt_eta_tilde[current_index,k] + dAt_aux[current_index,k])*decay_factor[j]

                    dAt_eta_tilde[current_index,last_index] += decay_factor[current_index]

                    if mask_equal[current_index,last_index]:
                        avg_value = (dA_eta_tilde[current_index,last_index] + dAt_eta_tilde[current_index,last_index]) / 2
                        dA_eta_tilde[current_index,last_index] = avg_value
                        dAt_eta_tilde[current_index,last_index] = avg_value
                    
                    dB_eta_tilde[current_index] = (dB_eta_tilde[current_index] - (jump_time-last_jump_time) * (intensities[2*d+current_index]) ) * decay_factor[current_index] + (dB_aux[current_index] - (jump_time-last_jump_time) * (intensities[d+current_index]+alpha_tilde[current_index,last_index] ) )*decay_factor[current_index]

                    dA_eta[current_index,:] = np.zeros(d)
                    dA_aux[current_index,:] = np.zeros(d)

                    dB_eta[current_index] = 0
                    dB_aux[current_index] = 0

                    dAt_eta[current_index,:] = np.zeros(d)
                    dAt_aux[current_index,:] = np.zeros(d)
            ########
            intensities[:d] = new_eta
            intensities[2*d:] = new_eta_tilde
            
            intensities[current_index] = 0
            intensities[2*d+current_index] += decay_factor[current_index]*(intensities[d+current_index]+alpha_tilde[current_index][last_index])

            intensities[d:2*d] = new_eta_aux
            intensities[d+current_index] = 0
        
        idx = 0
        new_grad_alpha_tilde = np.zeros(nb_alpha_tilde_nonzero) # Reconstruct grad alpha
        for i in range(d):
            for j in range(d):
                if not mask_alpha_tilde[i, j] and not mask_equal[i,j] :  # If the mask is False
                    new_grad_alpha_tilde[idx] = grad_alpha_tilde[i, j] 
                    idx += 1
        idx = 0
        new_grad_alpha = np.zeros(nb_alpha_nonzero) # Reconstruct grad alpha
        for i in range(d):
            for j in range(d):
                if not mask_alpha[i, j]:  # If the mask is False
                    new_grad_alpha[idx] = grad_alpha[i, j] 
                    if mask_equal[i,j]:
                        new_grad_alpha[idx] *= 2
                    idx += 1
        
        grad_comp = np.concatenate((grad_mu,new_grad_alpha,grad_beta, new_grad_alpha_tilde))

        return log_likelihood, grad_comp


@njit
def negloglikelihood_with_grad_single_jit_pen(parameters, realisation, model, d, t, lambda_pen, mu_pen):
    """
    Compute the penalised log-likelihood and its gradient for a single realisation of a Hawkes process.
    
    Parameters
    ----------
    parameters : array
        Model parameters. 
        - For 'hp' and 'vm': 
          [mu_1, ..., mu_d, alpha_11, alpha_12, ..., alpha_dd, beta_1, ..., beta_d].
        - For 'gvm': 
          [mu_1, ..., mu_d, alpha_11, alpha_12, ..., alpha_dd, beta_1, ..., beta_d, alpha_tilde_11, alpha_tilde_12, ..., alpha_tilde_dd].
    realisation : array-like of tuple (float, int)
        Sequence of event times and their corresponding dimensions.
    model : int
        Specifies the model to be used: 0 for 'hp', 1 for 'vm', 2 for 'gvm'.
    d : int
        The number of dimensions of the point process.
    t : float
        Softplus function parameter.
    lambda_pen : float
        Penalisation parameter for alpha_tilde (vm interactions).
    mu_pen : float
        Penalisation parameter for alpha and alpha_tilde (interaction sparsity).
        
    Returns
    -------
    log_likelihood : float
        The computed log-likelihood for the given realisation.
    gradient : array-like
        The gradient of the log-likelihood with respect to the model parameters.
        The shape of this array matches the input 'parameters' array.
    
    Notes
    -----
    - The function assumes the input 'parameters' array is correctly formatted according to the chosen model, dimension, and masks.
    - The structure of 'realisation' should match the dimensions ('d') and represent event times properly.
    """
    
    mask_alpha = np.zeros((d, d), dtype=np.bool_)
    mask_alpha_tilde = np.zeros((d, d), dtype=np.bool_)
    mask_equal = np.zeros((d, d), dtype=np.bool_)
    
    nll_value, grad = negloglikelihood_with_grad_single_jit(parameters, realisation, model, d, mask_alpha, mask_alpha_tilde, mask_equal, t)

    nb_alpha = d * d
    alpha = parameters[d : d + nb_alpha].reshape((d, d))
    eps = 1e-8

    if model==2:
        alpha_tilde = parameters[2*d+d*d:].reshape((d, d))
    
        # penalisation lambda
        
        row_norms_tilde = np.sqrt(np.sum(alpha_tilde**2, axis=1) + eps)
        penalty_lambda_value = lambda_pen * np.sum(row_norms_tilde)
        grad_alpha_tilde_lambda = lambda_pen * (alpha_tilde / row_norms_tilde[:, np.newaxis])
    
        # penalisation mu
        
        joint_norms = np.sqrt(alpha**2 + alpha_tilde**2 + eps)
        penalty_mu_value = mu_pen * np.sum(joint_norms)
        grad_alpha_pen = mu_pen * alpha / joint_norms
        grad_alpha_tilde_mu = mu_pen * alpha_tilde / joint_norms
    
        # add to totals
        
        penalty_value = penalty_lambda_value + penalty_mu_value
        
        grad_alpha_tilde_pen = grad_alpha_tilde_lambda + grad_alpha_tilde_mu
    
        grad_pen = np.zeros_like(parameters)
        grad_pen[d : d + d*d] = grad_alpha_pen.ravel()
        grad_pen[2*d+d*d :] = grad_alpha_tilde_pen.ravel()
    
        nll_value_pen = nll_value + penalty_value
        grad_pen_total = grad + grad_pen
    else:
        # penalisation mu
        
        joint_norms = np.sqrt(alpha**2 + eps)
        penalty_mu_value = mu_pen * np.sum(joint_norms)
        grad_alpha_pen = mu_pen * alpha / joint_norms
        
        # add to totals
        
        penalty_value = penalty_mu_value
    
        grad_pen = np.zeros_like(parameters)
        grad_pen[d : d + d*d] = grad_alpha_pen.ravel()
    
        nll_value_pen = nll_value + penalty_value
        grad_pen_total = grad + grad_pen
        
    return nll_value_pen, grad_pen_total


class ExponentialHawkesGVM():
    
    def __init__(self, model='gvm', mu=None, alpha=None, beta=None, alpha_tilde=None):
        """
        Parameters
        ----------
        model : str, optional
            Type of model ('hp', 'vm' or 'gvm'). 
            Default is 'gvm'.
        mu : array, optional
            Baseline intensity vector. Must match shapes of alpha, beta and alpha_tilde.
            Default is None.
        alpha : array, optional
            Interaction factors matrix for recent past. Must be a square matrix and match shapes of mu, beta and alpha_tilde.
            Default is None.
        beta : array, optional
            Decay factor vector. Must match shapes of mu, alpha and alpha_tilde.
            Default is None.
        alpha_tilde : array, optional
            Interaction factors matrix for ancient past. Must be a square matrix and match shapes of mu, alpha and beta.
            Default is alpha.

        Attributes
        ----------
        d : int
            Number of dimensions.
            Default is None.
        times : list of list of tuple (float, int)
            Lists of realisations of the Hawkes process. Each realisation is a sequence of event times and their corresponding dimensions.
            Default is None.
        nb_realisations : int
            Number of realisations of the Hawkes process.
            Default is 0.
        multiple_estimations : list of tuple of array
            Multiple parameter estimations, one for each realisation, i.e., [theta_hat(1), ..., theta_hat(len(times))].
            For 'hp' or 'vm': theta_hat = (mu_hat, alpha_hat, beta_hat). 
            For 'gvm': theta_hat = (mu_hat, alpha_hat, beta_hat, alpha_tilde_hat).
            Default is None.
        messages : list of str
            Messages from the minimisation process for each realisation.
            Default is None.
        results :
            Results from the minimisation process for each realisation.
            Default is None.
        average_estimation : tuple of array
            Single parameter estimation by averaging multiple estimations.
            For 'hp' or 'vm': average_theta_hat = (average_mu_hat, average_alpha_hat, average_beta_hat). 
            For 'gvm': average_theta_hat = (average_mu_hat, average_alpha_hat, average_beta_hat, average_alpha_tilde_hat).
            Default is None.
        estimation : tuple of array
            Single parameter estimation over all realisations.
            For 'hp' or 'vm': theta_hat = (mu_hat, alpha_hat, beta_hat). 
            For 'gvm': theta_hat = (mu_hat, alpha_hat, beta_hat, alpha_tilde_hat).
            Default is None.
        message : str
            Message from the minimisation process over all realisations.
        result :
            Result from the minimisation process over all realisations.
        model_int : int
            Encoded model type for efficient computation using njit: O for 'hp', 1 for 'vm', 2 for 'gvm'.
            Default is 2.
        nb_params : int
            Total number of parameters in the model. For 'hp' or 'vm': d*(d+2). For 'gvm': 2*d*(d+1).
            Default is None.
        alpha_zero_coefficients : array-like of bool, size (d,d) 
            Marks alpha parameters as null (True where alpha and alpha_tilde are zero).
        alpha_tilde_zero_coefficients : array-like of bool, size (d,d) 
            Marks alpha_tilde parameters as null (True where alpha and alpha_tilde are zero).
        equal_coefficients : array-like of bool, size (d,d) 
            Marks equal coefficients (True where alpha and alpha_tilde are equal).
            Default is None.
        """
        self.model = model  
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.alpha_tilde = alpha_tilde
        
        self.d = None
        self.times = None
        self.nb_realisations = 0
        self.multiple_estimations = None 
        self.messages = None
        self.results = None
        self.average_estimation = None
        self.estimation = None
        self.message = None
        self.result = None
        if self.model == 'hp':
            self.model_int = 0
        elif self.model == 'vm':
            self.model_int = 1
        else :
            self.model_int = 2
        self.nb_params = None
        self.alpha_zero_coefficients = None
        self.alpha_tilde_zero_coefficients = None
        self.equal_coefficients = None
        
        
    def simulate(self, size=None, horizon=None, nb_realisations=1):
        """
        Simulate multi-dimensionnal Hawkes processes under Model (GVM).
        
        Parameters
        ----------
        size : int, optional
            Maximal number of jumps.
            Default is None.
        horizon : float, optional
            Maximal time horizon.
            Default is None.
        nb_realisations : int, optional
            Number of realisations.
            Default is 1.

        Returns
        ----------
        times : list of list of tuple (float,int)
            Lists of realisations of the Hawkes process. Each realisation is a sequence of event times and their corresponding dimensions.
       
        Raises
        ----------
        ValueError  
            If any of the following conditions are met:  
            - There is an inconsistency in the maximum number of jumps or the time horizon.
            - One or more required parameters are missing.  
            - The spectral radius(es) of the interaction matrix(es) exceed(s) 1.  
        """
        # To launch simulation, either size or horizon must be other than None, and both can not be non None

        if size is None and horizon is None:
            raise ValueError("Either number of jumps (size) or maximal time (horizon) must be given.")
        if size is not None and horizon is not None:
            raise ValueError("Both number of jumps (size) and maximal time (horizon) can not be given.")
        if self.alpha is None or self.beta is None or self.mu is None: # To launch simulation, self.alpha, self.beta and self.mu must be other than None
            raise ValueError("One of the parameters is missing.")
        if self.alpha_tilde is None and self.model=='gvm':
            raise ValueError("One of the parameters is missing.")
        if size is None and horizon is not None:
            size = np.inf
        if size is not None and horizon is None:
            horizon = np.inf
            
        self.d = len(self.mu)  # We can now define self.d as no parameters are missing

        # Check that what we want to simulate is in fact a point process (i.e. does not explode on any finite interval). For this we check that the spectral radiuses are < 1
        
        if self.model == 'vm':
            self.alpha_tilde = np.zeros((self.d,self.d))
        elif self.model == 'hp':
            self.alpha_tilde = self.alpha

        spectral_radius = np.max(np.abs(np.linalg.eigvals(np.abs(self.alpha) / np.tile(self.beta, (self.d, 1)).T))) 
        spectral_radius_tilde = np.max(np.abs(np.linalg.eigvals(np.abs(self.alpha_tilde) / np.tile(self.beta, (self.d, 1)).T))) 

        if max(spectral_radius,spectral_radius_tilde) >= 1:
            raise ValueError("Spectral radius(es) is (are) larger than 1, which makes the process unstable.")

        # Launch simulation using thinning algorithm

        times = [ [] for _ in range(nb_realisations)]

        for k in range(nb_realisations): # Simulate nb_realisations independent Hawkes processes
            intensities = np.zeros(3 * self.d)  # Contains values of eta^1,...,eta^d,eta_tilde_aux^1,...eta_tilde_aux^d, eta_tilde^1,...eta_tilde^d, updated at each jump time
            upper_intensity = np.sum(self.mu)
            initialized = False  # Auxiliary variable to check if we have first jump time. If not, candidate intensities and upper bounds are slightly different
            current_time = 0

            while len(times[k]) < size and current_time < horizon : # Simulation is done until the maximal number of jumps (size) or maximal time (horizon) is attained
                current_time += np.random.exponential(1 / upper_intensity)

                if not initialized:
                    candidate_eta = np.zeros(self.d)
                    candidate_eta_tilde = np.zeros(self.d)
                else:
                    last_time = times[k][-1][0]
                    last_index = times[k][-1][1]
                    # Compute candidate intensities using recursion formulas
                    decay_factor = np.exp(-self.beta * (current_time-last_time))
                    candidate_eta = decay_factor * (intensities[:self.d] + self.alpha[:, last_index])
                    candidate_eta_tilde = decay_factor * intensities[2*self.d:]

                candidate_intensities = self.mu + candidate_eta + candidate_eta_tilde
                lambda_sum = max(0,np.sum(candidate_intensities))
                U = stats.uniform.rvs()

                if U <= lambda_sum / upper_intensity and current_time < horizon : # Reject or accept the candidate and check it does not go over horizon
                    next_index = np.searchsorted(np.cumsum(candidate_intensities), U * upper_intensity, side='left') # Find index of accepted candidate time

                    if not initialized:
                        initialized = True
                        upper_intensity = np.sum(self.mu) + max(0,np.sum(self.alpha[:, next_index]))
                    else:
                        intensities[:self.d] = candidate_eta
                        intensities[next_index] = 0   # Memory reset

                        intensities[2*self.d:] = candidate_eta_tilde
                        intensities[2*self.d + next_index] += decay_factor[next_index] * (intensities[self.d + next_index] + self.alpha_tilde[next_index, last_index])

                        intensities[self.d:2*self.d] = decay_factor * (intensities[self.d:2*self.d] + self.alpha_tilde[:, last_index]) # Update eta_tilde_aux using recursion formula
                        intensities[self.d + next_index] = 0

                        upper_intensity = np.sum(self.mu) + max(0, np.sum(intensities[:self.d] + intensities[2*self.d:]) + np.sum(self.alpha[:, next_index]))
                    times[k] += [(current_time,next_index)]
        return times

    # Auxiliary flattening and unflattening functions 

    def flatten_parameters(self, parameters):
        """
        Flattens the model parameters into a single array, excluding fixed values based on masks.
    
        Parameters
        ----------
        parameters : tuple of array-like
            Model parameters structured as:
            - parameters[0] (array-like): mu values.
            - parameters[1] (array-like): alpha values, with elements removed where 'self.alpha_zero_coefficients' is True.
            - parameters[2] (array-like): beta values.
            - parameters[3] (optional, array-like): alpha_tilde values, with elements removed where 'self.alpha_tilde_zero_coefficients' or 'self.equal_coefficients' is True.
    
        Returns
        -------
        flattened_parameters : array-like
            A 1D array containing the flattened parameters.
        """
        mu = parameters[0]
        alpha = parameters[1][~self.alpha_zero_coefficients]
        beta = parameters[2]
        if len(parameters) == 4:
            alpha_tilde = parameters[3][np.logical_and(~self.alpha_tilde_zero_coefficients, ~self.equal_coefficients)]
            return np.concatenate((mu,alpha, beta,alpha_tilde))
        return np.concatenate((mu,alpha, beta))

    def unflatten_parameters(self, parameters):
        """
        Reconstructs the structured model parameters from a flattened array.
    
        This function reverses the effect of 'flatten_parameters' by reconstructing the full parameter matrices, restoring zeroed-out values where necessary.
    
        Parameters
        ----------
        parameters : array-like
            A 1D array containing the flattened model parameters.
    
        Returns
        -------
        tuple
            - mu (array-like): The mu values of size (d,).
            - alpha (array-like): The alpha matrix of shape (d, d), reconstructed with masked values restored.
            - beta (array-like): The beta values of size (d,).
            - alpha_tilde (array-like, optional): The alpha_tilde matrix of shape (d, d), reconstructed if the model is 'gvm'.
    
        Notes
        -----
        - If 'self.model' is 'gvm', the function also reconstructs 'alpha_tilde', and ensures parameters marked by 'self.equal_coefficients' are set accordingly.
        - If 'self.alpha_zero_coefficients', 'self.alpha_tilde_zero_coefficients', or 'self.equal_coefficients' are 'None', they are initialised as 'False' matrices of shape (d, d).
        - The function assumes the input 'parameters' array is correctly formatted.
        """
        if self.alpha_zero_coefficients is None:
            self.alpha_zero_coefficients = np.full((self.d,self.d), False, dtype=bool)
        if self.alpha_tilde_zero_coefficients is None:
            self.alpha_tilde_zero_coefficients = np.full((self.d,self.d), False, dtype=bool)
        if self.equal_coefficients is None:
            self.equal_coefficients = np.full((self.d,self.d), False, dtype=bool)
            
        mu = parameters[:self.d]
        if self.model == 'gvm':
            nb_alpha_params = (~self.alpha_zero_coefficients).sum()
            alpha_masked = parameters[self.d : self.d + nb_alpha_params]
            beta = parameters[self.d + nb_alpha_params:2*self.d + nb_alpha_params]
            alpha_tilde_masked = parameters[2*self.d + nb_alpha_params:]
            alpha = np.zeros((self.d,self.d)) # Reconstruct alpha
            alpha[~self.alpha_zero_coefficients] = alpha_masked
            alpha_tilde = np.zeros((self.d,self.d)) # Reconstruct alpha
            alpha_tilde[np.logical_and(~self.alpha_tilde_zero_coefficients, ~self.equal_coefficients)] = alpha_tilde_masked
            for i in range(self.d):
                for j in range(self.d):
                    if self.equal_coefficients[i,j]:
                        alpha_tilde[i,j]=alpha[i,j]
            return mu,alpha,beta,alpha_tilde
        else:
            beta = parameters[-self.d:]
            alpha_masked = parameters[self.d:-self.d]
            alpha = np.zeros((self.d,self.d)) # Reconstruct alpha
            alpha[~self.alpha_zero_coefficients] = alpha_masked
            return mu,alpha,beta


    def fit(self, times, initial_guess_mu=None, initial_guess_alpha=None, initial_guess_beta=None, initial_guess_alpha_tilde=None, bounds_mu=None, bounds_alpha=None, bounds_beta=None, bounds_alpha_tilde=None, each_realisation=True, t=100, options={}):
        """
        Updates parameter estimations by MLE.

        Parameters
        ----------
        times : list of list of tuple (float, int)
            Lists of realisations of the Hawkes process. Each realisation is a sequence of event times and their corresponding dimensions.
        initial_guess_mu : 1D array, optional
            Initial guess for mu. Length must be dimension 'd'.
            Default is None.
        initial_guess_alpha : 1D array, optional
            Initial guess for alpha. Length must be consistent with dimension 'd' and masks.
            Default is None.
        initial_guess_beta : 1D array, optional
            Initial guess for beta. Length must be dimension 'd'.
            Default is None.
        initial_guess_alpha_tilde : 1D array, optional
            Initial guess for alpha_tilde. Length must be consistent with dimension 'd' and masks.
            Default is None.
        bounds_mu : list of (float,float), optional
            Bounds for parameter 'mu' for minimisation. Length must be dimension 'd'.
            Default is None.
        bounds_alpha : list of (float,float), optional
            Bounds for parameter 'alpha' for minimisation. Length must be consistent with dimension 'd' and masks.
            Default is None.
        bounds_beta : list of (float,float), optional
            Bounds for parameter 'beta' for minimisation. Length must be dimension 'd'.
            Default is None.
        bounds_alpha_tilde : list of (float,float), optional
            Bounds for parameter 'alpha_tilde' for minimisation. Length must be consistent with dimension 'd' and masks.
            Default is None.
        each_realisation : bool, optional
            If 'True', estimates parameters for each realisation individually. 
            If 'False', provides a single estimation across all realisations by minimising the sum of log-likelihoods.
            Default is True.
        t : int, optional
            Softplus function parameter.
            Default is 100.
        options : dict, optional
            Options for minimisation.
            Default is {}.
            
        Modifies
        --------
        If 'each_realisation' is True:
            - 'self.multiple_estimations': Stores the estimations for each individual realisation.
            - 'self.messages': Stores the messages associated with the optimisation process for each realisation.
            - 'self.average_estimations': Stores the averaged estimations across all realisations.
        
        If 'each_realisation' is False:
            - 'self.estimation': Stores the estimation result for the aggregated data.
            - 'self.message': Stores the summary message for the optimisation process over the aggregated data.
                    
        Raises
        ------
        ValueError
            If size of initial_guess or bounds are incompatible with the considered model.
        """
        self.times = times
        self.nb_realisations = len(self.times)
        self.d = max([m for l in self.times for t,m in l])+1  # To find number of dimensions    

        # Retrieve total number of parameters
        
        if self.model == 'gvm':
            self.nb_params = 2*self.d*(self.d + 1)
        else:
            self.nb_params = self.d*(self.d + 2)
            
        if self.alpha_zero_coefficients is None:
            self.alpha_zero_coefficients = np.full((self.d,self.d), False, dtype=bool)
            nb_alpha_params = self.d * self.d
        else :
            self.nb_params -= (self.alpha_zero_coefficients).sum()
            nb_alpha_params = (~self.alpha_zero_coefficients).sum()
            
        if self.alpha_tilde_zero_coefficients is None:
            self.alpha_tilde_zero_coefficients = np.full((self.d,self.d), False, dtype=bool)
            nb_alpha_tilde_params = self.d * self.d
        else :
            self.nb_params -= (self.alpha_tilde_zero_coefficients).sum()
            nb_alpha_tilde_params = (~self.alpha_tilde_zero_coefficients).sum()

        if self.equal_coefficients is None:
            self.equal_coefficients = np.full((self.d,self.d), False, dtype=bool)
        else :
            self.nb_params -= (np.logical_and(self.equal_coefficients, ~self.alpha_tilde_zero_coefficients)).sum()
            nb_alpha_tilde_params -= (np.logical_and(self.equal_coefficients, ~self.alpha_tilde_zero_coefficients)).sum()

        # Initialise guess and bounds
        
        if initial_guess_mu is None:
            initial_guess_mu = np.ones(self.d)
        if initial_guess_alpha is None:
            initial_guess_alpha = np.zeros(nb_alpha_params)
        if initial_guess_beta is None:
            initial_guess_beta = np.ones(self.d)
      
        if bounds_mu is None:
            bounds_mu = [(1e-15,None)] * self.d
        if bounds_alpha is None:
            bounds_alpha = [(None,None)] * nb_alpha_params 
        if bounds_beta is None:
            bounds_beta = [(1e-15,None)] * self.d

        if self.model == 'gvm':
            if initial_guess_alpha_tilde is None:
                initial_guess_alpha_tilde = np.zeros(nb_alpha_tilde_params)
            if bounds_alpha_tilde is None:
                bounds_alpha_tilde = [(None, None)] * nb_alpha_tilde_params
    
            initial_guess = np.concatenate((initial_guess_mu, initial_guess_alpha, initial_guess_beta, initial_guess_alpha_tilde))
            bounds = bounds_mu + bounds_alpha + bounds_beta + bounds_alpha_tilde

        else:
            initial_guess = np.concatenate((initial_guess_mu, initial_guess_alpha, initial_guess_beta))
            bounds = bounds_mu + bounds_alpha + bounds_beta 
        

        if each_realisation: # Multiple estimations
            self.multiple_estimations = [0 for _ in range(self.nb_realisations)]
            self.messages = ['' for _ in range(self.nb_realisations)]
            self.results = [0 for _ in range(self.nb_realisations)]
            for k in range(self.nb_realisations):
                res = minimize(self.negloglikelihood_with_grad_jit, initial_guess, args=(k, t), bounds=bounds, jac=True)
                self.multiple_estimations[k]= self.unflatten_parameters(res.x)       
                self.messages[k] = res.message
                self.results[k] = res
            self.calculate_average_estimation()

        else : # Estimation over all realisations   
            res = minimize(self.negloglikelihood_with_grad_jit, initial_guess, args=(None, t), bounds=bounds, jac=True)
            self.estimation = self.unflatten_parameters(res.x)
            self.message = res.message 
            self.result = res

            
    def fit_pen(self, times, initial_guess_mu=None, initial_guess_alpha=None, initial_guess_beta=None, initial_guess_alpha_tilde=None, bounds_mu=None, bounds_alpha=None, bounds_beta=None, bounds_alpha_tilde=None, each_realisation=True, t=100, options={}, lambda_pen=0.01, mu_pen=0.01):
        """
        Updates parameter estimations by penalised MLE.

        Parameters
        ----------
        times : list of list of tuple (float, int)
            Lists of realisations of the Hawkes process. Each realisation is a sequence of event times and their corresponding dimensions.
        initial_guess_mu : 1D array, optional
            Initial guess for mu. Length must be dimension 'd'.
            Default is None.
        initial_guess_alpha : 1D array, optional
            Initial guess for alpha. Length must be 'd'*'d'.
            Default is None.
        initial_guess_beta : 1D array, optional
            Initial guess for beta. Length must be dimension 'd'.
            Default is None.
        initial_guess_alpha_tilde : 1D array, optional
            Initial guess for alpha_tilde. Length must be 'd'*'d'.
            Default is None.
        bounds_mu : list of (float,float), optional
            Bounds for parameter 'mu' for minimisation. Length must be dimension 'd'.
            Default is None.
        bounds_alpha : list of (float,float), optional
            Bounds for parameter 'alpha' for minimisation. Length must be 'd'*'d'.
            Default is None.
        bounds_beta : list of (float,float), optional
            Bounds for parameter 'beta' for minimisation. Length must be dimension 'd'.
            Default is None.
        bounds_alpha_tilde : list of (float,float), optional
            Bounds for parameter 'alpha_tilde' for minimisation. Length must be 'd'*'d'.
            Default is None.
        each_realisation : bool, optional
            If 'True', estimates parameters for each realisation individually. 
            If 'False', provides a single estimation across all realisations by minimising the sum of log-likelihoods.
            Default is True.
        t : int, optional
            Softplus function parameter.
            Default is 100.
        options : dict, optional
            Options for minimisation.
            Default is {}.
        lambda_pen : float, optional
            Penalisation parameter for alpha_tilde (vm interactions).
            Default is 0.01.
        mu_pen : float, optional
            Penalisation parameter for alpha and alpha_tilde (interaction sparsity).
            Default is 0.01.
            
        Modifies
        --------
        If 'each_realisation' is True:
            - 'self.multiple_estimations': Stores the estimations for each individual realisation.
            - 'self.messages': Stores the messages associated with the optimisation process for each realisation.
            - 'self.average_estimations': Stores the averaged estimations across all realisations.
        
        If 'each_realisation' is False:
            - 'self.estimation': Stores the estimation result for the aggregated data.
            - 'self.message': Stores the summary message for the optimisation process over the aggregated data.
                    
        Raises
        ------
        ValueError
            If size of initial_guess or bounds are incompatible with the considered model.
        """
        self.times = times
        self.nb_realisations = len(self.times)
        self.d = max([m for l in self.times for t,m in l])+1  # To find number of dimensions    

        # Retrieve total number of parameters
        
        if self.model == 'gvm':
            self.nb_params = 2*self.d*(self.d + 1)
        else:
            self.nb_params = self.d*(self.d + 2)

        # Initialise guess and bounds

        if initial_guess_mu is None:
            initial_guess_mu = np.ones(self.d)
        if initial_guess_alpha is None:
            initial_guess_alpha = np.zeros(nb_alpha_params)
        if initial_guess_beta is None:
            initial_guess_beta = np.ones(self.d)
      
        if bounds_mu is None:
            bounds_mu = [(1e-15,None)] * self.d
        if bounds_alpha is None:
            bounds_alpha = [(None,None)] * nb_alpha_params 
        if bounds_beta is None:
            bounds_beta = [(1e-15,None)] * self.d

        if self.model == 'gvm':
            if initial_guess_alpha_tilde is None:
                initial_guess_alpha_tilde = np.zeros(nb_alpha_tilde_params)
            if bounds_alpha_tilde is None:
                bounds_alpha_tilde = [(None, None)] * nb_alpha_tilde_params
    
            initial_guess = np.concatenate((initial_guess_mu, initial_guess_alpha, initial_guess_beta, initial_guess_alpha_tilde))
            bounds = bounds_mu + bounds_alpha + bounds_beta + bounds_alpha_tilde

        else:
            initial_guess = np.concatenate((initial_guess_mu, initial_guess_alpha, initial_guess_beta))
            bounds = bounds_mu + bounds_alpha + bounds_beta 
        

        if each_realisation: # Multiple estimations
            self.multiple_estimations = [0 for _ in range(self.nb_realisations)]
            self.messages = ['' for _ in range(self.nb_realisations)]
            self.results = [0 for _ in range(self.nb_realisations)]
            for k in range(self.nb_realisations):
                res = minimize(self.negloglikelihood_with_grad_jit_pen, initial_guess, args=(k, t, lambda_pen, mu_pen), bounds=bounds, jac=True)
                self.multiple_estimations[k]= self.unflatten_parameters(res.x)       
                self.messages[k] = res.message
                self.results[k] = res
            self.calculate_average_estimation()

        else : # Estimation over all realisations   
            res = minimize(self.negloglikelihood_with_grad_jit_pen, initial_guess, args=(None, t, lambda_pen, mu_pen), bounds=bounds, jac=True)
            self.estimation = self.unflatten_parameters(res.x)
            self.message = res.message 
            self.result = res
            


    def pvalues(self, resampling=True, nb_iterations=1, independent_samples=None, av_estimation=False, distribution='expon', method='cramervonmises'):
        """
        Computes p-values for the goodness-of-fit test using resampling or provided samples, using average estimation or estimation over all realisations.

        Parameters
        ----------
        resampling : bool, optional
            If True, p-values are computed using the resampling procedure.
            If False, p-values are computed using the provided samples (shared or independent, depending on 'independent_samples').
            Default is True.
        nb_iterations : int, optional
            Number of resampling iterations to perform during the goodness-of-fit testing procedure if 'resampling' is True.
            Default is 1.
        independent_samples : list of list of tuple (float, int), optional
            Lists of independent realisations of the Hawkes process. Each realisation is a sequence of event times and their corresponding dimensions. The number of realisations must match self.nb_realisations.
            Default is None.
        av_estimation : bool, optional
            If True, p-values are computed using the averaged estimation over multiple realisations.
            If False, p-values are computed using the estimation over all realisations.
            Default is False.
        distribution : str, optional
            Str which indicates which distribution to use for goodness-of-fit (exponential or uniform). Must be 'expon' or 'uniform'.
            Default is 'expon'.
        method : str, optional
            Str which indicates which method to use for goodness-of-fit. Must be 'kstest' or 'cramervonmises'.
            Default is 'cramervonmises'.

        Returns
        ----------
        pvalues : array of float
            Array of computed p-values. Its length is equal to 'nb_iterations' if 'resampling' is True,  and 'self.nb_realisations' if 'resampling' is False.
        
        Raises
        --------
        ValueError
            If 'av_estimation' is True but 'self.average_estimation' is None, or if 'av_estimation' is False but 'self.estimation' is None.
            If 'distribution' is different than 'expon' or 'uniform'.
            If 'method' is different than 'kstest' or 'cramervonmises'.
        """
        if av_estimation and self.average_estimation is None:
            raise ValueError('Must have average estimation.')
        if not av_estimation and self.estimation is None:
            raise ValueError('Must have estimation over all realisations.')

        if resampling:
            pvalues = np.zeros(nb_iterations) # Initialise p-values
            sample_size = int(self.nb_realisations**(1/2)) # Sample size, could also be int(self.nb_realisations**(2/3))
    
            intervals_transformed = [0 for _ in range(self.nb_realisations)] # Transform times with compensator
            for k in range(self.nb_realisations):
                if av_estimation:
                    intervals_transformed[k] = self.calculate_test_values(self.average_estimation, self.times[k])
                else :
                    intervals_transformed[k] = self.calculate_test_values(self.estimation, self.times[k])

            for k in range(nb_iterations): # perform resampling procedure
                random_indexes = np.sort(np.random.choice(np.arange(self.nb_realisations), size=sample_size, replace=False)) # Draw subsample at random
            
                concatenated_times = np.array([]) # to concatenate times
            
                add_val = 0
                for index in random_indexes:
                    times = np.cumsum(intervals_transformed[index]) + add_val
                    
                    concatenated_times = np.concatenate((concatenated_times,times))
                    add_val += times[-1]
            
                max_time = concatenated_times[-1]
                theta = 0.9 * max_time
                cut_times = concatenated_times[concatenated_times <= max_time]

                if distribution=='expon':
                    test_times = cut_times[2:]- cut_times[1:-1]
                elif distribution=='uniform':
                    test_times = cut_times/max_time
                else:
                    raise ValueError("Distribution must be 'expon' or 'uniform'")

                if method=='kstest':
                    pvalues[k] = kstest(test_times, distribution).pvalue
                elif method=='cramervonmises':
                    pvalues[k] = cramervonmises(test_times, distribution).pvalue
                else:
                    raise ValueError("Method must be 'kstest' or 'cramervonmises'")
            
            return pvalues 

        else:
            pvalues = np.zeros(self.nb_realisations) # Initialise p-values
    
            if independent_samples is None:
                samples = self.times
            else :
                samples = independent_samples
    
            for k in range(self.nb_realisations): # transform times with compensator
                if av_estimation:
                    intervals_transformed = self.calculate_test_values(self.average_estimation, samples[k])
                else:
                    intervals_transformed = self.calculate_test_values(self.estimation, samples[k])

                if distribution=='expon':
                    test_times = intervals_transformed
                elif distribution=='uniform':
                    transformed_times = np.cumsum(intervals_transformed)
                    test_times = transformed_times/transformed_times[-1]
                else:
                    raise ValueError("Distribution must be 'expon' or 'uniform'")
                
                if method=='kstest':
                    pvalues[k] = kstest(test_times, distribution).pvalue
                elif method=='cramervonmises':
                    pvalues[k] = cramervonmises(test_times, distribution).pvalue
                else:
                    raise ValueError("Method must be 'kstest' or 'cramervonmises'")
            

            return pvalues




    def test_sparsity_alpha(self, level=0.05, asymptotic=True):
        """
        Calculate p-values and apply FDR control on the hypotheses H0: alpha_ij = 0.

        Parameters
        ----------
        level : float, optionnal
            Confidence level for the tests.
            Default is 0.05
        asymptotic : bool, optionnal
            If True, asymptotic normality-based confidence intervals are used. If False, empirical confidence intervals are used.
            Default is True.

        Modifies
        ----------
        self.alpha_tilde_zero_coefficients : array-like of bool, size (d,d) 
            Marks alpha_tilde parameters as null (True where alpha_tilde is zero).
            
        Raises
        --------
        ValueError
            If any of the following conditions are met:  
            - There are no estimations for each realisation.
            - self.model is not 'gvm'.
            - The number of realisations is too small.
        """
        if self.multiple_estimations is None:
            raise ValueError('Must have estimations for each realisation.')
        if self.nb_realisations < 2:
            raise ValueError('Not enough realisations.')

        nb_alpha_params = (~self.alpha_zero_coefficients).sum()
        
        if asymptotic:
            df = self.nb_realisations - 1 
            threshold = (np.arange(1, nb_alpha_params + 1) / nb_alpha_params) * level
            estimations_alpha = np.array( [e[1][~self.alpha_zero_coefficients] for e in self.multiple_estimations] ).T
    
            means = np.mean(estimations_alpha, axis=1)  
            std_errors = np.std(estimations_alpha, axis=1, ddof=1) / np.sqrt(self.nb_realisations)  
            t_stats = means / std_errors
            pvalues = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))  

        else:
            alpha_estimations = np.array( [e[1][~self.alpha_zero_coefficients] for e in self.multiple_estimations] )
            pvalues = 2 * np.minimum( np.mean(alpha_estimations <= 0, axis=0), np.mean(alpha_estimations >= 0, axis=0))
            
        flat_p_values = pvalues.flatten()
        sorted_indices = np.argsort(flat_p_values)
        sorted_p_values = flat_p_values[sorted_indices]
        
        # Apply BH procedure
        thresholds = (np.arange(1, nb_alpha_params + 1) / nb_alpha_params) * level
        below_threshold = sorted_p_values <= thresholds
         
        if np.any(below_threshold):
            max_index = np.max(np.where(below_threshold))
            fdr_cutoff = sorted_p_values[max_index]
        else:
            fdr_cutoff = 0  
        
        rejected_hypotheses = flat_p_values <= fdr_cutoff
        
        zero_matrix = np.zeros((self.d, self.d), dtype=bool)
        k = 0
        for i in range(self.d):
            for j in range(self.d):
                if self.alpha_zero_coefficients[i,j]:
                    zero_matrix[i,j] = True
                else:
                    if not rejected_hypotheses[k]:
                        zero_matrix[i,j] = True
                    k += 1

        self.alpha_zero_coefficients = zero_matrix


    def test_sparsity_alpha_tilde(self, level=0.05, asymptotic=True):
        """
        Calculate p-values and apply FDR control on the hypotheses H0: alpha_tilde_ij = 0.

        Parameters
        ----------
        level : float, optionnal
            Confidence level for the tests.
            Default is 0.05
        asymptotic : bool, optionnal
            If True, asymptotic normality-based confidence intervals are used. If False, empirical confidence intervals are used.
            Default is True.

        Modifies
        ----------
        self.alpha_tilde_zero_coefficients : array-like of bool, size (d,d) 
            Marks alpha_tilde parameters as null (True where alpha_tilde is zero).
            
        Raises
        --------
        ValueError
            If any of the following conditions are met:  
            - There are no estimations for each realisation.
            - self.model is not 'gvm'.
            - The number of realisations is too small.
        """
        if self.multiple_estimations is None:
            raise ValueError('Must have estimations for each realisation.')
        if self.nb_realisations < 2:
            raise ValueError('Not enough realisations.')
        if self.model != 'gvm' :
            raise ValueError('Estimations must be under model gvm.')

        nb_alpha_tilde_params = (~self.alpha_tilde_zero_coefficients).sum()
        
        if asymptotic:
            df = self.nb_realisations - 1 
            threshold = (np.arange(1, nb_alpha_tilde_params + 1) / nb_alpha_tilde_params) * level
            estimations_alpha_tilde = np.array( [e[3][~self.alpha_tilde_zero_coefficients] for e in self.multiple_estimations] ).T
    
            means = np.mean(estimations_alpha_tilde, axis=1)  
            std_errors = np.std(estimations_alpha_tilde, axis=1, ddof=1) / np.sqrt(self.nb_realisations)  
            t_stats = means / std_errors
            pvalues = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=df))  

        else:
            alpha_tilde_estimations = np.array( [e[3][~self.alpha_tilde_zero_coefficients] for e in self.multiple_estimations] )
            pvalues = 2 * np.minimum( np.mean(alpha_tilde_estimations <= 0, axis=0), np.mean(alpha_tilde_estimations >= 0, axis=0))
            
        flat_p_values = pvalues.flatten()
        sorted_indices = np.argsort(flat_p_values)
        sorted_p_values = flat_p_values[sorted_indices]
        
        # Apply BH procedure
        thresholds = (np.arange(1, nb_alpha_tilde_params + 1) / nb_alpha_tilde_params) * level
        below_threshold = sorted_p_values <= thresholds
         
        if np.any(below_threshold):
            max_index = np.max(np.where(below_threshold))
            fdr_cutoff = sorted_p_values[max_index]
        else:
            fdr_cutoff = 0  
        
        rejected_hypotheses = flat_p_values <= fdr_cutoff
        
        zero_matrix = np.zeros((self.d, self.d), dtype=bool)
        k = 0
        for i in range(self.d):
            for j in range(self.d):
                if self.alpha_tilde_zero_coefficients[i,j]:
                    zero_matrix[i,j] = True
                else:
                    if not rejected_hypotheses[k]:
                        zero_matrix[i,j] = True
                    k += 1

        self.alpha_tilde_zero_coefficients = zero_matrix

    
    def test_absence_interactions(self, level=0.05, asymptotic=True):
        """
        Calculate p-values and apply FDR control on the hypotheses H0: alpha_ij = alpha_tilde_ij = 0.
        
        Parameters
        ----------
        level : float, optionnal.
            Confidence level for the tests.
            Default is 0.05.
        asymptotic : bool, optionnal
            If True, asymptotic normality-based confidence intervals are used. If False, empirical confidence intervals are used.
            Default is True.
        
        Modifies
        ----------
        self.alpha_zero_coefficients : array-like of bool, size (d,d) 
            Marks alpha parameters as null (True where alpha and alpha_tilde are zero).
        self.alpha_tilde_zero_coefficients : array-like of bool, size (d,d) 
            Marks alpha_tilde parameters as null (True where alpha and alpha_tilde are zero).
        """
        if self.multiple_estimations is None:
            raise ValueError('Must have estimations for each realisation.')
        if self.nb_realisations < 2:
            raise ValueError('Not enough realisations.')
        
        if asymptotic:
            num_tests = self.d*self.d
            p_values = np.zeros((self.d,self.d))
            
            for i in range(self.d):
                for j in range(self.d):
                    data = np.zeros((self.nb_realisations, 2))
                    for l in range(self.nb_realisations):
                        data[l,0] = self.multiple_estimations[l][1][i,j]
                        data[l,1] = self.multiple_estimations[l][3][i,j]
                        
                    mean_vector = np.mean(data, axis=0)
                    covariance_matrix = np.cov(data, rowvar=False, ddof=1)
                    covariance_matrix_inv = np.linalg.inv(covariance_matrix)
                    t2_stat = self.nb_realisations * mean_vector.T @ covariance_matrix_inv @ mean_vector
                    f_stat = (t2_stat * (self.nb_realisations - 2)) / (2 * (self.nb_realisations - 1))
                    
                    # Compute the p-value
                    p_values[i,j] = 1 - stats.f.cdf(f_stat, 2, self.nb_realisations - 2)
    
            # Flatten the p-values for BH procedure
            flat_p_values = p_values.flatten()
            sorted_indices = np.argsort(flat_p_values)
            sorted_p_values = flat_p_values[sorted_indices]
            
            # Apply BH procedure
            thresholds = (np.arange(1, num_tests + 1) / num_tests) * level
            below_threshold = sorted_p_values <= thresholds
             
            if np.any(below_threshold):
                max_index = np.max(np.where(below_threshold))
                fdr_cutoff = sorted_p_values[max_index]
            else:
                fdr_cutoff = 0  
            
            rejected_hypotheses = flat_p_values <= fdr_cutoff
            zero_matrix = np.zeros((self.d, self.d), dtype=bool)
            zero_matrix.ravel()[~rejected_hypotheses] = True  
    
            self.alpha_zero_coefficients = zero_matrix
            self.alpha_tilde_zero_coefficients = zero_matrix
            
        else:
            nb_alpha_params = (~self.alpha_zero_coefficients).sum()
            alpha_estimations = np.array( [e[1][~self.alpha_zero_coefficients] for e in self.multiple_estimations] )
            pvalues = 2 * np.minimum( np.mean(alpha_estimations <= 0, axis=0), np.mean(alpha_estimations >= 0, axis=0))
            flat_pvalues = pvalues.flatten()
            
            nb_alpha_tilde_params = (~self.alpha_tilde_zero_coefficients).sum()
            alpha_tilde_estimations = np.array( [e[3][~self.alpha_tilde_zero_coefficients] for e in self.multiple_estimations] )
            pvalues_tilde = 2 * np.minimum( np.mean(alpha_tilde_estimations <= 0, axis=0), np.mean(alpha_tilde_estimations >= 0, axis=0))
            flat_pvalues_tilde = pvalues_tilde.flatten()

            all_pvalues = np.concatenate((pvalues,pvalues_tilde))
            num_tests = len(all_pvalues)
            sorted_indices = np.argsort(all_pvalues)
            sorted_p_values = all_pvalues[sorted_indices]
            
            # Apply BH procedure on 2*d*d p-values
            thresholds = (np.arange(1, num_tests + 1) / num_tests) * level
            below_threshold = sorted_p_values <= thresholds
             
            if np.any(below_threshold):
                max_index = np.max(np.where(below_threshold))
                fdr_cutoff = sorted_p_values[max_index]
            else:
                fdr_cutoff = 0  
            
            rejected_hypotheses = all_pvalues <= fdr_cutoff
            rejected_values = rejected_hypotheses[:self.d*self.d]
            rejected_values_tilde = rejected_hypotheses[self.d*self.d:]

            zero_matrix = np.logical_and(~rejected_values.reshape((self.d,self.d)), ~rejected_values_tilde.reshape((self.d,self.d)))
            
            self.alpha_zero_coefficients = zero_matrix
            self.alpha_tilde_zero_coefficients = zero_matrix

        
    def tests_hp_vm(self, level=0.05, asymptotic=True):
        """
        Calculate p-values and apply FDR control on the hypotheses H0: alpha_ij = alpha_tilde_ij and H0: alpha_tilde_ij = 0.

        Parameters
        ----------
        level : float, optionnal
            Confidence level for the tests.
            Default is 0.05.
        asymptotic : bool, optionnal
            If True, asymptotic normality-based confidence intervals are used. If False, empirical confidence intervals are used.
            Default is True.

        Modifies
        ----------
        self.alpha_tilde_zero_coefficients : array-like of bool, size (d,d) 
            Marks alpha_tilde parameters as null (True where alpha_tilde is zero).
        self.equal_coefficients : array-like of bool, shape (d, d)  
            Indicates where alpha parameters are equal to alpha_tilde (True if alpha=alpha_tilde, False otherwise).  
            
        Raises
        --------
        ValueError
            If any of the following conditions are met:  
            - There are no estimations for each realisation.
            - self.model is not 'gvm'.
            - The number of realisations is too small.
        """
        if self.multiple_estimations is None:
            raise ValueError('Must have estimations for each realisation.')
        if self.model != 'gvm' :
            raise ValueError('Estimations must be under model gvm.')
        if self.nb_realisations < 2:
            raise ValueError('Not enough realisations.')

        if asymptotic:
            # Test hp
    
            and_matrix = np.logical_and(self.alpha_zero_coefficients, self.alpha_tilde_zero_coefficients) # positions where both coeffs are zero
            num_params = (~and_matrix).sum()
                        
            df = self.nb_realisations - 1 
            threshold = (np.arange(1, num_params + 1) / num_params) * level
            estimations_alpha_tilde = np.array( [e[3][~and_matrix] for e in self.multiple_estimations] ).T
            estimations_alpha = np.array( [e[1][~and_matrix] for e in self.multiple_estimations] ).T
    
            test_values_hp = estimations_alpha_tilde - estimations_alpha
            means_hp = np.mean(test_values_hp, axis=1)  
            std_errors_hp = np.std(test_values_hp, axis=1, ddof=1) / np.sqrt(self.nb_realisations)  
            t_stats_hp = means_hp / std_errors_hp
            p_values_hp = 2 * (1 - stats.t.cdf(np.abs(t_stats_hp), df=df))  
            sorted_indices_hp = np.argsort(p_values_hp)
            sorted_p_values_hp = p_values_hp[sorted_indices_hp]
            below_threshold_hp = sorted_p_values_hp <= threshold
    
            if np.any(below_threshold_hp):
                max_index = np.max(np.where(below_threshold_hp))
                fdr_cutoff_hp = sorted_p_values_hp[max_index]
            else:
                fdr_cutoff_hp = 0  
                
            rejected_hypotheses_hp = p_values_hp <= fdr_cutoff_hp
            zero_matrix_hp = np.zeros((self.d, self.d), dtype=bool)
            idx=0
            for k in range(self.d*self.d):
                if and_matrix.ravel()[k]:
                    zero_matrix_hp.ravel()[k] = True
                else:
                    zero_matrix_hp.ravel()[k] = ~rejected_hypotheses_hp[idx]
                    idx+=1
                            
            self.equal_coefficients = zero_matrix_hp
            
            # Test vm
    
            self.test_sparsity_alpha_tilde(level, asymptotic=True)
    
            for i in range(self.d):
                for j in range(self.d):
                    if self.alpha_tilde_zero_coefficients[i,j] and self.equal_coefficients[i,j] and ~(self.alpha_zero_coefficients[i,j]):
                        #self.equal_coefficients[i,j] = False
                        self.alpha_tilde_zero_coefficients[i,j] = False

        else: 
            # Test hp
    
            and_matrix = np.logical_and(self.alpha_zero_coefficients, self.alpha_tilde_zero_coefficients) # positions where both coeffs are zero
            num_params = (~and_matrix).sum()
                        
            estimations_alpha_tilde = np.array( [e[3][~and_matrix] for e in self.multiple_estimations] )
            estimations_alpha = np.array( [e[1][~and_matrix] for e in self.multiple_estimations] )
            test_values_hp = estimations_alpha_tilde - estimations_alpha
            pvalues_hp = 2 * np.minimum( np.mean(test_values_hp <= 0, axis=0), np.mean(test_values_hp >= 0, axis=0))
            flat_pvalues_hp = pvalues_hp.flatten()
            sorted_indices_hp = np.argsort(flat_pvalues_hp)
            sorted_p_values_hp = flat_pvalues_hp[sorted_indices_hp]
            threshold = (np.arange(1, num_params + 1) / num_params) * level
            below_threshold_hp = sorted_p_values_hp <= threshold
    
            if np.any(below_threshold_hp):
                max_index = np.max(np.where(below_threshold_hp))
                fdr_cutoff_hp = sorted_p_values_hp[max_index]
            else:
                fdr_cutoff_hp = 0  
                
            rejected_hypotheses_hp = flat_pvalues_hp <= fdr_cutoff_hp
            zero_matrix_hp = np.zeros((self.d, self.d), dtype=bool)
            idx=0
            for k in range(self.d*self.d):
                if and_matrix.ravel()[k]:
                    zero_matrix_hp.ravel()[k] = True
                else:
                    zero_matrix_hp.ravel()[k] = ~rejected_hypotheses_hp[idx]
                    idx+=1
    
            self.equal_coefficients = zero_matrix_hp
    
            # Test vm
    
            self.test_sparsity_alpha_tilde(level, asymptotic=False)
    
            for i in range(self.d):
                for j in range(self.d):
                    if self.alpha_tilde_zero_coefficients[i,j] and self.equal_coefficients[i,j] and ~self.alpha_zero_coefficients[i,j]:
                        self.alpha_tilde_zero_coefficients[i,j] = False


    def negloglikelihood_with_grad_jit(self, parameters, realisation_index=None, t=100):
        """
        Compute the log-likelihood and gradient using njit, either for a specific realisation or as the sum across all realisations.

        Parameters
        ----------
        parameters : array
            Model parameters. 
            - For 'hp' and 'vm': 
              [mu_1, ..., mu_d, alpha_11, alpha_12, ..., alpha_dd, beta_1, ..., beta_d].
            - For 'gvm': 
              [mu_1, ..., mu_d, alpha_11, alpha_12, ..., alpha_dd, beta_1, ..., beta_d, alpha_tilde_11, alpha_tilde_12, ..., alpha_tilde_dd].
        realisation_index : int, optionnal
            Index of the realisation for which to compute the log-likelihood.
            If 'None' (default), computes the sum of log-likelihoods across all realisations.
        t : int, optionnal
            Softplus function parameter.
            Default is 100.

        Returns
        ----------
        loglikelihood : float
            - If 'realisation_index' is 'None': returns the sum of the log-likelihoods across all realisations.
            - Otherwise: returns the log-likelihood for the specified 'realisation_index'.
        gradient : array
            - If 'realisation_index' is 'None': returns the sum of gradients across all realizations.
            - Otherwise: returns the gradient for the specified 'realisation_index'.
            
        Raises
        --------
        ValueError
            If there are no times.
        """
        if self.times is None :
            raise ValueError("There are no times.")
        self.nb_realisations = len(self.times)
        if realisation_index is not None:  
            return(negloglikelihood_with_grad_single_jit(parameters,np.array(self.times[realisation_index]), self.model_int, self.d, self.alpha_zero_coefficients, self.alpha_tilde_zero_coefficients, self.equal_coefficients, t))
        else :
            sum_loglikelihoods = [0.0, np.zeros(self.nb_params)]
            for k in range(self.nb_realisations):
                negloglik = negloglikelihood_with_grad_single_jit(parameters,np.array(self.times[k]), self.model_int, self.d, self.alpha_zero_coefficients, self.alpha_tilde_zero_coefficients, self.equal_coefficients, t)
                sum_loglikelihoods[0] += negloglik[0]
                sum_loglikelihoods[1] += negloglik[1]
            return sum_loglikelihoods

        
    def negloglikelihood_with_grad_jit_pen(self, parameters, realisation_index=None, t=100, lambda_pen=0.01, mu_pen=0.01):
        """
        Compute the penalised log-likelihood and gradient using njit, either for a specific realisation or as the sum across all realisations.

        Parameters
        ----------
        parameters : array
            Model parameters. 
            - For 'hp' and 'vm': 
              [mu_1, ..., mu_d, alpha_11, alpha_12, ..., alpha_dd, beta_1, ..., beta_d].
            - For 'gvm': 
              [mu_1, ..., mu_d, alpha_11, alpha_12, ..., alpha_dd, beta_1, ..., beta_d, alpha_tilde_11, alpha_tilde_12, ..., alpha_tilde_dd].
        realisation_index : int, optionnal
            Index of the realisation for which to compute the log-likelihood.
            If 'None' (default), computes the sum of log-likelihoods across all realisations.
        t : int, optionnal
            Softplus function parameter.
            Default is 100.

        Returns
        ----------
        loglikelihood : float
            - If 'realisation_index' is 'None': returns the sum of the log-likelihoods across all realisations.
            - Otherwise: returns the log-likelihood for the specified 'realisation_index'.
        gradient : array
            - If 'realisation_index' is 'None': returns the sum of gradients across all realizations.
            - Otherwise: returns the gradient for the specified 'realisation_index'.
            
        Raises
        --------
        ValueError
            If there are no times.
        """
        if self.times is None :
            raise ValueError("There are no times.")
        self.nb_realisations = len(self.times)
        if realisation_index is not None:  
            return(negloglikelihood_with_grad_single_jit_pen(parameters,np.array(self.times[realisation_index]), self.model_int, self.d, t, lambda_pen, mu_pen))
        else :
            sum_loglikelihoods = [0.0, np.zeros(self.nb_params)]
            for k in range(self.nb_realisations):
                negloglik = negloglikelihood_with_grad_single_jit_pen(parameters,np.array(self.times[k]), self.model_int, self.d, t, lambda_pen, mu_pen)
                sum_loglikelihoods[0] += negloglik[0]
                sum_loglikelihoods[1] += negloglik[1]
            return sum_loglikelihoods


    def calculate_test_values(self, parameters, realisation):
        """
        Calculate the test values for the goodness-of-fit test based on compensator differences.
    
        The function computes the inter-arrival times of the transformed event times (Lambda(T_k)) via the compensator for the given realization.
    
        Parameters
        ----------
        parameters : array-like
            Model parameters, which vary depending on the model type:
            - For 'hp' and 'vm', the parameters are [mu, alpha, beta].
            - For 'gvm', the parameters are [mu, alpha, beta, alpha_tilde].
        realisation : list of tuple (float, int)
            Sequence of event times and their corresponding dimensions.
    
        Returns
        -------
        compensator_differences : array
            Array containing the inter-arrival times of (Lambda(T_k)) for each event time (T_k) in 'realisation'.
        """
        N = len(realisation)

        if self.model == 'hp' or self.model =='vm' :
            mu,alpha,beta = parameters
            # Initialise restart times, differences of compensators, compensators and  intensities
            restart_times = np.zeros(self.d)
            compensator_differences = np.zeros(N)
            compensator_differences[0] = np.sum(mu) * realisation[0][0]
            intensities = np.zeros(self.d)

            for i in range(1,N):
                jump_time = realisation[i][0]
                last_jump_time = realisation[i-1][0]

                index = realisation[i][1]
                last_index = realisation[i-1][1]

                # Use recursion formulas
                decay_factor = np.exp(-beta * (jump_time-last_jump_time))
                
                restart_times = np.minimum(jump_time,last_jump_time + ( np.log( np.where(mu+intensities+alpha[:,last_index] < 0,(-intensities-alpha[:,last_index])/mu,1))) / beta )
                j_i = mu * (jump_time - restart_times) + ( (intensities+alpha[:,last_index])/beta )* (np.exp(-beta*(restart_times-last_jump_time))-np.exp(-beta*(jump_time - last_jump_time))) 

                compensator_differences[i] = np.sum(j_i)

                intensities = decay_factor * (intensities + alpha[:,last_index])
                if self.model =='vm':
                    intensities[index]=0  # Memory reset
            return compensator_differences
        else:
            mu,alpha,beta,alpha_tilde = parameters
            # Initialise restart times, differences of compensators, compensators and  intensities
            restart_times = np.zeros(self.d)
            compensator_differences = np.zeros(N)
            compensator_differences[0] = np.sum(mu) * realisation[0][0]
            intensities = np.zeros(3*self.d)

            for i in range(1,N):
                jump_time = realisation[i][0]
                last_jump_time = realisation[i-1][0]

                index = realisation[i][1]
                last_index = realisation[i-1][1]

                decay_factor = np.exp(-beta * (jump_time-last_jump_time))

                restart_times = np.minimum(jump_time,last_jump_time + ( np.log( np.where(mu+intensities[:self.d]+intensities[2*self.d:]+alpha[:,last_index] < 0,(-intensities[:self.d]-intensities[2*self.d:]-alpha[:,last_index])/mu,1))) / beta )
                j_i = mu* (jump_time - restart_times) + ( (intensities[:self.d]+intensities[2*self.d:]+alpha[:,last_index]) * (np.exp(-beta*(restart_times-last_jump_time))-np.exp(-beta*(jump_time-last_jump_time))) ) /beta

                compensator_differences[i] = np.sum(j_i)

                intensities[:self.d] = decay_factor * (intensities[:self.d] + alpha[:,last_index])
                intensities[index] = 0

                intensities[2*self.d:] = decay_factor * intensities[2*self.d:]
                intensities[2*self.d+index] += decay_factor[index]*(intensities[self.d+index]+alpha_tilde[index][last_index])
                
                intensities[self.d:2*self.d] = decay_factor * (intensities[self.d:2*self.d] + alpha_tilde[:,last_index])
                intensities[self.d+index] = 0

                
            return compensator_differences


    def calculate_average_estimation(self):
        """
        Calculate the average estimation based on multiple realisations 'self.multiple_estimations' and stores result in 'self.average_estimation'.
    
        Modifies
        --------
        self.average_estimation : tuple of arrays
            The average parameter estimation, where each array is the mean of the respective parameter values across all realisations in 'self.multiple_estimations'.
    
        Raises
        ------
        ValueError
            If 'self.multiple_estimations' is 'None'.
        
        Notes
        -----
        - For models other than 'gvm', the average estimation consists of three arrays: 
          (mean_1, mean_2, mean_3).
        - For the 'gvm' model, the average estimation includes four arrays: 
          (mean_1, mean_2, mean_3, mean_4).
        """
        if self.multiple_estimations is None:
            raise ValueError("No multiple estimations available.")
            
        arrays_1 = np.array([x[0] for x in self.multiple_estimations])  # First array in each tuple
        arrays_2 = np.array([x[1] for x in self.multiple_estimations])  # Second array in each tuple
        arrays_3 = np.array([x[2] for x in self.multiple_estimations])  # Third array in each tuple
        
        mean_1 = np.mean(arrays_1, axis=0)
        mean_2 = np.mean(arrays_2, axis=0)
        mean_3 = np.mean(arrays_3, axis=0)
            
        if self.model == 'gvm':
            arrays_4 = np.array([x[3] for x in self.multiple_estimations])
            mean_4 = np.mean(arrays_4, axis=0)
            
            self.average_estimation = mean_1, mean_2, mean_3, mean_4
        else :
            self.average_estimation = mean_1, mean_2, mean_3
