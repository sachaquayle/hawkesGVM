# Estimation for Generalised Hawkes Processes with Variable Length Memory (GVM)

Python code for simulating and estimating Generalised Hawkes Processes with variable length memory.

This repository implements the methods described in:

> **S. Quayle, A. Bonnet, M. Sangnier**,  
> *Hawkes Processes with Variable Length Memory: Existence, Inference and Application to Neuronal Activity*.  
> arXiv:? (to be updated)

## Features

- Multivariate Hawkes processes
- Variable length memory with exponential decay kernels
- Simulation from specified parameters
- Parameter estimation by Maximum Likelihood Estimation (MLE)
- Interaction type inference via statistical confidence intervals
- Goodness-of-fit tests

## Dependencies

This code was implemented using Python >=3.8 and requires Numpy, Matplotlib, Scipy.

## Installation

Copy all files in the current working directory.

## Example 

Includes a basic simulation and estimation example using synthetic data.

The script in `examples/basic_usage.py` demonstrates how to:

- Define a Generalised Hawkes process with variable length memory
- Simulate spike trains from the model
- Estimate the model parameters via maximum likelihood
- Visualise the estimated intensity functions and confidence intervals

## Reproducibility

The parameters used in the synthetic experiments of our paper are provided in  
`examples/paper_parameters.py`.

The data used and preprocessing steps for applying the model to real neuronal spike train data (as described in the paper) are available in  `examples/neuronal_data` and `examples/preprocess_neuronal_data.py`.

## Author

Sacha Quayle

## References

S. Quayle, A. Bonnet, M. Sangnier, Hawkes Processes with Variable Length Memory: Existence, Inference and Application to Neuronal Activity. arXiv:?
