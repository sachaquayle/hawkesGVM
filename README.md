# Estimation for Generalised Hawkes Processes with Variable Length Memory (GVM)

Python code for simulating and estimating Generalised Hawkes Processes with variable length memory.

This repository implements the methods described in:

> **S. Quayle, A. Bonnet, M. Sangnier**,  
> *Hawkes Processes with Variable Length Memory: Existence, Inference and Application to Neuronal Activity*.  
> arXiv: https://arxiv.org/abs/2507.22867

## Features

- Multivariate Hawkes processes
- Variable length memory with exponential decay kernels
- Simulation
- Parameter estimation by Maximum Likelihood Estimation
- Estimation of interaction types via confidence intervals
- Goodness-of-fit tests

## Dependencies

This code was implemented using Python >=3.8 and requires Numpy, Matplotlib, Scipy.

## Installation

Copy all files in the current working directory.

## Simulation examples

The `simulations/` folder contains all simulations presented in the main article and supplementary material, illustrating the 5-step estimation procedure described in [1].

- `simulations.ipynb` contains all scripts required to reproduce the figures from the article.
- `results_sim/` contains the estimation results saved as `.txt` files.
- `supplementary_material.ipynb` contains the scripts used to reproduce the figures from the supplementary material.

## Results on neuronal data

The `data/` folder contains the neuronal datasets together with the corresponding estimation results.

- `neuronal_data/` contains the original neuronal recordings.
- `preprocessing.ipynb` preprocesses the original trials, saves the processed data in `scripts_and_data/7_trials.txt` using the required format, and generates a sample of resampled trials stored in `scripts_and_data/data_61.txt` for estimation.
- `scripts_and_data/` contains all scripts used for estimation and file generation.
- `results_estimation_trials.ipynb` presents the estimation results and reproduces the figures from the article.

## Author

Sacha Quayle

## References

[1] S. Quayle, A. Bonnet, M. Sangnier, Hawkes Processes with Variable Length Memory: Existence, Inference and Application to Neuronal Activity. arXiv: https://arxiv.org/abs/2507.22867
