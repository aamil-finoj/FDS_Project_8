"""Project 8 helper package: inverse problems with multiple valid solutions.

Modules
-------
config     : selected experimental settings (one place to change them)
data       : seeding + the generative model t ~ prior, x = t^2 + eps
reference  : exact posterior p(t | x) via 1-D quadrature (the "answer key")
models     : PyTorch networks (MSE point predictor, MDN; M=1 is the single Gaussian)
training   : one generic training loop with early stopping + result caching
evaluate   : densities, NLL, sampling, branch metrics, Wasserstein distance
plotting   : shared plotting style/helpers
"""
