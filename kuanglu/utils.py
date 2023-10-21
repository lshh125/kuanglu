import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.stats as stats
import seaborn as sns
import torch

# distributions:

def log_normal(x, mean, var):
    return -0.5 * torch.log(2 * torch.tensor(np.pi)) - 0.5 * torch.log(var) - (x - mean) ** 2 / (2 * var)

def normal(x, mean, var):
    return torch.exp(log_normal(x, mean, var))

def log_negative_binomial(x, mu, theta):
    return torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1) + theta * torch.log(theta) + x * torch.log(mu) - (x + theta) * torch.log(mu + theta)

def negative_binomial(x, mu, theta):
    return torch.exp(log_negative_binomial(x, mu, theta))

def log_zero_inflated_negative_binomial(x, pi, mu, theta):
    return torch.log(pi + (1 - pi) * negative_binomial(x, mu, theta) + 1e-8) if x == 0 else torch.log(1 - pi + 1e-8) + log_negative_binomial(x, mu, theta)

def zero_inflated_negative_binomial(x, pi, mu, theta):
    return torch.exp(log_zero_inflated_negative_binomial(x, pi, mu, theta))


