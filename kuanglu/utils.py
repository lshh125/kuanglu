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
import torch.nn as nn

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


# utils:

def _nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)

def _nelem(x):
    nelem = torch.sum(~torch.isnan(x)).type_as(x)
    return torch.where(torch.eq(nelem, 0.), torch.tensor(1.).type_as(x), nelem)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return torch.sum(x) / nelem

def masking(X, *, cell_rate=.6, gene_rate=.6, copy=True, device='cuda'):
    if copy:
        X = X.clone()
    row_mask = (torch.rand((X.shape[-2], 1)) < cell_rate) + 0.
    col_mask = (torch.rand((1, X.shape[-1])) < gene_rate) + 0.
    mask = 1 - (row_mask * col_mask).to(device)
    return X * mask, row_mask.squeeze(), col_mask.squeeze()


# losses:

def mse_loss(y_true, y_pred):
    ret = (y_pred - y_true) ** 2
    return _reduce_mean(ret)


def poisson_loss(y_true, y_pred):
    y_pred = y_pred.float()
    y_true = y_true.float()
    
    nelem = _nelem(y_true)
    y_true = _nan2zero(y_true)
    
    ret = y_pred - y_true * torch.log(y_pred + 1e-10) + (y_true + 1e-10).lgamma()
    
    return torch.sum(ret) / nelem


class NB(nn.Module):
    def __init__(self, theta=None, masking=False, scale_factor=1.0, debug=False):
        # For numerical stability
        super().__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.masking = masking
        self.theta = theta
        
    def forward(self, y_true, y_pred, mean=True):
        y_true = y_true.float()
        y_pred = y_pred.float() * self.scale_factor

        if self.masking:
            nelem = _nelem(y_true)
            y_true = _nan2zero(y_true)

        theta = torch.clamp(self.theta, max=1e6)

        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))

        final = t1 + t2
        final = _nan2inf(final)

        if mean:
            if self.masking:
                final = torch.sum(final) / nelem
            else:
                final = torch.mean(final)

        return final
    

class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):

        nb_case = super().forward(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + self.eps)

        y_true = y_true.float()
        y_pred = y_pred.float() * self.scale_factor
        theta = torch.clamp(self.theta, max=1e6)

        zero_nb = torch.pow(theta / (theta + y_pred + self.eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + self.eps)
        result = torch.where(y_true < 1e-8, zero_case, nb_case)
        ridge = self.ridge_lambda * torch.pow(self.pi, 2)
        result += ridge

        if mean:
            if self.masking:
                result = _reduce_mean(result)
            else:
                result = torch.mean(result)

        result = _nan2inf(result)
        
        return result