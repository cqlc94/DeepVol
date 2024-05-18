import math
import torch
import numpy as np
from scipy.stats import norm


### Loss Functions
def nll_loss(std, target):
    var = torch.square(std)
    return 0.5 * torch.log(2 * math.pi * var) + (target)**2 / (2 * var)

def smoothed_nll_loss(std, target, epsilon=1e-10):
    var = std**2
    pdf = 1 / torch.sqrt(2 * math.pi * var) * torch.exp(-(target**2) / (2 * var))
    return -torch.log(pdf + epsilon)

### Evaluation Metrics - Numpy
def nll(std, target):
    var = np.square(std)
    return 0.5 * np.log(2 * math.pi * var) + (target)**2 / (2 * var)

def mse(std, target):
    return (std-target)**2

def quantile_loss(std, target, alpha=0.01):
    quantile = target.mean() + std * norm.ppf(alpha)
    indicator = (target < quantile).astype(int)
    return (alpha - indicator) * (target - quantile)

def jointloss(std, target, alpha=0.01):
    quantile = target.mean() + std * norm.ppf(alpha)
    indicator = (target < quantile).astype(int)
    esn = norm.pdf(norm.ppf(alpha))/alpha
    es = target.mean() - std * esn
    return -np.log((alpha-1)/es) - (target - quantile) * (alpha - indicator) / es / alpha

def loss_ls(loss_fn, std_ls, target_ls, **kwargs):
    return np.array([loss_fn(std, target, **kwargs).mean() for std, target in zip(std_ls, target_ls)])

