import torch
import sys
sys.path.append("../..")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--selector', '-s',
choices=["TrivialSelector", "SpikeAndSlabSelector", "HorseshoeSelector"],
help='''
Selectors:
TrivialSelector
SpikeAndSlabSelector
HorseshoeSelector
''', default = "HorseshoeSelector")

parser.add_argument('--acq_fun', '-a',
choices=["UCB", "EI", "POI"],
help='''
UCB: Upper Confidence Bound
EI: Expected Improvement
POI: Probability of Improvement
''', default = "UCB")

parser.add_argument('--num_trial', '-t', type = int, default = 10, help = "Number of Bayesian Optimization Interations")
parser.add_argument('--batch_size', '-b', type = int, default = 4)
parser.add_argument('--num_raw_samples', '-r', type = int, default = 20)
parser.add_argument('--num_inducing', '-i', type = int, default = 10)
parser.add_argument('--n_kernels', '-k', type = int, default = 5)
parser.add_argument('--num_step', '-p', type = int, default = 50, help = "Number of steps to optimize surrogate model for each BO stages")
parser.add_argument('--learning_rate', '-l', type = float, default = 3e-6, help = "learning rate in Adam optimizer")

"""
##Deprecated
import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
"""
from scipy.optimize import minimize

from utils import branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock

norm = torch.distributions.normal.Normal(0, 1)

def acq_max(bounds, sur_model, y_max, acq_fun, n_warmup = 10000, iteration = 10):
    x_tries = torch.empty(n_warmup, bounds.shape[0])._rand() * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    ys = acq_fun(x_tries, sur_model, y_max, kappa)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    for iterate in range(iteration):
        locs = torch.empty(bounds.shape[0])._rand() * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        res = minimize(lambda x: -acq_fun(x),
                      locs,
                      bounds = bounds,
                      method = "L-BFGS-B")

        if not res.success:
            continue

        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    return torch.clip(x_max, bounds[:, 0], bounds[:, 1])

#acqusition functions
def UCB(x, sur_model, kappa = 2.576):
    mean, std = sur_model(x)
    return mean + kappa * std

def EI(x, sur_model, y_max):
    mean, std = sur_model(x)
    a = (mean - ymax - x)
    z = a / std
    return a * norm.cdf(z) + std * norm.pdf(z)

def POI(x, sur_model, y_max):
    mean, std = sur_model(x)
    z = (mean - y_max - x)/std
    return norm.cdf(z)

#models

    
if __name__ == "__main__":
    for obj_fun in [branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock]:
        model = 
        