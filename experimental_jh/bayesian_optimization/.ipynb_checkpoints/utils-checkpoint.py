import torch

import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood

from horseshoe_gp.src.structural_sgp import VariationalGP, StructuralSparseGP, \
TrivialSelector, SpikeAndSlabSelector, SpikeAndSlabSelectorV2, HorseshoeSelector
from horseshoe_gp.src.mean_field_hs import MeanFieldHorseshoe, VariatioalHorseshoe

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
def horseshoe():
    
    
    
    
