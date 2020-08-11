import math
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

###acqusition functions
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

###models
def horseshoe():
    
    
    
    

    
###Benchmark Black-Box Funtions
#https://arxiv.org/pdf/1308.4008.pdf

def branin_rcos(x1, x2):
    return (x2 - 5.1/(4*(math.pi**2))*(x1**2) + 5/math.pi*x1 - 6)**2 + 10*(1-1/(8*math.pi))+10
#range: −5 ≤ x1 ≤ 10, 0 ≤ x2 ≤ 15
#sol: 0.397887 at (-pi, 12.275), (pi, 2.275) and (9.42478, 2.475)

def six_hump_camel_back(x1, x2):
    return (4-2.1*x1**2+x1**(4/3))*(x1**2)+x1*x2+(-4+4*(x2**2))*(x2**2)
#range -5 to 5 fo all x_i
#sol: -1.0316 at (-0.0898, 0.7126) and (0.0898, -0.7126)

def hartman_6(x_vec):
    P = torch.Tensor([
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5586],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ])
    A = torch.Tensor([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],    
    ])
    c = torch.Tensor([1, 1.2, 3, 3.2])
    return -(c*(-A*((x_vec.repeat(4).view(4, -1) - P)**2)).sum(1).exp()).sum()
#range 0 to 1 for all x_i
#sol: -3.3062 at (0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301)

def goldstein_price(x1, x2):
    return (1 + (x1 + x2 + 1)**2 * (19 - 14 *x1 + 3*(x1**2) - 14*x2 + 6*x1*x2 + 3*(x2**2)))*(30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12* (x1**2) + 48 * x2 - 36*x1*x2 + 27*(x2**2)))
#range -2 to 2
#sol 3 at (0, -1)

def rosenbrock(x_vec):
    res = 0.0
    for dim in range(x_vec.shape[0] - 1):
        res += 100 * (x_vec[dim+1] - x_vec[dim]**2)**2 + (x_vec[dim] - 1)**2
    return res
    
#range: −30 ≤ xi ≤ 30
#sol: 0 at (1,1,1,1, ...)