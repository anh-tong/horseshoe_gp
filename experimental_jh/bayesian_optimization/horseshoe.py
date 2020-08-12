import math
import torch

import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood

from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement

from horseshoe_gp.src.structural_sgp import VariationalGP, StructuralSparseGP, \
TrivialSelector, SpikeAndSlabSelector, SpikeAndSlabSelectorV2, HorseshoeSelector
from horseshoe_gp.src.mean_field_hs import MeanFieldHorseshoe, VariatioalHorseshoe

selector = HorseshoeSelector(dim=n_kernels, A=1., B=1.)
model = StructuralSparseGP(gps, selector)

# set up kernels
n_kernels = 5
means = [ZeroMean()] * n_kernels
kernels = [RBFKernel()] * n_kernels

n_inducing = 10
inducing_points = torch.linspace(0, 1, n_inducing)

# GP for each kernel
gps = []
for mean, kernel in zip(means, kernels):
    gp = VariationalGP(mean, kernel, inducing_points)
    gps.append(gp)