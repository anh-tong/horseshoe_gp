from typing import List

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import SumLazyTensor
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from torch.nn import ModuleList

from src.sparse_selector import BaseSparseSelector


class VariationalGP(ApproximateGP):

    def __init__(self, mean, kernel, inducing_points):
        variational_dist = CholeskyVariationalDistribution(inducing_points.size(-1))
        variational_strat = VariationalStrategy(self, inducing_points, variational_dist)
        super().__init__(variational_strat)
        self.mean_module = mean
        self.covar_model = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_model(x)
        return MultivariateNormal(mean_x, covar_x)


class StructuralSparseGP(ApproximateGP):

    def __init__(self, gps: List[ApproximateGP], selector: BaseSparseSelector):
        super().__init__(None)
        # sanity check #kernels = #dimension
        assert len(gps) == selector.dim
        self.selector = selector
        self.gps = ModuleList(gps)
        self.variational_strategy = StructuralVariationalStrategy(self)

    def forward(self, x):
        pass

    def __call__(self, inputs, prior=False, **kwargs):
        posteriors = []
        for gp in self.gps:
            posterior = gp(inputs, prior, **kwargs)
            posteriors += [posterior]

        weights = self.selector().squeeze()
        means, covars = [], []
        for i in range(self.selector.dim):
            w_i = weights[i].squeeze()
            mean_i = posteriors[i].mean * w_i
            covar_i = posteriors[i].covariance_matrix * torch.square(w_i)
            means.append(mean_i)
            covars.append(covar_i)

        mean = sum(means)
        covar = SumLazyTensor(*covars)
        return MultivariateNormal(mean, covar)


class StructuralVariationalStrategy(object):

    def __init__(self, model: StructuralSparseGP):
        self.model = model

    def kl_divergence(self):
        kls = []
        for i, gp in enumerate(self.model.gps):
            variational_strategy = gp.variational_strategy
            kl = variational_strategy.kl_divergence()
            kls.append(kl)

        gp_kl = sum(kls)
        horseshoe_kl = self.model.selector.kl_divergence()
        return gp_kl + horseshoe_kl
