import torch
from torch.nn import ModuleList

import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.lazy import SumLazyTensor
from src.utils import trace_stats, expect_kl

from src.mean_field_hs import MeanFieldHorseshoe

from typing import List

def get_stats(variational_strategy:VariationalStrategy):
    """Get the statistics trace of (S + m^\top m) k_i(\vZ, \vZ) from variational strategy"""

    # q(u)
    variation_dist = variational_strategy.variational_distribution
    m = variation_dist.mean
    S = variation_dist.covariance_matrix
    S_m = S + m.t() @ m
    q_S_m = MultivariateNormal(torch.zeros_like(m), S_m)
    # p(u)
    prior = variational_strategy(variational_strategy.inducing_points, prior=True)

    return trace_stats(q_S_m, prior)

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

class StructuralVariationalStrategy(object):

    def __init__(self, gps, horseshoe:MeanFieldHorseshoe):
        self.gps = gps
        self.horseshoe = horseshoe

    def kl_divergence(self):
        expect_invese = self.horseshoe().squeeze()
        kls = []
        for i, gp in enumerate(self.gps):
            variational_strategy = gp.variational_strategy
            variation_dist = variational_strategy.variational_distribution
            prior = variational_strategy.prior_distribution
            new_prior = MultivariateNormal(prior.mean, prior.covariance_matrix * expect_invese[i])
            kl = torch.distributions.kl.kl_divergence(variation_dist, new_prior)
            # kl = variational_strategy.kl_divergence()
            kls.append(kl)

        gp_kl = sum(kls)
        horseshoe_kl = self.horseshoe.kl_divergence() #.detach() # make sure there is no gradient calculation
        return gp_kl + horseshoe_kl

class StructuralSparseGP(ApproximateGP):

    def __init__(self, gps: List[ApproximateGP], horseshoe:MeanFieldHorseshoe):
        super().__init__(None)
        # sanity check #kernels = #dimension
        assert len(gps) == horseshoe.n_dims
        self.horseshoe = horseshoe
        self.gps = ModuleList(gps)
        self.variational_strategy = StructuralVariationalStrategy(gps, horseshoe)

    def forward(self, x):
        pass

    def __call__(self, inputs, prior=False, **kwargs):
        posteriors = []
        for gp in self.gps:
            posterior = gp(inputs, prior, **kwargs)
            posteriors += [posterior]

        weights = self.horseshoe().squeeze()
        means, covars = [], []
        for i in range(self.horseshoe.n_dims):
            w_i = weights[i].squeeze()
            mean_i = posteriors[i].mean
            covar_i = posteriors[i].covariance_matrix * w_i
            means.append(mean_i)
            covars.append(covar_i)

        mean = sum(means)
        covar = SumLazyTensor(*covars)
        return MultivariateNormal(mean, covar)





