import torch
from torch.nn import ModuleList
from gpytorch.module import Module

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.models import ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.lazy import SumLazyTensor
from typing import List


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


class BaseSparseSelector(Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def kl_divergence(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class TrivialSelector(BaseSparseSelector):

    def __init__(self, dim):
        super(TrivialSelector, self).__init__(dim)

    def kl_divergence(self):
        return torch.tensor([0.])

    def __call__(self):
        return torch.ones(self.dim)


class SpikeAndSlabSelector(BaseSparseSelector):

    def __init__(self, dim, gumbel_temp=1.):
        super(SpikeAndSlabSelector, self).__init__(dim)
        self.gumbel_temp = gumbel_temp
        self.register_parameter(name="raw_prob",
                                parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        prob_constraint = Interval(0., 1.)
        self.register_constraint("raw_prob", prob_constraint)
        self.register_parameter(name='w_mean',
                                parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        self.register_parameter(name="log_w_var",
                                parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        self.register_parameter(name="log_w_zero",
                                parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))

    @property
    def prob(self):
        return self.raw_prob_constraint.transform(self.raw_prob)

    def kl_divergence(self):
        log_prior = torch.sum(self.prob * tor)
        entropy = 0.
        return entropy + log_prior

    def gumbel_sample(self):
        bar_prob = 1. - self.prob
        stacked = torch.cat([bar_prob, self.prob], dim=1)
        uniform = torch.rand(self.dim, 2)
        z = - torch.log(-torch.log(uniform)) + torch.log(stacked)
        applied_softmax = torch.softmax(z, dim=1) / self.gumbel_temp
        return applied_softmax[:,1]

    def __call__(self):

        g = self.gumbel_sample()[:,None]
        mean = g * self.w_mean
        var = g * torch.exp(self.log_w_var) + (1. - g) * torch.exp(self.log_w_zero)
        return mean + var * torch.randn(self.dim, 1)


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
            covar_i = posteriors[i].covariance_matrix * w_i ** 2
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
