import numpy as np
import torch
from torch.nn import ModuleList
from gpytorch.module import Module

import gpytorch
from gpytorch.constraints import Interval, Positive
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
        self.register_parameter(name="raw_sparsity",
                                parameter=torch.nn.Parameter(torch.zeros(1)))
        sparsity_constraint = Interval(0., 1.)
        self.register_constraint("raw_sparsity", sparsity_constraint)
        self.register_parameter(name="raw_scale",
                                parameter=torch.nn.Parameter(torch.zeros(1)))

    @property
    def prob(self):
        return self.raw_prob_constraint.transform(self.raw_prob)

    @property
    def sparsity_prior(self):
        return self.raw_sparsity_constraint.transform(self.raw_sparsity)

    @property
    def scale_prior(self):
        # return torch.tensor([1.])
        return torch.exp(self.raw_scale) + 1e-20

    def kl_divergence(self):

        eps = 1e-20 # make computation stable for log
        log_prior = - 0.5 * self.dim * (torch.log(2 * np.pi * self.scale_prior)) - \
                    0.5 * (torch.exp(self.log_w_var) + torch.square(self.w_mean)).sum() / self.scale_prior + \
                    (self.prob * torch.log(self.sparsity_prior + eps)).sum() + \
                    ((1. - self.prob) * torch.log(1. - self.sparsity_prior + eps)).sum()
        entropy = - self.prob * torch.log(self.prob) - \
                  (1. - self.prob) * torch.log(1. - self.prob) + \
                  0.5 * (1 - self.prob) * (torch.log(2. * np.pi * torch.exp(torch.tensor([1.]))) + self.log_w_zero)\
                  + 0.5 * self.prob * (torch.log(2. * np.pi * torch.exp(torch.tensor([1.]))) + self.log_w_var)
        entropy = entropy.sum()

        return -entropy - log_prior

    def gumbel_sample(self, eps=1e-20):
        bar_prob = 1. - self.prob
        stacked = torch.cat([bar_prob, self.prob], dim=1)
        uniform = torch.rand(self.dim, 2)
        z = - torch.log(-torch.log(uniform + eps) + eps) + torch.log(stacked)
        applied_softmax = torch.softmax(z, dim=1) / self.gumbel_temp
        return applied_softmax[:, 1]

    def __call__(self):
        g = self.gumbel_sample()[:, None]
        mean = g * self.w_mean
        # var = g * torch.exp(self.log_w_var) + (1. - g) * torch.exp(self.log_w_zero)
        var = g * torch.exp(self.log_w_var) + (1. - g) * self.scale_prior
        return mean + var * torch.randn(self.dim, 1)

class SpikeAndSlabSelectorV2(BaseSparseSelector):

    def __init__(self, dim, gumbel_temp=0.1):
        super().__init__(dim)
        self.gumbel_temp = gumbel_temp
        self.register_parameter(name="w_mu", parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        self.register_parameter(name="raw_w_s2", parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        w_s2_constraint = Positive()
        self.register_constraint("raw_w_s2", w_s2_constraint)
        self.register_parameter(name="raw_s_pi", parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        s_pi_constraint = Interval(0., 1.)
        self.register_constraint("raw_s_pi", s_pi_constraint)
        self.register_parameter(name="raw_pi_w", parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        pi_w_constraint = Interval(0., 1.)
        self.register_constraint("raw_pi_w", pi_w_constraint)
        self.register_parameter(name="raw_s2_w", parameter=torch.nn.Parameter(torch.zeros(1)))
        s2_w_constraint = Positive()
        self.register_constraint("raw_s2_w", s2_w_constraint)

        # Only for linear model
        self.register_parameter(name="raw_s2", parameter=torch.nn.Parameter(torch.zeros(1)))
        s2_constraint = Positive()
        self.register_constraint("raw_s2", s2_constraint)


    @property
    def w_s2(self):
        return self.raw_w_s2_constraint.transform(self.raw_w_s2)

    @property
    def s_pi(self):
        return self.raw_s_pi_constraint.transform(self.raw_s_pi)

    @property
    def pi_w(self):
        return self.raw_pi_w_constraint.transform(self.raw_pi_w)

    @property
    def s2_w(self):
        return self.raw_s2_w_constraint.transform(self.raw_s2_w)

    @property
    def s2(self):
        return self.raw_s2_constraint.transform(self.raw_s2)

    def gumbel_sample(self, eps=1e-20):
        bar_s_pi = 1. - self.s_pi
        stacked = torch.cat([bar_s_pi, self.s_pi], dim=1)
        uniform = torch.rand(self.dim, 2)
        z = - torch.log(-torch.log(uniform + eps) + eps) + torch.log(stacked)
        applied_softmax = torch.softmax(z, dim=1) / self.gumbel_temp
        return applied_softmax[:, 1]

    def __call__(self):
        g = self.gumbel_sample()[:, None]
        mean = g * self.w_mu
        var = g * self.w_s2 + (1-g) * self.s2_w
        return g * (mean + torch.sqrt(var) * torch.randn(self.dim, 1))


    def kl_divergence(self):

        esp = 1e-20

        g = self.gumbel_sample()[:, None]
        mean = g * self.w_mu
        var = g * self.w_s2 + (1-g) * self.s2_w
        w =  mean + torch.sqrt(var) * torch.randn(self.dim, 1)
        mean = mean.detach()
        var = var.detach()
        entropy = 0.5 * torch.sum(torch.log(2. * np.pi * var)+ 0.5 * torch.square(w-mean) / var) \
                  - torch.sum(g * torch.log(self.s_pi + esp) + (1.-g) * torch.log(1. - self.s_pi + esp))

        log_prior = - 0.5 * self.dim * torch.log(2. * np.pi * self.s2_w) \
                    - 0.5 * torch.sum(torch.square(w)) / self.s2_w + \
                    torch.sum(g * torch.log(self.pi_w + esp) + (1. - g) * torch.log(1. - self.pi_w + esp))

        return - log_prior - entropy


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
