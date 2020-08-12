import torch
import numpy as np

from torch.autograd import Variable
from gpytorch import Module
from gpytorch.constraints import Interval, Positive
from src.distributions import Normal, LogNormal, InverseGamma


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

    def __init__(self, dim, gumbel_temp=0.1):
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
        self.register_parameter(name="raw_scale", parameter=torch.nn.Parameter(torch.zeros(1)))
        # Only for linear model
        self.register_parameter(name="raw_s2", parameter=torch.nn.Parameter(torch.zeros(1)))
        s2_constraint = Positive()
        self.register_constraint("raw_s2", s2_constraint)

    @property
    def prob(self):
        return self.raw_prob_constraint.transform(self.raw_prob)

    @property
    def sparsity_prior(self):
        return self.raw_sparsity_constraint.transform(self.raw_sparsity)

    @property
    def scale_prior(self):
        # return torch.tensor([1.])
        return torch.exp(self.raw_scale)

    @property
    def s2(self):
        return self.raw_s2_constraint.transform(self.raw_s2)

    def entropy(self, eps=1e-20):
        entropy = - self.prob * torch.log(self.prob + eps) - \
                  (1. - self.prob) * torch.log(1. - self.prob + eps) \
                  - 0.5 * self.prob * (torch.log(2. * np.pi * torch.exp(torch.tensor([1.]))) + self.log_w_var)

        entropy = entropy.sum()
        return entropy

    def log_prior(self, eps=1e-20):
        log_prior_gaussian = -0.5 * (np.log(2. * np.pi) + self.raw_scale) - 0.5 * (
                torch.square(self.w_mean) + torch.exp(self.log_w_var)) / self.scale_prior
        log_prior_gaussian = self.prob * log_prior_gaussian
        log_prior_bern = (self.prob * torch.log(self.sparsity_prior + eps)) + \
                         ((1. - self.prob) * torch.log(1. - self.sparsity_prior + eps))
        log_prior = log_prior_gaussian.sum() + log_prior_bern.sum()
        return log_prior

    def kl_divergence(self):
        entropy = self.entropy()
        log_prior = self.log_prior()
        return -entropy - log_prior

    def gumbel_sample(self, eps=1e-20):
        bar_prob = 1. - self.prob
        stacked = torch.cat([bar_prob, self.prob], dim=1)
        uniform = torch.rand(self.dim, 2)
        z = - Variable(torch.log(-torch.log(uniform + eps) + eps)) + torch.log(stacked)
        applied_softmax = torch.softmax(z / self.gumbel_temp, dim=1)
        return applied_softmax[:, 1]

    def __call__(self):
        g = self.gumbel_sample()[:, None]
        mean = g * self.w_mean
        var = g * torch.exp(self.log_w_var) + (1. - g) * self.scale_prior
        ret = mean + torch.sqrt(var) * torch.randn(self.dim, 1)
        ret = g * ret
        return ret



class HorseshoeSelector(BaseSparseSelector):

    def __init__(self, dim, A, B):
        super().__init__(dim)

        # tau
        self.register_parameter("m_tau", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_parameter("log_var_tau", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.q_tau = LogNormal(self.m_tau, self.log_var_tau)
        self.phi_tau = InverseGamma(shape=torch.tensor(0.5), rate=torch.tensor(A))

        # lambda
        self.register_parameter("m_lambda", parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        self.register_parameter("log_var_lambda", parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        self.q_lambda = LogNormal(self.m_lambda, self.log_var_lambda)
        self.phi_lambda = InverseGamma(shape=torch.ones(self.dim)*0.5, rate=torch.ones(self.dim) * B)

        # w
        self.register_parameter("m_w", parameter=torch.nn.Parameter(torch.randn(self.dim, 1)))
        self.register_parameter("log_var_w", parameter=torch.nn.Parameter(torch.zeros(self.dim, 1)))
        self.q_w = Normal(self.m_w, self.log_var_w)

        # s2 for linear regression
        self.register_parameter("raw_s2", parameter=torch.nn.Parameter(torch.zeros(1)))
        s2_constraint = Positive()
        self.register_constraint("raw_s2", s2_constraint)

    @property
    def s2(self):
        return self.raw_s2_constraint.transform(self.raw_s2)

    def entropy(self):
        # entropy of log Gaussian for tau
        entropy_tau = self.q_tau.entropy()
        # entropy of log Gaussian for lambda
        entropy_lambda = self.q_lambda.entropy()
        entropy_lambda = entropy_lambda.sum()
        # TODO: there is entropy for phi but it does not effect optimization. Should I include it?
        entropy_phi_tau = self.phi_tau.entropy()
        entropy_phi_lambda = self.phi_lambda.entropy()
        return entropy_tau + entropy_lambda


    def log_prior(self):

        def log_density_inverse_gamma(x: LogNormal, alpha, beta:InverseGamma):
            """log PDF of IG(x; alpha, 1/beta)"""
            ret = - alpha * beta.expect_log() - torch.lgamma(alpha)\
                  - (alpha + 1) * x.expect_log_x() - beta.expect_inverse() * x.expect_inverse()
            return ret

        log_prior_tau = log_density_inverse_gamma(self.q_tau, torch.tensor(0.5), self.phi_tau)
        log_prior_lambda = log_density_inverse_gamma(self.q_lambda, torch.tensor(0.5), self.phi_lambda)

        # TODO: Like entropy function, there is log prior for phi variables

        return log_prior_tau + log_prior_lambda.sum()


    def kl_divergence(self):
        w_kl_divergence = self.q_w.kl_divergence().sum()
        return w_kl_divergence - self.entropy() - self.log_prior()

    def update_tau_lambda(self):
        new_shape = torch.tensor(1.)
        new_rate = self.q_tau.expect_inverse() + 1.
        self.phi_tau.update(new_shape, new_rate)
        new_rate_phi_lambda = self.q_lambda.expect_inverse() + 1.
        self.phi_lambda.update(new_shape, new_rate_phi_lambda)

    def __call__(self):
        tau = self.q_tau.sample()
        lamda = self.q_lambda.sample()
        w = self.q_w.sample()
        return w * tau * lamda
