import torch
import math
from torch.nn import Module
from torch.distributions import HalfCauchy, Normal


class ReparameterizedGaussian():

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    @property
    def std_dev(self):
        return torch.log1p(self.var.exp())

    def sample(self, n_samples=1):
        epsilon = Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
        return self.mean + self.std_dev * epsilon

    def logprob(self, target):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.std_dev)
                - ((target - self.mean) ** 2) / (2 * self.std_dev ** 2)).sum()

    def entropy(self):
        if self.mean.dim() > 1:
            n_inputs, n_outputs = self.mean.shape
        else:
            n_inputs = len(self.mean)
            n_outputs = 1

        part1 = (n_inputs * n_outputs) / 2 * (torch.log(torch.tensor([2 * math.pi])) + 1)
        part2 = torch.sum(torch.log(self.std_dev))

        return part1 + part2


class InverseGamma(object):

    def __init__(self, shape, rate):
        self.shape = shape
        self.rate = rate

    def expect_inverse(self):
        """Compute \mathbb{E}[1/x] of IG(x; shape, rate)
        """
        return self.shape / self.rate

    def expect_log(self):
        """Compute \mathbb{E}[\log(x)]"""
        return torch.log(self.rate) - torch.digamma(self.shape)

    def entropy(self):
        entropy = self.shape + torch.log(self.rate) + torch.lgamma(self.shape) \
                  - (1 + self.shape) * torch.digamma(self.shape)
        return torch.sum(entropy)

    def update(self, new_shape, new_rate):
        self.shape = new_shape
        self.rate = new_rate


def expectation_log_IG(alpha, expect_beta, expect_log_beta, expect_log_x, expect_inverse_x):
    """
    Inverse Gamma with pdf: \IG(x;\alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} (1/x)^{\alpha +1} \exp(-\beta / x)

    The expectation is taken over variational distribution where some expectation quantities are given.

    :param alpha: shape
    :param expect_beta: expectation of beta over variational distribution
    :param expect_log_beta: expectation of logarithm of beta over variational distribution
    :param expect_log_x: expectation of x over variational dist.
    :param expect_inverse_x: expectation of inverse of x over variation dist.
    :return:
    """
    term1 = - torch.lgamma(alpha)
    term2 = alpha * expect_log_beta
    term3 = - (alpha + 1) * expect_log_x
    term4 = -expect_beta * expect_inverse_x
    return term1 + term2 + term3 + term4


class VariationalHalfCauchy(Module):
    """Variational Half Cauchy with reprarameterization using Inverse Gamma

    p(\lambda_i^2) = Half-Cauchy(0, b), i = 1,...,n

    The equivalent form:
    p(\lambda_i^2) = Inv-Gamma(1/2, 1/\kappa_i)
    p(\kappa_i) = Inv-Gamma(1/2, 1/b^2)

    The variational distribution
    q(\lambda_i^2, \kappa_i) = log-Normal(\mu, \sigma) p(\kappa_i)
    """

    def __init__(self, n_dim, b):
        super().__init__()
        self.n_dim = n_dim
        self.b = b

        # prior and variatoinal over kappa
        # note that there is no variational distribution
        self.kappa_shape = torch.Tensor([0.5] * self.n_dim)
        self.kappa_rate = torch.Tensor([1. / self.b ** 2] * n_dim)
        self.kappa = InverseGamma(self.kappa_shape, self.kappa_rate)

        # variational over lambda
        dist = HalfCauchy(b)
        sample = dist.sample(sample_shape=torch.Size([self.n_dim]))
        self.register_parameter("lambda_mean", torch.nn.Parameter(torch.log(sample)))
        self.register_parameter("lambda_var", torch.nn.Parameter(-torch.ones_like(sample)))
        self.log_lambda = ReparameterizedGaussian(self.lambda_mean, self.lambda_var)

    def expect_log_prior(self):
        # \mathbb{E}_{\lambda_i, \kappa_i}[IG(\lambda_i^2| 1/2, 1/kappa_i)]
        alpha = self.kappa_shape  # 1/2
        expect_beta = self.kappa.expect_inverse()
        expect_log_beta = self.kappa.expect_log()
        expect_log_x = 0
        expect_inverse_x = 0

        expect1 = expectation_log_IG(alpha=alpha,
                                     expect_beta=expect_beta,
                                     expect_log_beta=expect_log_beta,
                                     expect_log_x=expect_log_x,
                                     expect_inverse_x=expect_inverse_x)

        # \mathbb{E}_{\kappa_i}[IG(\kappa_i| 1/2, 1/b^2)]
        alpha = self.kappa_shape
        expect_beta = self.kappa_rate
        expect_log_beta = torch.log(self.kappa_rate)
        expect_log_x = 0
        expect_inverse_x = 0

        expect2 = expectation_log_IG(alpha=alpha,
                                     expect_beta=expect_beta,
                                     expect_log_beta=expect_log_beta,
                                     expect_log_x=expect_log_x,
                                     expect_inverse_x=expect_inverse_x)

        return expect1 + expect2

    def entropy(self):
        log_normal_entropy = self.log_lambda.entropy() + torch.sum(self.log_lambda.mean)
        gamma_entropy = self.kappa.entropy()
        return log_normal_entropy + gamma_entropy

    def update_kappa(self):
        """Fixed point update for Kappa"""
        new_shape = torch.Tensor([1.] * self.n_dim)
        new_rate = torch.exp(-self.log_lambda.mean + 0.5 * self.log_lambda.std_dev ** 2) + self.kappa_rate
        self.kappa.update(new_shape, new_rate)

    def forward(self):
        """Return a sample of log-normal from variational dist of lambda^2"""
        log_lambda_sample = self.log_lambda.sample()  # A sample from normal dist
        return log_lambda_sample.exp()  # A sample from log-normal dist


class VariationalHorseshoe(Module):

    def __init__(self, n_dim, b_0, b_g):
        super().__init__()
        self.n_dim = n_dim
        self.b_0 = b_0
        self.b_g = b_g

        self.local_shrinkage = VariationalHalfCauchy(n_dim=self.n_dim, b=self.b_0)
        self.global_shrinkage = VariationalHalfCauchy(n_dim=1, b=self.b_g)

    def forward(self):
        # \lambda_i^2
        var_local = self.local_shrinkage.forward()
        # \tau^2
        var_global = self.global_shrinkage.forward()
        # a Gaussian sample \mathcal(N)(0, \lambda_i^2\tau^2)
        var = var_local * var_global

        # # regularized horseshoe
        c2 = 40
        lambda_tilde = torch.sqrt(c2 * var_local / (c2 + var))

        # return Normal(0,1).sample(sample_shape=var.size()) * var_global.sqrt() * lambda_tilde
        return Normal(0, 1).sample(sample_shape=var.size()) * var.sqrt()

    def expect_log_prior(self):
        return self.local_shrinkage.expect_log_prior() + self.global_shrinkage.expect_log_prior()

    def entropy(self):
        return self.local_shrinkage.entropy() + self.global_shrinkage.entropy()

    def kl_divergence(self):
        return self.expect_log_prior() - self.entropy()

    def update(self):
        self.local_shrinkage.update_kappa()
        self.global_shrinkage.update_kappa()
