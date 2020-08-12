import torch
import numpy as np

class InverseGamma(object):

    def __init__(self, shape, rate):
        self.shape = shape
        self.rate = rate

    def expect_inverse(self):
        """Compute \mathbb{E}[1/x] of IG(x; shape, rate)
        """
        return self.shape / self.rate

    def expect(self):
        return self.rate / (self.shape -1.)


    def expect_log(self):
        """Compute \mathbb{E}[\log(x)]"""
        return torch.log(self.rate) - torch.digamma(self.shape)

    def entropy(self):
        entropy = self.shape + torch.log(self.rate) + torch.lgamma(self.shape) \
                  - (1 + self.shape) * torch.digamma(self.shape)
        return torch.sum(entropy)

    def kl_divergence(self):
        return self.entropy() - self.expect_log()


    def update(self, new_shape, new_rate):
        self.shape = new_shape
        self.rate = new_rate

class LogNormal(object):

    def __init__(self, mean, log_var):
        self.mean = mean
        self.log_var = log_var

    def expect_log_x(self):
        return self.mean

    def entropy(self):
        return self.mean + self.log_var + 0.5 * (np.log(2. * np.pi) + 1)

    def expect_inverse(self):
        return torch.exp(-self.mean + 0.5 * torch.exp(self.log_var))

    def sample(self):
        var = torch.exp(self.log_var)
        s = self.mean + torch.sqrt(var) * torch.randn_like(var)
        return torch.exp(s)

class Normal(object):

    def __init__(self, mean, log_var):
        self.mean = mean
        self.log_var = log_var

    def entropy(self):
        return 0.5 * (self.log_var + np.log(2. * np.pi) + 1)

    def log_prior(self):
        """Expectation of log Normal(0, 1)"""
        ret = -0.5 * np.log(2. * np.pi) -0.5 * (torch.square(self.mean) + torch.exp(self.log_var))
        return ret

    def kl_divergence(self):
        return - self.log_prior() - self.entropy()

    def sample(self):
        sqrt_var = torch.exp(0.5 * self.log_var)
        return self.mean + sqrt_var * torch.randn_like(sqrt_var)

