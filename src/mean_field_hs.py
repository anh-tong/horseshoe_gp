import torch
from torch.distributions import Gamma
from gpytorch.constraints import Positive
from gpytorch import Module


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

class MeanFieldHorseshoe(object):

    def __init__(self, n_dims,
                 A,
                 B=1.,
                 init_tau_rate=None,
                 init_tau_shape=None,
                 init_lambda_rate=None,
                 init_lambda_shape=None,):

        init_tau_rate = 1. if init_tau_rate is None else init_tau_rate
        init_tau_shape = 2. if init_tau_shape is None else init_tau_shape
        init_lambda_rate = torch.ones(n_dims) if init_lambda_rate is None else init_lambda_rate
        init_lambda_shape = 2*torch.ones(n_dims) if init_lambda_shape is None else init_lambda_shape

        self.A = A
        self.B = B
        self.n_dims = n_dims

        self.tau2 = InverseGamma(rate=init_tau_rate, shape=init_tau_shape)
        self.a = InverseGamma(0.5, 1 / A**2)
        self.lambda2 = InverseGamma(rate=init_lambda_rate, shape=init_lambda_shape)
        self.lambda2.rate = torch.randn(self.lambda2.rate.size())**2
        self.b = InverseGamma(0.5, 1. / (torch.ones(n_dims)*B**2))


    def update_a(self):
        """Update rate and shape for auxiliary variable a"""
        new_shape = 1.
        new_rate = self.tau2.expect_inverse() + 1 / self.A ** 2
        self.a.update(new_shape, new_rate)

    def update_b(self):
        """Update rate and shape for auxiliary variables b"""
        new_shape = 1.
        new_rate = self.lambda2.expect_inverse() + 1 / self.B ** 2
        self.b.update(new_shape, new_rate)

    def update_tau2(self, n_inducings, trace_term):
        """
        Update rate and shape for tau
        :param n_inducings: Tensor of integers indicates the number of inducing points
        :param trace_term: Dim: n_dims. Value: trace of S_i * K_i^{-1}
        :return:
        """
        new_shape = 0.5 * (sum(n_inducings) + 1)
        new_rate = 0.5 * self.a.expect_inverse() + 0.5 * torch.sum(self.lambda2.expect_inverse() * trace_term)
        self.tau2.update(new_shape=new_shape, new_rate=new_rate)

    def update_lambda2(self, n_inducings, trace_term):
        """Update rate and shape for lambda"""
        new_shape = 0.5 * (n_inducings + 1)
        new_rate = 0.5 * self.b.expect_inverse() + 0.5 * self.tau2.expect_inverse() * trace_term
        self.lambda2.update(new_shape=new_shape, new_rate=new_rate)

    def update_all(self, n_inducings, trace_term):
        self.update_a()
        self.update_b()
        self.update_tau2(n_inducings, trace_term)
        self.update_lambda2(n_inducings, trace_term)

    def weight(self):
        return self.tau2.expect() * self.lambda2.expect()

    def inverse_weight(self):
        return self.tau2.expect_inverse() * self.lambda2.expect_inverse()

    def kl_divergence(self):
        # TODO: implement this. do not affect gradient computation
        return self.tau2.kl_divergence() + self.lambda2.kl_divergence()

class InverseGammaReparam(Module):

    def __init__(self, shape, n_dims, trainable=True):
        super().__init__()
        self.n_dims = n_dims
        self.trainable = trainable
        self.register_buffer("shape", tensor=torch.as_tensor(shape))
        if self.trainable:
            self.register_parameter("raw_rate", torch.nn.Parameter(torch.zeros(torch.Size([self.n_dims]))))
            self.register_constraint("raw_rate", Positive())
        else:
            self.register_buffer("raw_rate", tensor=torch.ones(torch.Size([self.n_dims])))
        self.base_gamma = Gamma(concentration=self.shape, rate=torch.ones_like(self.shape))

    @property
    def rate(self):
        if self.trainable:
            return self.raw_rate_constraint.transform(self.raw_rate)
        else:
            return self.raw_rate

    @rate.setter
    def rate(self, value):
        if self.trainable:
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.raw_rate)
            self.initialize(raw_rate=self.raw_rate_constraint.inverse_transform(value))
        else:
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(self.rate)
            self.raw_rate = value

    def expect_inverse(self):
        return self.shape / self.rate

    def expect(self):
        return self.rate / (self.shape - 1)

    def expect_log(self):
        return torch.log(self.rate) - torch.digamma(self.shape)

    def entropy(self):
        entropy = self.shape + torch.log(self.rate) + torch.lgamma(self.shape) \
                  - (1 + self.shape) * torch.digamma(self.shape)
        return torch.sum(entropy)

    def expect_log_prior(self, a):
        """Compute expect of log prior w.r.t. to variational distribution"""
        expect_log_a = a.expect_log()
        expect_inverse_a = a.expect_inverse()
        term1 = -0.5 * expect_log_a
        term2 = -1.5 * self.expect_log()
        term3 = -expect_inverse_a * self.expect_inverse()
        term4 = -torch.digamma(torch.as_tensor(0.5))
        return term1 + term2 + term3 + term4


    def forward(self):
        """Sample using reprameterization trick"""
        eps = self.base_gamma.sample(sample_shape=(1,))
        return self.rate / eps

    def update(self, new_shape, new_rate):
        if not self.trainable:
            self.shape = new_shape
            self.rate = new_rate
        else:
            raise RuntimeError("Trainble -> No hard update. Gradient update only")

class VariatioalHorseshoe(Module):

    def __init__(self, A, B, n_inducings):
        super().__init__()
        self.n_dims = len(n_inducings)
        # Half-Cauchy parameter
        self.A, self.B = A, B
        # shrinkage parameters
        shape_tau = 0.5 * (sum(n_inducings) + 1)
        self.variational_tau = InverseGammaReparam(shape=shape_tau, n_dims=1)
        shape_lambda = [0.5 * (n_i + 1.) for n_i in n_inducings]
        self.variational_lambda = InverseGammaReparam(shape=shape_lambda, n_dims=len(n_inducings))
        self.a = InverseGammaReparam(0.5, n_dims=1, trainable=False)
        self.a.rate = 1 / self.A ** 2
        self.b = InverseGammaReparam(0.5 * torch.ones(len(n_inducings)), n_dims=len(n_inducings), trainable=False)
        self.b.rate = torch.ones(len(n_inducings)) / self.B ** 2

    def forward(self):
        return self.variational_tau() * self.variational_lambda()

    def expect(self):
        return self.variational_tau.expect() * self.variational_lambda.expect()

    def expect_inverse(self):
        return self.variational_tau.expect_inverse() * self.variational_lambda.expect_inverse()

    def update_ab(self):
        a_new_shape = torch.as_tensor(1.)
        a_new_rate = self.variational_tau.expect_inverse() + 1. / self.A**2
        self.a.update(a_new_shape, a_new_rate)
        b_new_shape = torch.ones_like(self.b.shape)
        b_new_rate = self.variational_lambda.expect_inverse() + 1/ self.B**2
        self.b.update(b_new_shape, b_new_rate)

    def kl_divergence(self):
        entropy = self.variational_tau.entropy() + self.variational_lambda.entropy()
        expect_log_prior_tau = self.variational_tau.expect_log_prior(self.a)
        expect_log_prior_lambda = sum(self.variational_lambda.expect_log_prior(self.b))
        ret = expect_log_prior_tau + expect_log_prior_lambda - entropy
        return ret


def sanity_check_inverse_gamma():

    ig = InverseGammaReparam(shape=torch.ones(2)*8 + 1., n_dims=2)
    ig.rate = torch.randn(2)**2
    print("shape:", ig.shape)
    print("rate: ", ig.rate)
    print("expectation: ", ig.expect())
    n_samples = 100000
    samples = [ig() for _ in range(n_samples)]
    mean = sum(samples) / n_samples
    print("mean: ", mean)
    print("expect log: ", ig.expect_log())
    print("entropy: ", ig.entropy())
    print("expect_log_prior: ", ig.expect_log_prior(InverseGammaReparam(0.5, n_dims=2, trainable=False)))

def sanity_check_horseshoe():
    hs = VariatioalHorseshoe(1, 2, [3,3])
    sample = hs()
    print(sample)
    print(hs.kl_divergence())
    hs.update_ab()
    print(hs.a.rate, hs.a.shape)
    print(hs.b.rate, hs.b.shape)

# sanity_check_inverse_gamma()
# sanity_check_horseshoe()