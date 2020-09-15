import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import Module
from gpflow.base import Parameter
from gpflow.utilities import positive, to_default_float

from src.distributions import Normal, LogNormal, InverseGamma


class BaseSparseSelector(Module):

    def __init__(self, dim):
        super().__init__(name="selector")
        self.dim = dim

    def kl_divergence(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class TrivialSparseSelector(BaseSparseSelector):

    def __init__(self, dim):
        super().__init__(dim)

    def kl_divergence(self):
        return tf.zeros(1, dtype=tf.float64)

    def sample(self):
        return tf.ones(self.dim, dtype=tf.float64)


class LinearSelector(BaseSparseSelector):

    def __init__(self, dim):
        super().__init__(dim)
        self.w = Parameter(tf.random.normal((self.dim, 1)))


    def kl_divergence(self):
        return tf.zeros(1, dtype=tf.float64)

    def sample(self):
        return self.w

class SpikeAndSlabSelector(BaseSparseSelector):

    def __init__(self, dim, gumbel_temp=1.):
        super(SpikeAndSlabSelector, self).__init__(dim)
        self.gumbel_temp = gumbel_temp
        prob_transform = tfp.bijectors.Sigmoid()
        self.prob = Parameter(0.5 * tf.ones((self.dim, 1)), transform=prob_transform)
        self.w_mean = Parameter(tf.random.normal((self.dim, 1)))
        self.w_var = Parameter(tf.ones((self.dim, 1)), transform=positive())
        self.var_zero = Parameter(tf.ones((self.dim, 1)), transform=positive())
        sparsity_transform = tfp.bijectors.Sigmoid()
        self.sparsity = Parameter(0.5 * tf.ones((1, 1)), transform=sparsity_transform)
        self.s2 = Parameter(tf.ones((1, 1)), transform=positive())

    def entropy(self, eps=1e-20):
        entropy = - self.prob * tf.math.log(self.prob + eps) \
                  - (1. - self.prob) * tf.math.log(1 - self.prob + eps) \
                  - 0.5 * self.prob * tf.math.log(2. * np.pi * self.w_var)
        return tf.reduce_sum(entropy)

    def log_prior(self, eps=1e-20):
        log_prior_gaussian = -0.5 * tf.math.log(2. * np.pi * self.var_zero) \
                             - 0.5 * (tf.square(self.w_mean) + self.w_var) / self.var_zero
        log_prior_gaussian = self.prob * log_prior_gaussian
        log_prior_bern = (self.prob * tf.math.log(self.sparsity + eps)) \
                         + (1. - self.prob) * tf.math.log(1. - self.prob + eps)
        log_prior = tf.reduce_sum(log_prior_gaussian) + tf.reduce_sum(log_prior_bern)
        return log_prior

    def kl_divergence(self):
        return - self.entropy() - self.log_prior()

    def gumbel_sample(self, eps=1e-20):
        bar_prob = 1. - self.prob
        stacked = tf.concat([bar_prob, self.prob], axis=1)
        uniform = tf.random.uniform((self.dim, 2))
        uniform = to_default_float(uniform)
        z = tf.math.log(-tf.math.log(uniform + eps) + eps) + tf.math.log(stacked)
        applied_softmax = tf.nn.softmax(z / self.gumbel_temp, axis=1)
        return applied_softmax[:, 1]

    def sample(self):
        g = self.gumbel_sample()
        g = tf.expand_dims(g, axis=-1)
        mean = g * self.w_mean
        var = g * self.w_var + (1. - g) * self.var_zero
        ret = mean + tf.sqrt(var) * to_default_float(tf.random.normal((self.dim, 1)))
        ret = g * ret
        return ret


class HorseshoeSelector(BaseSparseSelector):

    def __init__(self, dim, A=1., B=1.):
        super().__init__(dim)

        # tau
        self.m_tau = Parameter(tf.random.normal([1, 1]))
        self.var_tau = Parameter(tf.ones([1, 1]),
                                 transform=positive())
        self.q_tau = LogNormal(mean=self.m_tau,
                               var=self.var_tau)
        self.q_phi_tau = InverseGamma(shape=0.5 * tf.ones([1, 1]),
                                      rate=tf.convert_to_tensor(A))

        # lambda
        self.m_lambda = Parameter(tf.random.normal([self.dim, 1]))
        self.var_lambda = Parameter(tf.ones([self.dim, 1]),
                                    transform=positive())
        self.q_lambda = LogNormal(mean=self.m_lambda,
                                  var=self.var_lambda)
        self.q_phi_lambda = InverseGamma(shape=0.5 * tf.ones([self.dim, 1]),
                                         rate=tf.convert_to_tensor(B))


    def entropy(self):
        entropy_tau = self.q_tau.entropy()
        entropy_lambda = self.q_lambda.entropy()
        entropy_lambda = tf.reduce_sum(entropy_lambda)
        # TODO: there is entropy for phi but it does not effect optimization.Should I include it?
        return entropy_tau + entropy_lambda

    def log_prior(self):

        def log_density_inverse_gamma(x: LogNormal, alpha, beta: InverseGamma):
            """log PDF of IG(x; alpha, 1/beta)"""
            ret = - alpha * beta.expect_log() - tf.math.lgamma(alpha) \
                  - (alpha + 1) * x.expect_log_x() - beta.expect_inverse() * x.expect_inverse()
            return ret
        half = to_default_float(0.5)
        log_prior_tau = log_density_inverse_gamma(self.q_tau, half, self.q_phi_tau)
        log_prior_lambda = log_density_inverse_gamma(self.q_lambda, half, self.q_phi_lambda)

        # TODO: Like entropy function, there is log prior for phi variables

        return log_prior_tau + tf.reduce_sum(log_prior_lambda)

    def kl_divergence(self):
        return - self.entropy() - self.log_prior()

    def update_tau_lambda(self):
        new_tau_shape = tf.ones((1, 1))
        new_tau_shape = to_default_float(new_tau_shape)
        new_tau_rate = self.q_tau.expect_inverse() + 1.
        new_tau_rate = to_default_float(new_tau_rate)
        self.q_phi_tau.update(new_tau_shape, new_tau_rate)
        new_lambda_shape = tf.ones((self.dim, 1))
        new_lambda_shape = to_default_float(new_lambda_shape)
        new_lambda_rate = self.q_lambda.expect_inverse() + 1.
        new_lambda_rate = to_default_float(new_lambda_rate)
        self.q_phi_lambda.update(new_lambda_shape, new_lambda_rate)

    def sample(self):
        mean = self.q_tau.mean + self.q_lambda.mean
        var = self.q_tau.var + self.q_lambda.var
        log_tau_lambda = mean + tf.sqrt(var) * tf.random.normal(shape=tf.shape(var), dtype=tf.float64)
        return tf.exp(0.5 * log_tau_lambda)

