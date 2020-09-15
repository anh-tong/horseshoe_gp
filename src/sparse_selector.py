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

