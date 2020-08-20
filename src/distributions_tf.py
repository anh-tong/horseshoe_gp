import numpy as np
import tensorflow as tf
from gpflow.utilities import to_default_float
from gpflow.config import default_float


class InverseGamma(object):

    def __init__(self, shape, rate):
        self.shape = to_default_float(shape)
        self.rate = to_default_float(rate)

    def expect_inverse(self):
        """Compute \mathbb{E}[1/x] of IG(x; shape, rate)
        """
        return self.shape / self.rate

    def expect(self):
        return self.rate / (self.shape - 1.)

    def expect_log(self):
        """Compute \mathbb{E}[\log(x)]"""
        return tf.math.log(self.rate) - tf.math.digamma(self.shape)

    def entropy(self):
        entropy = self.shape + tf.math.log(self.rate) + tf.math.lgamma(self.shape) \
                  - (1 + self.shape) * tf.math.digamma(self.shape)
        return tf.reduce_sum(entropy)

    def kl_divergence(self):
        return self.entropy() - self.expect_log()

    def update(self, new_shape, new_rate):
        self.shape = new_shape
        self.rate = new_rate


class LogNormal(object):

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def expect_log_x(self):
        return self.mean

    def entropy(self):
        return self.mean + tf.math.log(self.var) + 0.5 * (np.log(2. * np.pi) + 1)

    def expect_inverse(self):
        return tf.exp(-self.mean + 0.5 * self.var)

    def sample(self):
        s = self.mean + tf.sqrt(self.var) * tf.random.normal(shape=tf.shape(self.var), dtype=default_float())
        return tf.exp(s)


class Normal(object):

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def entropy(self):
        return 0.5 * (tf.math.log(self.var) + np.log(2. * np.pi) + 1)

    def log_prior(self):
        """Expectation of log Normal(0, 1)"""
        ret = -0.5 * np.log(2. * np.pi) - 0.5 * (tf.square(self.mean) + self.var)
        return ret

    def kl_divergence(self):
        return - self.log_prior() - self.entropy()

    def sample(self):
        sqrt_var = tf.sqrt(self.var)
        return self.mean + sqrt_var * tf.random.normal(shape=tf.shape(self.var), dtype=default_float())
