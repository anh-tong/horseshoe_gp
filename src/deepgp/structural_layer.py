import tensorflow as tf
import gpflow
from .layers import Layer
from src.sparse_selector_tf import BaseSparseSelector

from typing import List

class StructuralSVGPLayer(Layer):

    def __init__(self, gps: List[Layer], selector: BaseSparseSelector, output_dim, input_dim=None, mean_function=None):
        super().__init__(input_dim)
        self.output_dim = output_dim
        self.gps = gps
        self.selector = selector
        self.mean_function = mean_function

    def conditional_ND(self, X, full_cov=False):

        w = self.selector.sample()
        f_means = []
        f_vars = []
        for i, gp in enumerate(self.gps):
            mean, var = gp.conditional_ND(X, full_cov=full_cov)
            f_means += [mean * w[i]]
            f_vars += [var * w[i] ** 2]

        f_mean = tf.add_n(f_means)
        f_var = tf.add_n(f_vars)

        mean_X = self.mean_function(X)
        return f_mean + mean_X, f_var

    def KL(self):
        w_kl = self.selector.kl_divergence()
        gp_kl = [gp.KL() for gp in self.gps]
        return w_kl + tf.add_n(gp_kl)



