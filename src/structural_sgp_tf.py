from typing import List

import tensorflow as tf
from gpflow.models import SVGP, BayesianModel
from gpflow.models.model import MeanAndVariance, RegressionData, InputData
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin

from src.sparse_selector_tf import BaseSparseSelector


class StructuralSVGP(BayesianModel, ExternalDataTrainingLossMixin):

    def __init__(self, gps: List[SVGP], selector: BaseSparseSelector, likelihood, num_data=None):

        super(StructuralSVGP, self).__init__(name="structrural_gp")
        self.gps = gps
        self.selector = selector
        self.likelihood = likelihood
        self.num_data = num_data

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def prior_kl(self):
        kls = [gp.prior_kl() for gp in self.gps]
        kls.append(tf.squeeze(self.selector.kl_divergence()))
        return sum(kls)

    def elbo(self, data: RegressionData):
        X, Y = data
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        # f_var = tf.reshape(tf.linalg.diag_part(f_var), shape=tf.shape(f_mean))
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        kl = self.prior_kl()
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, dtype=kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], dtype=kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1., dtype=kl.dtype)

        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:

        w = self.selector.sample()
        w = tf.squeeze(w)
        means = []
        vars = []
        for i, gp in enumerate(self.gps):
            w_i = w[i]
            # w_i = truncate_small(w_i)
            mean, var = gp.predict_f(Xnew, full_cov, full_output_cov)
            means += [mean * w_i]
            w2_i = w_i ** 2
            w2_i = tf.clip_by_value(w2_i, clip_value_min=1e-2, clip_value_max=1e3)
            vars += [var * w2_i]

        f_mean = sum(means)
        f_var = sum(vars)
        return f_mean, f_var

    def predict_y(self, Xnew, full_cov=False, full_output_cov=False):
        f_mean, f_var = self.predict_f(Xnew, full_cov, full_output_cov)
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

def truncate_small(x, eps=1e-2):

    pos = 0.5 * (1. + tf.sign(x - eps)) * x
    neg = 0.5 * (1 + tf.sign(-x - eps)) * x
    return pos - neg


