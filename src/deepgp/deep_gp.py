import gpflow
import tensorflow as tf
from gpflow.models import BayesianModel
from gpflow.models.model import MeanAndVariance

from .likelihood import BroadcastingLikelihood

from gpflow.models.training_mixins import ExternalDataTrainingLossMixin

class DeepGPBase(BayesianModel, ExternalDataTrainingLossMixin):

    def __init__(self, likelihood, layers, num_samples=1, num_data=None, **kwargs):
        super().__init__(name="DeepGPBase")
        self.num_data = num_data
        self.num_samples = num_samples
        self.likelihood = BroadcastingLikelihood(likelihood)
        self.layers = layers

    def propagate(self, X, full_cov=False, num_samples=1, zs=None):
        sX = tf.tile(tf.expand_dims(X, 0), [num_samples, 1, 1])
        Fs, Fmeans, Fvars = [], [], []
        F = sX
        zs = zs or [None, ] * len(self.layers)
        for layer, z in zip(self.layers, zs):
            F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)
        return Fs, Fmeans, Fvars

    def predict_f(self, predict_at, num_samples, full_cov=False) -> MeanAndVariance:
        Fs, Fmeans, Fvars = self.propagate(predict_at, full_cov=full_cov,
                                           num_samples=num_samples)
        return Fmeans[-1], Fvars[-1]

    def predict_all_layers(self, predict_at, num_samples, full_cov=False):
        return self.propagate(predict_at, full_cov=full_cov,
                              num_samples=num_samples)

    def predict_y(self, predict_at, num_samples):
        Fmean, Fvar = self.predict_f(predict_at, num_samples=num_samples,
                                     full_cov=False)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    def predict_log_density(self, data, num_samples):
        Fmean, Fvar = self.predict_f(data[0], num_samples=num_samples,
                                     full_cov=False)
        l = self.likelihood.predict_density(Fmean, Fvar, data[1])
        log_num_samples = tf.math.log(tf.cast(self.num_samples, gpflow.base.default_float()))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

    def expected_data_log_likelihood(self, X, Y):
        F_mean, F_var = self.predict_f(X, num_samples=self.num_samples,
                                       full_cov=False)
        var_exp = self.likelihood.variational_expectations(F_mean, F_var, Y)  # Shape [S, N, D]
        return tf.reduce_mean(var_exp, 0)  # Shape [N, D]

    def elbo(self, data):

        X, Y = data
        num_data = X.shape[0]
        likelihood = tf.reduce_sum(self.expected_data_log_likelihood(X, Y))
        if self.num_data is not None:
            scale = tf.cast(num_data, gpflow.default_float())
            scale /= tf.cast(X.shape[0], gpflow.default_float())
        else:
            scale = tf.cast(1., gpflow.default_float())
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        return scale * likelihood - KL

    def maximum_log_likelihood_objective(self, data) -> tf.Tensor:
        return self.elbo(data)