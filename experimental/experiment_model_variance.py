import tensorflow as tf
import numpy as np
from gpflow.kernels import RBF, Combination, Periodic, White
from src.sparse_selector_tf import BaseSparseSelector, HorseshoeSelector
from gpflow.models import SVGP
from gpflow.likelihoods import Gaussian, Bernoulli
import tensorflow_probability as tfp
from src.structural_sgp_tf import StructuralSVGP
from gpflow import default_float
from gpflow.config import set_default_jitter
import sys

set_default_jitter(1e-3)

class StochasticKernel(Combination):

    def __init__(self, kernels, selector):
        super().__init__(kernels, name="stochastic_kernel")
        self.selector = selector
        self.set_w()
        assert len(kernels) == self.selector.dim


    def set_w(self):
        w = self.selector.sample()
        self.w = w

    def K(self, X, X2=None):
        w = self.w#tf.squeeze(self.selector.sample())
        ret = []
        for i in range(len(self.kernels)):
            w_i = w[i]
            K_i = self.kernels[i].K(X, X2)
            ret += [K_i * w_i ** 2]
        return tf.add_n(ret)

    def K_diag(self, X):
        w = self.w#tf.squeeze(self.selector.sample())
        ret = []
        for i in range(len(self.kernels)):
            w_i = w[i]
            K_i = self.kernels[i].K_diag(X)
            ret += [K_i * w_i ** 2]
        return tf.add_n(ret)


class NaiveModel(SVGP):

    def __init__(self, kernel: StochasticKernel, likelihood, inducing_points, num_data, n_sample=1):
        super(NaiveModel, self).__init__(kernel=kernel,
                                         likelihood=likelihood,
                                         inducing_variable=inducing_points,
                                         num_data=num_data)
        self.n_sample = n_sample

    def elbo(self, data) -> tf.Tensor:
        if self.n_sample > 1:
            ret = []
            for _ in range(self.n_sample):
                ret += [super(NaiveModel, self).elbo(data)]
            ret = tf.reduce_mean(tf.concat(ret, axis=0))
        else:
            ret = super().elbo(data)
        ret -= self.kernel.selector.kl_divergence()
        return ret


    def predict_y(self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        self.kernel.set_w()
        return super(NaiveModel, self).predict_y(Xnew, full_cov, full_output_cov)




def init_inducing_point(x, M):
    return x[:M]

def create_model(kernels, selector, inducing_points, num_data):
    stochastic_kernel = StochasticKernel(kernels, selector)
    likelihood = Gaussian(variance=1e-2)
    svgp = NaiveModel(kernel=stochastic_kernel,
                likelihood=likelihood,
                inducing_points=inducing_points,
                num_data=num_data,
                )
    return svgp

def create_our_model(kernels, selector, inducing_points, num_data):

    gps = []
    for kernel in kernels:
        gp = SVGP(kernel, likelihood=None, inducing_variable=inducing_points)
        gps += [gp]
    likelihood = Gaussian(variance=1e-2)
    model = StructuralSVGP(gps=gps, likelihood=likelihood, selector=selector, num_data=num_data)
    return model

def create_kernels():
    kernels = [RBF(variance=0.01, lengthscales=0.1),
               RBF(lengthscales=2.),
               Periodic(base_kernel=RBF(), period=0.1),
               Periodic(base_kernel=RBF(), period=2.)]
    return kernels


def train(data, model, n_iter=1000, lr=0.1, freq=1):

    optimizer = tf.optimizers.Adam(lr=lr)
    train_loss = model.training_loss_closure(data)

    @tf.function
    def optimize_step():
        optimizer.minimize(train_loss, model.trainable_variables)

    X, Y = data
    @tf.function
    def get_var_exp():
        f_mean, f_var = model.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = model.likelihood.variational_expectations(f_mean, f_var, Y)
        return tf.reduce_sum(var_exp)

    ret = []
    for i in range(n_iter):
        optimize_step()
        if isinstance(model, StructuralSVGP):
            model.selector.update_tau_lambda()
        else:
            model.kernel.selector.update_tau_lambda()

        if i % freq == 0:
            var_exp = -tf.squeeze(get_var_exp())
            print("Iter {} \t Variational expectation: {:.2f}".format(i, var_exp))
            ret += [var_exp.numpy()]
    ret = np.array(ret)
    return ret



def postprocessing():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    n_sample = 1
    choices = ["naive", "ours"]
    for choice in choices:
        ids = []
        iters = []
        elbos = []
        for id in range(1,11):
            save_path = "../model/elbo/{}_{}_{}.npy".format(choice, id, n_sample)
            data = np.load(save_path)
            for iter, value in enumerate(data[:100]):
                ids += [id]
                iters+= [iter]
                elbos += [value]

        ids = np.array(ids)[:, None]
        iters = np.array(iters)[:, None]
        elbos = np.array(elbos)[:, None]
        collected = np.hstack([ids, iters, elbos])
        collected = pd.DataFrame(data=collected, columns=["id", "iter", "ELBO"])
        sns.lineplot(x="iter", y="ELBO", data=collected)

    plt.show()






if __name__ == "__main__":

    load = True
    if load:
        postprocessing()
        exit(0)

    choose = sys.argv[1]
    id = sys.argv[2]
    if len(sys.argv) >= 4:
        n_sample = int(sys.argv[3])
    else:
        n_sample = 1

    # choose = "naive"
    # id = 1
    #choose = "ours"

    M = 50
    # num data
    N = 200
    x = tf.linspace(0., 5., N)[:, None]
    x = tf.cast(x, dtype=default_float())
    y = tf.math.cos(2*x) + 3*tf.math.cos(0.5 * x**2) + 0.2*tf.random.normal((N,1), dtype=default_float())

    inducing_points = init_inducing_point(x, M)




    kernels = create_kernels()
    selector = HorseshoeSelector(len(kernels))
    if choose == "naive":
        model = create_model(kernels, selector, inducing_points, N)
    else:
        model = create_our_model(kernels, selector, inducing_points, N)

    ret = train((x, y), model, n_iter=100, lr=0.01)
    save_path = "../model/elbo/{}_{}_{}.npy".format(choose, id, n_sample)
    np.save(save_path, ret)
    print("save file to {}".format(save_path))
