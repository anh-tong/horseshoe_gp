import numpy as np
import tensorflow as tf

from gpflow.kernels import RBF, Periodic, Linear, Product
from gpflow.models import SVGP
from gpflow.likelihoods import Gaussian
from src.kernel_generator_tf import Periodic2
from src.structural_sgp_tf import StructuralSVGP
from src.sparse_selector_tf import HorseshoeSelector, SpikeAndSlabSelector
from src.kernel_generator_tf import Generator
from src.utils import get_dataset
from gpflow import set_trainable


def load_data(name="airline"):
    dataset = get_dataset(name)
    return dataset


def init_inducing_points(x, M=100):
    return x[:M]


def make_data_iteration(x, y, batch_size=128, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.repeat().shuffle(len(y))

    data_iter = iter(dataset.batch(batch_size))
    return data_iter


def fix_kernel_variance(kernels):
    """kernel variance is fixed. the variance is decided by selector"""
    for kernel in kernels:
        if isinstance(kernel, Product):
            fix_kernel_variance(kernel.kernels)
        elif isinstance(kernel, Periodic2):
            set_trainable(kernel.base_kernel.variance, False)
        else:
            set_trainable(kernel.variance, False)


def create_model(inducing_point, num_data) -> StructuralSVGP:
    generator = Generator()
    # kernels = generator.create_upto(2)
    kernels = [RBF(),
               Periodic2(),
               Product([RBF(), Periodic2()]),
               RBF(),
               Periodic2(),
               Product([RBF(), Periodic2()])]

    fix_kernel_variance(kernels)
    gps = []
    for kernel in kernels:
        gp = SVGP(kernel, likelihood=None, inducing_variable=inducing_point)
        gps.append(gp)

    selector = HorseshoeSelector(dim=len(gps))
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data)
    return model


def train(model, train_iter, n_iter=10000, lr=0.01):
    optimizer = tf.optimizers.Adam(lr=lr)

    train_loss = model.training_loss_closure(train_iter)

    @tf.function
    def optimize_step():
        optimizer.minimize(train_loss, model.trainable_variables)

    for i in range(n_iter):
        optimize_step()
        # additional update for the horseshoe case
        if isinstance(model.selector, HorseshoeSelector):
            model.selector.update_tau_lambda()

        if i % 100 == 0:
            print("Iter {} \t Loss: {:.2f}".format(i, train_loss().numpy()))

    return model


def test(test_iter, model: StructuralSVGP):
    mus = []
    vars = []
    # lls = []
    for x_batch, y_batch in test_iter:
        pred_mean, pred_var = model.predict_f(x_batch, full_cov=False, full_output_cov=False)
        # ll = model.likelihood.log_prob(pred_mean, pred_var, y_batch)
        mus.append(pred_mean)
        vars.append(pred_var)
        # lls.append(ll)

    mu = tf.concat(mus, axis=0)
    var = tf.concat(vars, axis=0)
    # ll = tf.concat(lls, axis=0)
    return mu, var, None


def plot(x, y, x_prime, y_prime, upper, lower):
    import matplotlib.pyplot as plt
    plt.plot(x, y, "+")
    plt.plot(x_prime, y_prime)
    plt.fill_between(x_prime.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2)
    plt.show()


def run(name="airline", batch_size=128):

    # data
    dataset = load_data(name)
    x_train, y_train = dataset.get_train()
    train_iter = make_data_iteration(x_train, y_train, batch_size=batch_size)
    x_test, y_test = dataset.get_test()
    test_iter = make_data_iteration(x_test, y_test, batch_size=batch_size, shuffle=False)

    inducing_point = init_inducing_points(x_train)

    # create model
    model = create_model(inducing_point, num_data=len(y_train))

    # train
    model = train(model, train_iter, n_iter=50000, lr=0.01)

    # predict
    mu, var, ll = test(test_iter, model)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(mu - y_test)))
    # mean_ll = tf.reduce_mean(ll)

    print("RMSE: {} ".format(rmse.numpy()))

    # plot for 1D case
    n_test = 300
    x_test = tf.linspace(tf.reduce_min(x_train), tf.reduce_max(x_train), n_test)[:, None]
    y_test = tf.zeros_like(x_test)
    plot_iter = make_data_iteration(x_test, y_test, shuffle=False)
    mu, var, _ = test(plot_iter, model)
    lower = mu - 1.96 * tf.sqrt(var)
    upper = mu + 1.96 * tf.sqrt(var)

    plot(x_train, y_train, x_test.numpy(), mu.numpy(), lower.numpy(), upper.numpy())


if __name__ == "__main__":
    # run(name="solar") # TODO: have a problem running this
    # run(name="airline")
    # run(name="mauna")
    run(name="wheat-price")


