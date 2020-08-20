import tensorflow as tf
import gpflow as gf
import numpy as np
from gpflow.kernels import RBF
from gpflow.mean_functions import Zero
from gpflow.models import SVGP
from gpflow.optimizers import NaturalGradient
from  gpflow.likelihoods import Gaussian, GaussianMC
from src.sparse_selector_tf import TrivialSparseSelector, SpikeAndSlabSelector, HorseshoeSelector, LinearSelector
from src.structural_sgp_tf import StructuralSVGP
from gpflow import set_trainable

# tf.executing_eagerly()


import matplotlib.pyplot as plt

# toy data
train_x = tf.linspace(0., 1., 100)[:, None]
train_y = 3. * tf.cos(train_x * 2. * np.pi) + tf.random.normal((100,1)) * train_x ** 3

train_x = tf.cast(train_x, tf.float64)
train_y = tf.cast(train_y, tf.float64)

n_kernels = 5
gps = []
for i in range(n_kernels):
    kernel = RBF()
    set_trainable(kernel.variance, False)
    mean = Zero()
    Z = train_x[:50]
    gp = SVGP(kernel=kernel, mean_function=mean, likelihood=None, inducing_variable=Z)
    gps += [gp]

def create_linear_data():

    np.random.seed(123)
    N = 100
    sparsity = 0.05
    M = 200
    beta = np.zeros(M + 1)
    b1 = np.random.binomial(n=1, p=sparsity, size=M)
    b2 = np.random.binomial(n=1, p=0.5, size=M)
    for m in range(M):
        if b1[m]:
            if b2[m]:
                beta[m] = 10 + np.random.randn()
            else:
                beta[m] = -10 + np.random.randn()
        else:
            beta[m] = 0.25 * np.random.randn()

    beta[M] = 0.
    X_train = np.random.randn(N, M + 1)
    X_train[:, M] = 1
    y_train = np.matmul(X_train, beta) + np.random.randn(N)
    return M, N, X_train, y_train, beta


def test_trivial_model():
    selector = TrivialSparseSelector(n_kernels)
    test_gp(selector)
    return
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood)

    minibatch_size = 100
    train_dataset = tf.data.Dataset.\
        from_tensor_slices((train_x, train_y)).\
        repeat().\
        shuffle(len(train_y))
    train_iter = iter(train_dataset.batch(minibatch_size))
    opt = tf.optimizers.Adam()

    train_loss = model.training_loss_closure(train_iter)

    @tf.function
    def optimize_step():
        opt.minimize(train_loss, model.trainable_variables)

    for i in range(5000):
        optimize_step()
        if i % 10 == 0:
            print("Iter {}\t Loss{}".format(i, train_loss().numpy()))


    test_x = tf.linspace(-0.1, 1.1, 100)[:,None]
    test_x = tf.cast(test_x, dtype=tf.float64)
    mean, var = model.predict_f(test_x)

    plt.plot(train_x, train_y, "kx", mew=2)
    plt.plot(test_x, mean, "C0", lw=2)
    plt.fill_between(
        test_x[:, 0],
        mean[:, 0] - 1.96 * np.square(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:,0]),
        color='C0',
        alpha= 0.2
    )
    plt.show()

def test_spike_and_slab():

    ss = SpikeAndSlabSelector(dim=5)
    print(ss.entropy())
    print(ss.log_prior())
    print(ss.kl_divergence())
    print(ss.sample())

def test_horseshoe():
    horseshoe = HorseshoeSelector(dim=5)
    print(horseshoe.entropy())
    print(horseshoe.log_prior())
    print(horseshoe.kl_divergence())
    print(horseshoe.sample())


def test_linear_regression_spike_and_slab():

    M, N, X_train, y_train, beta = create_linear_data()
    spike_and_slab = SpikeAndSlabSelector(dim=M+1)

    def loss_closure():
        s2 = spike_and_slab.s2
        w = spike_and_slab.sample()
        y_mean = X_train @ w
        ll = -0.5 * N * tf.math.log(2. * np.pi * s2) - 0.5 * tf.reduce_sum(tf.square(y_train - tf.squeeze(y_mean))) / s2
        kl = spike_and_slab.kl_divergence()
        return - (ll - kl)



    optimizer = tf.optimizers.Adam()

    @tf.function
    def optimize_step():
        optimizer.minimize(loss_closure, list(spike_and_slab.trainable_variables))

    for i in range(20000):
        optimize_step()
        if i % 10 == 0:
            print("Iter: {} \t Loss: {}".format(i, loss_closure().numpy()))

    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(M), beta[:-1], \
            linewidth=3, color="black", label="ground truth")
    ax.scatter(np.arange(M), beta[:-1], \
               s=70, marker='+', color="black")
    w = spike_and_slab.w_mean * spike_and_slab.prob
    w = w.numpy()
    ax.plot(np.arange(M), w[:-1], \
            linewidth=3, color="red", \
            label="linear model with spike and slab prior")
    ax.set_xlim([0, M - 1])
    ax.set_ylabel("Slopes", fontsize=18)
    ax.hlines(0, 0, M - 1)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.legend(prop={'size': 14})

    fig.set_tight_layout(True)
    plt.show()


def test_linear_regression_horseshoe():

    M, N, X_train, y_train, beta = create_linear_data()
    horseshoe = HorseshoeSelector(dim=M+1)

    def loss_closure():
        s2 = horseshoe.s2
        w = horseshoe.sample()
        y_mean = X_train @ w
        ll = -0.5 * N * tf.math.log(2. * np.pi * s2) - 0.5 * tf.reduce_sum(tf.square(y_train - tf.squeeze(y_mean))) / s2
        kl = horseshoe.kl_divergence()
        return - (ll - kl)

    optimizer = tf.optimizers.Adam(lr=0.01)

    @tf.function
    def optimize_step():
        optimizer.minimize(loss_closure, list(horseshoe.trainable_variables))

    for i in range(20000):
        optimize_step()
        horseshoe.update_tau_lambda()
        if i % 10 == 0:
            print("Iter: {} \t Loss: {}".format(i, loss_closure().numpy()))

    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(M), beta[:-1], \
            linewidth=3, color="black", label="ground truth")
    ax.scatter(np.arange(M), beta[:-1], \
               s=70, marker='+', color="black")
    w = horseshoe.sample()
    w = w.numpy()
    ax.plot(np.arange(M), w[:-1], \
            linewidth=3, color="red", \
            label="linear model with spike and slab prior")
    ax.set_xlim([0, M - 1])
    ax.set_ylabel("Slopes", fontsize=18)
    ax.hlines(0, 0, M - 1)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().set_visible(False)
    ax.legend(prop={'size': 14})

    fig.set_tight_layout(True)
    plt.show()

def test_gp_natural_gradient(selector, func=None):

    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood)
    from gpflow import set_trainable
    natgrad_params = []
    for gp in model.gps:
        set_trainable(gp.q_mu, False)
        set_trainable(gp.q_sqrt, False)
        natgrad_params.append((gp.q_mu, gp.q_sqrt))

    adam_opt = tf.optimizers.Adam(lr=0.01)
    natgrad_opt = NaturalGradient(gamma=0.1)

    minibatch_size = 100
    train_dataset = tf.data.Dataset. \
        from_tensor_slices((train_x, train_y)). \
        repeat(). \
        shuffle(len(train_y))
    train_iter = iter(train_dataset.batch(minibatch_size))

    train_loss = model.training_loss_closure(train_iter)

    @tf.function
    def adam_step():
        adam_opt.minimize(train_loss, model.trainable_variables)

    @tf.function
    def natgrad_step():
        natgrad_opt.minimize(train_loss, var_list=natgrad_params)

    for i in range(10000):
        adam_step()
        natgrad_step()
        if func is not None:
            func()
        if i % 10 == 0:
            print("Iter {}\t Loss {:.3f}".format(i, tf.squeeze(train_loss()).numpy()))

    test_x = tf.linspace(-0.1, 1.1, 100)[:, None]
    test_x = tf.cast(test_x, dtype=tf.float64)
    mean, var = model.predict_f(test_x)

    plt.plot(train_x, train_y, "kx", mew=2)
    plt.plot(test_x, mean, "C0", lw=2)
    plt.fill_between(
        test_x[:, 0],
        mean[:, 0] - 1.96 * np.square(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color='C0',
        alpha=0.2
    )
    plt.show()

def test_gp(selector, func=None):

    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data=100)

    minibatch_size = 100
    train_dataset = tf.data.Dataset. \
        from_tensor_slices((train_x, train_y)). \
        repeat(). \
        shuffle(len(train_y))
    train_iter = iter(train_dataset.batch(minibatch_size))
    opt = tf.optimizers.Adam(lr=0.01)

    train_loss = model.training_loss_closure(train_iter)

    @tf.function
    def optimize_step():
        opt.minimize(train_loss, model.trainable_variables)

    for i in range(20000):
        optimize_step()
        if func is not None:
            func()
        if i % 100 == 0:
            print("Iter {}\t Loss {:.3f}".format(i, tf.squeeze(train_loss()).numpy()))

    test_x = tf.linspace(-0.1, 1.1, 100)[:, None]
    test_x = tf.cast(test_x, dtype=tf.float64)
    mean, var = model.predict_f(test_x)

    plt.plot(train_x, train_y, "kx", mew=2)
    plt.plot(test_x, mean, "C0", lw=2)
    plt.fill_between(
        test_x[:, 0],
        mean[:, 0] - 1.96 * np.square(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color='C0',
        alpha=0.2
    )
    plt.show()

def test_gp_double_opt(selector, func=None):
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data=100)

    minibatch_size = 100
    train_dataset = tf.data.Dataset. \
        from_tensor_slices((train_x, train_y)). \
        repeat(). \
        shuffle(len(train_y))
    train_iter = iter(train_dataset.batch(minibatch_size))
    opt_gp = tf.optimizers.Adam(lr=0.1)
    opt_selector = tf.optimizers.Adam(lr=0.01)

    selector_variables = selector.trainable_variables
    gp_variable = []
    for gp in model.gps:
        gp_variable.append(gp.trainable_variables)
    gp_variable.append(likelihood.trainable_variables)

    train_loss = model.training_loss_closure(train_iter)


    @tf.function
    def optimize_step():
        opt_gp.minimize(train_loss, gp_variable)
        opt_selector.minimize(train_loss, selector_variables)

    for i in range(10000):
        optimize_step()
        if func is not None:
            func()
        if i % 10 == 0:
            print("Iter {}\t Loss {:.3f}".format(i, tf.squeeze(train_loss()).numpy()))

    test_x = tf.linspace(-0.1, 1.1, 100)[:, None]
    test_x = tf.cast(test_x, dtype=tf.float64)
    mean, var = model.predict_f(test_x)

    plt.plot(train_x, train_y, "kx", mew=2)
    plt.plot(test_x, mean, "C0", lw=2)
    plt.fill_between(
        test_x[:, 0],
        mean[:, 0] - 1.96 * np.square(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color='C0',
        alpha=0.2
    )
    plt.show()


def test_gp_spike_and_slab():
    selector = SpikeAndSlabSelector(n_kernels, gumbel_temp=0.5)
    test_gp(selector)


def test_gp_horseshoe():
    selector = HorseshoeSelector(n_kernels)
    def func():
        selector.update_tau_lambda()

    test_gp(selector, func)


# test_trivial_model()
# test_spike_and_slab()
# test_horseshoe()
# test_linear_regression_spike_and_slab()
# test_linear_regression_horseshoe()
# test_gp_spike_and_slab()
test_gp_horseshoe()

## Natural gradient -> not work
# selector = HorseshoeSelector(n_kernels)
# def func():
#     selector.update_tau_lambda()
# test_gp_natural_gradient(selector, func)

# selector = SpikeAndSlabSelector(n_kernels)
# test_gp_double_opt(selector)

# selector = LinearSelector(n_kernels)
# test_gp(selector)
