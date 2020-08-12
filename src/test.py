import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood

from src.structural_sgp import VariationalGP, StructuralSparseGP

from src.sparse_selector import TrivialSelector, SpikeAndSlabSelector, HorseshoeSelector

# toy data
train_x = torch.linspace(0, 1, 100)
train_y = 3. * torch.cos(train_x * 2 * math.pi) + torch.randn(100).mul(train_x.pow(3) * 1.)

# set up kernels
n_kernels = 5
means = [ZeroMean()] * n_kernels
kernels = [RBFKernel()] * n_kernels

n_inducing = 50
inducing_points = torch.linspace(0, 1, n_inducing)

# GP for each kernel
gps = []
for mean, kernel in zip(means, kernels):
    gp = VariationalGP(mean, kernel, inducing_points)
    gps.append(gp)


def test_trivial_model():
    selector = TrivialSelector(n_kernels)
    # main model
    model = StructuralSparseGP(gps, selector)

    likelihood = GaussianLikelihood()
    elbo = VariationalELBO(likelihood, model, num_data=100)

    output = model(train_x)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)

    for i in range(500):
        optimizer.zero_grad()
        output = model(train_x)
        loss = - elbo(output, train_y)
        loss.backward()
        optimizer.step()
        print("Iter: {} \t Loss: {:.2f}".format(i, loss.item()))

    print(torch.mean(output.mean - train_y) ** 2)
    print(train_y)
    print(output.mean)
    plt.plot(train_x, train_y, '+')
    plt.plot(train_x, output.mean.detach().numpy())
    lower, upper = output.confidence_region()
    plt.fill_between(train_x.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.3)
    print(lower)
    print(upper)
    plt.show()


def test_object_spike_and_slab():
    ss = SpikeAndSlabSelector(5)
    sample = ss()
    print(sample)
    print(
        ss.kl_divergence())  ## KL divergence equal to 0 makes sense because the variational dist. and prior dist. is the same


def test_object_horseshoe():
    horseshoe = HorseshoeSelector(dim=5, A=1., B=1.)
    print("entropy", horseshoe.entropy())
    print("log prior", horseshoe.log_prior())
    print("kl", horseshoe.kl_divergence())
    # print("", horseshoe.q_w.kl_divergence())
    print(horseshoe())


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
    X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
    return M, N, X_train, y_train, beta


def test_linear_regression_spike_and_slab():
    ### CREATE DATA
    M, N, X_train, y_train, beta = create_linear_data()

    spike_and_slab = SpikeAndSlabSelector(dim=M + 1)

    def compute_loss():
        s2 = spike_and_slab.s2
        w = spike_and_slab()

        def log_likelihood(w):
            y_mean = torch.matmul(X_train, w)
            ll = - 0.5 * N * torch.log(2. * np.pi * s2) - 0.5 * torch.square(y_train - y_mean.squeeze()).sum() / s2
            return ll

        ll = log_likelihood(w)
        kl = spike_and_slab.kl_divergence()
        # elbo = log_likelihood(w) - spike_and_slab.kl_divergence()
        return ll, kl

    optimizer = torch.optim.Adam(spike_and_slab.parameters(), lr=0.005)

    for i in range(10000):
        optimizer.zero_grad()
        ll, kl = compute_loss()
        loss = - (ll - kl)
        loss.backward()
        optimizer.step()
        print("Iter: {} \t Loss: {} \t ll: {} \t kl: {} \t noise: {}".format(i,
                                                                             loss.item(),
                                                                             ll.item(),
                                                                             kl.item(),
                                                                             spike_and_slab.s2.data))

    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(M), beta[:-1], \
            linewidth=3, color="black", label="ground truth")
    ax.scatter(np.arange(M), beta[:-1], \
               s=70, marker='+', color="black")
    w = spike_and_slab.w_mean * spike_and_slab.prob
    w = w.detach().numpy()
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
    fig.savefig('foo.png')
    plt.show()


def test_linear_regression_horseshoe():

    M, N, X_train, y_train, beta = create_linear_data()
    horseshoe = HorseshoeSelector(dim=M+1, A=1., B=1.)

    optimizer = torch.optim.Adam(list(horseshoe.parameters()), lr=0.01)


    def compute_loss():
        horseshoe.update_tau_lambda()
        s2 = horseshoe.s2
        w = horseshoe()
        y_mean = X_train @ w
        ll = - 0.5 * N * torch.log(2. * np.pi * s2) \
             - 0.5 * torch.square(y_train - y_mean.squeeze()).sum() / s2
        kl = horseshoe.kl_divergence()
        return ll, kl
    for i in range(10000):
        optimizer.zero_grad()
        ll, kl = compute_loss()
        loss = - (ll - kl)
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Iter: {} \t Loss: {} \t ll: {} \t kl: {} \t noise: {}".format(i,
                                                                             loss.item(),
                                                                             ll.item(),
                                                                             kl.item(),
                                                                             horseshoe.s2.data))
    fig = plt.figure(figsize=(16, 8))

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(M), beta[:-1], \
            linewidth=3, color="black", label="ground truth")
    ax.scatter(np.arange(M), beta[:-1], \
               s=70, marker='+', color="black")
    w = horseshoe()
    w = w.detach().numpy()
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
    fig.savefig('foo.png')
    plt.show()


def test_gp_spike_and_slab():

    selector = SpikeAndSlabSelector(dim=n_kernels, gumbel_temp=.5)
    model = StructuralSparseGP(gps, selector)

    likelihood = GaussianLikelihood()
    elbo = PredictiveLogLikelihood(likelihood, model, num_data=100)

    output = model(train_x)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)

    for i in range(5000):
        optimizer.zero_grad()
        output = model(train_x)
        loss = - elbo(output, train_y)
        loss.backward()
        optimizer.step()
        print("Iter: {} \t Loss: {:.2f}".format(i, loss.item()))

    print(torch.mean(output.mean - train_y) ** 2)
    print(train_y)
    print(output.mean)
    plt.plot(train_x, train_y, '+')
    plt.plot(train_x, output.mean.detach().numpy())
    lower, upper = output.confidence_region()
    plt.fill_between(train_x.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.3)
    plt.show()

def test_gp_horseshoe():

    selector = HorseshoeSelector(dim=n_kernels, A=1., B=1.)
    model = StructuralSparseGP(gps, selector)

    likelihood = GaussianLikelihood()
    elbo = PredictiveLogLikelihood(likelihood, model, num_data=100)

    output = model(train_x)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)

    for i in range(5000):
        optimizer.zero_grad()
        selector.update_tau_lambda()
        output = model(train_x)
        loss = - elbo(output, train_y)
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Iter: {} \t Loss: {:.2f}".format(i, loss.item()))

    print(torch.mean(output.mean - train_y) ** 2)
    print(train_y)
    print(output.mean)
    plt.plot(train_x, train_y, '+')
    plt.plot(train_x, output.mean.detach().numpy())
    lower, upper = output.confidence_region()
    plt.fill_between(train_x.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.3)
    plt.show()


# test_trivial_model()
# test_object_spike_and_slab()
# test_linear_regression_spike_and_slab()

# test_gp_spike_and_slab()


# test_object_horseshoe()
# test_linear_regression_horseshoe()
test_gp_horseshoe()