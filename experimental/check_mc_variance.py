import torch
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.kernels import RBFKernel, PeriodicKernel
from matplotlib import rc
rc('font', **{'family':'serif'})
rc('text', usetex=True)


def create_kernels():
    k1 = RBFKernel()
    k1.lengthscale = 0.1
    k2 = RBFKernel()
    k2.lengthscale = 2.
    k3 = PeriodicKernel()
    k3.lengthscale = 1.
    k3.period_length = 0.1
    k4 = PeriodicKernel()
    k4.period_length = 2.
    kernels = [k1, k2, k3, k4]

    x = torch.linspace(0, 5, 50)

    Kx = [k(x).evaluate() for k in kernels]
    return Kx


def compute_decompose(kernels, weights):

    Ls = []
    for k, w in zip(kernels, weights):
        L = w*torch.cholesky_inverse(k)
        Ls += [L]

    return torch.norm(sum(Ls))

def compute_nondecompose(kernels, weights):
    Ks = []
    for k, w in zip(kernels, weights):
        Ks += [w *k]

    L = torch.cholesky_inverse(sum(Ks) + torch.eye(50) * 1e-3)

    return torch.norm(L)


def sample_weight(prob):
    return torch.bernoulli(prob)


def simulation(kernels, prob, func, n_sims=1000):

    record = []
    for i in range(n_sims):
        weights = sample_weight(prob)
        fx = func(kernels, weights)
        record += [fx]

    record = torch.tensor(record)
    return record.mean()  # return MCMC estimate


def replicate(*args, **kwargs):

    n_replicate = 100
    record = []
    for i in range(n_replicate):
        mc_estimate = simulation(*args, **kwargs)
        record += [mc_estimate]

    record = torch.tensor(record)
    return record.std()


def test_compute_decompose():
    kernels = [torch.eye(3), torch.eye(3)]
    weights = [torch.tensor([1.]), torch.tensor([1.])]
    print(compute_decompose(kernels, weights))


def test_compute_nondecompose():
    kernels = [torch.eye(3), torch.eye(3)]
    weights = [torch.tensor([1.]), torch.tensor([1.])]
    print(compute_nondecompose(kernels, weights))

# test_compute_decompose()
# test_compute_nondecompose()

kernels = create_kernels()
prob = torch.tensor([0.2, 0.4, 0.8, 0.5])

list_n_sims = [100, 1000, 10000]
# load or not
load = True

if not load:
    var_decompose = []
    var_nondecompose = []
    for n_sims in list_n_sims:
        error_variance = replicate(kernels, prob, func=compute_decompose, n_sims=n_sims)
        var_decompose += [error_variance.numpy()]
        print("Decomposed\t num simulations: {} \t Variance {}".format(n_sims, error_variance.numpy()))
        error_variance = replicate(kernels, prob, func=compute_nondecompose, n_sims=n_sims)
        print("Non decomposed\t num simulations: {} \t Variance {}".format(n_sims, error_variance.numpy()))
        var_nondecompose += [error_variance.numpy()]

    var_decompose = np.array(var_decompose)
    np.save('var_decompose.npy', var_decompose)
    var_nondecompose = np.array(var_nondecompose)
    np.save('var_nondecompose.npy', var_nondecompose)
else:
    var_decompose = np.load('var_decompose.npy')
    var_nondecompose = np.load('var_nondecompose.npy')

fig, ax = plt.subplots(1,1, figsize=(5,5))
font = {'family': 'serif', 'weight': 'normal', 'size': 18}

ax.plot(list_n_sims, var_decompose, marker='.', lw=4., markersize=16., label="Decompose")
ax.plot(list_n_sims, var_nondecompose, marker='.', lw=4., markersize=16., label="No Decompose")
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.xlabel("Number of MC sample", font)
plt.ylabel("STD of MC estimator", font)

fig.savefig("comparing_mcmc_std.png", dpi=300, bbox_inches='tight')
fig.savefig("comparing_mcmc_std.pdf", dpi=300, bbox_inches='tight')

plt.show()
