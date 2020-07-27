import numpy as np
import torch
from gpytorch.kernels import RBFKernel, PeriodicKernel
from src.mean_field_hs import InverseGammaReparam
from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

# fix plot Mac OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def uniform_sample(lower, upper):
    return lower + np.random.rand() * (upper - lower)

n_kernels = 50
n_points = 150
n_samples = 3

x = torch.linspace(0,10, n_points)

n_rbfs = 25
rbfs = [RBFKernel() for _ in range(n_rbfs)]

for k in rbfs:
    k.lengthscale = 0.5

n_periodics = 25
periodics = [PeriodicKernel() for _ in range(n_periodics)]

for k in periodics:
    k.period = 0.5

rbfs.extend(periodics)

kernels = rbfs

K_x = [k(x).evaluate() for k in kernels]

tau = InverseGammaReparam(shape=1., n_dims=1)
lambda_i = InverseGammaReparam(shape=1. * torch.ones(n_kernels), n_dims=n_kernels)
lambda_i.rate = 0.01 * torch.ones(n_kernels)

for _ in range(n_samples):
    sample_tau = tau().detach()
    sample_lambda_i = lambda_i().detach().squeeze()

    print(sample_lambda_i)

    sum_K = []
    for i, K in enumerate(K_x):
        coeff = sample_tau * sample_lambda_i[i]
        sum_K.append(coeff * K)


    covar_weighted = sum(sum_K)
    covar_weighted = covar_weighted + torch.eye(n_points) * 0.1

    dist = MultivariateNormal(mean=torch.zeros(n_points), covariance_matrix=covar_weighted)

    sample = dist.sample()

    plt.plot(x, sample)




covar_uniform = sum(K_x)
covar_uniform = covar_uniform + torch.eye(n_points) * 0.01

dist = MultivariateNormal(mean=torch.zeros(n_points), covariance_matrix=covar_uniform)

plt.subplots()
for _ in range(n_samples):
    sample = dist.sample()
    plt.plot(x, sample)

plt.show()

