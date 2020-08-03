import math
import numpy as np
import torch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel
from src.structural_sgp import VariationalGP, StructuralSparseGP
from src.mean_field_hs import MeanFieldHorseshoe, VariatioalHorseshoe
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
import matplotlib.pyplot as plt

# toy data
train_x = torch.linspace(0, 1, 100)
train_y = 3.*torch.cos(train_x * 2 * math.pi) + torch.randn(100).mul(train_x.pow(3) * 1.)

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

# declare Horseshoe object
# horseshoe = MeanFieldHorseshoe(n_dims=n_kernels, A=1.)
horseshoe = VariatioalHorseshoe(A=1., B=1., n_inducings=[n_inducing]*n_kernels)

# main model
model = StructuralSparseGP(gps, horseshoe)

likelihood = GaussianLikelihood()
elbo = VariationalELBO(likelihood, model, num_data=100)


output = model(train_x)
kl = model.variational_strategy.kl_divergence()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for i in range(500):
    optimizer.zero_grad()
    output = model(train_x)
    loss = - elbo(output, train_y)
    loss.backward()
    optimizer.step()
    print("Iter: {} \t Loss: {:.2f}".format(i, loss.item()))
    if i%10 == 0:
        print(model.horseshoe())
    model.horseshoe.update_ab()

print(model.horseshoe())
print(torch.mean(output.mean - train_y)**2)
print(train_y)
print(output.mean)

plt.plot(train_x, train_y, '+')
plt.plot(train_x, output.mean.detach().numpy())
lower, upper = output.confidence_region()
plt.fill_between(train_x.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.3)
print(lower)
print(upper)
plt.show()
