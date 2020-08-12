import os
import torch
import gpytorch

from torch.utils.data import TensorDataset, DataLoader
from gpytorch.models.deep_gps import DeepGP
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel
from gpytorch.means import ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from src.deep_structral_sgp import DeepStructuralLayer
from src.structural_sgp import StructuralSparseGP
from src.sparse_selector import TrivialSelector, SpikeAndSlabSelector, HorseshoeSelector


class DeepStructuralGP(DeepGP):

    def __init__(self, input_dims, intermedia_dims, gp1, gp2):
        hidden_layer = DeepStructuralLayer(input_dims=input_dims, output_dims=intermedia_dims, gp=gp1)
        last_layer = DeepStructuralLayer(input_dims=intermedia_dims, output_dims=None, gp=gp2)

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs):
        hidden = self.hidden_layer(inputs)
        output = self.last_layer(hidden)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


class VariationalGP(ApproximateGP):

    def __init__(self, mean, kernel, input_dims, output_dims, num_inducing=128):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_dist = CholeskyVariationalDistribution(num_inducing_points=num_inducing)
        variational_strat = VariationalStrategy(self, inducing_points, variational_dist)
        super().__init__(variational_strat)
        self.mean_module = mean
        self.mean_module.batch_shape = batch_shape
        self.covar_module = kernel
        self.covar_module.batch_shape = batch_shape

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

def create_data():
    import urllib.request
    import scipy.io as sio
    data_file = "../data/elevators.mat"
    if not os.path.isfile(data_file):
        urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', data_file)

    data = torch.Tensor(sio.loadmat(data_file)['data'])
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]

    n_train = int(0.8 * len(X))
    x_train = X[:n_train, :].contiguous()
    y_train = y[:n_train].contiguous()

    x_test = X[n_train:, :].contiguous()
    y_test = y[n_train:].contiguous()

    return x_train, y_train, x_test, y_test


def create_spike_and_slab_gp(input_dims, output_dims, num_inducing=128, num_kernels=5):

    means = [ZeroMean() for _ in range(num_kernels)]
    kernels = [RBFKernel() for _ in range(num_kernels)]

    gps = []
    for mean, kernel in zip(means, kernels):
        gp = VariationalGP(mean=mean,
                           kernel=kernel,
                           input_dims=input_dims,
                           output_dims=output_dims,
                           num_inducing=num_inducing)
        gps += [gp]

    selector = SpikeAndSlabSelector(dim=num_kernels)

    gp = StructuralSparseGP(gps=gps, selector=selector)

    return gp


def run_deep_gp(n_iter=500, lr=0.01):
    x_train, y_train, x_test, y_test = create_data()

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=128, shuffle=False)


    input_dims = x_train.size(-1)
    intermediate_dims = 5

    gp1 = create_spike_and_slab_gp(input_dims=input_dims, output_dims=intermediate_dims)
    gp2 = create_spike_and_slab_gp(input_dims=intermediate_dims, output_dims=None)

    deep_gp = DeepStructuralGP(input_dims=input_dims,
                               intermedia_dims=intermediate_dims,
                               gp1=gp1,
                               gp2=gp2)

    optimizer = torch.optim.Adam(deep_gp.parameters(), lr=lr)
    base_mll = VariationalELBO(likelihood=deep_gp.likelihood,
                               model=deep_gp,
                               num_data=x_train.shape[-2])
    mll = DeepApproximateMLL(base_mll)

    for i in range(n_iter):
        for x_batch, y_batch in train_loader:
            with gpytorch.settings.num_likelihood_samples(1):
                optimizer.zero_grad()
                output = deep_gp(x_batch)
                loss = - mll(output, y_batch)
                loss.backward()
                optimizer.step()

                print("Iter: {} \t Loss: {:.2f}".format(i, loss.item()))


run_deep_gp()






