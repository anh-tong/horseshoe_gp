"""
Regression experiment

Data set:
    - UCI
    - ABCD data set
    - Large scale data set
"""
import numpy as np
import torch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel, ProductKernel, LinearKernel, AdditiveKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import PredictiveLogLikelihood
from src.structural_sgp import StructuralSparseGP, VariationalGP
from src.sparse_selector import HorseshoeSelector, SpikeAndSlabSelector, TrivialSelector
from src.kernel_generator import Generator
from torch.utils.data import TensorDataset, DataLoader

def load_data():

    # TODO: create a data set factory to construct data set object. Let's start with hard-coded data set
    import scipy.io as sio
    from sklearn.model_selection import train_test_split
    data = sio.loadmat("../data/01-airline.mat")
    x = data["X"].astype(np.float)
    y = data["y"].astype(np.float)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=123)

    def to_tensor(*inputs):
        for input in inputs:
            yield torch.tensor(input, dtype=torch.float32)

    x_train, x_test, y_train, y_test = to_tensor(x_train, x_test, y_train.squeeze(), y_test.squeeze())
    return x_train, y_train, x_test, y_test


def make_loader(x, y, batch_size=128, shuffle=True):
    dataset = TensorDataset(x, y)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def init_inducing_point(train_loader:DataLoader, n_inducing=128):
    # TODO: k-mean for high-dimension data

    # for 1D data
    x = train_loader.dataset.tensors[0]
    inducing_point = torch.linspace(x.min(), x.max(), n_inducing)
    return inducing_point

def create_svgp_model(inducing_points):

    class SVGP(VariationalGP):

        def __init__(self, mean, kernel, likelihood, inducing_points):
            super().__init__(mean, kernel, inducing_points)
            self.likelihood = likelihood


    mean = ZeroMean()
    kernel = AdditiveKernel(RBFKernel(), PeriodicKernel())
    likelihood = GaussianLikelihood()
    model = SVGP(mean, kernel, likelihood, inducing_points)

    return model

def train_svgp(train_loader, n_iter=1000, lr=0.01, save_dir=None):

    inducing_points = init_inducing_point(train_loader, n_inducing=128)
    model = create_svgp_model(inducing_points)
    elbo = PredictiveLogLikelihood(model.likelihood, model, num_data=len(train_loader.dataset))
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)

    model.train()

    for i in range(n_iter):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = - elbo(output, y_batch)
            loss.backward()
            optimizer.step()
            print("Iter: {} \t Loss: {:.3f}".format(i, loss.item()))

    # save model
    if save_dir is not None:
        save_model(model, dir="../model/model")  # TODO: correct file name

    return model


def create_model(inducing_points):

    generator = Generator(data_stats=None, base_cls=[RBFKernel])
    kernels = generator.create_upto(upto_order=2)
    kernels = [RBFKernel()] *2
    gps = []
    for kernel in kernels:
        mean = ZeroMean()
        gp = VariationalGP(mean, kernel, inducing_points)  # it's OK to use the initial inducing point for all kernels
        gps.append(gp)
    # selector
    # selector = SpikeAndSlabSelector(dim=len(gps), gumbel_temp=0.5)
    selector = TrivialSelector(dim=len(gps))
    likelihood = GaussianLikelihood()
    model = StructuralSparseGP(gps, selector, likelihood)
    return model


def save_model(model, dir):
    state_dict = model.state_dict()
    torch.save(state_dict, dir)
    print("Save model to {}".format(dir))


def load_model(model, dir):
    if not torch.cuda.is_available():
        state_dict = torch.load(dir, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(dir)
    model.load_state_dict(state_dict)
    print("Load model from {}".format(dir))


def train(train_loader, n_iter=1000, lr=0.01, save_dir=None):

    inducing_points = init_inducing_point(train_loader, n_inducing=128)
    model = create_model(inducing_points)
    elbo = PredictiveLogLikelihood(model.likelihood, model, num_data=len(train_loader.dataset))
    optimizer = torch.optim.Adam(list(model.parameters()), lr=lr)

    model.train()

    for i in range(n_iter):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = - elbo(output, y_batch)
            loss.backward()
            optimizer.step()
            print("Iter: {} \t Loss: {:.3f}".format(i, loss.item()))

    # save model
    if save_dir is not None:
        save_model(model, dir="../model/model") # TODO: correct file name

    return model


def test(test_loader: DataLoader, model: StructuralSparseGP):
    model.eval()
    with torch.no_grad():
        mus = []
        vars = []
        lls = []
        for x_batch, y_batch in test_loader:
            pred = model.likelihood(model(x_batch))
            mus.append(pred.mean)
            vars.append(pred.variance)
            lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        mu, var, ll = torch.cat(mus, dim=-1), torch.cat(vars, dim=-1), torch.cat(lls, dim=-1)

    return mu, var, ll


def plot(x, y, x_prime, y_prime, upper, lower):

    import matplotlib.pyplot as plt
    plt.plot(x, y, "+")
    plt.plot(x_prime, y_prime)
    plt.fill_between(x_prime.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2)
    plt.show()

def run(batch_size=128):

    x_train, y_train, x_test, y_test = load_data()
    train_loader = make_loader(x_train, y_train, batch_size=batch_size)
    test_loader = make_loader(x_test, y_test, shuffle=False)

    # model = train(train_loader, n_iter=500, lr=0.1)
    model = train_svgp(train_loader, n_iter=500, lr=0.01)
    mu, var, ll = test(test_loader, model)

    # prediction
    rmse = torch.mean(torch.pow(mu - y_test, 2)).sqrt()
    print("RMSE: {} \t LL:{}".format(rmse.item(), ll.mean().item()))

    # plot 1D only
    n_test = 300
    x_test = torch.linspace(x_train.min(), x_train.max(), n_test)
    y_test = torch.zeros(n_test) # not gonna use anyway
    plot_loader = make_loader(x_test, y_test, shuffle=False)
    mu, var, _ = test(plot_loader, model)
    lower = mu - 0.5 * var.sqrt()
    upper = mu + 0.5 * var.sqrt()
    plot(x_train.numpy(), y_train.numpy(), x_test.numpy(), mu.numpy(), lower.numpy(), upper.numpy())


run()



