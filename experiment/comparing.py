import torch
from torch.nn import ModuleList
import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel, ProductKernel, AdditiveKernel
from gpytorch.models import ExactGP, VariationalGP, ApproximateGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
import matplotlib.pyplot as plt
from matplotlib import rc
import pickle

## plot setting
rc('font', **{'family':'serif'})
rc('text', usetex=True)

## seed setting
torch.manual_seed(0)

def create_data_from_gp():

    k1 = RBFKernel()
    k1.lengthscale = 1.
    k2 = RBFKernel()
    k2.lengthscale = 4.
    k3 = PeriodicKernel()
    k3.period_length = 1.5
    kernel = AdditiveKernel(k1, k2, k3)

    train_x = torch.linspace(0, 5, 100)
    test_x = torch.linspace(-1, 6, 200)
    Kx = kernel(train_x)
    dist = MultivariateNormal(torch.zeros(100), Kx)
    train_y = dist.sample()

    return train_x, train_y, test_x


load = True
if not load:
    train_x, train_y, test_x = create_data_from_gp()
    print("save data to file!")
    with open('synthetic_data.pkl', 'wb') as f:
        pickle.dump([train_x, train_y, test_x], f)
else:
    print("loading data from file")
    with open('synthetic_data.pkl', 'rb') as f:
        train_x, train_y, test_x = pickle.load(f)





class GPRegression(ExactGP):

    def __init__(self, train_x, train_y, kernel, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class VariationalGPRegression(VariationalGP):

    def __init__(self, kernel, inducing_points):
        variational_dist = CholeskyVariationalDistribution(inducing_points.size(-1))
        variational_strategy = VariationalStrategy(self,
                                                   inducing_points,
                                                   variational_dist,
                                                   learn_inducing_locations=True)
        super(VariationalGPRegression, self).__init__(variational_strategy)
        self.mean_module = ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class StructralVariationalStrategy(object):

    def __init__(self, gps):
        self.gps = gps

    def kl_divergence(self):
        return sum([gp.variational_strategy.kl_divergence() for gp in self.gps])


class StructuralGPRegression(ApproximateGP):

    def __init__(self, gps):
        super().__init__(None)
        self.gps = ModuleList(gps)
        self.variational_strategy = StructralVariationalStrategy(gps)

    def __call__(self, inputs, prior=False, **kwargs):
        posteriors = []
        for gp in self.gps:
            posterior = gp(inputs, prior, **kwargs)
            posteriors += [posterior]
        mean = sum([p.mean for p in posteriors])
        covar = sum([p.covariance_matrix for p in posteriors])
        return MultivariateNormal(mean, covar)


def normal_model(train_iter=100):
    k1 = RBFKernel()
    k2 = RBFKernel()
    k3 = PeriodicKernel()
    kernel = AdditiveKernel(k1, k2, k3)
    likelihood = GaussianLikelihood()
    model = GPRegression(train_x, train_y, kernel, likelihood)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(train_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # predict and plot
    # predict and plot
    return predict_and_plot(model, likelihood, test_x)


def variational_model(train_iter=100):
    k1 = RBFKernel()
    k2 = RBFKernel()
    k3 = PeriodicKernel()
    kernel = AdditiveKernel(k1, k2, k3)
    inducing_points = torch.linspace(0, 1, 50)
    likelihood = GaussianLikelihood()
    model = VariationalGPRegression(kernel, inducing_points)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
    mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.numel())

    model.train()
    likelihood.train()

    for _ in range(train_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # predict and plot

    # predict and plot
    return predict_and_plot(model, likelihood, test_x)
    # predict and plot


def independent_inducing(train_iter=100):
    kernels = [RBFKernel(), RBFKernel(), PeriodicKernel()]
    inducing_points = [torch.linspace(0, 1, 50), torch.linspace(0, 1, 50), torch.linspace(0, 1, 50)]

    def make_model(kernels, inducing_points):
        models = []
        for k, inducing_point in zip(kernels, inducing_points):
            model = VariationalGPRegression(k, inducing_point)
            models += [model]
        return models

    gps = make_model(kernels, inducing_points)
    likelihood = GaussianLikelihood()
    model = StructuralGPRegression(gps)
    optimizer = torch.optim.Adam(list(model.parameters()) + list((likelihood.parameters())),
                                 lr=0.01)
    mll = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.numel())

    model.train()
    likelihood.train()

    for _ in range(train_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

    # predict and plot
    return predict_and_plot(model, likelihood, test_x)


def predict_and_plot(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        f_dist = model(test_x)
        mean = f_dist.mean
        covar = f_dist.covariance_matrix
        noise = likelihood.noise
        covar = covar + noise * torch.eye(mean.numel())
        y_dist = likelihood(f_dist)

    return mean, covar, y_dist

def plot_gp(y_dist, color, name):

    fig, ax = plt.subplots(1,1, figsize=(5,3.5))
    mean = y_dist.mean
    lower, upper = y_dist.confidence_region()
    line, = ax.plot(test_x, mean, color=color, lw=1., alpha=0.9)
    ax.fill_between(test_x, lower, upper, color=color, alpha=0.15)
    ax.plot(test_x, lower, '--', color=color, lw=0.5, alpha=0.9)
    ax.plot(test_x, upper, '--', color=color, lw=0.5, alpha=0.9)
    ax.scatter(train_x, train_y, c='k', marker='.', lw=1, alpha=0.9)
    ax.set_xticks([0, 2.5, 5])
    ax.set_yticks([])
    ax.set_xlim([-1, 6])
    fig.savefig(name + ".png", bbox_inches='tight', dpi=300)
    fig.savefig(name + ".pdf", bbox_inches='tight', dpi=300)

def true_model():
    k1 = RBFKernel()
    k1.lengthscale = 1.
    k2 = RBFKernel()
    k2.lengthscale = 4.
    k3 = PeriodicKernel()
    k3.period_length = 1.5
    kernel = AdditiveKernel(k1, k2, k3)
    likelihood = GaussianLikelihood()
    likelihood.noise = 1e-3
    model = GPRegression(train_x, train_y, kernel, likelihood)

    return predict_and_plot(model, likelihood, test_x)



def matrix_sqrt(a):
    values, vectors = a.eig(eigenvectors=True)  # return Nx1 eigenvector, Nx2 eigenvalue (real, image)
    sqrt_values = torch.sqrt(torch.max(values[:, 0], torch.tensor([0.0])))
    return vectors @ (sqrt_values * vectors).t()


def test_matrix_sqrt():
    a = torch.eye(10)
    sqrt_a = matrix_sqrt(a)
    print("pass test")
    print(sqrt_a)


def wasserstein2_distance(m1, m2, S1, S2):
    sqrt_S1 = matrix_sqrt(S1)
    distance = torch.square(m1 - m2).sum() + \
               torch.trace(S1) + \
               torch.trace(S2) - \
               2 * torch.trace(matrix_sqrt(sqrt_S1 @ S2 @ sqrt_S1))

    return distance


def test_wassertein2():
    m1 = torch.zeros(10)
    m2 = torch.zeros(10)
    S1 = torch.eye(10)
    S2 = torch.eye(10)
    distance = wasserstein2_distance(m1, m2, S1, S2)
    print("pass test")
    print(distance)


# test_matrix_sqrt()
#test_wassertein2()
m0, S0, full_gp_dist = normal_model(train_iter=1000)
plot_gp(full_gp_dist, color='blue', name='full_gp')
m1, S1, variational_gp_dist = variational_model(train_iter=1000)
plot_gp(variational_gp_dist, color='orange', name='variational')
m2, S2, independent_gp_dist = independent_inducing(train_iter=1000)
plot_gp(independent_gp_dist, color='purple', name='independent_inducing')
m3, S3, true_gp_dist = true_model()
plot_gp(true_gp_dist, color='green', name='true_model')
#
variational_dist = wasserstein2_distance(m1, m3, S1, S3)
independent_dist = wasserstein2_distance(m2, m3, S2, S3)
true_dist = wasserstein2_distance(m0, m3, S0, S3)
print("variational distance: \t {}".format(variational_dist))
print("independence distance: \t {}".format(independent_dist))
print("true distance: \t {}".format(true_dist))
plt.show()
