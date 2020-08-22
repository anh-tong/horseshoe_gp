import torch
device = torch.device('cpu')
torch.manual_seed(2020)
import torch.optim as optim
import sys
sys.path.append("../..")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--selector', '-s',
choices=["TrivialSelector", "SpikeAndSlabSelector", "HorseshoeSelector"],
help='''
Selectors:
TrivialSelector
SpikeAndSlabSelector
HorseshoeSelector
''', default = "HorseshoeSelector")

parser.add_argument('--acq_fun', '-a',
choices=["EI", "UCB", "POI"],
help='''
EI: Expected Improvement
UCB: Upper Confidence Bound
POI: Probability of Improvement
''', default = "EI")

parser.add_argument('--num_trial', '-t', type = int, default = 10, help = "Number of Bayesian Optimization Interations")
parser.add_argument('--batch_size', '-b', type = int, default = 4)
parser.add_argument('--num_raw_samples', '-r', type = int, default = 20)
parser.add_argument('--num_inducing', '-i', type = int, default = 10)
parser.add_argument('--n_kernels', '-k', type = int, default = 5)
parser.add_argument('--num_step', '-p', type = int, default = 50, help = "Number of steps to optimize surrogate model for each BO stages")
parser.add_argument('--learning_rate', '-l', type = float, default = 3e-6, help = "learning rate in Adam optimizer")

"""
##Deprecated
import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
"""
from scipy.optimize import minimize, Bounds

from utils import branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock

norm = torch.distributions.normal.Normal(0, 1)

def acq_max(bounds, sur_model, y_max, acq_fun, n_warmup = 10000, iteration = 10):
    x_tries = torch.empty(n_warmup, bounds.shape[0])._rand() * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    ys = acq_fun(x_tries, sur_model, y_max, kappa)
    x_max = x_tries[ys.argmax()]
    max_acq = ys.max()

    for iterate in range(iteration):
        locs = torch.empty(bounds.shape[0])._rand() * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        res = minimize(lambda x: -acq_fun(x),
                      locs,
                      bounds = bounds,
                      method = "L-BFGS-B")

        if not res.success:
            continue

        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    return torch.clip(x_max, bounds[:, 0], bounds[:, 1])

#acqusition functions
def UCB(x, sur_model, kappa = 2.576):
    mean, std = sur_model(x)
    return mean + kappa * std

def EI(x, sur_model, y_max):
    mean, std = sur_model(x)
    a = (mean - ymax - x)
    z = a / std
    return a * norm.cdf(z) + std * norm.pdf(z)

def POI(x, sur_model, y_max):
    mean, std = sur_model(x)
    z = (mean - y_max - x)/std
    return norm.cdf(z)

#models
class exact_GP(nn.Module):
    def __init__(self, ndim_x, ndim_y):
        super(GP, self).__init__()
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y
        self.data_x = torch.zeros(0, self.ndim_x).to(device)
        self.data_y = torch.zeros(0, self.ndim_y).to(device)
        self.dist = torch.zeros((self.ndim_x, 0, 0)).to(device)
        
        self.ARD_weight = torch.nn.Parameter(torch.randn(self.ndim_x, device = device))
        self.raw_beta = torch.nn.Parameter(torch.rand(1, device = device))      
        self.raw_sigma = torch.nn.Parameter(torch.rand(1, device = device))

    def distance(self, data_x):
        N = data_x.shape[0]
        D = data_x.shape[1]
        dist = torch.zeros(D, N, N)
        for dim in range(D):
            term1 = (data_x[:, dim] ** 2).expand((N,N))
            term2 = data_x[:, dim].unsqueeze(0).transpose(0, 1) * data_x[:, dim].unsqueeze(0)
            dist[dim] = (term1 + term1.transpose(0,1) - 2 * term2).unsqueeze(0)
        return dist
    
    def sigma(self):
        return 1/self.raw_sigma.pow(2) * torch.eye(dim)

    def noise(self, dim = 1):
        return F.softplus(self.raw_beta.pow(2))

    def fit_x(self, x_new):
        self.data_x = torch.cat((self.data_x, x_new), 0)
        self.dist = self.distance(self.data_x)
         
    def fit_y(self, y_new):
        self.data_y = torch.cat((self.data_y, y_new), 0)
        
    def fit(self, x_new, y_new, max_step = 1000):
        self.fit_x(x_new)
        self.fit_y(y_new)
        optimizer = optim.Adam(self.parameters(), lr = 3e-4)
        for step in range(max_step):
            optimizer.zero_grad()
            
            optimizer.step()

    def forward(self):
        #Simple Kernel
        #term1 = torch.clamp(-0.5 * self.KHP(self.dist.unsqueeze(0)).squeeze(0).squeeze(0), max = 0)
        #term2 = self.noise(self.data_x.shape[0])
        #cov_fitted = torch.exp(term1) + term2
        
        #ARD Mattern Kernel
        ARD_dist = torch.sqrt(5) * torch.einsum("abb,a->bb", self.dist, self.ARD_weight)
        cov_fitted = self.sigma() * (1 + ARD_dist + ARD_dist.pow(2).div(3)) * (-ARD_dist).exp()
        return cov_fitted
    
    def predict(self, x_new, y_new):
        num_fitted = self.data_x.shape[0]
        dist = self.distance(torch.cat((self.data_x, x_new), 0))
        
        cov_fitted = torch.exp(
            -0.5 * self.KHP(dist[:, :num_fitted, :num_fitted].unsqueeze(0)).squeeze(0).squeeze(0)) +
            self.noise(num_fitted)
        cov_wing = torch.exp(
            -0.5 * self.KHP(dist[:, :num_fitted, num_fitted:].unsqueeze(0)).squeeze(0).squeeze(0))
        cov_new = torch.exp(
            -0.5 * self.KHP(dist[:, num_fitted:, num_fitted:].unsqueeze(0)).squeeze(0).squeeze(0)) +
            self.noise(x_new.shape[0])
        
        pred_mu = torch.matmul(torch.matmul(cov_wing.transpose(0,1), cov_fitted.inverse()), self.data_y)
        pred_cov = cov_new -
            torch.matmul(torch.matmul(cov_wing.transpose(0, 1), cov_fitted.inverse()), cov_wing) +
            self.noise(x_new.shape[0]) 
        
        return pred_mu, pred_cov

def NLL(y, cov_fitted):
    return torch.log(torch.clamp(torch.det(cov_fitted.squeeze(0).squeeze(0)), min = 1e-8)) +
    torch.matmul(torch.matmul(y.transpose(0,1), cov_fitted.inverse()), y)   
    
if __name__ == "__main__":
    exec("acq_fun = " + args.acq_fun)
    for obj_fun in [branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock]:
        model = exact_GP(acq_fun.dim, 1).to(device)
        bound = Bounds(obj_fun.lower_bound, obj_fun.upper_bound)
        x = torch.randn(args.num_raw_samples, acq_fun.dim) * (obj_fun.upper_bound - obj_fun.lower_bound) + obj_fun.lower_bound
        model.fit(initial_pnt) 
        
        for step in range(args.num_step):
            x_new = acq_max(bound, model)
            model.fit(torch.cat(x, x_new))
            
        res = acq_max(bound, model)
        
        
        
        