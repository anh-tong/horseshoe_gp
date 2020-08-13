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
choices=["UpperConfidenceBound", "ExpectedImprovement", "ProbabilityOfImprovement"],
help='''
Acqusition Functions:
UpperConfidenceBound
ExpectedImprovement
ProbabilityOfImprovement
''', default = "UpperConfidenceBound")

parser.add_argument('--num_trial', '-t', type = int, default = 10, help = "Number of Bayesian Optimization Interations")
parser.add_argument('--batch_size', '-b', type = int, default = 4)
parser.add_argument('--num_raw_samples', '-r', type = int, default = 20)
parser.add_argument('--num_inducing', '-i', type = int, default = 10)

args = parser.parse_args()

import math
import torch

import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood

from src.structural_sgp import VariationalGP, StructuralSparseGP
exec("from src.sparse_selector import " + args.selector)

from botorch.fit import fit_gpytorch_model
exec("from botorch.acquisition.analytic import " + args.acq_fun)

from utils import branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock

for opt in [branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock]:
    bench_fun = opt()
    
    n_inducing = 10
    inducing_points = torch.linspace(0, 1, n_inducing).repeat(bench_fun.dim).view(-1, 1)
    
    # set up kernels
    n_kernels = 5
    means = [ZeroMean()] * n_kernels
    kernels = [RBFKernel()] * n_kernels
    
    # GP for each kernel
    gps = []
    for mean, kernel in zip(means, kernels):
        gp = VariationalGP(mean, kernel, inducing_points)
        gps.append(gp)

    exec("selector = " + args.selector + "(dim=n_kernels, A=1., B=1.)")
    model = StructuralSparseGP(gps, selector)

    likelihood = GaussianLikelihood()
    elbo = PredictiveLogLikelihood(likelihood, model, num_data=args.num_raw_samples)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
    
    init_pnt = torch.empty(args.num_raw_samples).random_()
    
    for ind in range(100):
        optimizer.zero_grad()
        selector.update_tau_lambda()
        loss = - elbo(model(init_pnt), bench_fun(init_pnt))
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Iter: {} \t Loss: {:.2f}".format(i, loss.item()))

#    for tries in range(args.num_trial):
        
    
    
    