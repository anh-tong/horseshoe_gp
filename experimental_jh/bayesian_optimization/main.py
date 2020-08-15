#ExpectedImprovement as acquisition function and L-BFGS-D for bayes opt is standard setting in BO

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
choices=["ExpectedImprovement", "UpperConfidenceBound_", "ProbabilityOfImprovement"],
help='''
Acqusition Functions:
UpperConfidenceBound
ExpectedImprovement
ProbabilityOfImprovement
''', default = "ExpectedImprovement")

parser.add_argument('--num_trial', '-t', type = int, default = 10, help = "Number of Bayesian Optimization Interations")
parser.add_argument('--batch_size', '-b', type = int, default = 4)
parser.add_argument('--num_raw_samples', '-r', type = int, default = 20)
parser.add_argument('--num_inducing', '-i', type = int, default = 10)
parser.add_argument('--n_kernels', '-k', type = int, default = 5)
parser.add_argument('--num_step', '-p', type = int, default = 10, help = "Number of steps to optimize surrogate model for each BO stages")
parser.add_argument('--learning_rate', '-l', type = float, default = 3e-4, help = "learning rate in Adam optimizer")

args = parser.parse_args()

import math
import torch
is_cuda = torch.cuda.is_available()
torch.manual_seed(2020)

import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood

from src.structural_sgp import VariationalGP, StructuralSparseGP
exec("from src.sparse_selector import " + args.selector)

exec("from botorch.acquisition.analytic import " + args.acq_fun)
from botorch.generation.gen import gen_candidates_scipy

from utils import branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock

for opt in [branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock]:
    bench_fun = opt()
    
    n_inducing = args.num_inducing
    inducing_points = torch.stack([torch.linspace(0, 1, n_inducing)] * bench_fun.dim, dim = 1)

    # set up kernels
    n_kernels = args.n_kernels
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
    
    if args.acq_fun == "UpperConfidenceBound":
        exec("acq_fun = " + args.acq_fun + "(model, beta=0.2)")
    else:
        exec("acq_fun = " + args.acq_fun + "(model, best_f=0.2)")
        
    #Initial Points given
    x = torch.empty(args.num_raw_samples, bench_fun.dim) * (bench_fun.bound[1] - bench_fun.bound[0]).view(1, -1) + bench_fun.bound[0].view(1, -1)
    y = bench_fun.value(x)
    
    #Load modules to GPU
    if is_cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()
        elbo = elbo.cuda()
        
        inducing_points = inducing_points.cuda()
        x = x.cuda()
        y = y.cuda()
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=args.learning_rate)
    
    #Initiali Training
    print("---Initial training---")
    for ind in range(args.num_step):
        optimizer.zero_grad()
        selector.update_tau_lambda()
        output = model(x)
        loss = - elbo(output, y)
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Iter: {} \t Loss: {:.2f}".format(ind, loss.item()))
        
    batch_candidates, batch_acq_values = gen_candidates_scipy(
        initial_conditions=x.view(-1, 1, bench_fun.dim),
        acquisition_function=acq_fun,
        lower_bounds = bench_fun.bound[0],
        upper_bounds = bench_fun.bound[1])
    
    x = torch.cat([x, batch_candidates], dim = 0)
    y = torch.cat([y, bench_fun(batch_candidates)], dim = 0)
        
    #Bayesian Optimization iteration
    for tries in range(args.num_trial):
        print("try: %d" %tries)
        pass
        
        elbo = PredictiveLogLikelihood(likelihood, model, num_data=args.num_raw_samples + tries)
        if is_cuda:
            elbo = elbo.cuda()
        else:
            x = torch.cat([x], dim = 0)
            y = torch.cat([y], dim = 0)
                
        optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.01)
        
        for ind in range(args.num_step):
            optimizer.zero_grad()
            selector.update_tau_lambda()
            loss = - elbo(model(x), y)            
            loss.backward(retain_graph=True)
            optimizer.step()
    
    print(bench_fun.sol)
    
    
        
    
    
    