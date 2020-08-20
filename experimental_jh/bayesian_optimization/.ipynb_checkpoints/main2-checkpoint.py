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
choices=["qExpectedImprovement", "qUpperConfidenceBound", "qProbabilityOfImprovement"],
help='''
Acqusition Functions:
qUpperConfidenceBound
qExpectedImprovement
qProbabilityOfImprovement
''', default = "qUpperConfidenceBound")

parser.add_argument('--num_trial', '-t', type = int, default = 10, help = "Number of Bayesian Optimization Interations")
parser.add_argument('--batch_size', '-b', type = int, default = 4)
parser.add_argument('--num_raw_samples', '-r', type = int, default = 20)
parser.add_argument('--num_inducing', '-i', type = int, default = 10)
parser.add_argument('--n_kernels', '-k', type = int, default = 5)
parser.add_argument('--num_step', '-p', type = int, default = 50, help = "Number of steps to optimize surrogate model for each BO stages")
parser.add_argument('--learning_rate', '-l', type = float, default = 3e-6, help = "learning rate in Adam optimizer")

args = parser.parse_args()

import math
import torch
#if torch.cuda.is_available():
#    device = torch.device('cuda')
#else:
#    device = torch.device('cpu')
device = torch.device('cpu')
torch.manual_seed(2020)
#torch.autograd.set_detect_anomaly(True)

import gpytorch
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood

from src.structural_sgp import VariationalGP, StructuralSparseGP
exec("from src.sparse_selector import " + args.selector)

from botorch import fit_gpytorch_model
from botorch.generation.gen import gen_candidates_torch
from botorch.generation.gen import gen_candidates_scipy
#Standard setting is to use L-BFGS-D for optimizing acqusition function
#But due to numerical issue in Botorch(https://github.com/pytorch/botorch/issues/179), used Adam in torch 
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.models.utils import gpt_posterior_settings
from botorch.models.gpytorch import GPyTorchModel
if args.acq_fun == "qUpperConfidenceBound":
    exec("from botorch.acquisition.monte_carlo import " + args.acq_fun)
else:
    exec("from botorch.acquisition.analytic import " + args.acq_fun)
from botorch.sampling import IIDNormalSampler
from botorch.optim import optimize_acqf

from utils import branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock

from typing import Any, Union
#def posterior(self, X: torch.Tensor, observation_noise: Union[bool, torch.Tensor] = False, **kwargs: Any) -> GPyTorchPosterior:
    #self.eval()
    #with gpt_posterior_settings():
    
StructuralSparseGP.posterior = GPyTorchModel.posterior
    
#StructuralSparseGP.posterior = posterior

for opt in [branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock]:
    bench_fun = opt()
    
    n_inducing = args.num_inducing
    inducing_points = torch.stack([torch.linspace(0, 1, n_inducing)] * bench_fun.dim, dim = 1).to(device)

    # set up kernels
    n_kernels = args.n_kernels
    means = [ZeroMean().to(device)] * n_kernels
    kernels = [RBFKernel().to(device)] * n_kernels
    
    # GP for each kernel
    gps = []
    for mean, kernel in zip(means, kernels):
        gp = VariationalGP(mean, kernel, inducing_points).to(device)
        gps.append(gp)

    exec("selector = " + args.selector + "(dim=n_kernels, A=1., B=1.)")
    model = StructuralSparseGP(gps, selector).to(device)

    likelihood = GaussianLikelihood().to(device)
    elbo = PredictiveLogLikelihood(likelihood, model, num_data=args.num_raw_samples)
    
    if args.acq_fun == "qUpperConfidenceBound":
        exec("acq_fun = " + args.acq_fun + "(model, beta=0.1)")
    else:
        exec("acq_fun = " + args.acq_fun + "(model, best_f=0.1)")
        
    #Initial Points given
    x = torch.empty(args.num_raw_samples, bench_fun.dim).to(device)
    y = bench_fun(x)
    
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=args.learning_rate)
    
    fit_gpytorch_model(elbo)
    
    #Initiali Training
    print("---Initial training---")
    for ind in range(args.num_step):
        optimizer.zero_grad()
        selector.update_tau_lambda()
        loss = - elbo(model(x), y)
        loss.backward(retain_graph=True)
        optimizer.step()
        print("Iter: {} \t Loss: {:.2f}".format(ind, loss.item()))
        
    #Bayesian Optimization iteration
    for tries in range(args.num_trial):
        print("\ntry: %d" %(tries + 1))
        batch_candidates, batch_acq_values = gen_candidates_torch(
            initial_conditions=x.view(-1, 1, bench_fun.dim),
            acquisition_function=acq_fun,
            lower_bounds = bench_fun.lower_bound,
            upper_bounds = bench_fun.upper_bound)
        batch_candidates = batch_candidates.view(x.shape)

        x = torch.cat([x, batch_candidates], dim = 0)
        y = torch.cat([y, bench_fun(batch_candidates)], dim = 0)
        
        elbo.num_data += 1
        
        print("\n")
        
        for ind in range(args.num_step):
            optimizer.zero_grad()
            selector.update_tau_lambda()
            loss = - elbo(model(x), y)            
            loss.backward(retain_graph=True)
            optimizer.step()
            print("Iter: {} \t Loss: {:.2f}".format(ind, loss.item()))
    
    print("result:")
    print(model(x[-1]))
    print(y[-1])
    
    
        
    
    
    