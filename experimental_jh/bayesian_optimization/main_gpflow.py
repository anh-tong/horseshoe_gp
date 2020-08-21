import gpflow
import tensorflow as tf
import tensorflow_probability as tfp
tf.set_random_seed(2020)
tfd = tfp.distributions

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

norm = tfd.Normal(loc=0., scale=1.)

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
if __name__ == "__main__":
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
    