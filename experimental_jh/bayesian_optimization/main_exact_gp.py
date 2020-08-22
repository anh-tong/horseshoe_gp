import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gpflow
from gpflow.utilities import print_summary
from gpflow.models.util import data_input_to_tensor
from gpflow.models.model import RegressionData

import tensorflow as tf
import tensorflow_probability as tfp
tf.random.set_seed(2020)

import numpy as np

import sys
sys.path.append("../..")

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--show_plot', '-v', type = bool, default = True)

##This is argument for selector, but not used in baseline
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
parser.add_argument('--num_raw_samples', '-r', type = int, default = 20)

###This parts is not used in Baseline
parser.add_argument('--num_inducing', '-i', type = int, default = 10)
parser.add_argument('--n_kernels', '-k', type = int, default = 5)
parser.add_argument('--num_step', '-p', type = int, default = 50, help = "Number of steps to optimize surrogate model for each BO stages")
parser.add_argument('--learning_rate', '-l', type = float, default = 3e-4, help = "learning rate in Adam optimizer")

args = parser.parse_args()

if args.show_plot:
    import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds

from utils import branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock
exec("from utils import " + args.acq_fun)

def acq_max(bounds, sur_model, y_max, acq_fun, n_warmup = 10000, iteration = 10):
    x_tries = tf.random.uniform(
        [n_warmup, bench_fun.dim],
        dtype=tf.dtypes.float64) * tf.expand_dims(bounds.ub - bounds.lb, 0) + tf.expand_dims(bounds.lb, 0)
    ys = acq_fun(x_tries, y_max)
    x_max = x_tries[tf.squeeze(tf.argmax(ys))]
    max_acq = tf.reduce_max(ys)
    if tf.reduce_max(ys) > y_max:
        y_max = tf.reduce_max(ys)

    for iterate in range(iteration):
        locs = tf.random.uniform(
            [1, bench_fun.dim],
            dtype=tf.dtypes.float64) * tf.expand_dims(bounds.ub - bounds.lb, 0) + tf.expand_dims(bounds.lb, 0)
        res = minimize(
            lambda x: -acq_fun(x, y_max),
            locs,
            bounds = bounds,
            method = "L-BFGS-B")

        if not res.success:
            continue

        if max_acq is None or -res.fun[0] >= max_acq:
            x_max = res.x
            max_acq = -res.fun[0]

    return tf.clip_by_value(x_max, bounds[:, 0], bounds[:, 1])


#models
if __name__ == "__main__":
    for opt in [branin_rcos, six_hump_camel_back, hartman_6, goldstein_price, rosenbrock]:
        bench_fun = opt()

        #n_inducing = args.num_inducing

        #Initial Points given
        x = tf.random.uniform(
            (args.num_raw_samples, bench_fun.dim),
            dtype=tf.dtypes.float64
        )
        x = x * (bench_fun.upper_bound - bench_fun.lower_bound) + bench_fun.lower_bound
        y = tf.expand_dims(bench_fun(x), 1)
        
        bound = Bounds(bench_fun.lower_bound, bench_fun.upper_bound)

        #model
        model = gpflow.models.GPR(
            data=(x, y),
            kernel=gpflow.kernels.Matern52(),
            mean_function=None)
        optimizer = gpflow.optimizers.Scipy()
        
        exec("acq_fun = " + args.acq_fun + "(model)")
        model.likelihood.variance.assign(0.01)
        model.kernel.lengthscales.assign(0.3)

        #Initiali Training
        optimizer.minimize(
            model.training_loss,
            model.trainable_variables,
            options=dict(maxiter=100))

        #Bayesian Optimization iteration
        for tries in range(args.num_trial):
            x_new = acq_max(bound, model, tf.reduce_max(y), acq_fun)
            print(x_new.shape)

            x = tf.concat([x, x_new], 1)
            y = tf.concat([y, acq_fun(x_new)], 1)
            
            model.data = data_input_to_tensor(RegressionData((x, y)))
                  
            
            
            opt.minimize(model.training_loss, m.trainable_variables, options=dict(maxiter=100))
