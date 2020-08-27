import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append("../..")


import gpflow
from gpflow.kernels import SquaredExponential

import tensorflow as tf
tf.random.set_seed(2020)

import numpy as np
import pandas as pd

from scipy.optimize import Bounds, minimize

#-------------------------argparse-------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--show_plot', '-v', type = bool, default = True)

###This is argument for selector, but not used in baseline
parser.add_argument('--selector', '-s',
choices=["TrivialSelector", "SpikeAndSlabSelector", "HorseshoeSelector"],
help='''
Selectors:
TrivialSelector
SpikeAndSlabSelector
HorseshoeSelector
''', default = "HorseshoeSelector")

###This parts is not used in Baseline
parser.add_argument('--num_inducing', '-i', type = int, default = 10)
parser.add_argument('--n_kernels', '-k', type = int, default = 5)

"""
parser.add_argument('--bench_fun', '-b',
choices=["branin_rcos", "six_hump_camel_back", "goldstein_price", "rosenbrock", "hartman_6"],
default="branin_rcos")
"""

parser.add_argument('--acq_fun', '-a',
choices=["EI", "UCB", "POI"],
help='''
EI: Expected Improvement
UCB: Upper Confidence Bound
POI: Probability of Improvement
''', default = "EI")

parser.add_argument('--num_trial', '-t', type = int, default = 200, help = "Number of Bayesian Optimization Interations")

parser.add_argument('--num_init', '-n', type = int, default = 10,
                    help = "Number of runs for each benchmark function to change intial points randomly.")
parser.add_argument('--learning_rate', '-l', type = float, default = 3e-4, help = "learning rate in Adam optimizer")
parser.add_argument('--noise_level', '-e', type = float, default = 0.01, help = "Noise in function evaluation")


args = parser.parse_args()
#-------------------------argparse-------------------------

#exec("from utils import " + args.bench_fun)
#exec("bench_fun = " + args.bench_fun)
from utils import branin_rcos, six_hump_camel_back, goldstein_price, rosenbrock, hartman_6, Styblinski_Tang, Michalewicz

exec("from utils import " + args.acq_fun)
exec("acq_fun = " + args.acq_fun + "()")

def create_rbf(x):
    lengthscales = tf.random.normal([1,2], mean=0., stddev=1., dtype=tf.dtypes.float64) + tf.math.reduce_std(x)
    lengthscales = tf.math.exp(lengthscales)
    return SquaredExponential(lengthscales=lengthscales)

def acq_max(lb, ub, sur_model, y_max, acq_fun, n_warmup = 10000, iteration = 10):
    bounds = Bounds(lb, ub)
    
    x_tries = tf.random.uniform(
        [n_warmup, obj_fun.dim],
        dtype=tf.dtypes.float64) * (ub - lb) + lb
    ys = acq_fun(x_tries, sur_model, y_max)
    x_max = tf.expand_dims(x_tries[tf.squeeze(tf.argmax(ys))], 0)
    max_acq = tf.reduce_max(ys)
    
    if tf.reduce_max(ys) > y_max:
        y_max = tf.reduce_max(ys)
        
    for iterate in range(iteration):
        locs = tf.random.uniform(
            [1, obj_fun.dim],
            dtype=tf.dtypes.float64) * (ub - lb) + lb
        
        opt_result = minimize(
            lambda x: -acq_fun(tf.reshape(locs, (1, -1)), sur_model, y_max).numpy(),
            locs,
            bounds=bounds,
            method="L-BFGS-B")
        
        if not opt_result.success:
            continue

        if max_acq is None or -opt_result.fun >= max_acq:
            x_max = tf.expand_dims(opt_result.x, 0)
            max_acq = -opt_result.fun
            
    return tf.clip_by_value(x_max, lb, ub)


#main
if __name__ == "__main__":
    
    ###Result directory
    save_file = "./GP_SE/"
    
    for bench_fun in [branin_rcos, six_hump_camel_back, goldstein_price, rosenbrock, hartman_6, Styblinski_Tang, Michalewicz]:
        obj_fun = bench_fun()

        df_result = pd.DataFrame(
            0,
            index=range(args.num_trial+1),
            columns=range(args.num_init))

        num_test = 0
        while num_test < args.num_init:
            ###n_inducing = args.num_inducing

            #Initial Points given
            x = tf.random.uniform(
                (1, obj_fun.dim),
                dtype=tf.dtypes.float64
            )
            x = x * (obj_fun.upper_bound -obj_fun.lower_bound) + obj_fun.lower_bound
            y = tf.expand_dims(obj_fun(x), 1)
            y = y + tf.random.normal(
                y.shape, mean = 0.0, stddev = args.noise_level, dtype=tf.dtypes.float64)

            y_start = tf.reduce_min(y, axis=0).numpy()

            df_result.loc[0, num_test] = y_start

            #Initiali Training
            #optimizer = gpflow.optimizers.Scipy()
            optimizer = tf.keras.optimizers.Adam()
            
            ###model
            model = gpflow.models.GPR(
                data=(x, y),
                kernel=create_rbf(x),
                mean_function=None)

            optimizer.minimize(
                model.training_loss,
                model.trainable_variables)

            #Bayesian Optimization iteration
            for tries in range(args.num_trial):
                x_new = acq_max(
                    obj_fun.lower_bound,
                    obj_fun.upper_bound,
                    model,
                    tf.reduce_max(y),
                    acq_fun)

                #Evaluation of new points
                y_new = tf.expand_dims(obj_fun(x_new), 1)
                y_new = y_new + tf.random.normal(
                    y_new.shape, mean = 0.0, stddev = args.noise_level, dtype=tf.dtypes.float64)

                x = tf.concat([x, x_new], 0)
                y = tf.concat([y, y_new], 0)

                ###model initialization again
                model = gpflow.models.GPR(
                    data=(x, y),
                    kernel=gpflow.kernels.SquaredExponential(),
                    mean_function=None)

                optimizer.minimize(
                    model.training_loss,
                    model.trainable_variables)

                #Result
                y_end = tf.reduce_min(y, axis=0).numpy()
                df_result.loc[tries + 1, num_test] = y_end

            print(bench_fun.__name__ + "-test %d: %f->%f" %(num_test + 1, y_start, y_end))
            num_test += 1

        df_result.to_csv(save_file + args.acq_fun + "_" + bench_fun.__name__ + ".csv")
