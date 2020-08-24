import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gpflow
from gpflow.utilities import print_summary
from gpflow.models.util import data_input_to_tensor

import tensorflow as tf
import tensorflow_probability as tfp
tf.random.set_seed(2020)

import numpy as np
import pandas as pd

from scipy.optimize import Bounds, minimize

import sys
sys.path.append("../..")

#-------------------------argparse-------------------------
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

parser.add_argument('--bench_fun', '-b',
choices=["branin_rcos", "six_hump_camel_back", "hartman_6", "goldstein_price", "rosenbrock"],
default="branin_rcos")

parser.add_argument('--acq_fun', '-a',
choices=["EI", "UCB", "POI"],
help='''
EI: Expected Improvement
UCB: Upper Confidence Bound
POI: Probability of Improvement
''', default = "EI")

parser.add_argument('--num_trial', '-t', type = int, default = 100, help = "Number of Bayesian Optimization Interations")
parser.add_argument('--num_raw_samples', '-r', type = int, default = 20)

###This parts is not used in Baseline
parser.add_argument('--num_inducing', '-i', type = int, default = 10)
parser.add_argument('--n_kernels', '-k', type = int, default = 5)
parser.add_argument('--num_init', '-n', type = int, default = 10,
                    help = "Number of runs for each benchmark function to change intial points randomly.")
parser.add_argument('--num_step', '-p', type = int, default = 50,
                    help = "Number of steps to optimize surrogate model for each BO stages")
parser.add_argument('--learning_rate', '-l', type = float, default = 3e-4, help = "learning rate in Adam optimizer")

args = parser.parse_args()
#-------------------------argparse-------------------------

if args.show_plot:
    import matplotlib.pyplot as plt

exec("from utils import " + args.bench_fun)
exec("bench_fun = " + args.bench_fun)
exec("from utils import " + args.acq_fun)

def acq_max(lb, ub, sur_model, y_max, acq_fun, n_warmup = 10000, iteration = 10):
    x_tries = tf.random.uniform(
        [n_warmup, obj_fun.dim],
        dtype=tf.dtypes.float64) * (ub - lb) + lb
    ys = acq_fun(x_tries, y_max)
    x_max = tf.expand_dims(x_tries[tf.squeeze(tf.argmax(ys))], 0)
    max_acq = tf.reduce_max(ys)
    
    if tf.reduce_max(ys) > y_max:
        y_max = tf.reduce_max(ys)
        
    bound = Bounds(lb, ub)
        
    def acq_loss_and_gradient(x):
        return tfp.math.value_and_gradient(
            lambda x: -acq_fun(tf.clip_by_value(tf.reshape(x, (1, -1)), lb, ub), y_max), x)
        
    for iterate in range(iteration):
        locs = tf.random.uniform(
            [1, obj_fun.dim],
            dtype=tf.dtypes.float64) * (ub - lb) + lb
        
        try:
            opt_result = tfp.optimizer.lbfgs_minimize(
                acq_loss_and_gradient,
                initial_position=locs,
                num_correction_pairs=10,
                tolerance=1e-8)
        except:
            continue

        if not opt_result.converged.numpy() or opt_result.failed.numpy():
            continue
        
        loc_res = opt_result.position
        obj_res = opt_result.objective_value

        if max_acq is None or -obj_res >= max_acq:
            x_max = loc_res
            max_acq = -obj_res
            
    return x_max, tf.reshape(max_acq, (1, -1))


#main
if __name__ == "__main__":
    
    
    ###Result directory
    save_file = "./GP_mattern/"
    
    obj_fun = bench_fun()
    df_result = pd.DataFrame(
        0,
        index=range(args.num_trial),
        columns=range(args.num_init))

    num_test = 0
    while num_test < args.num_init:
        try:
            ###n_inducing = args.num_inducing

            #Initial Points given
            x = tf.random.uniform(
                (args.num_raw_samples, obj_fun.dim),
                dtype=tf.dtypes.float64
            )
            x = x * (obj_fun.upper_bound -obj_fun.lower_bound) + obj_fun.lower_bound
            y = tf.expand_dims(obj_fun(x), 1)

            ###model
            model = gpflow.models.GPR(
                data=(x, y),
                kernel=gpflow.kernels.Matern52(),
                mean_function=None)

            exec("acq_fun = " + args.acq_fun + "(model)")

            #Initiali Training
            optimizer = gpflow.optimizers.Scipy()

            optimizer.minimize(
                model.training_loss,
                model.trainable_variables,
                options=dict(maxiter=20))

            #Bayesian Optimization iteration
            for tries in range(args.num_trial):
                x_new, y_new = acq_max(
                    obj_fun.lower_bound,
                    obj_fun.upper_bound,
                    model,
                    tf.reduce_max(y),
                    acq_fun)

                x = tf.concat([x, x_new], 0)
                y = tf.concat([y, y_new], 0)

                #model.data = data_input_to_tensor((x, y))
                #model.num_latent_gps += 1

                model = gpflow.models.GPR(
                    data=(x, y),
                    kernel=gpflow.kernels.Matern52(),
                    mean_function=None)

                optimizer.minimize(
                    model.training_loss,
                    model.trainable_variables,
                    options=dict(maxiter=20))

                #Result
                df_result.loc[tries, num_test] = tf.reduce_min(y, axis=0).numpy()

            print(bench_fun.__name__ + "-test:%d" %(num_test + 1))
            num_test += 1
            
        except:
            continue
    
    df_result.to_csv(save_file + bench_fun.__name__ + ".csv")
