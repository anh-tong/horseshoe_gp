import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import sys
sys.path.append("../..")


import gpflow
from gpflow.optimizers import NaturalGradient
from gpflow.models import SVGP, BayesianModel
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF

from src.experiment_tf import init_inducing_points
from src.sparse_selector_tf import HorseshoeSelector
from src.structural_sgp_tf import StructuralSVGP
from src.kernel_generator_tf import Generator
from src.experiment_tf import fix_kernel_variance

#from gpflow.mean_functions import Zero

import tensorflow as tf
tf.random.set_seed(2020)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

import numpy as np
import pandas as pd


#-------------------------argparse-------------------------
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--show_plot', '-v', type = bool, default = True)

###This parts is not used in Baseline
parser.add_argument('--num_inducing', '-i', type = int, default = 10)
parser.add_argument('--n_kernels', '-k', type = int, default = 2)

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
parser.add_argument('--learning_rate', '-l', type = float, default = 0.1, help = "learning rate in Adam optimizer")
parser.add_argument('--num_step', '-s', type = int, default = 10, help = "number of steps in each BO iteration")

args = parser.parse_args()
#-------------------------argparse-------------------------

#exec("from utils import " + args.bench_fun)
#exec("bench_fun = " + args.bench_fun)
from utils import branin_rcos, six_hump_camel_back, goldstein_price, rosenbrock, hartman_6,  Styblinski_Tang, Michalewicz

exec("from utils import " + args.acq_fun)
exec("acq_fun = " + args.acq_fun + "()")

from utils import get_data_shape

from src.kernels import create_rbf, create_se_per

def acq_max(lb, ub, sur_model, y_max, acq_fun, n_warmup = 10000, iteration = 10):
    x_tries = tf.random.uniform(
        [n_warmup, obj_fun.dim],
        dtype=tf.dtypes.float64) * (ub - lb) + lb
    ys = acq_fun(
        x = x_tries,
        model = sur_model,
        ymax = y_max)
    x_max = tf.expand_dims(x_tries[tf.squeeze(tf.argmax(ys))], 0)
    max_acq = tf.reduce_max(ys)
    
    if tf.reduce_max(ys) > y_max:
        y_max = tf.reduce_max(ys)
            
    return tf.clip_by_value(x_max, lb, ub)


#main
if __name__ == "__main__":
    
    ###Result directory
    save_file = "./GP_Horseshoe_manual/"
    
    for bench_fun in [branin_rcos]:
        obj_fun = bench_fun()

        df_result = pd.DataFrame(
            0,
            index=range(args.num_trial+1),
            columns=range(args.num_init))  

        num_test = 0
        while num_test < args.num_init:
            #Initial Points given
            tf.random.set_seed(2020 + num_test)
            x = tf.random.uniform(
                (10, obj_fun.dim),
                dtype=tf.dtypes.float64
            )
            x = x * (obj_fun.upper_bound - obj_fun.lower_bound) + obj_fun.lower_bound
            y = tf.expand_dims(obj_fun(x), 1)

            y_start = tf.reduce_min(y, axis=0).numpy()

            df_result.loc[0, num_test] = y_start

            ###number of inducing variables
            inducing_point = obj_fun.lower_bound +  tf.random.uniform(
                (200, obj_fun.dim),
                dtype=tf.dtypes.float64
            ) * (obj_fun.upper_bound - obj_fun.lower_bound)
            
            #Initialize Optimizer
            optimizer = tf.optimizers.Adam(learning_rate=args.learning_rate)
            
            ###model
            generator = Generator(get_data_shape(x))
            kernels = [create_rbf(get_data_shape(x)), create_se_per(get_data_shape(x))] * args.n_kernels
            fix_kernel_variance(kernels)

            gps = []
            for kernel in kernels:
                gp = SVGP(kernel, likelihood=None, inducing_variable=inducing_point)
                gps.append(gp)
                
            selector = HorseshoeSelector(dim=len(gps))
            likelihood = Gaussian()
            model = StructuralSVGP(gps, selector, likelihood)
        
            #Bayesian Optimization iteration
            for tries in range(args.num_trial):
                model.num_data = len(y)
                
                train_loss = model.training_loss_closure((x, y))         
                
                @tf.function
                def optimize_step():
                    optimizer.minimize(
                        train_loss,
                        model.trainable_variables)
                    
                prev_loss = train_loss().numpy()
                
                # optimize GP
                while train_loss() + 1 < prev_loss:
                    prev_loss = train_loss().numpy()
                    for step in range(args.num_step):
                        optimize_step()
                    print(prev_loss)

                x_new = acq_max(
                    obj_fun.lower_bound,
                    obj_fun.upper_bound,
                    model,
                    tf.reduce_max(y),
                    acq_fun)

                #Evaluation of new points
                y_new = tf.expand_dims(obj_fun(x_new), 1)

                x = tf.concat([x, x_new], 0)
                y = tf.concat([y, y_new], 0)
                
                #Result
                y_end = tf.reduce_min(y, axis=0).numpy()
                df_result.loc[tries + 1, num_test] = y_end
                
            print(bench_fun.__name__ + "-test %d: %f->%f" %(num_test + 1, y_start, y_end))
            num_test += 1

        df_result.to_csv(save_file + args.acq_fun + "_" + bench_fun.__name__ + ".csv")
