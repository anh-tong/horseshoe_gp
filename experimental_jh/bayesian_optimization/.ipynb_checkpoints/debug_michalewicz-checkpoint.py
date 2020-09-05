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

import seaborn as sns

#-------------------------argparse-------------------------
import argparse
parser = argparse.ArgumentParser()

###This parts is not used in Baseline
parser.add_argument('--num_inducing', '-i', type = int, default = 10)
parser.add_argument('--n_kernels', '-k', type = int, default = 2)

"""
parser.add_argument('--bench_fun', '-b',
choices=["branin_rcos", "six_hump_camel_back", "goldstein_price", "rosenbrock", "hartman_6"],
default="branin_rcos")
"""

parser.add_argument('--learning_rate', '-l', type = float, default = 0.1, help = "learning rate in Adam optimizer")
parser.add_argument('--num_step', '-s', type = int, default = 10, help = "number of steps in each BO iteration")

args = parser.parse_args()
#-------------------------argparse-------------------------

#exec("from utils import " + args.bench_fun)
#exec("bench_fun = " + args.bench_fun)
from utils import Michalewicz

from utils import get_data_shape

from src.kernels import create_rbf, create_se_per


#main
if __name__ == "__main__":
    obj_fun = Michalewicz()
    
    train_x = tf.random.uniform(
        (100, obj_fun.dim),
        dtype=tf.dtypes.float64
    ) * (obj_fun.upper_bound - obj_fun.lower_bound) + obj_fun.lower_bound
    train_y = tf.expand_dims(obj_fun(train_x), 1)
    
    valid_x = tf.random.uniform(
        (100, obj_fun.dim),
        dtype=tf.dtypes.float64
    ) * (obj_fun.upper_bound - obj_fun.lower_bound) + obj_fun.lower_bound
    valid_y = tf.expand_dims(obj_fun(valid_x), 1)
    
    
   #Horseshoe model
    optimizer_horseshoe = tf.optimizers.Adam(learning_rate=args.learning_rate)
    
    inducing_point = obj_fun.lower_bound +  tf.random.uniform(
        (args.num_inducing, obj_fun.dim),
        dtype=tf.dtypes.float64
    ) * (obj_fun.upper_bound - obj_fun.lower_bound)
    
    kernels = [create_rbf(get_data_shape(x)), create_se_per(get_data_shape(x))] * args.n_kernels
    fix_kernel_variance(kernels)
    
    gps = []
    for kernel in kernels:
        gp = SVGP(kernel, likelihood=None, inducing_variable=inducing_point)
        gps.append(gp)
        
    selector = HorseshoeSelector(dim=len(gps))
    likelihood = Gaussian()
    model_horseshoe = StructuralSVGP(gps, selector, likelihood)
    
    train_loss_horseshoe = model_horseshoe.training_loss_closure((train_x, train_y))
    valid_loss_horseshoe = model_horseshoe.training_loss_closure((valid_x, valid_y))
    
    @tf.function
    def optimize_step_horseshoe():
        optimizer.minimize(
            train_loss_horseshoe,
            model_horseshoe.trainable_variables)
        
    optimizer_SE = tf.optimizers.Adam(learning_rate=args.learning_rate)
    
    
    
    #SE model
    model = gpflow.models.GPR(
        data=(train_x, train_y),
        kernel=create_rbf(get_data_shape(x)),
        mean_function=None)
    
    train_loss_SE = model_horseshoe.training_loss_closure((train_x, train_y))
    valid_loss_SE = model_horseshoe.training_loss_closure((valid_x, valid_y))
        
    @tf.function
    def optimize_step_SE():
        optimizer.minimize(
            train_loss_SE,
            model_SE.trainable_variables)    
    
    
    #Training
    num_epoch = 1000
    
    df_res = pd.DataFrame(0, index = range(num_epoch), col=["train_horseshoe", "valid_horseshoe", "train_SE", "valid_SE"])
    
    for step in range(df_res):
        optimizer.step()
        
        df_res.loc[step] = [train_loss_horseshoe, valid_loss_horseshoe, train_loss_SE, valid_loss_SE]
        
        if step%100 == 0:
            print(train_loss_SE, valid_loss_SE, train_loss_horseshoe, valid_loss_horseshoe)