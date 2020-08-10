import torch
import gpytorch
from sklearn import svm
#import lightgbm as lgb
#import xgboost as xgb

from src.structural_sgp import VariationalGP, StructuralSparseGP, TrivialSelector, SpikeAndSlabSelector, SpikeAndSlabSelectorV2, HorseshoeSelector
from src.mean_field_hs import MeanFieldHorseshoe, VariatioalHorseshoe

from bayes_opt import BayesianOptimization

def get_objective_svm(svm.SVC():