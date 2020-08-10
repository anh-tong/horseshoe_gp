import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood

import src.structural_sgp import VariationalGP, StructuralSparseGP, TrivialSelector, SpikeAndSlabSelector, SpikeAndSlabSelectorV2, HorseshoeSelector
from src.mean_field_hs import MeanFieldHorseshoe, VariatioalHorseshoe

import matplotlib.pyplot as plt

#Space
train_x = torch.linspace(-1, 1, 51)
