from src.experiment_tf import *
from src.experiment_tf import create_model as old_create_model

def create_model(inducing_point, data_shape, num_data):
    generator = Generator(data_shape, base_fn=[create_rbf, create_period])
    kernels = []
    for _ in range(2):
        kernels.extend()