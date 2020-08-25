"""
The follows kernels contains some features:

    - Linear kernel has parameter location
    - kernel is added parameterization
"""


import tensorflow as tf
import numpy as np
import gpflow
from gpflow import Parameter
from gpflow.utilities import positive
from gpflow.kernels import (Linear as Linear_gpflow,
                            RBF,
                            Periodic,
                            Product
                            )

class Linear(Linear_gpflow):

    def __init__(self, location=0., bound=None, active_dims=None):
        super().__init__(variance=1., active_dims=active_dims)
        if bound is None:
            self.location = Parameter(location)
        else:
            raise NotImplementedError

    @property
    def ard(self):
        return self.location.shape.ndims > 0

    def K(self, X, X2=None):
        X_shifted = X - self.location
        if X2 is None:
            return super().K(X_shifted, None)
        else:
            X2_shifted = X2 - self.location
            return super().K(X_shifted, X2_shifted)

    def K_diag(self, X):
        X_shifted = X - self.location
        return super().K_diag(X_shifted)


def create_linear(data_shape):
    location = np.random.uniform(low=1.5 * data_shape["x_min"], high=1.5 * data_shape["x_max"])
    return Linear(location=location)

def create_rbf(data_shape, sd=1.):
    r = np.random.rand()
    if r < 0.5:
        lengthscales = np.random.normal(loc=data_shape['x_sd'], scale=sd)
        lengthscales = np.exp(lengthscales)
    else:
        dist = data_shape['x_max'] - data_shape['x_min']
        lengthscales = np.random.normal(loc=np.log(2 * dist), scale=sd)
        lengthscales = np.exp(lengthscales)


    return RBF(lengthscales=lengthscales)

def create_period(data_shape, sd=1.):
    base_kernel = create_rbf(data_shape)
    r = np.random.rand()
    if r < 0.33:
        value = data_shape["x_sd"] - 2.
        period = np.random.normal(loc=value, scale=sd)
        period = np.exp(period)
    elif r < 0.5:
        dist = data_shape["x_max"] - data_shape["x_min"]
        period = np.random.normal(loc=np.log(dist) - 3.2, scale=sd)
        period = np.exp(period)
    else:
        x_min_abs_diff = data_shape["x_min_abs_diff"]
        period = np.random.normal(loc=x_min_abs_diff + 3.2, scale=sd)
        period = np.exp(period)

    return Periodic(base_kernel, period=period)





if __name__ == "__main__":

    from src.utils import get_dataset, get_data_shape

    dataset = get_dataset("airline")
    data_shape = get_data_shape(dataset)
    print(data_shape)


    kernels = [create_linear(data_shape),
               create_rbf(data_shape),
               create_period(data_shape), Product([create_linear(data_shape),
                                                   create_rbf(data_shape),
                                                   create_period(data_shape)])]

    for kernel in kernels:
        gpflow.utilities.print_summary(kernel)







