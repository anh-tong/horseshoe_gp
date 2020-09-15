"""
The follows kernels contains some features:

    - Linear kernel has parameter location
    - kernel is added parameterization
"""


import numpy as np
import itertools
import gpflow
from gpflow import Parameter
from scipy.stats import truncnorm
from gpflow.kernels import (Linear as Linear_gpflow,
                            RBF,
                            Periodic,
                            Product,
                            White,
                            ChangePoints)

class Linear(Linear_gpflow):

    def __init__(self, variance=1., location=0., bound=None, active_dims=None):
        super().__init__(variance=variance, active_dims=active_dims)
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


def create_linear(data_shape, sd=1., active_dims=None, type="weak"):
    diff = data_shape["x_max"] - data_shape["x_min"]
    location = np.random.uniform(low=data_shape["x_min"] + 0.1 * diff,
                                 high=data_shape["x_max"] - 0.1*diff)
    if active_dims is not None:
        location = location[active_dims]
    return Linear(location=location, active_dims=active_dims)

def create_rbf(data_shape, sd=1., active_dims=None, type="weak"):

    # weak prior
    lengthscales = np.random.rand(data_shape["n_dims"]) * 0.1
    lengthscales = np.exp(lengthscales)

    if active_dims is not None:
        lengthscales =lengthscales[active_dims]

    return RBF(lengthscales=lengthscales, active_dims=active_dims)

def create_period(data_shape, sd=1., active_dims=None, type="weak"):
    # base_kernel = create_rbf(data_shape, active_dims=active_dims)
    base_kernel = RBF(active_dims=active_dims)

    # weak prior
    if type == "weak":
        p_min = np.log(10*(data_shape["x_max"] - data_shape["x_min"])/ data_shape["N"])
        W = truncnorm.rvs(0., np.infty, loc=-0.5, size=data_shape["n_dims"])
        period = np.exp(p_min + W)

    # strong prior
    elif type == "strong":
        p_max = np.log(data_shape["x_max"] - data_shape["x_min"]) - np.log(5.)
        U = truncnorm.rvs(-np.infty, 0., loc=-0.5 , size=data_shape["n_dims"])
        period = np.exp(p_max + U)
    else:
        raise ValueError("Wrong prior [{}] for hyperparameter intialization ".format(type))

    if active_dims is not None:
        period = period[active_dims]

    return Periodic(base_kernel, period=period)

def create_se_per(data_shape, sd=1., active_dims=None, type="weak"):

    se = create_rbf(data_shape, sd, active_dims)
    per = create_period(data_shape, sd, active_dims)
    return Product([se, per])


def create_changepoint(data_shape, sd=1., active_dims=None, base_fn=create_rbf):
    kernel = base_fn(data_shape, sd, active_dims)
    kernel1 = base_fn(data_shape,sd, active_dims)
    diff = data_shape["x_max"] - data_shape["x_min"]
    locations = np.random.uniform(low=data_shape["x_min"] + 0.1 * diff,
                                 high=data_shape["x_max"] - 0.1 * diff)
    # TODO: init steepness
    return ChangePoints(kernels=[kernel, kernel1], locations=locations)



def additive(kernel_gen_fn, data_shape, num_active_dims_per_kernel=1, sd=1.):

    D = data_shape["n_dims"]
    assert D >= num_active_dims_per_kernel
    all_dims = np.arange(0, D)
    kernels = []
    for indices in itertools.combinations(all_dims, num_active_dims_per_kernel):
        active_dims = np.array(indices)
        kernels += [kernel_gen_fn(data_shape, sd, active_dims)]

    return kernels







if __name__ == "__main__":

    from src.utils import get_dataset, get_data_shape

    # dataset = get_dataset("airline")
    dataset = get_dataset("housing")
    data_shape = get_data_shape(dataset)
    print(data_shape)


    # kernels = [create_linear(data_shape),
    #            create_rbf(data_shape),
    #            create_period(data_shape), Product([create_linear(data_shape),
    #                                                create_rbf(data_shape),
    #                                                create_period(data_shape)])]

    # kernels = additive(create_linear, data_shape, num_active_dims_per_kernel=1)
    kernels = additive(create_rbf,data_shape, num_active_dims_per_kernel=1)
    kernels = additive(create_period, data_shape, num_active_dims_per_kernel=1, sd=1.)


    for kernel in kernels:
        gpflow.utilities.print_summary(kernel)







