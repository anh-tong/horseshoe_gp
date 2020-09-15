import os
import sys

from src.experiment_tf import *
from src.utils import create_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def create_kernels(data_shape):

    """Additive kernel for this models"""
    kernels = []
    # CONCRETE has additive order 1
    order = 1
    print("Create additive order: {}".format(order))
    ks = additive(create_rbf, data_shape=data_shape, num_active_dims_per_kernel=1)
    kernels.extend(ks)
    ks2 = additive(create_rbf, data_shape, num_active_dims_per_kernel=1)
    kernels.extend(ks2)
    fix_kernel_variance(kernels)
    return kernels


def create_model(inducing_point, data_shape, num_data, n_inducing):
    kernels = create_kernels(data_shape)
    fix_kernel_variance(kernels)
    print("Number of kernels is {}".format(len(kernels)))
    gps = []
    for kernel in kernels:
        gp = SVGP(kernel,
                  likelihood=None,
                  inducing_variable=inducing_point,
                  q_mu=np.random.randn(n_inducing, 1))
        gps.append(gp)

    selector = HorseshoeSelector(dim=len(gps))
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data)

    return model


if __name__ == "__main__":
    date = sys.argv[1]
    dataset_name = "concrete"
    n_inducing = 200
    batch_size = 128
    n_iter =50000
    lr= 0.005

    date = "{}_additive".format(date)


    unique_name = create_unique_name(date, dataset_name, kernel_order=None, repetition=None)

    logger = create_logger("../log", unique_name, __file__)

    # data
    dataset = load_data(dataset_name)
    x_train, y_train = dataset.get_train()
    train_iter = make_data_iteration(x_train, y_train, batch_size=batch_size)
    x_test, y_test = dataset.get_test()
    test_iter = make_data_iteration(x_test, y_test, batch_size=batch_size, shuffle=False)

    inducing_point = init_inducing_points(x_train, M=n_inducing)

    data_shape = get_data_shape(dataset)

    # create model
    model = create_model(inducing_point,
                         data_shape=data_shape,
                         num_data=len(y_train),
                         n_inducing=n_inducing,
                         )

    ckpt_dir = "../model/{}".format(unique_name)

    train_and_test(model, train_iter, x_test, y_test, dataset.std_y_train, ckpt_dir,
                   n_iter=n_iter,
                   lr=lr,
                   logger=logger)
