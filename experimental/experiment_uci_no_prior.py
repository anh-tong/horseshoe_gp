import os
import sys

from gpflow.kernels import Sum

from src.experiment_tf import *
from src.utils import create_logger
from src.sparse_selector_tf import LinearSelector

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# BASE KERNEL config here
BASE_FN = [create_rbf, create_period]


def create_kernels(data_shape, kernel_order, repetition):
    generator = Generator(data_shape, base_fn=BASE_FN)
    kernels = []
    for _ in range(repetition):
        generated = generator.create_upto(kernel_order)
        kernels.extend(generated)
    return kernels



def create_model(inducing_point, data_shape, num_data, n_inducing, kernel_order, repetition):
    kernels = create_kernels(data_shape, kernel_order, repetition)
    fix_kernel_variance(kernels)
    print("Number of kernels is {}".format(len(kernels)))
    gps = []
    for kernel in kernels:
        gp = SVGP(kernel,
                  likelihood=None,
                  inducing_variable=inducing_point,
                  q_mu=np.random.randn(n_inducing, 1))
        gps.append(gp)

    selector = LinearSelector(dim=len(gps))
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data)

    return model

if __name__ == "__main__":
    date = sys.argv[1]
    dataset_name = sys.argv[2]
    n_inducing = 100
    batch_size = 128
    kernel_order = 2
    n_iter =30000
    lr=0.01
    repetition = 2


    unique_name = create_unique_name(date, dataset_name, kernel_order, repetition)

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
                         kernel_order=kernel_order,
                         repetition=repetition
                         )

    ckpt_dir = "../model/no_prior_{}".format(unique_name)

    train_and_test(model, train_iter, x_test, y_test, dataset.std_y_train, ckpt_dir,
                   n_iter=n_iter,
                   lr=lr,
                   logger=logger)
