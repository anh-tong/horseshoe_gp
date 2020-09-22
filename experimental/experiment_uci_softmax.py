import os
import sys

from src.experiment_tf import *
from src.utils import create_logger
from gpflow.kernels import RBF, Sum, RationalQuadratic as RQ
from src.sparse_selector_tf import BaseSparseSelector
from gpflow import Parameter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# BASE KERNEL config here
BASE_FN = [create_rbf, create_period]


class SoftmaxSelector(BaseSparseSelector):

    def __init__(self, dim):
        super().__init__(dim)
        self.g = Parameter(tf.random.normal((self.dim, 1)))


    def kl_divergence(self):
        return to_default_float(0.0)

    def sample(self):
        return tf.nn.softmax(self.g, axis=0)


def create_kernels():
    def create_per():
        return Periodic(base_kernel=RBF())
    k1 = Product([RBF(), RQ()])
    k2 = Sum([Product([RBF(), RQ()]), RBF()])
    k3 = Sum([Product([RBF(), RQ()]), create_per()])
    k4 = Sum([create_per(), RQ(), RBF()])
    k5 = Sum([create_per(), RQ(), RBF()])
    k6 = Sum([create_per(), create_per(), RBF()])
    k7 = Sum([Product([RBF(), create_per()]), RBF()])
    k8 = Sum([Product([create_per(), RQ()]), RBF()])
    k9 = Sum([Product([create_per(), RBF()]), RBF()])
    k10 = Product([create_per(), RBF(), RBF()])
    k11 = Product([create_per(), RBF(), RQ()])
    k12 = Product([Sum([create_per(), RQ()]), RBF()])

    return [
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
        k9,
        k10,
        k11,
        k12]



def create_model(inducing_point, data_shape, num_data, n_inducing, kernel_order, repetition):
    kernels = create_kernels()
    print("Number of kernels is {}".format(len(kernels)))
    gps = []
    for kernel in kernels:
        gp = SVGP(kernel,
                  likelihood=None,
                  inducing_variable=inducing_point,
                  q_mu=np.random.randn(n_inducing, 1))
        gps.append(gp)

    selector = SoftmaxSelector(dim=len(gps))
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data)

    return model


if __name__ == "__main__":
    date = sys.argv[1]
    dataset_name = sys.argv[2]
    n_inducing = 150
    batch_size = 128
    kernel_order = 2
    n_iter =30000
    lr=0.01
    repetition = 2


    unique_name = create_unique_name(date, dataset_name, kernel_order, repetition)

    logger = create_logger("../log", unique_name, __file__)
    print("Running softmax")
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

    ckpt_dir = "../model/softmax_{}".format(unique_name)

    train_and_test(model, train_iter, x_test, y_test, dataset.std_y_train, ckpt_dir,
                   n_iter=n_iter,
                   lr=lr,
                   logger=logger)
