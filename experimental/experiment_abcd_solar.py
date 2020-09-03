import matplotlib.pyplot as plt
from gpflow.config import set_default_jitter

import scipy.io as sio
from src.experiment_tf import *
from src.kernels import create_linear, create_rbf, create_period
from src.utils import ABCDDataset, standardize
from gpflow.kernels import ChangePoints, White

set_default_jitter(1e-3)

# set random seed for reproducibility
tf.random.set_seed(1)
np.random.seed(1)

golden_ratio = (1 + 5 ** 0.5) / 2

# BASE KERNEL config here
BASE_FN = [create_linear, create_rbf, create_period]


def create_kernels(data_shape, kernel_order, repetition):
    generator = Generator(data_shape, base_fn=BASE_FN)
    kernels = []
    for _ in range(repetition):
        generated = generator.create_upto(kernel_order)
        kernels.extend(generated)

    # fix kernel variance
    fix_kernel_variance(kernels)

    # r1 = create_se_per(data_shape)
    # l1 = White(variance=1e-3)
    # k1 = [l1, r1]
    # fix_kernel_variance(k1)
    # cp1 = ChangePoints(kernels=k1, locations=0.5*(data_shape["x_max"] + data_shape["x_min"]))
    #
    # r2 = White(variance=1e-3)
    # l2 = create_se_per(data_shape)
    # k2 = [l2, r2]
    # fix_kernel_variance(k2)
    # cp2 = ChangePoints(kernels=k2, locations=0.5*(data_shape["x_max"] + data_shape["x_min"]))
    #
    # kernels += [cp1]
    # kernels += [cp2]


    return kernels


def create_model(inducing_point, data_shape, num_data, n_inducing, kernel_order, repetition):
    kernels = create_kernels(data_shape, kernel_order, repetition)
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

class SolarDataSet(ABCDDataset):

    def retrieve(self):
        data = sio.loadmat(self.data_dir)
        x = data["X"]
        y = data["y"]
        y, _, _ = standardize(y)
        self.n = x.shape[0]
        self.d = x.shape[1]
        return x, y


def plot_abcd(x_train, y_train, x_test, y_test, x_extra, mu, lower, upper):
    plt.figure(figsize=(3 * golden_ratio, 3))
    plt.plot(x_train, y_train, "k.")
    plt.plot(x_test, y_test, "*")
    plt.plot(x_extra, mu)
    plt.fill_between(x_extra.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2)
    plt.savefig("../figure/solar.png", dpi=300, bbox_inches="tight")
    plt.savefig("../figure/solar.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # All parameters are here
    date = "0903-2"
    dataset_name = "solar"
    kernel_order = 2
    repetition = 1
    n_inducing = 200
    batch_size = 128
    n_iter = 20000
    lr = 0.1

    # train or load
    load = False
    if load:
        n_iter = 0

    unique_name = create_unique_name(date,
                                     dataset_name,
                                     kernel_order,
                                     repetition)

    # data
    dataset = SolarDataSet("../data/02-solar.mat")
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

    # train
    ckpt_dir = "../model/{}".format(unique_name)

    model = train(model, train_iter, ckpt_dir, n_iter=n_iter, lr=lr, dataset=dataset)

    x_extra = np.linspace(x_train[0], x_test[-1], 300)
    mu, var = model.predict_y(x_extra)
    lower = mu - tf.sqrt(var)
    upper = mu + tf.sqrt(var)
    mu, lower, upper = mu.numpy(), lower.numpy(), upper.numpy()

    x_train = dataset.x_train
    x_test = dataset.x_test
    y_train = dataset.y_train
    y_test = dataset.y_test
    plot_abcd(x_train, y_train, x_test, y_test, x_extra, mu, lower, upper)
    plt.show()
