from src.experiment_tf import *
import matplotlib.pyplot as plt
from gpflow.config import set_default_jitter
from gpflow.kernels import ChangePoints, White
from src.kernels import create_changepoint
from src.utils import ABCDDataset, standardize

set_default_jitter(1e-3)

# set random seed for reproducibility
tf.random.set_seed(1)
np.random.seed(1)

golden_ratio = (1 + 5 ** 0.5) / 2

import scipy.io as sio
class SolarDataSet(ABCDDataset):

    def retrieve(self):
        data = sio.loadmat(self.data_dir)
        x = data["X"]
        y = data["y"]
        y, _, _ = standardize(y)
        self.n = x.shape[0]
        self.d = x.shape[1]
        return x, y


def create_model(inducing_point, data_shape, num_data, selector="horseshoe", kernel_order=2, repetition=2) -> StructuralSVGP:

    print("Create model with changepoint")
    ## Add change point of this data sets
    generator = Generator(data_shape, base_fn=[create_rbf, create_period])
    kernels = []
    for _ in range(repetition):
        kernels.extend(generator.create_upto(upto_order=kernel_order))

    # # create changepoint
    # cps = []
    # for side in ["left", "right"]:
    #     for base_fn in [create_rbf, create_period]:
    #         cp = create_changepoint(data_shape, side=side, base_fn=base_fn)
    #         cps.append(cp)
    # kernels.extend(cps)

    print("NUMBER OF KERNELS: {}".format(len(kernels)))

    gps = []
    for kernel in kernels:
        gp = SVGP(kernel, likelihood=None, inducing_variable=inducing_point, q_mu=np.random.randn(100, 1))
        gps.append(gp)

    selector = HorseshoeSelector(dim=len(gps))
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data)
    return model


def plot_abcd(x_train, y_train, x_test, y_test, x_extra, mu, lower, upper):
    plt.figure(figsize=(3 * golden_ratio, 3))
    plt.plot(x_train, y_train, "k.")
    plt.plot(x_test, y_test, "*")
    plt.plot(x_extra, mu)
    plt.fill_between(x_extra.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2 )


if __name__ == "__main__":
    # All parameters are here
    date = "0901"
    dataset_name = "solar"
    kernel_order = 2
    repetition = 2
    selector = "horseshoe"
    batch_size = 300
    n_iter = 20000
    lr = 0.01

    # train or load
    load = False
    if load:
        n_iter = 0

    unique_name = create_unique_name(date,
                                     dataset_name,
                                     kernel_order,
                                     repetition,
                                     selector)

    # data
    dataset = SolarDataSet("../data/02-solar.mat")
    x_train, y_train = dataset.get_train()
    train_iter = make_data_iteration(x_train, y_train, batch_size=batch_size)
    x_test, y_test = dataset.get_test()
    test_iter = make_data_iteration(x_test, y_test, batch_size=batch_size, shuffle=False)

    inducing_point = init_inducing_points(x_train)

    data_shape = get_data_shape(dataset)

    # create model
    model = create_model(inducing_point,
                         selector=selector,
                         data_shape=data_shape,
                         num_data=len(y_train),
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