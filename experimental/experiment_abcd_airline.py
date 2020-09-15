from src.experiment import *
from src.kernels import create_linear, create_rbf, create_period
import matplotlib.pyplot as plt
from gpflow.config import set_default_jitter
from gpflow.utilities import print_summary


# plotting style
import matplotlib
matplotlib.rcParams.update({'font.size':12,'figure.subplot.bottom':0.125})
matplotlib.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    # "font.serif": ["Palatino"],
})
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# import seaborn as sns
# sns.set_style( {"text.usetex":True,
#                 "font.family":"serif"})


set_default_jitter(1e-3)

# set random seed for reproducibility
tf.random.set_seed(123)
np.random.seed(123)

golden_ratio = (1 + 5 ** 0.5) / 2

## BASE KERNEL config here
BASE_FN = [create_linear, create_rbf, create_period]


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

    selector = HorseshoeSelector(dim=len(gps))
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data)

    return model

def plot_weights(selector):
    w = selector.sample()
    w = w.numpy().squeeze()
    print(w)
    plt.figure(figsize=(golden_ratio*2, 1))
    plt.bar(np.arange(24), w)
    plt.xticks([])
    plt.xlim(-1, 24)
    plt.xlabel(r"kernels $k_i$")
    plt.ylabel(r"$w_i$")
    plt.savefig("../figure/airline_weight.png", dpi=300, bbox_inches="tight")
    plt.savefig("../figure/airline_weight.pdf", dpi=300, bbox_inches="tight")

def plot_abcd(x_train, y_train, x_test, y_test, x_extra, mu, lower, upper):
    plt.figure(figsize=(3 * golden_ratio, 3))
    plt.plot(x_train, y_train, "k.", label="train")
    plt.plot(x_test, y_test, "*", label="test")
    plt.plot(x_extra, mu, label="predict")
    plt.xticks([1950, 1952, 1954, 1956, 1958, 1960])
    plt.yticks([200, 400, 600])
    plt.xlim(1949, 1961)
    plt.ylim(60, 690)
    sep = x_train[-1].numpy()
    plt.plot([sep, sep], [60, 690], "k--", alpha=0.5 )
    plt.fill_between(x_extra.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2, label="confidence")
    plt.legend(loc="upper left")
    plt.savefig("../figure/airline.png", dpi=300, bbox_inches="tight")
    plt.savefig("../figure/airline.pdf", dpi=300, bbox_inches="tight")


def plot_decompostion(selector, gps, x_extra, n_components=3):

    w = selector.sample()
    w = w.numpy().squeeze()
    print(w)
    sorted_index = np.argsort(w)
    selected_gps = []
    for i in sorted_index[-n_components:]:
        print(w[i])
        print_summary(gps[i].kernel)
        selected_gps.append(gps[i])
        mu, var = gps[i].predict_f(x_extra)
        lower = mu - 1.96*tf.sqrt(var)
        upper = mu + 1.96 * tf.sqrt(var)
        mu, var = mu.numpy(), var.numpy()
        lower, upper = lower.numpy(), upper.numpy()
        plt.figure(figsize=(1.5*golden_ratio, 1.5))
        plt.plot(x_extra, mu)
        plt.fill_between(x_extra.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2)
        plt.xticks([1950, 1955, 1960])
        plt.savefig("../figure/airline_c_{}.png".format(i), dpi=300, bbox_inches="tight")
        plt.savefig("../figure/airline_c_{}.pdf".format(i), dpi=300, bbox_inches="tight")






if __name__ == "__main__":
    # All parameters are here
    date = "0904"
    dataset_name = "airline"
    kernel_order = 2
    repetition = 2
    selector = "horseshoe"
    batch_size = 128
    n_iter = 30000
    lr = 0.01
    n_inducing = 100

    # train or load
    load = True
    if load:
        n_iter = 0

    unique_name = create_unique_name(date,
                                     dataset_name,
                                     kernel_order,
                                     repetition)

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

    # train
    ckpt_dir = "../model/{}".format(unique_name)

    model = train(model, train_iter, ckpt_dir, n_iter=n_iter, lr=lr, dataset=dataset)

    x_extra = np.linspace(x_train[0], x_test[-1], 300)
    mu, var = model.predict_y(x_extra)
    lower = mu - 1.96 * tf.sqrt(var)
    upper = mu + 1.96 * tf.sqrt(var)
    mu, lower, upper = mu.numpy(), lower.numpy(), upper.numpy()

    x, y = dataset.retrieve()
    print(np.min(x), np.max(x))
    print(np.min(y), np.max(y))
    x_train = dataset.x_train
    x_test = dataset.x_test
    y_train = dataset.y_train
    y_test = dataset.y_test
    plot_weights(model.selector)
    # plot_decompostion(model.selector, list(model.gps), x_extra)
    # plot_abcd(x_train, y_train, x_test, y_test, x_extra, mu, lower, upper)
    plt.show()
