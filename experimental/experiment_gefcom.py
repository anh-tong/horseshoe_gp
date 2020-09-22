from src.experiment_tf import *
from src.kernels import create_linear, create_rbf, create_period
import matplotlib.pyplot as plt
from gpflow.config import set_default_jitter
from gpflow.utilities import print_summary

# plotting style
import matplotlib
matplotlib.rcParams.update({'font.size':14,'figure.subplot.bottom':0.125})
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import seaborn as sns
sns.set_style( {"font.family":"serif"})

set_default_jitter(1e-3)

# set random seed for reproducibility
tf.random.set_seed(123)
np.random.seed(123)

golden_ratio = (1 + 5 ** 0.5) / 2

## BASE KERNEL config here
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

    selector = HorseshoeSelector(dim=len(gps))
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data)

    return model


def plot_train_gefcom(x, y, x_min):
    plt.figure(figsize=(10, 2.5))
    plt.plot(2004 + (x -x_min) / 365, y, "k.", markersize=0.4)
    plt.xlim(2004, 2008.5)
    plt.xticks([2004, 2005, 2006, 2007, 2008])
    plt.ylim(-2.5, 4.8)
    plt.savefig("../figure/gefcom_train.png", dpi=300, bbox_inches="tight")
    plt.savefig("../figure/gefcom_train.pdf", dpi=300, bbox_inches="tight")

def plot_posterior_on_test(x_test, y_test, x_min, mu, lower, upper):

    x_test = x_test.squeeze()
    y_test = y_test.squeeze()
    mu = mu.squeeze()
    upper = upper.squeeze()
    lower = lower.squeeze()
    plt.figure(figsize=(10,2.5))
    x_transformed = 2004 + (x_test - x_min) / 365
    sorted_indices = np.argsort(x_transformed)
    x_transformed = x_transformed[sorted_indices]
    mu = mu[sorted_indices]
    y_test = y_test[sorted_indices]
    lower = lower[sorted_indices]
    upper = upper[sorted_indices]
    plt.xlim(2004, 2008.5)
    plt.xticks([2004, 2005, 2006, 2007, 2008])
    plt.ylim(-2.5, 4.8)
    plt.plot(x_transformed, y_test, ".", markersize=0.4, label="test")
    plt.plot(x_transformed, mu, linewidth=0.5, alpha=0.7, label="predict mean")
    plt.fill_between(x_transformed.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2, label="confidence")
    lg = plt.legend(loc="upper right")
    plt.savefig("../figure/gefcom_test_ours.png", dgp=300, bbox_inches="tight")
    plt.savefig("../figure/gefcom_test_ours.pdf", dgp=300, bbox_inches="tight")

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
        # plt.xticks([1950, 1955, 1960])
        plt.savefig("../figure/gefcom_c_{}.png".format(i), dpi=300, bbox_inches="tight")
        plt.savefig("../figure/gefcom_c_{}.pdf".format(i), dpi=300, bbox_inches="tight")

def plot_weights(selector):
    w = selector.sample()
    w = w.numpy().squeeze()
    plt.figure(figsize=(golden_ratio*2, 1))
    plt.bar(np.arange(12), np.abs(w))
    plt.xticks([])
    plt.xlim(-1, 12)
    plt.xlabel(r"kernels $k_i$")
    plt.ylabel(r"$w_i$")
    plt.savefig("../figure/gefcom_weight.png", dpi=300, bbox_inches="tight")
    plt.savefig("../figure/gef_weight.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # All parameters are here
    date = "0903"
    dataset_name = "gefcom"
    kernel_order = 2
    repetition = 2
    batch_size = 128
    n_iter = 15000
    lr = 0.01
    n_inducing = 300

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

    mu, var = model.predict_y(x_test)

    ll = model.likelihood.predict_log_density(mu, var, y_test)
    ll = tf.squeeze(tf.reduce_mean(ll))
    rmse = tf.sqrt(tf.reduce_mean(
        tf.square(
            tf.squeeze(mu) - tf.squeeze(y_test)
        )
    )
    )
    print("RMSE: {} \t Test LL: {}".format(rmse.numpy(), ll.numpy()))

    lower = mu - 1.96 * tf.sqrt(var)
    upper = mu + 1.96 * tf.sqrt(var)
    mu, lower, upper = mu.numpy(), lower.numpy(), upper.numpy()

    x_train = dataset.x_train
    x_test = dataset.x_test
    y_train = dataset.y_train
    y_test = dataset.y_test
    x, y = dataset.retrieve()
    x_min = np.min(x)
    x_max = np.max(x)
    print(2004 + (x_max - x_min) / 365)
    plot_weights(model.selector)
    plot_decompostion(model.selector, model.gps, x_extra=x_test.numpy(), n_components=12)
    #plot_train_gefcom(x_train, y_train, x_min)
    #plot_posterior_on_test(x_test.numpy(), y_test.numpy(), x_min, mu, lower, upper)
    plt.show()