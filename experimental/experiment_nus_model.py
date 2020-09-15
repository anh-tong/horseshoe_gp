from src.experiment_tf import *
from src.sparse_selector_tf import BaseSparseSelector
from gpflow import Parameter
from gpflow.kernels import Sum, RationalQuadratic as RQ, RBF, Periodic
import matplotlib.pyplot as plt

# plotting style
import matplotlib
matplotlib.rcParams.update({'font.size':14,'figure.subplot.bottom':0.125})
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import seaborn as sns
sns.set_style( {"font.family":"serif"})

# set random seed for reproducibility
tf.random.set_seed(123)
np.random.seed(123)

golden_ratio = (1 + 5 ** 0.5) / 2


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


def create_model(inducing_point, data_shape, num_data, n_inducing):
    kernels = create_kernels()
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
    plt.savefig("../figure/gefcom_test_softmax.png", dgp=300, bbox_inches="tight")
    plt.savefig("../figure/gefcom_test_softmax.pdf", dgp=300, bbox_inches="tight")


if __name__ == "__main__":

    date = "0908"
    dataset_name = "gefcom"
    batch_size = 128
    n_iter = 5000
    lr = 0.01

    load = True
    if load:
        n_iter = 0

    dataset = load_data(dataset_name)

    x_train, y_train = dataset.get_train()
    train_iter = make_data_iteration(x_train, y_train, batch_size=batch_size)
    x_test, y_test = dataset.get_test()
    test_iter = make_data_iteration(x_test, y_test, batch_size=batch_size, shuffle=False)

    data_shape = get_data_shape(dataset)
    inducing_point = init_inducing_points(x_train, 200)

    model = create_model(inducing_point, data_shape, len(y_train), 200)

    ckpt_dir = "../model/gefcom_{}_softmax".format(date)

    model = train(model, train_iter, ckpt_dir, n_iter=n_iter, lr=lr, dataset=dataset)


    w = model.selector.sample()
    i = np.argmax(w.numpy())
    gp = model.gps[i]
    w_i = w[i]
    print(i)
    print(w.numpy())
    mu, var = model.predict_y(x_test)
    ll = model.likelihood.predict_log_density(mu, var, y_test)
    ll = tf.squeeze(tf.reduce_mean(ll))
    rmse = tf.sqrt(tf.reduce_mean(
        tf.square(
            tf.squeeze(mu) -tf.squeeze(y_test)
        )
    )
    )
    print("RMSE: {} \t Test LL: {}".format(rmse.numpy(), ll.numpy()))
    # mu, var = gp.predict_f(x_test)
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
    plot_posterior_on_test(x_test.numpy(), y_test.numpy(), x_min, mu, lower, upper)
    model.likelihood.predict_log_density
    plt.show()