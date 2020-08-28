import matplotlib.pyplot as plt
from gpflow.kernels import Sum
from gpflow.utilities import print_summary

from src.experiment_tf import *
from src.kernels import *
from src.utils import get_data_shape_from_XY

np.random.seed(123)
tf.random.set_seed(123)


def create_kernel_pool(data_shape):
    se1 = create_rbf(data_shape)
    per1 = create_period(data_shape)
    se_per1 = create_se_per(data_shape)
    per_per = Product([create_period(data_shape), create_period(data_shape)])
    return [se1, per1, se_per1, per_per]


def create_true_kernel_2():
    small_per = Periodic(RBF(), period=2.)
    big_per = Periodic(RBF(), period=5.)

    se_per = Product([RBF(lengthscales=2), Periodic(RBF(), period=3.)])
    # per = Periodic(RBF(lengthscales=2.), period=0.5)
    # se2 = RBF(variance=1.5, lengthscales=1.5)
    # per2 = Periodic(RBF(), period=1.5)
    # se_per2 = Product([se2, per2])
    return Sum([
        Product([small_per, big_per]),
        # se_per
    ])
    # return Sum([per, se_per2])


def generate_data(kernel, noise=0.01, n_data=250, lower=-5, upper=5):
    x = np.linspace(lower, upper, n_data)[:, None]
    K = kernel(x)
    mean = np.zeros(n_data)
    cov = K.numpy() + np.eye(n_data) * noise
    sample = np.random.multivariate_normal(mean=mean, cov=cov)
    return x, sample


kernel = create_true_kernel_2()

n_data = 250
n_train = 200
n_extrapol = 50

noise=0.0001
print("introduce noise: {}".format(noise))
X, Y = generate_data(kernel, n_data=n_data, noise=noise)
Y = Y[:, None]

# plt.plot(X, Y)
# plt.show()
# exit(0)

def split():
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_extra = X[n_train:]
    Y_extra = Y[n_train:]
    return X_train, Y_train, X_extra, Y_extra


X, Y, X_extra, Y_extra = split()


data_shape = get_data_shape_from_XY(X, Y)

kernel_pools = create_kernel_pool(data_shape)
kernel_pools.extend(create_kernel_pool(data_shape))
kernel_pools.extend(create_kernel_pool(data_shape))

print("Number of kernel is {}".format(len(kernel_pools)))


def create_sgp_no_regularize():
    inducing_points = init_inducing_points(X, M=100)

    kernel = Sum(kernel_pools)

    likelihood = Gaussian()
    model = SVGP(kernel, likelihood, inducing_variable=inducing_points, num_data=200)
    return model


def create_structral_gp():
    inducing_points = init_inducing_points(X, M=100)
    fix_kernel_variance(kernel_pools)
    gps = []
    for kernel in kernel_pools:
        gp = SVGP(kernel, likelihood=None, inducing_variable=inducing_points)
        gps.append(gp)

    selector = HorseshoeSelector(len(gps))
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data=200)
    return model


def run_justify(load=False, create_model_fn=create_structral_gp, chkpt_dir="../model/justify_our_model"):
    model = create_model_fn()
    optimizer = tf.optimizers.Adam(lr=0.01)
    train_loss = model.training_loss_closure((X, Y))

    chkpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(chkpt, chkpt_dir, max_to_keep=1)

    @tf.function
    def optimize_step():
        optimizer.minimize(train_loss, model.trainable_variables)
        if not isinstance(model, SVGP):
            if isinstance(model.selector, HorseshoeSelector):
                model.selector.update_tau_lambda()

    if not load:
        for i in range(20000):
            optimize_step()
            if i % 1000 == 0:
                print("Iter: {} \t Loss: {:.2f}".format(i, tf.squeeze(train_loss()).numpy()))
        save_path = manager.save()
        print("Save checkpoint in {}".format(save_path))
    else:
        chkpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from save model in {}".format(manager.latest_checkpoint))
        else:
            print("No saved model available")

    x_test = np.linspace(-5.1, 5.1, 200)[:, None]
    mu, var = model.predict_y(x_test)

    lower = mu - 1.96 * tf.sqrt(var)
    upper = mu + 1.96 * tf.sqrt(var)
    mu, var, lower, upper = mu.numpy(), var.numpy(), lower.numpy(), upper.numpy()

    if not isinstance(model, SVGP):
        w = sum([0.2 * model.selector.sample() for i in range(5)])
        w_2 = tf.square(tf.squeeze(w))
        normalized = w_2 / tf.reduce_sum(w_2)
        print("Normalized weight: ", normalized)
    else:
        kernel = model.kernel

        def get_variance(k):
            if isinstance(k, Periodic):
                return k.base_kernel.variance.numpy()
            elif isinstance(k, Product):
                v = 1.
                for j in k.kernels:
                    v *= get_variance(j)
                return v
            else:
                return k.variance.numpy()

        w_2 = np.array([get_variance(k) ** 2 for k in kernel.kernels])
        normalized = w_2 / np.sum(w_2)
        print("Normalized weight: ", normalized)
    for kernel in kernel_pools:
        print_summary(kernel)

    plt.figure()
    plt.plot(X, Y, "+")
    plt.plot(x_test, mu)
    plt.plot(X_extra, Y_extra)
    plt.fill_between(x_test.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2)


# OUR MODEL
chkpt_dir = "../model/justify_our_model_2"
run_justify(create_model_fn=create_structral_gp, chkpt_dir=chkpt_dir, load=False)

# BASELINE MODEL
kernel_pools = create_kernel_pool(data_shape)
kernel_pools.extend(create_kernel_pool(data_shape))
kernel_pools.extend(create_kernel_pool(data_shape))
chkpt_dir = "../model/justify_baseline_2"
run_justify(create_model_fn=create_sgp_no_regularize,
            chkpt_dir=chkpt_dir,
            load=True)
plt.show()
