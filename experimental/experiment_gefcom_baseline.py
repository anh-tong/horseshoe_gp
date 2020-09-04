# plotting style
import matplotlib
import matplotlib.pyplot as plt
from gpflow.config import set_default_jitter
from gpflow.kernels import Sum

from src.experiment_tf import *
from src.kernels import create_rbf, create_period

matplotlib.rcParams.update({'font.size': 14, 'figure.subplot.bottom': 0.125})
# from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
import seaborn as sns

sns.set_style({"font.family": "serif"})

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


def create_no_prior_model(inducing_point, data_shape, num_data, kernel_order, repetition) -> SVGP:
    kernels = create_kernels(data_shape, kernel_order, repetition)
    print("Number of kernels is {}".format(len(kernels)))
    sum_kernel = Sum(kernels)
    likelihood = Gaussian()
    model = SVGP(kernel=sum_kernel,
                 likelihood=likelihood,
                 num_data=num_data,
                 inducing_variable=inducing_point
                 )

    return model


def create_se_model(inducing_point, data_shape, num_data):
    kernel = create_rbf(data_shape)
    likelihood = Gaussian()
    model = SVGP(kernel=kernel,
                 likelihood=likelihood,
                 num_data=num_data,
                 inducing_variable=inducing_point)
    return model


def plot_posterior_on_test(x_test, y_test, x_min, mu, lower, upper, name="no_prior"):
    x_test = x_test.squeeze()
    y_test = y_test.squeeze()
    mu = mu.squeeze()
    upper = upper.squeeze()
    lower = lower.squeeze()
    plt.figure(figsize=(10, 2.5))
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
    plt.savefig("../figure/gefcom_test_{}.png".format(name), dgp=300, bbox_inches="tight")
    plt.savefig("../figure/gefcom_test_{}.pdf".format(name), dgp=300, bbox_inches="tight")


if __name__ == "__main__":
    ## LOAD OR TRAIN
    load = True
    ## SELECT BETWEEN no_prior or se
    # name = "no_prior"
    name = "se"

    # All parameters are here
    dataset_name = "gefcom"
    kernel_order = 2
    repetition = 2
    batch_size = 128
    n_iter = 15000
    lr = 0.01
    n_inducing = 300



    # data
    dataset = load_data(dataset_name)
    x_train, y_train = dataset.get_train()
    train_iter = make_data_iteration(x_train, y_train, batch_size=batch_size)
    x_test, y_test = dataset.get_test()
    test_iter = make_data_iteration(x_test, y_test, batch_size=batch_size, shuffle=False)

    inducing_point = init_inducing_points(x_train, M=n_inducing)

    data_shape = get_data_shape(dataset)

    # create model
    # NO PRIOR
    if name == "no_prior":
        model = create_no_prior_model(inducing_point,
                                      data_shape=data_shape,
                                      num_data=len(y_train),
                                      kernel_order=2,
                                      repetition=2
                                      )
    else:
        # ONLY SE
        model = create_se_model(inducing_point, data_shape, len(y_train))



    # train
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, "../model/gefcom_{}".format(name), max_to_keep=1)
    optimizer = tf.optimizers.Adam(lr=lr)
    train_loss = model.training_loss_closure(train_iter)


    @tf.function
    def optimize_step():
        optimizer.minimize(train_loss, model.trainable_variables)

    if not load:
        for i in range(n_iter):
            optimize_step()
            if i % 1000 == 0 and not i == 0:
                print("Iter {} \t Loss: {:.2f}".format(i, train_loss().numpy()))

        save_path = manager.save()
        print("Save model to {}".format(save_path))
    else:
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("restoring model")
        else:
            print("cannot find model to restore")
            exit(0)

    mu, var = model.predict_y(x_test)
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
    plot_posterior_on_test(x_test.numpy(), y_test.numpy(), x_min, mu, lower, upper, name=name)
    plt.show()
