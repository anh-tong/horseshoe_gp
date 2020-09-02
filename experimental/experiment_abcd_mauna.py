from src.experiment_tf import *
import matplotlib.pyplot as plt
from gpflow.config import set_default_jitter

set_default_jitter(1e-3)

# set random seed for reproducibility
tf.random.set_seed(123)
np.random.seed(123)

golden_ratio = (1 + 5 ** 0.5) / 2


def plot_abcd(x_train, y_train, x_test, y_test, x_extra, mu, lower, upper):
    plt.figure(figsize=(3 * golden_ratio, 3))
    plt.plot(x_train, y_train, "k.")
    plt.plot(x_test, y_test, "*")
    plt.plot(x_extra, mu)
    plt.fill_between(x_extra.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2 )


if __name__ == "__main__":
    # All parameters are here
    date = "0901"
    dataset_name = "mauna"
    kernel_order = 2
    repetition = 2
    selector = "horseshoe"
    batch_size = 128
    n_iter = 30000
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
    dataset = load_data(dataset_name)
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