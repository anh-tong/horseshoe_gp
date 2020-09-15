import logging

import numpy as np
import tensorflow as tf
from gpflow import set_trainable
from gpflow.kernels import Periodic, Product
from gpflow.likelihoods import Gaussian
from gpflow.models import SVGP
from gpflow.utilities import to_default_float

from src.kernel_generator import Generator
from src.kernels import create_rbf, create_period, additive, create_se_per
from src.sparse_selector import HorseshoeSelector, SpikeAndSlabSelector
from src.structural_sgp import StructuralSVGP
from src.utils import get_dataset, get_data_shape


def load_data(name="airline"):
    dataset = get_dataset(name)
    return dataset


def init_inducing_points(x, M=100):
    x_perm = tf.random.shuffle(x)
    return x_perm[:M]


def make_data_iteration(x, y, batch_size=128, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.repeat().shuffle(buffer_size=1024, seed=123)

    data_iter = iter(dataset.batch(batch_size))
    return data_iter


def fix_kernel_variance(kernels):
    """kernel variance is fixed. the variance is decided by selector"""
    for kernel in kernels:
        if isinstance(kernel, Product):
            fix_kernel_variance(kernel.kernels)
        elif isinstance(kernel, Periodic):
            set_trainable(kernel.base_kernel.variance, False)
        else:
            set_trainable(kernel.variance, False)


def create_model(inducing_point, data_shape, num_data, selector="horseshoe", kernel_order=2,
                 repetition=2) -> StructuralSVGP:
    generator = Generator(data_shape
                          , base_fn=[create_rbf, create_period]
                          )
    kernels = []
    for _ in range(repetition):
        kernels.extend(generator.create_upto(upto_order=kernel_order))

#    kernels = additive(create_se_per, data_shape=data_shape, num_active_dims_per_kernel=1)
#    kernels.extend(additive(create_se_per, data_shape=data_shape, num_active_dims_per_kernel=1))
    print("NUMBER OF KERNELS: {}".format(len(kernels)))
    fix_kernel_variance(kernels)
    gps = []
    for kernel in kernels:
        gp = SVGP(kernel, likelihood=None, inducing_variable=inducing_point, q_mu=np.random.randn(100,1))
        gps.append(gp)

    if selector == "horseshoe":
        selector = HorseshoeSelector(dim=len(gps))
        # from src.sparse_selector_tf import TrivialSparseSelector
        # selector = TrivialSparseSelector(len(gps))
    elif selector == "spike_n_slab":
        selector = SpikeAndSlabSelector(dim=len(gps))
    else:
        raise ValueError("Invalid selector name. Pick either [horseshoe] or [spike_n_slab]")

    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data)
    return model


def train_and_test(model,
                   train_iter,
                   x_test,
                   y_test,
                   std_y_train,
                   ckpt_dir,
                   ckpt_feq=1000,
                   n_iter=10000,
                   lr=0.01,
                   logger=logging.getLogger("default")):
    optimizer = tf.optimizers.Adam(lr=lr)
    train_loss = model.training_loss_closure(train_iter)
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    @tf.function
    def optimize_step():
        optimizer.minimize(train_loss, model.trainable_variables)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restore from {} !!!".format(manager.latest_checkpoint))
    else:
        logger.info("Initialize from scratch !!!")

    for i in range(n_iter):
        # optimizer step
        optimize_step()
        # horseshoe update
        if isinstance(model, StructuralSVGP) and isinstance(model.selector, HorseshoeSelector):
            model.selector.update_tau_lambda()

        # save checkpoint
        if (i + 1) % ckpt_feq == 0:
            save_path = manager.save()
            mu, var = model.predict_y(x_test)
            ll = tf.reduce_mean(model.likelihood.predict_log_density(mu, var, y_test)) - np.log(std_y_train)
            RMSE = tf.sqrt(tf.reduce_mean(tf.square(mu - y_test))) * std_y_train
            logger.info("Saved checkpoint for step {}: {}".format(i + 1, save_path))
            logger.info("Iter {} \t Loss: {:.2f} \t Test RMSE:{} \t Test LL{}".format(i,
                                                                                      train_loss().numpy(),
                                                                                      RMSE.numpy(),
                                                                                      ll.numpy()))


def train(model, train_iter, ckpt_dir, ckpt_freq=1000, n_iter=10000, lr=0.01, dataset=None):
    optimizer = tf.optimizers.Adam(lr=lr)

    train_loss = model.training_loss_closure(train_iter)

    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    @tf.function
    def optimize_step():
        optimizer.minimize(train_loss, model.trainable_variables)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restore from {} !!!".format(manager.latest_checkpoint))
    else:
        print("Initialize from scratch !!!")

    for i in range(n_iter):
        # model.elbo(next(train_iter))
        optimize_step()
        # additional update for the horseshoe case
        if isinstance(model, StructuralSVGP) and isinstance(model.selector, HorseshoeSelector):
            model.selector.update_tau_lambda()

        # save check point
        if (i + 1) % ckpt_freq == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(i + 1, save_path))
            x_test, y_test = dataset.get_test()
            test_iter = make_data_iteration(x_test, y_test, batch_size=128, shuffle=False)
            if test_iter is None:
                print("Iter {} \t Loss: {:.2f}".format(i, train_loss().numpy()))
            else:
                error = []
                for x_batch, y_batch in test_iter:
                    mu, var = model.predict_y(x_batch)
                    error += [tf.square(tf.squeeze(mu) - tf.squeeze(y_batch))]
                error = tf.concat(error, axis=0)
                rmse = tf.sqrt(tf.reduce_mean(error))
                print("Iter {} \t Loss: {:.2f} \t RMSE: {:.2f}".format(i, train_loss().numpy(), rmse.numpy()))


    return model


def test_from_checkpoint(date, dataset_name, selector, kernel_order, repetition):
    unique_name = create_unique_name(date, dataset_name, kernel_order, repetition, selector)
    ckpt_dir = "../model/{}".format(unique_name)
    dataset = load_data(dataset_name)
    x_train, y_train = dataset.get_train()
    x_test, y_test = dataset.get_test()

    test_iter = make_data_iteration(x_test, y_test, batch_size=128, shuffle=False)

    data_shape = get_data_shape(dataset)

    inducing_points = init_inducing_points(x_train)

    model = create_model(inducing_points, data_shape,
                         selector=selector,
                         kernel_order=kernel_order,
                         repetition=repetition,
                         num_data=len(y_train))

    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("No available checkpoint")
        return

    # predict
    mu, var, ll = test(test_iter, model)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(mu - y_test)))
    # mean_ll = tf.reduce_mean(ll)

    print("RMSE: {} ".format(rmse.numpy()))

    # plot for 1D case
    n_test = 300
    x_test = tf.linspace(tf.reduce_min(x_train), tf.reduce_max(x_train), n_test)[:, None]
    y_test = tf.zeros_like(x_test)
    plot_iter = make_data_iteration(x_test, y_test, shuffle=False)
    mu, var, _ = test(plot_iter, model)
    lower = mu - 1.96 * tf.sqrt(var)
    upper = mu + 1.96 * tf.sqrt(var)

    plot(x_train, y_train, x_test.numpy(), mu.numpy(), lower.numpy(), upper.numpy())


def test(test_iter, model: StructuralSVGP):
    mus = []
    vars = []
    # lls = []
    for x_batch, y_batch in test_iter:
        pred_mean, pred_var = model.predict_y(x_batch, full_cov=False, full_output_cov=False)
        # ll = model.likelihood.log_prob(pred_mean, pred_var, y_batch)
        mus.append(pred_mean)
        vars.append(pred_var)
        # lls.append(ll)

    mu = tf.concat(mus, axis=0)
    var = tf.concat(vars, axis=0)
    # ll = tf.concat(lls, axis=0)
    return mu, var, None


def plot(x, y, x_prime, y_prime, upper, lower):
    import matplotlib.pyplot as plt
    plt.plot(x, y, "+")
    plt.plot(x_prime, y_prime)
    plt.fill_between(x_prime.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.2)
    plt.show()


def run_train_and_test(date,
                       dataset_name,
                       selector="horseshoe",
                       kernel_order=2,
                       repetition=2,
                       n_iter=50000,
                       lr=0.01,
                       batch_size=128,
                       logger=logging.getLogger("default")
                       ):
    unique_name = create_unique_name(date, dataset_name, kernel_order, repetition, selector)

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
    train_and_test(model, train_iter, x_test, y_test, dataset.std_y_train, ckpt_dir, n_iter=n_iter, lr=lr,
                   logger=logger)


def run(date,
        dataset_name,
        selector="horseshoe",
        kernel_order=2,
        repetition=2,
        n_iter=50000,
        lr=0.01,
        batch_size=128,
        plot_n_predict=True,
        ):
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

    if plot_n_predict:
        # predict
        mu, var, ll = test(test_iter, model)
        rmse = tf.sqrt(tf.reduce_mean(tf.square(mu - y_test)))
        # mean_ll = tf.reduce_mean(ll)

        print("RMSE: {} ".format(rmse.numpy()))

        # plot for 1D case
        n_test = 300
        x_test = tf.linspace(tf.reduce_min(x_train), tf.reduce_max(x_train), n_test)[:, None]
        y_test = tf.zeros_like(x_test)
        plot_iter = make_data_iteration(x_test, y_test, shuffle=False)
        mu, var, _ = test(plot_iter, model)
        lower = mu - 1.96 * tf.sqrt(var)
        upper = mu + 1.96 * tf.sqrt(var)

        plot(x_train, y_train, x_test.numpy(), mu.numpy(), lower.numpy(), upper.numpy())


def create_unique_name(date, dataset_name, kernel_order, repetition):
    name = "{}_{}_kernel_{}{}".format(dataset_name, date,kernel_order, repetition)
    return name


if __name__ == "__main__":
    # run(date="0901", dataset_name="solar", lr=0.01, n_iter=10000) # TODO: have a problem running this
    run(date="0831", dataset_name="airline")
    # run(name="mauna")
    # run(name="wheat-price")
    run(date="0901", dataset_name="gefcom", lr=0.01, n_iter=0)
    # test_from_checkpoint(date="0831_4", dataset_name="solar", selector="horseshoe", kernel_order=2, repetition=2)
