import os
import sys

from gpflow.likelihoods import Bernoulli

from src.experiment_tf import *
from src.utils import create_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def create_kernels(data_shape, additive_order):
    kernels = []
    for order in additive_order:
        print("Create additive order: {}".format(order))
        ks = additive(create_rbf, data_shape=data_shape, num_active_dims_per_kernel=order)
        kernels.extend(ks)
    fix_kernel_variance(kernels)
    return kernels


def create_model_from_kernel(kernels, inducing_point, num_data):
    print("number of kernels: {}".format(len(kernels)))

    gps = []
    for kernel in kernels:
        gp = SVGP(kernel, likelihood=None, inducing_variable=inducing_point, q_mu=np.random.random((100, 1)))
        gps += [gp]

    selector = HorseshoeSelector(dim=len(gps))
    likelihood = Gaussian()
    model = StructuralSVGP(gps, selector, likelihood, num_data=num_data)
    return model


def create_classification_model(kernels, inducing_point, num_data):
    print("number of kernels: {}".format(len(kernels)))

    gps = []
    for kernel in kernels:
        gp = SVGP(kernel, likelihood=None, inducing_variable=inducing_point, q_mu=np.random.random((100, 1)))
        gps += [gp]

    selector = HorseshoeSelector(dim=len(gps))
    likelihood = Bernoulli()
    model = StructuralSVGP(gps, selector, likelihood, num_data=num_data)
    return model


def create_regression_model(inducing_point, data_shape, additive_order, num_data, classify):
    print("New implementation of creating model!!!")
    kernels = create_kernels(data_shape, additive_order)
    if classify:
        return create_classification_model(kernels, inducing_point, num_data)
    else:
        return create_model_from_kernel(kernels, inducing_point, num_data)


def run_additive(date,
                 dataset_name,
                 additive_order,
                 classify=False,
                 n_iter=50000,
                 lr=0.01,
                 ckpt_freq=1000,
                 batch_size=128,
                 logger=logging.getLogger("default")

                 ):
    unique_name = create_unique_name(date, dataset_name, None, None)

    # data
    dataset = load_data(dataset_name)
    x, y = dataset.retrieve()
    print(x[1:3,:])
    x_train, y_train = dataset.get_train()
    train_iter = make_data_iteration(x_train, y_train, batch_size=batch_size)
    x_test, y_test = dataset.get_test()
    test_iter = make_data_iteration(x_test, y_test, batch_size=batch_size, shuffle=False)

    inducing_point = init_inducing_points(x_train)

    data_shape = get_data_shape(dataset)

    # create model
    model = create_regression_model(inducing_point,
                                    additive_order=additive_order,
                                    data_shape=data_shape,
                                    num_data=len(y_train),
                                    classify=classify
                                    )

    # train
    ckpt_dir = "../model/{}".format(unique_name)
    optimizer = tf.optimizers.Adam(lr=lr)
    train_loss = model.training_loss_closure(train_iter)
    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    @tf.function
    def optimize_step():
        optimizer.minimize(train_loss, model.trainable_variables)
        # horseshoe update
        if isinstance(model.selector, HorseshoeSelector):
            model.selector.update_tau_lambda()

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restore from {} !!!".format(manager.latest_checkpoint))
    else:
        logger.info("Initialize from scratch !!!")

    for i in range(n_iter):
        # optimizer step
        optimize_step()
        # horseshoe update
        if isinstance(model.selector, HorseshoeSelector):
            model.selector.update_tau_lambda()

        # save checkpoint
        if (i + 1) % ckpt_freq == 0:
            save_path = manager.save()
            logger.info("Saved checkpoint for step {}: {}".format(i + 1, save_path))
            if not classify:
                mu, var = model.predict_y(x_test)
                ll = tf.reduce_mean(model.likelihood.predict_log_density(mu, var, y_test))
                RMSE = tf.sqrt(tf.reduce_mean(tf.square(mu - y_test)))
                logger.info("Iter {} \t Loss: {:.2f} \t Test RMSE:{} \t Test LL{}".format(i,
                                                                                          train_loss().numpy(),
                                                                                          RMSE.numpy(),
                                                                                          ll.numpy()))
            else:
                mu, var = model.predict_y(x_test)
                ll = tf.reduce_mean(model.likelihood.predict_log_density(mu, var, y_test))
                pred = np.argmax(tf.concat([1. - mu, mu], axis=1), 1).reshape(y_test.numpy().shape)
                correct = pred == y_test.numpy().astype(int)
                acc = np.average(correct.astype(float)) * 100

                logger.info("Iter {} \t Loss: {:.2f} \t Test ACC:{} \t Test LL{}".format(i,
                                                                                         train_loss().numpy(),
                                                                                         acc,
                                                                                         ll.numpy()))

    get_weight(model.selector, model.gps)
    # do prediction
    if not classify:
        mu, var = model.predict_y(x_test)
        ll = tf.reduce_mean(model.likelihood.predict_log_density(mu, var, y_test))
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(mu - y_test)))
        logger.info("Iter {} \t Loss: {:.2f} \t Test RMSE:{} \t Test LL{}".format(i,
                                                                                  train_loss().numpy(),
                                                                                  RMSE.numpy(),
                                                                                  ll.numpy()))
    else:
        mu, var = model.predict_y(x_test)
        ll = tf.reduce_mean(model.likelihood.predict_log_density(mu, var, y_test))
        pred = np.argmax(tf.concat([1. - mu, mu], axis=1), 1).reshape(y_test.numpy().shape)
        correct = pred == y_test.numpy().astype(int)
        acc = np.average(correct.astype(float)) * 100

        logger.info("Test ACC:{} \t Test LL{}".format(
            acc,
            ll.numpy()))


def get_weight(selector, gps, n_components=3):

    w = selector.sample()
    w = w.numpy().squeeze()
    print(w)
    sorted_index = np.argsort(w)
    for i in sorted_index[-n_components:]:
        print(w[i])
        # print_summary(gps[i].kernel)
        gp = gps[i]
        kernel = gp.kernel
        print("w {}\t active_dims={}".format(w[i], kernel.active_dims))

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.size': 12, 'figure.subplot.bottom': 0.125})
    matplotlib.rcParams.update({
        "text.usetex": True,
        # "font.family": "serif",
        # "font.serif": ["Palatino"],
    })
    from matplotlib import rc
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    plt.figure(figsize=(5, 1.5))
    plt.bar(np.arange(len(w)), w**2)
    plt.xticks([])
    # plt.xlim(-1, 12)
    plt.xlabel(r"kernels $k_i$")
    plt.ylabel(r"$w_i$")
    plt.show()


if __name__ == "__main__":
    # date = sys.argv[1]
    # dataset_name = sys.argv[2]
    # LOAD OR NOT


    n_iter = 5000
    lr = 0.01

    date = "0909_1"
    dataset_name = "liver"
    # if len(sys.argv) > 3:
    n_iter = 0


    if dataset_name == "housing":
        additive_order = [3]
        raise RuntimeError("Too resource demanding")
    elif dataset_name == "concrete":
        additive_order = [1, 2, 3]
        classify = False
    elif dataset_name == "servo":
        additive_order = [1, 2, 4]
    elif dataset_name == "pima":
        additive_order = [6]
        classify = True
    elif dataset_name == "liver":
        additive_order = [3]
        classify = True
    elif dataset_name == "heart":
        # 10 dimension is too big
        additive_order = [1]
        classify = True
    else:
        raise ValueError("cannot find data set")


    unique_name = create_unique_name(date, dataset_name, kernel_order=None, repetition=None)

    logger = create_logger("../log", unique_name, __file__)

    run_additive(date, dataset_name, additive_order, ckpt_freq=100, classify=classify, logger=logger, n_iter=n_iter,
                 lr=lr)
