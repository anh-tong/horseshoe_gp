import logging
import sys

from gpflow.kernels import RBF, Product, Periodic
from gpflow.mean_functions import Zero

from src.deepgp.deep_gp import DeepGPBase
from src.deepgp.layers import SVGPLayer
from src.deepgp.structural_layer import StructuralSVGPLayer
from src.experiment_tf import *
from src.utils import create_logger


class StructuralDeepGP(DeepGPBase):

    def __init__(self, layers, likelihood, num_samples=1):
        super(StructuralDeepGP, self).__init__(likelihood, layers, num_samples)


def create_dgp_regression(X, Y, Z, layer_sizes):
    likelihood = Gaussian()
    layers = init_layers(X, Y, Z, layer_sizes)
    model = StructuralDeepGP(layers, likelihood)
    return model


def create_layer(output_dim, inducing_points, whiten):
    # TODO: change kernel herew
    kernels = [RBF(), Product([RBF(), Periodic(RBF())])] * 2
    fix_kernel_variance(kernels)
    gps = []
    for kernel in kernels:
        svgp = SVGPLayer(kern=kernel,
                         Z=inducing_points,
                         num_outputs=output_dim,
                         mean_function=Zero(),
                         white=whiten)
        gps.append(svgp)

    selector = HorseshoeSelector(dim=len(kernels))
    layer = StructuralSVGPLayer(gps, selector, output_dim)
    return layer


def init_layers(X, Y, Z, layer_sizes, output_dim=None, whiten=False):
    depth = len(layer_sizes)
    output_dim = output_dim or Y.shape[1]
    layers = []
    X_running, Z_running = X.copy(), Z.copy()
    for i in range(depth - 1):
        dim_in = layer_sizes[i]
        dim_out = layer_sizes[i + 1]
        new_layer = create_layer(dim_out, inducing_points=Z_running, whiten=whiten)
        layers += [new_layer]

        if dim_in != dim_out:
            if dim_in > dim_out:
                _, _, V = np.linalg.svd(X_running, full_matrices=False)
                W = V[:dim_out, :].T
            else:
                W = np.concatenate([np.eye(dim_in),
                                    np.zeros((dim_in, dim_out - dim_in))], 1)
            Z_running = Z_running.dot(W)
            X_running = X_running.dot(W)

    last_layer = create_layer(output_dim, inducing_points=Z_running, whiten=whiten)
    layers += [last_layer]
    return layers


def run_deepgp(date,
               dataset_name="housing",
               kernel_order=2,
               repetition=2,
               n_iter=1000,
               lr=0.01,
               ckpt_feq=1000,
               logger=logging.getLogger()):
    unique_name = create_unique_name(date, dataset_name, kernel_order=kernel_order, repetition=repetition,
                                     selector="horseshoe")
    dataset = load_data(dataset_name)
    x_train, y_train = dataset.get_train()
    x_test, y_test = dataset.get_test()
    std_y_train = dataset.std_y_train
    z = init_inducing_points(x_train, M=100)
    input_dim = x_train.shape[1]
    model = create_dgp_regression(x_train.numpy(), y_train.numpy(), z.numpy(), layer_sizes=[input_dim, 5, 5])

    ckpt = tf.train.Checkpoint(model=model)
    ckpt_dir = "../model/dgp_{}".format(unique_name)
    manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=3)

    train_iter = make_data_iteration(x_train, y_train, batch_size=128, shuffle=True)
    train_loss = model.training_loss_closure(train_iter, compile=False)
    optimizer = tf.optimizers.Adam(lr=lr)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        logger.info("Restore from {}".format(manager.latest_checkpoint))
    else:
        logger.info("Initialize from scratch")

    @tf.function
    def train_step():
        optimizer.minimize(train_loss, model.trainable_variables)

    for i in range(n_iter):
        train_step()

        # update phi_tau and phi_lambda in each layer
        for layer in model.layers:
            layer.selector.update_tau_lambda()

        if (i + 1) % ckpt_feq == 0:
            save_path = manager.save()
            mu, var = model.predict_y(x_test, num_samples=5)
            ll = tf.reduce_mean(model.likelihood.predict_log_density(mu, var, y_test)) - np.log(std_y_train)
            RMSE = tf.sqrt(tf.reduce_mean(tf.square(mu - y_test))) * std_y_train
            logger.info("Save checkpoint at {}".format(save_path))
            logger.info("Iter: {} \t loss:{:.2f} \t RMSE: {} \t Test LL: {} ".format(i + 1,
                                                                                     tf.squeeze(train_loss()).numpy(),
                                                                                     RMSE.numpy(),
                                                                                     tf.squeeze(ll).numpy()))


if __name__ == "__main__":
    argv = sys.argv

    date = argv[0]
    dataset_name = argv[1]
    kernel_order = 2
    repetition = 2
    unique_name = create_unique_name(date, dataset_name, kernel_order, repetition, selector="horseshoe")

    logger = create_logger(output_path="../log", name=unique_name, run_file=__file__)
    run_deepgp(date, dataset_name, n_iter=50000, logger=logger, ckpt_feq=1000, lr=0.01)
