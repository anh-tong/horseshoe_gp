from src.experiment_tf import *
from src.deepgp.deep_gp import DeepGPBase
from src.deepgp.structural_layer import StructuralSVGPLayer
from src.deepgp.layers import SVGPLayer
from gpflow.mean_functions import Zero



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
    for i in range(depth-1):
        dim_in = layer_sizes[i]
        dim_out = layer_sizes[i+1]
        new_layer = create_layer(dim_out, inducing_points=Z_running, whiten=whiten)
        layers += [new_layer]

        if dim_in != dim_out:
            if dim_in > dim_out:
                _, _, V = np.linalg.svd(X_running, full_matrices=False)
                W = V[:output_dim, :].T
            else:
                W = np.concatenate([np.eye(dim_in),
                                    np.zeros((dim_in, dim_out - dim_in))], 1)
            Z_running = Z_running.dot(W)
            X_running = X_running.dot(W)

    last_layer = create_layer(output_dim, inducing_points=Z_running, whiten=whiten)
    layers += [last_layer]
    return layers


def run_deepgp(n_iter=1000, lr=0.01):

    dataset = load_data()
    x_train, y_train = dataset.get_train()
    x_test, y_test = dataset.get_test()
    z = init_inducing_points(x_train, M=100)
    input_dim = x_train.shape[1]
    model = create_dgp_regression(x_train.numpy(), y_train.numpy(), z.numpy(), layer_sizes=[input_dim, 5,5])

    train_iter = make_data_iteration(x_train, y_train, batch_size=128, shuffle=True)
    optimizer = tf.optimizers.Adam(lr=lr)


    def train_step():
        data_batch = next(train_iter)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.trainable_variables)
            objective = -model.elbo(data_batch)
            gradients = tape.gradient(objective, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    for i in range(n_iter):
        train_step()


run_deepgp()

