import os
import pickle

from src.experiment_tf import *
from gpflow.likelihoods import MultiClass
from scipy.cluster.vq import kmeans2


def init_mnist_inducing_points(train_dataset, M=100, save_dir="../model/mnist_inducing_M_{}.pkl"):

    save_dir = save_dir.format(M)
    if not os.path.exists(save_dir):
        train_dataset = list(train_dataset)
        X = [a[0].numpy() for a in train_dataset]
        X = np.vstack(X)
        inducing_points = kmeans2(X, M, minit="points")[0]
        with open(save_dir, 'wb') as f:
            pickle.dump(inducing_points, f)
        print("Persist inducing points!!!")
    else:
        with open(save_dir, 'rb') as f:
            inducing_points = pickle.load(f)
        print("Loading inducing point from [{}]".format(save_dir))

    return inducing_points



def create_classifier_model(inducing_point, input_dim, num_data, num_class, n_kernels=5) -> StructuralSVGP:

    # TODO: initialize lengthscales
    kernels = [RBF(lengthscales=np.ones(input_dim)),
               Product([RBF(lengthscales=np.ones(input_dim)),
                        Periodic2(period=np.ones(input_dim))])] * n_kernels

    fix_kernel_variance(kernels)

    gps = []
    for kernel in kernels:
        gp = SVGP(kernel=kernel,
                  likelihood=None,
                  inducing_variable=inducing_point,
                  num_data=num_data,
                  num_latent_gps=num_class)
        gps += [gp]

    selector = HorseshoeSelector(dim=len(gps))
    likelihood = MultiClass(num_class)
    model = StructuralSVGP(gps, selector, likelihood, num_data)
    return model

def test_mnist(test_iter, model):

    corrects = []
    for x_batch, y_batch in test_iter:
        mu, var = model.predict_y(x_batch)
        pred = np.argmax(mu, 1).reshape(y_batch.numpy().shape)
        correct = pred == y_batch.numpy().astype(int)
        corrects.extend(correct)

    corrects = np.vstack(corrects)
    acc = np.average(corrects.astype(float)) * 100.
    print("Accuracy is {:.4f}".format(acc))


def run_mnist():

    dataset = load_data("mnist")
    train_iter = iter(dataset.train_dataset)
    test_iter = iter(dataset.test_dataset)

    inducing_points = init_mnist_inducing_points(dataset.train_dataset_norepeat)

    model = create_classifier_model(inducing_points, input_dim=dataset.d, num_data=dataset.n_train, num_class=10)

    model = train(model, train_iter, n_iter=100000, lr=0.01)

    test_mnist(test_iter, model)


if __name__ == "__main__":
    run_mnist()







