import os
import logging
import time
import numpy as np
import tensorflow as tf
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from gpflow.utilities import to_default_float
from typing import Tuple, Dict
import matplotlib

class Dataset(object):

    def __init__(self, name=None):
        self.name = name
        self.n, self.d, self.n_train, self.n_test = None, None, None, None
        self.data()

    def retrieve(self):
        raise NotImplementedError

    def preprocess(self, x, y):
        return x, y


    def data(self, test_size=0.1, random_state=123):
        x, y = self.retrieve()
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        self.n_train = x_train.shape[0]
        self.n_test = x_test.shape[0]

        self.x_train = to_default_float(x_train)
        self.x_test = to_default_float(x_test)
        self.y_train = to_default_float(y_train)
        self.y_test = to_default_float(y_test)


    def get_train(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.x_train, self.y_train


    def get_test(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.x_test, self.y_test


class ABCDDataset(Dataset):

    def __init__(self, data_dir, name=None):
        if not name is None:
            name = "ABCD_" + name
        else:
            name = "ABCD"
        self.data_dir = data_dir
        super().__init__(name)

    def data(self, test_size=0.1, random_state=123):
        x, y = self.retrieve()
        # since this focus on the extrapolation for future prediction
        self.n_test = int(test_size * self.n)
        self.n_train = self.n - self.n_test
        x_train = x[:self.n_train]
        x_test = x[self.n_train:]
        y_train = y[:self.n_train]
        y_test = y[self.n_train:]
        self.x_train = to_default_float(x_train)
        self.x_test = to_default_float(x_test)
        self.y_train = to_default_float(y_train)
        self.y_test = to_default_float(y_test)

    def retrieve(self):
        data = sio.loadmat(self.data_dir)
        x = data["X"]
        y = data["y"]
        self.n = x.shape[0]
        self.d = x.shape[1]
        return x, y



class UCIDataset(Dataset):

    def __init__(self, data_dir, name=None):
        self.data_dir = data_dir
        super().__init__(name)

    def data(self, test_size=0.1, random_state=123):
        data, target = self.retrieve()
        if data.shape[0] < 10000:
            d = data
        else:
            indices = np.random.permutation(range(data.shape[0]))
            d = data[indices][:10000]

        pca = PCA(n_components=1)
        pca.fit(d)
        projected = pca.transform(data)

        ind = list(np.argsort(projected.squeeze()))
        len_ = len(ind)
        test_ind = np.array(ind[: len_ // 15] + ind[-len_ // 15:])
        train_ind = np.array(ind[len_ // 15: -len_ // 15])
        test_ind = test_ind[np.random.permutation(len(test_ind))]
        train_ind = train_ind[np.random.permutation(len(train_ind))]

        assert len(test_ind) + len(train_ind) == len_, 'train set and test set should add to the whole set'
        assert set(test_ind) - set(train_ind) == set(test_ind), 'train set and test set should be exclusive'

        x_train, y_train = data[train_ind], target[train_ind]
        x_test, y_test = data[test_ind], target[test_ind]

        x_train, x_test, _, _ = standardize(x_train, x_test)
        y_train, y_test, _, train_std = standardize(y_train, y_test)

        self.n_train = x_train.shape[0]
        self.n_test = x_test.shape[0]
        self.std_y_train = train_std

        self.x_train = to_default_float(x_train)
        self.x_test = to_default_float(x_test)
        self.y_train = to_default_float(y_train)
        self.y_test = to_default_float(y_test)



    def retrieve(self):
        data = np.loadtxt(self.data_dir)
        x, y = data[:, :-1], data[:, -1]
        y = y[:, None]
        self.n = x.shape[0]
        self.d = x.shape[1]
        return x, y

class NavalDataset(UCIDataset):

    def retrieve(self):
        x, y = super(NavalDataset, self).retrieve()
        x = np.delete(x, axis=1, obj=[8, 11])
        return x, y



class GEFCOM(Dataset):

    def __init__(self, data_dir="../data/gefcom.mat"):
        self.data_dir = data_dir
        super().__init__(name="gefcom")

    def retrieve(self):
        data = sio.loadmat(self.data_dir)
        X = data["times"]
        Y = data["loads"][:,0]
        Y = Y[:,None]
        Y, _, _ = standardize(Y)
        self.n = X.shape[0]
        self.d = X.shape[1]
        assert len(Y.shape) == 2
        assert len(X.shape) == 2

        return X, Y


def get_dataset(name) -> Dataset:

    if name == "airline":
        dataset = ABCDDataset("../data/01-airline.mat")
    elif name == "solar":
        dataset = ABCDDataset("../data/02-solar.mat")
    elif name == "mauna":
        dataset = ABCDDataset("../data/03-mauna2003.mat")
    elif name == "wheat-price":
        dataset = ABCDDataset("../data/beveridge-wheat-price-index-1500.mat")
    elif name == "housing":
        dataset = UCIDataset("../data/housing.data")
    elif name == "concrete":
        dataset = UCIDataset("../data/concrete.data.txt")
    elif name == "energy":
        dataset = UCIDataset("../data/energy.data")
    elif name == "kin8nm":
        dataset = UCIDataset("../data/kin8nm.data")
    elif name == "naval":
        dataset = NavalDataset("../data/naval.data")
    elif name == "power_plant":
        dataset = UCIDataset("../data/power_plant.data")
    elif name == "wine":
        dataset = UCIDataset("../data/wine.data")
    elif name == "yatch":
        dataset = UCIDataset("../data/yacht_hydrodynamics.data")
    elif name == "gefcom":
        dataset = GEFCOM("../data/gefcom.mat")
    elif name == "servo":
        dataset = ServoDataset("../data/servo.mat")
    elif name == "pima":
        dataset = PimaDataset("../data/r_pima.mat")
    elif name == "liver":
        dataset = LiverDataset("../data/r_liver.mat")
    elif name == "heart":
        dataset = HeartDataset("../data/r_heart.mat")
    else:
        raise ValueError("Cannot find the dataset [{}]".format(name))

    return dataset


class ServoDataset(Dataset):

    def __init__(self, data_dir="../data/servo.mat"):
        self.data_dir = data_dir
        self.std_y_train = 1. # no normalization in this data set
        super().__init__()


    def retrieve(self):
        data = sio.loadmat(self.data_dir)
        x = data["X"]
        y = data["y"]
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        return x, y

class PimaDataset(Dataset):

    def __init__(self, data_dir="../data/r_pima.mat"):
        self.data_dir = data_dir
        self.std_y_train = 1.
        self.n_class = 2
        super().__init__()

    def retrieve(self):
        data = sio.loadmat(self.data_dir)
        x = data["X"]
        y = data["y"]
        y[y == -1] = 0.
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        return x, y

class LiverDataset(PimaDataset):

    pass

class HeartDataset(PimaDataset):

    pass


def get_data_shape_from_XY(X, Y):
    data_shape = dict()

    data_shape["n_dims"] = X.shape[1]
    data_shape["N"] = X.shape[0]

    data_shape["x_mean"] = np.mean(X, axis=0)
    data_shape["y_mean"] = np.mean(Y)
    data_shape["x_sd"] = np.std(X, axis=0)
    data_shape["y_sd"] = np.std(Y)
    data_shape["x_min"] = np.min(X, axis=0)
    data_shape["y_min"] = np.min(Y)
    data_shape["x_max"] = np.max(X, axis=0)
    data_shape["y_max"] = np.max(Y)

    return data_shape

def get_data_shape(dataset:ABCDDataset):

    X, Y = dataset.get_train()
    X, Y = X.numpy(), Y.numpy()

    data_shape = get_data_shape_from_XY(X, Y)

    return data_shape

def standardize(data_train, *args):
    """
    Standardize a dataset to have zero mean and unit standard deviation.
    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.
    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    output = [data_train_standardized]
    for d in args:
        dd = (d - mean) / std
        output.append(dd)
    output.append(mean)
    output.append(std)
    return output

def makedirs(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

def create_logger(output_path, name, run_file):

    log_file = "{}_{}.log".format(name, time.strftime('%m-%d-%H-%M-%S'))
    head = '%(asctime)-15s %(message)s'
    filename = os.path.join(output_path, log_file)
    makedirs(filename)
    logging.basicConfig(filename=filename, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    with open(run_file, 'r') as f:
        logger.info(f.read())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger