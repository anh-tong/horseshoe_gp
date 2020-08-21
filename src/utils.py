
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import scipy.io as sio
from sklearn.model_selection import train_test_split
from gpflow.utilities import to_default_float
from typing import Tuple, Dict

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

    def retrieve(self):
        data = sio.loadmat(self.data_dir)
        x = data["X"]
        y = data["y"]
        self.n = x.shape[0]
        self.d = x.shape[1]
        return x, y

class MNISTDataset(Dataset):

    def __init__(self, name="MNIST", batch_size=128):

        self.batch_size = batch_size
        super().__init__(name)


    def data(self, test_size=0.1, random_state=123):
        train, train_info = tfds.load(name="mnist", split=tfds.Split.TRAIN, with_info=True)
        test, test_info = tfds.load(name="mnist", split=tfds.Split.TEST, with_info=True)
        self.n_train = train_info.splits["train"].num_examples
        image_shape = train_info.features["image"].shape
        self.d = tf.reduce_prod(image_shape)

        self.n_test = test_info.splits["test"].num_examples
        self.n = self.n_train + self.n_test


        autotune = tf.data.experimental.AUTOTUNE

        def map_fn(input_slice: Dict[str, tf.Tensor]):
            updated = input_slice
            image = to_default_float(updated["image"]) / 255.0
            label = to_default_float(updated["label"])
            return tf.reshape(image, [-1, self.d]), label


        self.train_dataset_norepeat = train.map(map_fn, num_parallel_calls=autotune)

        self.train_dataset = (train.shuffle(1024)
                              .batch(self.batch_size, drop_remainder=True)
                              .map(map_fn, num_parallel_calls=autotune)
                              .prefetch(autotune)
                              .repeat())
        self.test_dataset = (test.batch(self.batch_size).map(map_fn, num_parallel_calls=autotune))



def get_dataset(name) -> Dataset:

    if name == "airline":
        dataset = ABCDDataset("../data/01-airline.mat")
    elif name == "solar":
        dataset = ABCDDataset("../data/02-solar.mat")
    elif name == "mauna":
        dataset = ABCDDataset("../data/03-mauna2003.mat")
    elif name == "wheat-price":
        dataset = ABCDDataset("../data/beveridge-wheat-price-index-1500.mat")
    elif name == "mnist":
        dataset = MNISTDataset()
    else:
        raise ValueError("Cannot find the dataset [{}]".format(name))

    return dataset

