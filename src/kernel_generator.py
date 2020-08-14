import torch
import itertools

from gpytorch.kernels import (RBFKernel, PeriodicKernel,
                              LinearKernel, ProductKernel)

class DataStats:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.compute_stats()

    def compute_stats(self):
        pass

class Generator():

    def __init__(self, data_stats, base_cls=None):
        self.data_stats = data_stats
        self.base_cls = [RBFKernel,
                         PeriodicKernel,
                         LinearKernel] if base_cls is None else base_cls

    def create_first_order(self):
        kernels = []
        for cls in self.base_cls:
            k = cls()
            self.init_param(k)
            kernels += [k]
        return kernels

    def create(self, order):
        if order == 1:
            return self.create_first_order()
        else:
            comb = itertools.product(self.base_cls, repeat=order)
            kernels = []
            for c in comb:
                kernel = []
                for cls in c:
                    k = cls()
                    self.init_param(k)
                    kernel += [k]
                kernels.append(ProductKernel(*kernel))

        return kernels

    def init_param(self, k):
        """TODO: intialize parameter accounding to data"""
        pass


generator = Generator(None)
# print(generator.create_first_order())
# print(generator.create(2))
print(len(generator.create(2)))
print(len(generator.create(3)))
