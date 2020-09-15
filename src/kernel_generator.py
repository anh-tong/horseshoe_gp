import itertools
from typing import List

from gpflow.kernels import Product, RBF, Kernel

from src.kernels import create_rbf, create_period, create_linear


class Generator(object):

    def __init__(self, data_shape, base_fn=None):
        self.data_shape = data_shape
        self.base_fn = [create_rbf, create_period, create_linear] if base_fn is None else base_fn

    def create_first_order(self, type="weak"):
        kernels = []
        for generate_fn in self.base_fn:
            k = generate_fn(self.data_shape, type=type)
            kernels.append(k)
        return kernels

    def create(self, order, type):
        if order == 1:
            return self.create_first_order()
        else:
            comb = itertools.product(self.base_fn, repeat=order)
            kernels = []
            for c in comb:
                kernel = []
                for generate_fn in c:
                    k = generate_fn(self.data_shape, type=type)
                    kernel.append(k)
                kernel = reduce_rbf(kernel)
                if len(kernel) == 1:
                    # after reduction just single kernel
                    kernels.append(kernel[0])
                else:
                    # multiple kernel
                    kernels.append(Product(kernel))

        return kernels

    def create_upto(self, upto_order, type="weak"):
        ret = []
        for i in range(upto_order):
            kernels = self.create(order=i + 1, type=type)
            ret.extend(kernels)

        return ret


def reduce_rbf(kernels: List[Kernel]) -> List[Kernel]:
    rbf_indices = []
    for i, kernel in enumerate(kernels):
        if isinstance(kernel, RBF):
            rbf_indices += [i]

    if len(rbf_indices) <= 1:
        return kernels
    else:
        non_rbf_indices = [index for index in list(range(len(kernels))) if index not in rbf_indices]
        non_rbf_indices += [rbf_indices[0]]
        return [kernels[index] for index in non_rbf_indices]

# # test only
# if __name__ == "__main__":
#     from gpflow.kernels import Periodic, Linear
#     kernels = [RBF(), Linear(), Periodic(RBF())]
#     assert len(reduce_rbf(kernels)) == 3
#     kernels = [RBF(), RBF()]
#     assert len(reduce_rbf(kernels)) == 1
#     kernels = [RBF(), RBF(), Linear()]
#     assert len(reduce_rbf(kernels)) == 2
