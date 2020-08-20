import itertools

from gpflow.kernels import RBF, Periodic, Linear, Product

class Periodic2(Periodic):

    def __init__(self, period=1.):
        super().__init__(base_kernel=RBF(), period=period)


class Generator(object):

    def __init__(self, base_cls=None):
        self.base_cls = [RBF, Periodic2, Linear] if base_cls is None else base_cls

    def init_params(self, k):
        pass

    def create_first_order(self):
        kernels = []
        for cls in self.base_cls:
            k = cls()
            self.init_params(k)
            kernels.append(k)
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
                    self.init_params(k)
                    kernel.append(k)
                kernels.append(Product(kernel))

        return kernels

    def create_upto(self, upto_order):
        ret = []
        for i in range(upto_order):
            kernels = self.create(order=i+1)
            ret.extend(kernels)

        return ret


