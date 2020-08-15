import math
import torch

###Benchmark Black-Box Funtions
#https://arxiv.org/pdf/1308.4008.pdf

class acq_fun:
    def __init__(self, bound, sols, sol_val):
        self.dim = None
        self.bound = torch.Tensor(bound)
        self.sols = sols
        self.sol_val = sol_val
        
    def check_fun(self):
        for ind, pnt in enumerate(self.sols):
            if self.value(pnt) == self.sol_val:
                pass
            else:
                print(ind)
                return False
        return True

    def value(self, x):
        raise NotImplementedError()

class branin_rcos(acq_fun):
    def __init__(self):
        super().__init__(
                         [[-5, 0], [10, 15]],
                         [(-math.pi, 12.275), (math.pi, 2.275), (3*math.pi, 2.475)],
                         0.39788735772973816)
        self.dim = 2
        
    def value(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return (x2 - 5.1/(4*(math.pi**2))*(x1.pow(2)) + 5/math.pi*x1 - 6).pow(2) + 10*(1-1/(8*math.pi))*torch.cos(x1)+10

class six_hump_camel_back(acq_fun):
    def __init__(self):
        super().__init__(
                        [[-5] * 2, [5] * 2],
                        [(-0.0898, 0.7126), (0.0898, -0.7126)],
                        -1.0316284229280819)
        self.dim = 2
        
    def value(self, x):
        x1, x2 = x
        return (4-2.1*(x1.pow(2))+(x1.pow(4))/3)*(x1.pow(2))+x1*x2+(4*(x2.pow(2))-4)*(x2.pow(2))

class hartman_6(acq_fun):
    def __init__(self):
        super().__init__(
                         [[0]*6, [1]*6],
                         [(0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301)],
                         -3.306248188018799)
        self.dim = 6
        self.P = torch.Tensor([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5586],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ])
        self.A = torch.Tensor([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],    
        ])
        self.c = torch.Tensor([1, 1.2, 3, 3.2])
        
    def value(self, x): 
        return float(-(self.c*(-self.A*((torch.Tensor(x).repeat(4).view(4, -1) - self.P).pow(2))).sum(1).exp()).sum())

class goldstein_price(acq_fun):
    def __init__(self):
        super().__init__(
                         [[-2]*2, [2]*2],
                         [(0, -1)],
                         3)
        self.dim = 2
        
    def value(self, x):
        x1, x2 = x
        return (1 + (x1 + x2 + 1).pow(2) * (19 - 14 *x1 + 3*(x1.pow(2)) - 14*x2 + 6*x1*x2 + 3*(x2.pow(2))))*(30 + (2*x1 - 3*x2).pow(2) * (18 - 32*x1 + 12* (x1.pow(2)) + 48 * x2 - 36*x1*x2 + 27*(x2.pow(2))))

class rosenbrock(acq_fun):
    def __init__(self):
        super().__init__(
                         [[-30] * 10, [30] * 10],
                         [(1) * 10],
                         0)
        self.dim = 10
                         
    def value(self, x):
        if isinstance(x, int):
            x = [x]
        res = 0.0
        for dim in range(self.dim - 1):
            res += 100 * (x[dim+1] - x[dim]**2)**2 + (x[dim] - 1)**2
        return res
