import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

###Benchmark Black-Box Funtions
#https://arxiv.org/pdf/1308.4008.pdf

###Acqusition Functions
class UCB:
    def __init__(self, model):
        self.model = model
        self.kappa = 2.576
        
    def __call__(self, x, ymax = None):
        mean, std = self.model.predict_y(x)
        return mean + self.kappa * std

class EI:
    def __init__(self, model):
        self.model = model
        self.eps = 0.0
        self.norm = tfp.distributions.Normal(tf.zeros(1,  dtype=tf.dtypes.float64), tf.ones(1,  dtype=tf.dtypes.float64))
        
    def __call__(self, x, ymax):
        mean, std = self.model.predict_y(x)
        a = (mean - ymax - self.eps)
        z = a / std
        return a * self.norm.cdf(z) + std * self.norm.prob(z)

class POI:
    def __init__(self, model):
        self.model = model
        self.eps = 0.0
        self.norm = tfp.distributions.Normal(tf.zeros(1,  dtype=tf.dtypes.float64), tf.ones(1,  dtype=tf.dtypes.float64))
        
    def __self__(self, x, ymax):
        mean, std = self.model.predict_y(x)        
        z = (mean - ymax - self.eps)/std
        return self.norm.cdf(z)

###Test Functions
class test_fun:
    def __init__(self, lower_bound, upper_bound, sols, sol_val):
        self.dim = None
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.sols = sols
        self.sol_val = sol_val
        
    def check_fun(self):
        for ind, pnt in enumerate(self.sols):
            if self(pnt) == self.sol_val:
                pass
            else:
                print(ind + 1)
                return False
        return True

    def __call__(self, x):
        raise NotImplementedError()

class branin_rcos(test_fun):
    def __init__(self):
        super().__init__(
            np.array([-5, 0], dtype = np.float64),
            np.array([10, 15], dtype = np.float64),
            np.array([(-np.pi, 12.275), (np.pi, 2.275), (3*np.pi, 2.475)], dtype = np.float64),
            0.39788735772973816)
        self.dim = 2
        
    def __call__(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        return np.power((x2 - 5.1/(4*(np.power(np.pi, 2)))*np.power(x1, 2) + 5/np.pi*x1 - 6), 2) + 10*(1-1/(8*np.pi))*np.cos(x1)+10

class six_hump_camel_back(test_fun):
    def __init__(self):
        super().__init__(
            np.array([-5] * 2, dtype = np.float64),
            np.array([5] * 2, dtype = np.float64),
            np.array([(-0.0898, 0.7126), (0.0898, -0.7126)], dtype = np.float64),
            -1.0316284229280819)
        self.dim = 2
        
    def __call__(self, x):
        x1, x2 = x
        return (4-2.1*np.power(x1, 2)+np.power(x1, 4)/3)*np.power(x1, 2)+x1*x2+(4*np.power(x2, 2)-4)*np.power(x2, 2)

class hartman_6(test_fun):
    def __init__(self):
        super().__init__(
            np.array([0]*6, dtype = np.float64),
            np.array([1]*6, dtype = np.float64),
            np.array([(0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301)], dtype = np.float64),
            -3.306248140411212)
        self.dim = 6
        self.P = np.array([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5586],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ], dtype=np.float64)
        self.A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],    
        ], dtype=np.float64)
        self.c = np.array([1, 1.2, 3, 3.2], dtype=np.float64)
        
    def __call__(self, x):
        x = x
        return -np.sum(self.c*np.exp(np.sum(-self.A*np.power(np.reshape(np.repeat(x, 4, axis = 0), (4,6)) - self.P, 2), axis = 1)))

class goldstein_price(test_fun):
    def __init__(self):
        super().__init__(
            np.array([-2]*2, dtype=np.float64),
            np.array([2]*2, dtype=np.float64),
            np.array([(0, -1)], dtype = np.float64),
            3)
        self.dim = 2
        
    def __call__(self, x):
        x1, x2 = x
        return (1 + np.power(x1 + x2 + 1, 2) * (19 - 14 *x1 + 3*np.power(x1, 2) - 14*x2 + 6*x1*x2 + 3*np.power(x2, 2))*(30 + np.power(2*x1 - 3*x2, 2) * (18 - 32*x1 + 12* (np.power(x1, 2)) + 48 * x2 - 36*x1*x2 + 27*np.power(x2, 2))))

class rosenbrock(test_fun):
    def __init__(self):
        super().__init__(
            np.array([-30] * 10, dtype=np.float64),
            np.array([30] * 10, dtype=np.float64),
            np.array([(1) * 10], dtype = np.float64),
            0)
        self.dim = 10
                         
    def __call__(self, x):
        if isinstance(x, int):
            x = [x]
        x = x
        res = 0.0
        for dim in range(self.dim - 1):
            res += 100 * (x[dim+1] - x[dim]**2)**2 + (x[dim] - 1)**2
        return res
