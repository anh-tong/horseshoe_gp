import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.kernels import SquaredExponential, Product, Periodic 

###Benchmark Black-Box Funtions
#https://arxiv.org/pdf/1308.4008.pdf

pi = tf.constant(np.pi, dtype = tf.dtypes.float64)

#Data_shape object
def get_data_shape(x):
    ranked = tf.sort(x, axis = -1)
    return {
        "x_max":tf.reduce_max(x, axis = 0),
        "x_min":tf.reduce_min(x, axis = 0),
        "x_sd":tf.math.reduce_std(x, axis = 0),
        "x_min_abs_diff":ranked[0, :] - ranked[1, :]
    }


###Acqusition Functions
class UCB:
    def __init__(self):
        self.norm = tfp.distributions.Normal(
            tf.zeros(1, dtype=tf.dtypes.float64),
            tf.ones(1, dtype=tf.dtypes.float64))
        
    def __call__(self, x, model, num_fitted, ymax = None):
        mean, var = model.predict_f(x)
        #I put the value of 0.005 normal quantile
        return tf.squeeze(mean + 2.807034 * tf.math.sqrt(var))
        #return tf.squeeze(mean + self.norm.quantile(1 - 1 / num_fitted) * tf.math.sqrt(var))

class EI:
    def __init__(self):
        self.norm = tfp.distributions.Normal(
            tf.zeros(1, dtype=tf.dtypes.float64),
            tf.ones(1, dtype=tf.dtypes.float64))
        
    def __call__(self, x, model, ymax, num_fitted):
        mean, var = model.predict_f(x)
        std = tf.sqrt(var)
        z = (mean - ymax - 1/num_fitted) / std
        return tf.squeeze(std * (z * self.norm.cdf(z) + self.norm.prob(z)))

class POI:
    def __init__(self):
        self.eps = 0.01
        self.norm = tfp.distributions.Normal(
            tf.zeros(1,  dtype=tf.dtypes.float64),
            tf.ones(1,  dtype=tf.dtypes.float64))
        
    def __call__(self, x, model, ymax, num_fitted):
        mean, var = model.predict_f(x)        
        z = (mean - ymax -1/num_fitted)/tf.sqrt(var)
        return tf.squeeze(self.norm.cdf(z))

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
            if self(tf.expand_dims(pnt, 0)) == self.sol_val:
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
            tf.constant([-5, 0], dtype = tf.dtypes.float64),
            tf.constant([10, 15], dtype = tf.dtypes.float64),
            tf.constant([(-np.pi, 12.275), (np.pi, 2.275), (3*np.pi, 2.475)], dtype = tf.dtypes.float64),
            0.39788735772973816)
        self.dim = 2
        
    def __call__(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        return tf.pow((x2 - 5.1/(4*(tf.pow(pi, 2)))*tf.pow(x1, 2) + 5/pi*x1 - 6), 2) + 10*(1-1/(8*pi))*tf.math.cos(x1)+10

class six_hump_camel_back(test_fun):
    def __init__(self):
        super().__init__(
            tf.constant([-5] * 2, dtype = tf.dtypes.float64),
            tf.constant([5] * 2, dtype = tf.dtypes.float64),
            tf.constant([(-0.0898, 0.7126), (0.0898, -0.7126)], dtype = tf.dtypes.float64),
            -1.0316284229280819)
        self.dim = 2
        
    def __call__(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        return (4-2.1*tf.pow(x1, 2)+tf.pow(x1, 4)/3)*tf.pow(x1, 2)+x1*x2+(4*tf.pow(x2, 2)-4)*tf.pow(x2, 2)

class hartman_6(test_fun):
    def __init__(self):
        super().__init__(
            tf.constant([0]*6, dtype = tf.dtypes.float64),
            tf.constant([1]*6, dtype = tf.dtypes.float64),
            tf.constant([(0.201690, 0.150011, 0.476874, 0.275332, 0.311652, 0.657301)], dtype = tf.dtypes.float64),
            -3.306248140411212)
        self.dim = 6
        self.P = tf.constant([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5586],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
        ], dtype=tf.dtypes.float64)
        self.A = tf.constant([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],    
        ], dtype=tf.dtypes.float64)
        self.c = tf.constant([1, 1.2, 3, 3.2], dtype=tf.dtypes.float64)
        
    def __call__(self, x):
        x = tf.repeat(tf.reshape(x, (-1, 1, 6)), 4, axis = 1)
        x = -self.A*tf.pow(x - self.P, 2)
        x = tf.exp(tf.reduce_sum(x, axis = 2))
        x = self.c*x
        x = -tf.reduce_sum(x, axis = 1)
        return x

class goldstein_price(test_fun):
    def __init__(self):
        super().__init__(
            tf.constant([-2]*2, dtype = tf.dtypes.float64),
            tf.constant([2]*2, dtype = tf.dtypes.float64),
            tf.constant([(0, -1)], dtype = tf.dtypes.float64),
            3)
        self.dim = 2
        
    def __call__(self, x):
        x1 = x[:,0]
        x2 = x[:,1]
        return (1 + tf.pow(x1 + x2 + 1, 2) * (19 - 14 *x1 + 3*tf.pow(x1, 2) - 14*x2 + 6*x1*x2 + 3*tf.pow(x2, 2))*(30 + tf.pow(2*x1 - 3*x2, 2) * (18 - 32*x1 + 12* (tf.pow(x1, 2)) + 48 * x2 - 36*x1*x2 + 27*tf.pow(x2, 2))))

class rosenbrock(test_fun):
    def __init__(self, dim = 10):
        super().__init__(
            tf.constant([-30] * dim, dtype = tf.dtypes.float64),
            tf.constant([30] * dim, dtype = tf.dtypes.float64),
            tf.constant([(1) * dim], dtype = tf.dtypes.float64),
            0)
        self.dim = dim
                         
    def __call__(self, x):
        res = 0.0
        for dim in range(self.dim - 1):
            res += 100 * (x[:, dim+1] - x[:, dim]**2)**2 + (x[:, dim] - 1)**2
        return res
    
class Styblinski_Tang(test_fun):
    def __init__(self, dim = 10):
        super().__init__(
            tf.constant([-4] * dim, dtype = tf.dtypes.float64),
            tf.constant([4] * dim, dtype = tf.dtypes.float64),
            tf.constant([(-2.9) * dim], dtype = tf.dtypes.float64),
            -391.65950000000004
        )
        self.dim = dim
    
    def __call__(self, x):
        return tf.reduce_sum(tf.pow(x, 4) - 16*tf.pow(x, 2) + 5 * x, axis = 1) / 2
    
class Michalewicz(test_fun):
    def __init__(self, dim = 10):
        super().__init__(
            tf.constant([0] * dim, dtype = tf.dtypes.float64),
            tf.constant(tf.repeat(tf.expand_dims(pi, axis = 0), dim, axis = 0), dtype = tf.dtypes.float64),
            tf.constant([(0) * dim], dtype = tf.dtypes.float64),
            -9.66
        )
        self.dim = dim
        self.const = tf.expand_dims(tf.constant(range(self.dim), dtype = tf.dtypes.float64), 0)
        
    def __call__(self, x):
        return -tf.reduce_sum(tf.math.sin(x) * tf.pow(tf.math.sin(tf.math.sin(self.const * tf.pow(x,2) / pi)), 20), axis = 1)
