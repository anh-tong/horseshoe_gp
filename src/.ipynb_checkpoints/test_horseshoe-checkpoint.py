import torch
import numpy as np
import matplotlib.pyplot as plt
from src.horseshoe import *

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
This script tests the model described in

https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html

"""

m = 200
n = 100
alpha = 3.
sigma = 1.
sig_prob = 0.05

def generate_data():

    beta = np.zeros((m,1))
    for i in range(m):
        if np.random.random() < sig_prob:
            if np.random.random() < 0.5:
                beta[i] = np.random.randn() + 10
            else:
                beta[i] = np.random.randn() - 10
        else:
            beta[i] = np.random.rand()*0.25

    X = np.random.randn(m,n)
    y = np.zeros((n,1))
    for i in range(n):
        mean = X[:, i].transpose() @ beta + alpha
        y[i] = np.random.randn() * sigma + mean

    return X, y, beta

def plot_hist(beta):
    plt.subplots()
    plt.hist(beta, bins=100)

def plot_value(beta):
    plt.subplots()
    plt.plot(np.arange(0,m), beta)


X, y, beta = generate_data()

# plot histogram of beta -> sparsity
plot_hist(beta)
plot_value(beta)


class BayesianSparseRegression(Module):

    def __init__(self, X, y, alpha, sigma):
        super().__init__()
        # convert to tensor
        X = torch.Tensor(X)
        y = torch.Tensor(y)
        alpha = torch.Tensor([alpha])
        sigma = torch.Tensor([sigma])

        self.X = X
        self.y = y

        self.var_hs = VariationalHorseshoe(n_dim=m,
                                      b_0=5.,
                                      b_g=1.)
        self.alpha = alpha
        self.sigma = sigma

    def compute_likelihood(self):
        beta_sample = self.var_hs().t()
        y_hat = self.X.t() @ beta_sample + self.alpha
        diff = self.y - y_hat
        return - torch.sum(diff ** 2) / sigma**2 \
               - 0.5 * torch.log(2.*math.pi*self.sigma**2)

    def elbo(self, break_down=False):
        ll = self.compute_likelihood()
        penality = self.var_hs.kl_divergence()
        if break_down:
            return ll.sum(), penality.sum()
        else:
            ret = ll - penality
            return ret.sum()

    def update_kappa(self):
        self.var_hs.update()



bsr = BayesianSparseRegression(X, y, alpha, sigma)
opt = torch.optim.Adam(bsr.parameters(), lr=0.01)
for i in range(500):
    opt.zero_grad()
    ll, penalty = bsr.elbo(break_down=True)
    loss = -(ll - penalty)
    loss.backward()
    opt.step()
    bsr.update_kappa()
    print("Iter: {} \t ll:{} \t pen: {} \t Loss:{}".format(i,
                                                           ll.item(),
                                                           penalty.item(),
                                                           loss.item()))

with torch.no_grad():
    for i in range(2):
        beta = bsr.var_hs()
        beta = beta.detach().numpy()
        beta = beta.squeeze()
        plot_hist(beta)
        plot_value(beta)
    plt.show()






