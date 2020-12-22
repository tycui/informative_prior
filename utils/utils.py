from __future__ import absolute_import
import torch
import numpy as np

DATA_PATH = 'D:/Work/informative_prior/data/'
RESULT_PATH = 'D:/Work/informative_prior/results/'

def save_result(s, name):
    s = np.array(s)
    np.savetxt(name + '.txt', s, delimiter=',')

def PVE_test(mlp, x_test, y_test): 
    mlp.eval()
    y_pred, _ = mlp(x_test)
    ptve = 1. - torch.var(y_pred - y_test) / torch.var(y_test)
    return ptve.item()

def PVE_posterior(mlp, x_test, y_test): ## posterior distribution of pve
    iteration = 1000
    PTVE_pos = []
    mlp.train()
    for i in range(iteration):
        y_pred, _ = mlp(x_test)
        ptve = 1. - torch.var(y_pred- y_test) / torch.var(y_test)
        PTVE_pos.append(ptve.item())
    return np.mean(PTVE_pos)

def NLL_noise(mlp, x_test, y_test):
    """
    Estimate the predictive log-likelihood per datapoint
    L: the data log-likelihood of p(y|x, w), where w~q(w)
    LL: p(y|x, D) calculated via log-sum-exp trick
    
    """
    iteration = 100
    L = []
    mlp.train()
    for i in range(iteration):
        y_pred, _, sigma_noise = mlp(x_test)
        sigma_noise = sigma_noise.detach()
        test_ll = -0.5 * (y_pred - y_test) ** 2 / (sigma_noise**2) -  np.log(sigma_noise) - 0.5 * np.log(2 * np.pi)
        L.append(torch.sum(test_ll).item())
    L = np.array(L)
    a_max = L.max()
    LL = np.log(np.sum(np.exp(L-a_max)))+a_max - np.log(L.shape[0])
    return -LL/x_test.shape[0]

def PVE_test_noise(mlp, x_test, y_test): 
    mlp.eval()
    y_pred, _, _  = mlp(x_test)
    ptve = 1. - torch.var(y_pred - y_test) / torch.var(y_test)
    return ptve.item()
