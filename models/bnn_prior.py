import torch.nn as nn
from models.common_distributions import *
import torch.distributions as Dis
import random

class Gaussian_GammaARD_Layer_Prior(nn.Module):
    """
    BNN layers prior: Mean-field Gaussian with inv-Gamma(2, beta) ARD hyper prior over local scales
    """
    def __init__(self, input_dim, output_dim, alpha=2.):
        super(Gaussian_GammaARD_Layer_Prior, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = Dis.gamma.Gamma(alpha, 1.)

    def forward(self, x, beta):
        w = torch.randn(self.input_dim, self.output_dim)
        tau = beta * 1. / (self.gamma.sample(torch.tensor([self.input_dim])))  # Inverse Gamma ARD prior over local scales
        y = torch.mm(x * tau, w)
        return y

class Info_Gaussian_GammaARD_Layer_Prior(nn.Module):
    """
    BNN layers prior: Mean-field Gaussian with inv-Gamma(2, beta) ARD hyper prior over local scales
    With an informative spike-and-slab prior for feature selection
    """
    def __init__(self, input_dim, output_dim, k_min, k_max, tau, alpha=2):
        super(Info_Gaussian_GammaARD_Layer_Prior, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_m = Discrete_Flatten_Laplace(k_min, k_max, tau, input_dim)
        self.gamma = Dis.gamma.Gamma(alpha, 1.)

    def forward(self, x, beta):
        w = torch.randn(self.input_dim, self.output_dim)
        tau = beta * 1. / (self.gamma.sample(torch.tensor([self.input_dim])))  # Inverse Gamma ARD prior over local scales
        indentity = torch.zeros(self.input_dim)
        indentity_index = random.sample(population=[i for i in range(self.input_dim)], k=self.prior_m.sample().item())
        indentity[indentity_index] = 1
        y = torch.mm(x * tau * indentity, w)
        return y

class Gaussian_GammaARD_NN_Prior(nn.Module):
    """
    BNN prior: Mean-field Gaussian with inv-Gamma(2, beta) ARD hyper prior over local scales
    """
    def __init__(self, num_feature, num_hidden_nodes):
        super(Gaussian_GammaARD_NN_Prior, self).__init__()
        self.num_feature = num_feature
        self.Layer1 = Gaussian_GammaARD_Layer_Prior(num_feature, num_hidden_nodes[0])
        self.Layer2 = Gaussian_GammaARD_Layer_Prior(num_hidden_nodes[0], num_hidden_nodes[1])
        self.Layer3 = Gaussian_GammaARD_Layer_Prior(num_hidden_nodes[1], 1)
        self.relu = nn.ReLU()

    def forward(self, x, beta):
        x1 = self.relu(self.Layer1(x, beta))
        x2 = self.relu(self.Layer2(x1, beta))
        x3 = self.Layer3(x2, beta)
        return x3

class Info_Gaussian_GammaARD_NN_Prior(nn.Module):
    """
    BNN prior: Informative Mean-field Gaussian with inv-Gamma(2, beta) ARD hyper prior over local scales
    """
    def __init__(self, num_feature, num_hidden_nodes, k_min, k_max, tau):
        super(Info_Gaussian_GammaARD_NN_Prior, self).__init__()
        self.num_feature = num_feature
        self.Layer1 = Info_Gaussian_GammaARD_Layer_Prior(num_feature, num_hidden_nodes[0], k_min, k_max, tau)
        self.Layer2 = Gaussian_GammaARD_Layer_Prior(num_hidden_nodes[0], num_hidden_nodes[1])
        self.Layer3 = Gaussian_GammaARD_Layer_Prior(num_hidden_nodes[1], 1)
        self.relu = nn.ReLU()

    def forward(self, x, beta):
        x1 = self.relu(self.Layer1(x, beta))
        x2 = self.relu(self.Layer2(x1, beta))
        x3 = self.Layer3(x2, beta)
        return x3

class Gaussian_Layer_Prior(nn.Module):
    """
    BNN layers prior: Mean-field Gaussian prior
    """
    def __init__(self, input_dim, output_dim):
        super(Gaussian_Layer_Prior, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x, scale):
        w = torch.randn(self.input_dim, self.output_dim) * scale
        y = torch.mm(x, w)
        return y

class Info_Gaussian_Layer_Prior(nn.Module):
    """
    BNN layers prior: Mean-field Gaussian prior
    With an informative spike-and-slab prior for feature selection
    """
    def __init__(self, input_dim, output_dim, k_min, k_max, tau):
        super(Info_Gaussian_Layer_Prior, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_m = Discrete_Flatten_Laplace(k_min, k_max, tau, input_dim)

    def forward(self, x, scale):
        w = torch.randn(self.input_dim, self.output_dim) * scale
        indentity = torch.zeros(self.input_dim)
        indentity_index = random.sample(population=[i for i in range(self.input_dim)], k=self.prior_m.sample().item())
        indentity[indentity_index] = 1
        y = torch.mm(x * indentity, w)
        return y

class Gaussian_NN_prior(nn.Module):
    """
    BNN prior: Mean-field Gaussian prior over local scales
    """
    def __init__(self, num_feature, num_hidden_nodes):
        super(Gaussian_NN_prior, self).__init__()
        self.num_feature = num_feature
        self.Layer1 = Gaussian_Layer_Prior(num_feature, num_hidden_nodes[0])
        self.Layer2 = Gaussian_Layer_Prior(num_hidden_nodes[0], 1)
        self.relu = nn.ReLU()

    def forward(self, x, scale):
        x1 = self.relu(self.Layer1(x, scale))
        x2 = self.Layer2(x1, scale)
        return x2

class Info_Gaussian_NN_prior(nn.Module):
    """
    BNN prior: Informative Mean-field Gaussian prior over local scales
    """
    def __init__(self, num_feature, num_hidden_nodes, k_min, k_max, tau):
        super(Info_Gaussian_NN_prior, self).__init__()
        self.num_feature = num_feature
        self.Layer1 = Info_Gaussian_Layer_Prior(num_feature, num_hidden_nodes[0], k_min, k_max, tau)
        self.Layer2 = Gaussian_Layer_Prior(num_hidden_nodes[0], 1)
        self.relu = nn.ReLU()

    def forward(self, x, scale):
        x1 = self.relu(self.Layer1(x, scale))
        x2 = self.Layer2(x1, scale)
        return x2
