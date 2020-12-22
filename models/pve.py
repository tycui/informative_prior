from models.bnn_prior import *

class BetaPrior(object):
    """
    Beta(alpha, beta) prior over PVE, with (log) probability density function.
    """
    def __init__(self, alpha, beta):
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)

    def average_log_density(self, target):
        eps = 1e-7
        return torch.mean((self.alpha - 1) * torch.log(target + eps) + (self.beta - 1) * torch.log(1 - target + eps) - torch.lgamma(
            self.alpha) - torch.lgamma(self.beta) + torch.lgamma(self.alpha + self.beta))

    def pdf(self, target):
        eps = 1e-7
        return torch.exp((self.alpha - 1) * torch.log(target + eps) + (self.beta - 1) * torch.log(1 - target + eps) - torch.lgamma(
            self.alpha) - torch.lgamma(self.beta) + torch.lgamma(self.alpha + self.beta))

class PVESampler_ARD(nn.Module):
    """
    Empirical pve draw from BNN priors (ARD hyper prior). M samples are drawn each time.
    """
    def __init__(self, x, M, num_hidden_nodes):
        super(PVESampler_ARD, self).__init__()
        self.scale_ = nn.Parameter(torch.Tensor(1).uniform_(-2, -1))
        self.x = x
        self.M = M
        self.BNNs = nn.ModuleList([Gaussian_GammaARD_NN_Prior(x.shape[1], num_hidden_nodes) for i in range(M)])

    def forward(self):
        PVE = torch.zeros(self.M, 1)
        beta = torch.log(1 + torch.exp(self.scale_))

        for i, j in enumerate(self.BNNs):
            # forward passing for each gene
            var_f_i = torch.var(j(self.x, beta))
            pve_i = 1. - 1. / (1. + var_f_i)

            index_i = torch.zeros(self.M, 1);
            index_i[i, :] = 1
            PVE = PVE + index_i * pve_i

        return PVE

class PVESampler_InfoARD(nn.Module):
    """
    Empirical pve draw from BNN priors (NON-ARD hyper prior). M samples are drawn each time.
    """
    def __init__(self, x, M, num_hidden_nodes, k_min, k_max, tau):
        super(PVESampler_InfoARD, self).__init__()
        self.scale_ = nn.Parameter(torch.Tensor(1).uniform_(-2, -1))
        self.x = x
        self.M = M
        self.BNNs = nn.ModuleList([Info_Gaussian_GammaARD_NN_Prior(x.shape[1], num_hidden_nodes, k_min, k_max, tau) for i in range(M)])

    def forward(self):
        PVE = torch.zeros(self.M, 1)
        beta = torch.log(1 + torch.exp(self.scale_))
        for i, j in enumerate(self.BNNs):
            # forward passing for each gene
            var_f_i = torch.var(j(self.x, beta))
            pve_i = 1. - 1. / (1. + var_f_i)
            index_i = torch.zeros(self.M, 1);
            index_i[i, :] = 1
            PVE = PVE + index_i * pve_i
        return PVE

class PVESampler_Gaussian(nn.Module):
    """
    Empirical pve draw from BNN priors (Gaussian prior). M samples are drawn each time.
    """
    def __init__(self, x, M, num_hidden_nodes):
        super(PVESampler_Gaussian, self).__init__()
        self.scale_ = nn.Parameter(torch.Tensor(1).uniform_(-2, 0))
        self.x = x
        self.M = M
        self.BNNs = nn.ModuleList([Gaussian_NN_prior(x.shape[1], num_hidden_nodes) for i in range(M)])

    def forward(self):
        PVE = torch.zeros(self.M, 1)
        scale = torch.log(1 + torch.exp(self.scale_))
        for i, j in enumerate(self.BNNs):
            # forward passing for each gene
            var_f_i = torch.var(j(self.x, scale))
            pve_i = 1. - 1. / (1. + var_f_i)
            index_i = torch.zeros(self.M, 1);
            index_i[i, :] = 1
            PVE = PVE + index_i * pve_i
        return PVE

class PVESampler_InfoGaussian(nn.Module):
    """
    Empirical pve draw from BNN priors (Informative Gaussian prior). M samples are drawn each time.
    """
    def __init__(self, x, M, num_hidden_nodes, k_min, k_max, tau):
        super(PVESampler_InfoGaussian, self).__init__()
        self.scale_ = nn.Parameter(torch.Tensor(1).uniform_(-2, -1))
        self.x = x
        self.M = M
        self.BNNs = nn.ModuleList([Info_Gaussian_NN_prior(x.shape[1], num_hidden_nodes, k_min, k_max, tau) for i in range(M)])

    def forward(self):
        PVE = torch.zeros(self.M, 1)
        scale = torch.log(1 + torch.exp(self.scale_))
        for i, j in enumerate(self.BNNs):
            # forward passing for each gene
            var_f_i = torch.var(j(self.x, scale))
            pve_i = 1. - 1. / (1. + var_f_i)
            index_i = torch.zeros(self.M, 1);
            index_i[i, :] = 1
            PVE = PVE + index_i * pve_i
        return PVE


