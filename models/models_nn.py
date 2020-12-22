from models.layers import *

class Hier_Meanfield_NN(nn.Module):
    """
    Hiericical Meanfield Gaussian BNN
    """

    def __init__(self, num_feature, num_hidden_nodes, alpha_s=0.5, beta_s=0.5, alpha_t=0.5, beta_t=0.5):
        super(Hier_Meanfield_NN, self).__init__()
        self.num_feature = num_feature
        self.alpha = alpha_s
        self.beta = beta_s
        scale = 1. * np.sqrt(6. / (num_feature + 1))
        self.mu_logsigma = nn.Parameter(torch.Tensor(1).uniform_(-scale, scale))
        self.rho_logsigma = nn.Parameter(torch.Tensor(1).uniform_(-4, -2))
        self.relu = nn.ReLU()

        self.Layer1 = Hier_Meanfield_Layer(num_feature, num_hidden_nodes[0], alpha_t, beta_t)
        self.Layer2 = Hier_Meanfield_Layer(num_hidden_nodes[0], num_hidden_nodes[1], alpha_t, beta_t)
        self.Layer3 = Hier_Meanfield_Layer(num_hidden_nodes[1], 1, alpha_t, beta_t)

    def forward(self, x):
        var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))
        if self.training:
            epsilon_sigma = torch.randn(1)
            sigma_n = torch.exp(self.mu_logsigma + var_sigma * epsilon_sigma)
            x1, kl1 = self.Layer1(x, sigma_n, self.training)
            x1 = self.relu(x1)
            x2, kl2 = self.Layer2(x1, sigma_n, self.training)
            x2 = self.relu(x2)
            x3, kl3 = self.Layer3(x2, sigma_n, self.training)
        else:
            sigma_n = torch.exp(self.mu_logsigma)
            x1, kl1 = self.Layer1(x, sigma_n, self.training)
            x1 = self.relu(x1)
            x2, kl2 = self.Layer2(x1, sigma_n, self.training)
            x2 = self.relu(x2)
            x3, kl3 = self.Layer3(x2, sigma_n, self.training)

        KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
        KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
        KL_scale = KL_entropy_sigma - KL_prior_sigma

        return x3, kl1 + kl2 + kl3 + KL_scale, sigma_n

    def _entropy_inverse_gamma(self, mu, sigma):
        return torch.sum(mu + torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5)

    def _expected_prior_inverse_gamma(self, mu, sigma):
        return torch.sum(self.alpha * np.log(self.beta) - (self.alpha + 1) * mu - self.beta * torch.exp(
            -mu + 0.5 * sigma ** 2) - torch.lgamma(torch.tensor(self.alpha)))

class Info_Hier_Meanfield_NN(nn.Module):
    """
    Informative Hiericical Meanfield Gaussian BNN
    """

    def __init__(self, num_feature, num_hidden_nodes, k_min, k_max, tau=1., alpha_s=0.5, beta_s=0.5, alpha_t=0.5, beta_t=0.5):
        super(Info_Hier_Meanfield_NN, self).__init__()
        self.num_feature = num_feature
        self.alpha = alpha_s
        self.beta = beta_s
        scale = 1. * np.sqrt(6. / (num_feature + 1))
        self.mu_logsigma = nn.Parameter(torch.Tensor(1).uniform_(-scale, scale))
        self.rho_logsigma = nn.Parameter(torch.Tensor(1).uniform_(-4, -2))
        self.relu = nn.ReLU()

        self.Layer1 = Info_Hier_Meanfield_Layer(num_feature, num_hidden_nodes[0], k_min, k_max, tau, alpha_t, beta_t)
        self.Layer2 = Hier_Meanfield_Layer(num_hidden_nodes[0], num_hidden_nodes[1], alpha_t, beta_t)
        self.Layer3 = Hier_Meanfield_Layer(num_hidden_nodes[1], 1, alpha_t, beta_t)

    def forward(self, x):
        var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))
        if self.training:
            epsilon_sigma = torch.randn(1)
            sigma_n = torch.exp(self.mu_logsigma + var_sigma * epsilon_sigma)
            x1, kl1 = self.Layer1(x, sigma_n, self.training)
            x1 = self.relu(x1)
            x2, kl2 = self.Layer2(x1, sigma_n, self.training)
            x2 = self.relu(x2)
            x3, kl3 = self.Layer3(x2, sigma_n, self.training)
        else:
            sigma_n = torch.exp(self.mu_logsigma)
            x1, kl1 = self.Layer1(x, sigma_n, self.training)
            x1 = self.relu(x1)
            x2, kl2 = self.Layer2(x1, sigma_n, self.training)
            x2 = self.relu(x2)
            x3, kl3 = self.Layer3(x2, sigma_n, self.training)

        KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
        KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
        KL_scale = KL_entropy_sigma - KL_prior_sigma

        return x3, kl1 + kl2 + kl3 + KL_scale, sigma_n

    def _entropy_inverse_gamma(self, mu, sigma):
        return torch.sum(mu + torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5)

    def _expected_prior_inverse_gamma(self, mu, sigma):
        return torch.sum(self.alpha * np.log(self.beta) - (self.alpha + 1) * mu - self.beta * torch.exp(
            -mu + 0.5 * sigma ** 2) - torch.lgamma(torch.tensor(self.alpha)))

class Hier_Meanfield_PVE_NN(nn.Module):
    """
    Hiericical Meanfield Gaussian BNN
    """

    def __init__(self, num_feature, num_hidden_nodes, alpha_s=0.5, beta_s=0.5, alpha_t=0.5, beta_t=0.5):
        super(Hier_Meanfield_PVE_NN, self).__init__()
        self.num_feature = num_feature
        self.alpha = alpha_s
        self.beta = beta_s
        scale = 1. * np.sqrt(6. / (num_feature + 1))
        self.mu_logsigma = nn.Parameter(torch.Tensor(1).uniform_(-scale, scale))
        self.rho_logsigma = nn.Parameter(torch.Tensor(1).uniform_(-4, -2))
        self.relu = nn.ReLU()

        self.Layer1 = Hier_Meanfield_Layer(num_feature, num_hidden_nodes[0], alpha_t, beta_t)
        self.Layer2 = Hier_Meanfield_Layer(num_hidden_nodes[0], num_hidden_nodes[1], alpha_t, beta_t)
        self.Layer3 = Hier_Meanfield_Layer(num_hidden_nodes[1], 1, alpha_t, beta_t)

    def forward(self, x):
        var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))
        if self.training:
            epsilon_sigma = torch.randn(1)
            sigma_n = torch.exp(self.mu_logsigma + var_sigma * epsilon_sigma)
            x1, kl1 = self.Layer1(x, 1., self.training)
            x1 = self.relu(x1)
            x2, kl2 = self.Layer2(x1, 1., self.training)
            x2 = self.relu(x2)
            x3, kl3 = self.Layer3(x2, sigma_n, self.training)
        else:
            sigma_n = torch.exp(self.mu_logsigma)
            x1, kl1 = self.Layer1(x, 1., self.training)
            x1 = self.relu(x1)
            x2, kl2 = self.Layer2(x1, 1., self.training)
            x2 = self.relu(x2)
            x3, kl3 = self.Layer3(x2, sigma_n, self.training)

        KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
        KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
        KL_scale = KL_entropy_sigma - KL_prior_sigma

        return x3, kl1 + kl2 + kl3 + KL_scale, sigma_n

    def _entropy_inverse_gamma(self, mu, sigma):
        return torch.sum(mu + torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5)

    def _expected_prior_inverse_gamma(self, mu, sigma):
        return torch.sum(self.alpha * np.log(self.beta) - (self.alpha + 1) * mu - self.beta * torch.exp(
            -mu + 0.5 * sigma ** 2) - torch.lgamma(torch.tensor(self.alpha)))

class Info_Hier_Meanfield_PVE_NN(nn.Module):
    """
    Informative Hiericical Meanfield Gaussian BNN
    """

    def __init__(self, num_feature, num_hidden_nodes, k_min, k_max, tau=1., alpha_s=0.5, beta_s=0.5, alpha_t=0.5, beta_t=0.5):
        super(Info_Hier_Meanfield_PVE_NN, self).__init__()
        self.num_feature = num_feature
        self.alpha = alpha_s
        self.beta = beta_s
        scale = 1. * np.sqrt(6. / (num_feature + 1))
        self.mu_logsigma = nn.Parameter(torch.Tensor(1).uniform_(-scale, scale))
        self.rho_logsigma = nn.Parameter(torch.Tensor(1).uniform_(-4, -2))
        self.relu = nn.ReLU()

        self.Layer1 = Info_Hier_Meanfield_Layer(num_feature, num_hidden_nodes[0], k_min, k_max, tau, alpha_t, beta_t)
        self.Layer2 = Hier_Meanfield_Layer(num_hidden_nodes[0], num_hidden_nodes[1], alpha_t, beta_t)
        self.Layer3 = Hier_Meanfield_Layer(num_hidden_nodes[1], 1, alpha_t, beta_t)

    def forward(self, x):
        var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))
        if self.training:
            epsilon_sigma = torch.randn(1)
            sigma_n = torch.exp(self.mu_logsigma + var_sigma * epsilon_sigma)
            x1, kl1 = self.Layer1(x, 1., self.training)
            x1 = self.relu(x1)
            x2, kl2 = self.Layer2(x1, 1., self.training)
            x2 = self.relu(x2)
            x3, kl3 = self.Layer3(x2, sigma_n, self.training)
        else:
            sigma_n = torch.exp(self.mu_logsigma)
            x1, kl1 = self.Layer1(x, 1., self.training)
            x1 = self.relu(x1)
            x2, kl2 = self.Layer2(x1, 1., self.training)
            x2 = self.relu(x2)
            x3, kl3 = self.Layer3(x2, sigma_n, self.training)

        KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
        KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
        KL_scale = KL_entropy_sigma - KL_prior_sigma

        return x3, kl1 + kl2 + kl3 + KL_scale, sigma_n

    def _entropy_inverse_gamma(self, mu, sigma):
        return torch.sum(mu + torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5)

    def _expected_prior_inverse_gamma(self, mu, sigma):
        return torch.sum(self.alpha * np.log(self.beta) - (self.alpha + 1) * mu - self.beta * torch.exp(
            -mu + 0.5 * sigma ** 2) - torch.lgamma(torch.tensor(self.alpha)))