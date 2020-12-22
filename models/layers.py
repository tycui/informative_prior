import torch.nn as nn
from models.common_distributions import *

class Horseshoe_Layer_Withoutbias(nn.Module):
    """ Variational Horseshoe prior layer without bias term """
    def __init__(self, input_dim, output_dim, b_g=1e-5, b_0=1):
        super(Horseshoe_Layer_Withoutbias, self).__init__()
        self.num_features = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        scale = 1. * np.sqrt(6. / (input_dim + output_dim))

        ## Prior Distribution
        self.shape = torch.tensor([0.5])
        self.b_0 = torch.tensor([1 / b_0 ** 2])
        self.b_g = torch.tensor([1 / b_g ** 2])

        ## Variational Posterior
        # layer-wise sparsity parameter
        self.mu_lognu = nn.Parameter(torch.Tensor(1).uniform_(-scale, -scale))
        self.rho_lognu = nn.Parameter(torch.Tensor(1).uniform_(-4, -2))
        self.lognu = ReparametrizedGaussian(self.mu_lognu, self.rho_lognu)

        # node_wise sparsity parameter
        self.mu_logtau = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-scale, -scale))
        self.rho_logtau = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-4, -2))
        self.logtau = ReparametrizedGaussian(self.mu_logtau, self.rho_logtau)

        # lambda of layer-wise sparsity (nu)
        self.lambda_nu_shape = self.shape
        self.lambda_nu_rate = self.b_g
        self.lambda_nu = InverseGamma(self.lambda_nu_shape, self.lambda_nu_rate)

        # lambda of node-wise sparsity (tau_i)
        self.lambda_tau_shape = self.shape * torch.ones(self.input_dim)
        self.lambda_tau_rate = self.b_0 * torch.ones(self.input_dim)
        self.lambda_tau = InverseGamma(self.lambda_tau_shape, self.lambda_tau_rate)

        # neural network weights
        self.mu_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.rho_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))
        self.beta = ReparametrizedGaussian(self.mu_beta, self.rho_beta)

    def log_prior(self):
        def exp_log_inverse_gamma(shape, exp_rate, exp_log_rate, exp_log_x, exp_x_inverse):
            exp_log = - torch.lgamma(shape) + shape * exp_log_rate - (shape + 1) * exp_log_x \
                      - exp_rate * exp_x_inverse

            return torch.sum(exp_log)

        def exp_log_gaussian(mean, std):
            dim = mean.shape[0] * mean.shape[1]
            exp_gaus = - 0.5 * dim * (torch.log(torch.tensor(2 * math.pi))) - 0.5 * (
                        torch.sum(mean ** 2) + torch.sum(std ** 2))
            return exp_gaus

        # Calculate E_q[ln p(\tau | \lambda)] for node-wise sparsity
        shape = self.shape
        exp_lambda_inverse = self.lambda_tau.exp_inverse()
        exp_log_lambda = self.lambda_tau.exp_log()
        exp_log_tau = self.logtau.mean
        exp_tau_inverse = torch.exp(-self.logtau.mean + 0.5 * self.logtau.std_dev ** 2)
        log_inv_gammas_weight = exp_log_inverse_gamma(shape, exp_lambda_inverse, -exp_log_lambda,
                                                      exp_log_tau, exp_tau_inverse)

        # E_q[ln p(\lambda)] for lambda of node-wise sparsity
        shape = self.shape
        rate = self.b_0
        log_inv_gammas_weight += exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_lambda, exp_lambda_inverse)

        # E_q[ln p(v | \lambda)] for layer-wise sparsity
        shape = self.shape
        exp_theta_inverse = self.lambda_nu.exp_inverse()
        exp_log_theta = self.lambda_nu.exp_log()
        exp_log_v = self.lognu.mean
        exp_v_inverse = torch.exp(-self.lognu.mean + 0.5 * self.lognu.std_dev ** 2)
        log_inv_gammas_global = exp_log_inverse_gamma(shape, exp_theta_inverse, -exp_log_theta,
                                                      exp_log_v, exp_v_inverse)

        # E_q[ln p(\lambda)] for lambda of layer-wise sparsity
        shape = self.shape
        rate = self.b_g
        log_inv_gammas_global += exp_log_inverse_gamma(shape, rate, np.log(rate),
                                                       exp_log_theta, exp_theta_inverse)

        # Add all expectations
        log_inv_gammas = log_inv_gammas_weight + log_inv_gammas_global

        # E_q[N(beta)]
        log_gaussian = exp_log_gaussian(self.beta.mean, self.beta.std_dev)

        return log_gaussian + log_inv_gammas

    def log_variational_posterior(self):

        entropy = self.beta.entropy() + 1. * (
                    self.logtau.entropy() + torch.sum(self.logtau.mean) + self.lambda_tau.entropy()) \
                  + 1. * (self.lognu.entropy() + torch.sum(self.lognu.mean) + self.lambda_nu.entropy())

        if sum(torch.isnan(entropy)).item() != 0:
            raise Exception("entropy/log_variational_posterior computation ran into nan!")
            print('self.beta.entropy(): ', self.beta.entropy())
            print('beta mean: ', self.beta.mean)
            print('beta std: ', self.beta.std_dev)

        return -entropy

    def forward(self, x, training):
        if training:
            beta = self.beta.sample()
            log_tau = self.logtau.sample()
            log_v = self.lognu.sample()
            mask = torch.exp(log_v) * torch.exp(log_tau).view(-1, 1)
            weight = beta * mask
            temp = 0.1;
            self.num_features = torch.sum(mask / (mask + temp))  # effective number of features
            output = torch.mm(x, weight)
            KL = self.log_variational_posterior() - self.log_prior()

            return output, KL

        else:

            beta = self.beta.mean
            log_tau = self.logtau.mean
            log_v = self.lognu.mean
            mask = torch.exp(log_v) * torch.exp(log_tau).view(-1, 1)
            weight = beta * mask
            temp = 0.1;
            self.num_features = torch.sum(mask / (mask + temp))  # effective number of features
            output = torch.mm(x, weight)
            KL = self.log_variational_posterior() - self.log_prior()

            return output, KL

    def analytic_update(self):

        new_shape = torch.Tensor([1])
        # new lambda rate is given by E[1/tau_i] + 1/b_0^2
        new_lambda_rate = torch.exp(-self.logtau.mean + 0.5 * (self.logtau.std_dev ** 2)) \
                          + self.b_0

        # new theta rate is given by E[1/v] + 1/b_g^2
        new_theta_rate = torch.exp(-self.lognu.mean + 0.5 * (self.lognu.std_dev ** 2)) \
                         + self.b_g

        self.lambda_tau.update(new_shape, new_lambda_rate)
        self.lambda_nu.update(new_shape, new_theta_rate)


class Delta_SpikeSlab_Layer_Withoutbias(nn.Module):
    """ Variational Delta spike-and-slab prior layer without bias term """
    def __init__(self, input_dim, output_dim=1., prior=0.1, sigma=1.0, alpha=1., beta=0.5):
        super(Delta_SpikeSlab_Layer_Withoutbias, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features = input_dim
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta

        self.p_prior = torch.tensor(prior)

        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.mu_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.rho_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))

        init = 0.5
        init_min = np.log(init) - np.log(1. - init)
        init_max = np.log(init) - np.log(1. - init)

        self.p_logit = nn.Parameter(torch.empty(input_dim).uniform_(init_min, init_max))

        ## approximated posterior of local scale
        self.mu_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-scale, scale))
        self.rho_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-4, -2))

    def forward(self, x, training):
        p = torch.sigmoid(self.p_logit)
        eps = 1e-7
        if training:
            # forward passing with stochastic
            epsilon_beta = torch.randn(self.input_dim, self.output_dim);
            epsilon_sigma = torch.randn(self.input_dim)

            var_beta = torch.log(1 + torch.exp(self.rho_beta));
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            beta = self.mu_beta + var_beta * epsilon_beta;
            sigma = torch.exp(self.mu_logsigma + var_sigma * epsilon_sigma)

            x = self._hard_concrete_relaxation(p, x)

            output = torch.mm(x * sigma, beta)

            # calculate KL
            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy_spike = torch.sum(p * torch.log(p + eps) + (1. - p) * torch.log(1. - p + eps))
            KL_prior_spike = -torch.sum(
                p * torch.log(self.p_prior + eps) + (1. - p) * torch.log(1. - self.p_prior + eps))
            KL_spike = KL_entropy_spike + KL_prior_spike

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL = KL_beta + KL_spike + KL_scale

            return output, KL

        else:

            output = torch.mm(x * p * torch.exp(self.mu_logsigma), self.mu_beta)
            var_beta = torch.log(1 + torch.exp(self.rho_beta));
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            # calculate KL
            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy_spike = torch.sum(p * torch.log(p + eps) + (1. - p) * torch.log(1. - p + eps))
            KL_prior_spike = -torch.sum(
                p * torch.log(self.p_prior + eps) + (1. - p) * torch.log(1. - self.p_prior + eps))
            KL_spike = KL_entropy_spike + KL_prior_spike

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL = KL_beta + KL_spike + KL_scale

            return output, KL

    def _hard_concrete_relaxation(self, p, x):
        eps = 1e-4
        temp = 0.3
        unif_noise = torch.rand(x.shape[1])
        s = (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(
            1 - unif_noise + eps))
        s = torch.sigmoid(s / temp)
        keep_prob = s

        x = x * keep_prob
        self.num_features = torch.sum(keep_prob)
        return x

    def _entropy_inverse_gamma(self, mu, sigma):
        return torch.sum(mu + torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5)

    def _expected_prior_inverse_gamma(self, mu, sigma):
        return torch.sum(self.alpha * np.log(self.beta) - (self.alpha + 1) * mu - self.beta * torch.exp(
            -mu + 0.5 * sigma ** 2) - torch.lgamma(torch.tensor(self.alpha)))


class Beta_SpikeSlab_Layer_Withoutbias(nn.Module):
    """ Variational Beta spike-and-slab prior layer without bias term """
    def __init__(self, input_dim, output_dim=1, alpha_b=1., beta_b=1., alpha_g=1., beta_g=0.5, sigma=0.6):
        super(Beta_SpikeSlab_Layer_Withoutbias, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_features = input_dim
        self.alpha_b = torch.tensor(alpha_b)
        self.beta_b = torch.tensor(beta_b)
        self.alpha_g = torch.tensor(alpha_g)
        self.beta_g = torch.tensor(beta_g)
        self.sigma = sigma

        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.mu_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.rho_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))

        init = 0.5
        init_min = np.log(init) - np.log(1. - init)
        init_max = np.log(init) - np.log(1. - init)

        self.p_logit = nn.Parameter(torch.empty(input_dim).uniform_(init_min, init_max))

        ## approximated posterior of local scale
        self.mu_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-scale, scale))
        self.rho_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-4, -2))

        self.pi_posterior = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, training):
        p = torch.sigmoid(self.p_logit)
        eps = 1e-7
        if training:
            # forward passing with stochastic
            epsilon_beta = torch.randn(self.input_dim, self.output_dim);
            epsilon_sigma = torch.randn(self.input_dim)

            var_beta = torch.log(1 + torch.exp(self.rho_beta));
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            beta = self.mu_beta + var_beta * epsilon_beta;
            sigma = torch.exp(self.mu_logsigma + var_sigma * epsilon_sigma)

            x = self._hard_concrete_relaxation(p, x)

            output = torch.mm(x * sigma, beta)

            # calculate KL
            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy = torch.sum(p * torch.log(p + eps) + (1. - p) * torch.log(1. - p + eps))
            Exp_prior = -torch.sum(
                p * torch.log(self.pi_posterior + eps) + (1. - p) * torch.log(1. - self.pi_posterior + eps))
            Exp_beta = (self.alpha_b - 1.) * torch.log(self.pi_posterior + eps) + (self.beta_b - 1.) * torch.log(
                1. - self.pi_posterior + eps) - torch.lgamma(self.alpha_b) - torch.lgamma(self.beta_b) + torch.lgamma(
                self.alpha_b + self.beta_b)
            KL_spike = KL_entropy + Exp_prior - Exp_beta

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL = KL_beta + KL_spike + KL_scale

            return output, KL

        else:

            output = torch.mm(x * p * torch.exp(self.mu_logsigma), self.mu_beta)
            var_beta = torch.log(1 + torch.exp(self.rho_beta));
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            # calculate KL
            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy = torch.sum(p * torch.log(p + eps) + (1. - p) * torch.log(1. - p + eps))
            Exp_prior = -torch.sum(
                p * torch.log(self.pi_posterior + eps) + (1. - p) * torch.log(1. - self.pi_posterior + eps))
            Exp_beta = (self.alpha_b - 1.) * torch.log(self.pi_posterior + eps) + (self.beta_b - 1.) * torch.log(
                1. - self.pi_posterior + eps) - torch.lgamma(self.alpha_b) - torch.lgamma(self.beta_b) + torch.lgamma(
                self.alpha_b + self.beta_b)
            KL_spike = KL_entropy + Exp_prior - Exp_beta

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL = KL_beta + KL_spike + KL_scale

            return output, KL

    def _hard_concrete_relaxation(self, p, x):
        eps = 1e-4
        temp = 0.3
        unif_noise = torch.rand(x.shape[1])
        s = (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(
            1 - unif_noise + eps))
        s = torch.sigmoid(s / temp);
        keep_prob = s

        x = x * keep_prob
        self.num_features = torch.sum(keep_prob)
        return x

    def _entropy_inverse_gamma(self, mu, sigma):
        return torch.sum(mu + torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5)

    def _expected_prior_inverse_gamma(self, mu, sigma):
        return torch.sum(self.alpha_g * np.log(self.beta_g) - (self.alpha_g + 1) * mu - self.beta_g * torch.exp(
            -mu + 0.5 * sigma ** 2) - torch.lgamma(self.alpha_g))


class Info_SpikeSlab_Layer_Withoutbias(nn.Module):
    """ Variational Informativve spike-and-slab prior layer without bias term """
    def __init__(self, input_dim, output_dim, k_min, k_max, tau=1., alpha_g=1., beta_g=1., sigma=1.):
        super(Info_SpikeSlab_Layer_Withoutbias, self).__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.tau = tau
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = sigma
        self.alpha_g = alpha_g
        self.beta_g = beta_g
        self.num_features = input_dim

        scale = 1. * np.sqrt(6. / (input_dim + output_dim))

        # approximated posterior
        self.mu_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.rho_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))

        init = 0.5
        init_min = np.log(init) - np.log(1. - init)
        init_max = np.log(init) - np.log(1. - init)

        self.p_logit = nn.Parameter(torch.empty(input_dim).uniform_(init_min, init_max))

        ## approximated posterior of local scale
        self.mu_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-scale, scale))
        self.rho_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-4, -2))

    def forward(self, x, training):
        p = torch.sigmoid(self.p_logit)
        eps = 1e-7
        if training:
            # forward passing with stochastic
            epsilon_beta = torch.randn(self.input_dim, self.output_dim);
            epsilon_sigma = torch.randn(self.input_dim)

            var_beta = torch.log(1 + torch.exp(self.rho_beta));
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            beta = self.mu_beta + var_beta * epsilon_beta;
            sigma = torch.exp(self.mu_logsigma + var_sigma * epsilon_sigma)

            x, keep_prob = self._hard_concrete_relaxation(p, x)
            # update number of features
            self.num_features = torch.sum(keep_prob)

            output = torch.mm(x * sigma, beta)

            # calculate KL
            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy = torch.sum(p * torch.log(p) + (1. - p) * torch.log(1. - p))
            KL_prior = self._expected_log_prior(keep_prob)
            KL_info = KL_entropy + KL_prior

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL = KL_beta + KL_info + KL_scale

            return output, KL

        else:
            _, keep_prob = self._hard_concrete_relaxation(p, x)

            output = torch.mm(x * p * torch.exp(self.mu_logsigma), self.mu_beta)
            var_beta = torch.log(1 + torch.exp(self.rho_beta));
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            # calculate KL
            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy = torch.sum(p * torch.log(p) + (1. - p) * torch.log(1. - p))
            KL_prior = self._expected_log_prior(keep_prob)
            KL_info = KL_entropy + KL_prior

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL = KL_beta + KL_info + KL_scale

            return output, KL

    def _hard_concrete_relaxation(self, p, x):
        eps = 1e-4
        temp = 0.3
        unif_noise = torch.rand(x.shape[1])
        s = (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(
            1 - unif_noise + eps))
        s = torch.sigmoid(s / temp);
        keep_prob = s

        x = x * keep_prob

        return x, keep_prob

    def _expected_log_prior(self, M):
        eps = 1e-7
        count = torch.sum(M)
        reg = torch.max(
            torch.max(torch.max(self.k_min - count, torch.zeros(1)), torch.max(-self.k_max + count, torch.zeros(1))),
            torch.zeros(1))
        reg = self.tau * reg - count * torch.log(count + eps) - (self.input_dim - count) * torch.log(
            self.input_dim - count + eps) + self.input_dim * torch.log(torch.tensor(self.input_dim) + eps)
        return reg

    def _entropy_inverse_gamma(self, mu, sigma):
        return torch.sum(mu + torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5)

    def _expected_prior_inverse_gamma(self, mu, sigma):
        return torch.sum(self.alpha_g * np.log(self.beta_g) - (self.alpha_g + 1) * mu - self.beta_g * torch.exp(
            -mu + 0.5 * sigma ** 2) - torch.lgamma(torch.tensor(self.alpha_g)))


class Hier_Meanfield_Layer(nn.Module):
    """
    Hiericical Meanfield Gaussian BNN layers
    """
    def __init__(self, input_dim, output_dim, alpha=0.5, beta=0.5):
        super(Hier_Meanfield_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = 1.
        self.alpha = alpha
        self.beta = beta
        self.num_features = input_dim

        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.mu_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.rho_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))

        self.mu_bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale))
        self.rho_bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -2))

        ## approximated posterior of local scale
        self.mu_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-scale, scale))
        self.rho_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-4, -2))

    def forward(self, x, sigma_noise, training):
        eps = 1e-7
        if training:
            # forward passing with stochastic
            epsilon_beta = torch.randn(self.input_dim, self.output_dim);
            epsilon_bias = torch.randn(self.output_dim)
            epsilon_sigma = torch.randn(self.input_dim)

            var_beta = torch.log(1 + torch.exp(self.rho_beta));
            var_bias = torch.log(1 + torch.exp(self.rho_bias))
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            beta = self.mu_beta + var_beta * epsilon_beta;
            bias = self.mu_bias + var_bias * epsilon_bias
            sigma = torch.exp(self.mu_logsigma + var_sigma * epsilon_sigma)

            output = torch.mm(x * sigma, beta * sigma_noise) + bias

            # calculate KL
            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)
            KL_bias = torch.sum(
                (var_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma ** 2) - torch.log(var_bias + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL = KL_beta + KL_bias + KL_scale

            return output, KL

        else:

            output = torch.mm(x * torch.exp(self.mu_logsigma), self.mu_beta * sigma_noise) + self.mu_bias
            var_beta = torch.log(1 + torch.exp(self.rho_beta));
            var_bias = torch.log(1 + torch.exp(self.rho_bias))
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            # calculate KL
            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)
            KL_bias = torch.sum(
                (var_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma ** 2) - torch.log(var_bias + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL = KL_beta + KL_bias + KL_scale

            return output, KL

    def _entropy_inverse_gamma(self, mu, sigma):
        return torch.sum(mu + torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5)

    def _expected_prior_inverse_gamma(self, mu, sigma):
        return torch.sum(self.alpha * np.log(self.beta) - (self.alpha + 1) * mu - self.beta * torch.exp(
            -mu + 0.5 * sigma ** 2) - torch.lgamma(torch.tensor(self.alpha)))


class Info_Hier_Meanfield_Layer(nn.Module):
    """
    Hiericical Meanfield Gaussian BNN layers with informative spike-and-slab scale for feature selection
    """
    def __init__(self, input_dim, output_dim, k_min, k_max, tau=1., alpha=0.5, beta=0.5):
        super(Info_Hier_Meanfield_Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = 1.
        self.alpha = alpha
        self.beta = beta
        self.k_min = k_min
        self.k_max = k_max
        self.tau = tau
        self.num_features = input_dim

        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.mu_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.rho_beta = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2))

        self.mu_bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale))
        self.rho_bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-4, -2))

        ## approximated posterior of local scale
        self.mu_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-scale, scale))
        self.rho_logsigma = nn.Parameter(torch.Tensor(self.input_dim).uniform_(-4, -2))

        init = 0.9
        init_min = np.log(init) - np.log(1. - init)
        init_max = np.log(init) - np.log(1. - init)
        self.p_logit = nn.Parameter(torch.empty(input_dim).uniform_(init_min, init_max))


    def forward(self, x, sigma_noise, training):
        eps = 1e-7
        p = torch.sigmoid(self.p_logit)
        if training:
            x, keep_prob = self._hard_concrete_relaxation(p, x)
            # forward passing with stochastic
            epsilon_beta = torch.randn(self.input_dim, self.output_dim)
            epsilon_bias = torch.randn(self.output_dim)
            epsilon_sigma = torch.randn(self.input_dim)

            var_beta = torch.log(1 + torch.exp(self.rho_beta))
            var_bias = torch.log(1 + torch.exp(self.rho_bias))
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            beta = self.mu_beta + var_beta * epsilon_beta
            bias = self.mu_bias + var_bias * epsilon_bias
            sigma = torch.exp(self.mu_logsigma + var_sigma * epsilon_sigma)

            output = torch.mm(x * sigma, beta * sigma_noise) + bias

            # calculate KL
            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)
            KL_bias = torch.sum(
                (var_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma ** 2) - torch.log(var_bias + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL_entropy = torch.sum(p * torch.log(p) + (1. - p) * torch.log(1. - p))
            KL_prior = self._expected_log_prior(keep_prob)
            KL_info = KL_entropy + KL_prior

            KL = KL_beta + KL_bias + KL_scale + KL_info

            return output, KL

        else:

            output = torch.mm(x * torch.exp(self.mu_logsigma) * p, self.mu_beta * sigma_noise) + self.mu_bias
            var_beta = torch.log(1 + torch.exp(self.rho_beta));
            var_bias = torch.log(1 + torch.exp(self.rho_bias))
            var_sigma = torch.log(1 + torch.exp(self.rho_logsigma))

            # calculate KL
            _, keep_prob = self._hard_concrete_relaxation(p, x)

            KL_beta = torch.sum(
                (var_beta ** 2 + self.mu_beta ** 2) / (2 * self.sigma ** 2) - torch.log(var_beta + eps) + np.log(
                    self.sigma) - 0.5)
            KL_bias = torch.sum(
                (var_bias ** 2 + self.mu_bias ** 2) / (2 * self.sigma ** 2) - torch.log(var_bias + eps) + np.log(
                    self.sigma) - 0.5)

            KL_entropy_sigma = -self._entropy_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_prior_sigma = self._expected_prior_inverse_gamma(self.mu_logsigma, var_sigma)
            KL_scale = KL_entropy_sigma - KL_prior_sigma

            KL_entropy = torch.sum(p * torch.log(p) + (1. - p) * torch.log(1. - p))
            KL_prior = self._expected_log_prior(keep_prob)
            KL_info = KL_entropy + KL_prior

            KL = KL_beta + KL_bias + KL_scale + KL_info

            return output, KL

    def _entropy_inverse_gamma(self, mu, sigma):
        return torch.sum(mu + torch.log(sigma) + 0.5 * np.log(2 * np.pi) + 0.5)

    def _expected_prior_inverse_gamma(self, mu, sigma):
        return torch.sum(self.alpha * np.log(self.beta) - (self.alpha + 1) * mu - self.beta * torch.exp(
            -mu + 0.5 * sigma ** 2) - torch.lgamma(torch.tensor(self.alpha)))

    def _hard_concrete_relaxation(self, p, x):
        eps = 1e-4
        temp = 0.1
        # limit_left = -0.1;
        # limit_right = 1.1
        unif_noise = torch.rand(x.shape[1])
        s = (torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(unif_noise + eps) - torch.log(
            1 - unif_noise + eps))
        s = torch.sigmoid(s / temp);
        keep_prob = s
        #         s_bar = s * (limit_right - limit_left) + limit_left
        #         keep_prob = torch.min(torch.ones_like(s_bar), torch.max(torch.zeros_like(s_bar), s_bar))
        x = x * keep_prob
        self.num_features = torch.sum(keep_prob)
        return x, keep_prob

    def _expected_log_prior(self, M):
        eps = 1e-7
        count = torch.sum(M)
        reg = torch.max(
            torch.max(torch.max(self.k_min - count, torch.zeros(1)), torch.max(-self.k_max + count, torch.zeros(1))),
            torch.zeros(1))
        reg = self.tau * reg - count * torch.log(count + eps) - (self.input_dim - count) * torch.log(
            self.input_dim - count + eps) + self.input_dim * torch.log(torch.tensor(self.input_dim) + eps)
        return reg
