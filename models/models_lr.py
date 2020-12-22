from models.layers import *

class LR_DeltaSS(nn.Module):
    def __init__(self, num_feature=20, sigma=1, p=0.1, alpha_g=1., beta_g=0.5):
        super(LR_DeltaSS, self).__init__()
        self.num_feature = num_feature
        self.p = p
        self.sigma = sigma
        self.Layer1 = Delta_SpikeSlab_Layer_Withoutbias(num_feature, 1, p, sigma, alpha_g, beta_g)

    def forward(self, x):
        y, kl = self.Layer1(x, self.training)
        return y, kl


class LR_BetaSS(nn.Module):
    def __init__(self, num_feature=20, sigma=1, alpha_b=1., beta_b=1., alpha_g=1., beta_g=0.5):
        super(LR_BetaSS, self).__init__()
        self.num_feature = num_feature
        self.sigma = sigma
        self.Layer1 = Beta_SpikeSlab_Layer_Withoutbias(num_feature, 1, alpha_b, beta_b, alpha_g, beta_g, sigma)

    def forward(self, x):
        y, kl = self.Layer1(x, self.training)
        return y, kl


class LR_InfoSS(nn.Module):
    def __init__(self, num_feature=20, sigma=1, alpha_g=4., beta_g=4., k_min=0., k_max=10., tau=1.):
        super(LR_InfoSS, self).__init__()
        self.num_feature = num_feature
        self.sigma = sigma
        self.Layer1 = Info_SpikeSlab_Layer_Withoutbias(num_feature, 1, k_min, k_max, tau, alpha_g, beta_g, sigma)

    def forward(self, x):
        y, kl = self.Layer1(x, self.training)
        return y, kl


class LR_HS(nn.Module):
    def __init__(self, num_feature=20, b_g=1e-5, b_0=1):
        super(LR_HS, self).__init__()
        self.num_feature = num_feature

        self.Layer1 = Horseshoe_Layer_Withoutbias(num_feature, 1, b_g, b_0)

    def forward(self, x):
        y, kl = self.Layer1(x, self.training)
        return y, kl

    def _updates(self):
        self.Layer1.analytic_update()