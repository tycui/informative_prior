import torch


class SteinGradientEstimator(object):
    """
    Implementation of stein gradient estimator of implicit distributions. Only samples the from implicit distribution are required.
    """
    def __init__(self, eta):
        self.eta = eta

    def rbf_kernel(self, x1, x2, kernel_width):
        kernel = torch.exp(-torch.sum((x1 - x2) ** 2, dim=-1) / (2. * (kernel_width ** 2)))
        return kernel

    def gram(self, x1, x2, kernel_width):
        x_row = x1.unsqueeze(-2)  # x_row = [n1, 1, x_dim]
        x_col = x2.unsqueeze(-3)  # x_col = [1, n1, x_dim]
        gram_matrix = self.rbf_kernel(x_row, x_col, kernel_width)
        return gram_matrix

    def grad_gram(self, x1, x2, kernel_width):
        x_row = x1.unsqueeze(-2)
        x_col = x2.unsqueeze(-3)
        G = self.rbf_kernel(x_row, x_col, kernel_width)
        diff = (x_row - x_col) / (kernel_width ** 2)
        G_expand = G.unsqueeze(-1)
        grad_x2 = G_expand * diff
        grad_x1 = G_expand * (-diff)
        return G, grad_x1, grad_x2

    def heuristic_kernel_width(self, x1, x2):
        x_row = x1.unsqueeze(-2)
        x_col = x2.unsqueeze(-3)
        pairwise_distance = torch.sqrt(torch.sum((x_row - x_col) ** 2, dim=-1))
        kernel_width = torch.median(pairwise_distance).detach()
        return kernel_width

    def compute_gradients(self, samples):
        M = samples.shape[0]
        l = self.heuristic_kernel_width(samples, samples)
        K, grad_K1, grad_K2 = self.grad_gram(samples, samples, l)
        Kinv = torch.inverse(K + self.eta * torch.eye(M))
        H_dh = torch.sum(grad_K2, dim=-2)
        grads = -torch.matmul(Kinv, H_dh)
        return grads


def minimize_kldivergence(sampler, prior, optimizer, estimator, num_epoch):
    """
    minimizing the KL divergence between the empirical PVE and prior PVE.
    """
    for i in range(num_epoch):
        samples = sampler()  # Draw samples from sampler
        optimizer.zero_grad()  # THIS LINE IS VERY IMPORTANT!!
        dlog_q_samples = estimator.compute_gradients(
            samples)  # Estimate the gradient of implicit distribution via stein gradient estimator
        kl_loss = torch.mean(dlog_q_samples.detach() * samples) - prior.average_log_density(
            samples)  # Objective function of KL
        kl_loss.backward()  # backpropogate the gradient
        optimizer.step()  # optimize with SGD
        #         if (i % int(num_epoch / 10)) == 0:
        if (i % 10) == 0:
            print('EPOCH %d: KL: %.4f.' % (i, kl_loss.detach().numpy()))

    return torch.log(1 + torch.exp(sampler.scale_)).detach().numpy()