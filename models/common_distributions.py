#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
import math
from scipy.special import gamma
from abc import ABCMeta, abstractmethod
from utils.utils import *

class Distribution(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass
class ReparametrizedGaussian(Distribution):
    """
    Diagonal ReparametrizedGaussian distribution with parameters mu (mean) and rho. The standard
    deviation is parametrized as sigma = log(1 + exp(rho))
    A sample from the distribution can be obtained by sampling from a unit Gaussian,
    shifting the samples by the mean and scaling by the standard deviation:
    w = mu + log(1 + exp(rho)) * epsilon
    """
    def __init__(self, mu, rho):
        self.mean = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)
        self.point_estimate = self.mean

    @property
    def std_dev(self):
        return torch.log1p(torch.exp(self.rho))

#     def sample(self, n_samples=1):
#         epsilon = torch.distributions.Normal(0, 1).sample(sample_shape=(n_samples, *self.mean.size()))
#         return self.mean + self.std_dev * epsilon

    def sample(self):
        epsilon = torch.distributions.Normal(0, 1).sample(self.mean.size())
        return self.mean + self.std_dev * epsilon
    
    def logprob(self, target):
        return (-math.log(math.sqrt(2 * math.pi))
                    - torch.log(self.std_dev)
                    - ((target - self.mean) ** 2) / (2 * self.std_dev ** 2)).sum()

    def entropy(self):
        """
        Computes the entropy of the Diagonal Gaussian distribution.
        Details on the computation can be found in the 'diagonal_gaussian_entropy' notes in the repo
        """
        if self.mean.dim() > 1:
            n_inputs, n_outputs = self.mean.shape
        else:
            n_inputs = len(self.mean)
            n_outputs = 1

        part1 = (n_inputs * n_outputs) / 2 * (torch.log(torch.tensor([2 * math.pi])) + 1)
        part2 = torch.sum(torch.log(self.std_dev))

        return part1 + part2

class Gamma(Distribution):
    """ Gamma distribution """
    def __init__(self, shape, rate):
        """
        Class constructor, sets parameters
        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        Raises:
            TypeError: if given rate or shape are not floats
            ValueError: if given rate or shape are not positive
        """
        if not isinstance(shape, float) or not isinstance(rate, float):
            raise TypeError("Shape and rate should be floats!")

        if shape < 0 or rate < 0:
            raise ValueError("Shape and rate must be positive!")

        self.shape = shape
        self.rate = rate
        self.mean = self.shape / self.rate
        self.variance = self.shape / self.rate**2
        self.point_estimate = self.mean

    def update(self, shape, rate):
        """
        Updates mean and variance automatically when a and b get updated
        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        Raises:
            TypeError: if given rate or shape are not floats
            ValueError: if given rate or shape are not positive
        """
        if not isinstance(shape, float) or not isinstance(rate, float):
            raise TypeError("Shape and rate should be floats!")

        if shape < 0 or rate < 0:
            raise ValueError("Shape and rate must be positive!")

        self.shape = shape
        self.rate = rate
        self.mean = shape / rate
        self.variance = shape / rate ** 2

class InverseGamma(Distribution):
    """ Inverse Gamma distribution """
    def __init__(self, shape, rate):
        """
        Class constructor, sets parameters of the distribution.
        Args:
            shape: torch tensor of floats, shape parameters of the distribution
            rate: torch tensor of floats, rate parameters of the distribution
        """
        self.shape = shape
        self.rate = rate

    def exp_inverse(self):
        """
        Calculates the expectation E[1/x], where x follows
        the inverse gamma distribution
        """
        return self.shape / self.rate

    def exp_log(self):
        """
        Calculates the expectation E[log(x)], where x follows
        the inverse gamma distribution
        """
        exp_log = torch.log(self.rate) - torch.digamma(self.shape)
        return exp_log

    def entropy(self):
        """
        Calculates the entropy of the inverse gamma distribution
        """
        entropy =  self.shape + torch.log(self.rate) + torch.lgamma(self.shape) \
                     - (1 + self.shape) * torch.digamma(self.shape)
        return torch.sum(entropy)

    def logprob(self, target):
        """
        Computes the value of the predictive log likelihood at the target value
        Args:
            target: Torch tensor of floats, point(s) to evaluate the logprob
        Returns:
            loglike: float, the log likelihood
        """
        part1 = (self.rate**self.shape) / gamma(self.shape)
        part2 = target**(-self.shape - 1)
        part3 = torch.exp(-self.rate / target)

        return torch.log(part1 * part2 * part3)

    def update(self, shape, rate):
        """
        Updates shape and rate of the distribution
        Args:
            shape: float, shape parameter of the distribution
            rate: float, rate parameter of the distribution
        """
        self.shape = shape
        self.rate = rate

class Discrete_Flatten_Laplace(Distribution):
    """
    Flatten Laplace Distribution with mode interval [mu_dowm, mu_up], and precision parameter tau.
    """
    def __init__(self, mu_down, mu_up, tau, D):
        self.mu_up = torch.tensor(mu_up)
        self.mu_down = torch.tensor(mu_down)
        self.tau = tau
        self.domain = torch.tensor(np.linspace(0, D, D + 1)).float()

    @property
    def constant(self):
        return torch.sum(torch.exp(- self.tau * torch.sqrt(self.flatten(self.domain) ** 2)))

    def pmf(self):
        return torch.exp(- self.tau * torch.sqrt(self.flatten(self.domain) ** 2)) / self.constant

    def sample(self):
        return torch.multinomial(self.pmf(), 1, replacement=False)

    def flatten(self, x):
        return torch.max(
            torch.max(torch.max(self.mu_down - x, torch.zeros(1)), torch.max(-self.mu_up + x, torch.zeros(1))),
            torch.zeros(1))
