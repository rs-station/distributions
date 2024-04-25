import numpy as np
import torch
import math
from torch import distributions as dist


class RiceIRSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nu, sigma, samples, dnu, dsigma):
        ctx.save_for_backward(dnu, dsigma)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        (
            grad_nu,
            grad_sigma,
        ) = ctx.saved_tensors
        return grad_output * grad_nu, grad_output * grad_sigma, None, None, None


class Rice(dist.Distribution):
    """
    The Rice distribution is useful for modeling acentric structure factor amplitudes in
    X-ray crystallography. It is the amplitude distribution corresponding to a bivariate
    normal in the complex plane.

    ```
    x ~ MVN([ν, 0], σI)
    y = sqrt(x[0] * x[0] + x[1] * x[1])
    ```
    The parameters ν and σ represent the location and standard deviation of an isotropic, bivariate normal.
    If x is drawn from the normal with location [ν, 0i] and covariance,
    ```
    | σ   0 |
    | 0  σi |
    ```
    the distribution of amplitudes, `y = sqrt(x * conjugate(x))`, follows a Rician distribution.

    Args:
        nu (float or Tensor): location parameter of the underlying bivariate normal
        sigma (float or Tensor): standard deviation of the underlying bivariate normal (must be positive)
        validate_args (bool, optional): Whether to validate the arguments of the distribution.
        Default is None.
    """

    arg_constraints = {
        "nu": dist.constraints.nonnegative,
        "sigma": dist.constraints.positive,
    }
    support = torch.distributions.constraints.nonnegative

    def __init__(self, nu, sigma, validate_args=None):
        self.nu, self.sigma = torch.distributions.utils.broadcast_all(nu, sigma)
        batch_shape = self.nu.shape
        super().__init__(batch_shape, validate_args=validate_args)
        self._irsample = RiceIRSample().apply

    def log_prob(self, value):
        """
        Compute the log-probability of the given values under the Rice distribution

        ```
        Rice(x | nu, sigma) = \
            x * sigma**-2 * exp(-0.5 * (x**2 + nu**2) * sigma ** -2) * I_0(x * nu * sigma **-2)
        ```

        Args:
            value (Tensor): The values at which to evaluate the log-probability

        Returns:
            Tensor: The log-probabilities of the given values
        """
        if self._validate_args:
            self._validate_sample(value)
        nu, sigma = self.nu, self.sigma
        x = value
        log_prob = \
            torch.log(x) - 2.*torch.log(sigma) - \
            0.5 * torch.square((x-nu)/sigma) + torch.log(
            torch.special.i0e(nu * x / (sigma*sigma))
        )
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        """
        Generate random samples from the Rice distribution

        Args:
            sample_shape (torch.Size, optional): The shape of the samples to generate.
            Default is an empty shape

        Returns:
            Tensor: The generated random samples
        """
        shape = self._extended_shape(sample_shape)
        nu, sigma = self.nu, self.sigma
        nu = nu.expand(shape)
        sigma = sigma.expand(shape)
        with torch.no_grad():
            A = torch.normal(nu, sigma)
            B = torch.normal(torch.zeros_like(nu), sigma)
            z = torch.sqrt(A * A + B * B)
            return z

    @property
    def mean(self):
        """
        Compute the mean of the Rice distribution

        Returns:
            Tensor: The mean of the distribution.
        """
        sigma = self.sigma
        nu = self.nu

        x = -0.5 * torch.square(nu / sigma)
        L = (1. - x) * torch.special.i0e(-0.5*x) - x * torch.special.i1e(-0.5 * x)
        mean = sigma * math.sqrt(math.pi / 2.0) * L
        return mean

    @property
    def variance(self):
        """
        Compute the variance of the Rice distribution

        Returns:
            Tensor: The variance of the distribution
        """
        nu,sigma = self.nu,self.sigma
        n2 = nu * nu
        sigma = self.sigma
        return 2*sigma*sigma + nu*nu - torch.square(self.mean)

    def cdf(self, value):
        """
        Args:
            value (Tensor): The values at which to evaluate the CDF

        Returns:
            Tensor: The CDF values at the given values
        """
        raise NotImplementedError("The CDF is not implemented")

    def _grad_z(self, samples):
        """
        Return the gradient of samples from this distribution

        Args:
            samples (Tensor): samples from this distribution

        Returns:
            dnu: gradient with respect to the loc parameter, nu
            dsigma: gradient with respect to the underlying normal's scale parameter, sigma
        """
        z = samples
        nu, sigma = self.nu, self.sigma
        ab = z * nu / (sigma * sigma)
        dnu = torch.special.i1e(ab) / torch.special.i0e(ab) #== i1(ab)/i0(ab)
        dsigma = (z - nu * dnu)/sigma
        return dnu, dsigma

    def pdf(self, value):
        return torch.exp(self.log_prob(value))

    def rsample(self, sample_shape=torch.Size()):
        """
        Generate differentiable random samples from the Rice distribution.
        Gradients are implemented using implicit reparameterization (https://arxiv.org/abs/1805.08498).

        Args:
            sample_shape (torch.Size, optional): The shape of the samples to generate.
            Default is an empty shape

        Returns:
            Tensor: The generated random samples
        """
        samples = self.sample(sample_shape)
        dnu, dsigma = self._grad_z(samples)
        samples.requires_grad_(True)
        return self._irsample(self.nu, self.sigma, samples, dnu, dsigma)
