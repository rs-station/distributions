import numpy as np
import torch
from torch import distributions as dist



class NormalIRSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loc, scale, samples, dFdmu, dFdsig, q):
        dzdmu = -dFdmu/q
        dzdsig = -dFdsig/q
        ctx.save_for_backward(dzdmu, dzdsig)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        dzdmu, dzdsig, = ctx.saved_tensors
        return grad_output * dzdmu, grad_output * dzdsig, None, None, None, None


class FoldedNormal(dist.Distribution):
    """
    Folded Normal distribution class

    Args:
        loc (float or Tensor): location parameter of the distribution
        scale (float or Tensor): scale parameter of the distribution (must be positive)
        validate_args (bool, optional): Whether to validate the arguments of the distribution.
        Default is None.
    """

    arg_constraints = {"loc": dist.constraints.real, "scale": dist.constraints.positive}

    def __init__(self, loc, scale, validate_args=None):
        self.loc = torch.as_tensor(loc)
        self.scale = torch.as_tensor(scale)
        batch_shape = self.loc.shape
        super().__init__(batch_shape, validate_args=validate_args)
        self._irsample = NormalIRSample().apply

    def log_prob(self, value):
        """
        Compute the log-probability of the given values under the Folded Normal distribution

        Args:
            value (Tensor): The values at which to evaluate the log-probability

        Returns:
            Tensor: The log-probabilities of the given values
        """
        loc = self.loc
        scale = self.scale
        log_prob = torch.logaddexp(
            dist.Normal(loc, scale).log_prob(value),
            dist.Normal(-loc, scale).log_prob(value),
        )
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        """
        Generate random samples from the Folded Normal distribution

        Args:
            sample_shape (torch.Size, optional): The shape of the samples to generate.
            Default is an empty shape

        Returns:
            Tensor: The generated random samples
        """
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        samples = torch.abs(eps * self.scale + self.loc)

        return samples
        
    def mean(self):
        """
        Compute the mean of the Folded Normal distribution

        Returns:
            Tensor: The mean of the distribution.
        """
        loc = self.loc
        scale = self.scale
        return scale * torch.sqrt(torch.tensor(2.0) / torch.pi) * torch.exp(
            -0.5 * (loc / scale) ** 2
        ) + loc * (1 - 2 * dist.Normal(0, 1).cdf(-loc / scale))

    def variance(self):
        """
        Compute the variance of the Folded Normal distribution

        Returns:
            Tensor: The variance of the distribution
        """
        loc = self.loc
        scale = self.scale
        return loc**2 + scale**2 - self.mean() ** 2

    def cdf(self, value):
        """
        Args:
            value (Tensor): The values at which to evaluate the CDF

        Returns:
            Tensor: The CDF values at the given values
        """
        value = torch.as_tensor(value, dtype=self.loc.dtype, device=self.loc.device)
        # return dist.Normal(loc, scale).cdf(value) - dist.Normal(-loc, scale).cdf(-value)
        return 0.5 * (torch.erf((value + self.loc)/(self.scale * np.sqrt(2.0)))  + torch.erf((value - self.loc)/(self.scale * np.sqrt(2.0))))

    def dcdfdmu(self, value):
        return torch.exp(dist.Normal(-self.loc, self.scale).log_prob(value)) - torch.exp(dist.Normal(self.loc, self.scale).log_prob(value))

    def dcdfdsigma(self, value):
        A = (-(value + self.loc)/self.scale) * torch.exp(dist.Normal(-self.loc, self.scale).log_prob(value))
        B = (-(value - self.loc)/self.scale) * torch.exp(dist.Normal(self.loc, self.scale).log_prob(value))
        return A + B

    def pdf(self, value):
        return torch.exp(self.log_prob(value))

    def rsample(self, sample_shape=torch.Size()):
        samples = self.sample(sample_shape)
        F = self.cdf(samples)
        q = self.pdf(samples)
        dFdmu = self.dcdfdmu(samples)
        dFdsigma = self.dcdfdsigma(samples)
        samples.requires_grad_(True)
        return self._irsample(self.loc, self.scale, samples, dFdmu, dFdsigma, q)

