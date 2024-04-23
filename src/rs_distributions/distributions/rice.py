import numpy as np
import torch
from torch import distributions as dist


class RiceIRSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loc, scale, samples, dloc, dscale, dz):
        grad_loc = -dloc / dz
        grad_scale = -dscale / dz
        ctx.save_for_backward(grad_loc, grad_scale)
        return samples

    @staticmethod
    def backward(ctx, grad_output):
        (
            grad_loc,
            grad_scale,
        ) = ctx.saved_tensors
        return grad_output * grad_loc, grad_output * grad_scale, None, None, None, None


class Rice(dist.Distribution):
    """
    The Rice distribution is useful for modeling acentric structure factor amplitudes in
    X-ray crystallography. It is the amplitude distribution corresponding to a bivariate
    normal in the complex plane. 

    ```
    x ~ MVN([ν, 0], σI)
    y = sqrt(x[0] * x[0] + x[1] * x[1])
    ```
    The parameters ν and σ represent the location and variance of a bivariate normal. 
    If x is drawn from the normal with location ν + 0i and covariance,
    ```
    | σ   0 |
    | 0  σi |
    ```
    the distribution of amplitudes, `y = sqrt(x * conjugate(x))`, follows a Rician distribution.

    Args:
        loc (float or Tensor): location parameter of the underlying bivariate normal
        scale (float or Tensor): scale parameter of the underlying bivariate normal (must be positive)
        validate_args (bool, optional): Whether to validate the arguments of the distribution.
        Default is None.
    """

    arg_constraints = {
        "loc": dist.constraints.nonnegative , 
        "scale": dist.constraints.positive,
    }
    support = torch.distributions.constraints.nonnegative

    def __init__(self, loc, scale, validate_args=None):
        self.loc = torch.as_tensor(loc)
        self.scale = torch.as_tensor(scale)
        batch_shape = self.loc.shape
        super().__init__(batch_shape, validate_args=validate_args)
        self._irsample = RiceIRSample().apply

    def _log_bessel_i0(self, x):
        return torch.log(torch.special.i0e(x)) + torch.abs(x)

    def _log_bessel_i1(self, x):
        return torch.log(torch.special.i1e(x)) + torch.abs(x)

    def _laguerre_half(self, x):
        return (1. - x) * torch.exp(x / 2. + self._log_bessel_i0(-0.5 * x)) - x * torch.exp(x / 2.  + self._log_bessel_i1(-0.5 * x) )

    def log_prob(self, value):
        """
        Compute the log-probability of the given values under the Folded Normal distribution

        ```
        Rice(x | loc, scale) = \
            x * scale**-2 * exp(-0.5 * (x**2 + loc**2) * scale ** -2) * I_0(x * loc * scale **-2)
        ```

        Args:
            value (Tensor): The values at which to evaluate the log-probability

        Returns:
            Tensor: The log-probabilities of the given values
        """
        loc,scale = self.loc,self.scale

        if self._validate_args:
            self._validate_sample(value)

        log_scale = torch.log(scale)
        x = value
        log_x = torch.log(value)
        log_loc = torch.log(loc)
        i0_arg= torch.exp(log_x + log_loc - 2. * log_scale)

        log_prob = log_x - 2.*log_scale - 0.5*(x * x + loc * loc) / (scale * scale)
        log_prob += self._log_bessel_i0(i0_arg)

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
        loc, scale = self.loc, self.scale
        A = scale * torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device) + loc
        B = scale * torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device) 
        z = torch.sqrt(A*A + B*B)
        return z

    @property
    def mean(self):
        """
        Compute the mean of the Rice distribution

        Returns:
            Tensor: The mean of the distribution.
        """

        raise NotImplementedError()
        loc = self.loc
        scale = self.scale
        return scale * torch.sqrt(torch.tensor(2.0) / torch.pi) * torch.exp(
            -0.5 * (loc / scale) ** 2
        ) + loc * (1 - 2 * dist.Normal(0, 1).cdf(-loc / scale))

    @property
    def variance(self):
        """
        Compute the variance of the Folded Normal distribution

        Returns:
            Tensor: The variance of the distribution
        """
        raise NotImplementedError()
        loc = self.loc
        scale = self.scale
        return loc**2 + scale**2 - self.mean**2

    def cdf(self, value):
        """
        Args:
            value (Tensor): The values at which to evaluate the CDF

        Returns:
            Tensor: The CDF values at the given values
        """
        raise NotImplementedError("The CDF is not implemented")

    def grad_cdf(self, samples):
        """
        Return the gradient of the CDF
        Args: 
            samples (Tensor): samples from this distribution

        Returns: 
            dloc: gradient with respect to the loc parameter, nu
            dscale: gradient with respect to the scale parameter, sigma
        """
        z = samples
        loc, scale = self.loc, self.scale
        log_z,log_loc,log_scale = torch.log(z),torch.log(loc),torch.log(scale)
        log_a = log_loc - log_scale
        log_b = log_z - log_scale
        ab = torch.exp(log_a + log_b)

        # dQ = b*exp(-0.5*(a*a + b*b)) (shared term)
        # log_dQ = log(b) -0.5*(a*a + b*b)
        # da = -dQ * I_1(a*b)
        # -log_da = log(dQ) + log_I1(a*b)
        # da = dQ * I_1(a*b)
        # log_db = log_dQ + log_I0(a*b) 
        log_dQ = log_b - 0.5 * (torch.exp(2.*log_a) + torch.exp(2.*log_b))
        log_da = log_dQ + self._log_bessel_i1(ab)
        log_db = log_dQ + self._log_bessel_i0(ab) 

        dz = torch.exp(log_db - log_scale)
        dloc = -torch.exp(log_da - log_scale)
        ### TODO: why is this correct? i thought it was the negative of this????
        dscale = torch.exp(log_da + log_loc - 2*log_scale) - \
                torch.exp(log_db + log_z - 2*log_scale)
        return dloc, dscale, dz

    def pdf(self, value):
        return torch.exp(self.log_prob(value))

    def rsample(self, sample_shape=torch.Size()):
        """
        Generate differentiable random samples from the Folded Normal distribution.
        Gradients are implemented using implicit reparameterization (https://arxiv.org/abs/1805.08498).

        Args:
            sample_shape (torch.Size, optional): The shape of the samples to generate.
            Default is an empty shape

        Returns:
            Tensor: The generated random samples
        """
        samples = self.sample(sample_shape)
        dloc,dscale,dz = self.grad_cdf(samples)
        samples.requires_grad_(True)
        return self._irsample(self.loc, self.scale, samples, dloc, dscale, dz)

