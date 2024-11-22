import pytest
from rs_distributions.distributions.rice import Rice
from scipy.stats import rice
import torch
import numpy as np


@pytest.mark.parametrize("test_float_broadcasting", [False, True])
@pytest.mark.parametrize("batch_shape", [3, 10])
@pytest.mark.parametrize("sample_shape", [(), (10,)])
def test_rice_execution(test_float_broadcasting, batch_shape, sample_shape):
    params = torch.ones((2, batch_shape), requires_grad=True)
    loc, scale = params
    if test_float_broadcasting:
        loc = 1.0
    q = Rice(loc, scale)
    z = q.rsample(sample_shape)
    q.mean
    q.variance
    # q.cdf(z) #<-- no cdf implementation
    q.pdf(z)
    q.log_prob(z)
    torch.autograd.grad(z.sum(), params)


def test_rice_against_scipy(
    dtype="float32", snr_cutoff=10.0, log_min=-12, log_max=2, rtol=1e-5
):
    """
    Test the following attributes of Rice against scipy.stats.rice
     - mean

    Args:
      dtype (string) : float dtype to determine the precision of the test
      snr_cutoff (float) : do not test combinations with nu/sigma values above this threshold
      log_min (int) : 10**log_min will be the minimum value tested for nu and sigma
      log_max (int) : 10**log_max will be the maximum value tested for nu and sigma
      rtol (float) : the relative tolerance for equivalency in tests
    """
    log_min, log_max = -12, 2
    nu = sigma = np.logspace(log_min, log_max, log_max - log_min + 1, dtype=dtype)
    nu, sigma = np.meshgrid(nu, sigma)
    nu, sigma = nu.flatten(), sigma.flatten()
    idx = nu / sigma < snr_cutoff
    nu, sigma = nu[idx], sigma[idx]

    q = Rice(
        torch.as_tensor(nu, dtype=torch.float32),
        torch.as_tensor(sigma, dtype=torch.float32),
    )

    mean = rice.mean(nu / sigma, scale=sigma).astype(dtype)
    result = q.mean.detach().numpy()
    assert np.allclose(mean, result, rtol=rtol)

    stddev = rice.std(nu / sigma, scale=sigma).astype(dtype)
    result = q.stddev.detach().numpy()
    assert np.allclose(stddev, result, rtol=rtol)

    z = np.linspace(
        np.maximum(mean - 3.0 * stddev, 0.0),
        mean + 3.0 * stddev,
        10,
    )

    log_prob = rice.logpdf(z, nu / sigma, scale=sigma)
    result = q.log_prob(torch.as_tensor(z)).detach().numpy()
    assert np.allclose(log_prob, result, rtol=rtol)

    pdf = rice.pdf(z, nu / sigma, scale=sigma)
    result = q.pdf(torch.as_tensor(z)).detach().numpy()
    assert np.allclose(pdf, result, rtol=rtol)
