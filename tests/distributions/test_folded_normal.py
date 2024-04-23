import pytest
from rs_distributions.distributions.folded_normal import FoldedNormal
import torch


@pytest.mark.parametrize("test_float_broadcasting", [False, True])
@pytest.mark.parametrize("batch_shape", [3, 10])
@pytest.mark.parametrize("sample_shape", [(), (10,)])
def test_folded_normal_execution(test_float_broadcasting, batch_shape, sample_shape):
    params = torch.ones((2, batch_shape), requires_grad=True)
    loc, scale = params
    if test_float_broadcasting:
        loc = 1.0
    q = FoldedNormal(loc, scale)
    z = q.rsample(sample_shape)
    q.mean
    q.variance
    q.cdf(z)
    q.pdf(z)
    q.log_prob(z)
    torch.autograd.grad(z.sum(), params)
