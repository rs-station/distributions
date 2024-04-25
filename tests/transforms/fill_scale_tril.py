import pytest
from rs_distributions.transforms.fill_scale_tril import FillScaleTriL
import torch
from torch.distributions.constraints import lower_cholesky


@pytest.mark.parametrize("input_shape", [(6,), (10,)])
def test_forward_transform(input_shape):
    transform = FillScaleTriL()
    input_vector = torch.randn(input_shape)
    transformed_vector = transform._call(input_vector)

    assert isinstance(transformed_vector, torch.Tensor)
    assert transformed_vector.shape == (
        (-1 + torch.sqrt(torch.tensor(1 + input_shape[0] * 8))) / 2,
        (-1 + torch.sqrt(torch.tensor(1 + input_shape[0] * 8))) / 2,
    )
    assert lower_cholesky.check(transformed_vector)


@pytest.mark.parametrize("input_vector", [torch.randn(3), torch.randn(6)])
def test_forward_equals_inverse(input_vector):
    transform = FillScaleTriL()
    L = transform._call(input_vector)
    invL = transform._inverse(L)

    assert torch.allclose(input_vector, invL, atol=1e-6)
