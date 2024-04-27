import pytest
from rs_distributions.transforms.fill_scale_tril import FillScaleTriL
import torch
from torch.distributions.constraints import lower_cholesky


@pytest.mark.parametrize("batch_shape, d", [((2, 3), 6), ((1, 4, 5), 10)])
def test_forward_transform(batch_shape, d):
    transform = FillScaleTriL()
    input_shape = batch_shape + (d,)
    input_vector = torch.randn(input_shape)
    transformed_vector = transform._call(input_vector)

    n = int((-1 + torch.sqrt(torch.tensor(1 + 8 * d))) / 2)
    expected_output_shape = batch_shape + (n, n)
    cholesky_constraint_check = lower_cholesky.check(transformed_vector)

    assert isinstance(transformed_vector, torch.Tensor), "Output is not a torch.Tensor"
    assert (
        transformed_vector.shape == expected_output_shape
    ), f"Expected shape {expected_output_shape}, got {transformed_vector.shape}"
    assert cholesky_constraint_check.all()


@pytest.mark.parametrize("batch_shape, d", [((2, 3), 6), ((1, 4, 5), 10)])
def test_forward_equals_inverse(batch_shape, d):
    transform = FillScaleTriL()
    input_shape = batch_shape + (d,)
    input_vector = torch.randn(input_shape)
    L = transform._call(input_vector)
    invL = transform._inverse(L)

    n = int((-1 + torch.sqrt(torch.tensor(1 + 8 * d))) / 2)

    assert torch.allclose(
        input_vector, invL, atol=1e-4
    ), "Original input and the result of applying inverse transformation are not close enough"
