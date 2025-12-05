import pytest
from rs_distributions.transforms.fill_scale_tril import (
    FillScaleTriL,
    FillTriL,
    DiagTransform,
)
import torch
from torch.distributions.constraints import lower_cholesky
from torch.distributions.transforms import SoftplusTransform, ExpTransform


@pytest.mark.parametrize("batch_shape, d", [((2, 3), 6), ((1, 4, 5), 10)])
def test_forward_transform(batch_shape, d):
    transform = FillScaleTriL()
    input_shape = batch_shape + (d,)
    input_vector = torch.randn(input_shape)
    transformed_vector = transform(input_vector)

    n = int((-1 + torch.sqrt(torch.tensor(1 + 8 * d))) / 2)
    expected_output_shape = batch_shape + (n, n)
    cholesky_constraint_check = lower_cholesky.check(transformed_vector)

    assert isinstance(transformed_vector, torch.Tensor), "Output is not a torch.Tensor"
    assert transformed_vector.shape == expected_output_shape, (
        f"Expected shape {expected_output_shape}, got {transformed_vector.shape}"
    )
    assert cholesky_constraint_check.all()


@pytest.mark.parametrize("batch_shape, d", [((2, 3), 6), ((1, 4, 5), 10)])
def test_forward_equals_inverse(batch_shape, d):
    transform = FillScaleTriL()
    input_shape = batch_shape + (d,)
    input_vector = torch.randn(input_shape)
    L = transform(input_vector)
    invL = transform.inv(L)

    assert torch.allclose(input_vector, invL, atol=1e-4), (
        "Original input and the result of applying inverse transformation are not close enough"
    )


@pytest.mark.parametrize(
    "batch_shape, d, diag_transform",
    [
        ((2, 3), 6, SoftplusTransform()),
        ((1, 4, 5), 10, SoftplusTransform()),
        ((2, 3), 6, ExpTransform()),
        ((1, 4, 5), 10, ExpTransform()),
    ],
)
def test_log_abs_det_jacobian_softplus_and_exp(batch_shape, d, diag_transform):
    transform = FillScaleTriL(diag_transform=diag_transform)
    filltril = FillTriL()
    diagtransform = DiagTransform(diag_transform=diag_transform)
    input_shape = batch_shape + (d,)
    input_vector = torch.randn(input_shape, requires_grad=True)
    transformed_vector = transform(input_vector)

    # Calculate gradients log_abs_det_jacobian from FillScaleTriL
    log_abs_det_jacobian = transform.log_abs_det_jacobian(
        input_vector, transformed_vector
    )

    # Extract diagonal elements from input and transformed vectors
    tril = filltril(input_vector)
    diagonal_transformed = diagtransform(tril)

    # Calculate diagonal gradients
    diag_jacobian = diagtransform.log_abs_det_jacobian(tril, diagonal_transformed)

    # Assert diagonal gradients are approximately equal
    assert torch.allclose(diag_jacobian, log_abs_det_jacobian, atol=1e-4)
