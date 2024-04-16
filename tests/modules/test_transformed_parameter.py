import pytest
from rs_distributions.modules import TransformedParameter
import torch


@pytest.mark.parametrize("shape", [(), 10])
def test_transformed_parameter(shape):
    value = 10.0
    eps = 1e-6
    transform = torch.distributions.ComposeTransform(
        [
            torch.distributions.AffineTransform(eps, 1.0),
            torch.distributions.ExpTransform(),
        ]
    )
    variable = TransformedParameter(value, transform)
    assert variable() == value

    params = list(variable.parameters())
    assert len(params) == 1

    loss = variable().square().sum()
    loss.backward()
    assert params[0].grad.isfinite().all()
    assert (params[0] != 0).all()
