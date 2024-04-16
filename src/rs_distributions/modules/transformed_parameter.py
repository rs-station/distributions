import torch


class TransformedParameter(torch.nn.Module):
    """
    A `torch.nn.Module` subclass representing a constrained variabled.
    """

    def __init__(self, value, transform):
        """
        Args:
            value : Tensor
                The initial value of this learnable parameter
            transform : torch.distributions.Transform
                A transform instance which is applied to the underlying, unconstrained value
        """
        super().__init__()
        value = torch.as_tensor(value)  # support floats
        if isinstance(value, torch.nn.Parameter):
            self._value = value
            value.data = transform.inv(value)
        else:
            self._value = torch.nn.Parameter(transform.inv(value))
        self.transform = transform

    def forward(self):
        return self.transform(self._value)
