import torch
from torch.distributions import Transform, constraints
from torch.distributions.transforms import SoftplusTransform
from torch.distributions.utils import vec_to_tril_matrix, tril_matrix_to_vec


class FillScaleTriL(Transform):
    def __init__(self, diag_transform=None, diag_shift=1e-06):
        """
        Converts a tensor into a lower triangular matrix with positive diagonal entries.

        Args:
            diag_transform: transformation used on diagonal to ensure positive values.
            Default is SoftplusTransform
            diag_shift (float): small offset to avoid diagonals very close to zero.
            Default offset is 1e-06

        """
        super().__init__()
        self.diag_transform = (
            diag_transform if diag_transform is not None else SoftplusTransform()
        )
        self.diag_shift = diag_shift

    @property
    def domain(self):
        return constraints.real_vector

    @property
    def codomain(self):
        return constraints.lower_cholesky

    @property
    def bijective(self):
        return True

    def _call(self, x):
        """
        Transform input vector to lower triangular.

        Args:
            x (torch.Tensor): Input vector to transform
        Returns:
            torch.Tensor: Transformed lower triangular matrix
        """
        x = vec_to_tril_matrix(x)
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        if self.diag_shift is not None:
            result = x.diagonal_scatter(
                self.diag_transform(diagonal + self.diag_shift), dim1=-2, dim2=-1
            )
        else:
            result = x.diagonal_scatter(self.diag_transform(diagonal), dim1=-2, dim2=-1)
        return result

    def _inverse(self, y):
        """
        Apply the inverse transformation to the input lower triangular matrix.

        Args:
            y (torch.Tensor): Invertible lower triangular matrix

        Returns:
            torch.Tensor: Inversely transformed vector

        """
        diagonal = y.diagonal(dim1=-2, dim2=-1)
        if self.diag_shift is not None:
            result = y.diagonal_scatter(
                self.diag_transform.inv(diagonal - self.diag_shift), dim1=-2, dim2=-1
            )
        else:
            result = y.diagonal_scatter(
                self.diag_transform.inv(diagonal), dim1=-2, dim2=-1
            )
        return tril_matrix_to_vec(result)

    def log_abs_det_jacobian(self, x, y):
        L = vec_to_tril_matrix(x)
        diag = L.diagonal(dim1=-2, dim2=-1)
        diag.requires_grad_(True)
        if self.diag_shift is not None:
            transformed_diag = self.diag_transform(diag + self.diag_shift)
        else:
            transformed_diag = self.diag_transform(diag)
        derivatives = torch.autograd.grad(
            outputs=transformed_diag,
            inputs=diag,
            grad_outputs=torch.ones_like(transformed_diag),
        )[0]
        log_det_jacobian = torch.log(torch.abs(derivatives)).sum()
        return log_det_jacobian
