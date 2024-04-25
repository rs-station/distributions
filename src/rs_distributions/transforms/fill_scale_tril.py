import torch
from torch.distributions import Transform, constraints
from torch.distributions.transforms import SoftplusTransform
from torch.distributions.utils import vec_to_tril_matrix, tril_matrix_to_vec


class FillScaleTriL(Transform):
    def __init__(self, diag_transform=None, diag_shift=1e-05):
        """
        Converts a tensor into a lower triangular matrix with positive diagonal entries.

        Args:
            diag_transform: transformation used on diagonal to ensure positive values.
            Default is SoftplusTransform
            diag_shift (float): small offset to avoid diagonals very close to zero.
            Default offset is 1e-05

        """
        super().__init__()
        self.diag_transform = (
            diag_transform if diag_transform is not None else SoftplusTransform()
        )
        self.diag_shift = diag_shift

        domain = constraints.real_vector
        codomain = constraints.lower_cholesky
        bijective = True

    def _call(self, x):
        """
        Transform input vector to lower triangular.

        Args:
            x (torch.Tensor): Input vector to transform
        Returns:
            torch.Tensor: Transformed lower triangular matrix
        """
        x = vec_to_tril_matrix(x)
        diagonal_elements = x.diagonal(dim1=-2, dim2=-1)
        transformed_diagonal = self.diag_transform(diagonal_elements)
        if self.diag_shift is not None:
            transformed_diagonal += self.diag_shift
        x.diagonal(dim1=-2, dim2=-1).copy_(transformed_diagonal)
        return x

    def _inverse(self, y):
        """
        Apply the inverse transformation to the input lower triangular matrix.

        Args:
            y (torch.Tensor): Invertible lower triangular matrix

        Returns:
            torch.Tensor: Inversely transformed vector

        """
        diagonal_elements = y.diagonal(dim1=-2, dim2=-1)
        if self.diag_shift is not None:
            transformed_diagonal = self.diag_transform.inv(
                diagonal_elements - self.diag_shift
            )
        else:
            transformed_diagonal = self.diag_transform.inv(diagonal_elements)
        y.diagonal(dim1=-2, dim2=-1).copy_(transformed_diagonal)
        return tril_matrix_to_vec(y)

    def log_abs_det_jacobian(self, x, y):
        """
        Computes the log absolute determinant of the Jacobian matrix for the transformation.

        Assumes that Softplus is used on the diagonal.
        The derivative of the softplus function is the sigmoid function.

        Args:
            x (torch.Tensor): Input vector before transformation
            y (torch.Tensor): Output lower triangular matrix from _call

        Returns:
            torch.Tensor: Log absolute determinant of the Jacobian matrix
        """
        diag_elements = y.diagonal(dim1=-2, dim2=-1)
        derivatives = torch.sigmoid(diag_elements)
        log_det_jacobian = torch.log(derivatives).sum()
        return log_det_jacobian
