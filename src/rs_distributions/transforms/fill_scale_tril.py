import torch
from torch.distributions import Transform, ComposeTransform, constraints
from torch.distributions.transforms import SoftplusTransform
from torch.distributions.utils import vec_to_tril_matrix, tril_matrix_to_vec


class FillTriL(Transform):
    """
    Transform for converting a real-valued vector into a lower triangular matrix
    """

    def __init__(self):
        super().__init__()

    @property
    def domain(self):
        return constraints.real_vector

    @property
    def codomain(self):
        return constraints.lower_triangular

    @property
    def bijective(self):
        return True

    def _call(self, x):
        """
        Converts real-valued vector to lower triangular matrix.

        Args:
            x (torch.Tensor): input real-valued vector
        Returns:
            torch.Tensor: Lower triangular matrix
        """

        return vec_to_tril_matrix(x)

    def _inverse(self, y):
        return tril_matrix_to_vec(y)

    def log_abs_det_jacobian(self, x, y):
        batch_shape = x.shape[:-1]
        return torch.zeros(batch_shape, dtype=x.dtype, device=x.device)


class DiagTransform(Transform):
    """
    Applies transformation to the diagonal of a square matrix
    """

    def __init__(self, diag_transform):
        super().__init__()
        self.diag_transform = diag_transform

    @property
    def domain(self):
        return self.diag_transform.domain

    @property
    def codomain(self):
        return self.diag_transform.codomain

    @property
    def bijective(self):
        return self.diag_transform.bijective

    def _call(self, x):
        """
        Args:
            x (torch.Tensor): Input matrix
        Returns
            torch.Tensor: Transformed matrix
        """
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        transformed_diagonal = self.diag_transform(diagonal)
        result = x.diagonal_scatter(transformed_diagonal, dim1=-2, dim2=-1)

        return result

    def _inverse(self, y):
        diagonal = y.diagonal(dim1=-2, dim2=-1)
        result = y.diagonal_scatter(self.diag_transform.inv(diagonal), dim1=-2, dim2=-1)
        return result

    def log_abs_det_jacobian(self, x, y):
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        return self.diag_transform.log_abs_det_jacobian(diagonal, y)


class FillScaleTriL(ComposeTransform):
    """
    A `ComposeTransform` that reshapes a real-valued vector into a lower triangular matrix.
    The diagonal of the matrix is transformed with `diag_transform`.
    """

    def __init__(self, diag_transform=SoftplusTransform()):
        super().__init__([FillTriL(), DiagTransform(diag_transform=diag_transform)])
        self.diag_transform = diag_transform

    @property
    def bijective(self):
        return True

    def log_abs_det_jacobian(self, x, y):
        x = FillTriL()._call(x)
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        return self.diag_transform.log_abs_det_jacobian(diagonal, diagonal)
