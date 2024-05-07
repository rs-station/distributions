import torch
from torch.distributions import Transform, ComposeTransform, constraints
from torch.distributions.transforms import SoftplusTransform
from torch.distributions.utils import vec_to_tril_matrix, tril_matrix_to_vec


class FillTriL(Transform):
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
        return vec_to_tril_matrix(x)

    def _inverse(self, y):
        return tril_matrix_to_vec(y)

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)


class DiagTransform(Transform):
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
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        transformed_diagonal = self.diag_transform(diagonal)
        shifted_diag = transformed_diagonal
        result = x.diagonal_scatter(shifted_diag, dim1=-2, dim2=-1)

        return result

    def _inverse(self, y):
        diagonal = y.diagonal(dim1=-2, dim2=-1)
        result = y.diagonal_scatter(self.diag_transform.inv(diagonal), dim1=-2, dim2=-1)
        return result

    def log_abs_det_jacobian(self, x, y):
        diagonal = x.diagonal(dim1=-2, dim2=-1)
        return self.diag_transform.log_abs_det_jacobian(diagonal, y)


class FillScaleTriL(ComposeTransform):
    def __init__(self, diag_transform=SoftplusTransform()):
        super().__init__([FillTriL(), DiagTransform(diag_transform=diag_transform)])

    @property
    def bijective(self):
        return True
