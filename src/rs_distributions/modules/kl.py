from functools import wraps
from rs_distributions import modules as rsm
import torch


@wraps(torch.distributions.kl.kl_divergence)
def kl_divergence(p, q):
    if isinstance(p, rsm.DistributionModule):
        p = p._distribution()
    if isinstance(q, rsm.DistributionModule):
        q = q._distribution()
    return torch.distributions.kl.kl_divergence(p, q)
