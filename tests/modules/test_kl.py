from rs_distributions import modules as rsm
import torch


def test_kl_divergence():
    q = rsm.Normal(0.0, 1.0)
    p = torch.distributions.Normal(0.0, 1.0)

    assert all([param.grad is None for param in q.parameters()])
    kl = rsm.kl_divergence(q, p)
    kl.backward()
    assert all([torch.isfinite(param.grad) for param in q.parameters()])
