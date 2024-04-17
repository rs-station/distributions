import pytest
from rs_distributions import modules as rsm
import torch


distribution_classes = rsm.DistributionModule._extract_distributions(
    rsm, base_class=rsm.DistributionModule
)

# It is common to have arguments which are equivalent and mutually exclusive
# in distribution classes.
exlusive_args = [
    ("logits", "probs"),
    ("covariance_matrix", "precision_matrix", "scale_tril"),
]

# Workarounds for distributions with additional, non-parameter arguments
special_kwargs = {
    "RelaxedBernoulli": {"temperature": torch.ones(())},
    "RelaxedOneHotCategorical": {"temperature": torch.ones(())},
    "TransformedDistribution": {
        "base_distribution": rsm.Normal(0.0, 1.0),
        "transforms": torch.distributions.AffineTransform(0.0, 1.0),
    },
    "LKJCholesky": {"dim": 3},
    "Independent": {
        "base_distribution": rsm.Normal(torch.zeros(3), torch.ones(3)),
        "reinterpreted_batch_ndims": 1,
    },
}


@pytest.mark.parametrize("distribution_class_name", distribution_classes.keys())
@pytest.mark.parametrize("serialize", [False, True])
def test_distribution_module(distribution_class_name, serialize):
    distribution_class = distribution_classes[distribution_class_name]

    assert distribution_class.__doc__ == distribution_class.distribution_class.__doc__
    assert (
        distribution_class.arg_constraints
        == distribution_class.distribution_class.arg_constraints
    )

    shape = (3, 3)
    kwargs = {}
    cons = distribution_class.arg_constraints
    for group in exlusive_args:
        matches_group = all([g in cons for g in group])
        if matches_group:
            for con in group[1:]:
                del cons[con]
    for k, v in cons.items():
        try:
            t = torch.distributions.constraint_registry.transform_to(v)
            kwargs[k] = t(torch.ones(shape))
        except NotImplementedError:
            t = torch.distributions.AffineTransform(0.0, 1.0)
            kwargs[k] = rsm.TransformedParameter(v, t)

    if distribution_class_name in special_kwargs:
        kwargs.update(special_kwargs[distribution_class_name])
    q = distribution_class(**kwargs)

    if serialize:
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(delete=False) as f:
            torch.save(q, f)
            q = torch.load(f.name)

    # Not all distributions have these attributes implemented
    try:
        q.mean
        q.variance
        q.stddev
    except NotImplementedError:
        pass

    if q.has_rsample:
        z = q.rsample()
    else:
        z = q.sample()

    ll = q.log_prob(z)

    params = list(q.parameters())
    if q.has_rsample:
        loss = -ll.sum()
        loss.backward()
        for x in params:
            assert torch.isfinite(x.grad).all()
