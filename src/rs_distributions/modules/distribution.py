import torch
from rs_distributions.modules import TransformedParameter
from rs_distributions import distributions as rsd
from inspect import signature
from functools import update_wrapper


# TODO: decide whether to use "ignore" or "include" pattern here
# Distributions which are currently not supported
ignored_distributions = (
    "Uniform",  # has weird "dependent" constraints
    "Binomial",  # has_rsample == False
    "MixtureSameFamily",  # has_rsample == False
)


class DistributionModule(torch.nn.Module):
    """
    Base class for constructing learnable distributions.
    This subclass of `torch.nn.Module` acts like a `torch.distributions.Distribution`
    object with learnable `torch.nn.Parameter` attributes.
    It works by lazily constructing distributions as needed.
    Here is a simple example of distribution matching using learnable distributions with reparameterized gradients.

    ```python
    from rs_distributions import modules as rsm
    import torch

    q = rsm.FoldedNormal(10., 5.)
    p = torch.distributions.HalfNormal(1.)

    opt = torch.optim.Adam(q.parameters())

    steps = 10_000
    num_samples = 256
    for i in range(steps):
        opt.zero_grad()
        z = q.rsample((num_samples,))
        kl = (q.log_prob(z) - p.log_prob(z)).mean()
        kl.backward()
        opt.step()
    ```
    """

    distribution_class = torch.distributions.Distribution
    __doc__ = distribution_class.__doc__
    arg_constraints = distribution_class.arg_constraints

    def __init__(self, *args, **kwargs):
        super().__init__()
        sig = signature(self.distribution_class)
        bargs = sig.bind(*args, **kwargs)
        bargs.apply_defaults()
        for arg in self.distribution_class.arg_constraints:
            param = bargs.arguments.pop(arg)
            param = self._constrain_arg_if_needed(arg, param)
            setattr(self, f"_transformed_{arg}", param)
        self._extra_args = bargs.arguments

    def __repr__(self):
        rstring = super().__repr__().split("\n")[1:]
        rstring = [str(self.distribution_class) + " DistributionModule("] + rstring
        return "\n".join(rstring)

    def _distribution(self):
        kwargs = {
            k: self._realize_parameter(getattr(self, f"_transformed_{k}"))
            for k in self.distribution_class.arg_constraints
        }
        kwargs.update(self._extra_args)
        return self.distribution_class(**kwargs)

    def _constrain_arg_if_needed(self, name, value):
        if isinstance(value, TransformedParameter):
            return value
        cons = self.distribution_class.arg_constraints[name]
        if cons == torch.distributions.constraints.dependent:
            transform = torch.distributions.AffineTransform(0.0, 1.0)
        else:
            transform = torch.distributions.constraint_registry.transform_to(cons)
        return TransformedParameter(value, transform)

    @staticmethod
    def _realize_parameter(param):
        if isinstance(param, TransformedParameter):
            return param()
        return param

    def __getattr__(self, name: str):
        if name in self.distribution_class.arg_constraints or hasattr(
            self.distribution_class, name
        ):
            q = self._distribution()
            return getattr(q, name)
        return super().__getattr__(name)

    @staticmethod
    def _extract_distributions(*modules, base_class=torch.distributions.Distribution):
        """
        extract all torch.distributions.Distribution subclasses from a module(s)
        into a dict {name: cls}
        """
        d = {}
        for module in modules:
            for k in module.__all__:
                distribution_class = getattr(module, k)
                if not hasattr(distribution_class, "arg_constraints"):
                    continue
                if not hasattr(distribution_class.arg_constraints, "items"):
                    continue
                if issubclass(distribution_class, base_class):
                    d[k] = distribution_class
        return d

    def __init_subclass__(cls, /, distribution_class, **kwargs):
        super().__init_subclass__(**kwargs)
        update_wrapper(
            cls.__init__,
            distribution_class.__init__,
        )
        cls.distribution_class = distribution_class
        cls.arg_constraints = distribution_class.arg_constraints
        cls.__doc__ = distribution_class.__doc__


distributions_to_transform = DistributionModule._extract_distributions(
    torch.distributions,
    rsd,
)

for k in ignored_distributions:
    del distributions_to_transform[k]

__all__ = ["DistributionModule"]
for k, v in distributions_to_transform.items():
    # This pattern is required for pickling to work properly
    globals()[k] = type(k, (DistributionModule,), {}, distribution_class=v)
    __all__.append(k)
