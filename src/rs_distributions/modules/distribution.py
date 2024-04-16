import torch
from rs_distributions.modules import TransformedParameter
from rs_distributions import distributions as rsd
from inspect import signature
from functools import wraps


class DistributionModule(torch.nn.Module):
    """Base class for learnable distribution classes"""

    def __init__(self, distribution_class, *args, **kwargs):
        super().__init__()
        self.distribution_class = distribution_class
        sig = signature(distribution_class)
        bargs = sig.bind(*args, **kwargs)
        bargs.apply_defaults()
        for arg in distribution_class.arg_constraints:
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

    @classmethod
    def generate_subclass(cls, distribution_class):
        class DistributionModuleSubclass(cls):
            __doc__ = distribution_class.__doc__
            arg_constraints = distribution_class.arg_constraints

            @wraps(distribution_class.__init__)
            def __init__(self, *args, **kwargs):
                super().__init__(distribution_class, *args, **kwargs)

        return DistributionModuleSubclass

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


distributions_to_transform = DistributionModule._extract_distributions(
    torch.distributions,
    rsd,
)

# TODO: decide whether to use "ignore" or "include" pattern here
# Distributions which are currently not supported
ignore = (
    "Uniform",  # has weird "dependent" constraints
    "Binomial",  # has_rsample == False
    "MixtureSameFamily",  # has_rsample == False
)

for k in ignore:
    del distributions_to_transform[k]

__all__ = ["DistributionModule"]
for k, v in distributions_to_transform.items():
    globals()[k] = DistributionModule.generate_subclass(v)
    __all__.append(k)
