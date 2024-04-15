import torch
from rs_distributions.modules import TransformedParameter
from rs_distributions import distributions as rsd
from inspect import signature, isclass
from functools import wraps


def distribution_module_factory(distribution_class):
    class DistributionModule(torch.nn.Module):
        __doc__ = distribution_class.__doc__

        @wraps(distribution_class.__init__)
        def __init__(self, *args, **kwargs):
            super().__init__()
            sig = signature(distribution_class)
            bargs = sig.bind(*args, **kwargs)
            bargs.apply_defaults()
            for arg in distribution_class.arg_constraints:
                param = bargs.arguments.pop(arg)
                param = self._constrain_arg_if_needed(arg, param)
                setattr(self, f"_{arg}", param)
            self._extra_args = bargs.arguments

        def __repr__(self):
            rstring = super().__repr__().split("\n")[1:]
            rstring = [str(distribution_class) + " DistributionModule("] + rstring
            return "\n".join(rstring)

        def _distribution(self):
            kwargs = {
                k: self._realize_parameter(getattr(self, f"_{k}"))
                for k in distribution_class.arg_constraints
            }
            kwargs.update(self._extra_args)
            return distribution_class(**kwargs)

        @staticmethod
        def _constrain_arg_if_needed(name, value):
            if isinstance(value, TransformedParameter):
                return value
            cons = distribution_class.arg_constraints[name]
            transform = torch.distributions.constraint_registry.transform_to(cons)
            return TransformedParameter(value, transform)

        @staticmethod
        def _realize_parameter(param):
            if isinstance(param, TransformedParameter):
                return param()
            return param

        def __getattr__(self, name: str):
            if name in distribution_class.arg_constraints or hasattr(
                distribution_class, name
            ):
                q = self._distribution()
                return getattr(q, name)
            return super().__getattr__(name)

    return DistributionModule


distributions_to_transform = {}


def extract_distributions(module):
    """extract distributions from a module into a dict {name: cls}"""
    d = {}
    for k in module.__all__:
        cls = getattr(module, k)
        if not isclass(cls):
            continue
        if issubclass(cls, torch.distributions.Distribution):
            d[k] = cls
    return d


distributions_to_transform = extract_distributions(torch.distributions)
distributions_to_transform.update(extract_distributions(rsd))

__all__ = []
for k, v in distributions_to_transform.items():
    globals()[k] = distribution_module_factory(v)
    __all__.append(k)
