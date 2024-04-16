# rs-distributions

[![Documentation](https://github.com/rs-station/distributions/workflows/Documentation/badge.svg)](https://rs-station.github.io/distributions)
[![Build](https://github.com/rs-station/distributions/actions/workflows/test.yml/badge.svg)](https://github.com/rs-station/distributions/actions/workflows/test.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/rs-distributions.svg)](https://pypi.org/project/rs-distributions)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rs-distributions.svg)](https://pypi.org/project/rs-distributions)

-----

**Table of Contents**

- [Installation](#installation)
- [Distributions](#distributions)
- [Modules](#modules)
- [License](#license)

rs-distributions provides statistical tools which are helpful for structural biologists who wish to model their data using variational inference. 

## Installation

```console
pip install rs-distributions
```

## Distributions
`rs_distributions.distributions` provides learnable distributions that are important in structural biology. 
These distributions follow the conventions in `torch.dist`. 
Here's a small example of distribution matching between a learnable distribution, `q`, and a target distribion, `p`. 
The example works by minimizing the Kullback-Leibler divergence between `q` and `p` using gradients calculated by the [implicit reparameterization method](https://arxiv.org/abs/1805.08498). 

```python
import torch
from rs_distributions import distributions as rsd

target_loc = 4.
target_scale = 2.

loc_initial_guess = 10.
scale_initial_guess  = 3.

loc = torch.tensor(loc_initial_guess, requires_grad=True)

scale_transform = torch.distributions.transform_to(
    rsd.FoldedNormal.arg_constraints['scale']
)
scale_initial_guess = scale_transform.inv(
    torch.tensor(scale_initial_guess)
)
unconstrained_scale = torch.tensor(
    torch.tensor(scale_initial_guess),
    requires_grad=True
)

p = rsd.FoldedNormal(
    target_loc,
    target_scale,
)

opt = torch.optim.Adam([loc, unconstrained_scale])

steps = 10_000
num_samples = 100
for i in range(steps):
    opt.zero_grad()
    scale = scale_transform(unconstrained_scale)
    q = rsd.FoldedNormal(loc, scale)
    z = q.sample((num_samples,))
    kl_div = q.log_prob(z) - p.log_prob(z)
    kl_div = kl_div.mean()
    kl_div.backward()
    opt.step()
```
This example uses the folded normal distribution which is important in X-ray crystallography. 

## Modules
Working with PyTorch distributions can be a little verbose. 
So in addition to the `torch.distributions` style implementation, we provide `DistributionModule` classes which enable learnable distributions with automatic bijections in less code. 
These `DistributionModule` classes are subclasses of `torch.nn.Module`. 
They automatically instantiate problem parameters as `TransformedParameter` modules following the constraints in the distribution definition.
In the following example, a `FoldedNormal` `DistributionModule` is instantiated with an initial location and scale and trained to match a target distribution. 

```python
from rs_distributions import modules as rsm
import torch

loc_init = 10.
scale_init = 5.

q = rsm.FoldedNormal(loc_init, scale_init)
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

## License

`rs-distributions` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
