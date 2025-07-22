#!/usr/bin/env python3

from typing import Tuple, Optional, Dict, Callable, Union, List
import math
from numbers import Number

import torch as th
from torch import distributions as pyd
from torch.distributions import (Normal, Independent, Categorical, TransformedDistribution)
from torch.distributions.transforms import TanhTransform

import nvtx
from icecream import ic

T = th.Tensor


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,)
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=-1)
    else:
        tensor = tensor.sum()
    return tensor


class StableNormal(Normal):
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        log_scale = math.log(
            self.scale) if isinstance(
            self.scale,
            Number) else self.scale.log()
        return - 0.5 * ((value - self.loc) / self.scale)**2 - \
            log_scale - math.log(math.sqrt(2 * math.pi))


class SDENormal:
    def __init__(self,
                 action_dim: int,
                 latent_sde_dim: int,
                 epsilon: float = 1e-6,
                 ):
        self.action_dim = action_dim
        self.latent_sde_dim = latent_sde_dim
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self._latent_sde = None
        self.epsilon = epsilon

    def sample_weights(self, log_std: th.Tensor,
                       batch_size: List[int] = 1) -> None:
        """
        Sample weights for the noise exploration matrix,
        using a centered Gaussian distribution.

        :param log_std:
        :param batch_size:
        """
        if isinstance(batch_size, int):
            batch_size = [batch_size]
        std = th.exp(log_std)
        self.weights_dist = Normal(th.zeros_like(std), std)
        # Pre-compute matrices in case of parallel exploration
        self.exploration_matrices = self.weights_dist.rsample((*batch_size,))

    def log_prob(self, actions: th.Tensor) -> th.Tensor:

        gaussian_actions = actions
        # log likelihood for a gaussian
        log_prob = self.distribution.log_prob(gaussian_actions)
        # Sum along action dim
        log_prob = sum_independent_dims(log_prob)
        return log_prob

    def get_noise(self, latent_sde: th.Tensor) -> th.Tensor:
        latent_sde = latent_sde.detach()
        # Use batch matrix multiplication for efficient computation
        # (batch_size, n_features) -> (batch_size, 1, n_features)
        latent_sde = latent_sde.unsqueeze(dim=1)
        # (batch_size, 1, n_actions)
        noise = th.bmm(latent_sde, self.exploration_matrices)
        return noise.squeeze(dim=1)

    def proba_distribution(
            self, mean_actions: th.Tensor, log_std: th.Tensor,
            latent_sde: th.Tensor):
        self._latent_sde = latent_sde.detach()
        std = th.exp(log_std)
        variance = th.einsum('...ij,jk->...ik', self._latent_sde**2, std ** 2)
        self.distribution = Normal(
            mean_actions, th.sqrt(
                variance + self.epsilon))
        return self

    def entropy(self) -> th.Tensor:
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        noise = self.get_noise(self._latent_sde)
        actions = self.distribution.mean + noise
        return actions


class StableTanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1, threshold=20):
        super().__init__(cache_size=cache_size)
        self.softplus = th.nn.Softplus(threshold=threshold)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2. * (math.log(2.) - x - self.softplus(-2. * x))


class StableSquashedNormal(
        pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale, threshold: float = 20, stable=False):
        self.loc = loc
        self.scale = scale

        if stable:
            self.base_dist = StableNormal(loc, scale)
        else:
            self.base_dist = pyd.Normal(loc, scale)

        transforms = [StableTanhTransform(threshold=threshold)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def tanh_transform(
        d: th.distributions.Distribution) -> th.distributions.Distribution:
    # NOTE:
    # While I've had some random issues with `cache_size`,
    # maybe it's worth reviving this option.
    # It's certainly enabled in RL_GAMES...
    # Let's see.
    return TransformedDistribution(d, TanhTransform(cache_size=1))


class SquashedNormal(TransformedDistribution):
    def __init__(self, loc: th.Tensor, scale: th.Tensor,
                 validate_args: bool = True):
        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale, validate_args)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self) -> th.Tensor:
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()


@nvtx.annotate("get_action_distribution()", color="green")
def get_action_distribution(obs: th.Tensor,
                            policy_fn: Callable[[th.Tensor], Union[T, Tuple[T, T]]],
                            discrete: bool,
                            tanh_xfm: bool,
                            aux: Optional[Dict[str, th.Tensor]] = None,
                            stable_normal: bool = True
                            ) -> th.distributions.Distribution:

    if discrete:
        logits = policy_fn(obs, aux=aux)
        dist = Categorical(logits=logits)
        if aux is not None:
            aux['logits'] = logits
    else:
        mu, ls = policy_fn(obs, aux=aux)
        if stable_normal:
            if tanh_xfm:
                dist = StableSquashedNormal(mu, ls.exp(),
                                            stable=True)
            else:
                dist = StableNormal(mu, ls.exp(), validate_args=False)
        else:
            dist = Normal(mu, ls.exp(), validate_args=False)
            if tanh_xfm:
                dist = tanh_transform(dist)
        dist = Independent(dist, 1)
        if aux is not None:
            aux['mu'] = mu
            aux['ls'] = ls
    return dist


@nvtx.annotate("get_action()", color="green")
def get_action(obs: th.Tensor,
               policy_fn: Callable[[th.Tensor], Union[T, Tuple[T, T]]],
               aux: Optional[Dict[str, th.Tensor]] = None,
               sample: bool = True,
               clip_action: Optional[Tuple[th.Tensor, th.Tensor]] = None,
               **kwds):
    discrete = kwds.get('discrete')
    tanh_xfm = kwds.get('tanh_xfm')
    stable = kwds.get('stable', True)

    # [2] get action distribution.
    dist = get_action_distribution(
        obs, policy_fn, aux=aux, **kwds)

    if sample:
        action = dist.sample()
    else:
        if discrete:
            # WARN: `dist.logits` gets overwritten internally
            # compared to the logits passed into e.g. Categorical(...)
            # (but the argmax is still consistent)
            action = dist.logits.argmax(dim=-1)
        else:
            if tanh_xfm and (not stable):
                # tanh-squashed gaussian
                action = th.tanh(aux['mu'])
            else:
                # gaussian
                action = dist.mean

    # [4] Optionally apply clipping.
    if clip_action is not None:
        clipped_action = action.clip(*clip_action)
    else:
        clipped_action = action
    return clipped_action
