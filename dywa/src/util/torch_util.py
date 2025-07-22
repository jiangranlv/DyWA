#!/usr/bin/env python3

from typing import (Mapping, Callable, Iterable, Union,
                    Tuple, List)
from contextlib import contextmanager

import os
import random
import time
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad


def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        th.backends.cudnn.benchmark = False
        th.backends.cudnn.deterministic = True
        th.use_deterministic_algorithms(True)
    else:
        th.backends.cudnn.benchmark = True
        th.backends.cudnn.deterministic = False

    return seed


def merge_shapes(*dims) -> Tuple[int, ...]:
    """
    Concatenate multiple dims into one.
    Args:
        dims: either an int or a list of ints.
    Returns:
        concatenated dims.
    """
    out: List[int] = []
    for d in dims:
        if isinstance(d, Iterable):
            out.extend(d)
        else:
            out.append(d)
    return tuple(out)


def dot(x: th.Tensor, y: th.Tensor) -> th.Tensor:
    return th.einsum('...i,...i->...', x, y)


@contextmanager
def module_requires_grad(model: nn.Module, enable: bool):
    old = []
    try:
        for p in model.parameters():
            old.append(p.requires_grad)
            p.requires_grad = enable
        yield
    finally:
        for o, p in zip(old, model.parameters()):
            p.requires_grad = o


@contextmanager
def module_train_mode(model: nn.Module, enable: bool):
    old: bool = model.training
    try:
        model.train(enable)
        yield
    finally:
        model.train(old)


def gradient(x: th.Tensor, y: th.Tensor, **kwds) -> th.Tensor:
    """ dy/dx """
    return grad(y, [x],
                grad_outputs=th.ones_like(y,
                requires_grad=False),
                retain_graph=True,
                **kwds)[0]


def dcn(x: th.Tensor) -> np.ndarray:
    """
    Convert torch tensor into numpy array.
    """
    if isinstance(x, th.Tensor):
        return x.detach().cpu().numpy()
    return x


def dict_to(x: Union[th.Tensor, Mapping],
            device: Union[str, th.device] = 'cuda'):
    """
    Send a structured dict of torch tensors to target device.
    """
    device = th.device(device)
    if isinstance(x, Mapping):
        return {k: dict_to(x, device) for k, v in x.items()}
    else:
        if isinstance(x, th.Tensor):
            return x.to(device)
        else:
            return x


@th.jit.script
def masked_mean(x: th.Tensor, m: th.Tensor) -> th.Tensor:
    m = m.expand(x.shape)
    # assert (m.shape == x.shape)

    # I guess count_nonzero is non-differentiable
    # denom = th.count_nonzero(m)
    mf = m.float()
    denom = mf.sum().clamp_min(1.0)

    # if denom <= 0:
    #     return 0.0
    # This only works because of m.float()
    numer = th.sum(x * mf)
    # Would this maybe prevent CPU<->GPU data transfer?
    # if denom <= 0:
    #     return 0.0
    # else:
    #     return numer / denom
    return numer / denom
    # return th.where(denom <= 0,
    #                 th.zeros_like(numer),
    #                 numer / denom)


@th.jit.script
def masked_var_mean(
        values: th.Tensor,
        masks: th.Tensor,
        dim: int = 0) -> Tuple[th.Tensor, th.Tensor]:
    # WARN: dim default being 0 is very dangerous...
    count = masks.sum(dim=dim)
    masked_values = values * masks
    values_mean = masked_values.sum(dim=dim) / count
    min_sqr = (
        (th.square(masked_values) / count).sum(dim=dim)
        - th.square((masked_values / count).sum(dim=dim))
    )
    values_var = min_sqr * count / (count - 1)
    return values_var, values_mean


@th.jit.script
def masked_sample(value: th.Tensor,
                  mask: th.Tensor,
                  num_samples: int,
                  eps: float = 1e-6):
    prob = mask.float().add_(eps)
    prob = prob.div_(prob.sum(keepdim=True, dim=-1))
    indices = th.multinomial(prob,
                             num_samples=num_samples,
                             replacement=True)
    return th.take_along_dim(value,
                             indices[..., None],
                             dim=-2)


class MaskedLoss(nn.Module):
    def __init__(self, loss_fn: Callable):
        self.loss_fn = loss_fn
        # `reduction` better be `none`

    def forward(self, x: th.Tensor, y: th.Tensor, m: th.Tensor):
        return masked_mean(self.loss_fn(x, y), m)


def test_masked_var_mean_no_mask():
    x = th.randn(size=(128, 3), dtype=th.float32,
                 device='cuda:0')
    x.requires_grad_(True)
    m = (x <= 0.0) * 0
    mm = masked_var_mean(x, m)
    print(mm)


def time_masked_var_mean():
    x = th.randn(size=(128, 3), dtype=th.float32,
                 device='cuda:0')
    x.requires_grad_(True)
    m = (x <= 0.0)
    for _ in range(4):
        mm = masked_mean(x, m)
    t0 = time.time()
    for _ in range(128):
        mm = masked_mean(x, m)
    t1 = time.time()
    print(t1 - t0)
    print(mm)
    g = th.autograd.grad(mm, x)
    # print(g)
    # numpy version
    # print(np.sum(x.detach().cpu().numpy() * m.detach().cpu().numpy())
    #       / np.sum(m.detach().cpu().numpy()))


def main():
    test_masked_var_mean_no_mask()


if __name__ == '__main__':
    main()
