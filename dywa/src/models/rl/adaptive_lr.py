#!/usr/bin/env python3

from typing import Tuple
import torch as th


def update_lr_scale(
    prev_scale: float,
    kl_div: float,
    kl_target: float = 0.008,
    kl_bounds: Tuple[float, float] = (0.5, 2.0),
    factor: float = 1.5,
    scale_bounds: Tuple[float, float] = (1e-6, 1e-2)
) -> float:
    """ Update the learning rate by monitoring kl divergence, apparently.

    kl_div:
        KL divergence between old/new policies.
        Assumed to be of form KL(P_old || P_new)

    """
    scale = prev_scale
    if kl_div < (kl_bounds[0] * kl_target):
        # increase learning rate
        scale = min(prev_scale * factor, scale_bounds[1])
    if kl_div > (kl_bounds[1] * kl_target):
        # decrease learning rate
        scale = max(prev_scale / factor, scale_bounds[0])
    return scale


class AdaptiveScheduler:
    def __init__(self,
                 optimizer: th.optim.Optimizer,
                 init_lr: float = 3e-4,
                 kl_target: float = 0.008):
        super().__init__()
        self.optimizer = optimizer
        self.scale: float = 1.0
        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.get_scale)
        self.init_lr = init_lr
        self.min_scale: float = 1e-6 / init_lr
        self.max_scale: float = 1e-2 / init_lr
        self.kl_target = kl_target

    def update(self, kl_dist: float) -> Tuple[float, float]:
        scale = self.scale
        if kl_dist > (2.0 * self.kl_target):
            scale = max(scale / 1.5, self.min_scale)
        if kl_dist < (0.5 * self.kl_target):
            scale = min(scale * 1.5, self.max_scale)
        self.scale = scale
        return self.scale, self.init_lr * self.scale

    def get_scale(self) -> float:
        return self.scale

    def step(self, *args, **kwds):
        return self.scheduler.step(*args, **kwds)
