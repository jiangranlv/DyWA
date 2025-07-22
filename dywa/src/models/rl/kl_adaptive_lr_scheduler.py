#!/usr/bin/env python3

from dataclasses import dataclass

import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR

from models.rl.ppo_config import AdaptiveLearningRateConfig
from models.rl.adaptive_lr import update_lr_scale


class KLAdaptiveLRScheduler(LambdaLR):

    @dataclass
    class Config(AdaptiveLearningRateConfig):
        pass

    def __init__(self, cfg: Config,
                 optimizer: th.optim.Optimizer,
                 last_epoch: int = -1,
                 verbose: bool = False):
        self.cfg = cfg
        self.scale: float = cfg.initial_scale
        super().__init__(optimizer, self.get_lr_scale, last_epoch, verbose)

    def get_lr_scale(self, *args, **kwds):
        return self.scale

    def update_kl(self, kl: float):
        self.scale = update_lr_scale(self.scale, kl,
                                     self.cfg.kl_target,
                                     self.cfg.kl_bounds,
                                     self.cfg.factor,
                                     self.cfg.scale_bounds)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {
            key: value for key,
            value in self.__dict__.items() if key not in (
                'optimizer',
                'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.")

        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def main():
    net = nn.Linear(6, 8)
    optimizer = th.optim.Adam(net.parameters(), 3e-4)
    scheduler = KLAdaptiveLRScheduler(
        KLAdaptiveLRScheduler.Config(),
        optimizer)
    print(net.weight.mean())

    optimizer.zero_grad(set_to_none=True)
    net(th.randn((8, 6))).sum().backward()
    optimizer.step()
    scheduler.step()
    print(net.weight.mean())
    print(scheduler.scale)
    scheduler.update_kl(0.005)
    print(scheduler.scale)
    scheduler.update_kl(0.0005)
    print(scheduler.scale)
    scheduler.update_kl(0.01)
    print(scheduler.scale)
    scheduler.update_kl(0.021)
    print(scheduler.scale)  # dec
    scheduler.update_kl(0.05)
    print(scheduler.scale)  # dec


if __name__ == '__main__':
    main()
