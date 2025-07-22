#!/usr/bin/env python3

from dataclasses import dataclass
from util.config import ConfigBase
from typing import Tuple, Optional, Union

import numpy as np
import torch as th


def _filter_coefs(c: float):
    """Premultiplied butterworth filter coefs."""
    s = (1 / (1 + c * c + 1.414 * c))
    w = [1.0, 2.0, 1.0, -(c * c - 1.414 * c + 1), -(-2 * c * c + 2)]
    return np.multiply(w, s)


class PID(object):
    """Simple PID class.

    Supported features:

    * Max Windup clamping
    * Smooth derivative (2nd order butterworth)
    * Vectorized operation - (input doesn't need to be scalars)

    Reference:
    https://bitbucket.org/AndyZe/pid/src/master/
    """

    @dataclass
    class Config(ConfigBase):
        kp: float
        ki: float
        kd: float

        max_i: float  # windup
        max_u: float  # max effort

        cutoff_freq: float  # used for derivative filter coef

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # ~raw state variables...
        self.error_ = None
        self.error_i_ = None
        self.error_d_ = None

        # filtered (smooth) state.
        self.f_error_ = None
        self.f_error_d_ = None

    @property
    def kp(self):
        return self.cfg.kp

    @property
    def ki(self):
        return self.cfg.ki

    @property
    def kd(self):
        return self.cfg.kd

    def set_gains(self, kp: float, ki: float, kd: float):
        self.cfg.kp = kp
        self.cfg.ki = ki
        self.cfg.kd = kd

    def set_max_i(self, max_i: float):
        self.cfg.max_i = max_i

    def reset(self, soft=True, mask: Optional[th.Tensor] = None):
        if soft:
            if self.error_ is not None:
                if mask is not None:
                    self.error_.masked_fill_(mask[None], 0.0,)
                    self.error_i_.masked_fill_(mask, 0.0)
                    self.error_d_.masked_fill_(mask[None], 0.0)
                    self.f_error_.masked_fill_(mask[None], 0.0)
                    self.f_error_d_.masked_fill_(mask[None], 0.0)
                else:
                    self.error_.fill_(0.0)
                    self.error_i_.fill_(0.0)
                    self.error_d_.fill_(0.0)
                    self.f_error_.fill_(0.0)
                    self.f_error_d_.fill_(0.0)
        else:
            self.error_ = None
            self.error_i_ = None
            self.error_d_ = None
            self.f_error_ = None
            self.f_error_d_ = None

    def _allocate(self, shape: Tuple[int, ...],
                  device: Union[str, th.device, None] = None):
        # NOTE: `3` here is just the buffer length for
        # maintaining a smooth derivative.
        shape = tuple(shape)
        self.error_ = th.zeros((3,) + shape, dtype=th.float32,
                               device=device)  # 3xN
        self.error_i_ = th.zeros(shape, dtype=th.float32,
                                 device=device)  # N
        self.error_d_ = th.zeros_like(self.error_,
                                      device=device)  # 3xN
        # Filtered ...
        self.f_error_ = th.zeros_like(self.error_,
                                      device=device)  # 3xN
        self.f_error_d_ = th.zeros_like(self.error_d_,
                                        device=device)  # 3xN

    def __call__(self, err: th.Tensor, dt: float) -> th.Tensor:
        if isinstance(dt, float):
            dt = th.as_tensor(dt, dtype=th.float32)

        # If this is the first invocation since reset,
        # Configure the controller buffers.
        if self.error_ is None:
            self._allocate(err.shape, err.device)

        # Set the current error.
        self.error_ = th.roll(self.error_, -1, dims=0)
        self.error_[-1] = err

        # Apply numerical integration and clip the results.
        self.error_i_ += self.error_[-1] * dt
        self.error_i_ = th.clip(
            self.error_i_, -self.cfg.max_i, self.cfg.max_i,
            out=self.error_i_)

        # Apply (smooth) numerical differentiation.
        t = th.tan((self.cfg.cutoff_freq * 2 * np.pi) * 0.5 * dt)
        # FIXME: Remove hardcoded epsilon (0.01),
        # Or reparametrize filter coefficients to be numerically stable at or
        # near 0 (if applicable).
        t = th.where(th.abs(t) <= 0.01, 0.01 * th.sign(t), t)

        c = 1.0 / t
        k = _filter_coefs(c)
        self.f_error_ = th.roll(self.f_error_, -1, dims=0)

        q = th.cat([self.error_,
                    self.f_error_[:2]],
                   dim=0)
        k = th.as_tensor(k,
                         dtype=self.error_.dtype,
                         device=self.error_.device,
                         )
        self.f_error_[-1] = th.einsum('i,i...', k,
                                      th.cat([self.error_,
                                              self.f_error_[:2]],
                                             dim=0))

        self.error_d_ = th.roll(self.error_d_, -1, dims=0)
        self.error_d_[-1] = (1.0 / dt) * (self.error_[2] - self.error_[1])

        self.f_error_d_ = th.roll(self.f_error_d_, -1, dims=0)
        self.f_error_d_[-1] = th.einsum('i,i...',
                                        k,
                                        th.cat([self.error_d_,
                                                self.f_error_d_[:2]],
                                               dim=0))

        # Collect contributions.
        u_p = self.kp * self.error_[-1]
        u_i = self.ki * self.error_i_
        u_d = self.kd * self.f_error_d_[-1]

        u = th.zeros_like(u_p)
        if self.kp > 0:
            u += u_p
        if self.ki > 0:
            u += u_i
        if abs(self.kd) > 0:
            u += u_d

        # Clip output, and return.
        u = th.clip(u, -self.cfg.max_u, self.cfg.max_u, out=u)
        return u


def main():

    pid = PID(PID.Config(
        1.0, 0.1, 0.01,
        1.0, 1.0, 8.0))
    dt: float = 0.001
    x = th.randn((8, 3))
    target = x * 0 + 0.2
    for _ in range(8):
        err = target - x
        u = pid(err, dt)
        x += u
    print(x)


if __name__ == '__main__':
    main()
