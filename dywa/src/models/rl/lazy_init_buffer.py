#!/usr/bin/env python3

from typing import Tuple, List, Optional
import torch as th

T = th.Tensor


class LazyInitBuffer:
    """
    Stores data layout as:
        T X N X [...]
    where
        T = number of timesteps,
        N = number of (parallel) environments,
            where the env at each index is
            expected to be consistent.
    """
    KEYS: Tuple[str, ...] = ('obs0', 'actn', 'rewd',
                             'obs1', 'logp', 'end0', 'end1')

    def __init__(self, size: int,
                 device: th.device):
        self.size = size
        self.pos = 0
        self.full = False

        self.device = device

        self.obs0 = None
        self.actn = None
        self.rewd = None
        self.obs1 = None
        self.logp = None
        self.end0 = None
        self.end1 = None

    def __len__(self):
        return (self.size if self.full else self.pos)

    def allocate(self, buf: Tuple[T, T, T, T, T, T, T]):
        (obs0, actn, rewd,
         obs1, logp,
         end0, end1) = buf

        # Slight hack I guess.
        for s in self.KEYS:
            src = locals().get(s)
            setattr(self, s,
                    th.empty((self.size,) + src.shape,
                             dtype=src.dtype,
                             device=self.device))

    def append(self, buf: Tuple[T, T, T, T, T, T, T]):
        (obs0, actn, rewd,
         obs1, logp,
         end0, end1) = buf

        if self.obs0 is None:
            self.allocate(buf)
        for s in self.KEYS:
            getattr(self, s).__setitem__(self.pos,
                                         locals().get(s))

        pos = (self.pos + 1)
        if pos >= self.size:
            self.full = True
            pos = pos % self.size
        self.pos = pos

    def clear(self):
        """ reset the flags but preserve the data. """
        self.pos = 0
        self.full = False

    def as_list(self, horizon: Optional[int] = None) -> List[T]:
        # Extract and reshape.
        D = []
        for s in self.KEYS:
            d = getattr(self, s)
            if horizon is not None:
                d = d.view(-1, horizon,
                           *d.shape[1:])
            D.append(d)
        return D

    def sample(self, batch_size: int) -> List[T]:
        if self.full:
            indices = th.randint(self.size, size=(batch_size,))
        else:
            indices = th.randint(self.pos, size=(batch_size,))

        D = []
        for s in self.KEYS:
            d = getattr(self, s)
            d = d[indices]
            D.append(d)
        return D
