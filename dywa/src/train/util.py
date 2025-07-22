#!/usr/bin/env python3

from typing import Iterable
import logging
from git import Repo
import numpy as np
import torch as th


def constant_learning_rate(global_step: int,
                           *args, **kwds):
    return kwds.pop('base_learning_rate')


def google_learning_rate(
        global_step: int,
        warmup_steps: int,
        base_learning_rate: float,
        decay_steps: int,
        decay_rate: float) -> float:
    """ learning rate decay + warmup """
    if global_step < warmup_steps:
        learning_rate = base_learning_rate * float(
            global_step) / float(warmup_steps)
    else:
        learning_rate = base_learning_rate
    learning_rate = learning_rate * (decay_rate **
                                     (float(global_step) / float(decay_steps)))
    return learning_rate


def assert_committed(pwd='.', force_commit: bool = True) -> str:
    repo = Repo(pwd, search_parent_directories=True)
    if force_commit:
        if repo.is_dirty():
            raise ValueError('Must commit!')
    else:
        logging.warn(
            F'`force_commit` is set to {force_commit}!'
            + ' Experiment may not be repeatable.')
    return repo.head.commit.hexsha


def add_histogram(writer, k: str, v: th.Tensor, *args, **kwds):
    raise ValueError(
        'add_histogram does not currently work due to' +
        'torch <=> numpy compatibility issue with cumsum()!'
    )
    if isinstance(v, list) and len(v) > 0:
        if isinstance(v[0], np.ndarray):
            v = np.stack(v, axis=0)
        else:
            v = th.stack(v, dim=0)
    writer.add_histogram(k, v, *args, **kwds)


def _to_scalar(x) -> float:
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, th.Tensor):
        return th.mean(x).item()
    if isinstance(x, np.ndarray):
        return np.mean(x)
    if isinstance(x, Iterable):
        return np.mean([_to_scalar(e) for e in x])
    raise ValueError(F'Unknown type = {x}, {type(x)}')


def add_scalar(writer, k: str, v: th.Tensor, *args, **kwds):
    writer.add_scalar(k, _to_scalar(v), *args, **kwds)
