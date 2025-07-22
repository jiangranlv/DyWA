#!/usr/bin/env python3

from isaacgym.gymapi import KeyboardInput

from typing import Optional, Dict, Tuple, Union
from functools import partial
from gym import spaces
import torch as th
import numpy as np

from env.env.base import EnvBase
from models.common import merge_shapes


def get_code(x: Union[KeyboardInput, str]) -> KeyboardInput:
    if isinstance(x, KeyboardInput):
        return x
    if isinstance(x, str):
        X = x.upper()
        return getattr(KeyboardInput, F'KEY_{X}')
    raise ValueError(F'Unknown input = {x}')


def get_qwerty() -> str:
    out = 'QWERTYUIOPASDFGHJKL;ZXCVBNM,.'
    out = list(out)
    out[out.index(';')] = 'SEMICOLON'
    out[out.index(',')] = 'COMMA'
    out[out.index('.')] = 'PERIOD'
    return out


def get_key_map_xyzrpy():
    return {
        # XYZ
        'W': (0, +1.0),
        'S': (0, -1.0),
        'A': (1, +1.0),
        'D': (1, -1.0),
        'Q': (2, -1.0),
        'E': (2, +1.0),

        # RPY
        'I': (3, +1.0),
        'K': (3, -1.0),
        'J': (4, +1.0),
        'L': (4, -1.0),
        'U': (5, -1.0),
        'O': (5, +1.0)
    }


class KeyControl:
    def __init__(
            self, env: EnvBase,
            key_map: Optional[Dict[str, Tuple[int, float]]] = None,
            duration: int = 1):

        self.duration = duration
        self.count = 0
        self.device = th.device(env.device)

        # Determine action space.
        if isinstance(env.action_space, spaces.Box):
            self.actions = th.zeros(
                merge_shapes(env.num_env, env.action_space.shape),
                dtype=th.float,
                device=env.device)
            num_key: int = 3 * int(np.prod(env.action_space.shape))
            self.discrete = False
        elif isinstance(env.action_space. spaces.Discrete):
            self.actions = th.zeros((env.num_env, env.action_space.n),
                                    dtype=th.bool,
                                    device=env.device)
            num_key: int = env.action_space.n
            self.discrete = True
        else:
            raise ValueError(F'Unknown action_space = {env.action_space}')

        if key_map is None:
            # Automatically configure keymap.
            keys = get_qwerty()
            if len(keys) < num_key:
                raise ValueError(
                    'Cannot automatically configure key_map! '
                    + F'{len(keys)} < {num_key}'
                )

            if self.discrete:
                key_map = {k: (i, True)
                           for (i, k) in enumerate(keys[:num_key])}
            else:
                M = num_key // 3
                hi = env.action_space.high
                lo = env.action_space.low
                scale = 0.5 * (hi - lo)
                key_map = {k: (i % M, lo + scale * (i // M))
                           for (i, k) in enumerate(keys[:num_key])}
        print('key_map', key_map)

        # Sanitize key args to code args.
        code_map = {get_code(k): v for k, v in key_map.items()}

        for k, (i, v) in code_map.items():
            env.on_key(k, partial(self._set_action, i, v))

    def _set_action(self, index: int, value: float):
        self.actions[:, index] = value
        self.count = self.duration

    def reset(self, done: th.Tensor):
        self.actions[done] = 0

    def __call__(self, obs: th.Tensor):
        if self.count <= 0:
            self.actions.fill_(0)
        self.count = max(self.count - 1, 0)
        return self.actions
