#!/usr/bin/env python3

from env.env.wrap.base import ObservationWrapper

import logging
from typing import Iterable, Dict, Optional
import numpy as np
import torch as th
from gym import spaces

from icecream import ic


class DictToTensor(ObservationWrapper):
    def __init__(self, env, keys: Optional[Iterable[str]] = None):
        super().__init__(env, self._to_tensor)
        if keys is None:
            assert (isinstance(env.observation_space, spaces.Dict))
            keys = sorted(list(env.observation_space.keys()))
        self._keys = list(keys)
        lo = np.concatenate([env.observation_space[k].low
                             for k in self._keys], axis=0)
        hi = np.concatenate([env.observation_space[k].high
                             for k in self._keys], axis=0)
        self._observation_space = spaces.Box(lo, hi)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _to_tensor(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        return th.cat([obs[k] for k in self._keys], dim=-1)


class MergeKeys(ObservationWrapper):
    def __init__(self, env,
                 src_keys: Optional[Iterable[str]] = None,
                 dst_key: str = 'obs'):
        super().__init__(env, self._wrap_obs)
        assert (isinstance(env.observation_space, spaces.Dict))
        if src_keys is None:
            src_keys = sorted(list(env.observation_space.src_keys()))

        self.__src_keys = src_keys
        self.__dst_key = dst_key

        # Build obs_space
        obs_space = {k: v for (k, v) in env.observation_space.items()}

        src_spaces = []

        lo = []
        hi = []
        dst_size: int = 0
        for k in src_keys:
            s = obs_space.pop(k, None)
            if s is None:
                logging.warn(
                    F'Requested key = {k} not available while merging keys!')
                continue
            assert (isinstance(s, spaces.Box))
            src_spaces.append(s)
            # NOTE: `initial` is already 1 by default for np.prod,
            # but we explicitly mark it as 1 to ensure correctness
            # for scalars in the future.
            dst_size += np.prod(s.shape, initial=1)
            lo.append(s.low.reshape(-1))
            hi.append(s.high.reshape(-1))
        lo = np.concatenate(lo, axis=0)
        hi = np.concatenate(hi, axis=0)
        # FIXME: may result in unexpected behavior for
        # scalar inputs going from shape=() to shape=(1,)
        obs_space[dst_key] = spaces.Box(lo, hi,
                                        shape=(dst_size,))
        ic('obs_space', obs_space)
        self._observation_space = spaces.Dict(obs_space)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    def _wrap_obs(self, obs):
        obs = dict(obs)
        obs[self.__dst_key] = th.cat([
            obs.pop(k).reshape(self.num_env, -1)
            for k in self.__src_keys], dim=-1)
        return obs


def test_merge_keys():
    obs_spec = {'a': (3,), 'b': (4, 5)}
    b: int = 7  # batch size

    class Env:
        @property
        def observation_space(self):
            return spaces.Dict({k: spaces.Box(low=-1, high=+1, shape=v) for
                                k, v in obs_spec.items()})

        @property
        def num_env(self):
            return b

        def reset(self):
            return {k: th.randn(size=(b, *v)) for (k, v) in obs_spec.items()}
    th.random.manual_seed(0)
    env = Env()
    print(env.reset())

    th.random.manual_seed(0)
    env = MergeKeys(env, ['a', 'b'], 'c')
    print(env.reset())


def main():
    test_merge_keys()


if __name__ == '__main__':
    main()
