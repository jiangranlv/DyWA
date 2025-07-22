#!/usr/bin/env python3

from typing import Mapping, Optional, Iterable

from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper
from models.common import merge_shapes, map_struct

import numpy as np
import torch as th
import einops

from gym import spaces


class CatObs(ObservationWrapper):
    """
    Temporally concatenate the frames.
    """

    def __init__(self,
                 env,
                 num_cat: int = 4,
                 flatten: bool = False,
                 keys: Optional[Iterable[str]] = None):
        super().__init__(env, self._wrap_obs)
        self._num_cat = num_cat
        self.flatten = flatten

        # Read keys from default (obs_space)
        if keys is None and isinstance(env.observation_space, spaces.Dict):
            keys = list(env.observation_space.keys())
        self.keys = list(keys) if (keys is not None) else None
        if not isinstance(env.observation_space, spaces.Dict):
            self.keys = None

        if flatten:
            raise ValueError('`flatten` arg might cause some issues...')

        def cat(src: spaces.Box) -> spaces.Box:
            assert (isinstance(src, spaces.Box))

            lo = einops.repeat(src.low, '... -> r ...',
                               r=num_cat)
            hi = einops.repeat(src.high, '... -> r ...',
                               r=num_cat)
            if flatten:
                return spaces.Box(
                    lo.reshape(-1),
                    hi.reshape(-1))
            else:
                return spaces.Box(lo, hi)

        # TODO: support `spaces.Dict`
        if isinstance(env.observation_space, spaces.Box):
            # spaces.Box
            self._obs_space = cat(env.observation_space)
        else:
            # spaces.Dict
            # NOTE: assumes dictionary of boxes.

            assert(isinstance(env.observation_space, spaces.Dict))
            obs_spaces = {}
            for key, obs_space in env.observation_space.items():
                if key in self.keys:
                    obs_space = cat(obs_space)
                obs_spaces[key] = obs_space
            # obs_spaces = map_struct(
            #     env.observation_space,
            #     lambda src, dst: (cat(src) if dst in self.keys else src),
            #     self.keys,
            #     base_cls=spaces.Box,
            #     dict_cls=(Mapping, spaces.Dict)
            # )
            # FIXME: this still only works for single-level nesting...
            self._obs_space = spaces.Dict(obs_spaces)

        def _alloc(src: spaces.Box, _):
            shape = merge_shapes(self.num_env, src.shape)
            return th.zeros(shape,
                            dtype=th.float,
                            device=self.device)

        # By default, shape=(N,T,D...)
        # The buffer does not exactly match
        # the observation space, in the case `flatten` is `True`.
        self._obs_buf = map_struct(self._obs_space, _alloc,
                                   base_cls=spaces.Box,
                                   dict_cls=(Mapping, spaces.Dict))
        self._prev_done: th.Tensor = None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs: th.Tensor) -> th.Tensor:
        # Option number one : roll every frame
        # Option number two : circular buffer + slice
        # roll only every N frames.
        # Which option is better? I'm not sure
        def _set_obs(src, dst):
            # FIXME: lots of wasted memory op??
            dst[...] = th.roll(dst, -1, 1)
            dst[:, -1] = src
            return dst

        if self.keys is None:
            _set_obs(obs, self._obs_buf)
        else:
            for key in self.keys:
                _set_obs(obs[key], self._obs_buf[key])

        if self.flatten:
            return map_struct(self._obs_buf,
                              lambda src, dst: src.reshape(src.shape[0], -1),
                              base_cls=th.tensor)
        else:
            return self._obs_buf

    def reset(self):
        obs = self.env.reset()
        return self._wrap_obs(obs)

    def step(self, *args, **kwds):
        obs, rew, done, info = self.env.step(*args, **kwds)

        # reset `obs` where it was previously marked `done`.
        # FIXME: what if "zero" obs is not "correct"??
        if self._prev_done is not None:
            # FIXME: below code assumes `obs_buf`
            # has rank==1
            # self._obs_buf *= ~self._prev_done[:, None, None]
            prev_mask = ~self._prev_done

            def _reset_buf(src, _):
                src.reshape(*prev_mask.shape, -1).mul_(prev_mask[:, None])

            map_struct(self._obs_buf, _reset_buf,
                       base_cls=th.Tensor)

        obs = self._wrap_obs(obs)
        # NOTE: `_prev_done` is a
        # reminder to reset `obs` later.
        self._prev_done = done
        return (obs, rew, done, info)


def test_with_vector_spaces():
    N: int = 2
    T: int = 4
    device = 'cpu'

    class DummyEnv(EnvIface):
        def __init__(self):
            self._device = device
            self.obs = th.zeros((N, 1), dtype=th.float,
                                device=device)
            self.prev_done = None

        @property
        def action_space(self):
            return spaces.Box(-1, +1, (1,))

        @property
        def observation_space(self):
            return spaces.Box(-1, +1, (1,))

        @property
        def device(self):
            return self._device

        @property
        def num_env(self):
            return N

        @property
        def timeout(self):
            return np.inf

        def reset_indexed(self):
            raise NotImplementedError('no reset_indexed')

        def setup(self):
            pass

        def reset(self):
            self.obs.fill_(0)
            return self.obs

        def step(self, actions):
            # == "apply actions + simulate" ==
            self.obs += 1
            if self.prev_done is not None:
                self.obs[self.prev_done] = 0

            # == "compute observations and return" ==
            rew = th.zeros((N,), device=self.device)
            done = th.rand(size=(N,), device=self.device) <= 0.05
            info = {}
            self.prev_done = done
            return self.obs, rew, done, info

    env = DummyEnv()
    env = CatObs(env, T)
    print(env.reset())
    for _ in range(16):
        obs, rew, done, info = env.step(None)
        print('obs', obs, obs.shape)  # .ravel())
        print('done', done)  # .ravel())


def main():
    from models.rl.v2.ppo_config import DomainConfig
    from env.random_env import RandomEnv
    cfg = DomainConfig(num_env=7,
                       obs_space={'a': (4,), 'b': (5, 6)},
                       num_act=1)
    env = RandomEnv(cfg)
    env = CatObs(env, num_cat=3)
    obs = env.reset()

    def _print(src, _):
        print(src.dtype, src.shape)

    map_struct(obs, _print,
               base_cls=th.Tensor)
    print('initially...')
    print(obs['a'])

    for i in range(4):
        obs, rew, done, info = env.step(actions=None)
        map_struct(obs, _print,
                   base_cls=th.Tensor)
        print(F'reset {th.argwhere(done).ravel()}')
        print(F'step {i}')
        print(obs['a'])


if __name__ == '__main__':
    main()
