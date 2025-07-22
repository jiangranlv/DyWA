#!/usr/bin/env python3

from dataclasses import dataclass
from util.config import ConfigBase
from gym import spaces
from env.env.iface import EnvIface
from env.env.wrap.base import ActionWrapper
import torch as th
from models.common import merge_shapes


def cat_table(act_range: th.Tensor):
    num_act, num_bin = act_range.shape
    act_table = th.zeros(
        (num_act * num_bin, num_act),
        dtype=th.float32, device=act_range.device)
    ii = th.arange(num_act * num_bin, device=act_range.device)
    act_table[ii, ii // num_bin] = act_range.ravel()
    return act_table


class TabulateAction(ActionWrapper):

    @dataclass
    class Config(ConfigBase):
        num_bin: int = 3
        # whether to "center" the actions in the middle of action "cells"
        center: bool = False
        lookup: bool = True
        method: str = 'cat'

    def __init__(self, cfg: Config, env: EnvIface, **kwds):
        self.cfg = cfg
        self.env = env
        super().__init__(env, self._wrap_act)

        # Required for tabulating.
        # assert (isinstance(env.action_space, spaces.Box))
        # assert (env.action_space.is_bounded())

        # spaces.Box().shape
        self._actions_buf = th.zeros(
            merge_shapes(env.num_env, env.action_space.shape),
            dtype=th.float32,
            device=env.device  # or env.device
        )

        lo = th.as_tensor(env.action_space.low,
                          dtype=th.float32, device=self.device)
        hi = th.as_tensor(env.action_space.high,
                          dtype=th.float32, device=self.device)
        self.lo = lo
        self.hi = hi

        if cfg.lookup:
            if cfg.center:
                scale = (hi - lo) / (2 * cfg.num_bin)
                act_range = (
                    lo[:, None] +
                    th.arange(1, 2 * cfg.num_bin, 2,
                              device=self.device)[None, :] * scale[:, None]
                )

            else:
                act_range = (lo[:, None] + th.linspace(0, 1, cfg.num_bin,
                             device=self.device)[None, :] * (hi - lo)[:, None])
            # Convert action range -> table
            if cfg.method == 'prod':
                act_table = th.cartesian_prod(*act_range)
                assert (len(env.action_space.shape) == 1)
                self._act_space = spaces.Discrete(
                    cfg.num_bin ** env.action_space.shape[0])
            elif cfg.method == 'cat':
                act_table = cat_table(act_range)
                assert (len(env.action_space.shape) == 1)
                self._act_space = spaces.Discrete(
                    cfg.num_bin * env.action_space.shape[0])
            elif cfg.method == 'set':
                act_table = th.as_tensor(kwds.pop('act_table'),
                                         dtype=th.float,
                                         device=self.device)
                self._act_space = spaces.Discrete(len(act_table))
            else:
                raise ValueError(F'Unknown method = {cfg.method}')
            self._act_table = act_table
        else:
            raise ValueError('compute branch is not currently functional.')
            if cfg.center:
                self.scale = (self.hi - self.lo) / cfg.num_bin
                self.bias = self.lo + 0.5 * self.scale
            else:
                self.bias = self.lo
                self.scale = (self.hi - self.lo) / (cfg.num_bin - 1)
            self._act_space = spaces.Discrete(
                cfg.num_bin ** env.action_space.shape[0])

    def _wrap_act(self, action: th.Tensor):
        cfg = self.cfg

        if action is None:
            return action

        # [1] Convert action to index.
        if th.is_floating_point(action):
            # Given as logits
            index = th.argmax(action, dim=-1, keepdim=True)
        else:
            # Given as indices
            index = action

        # [1] index ->
        if cfg.lookup:
            # Option #1 : lookup from table.
            self._actions_buf.fill_(0)
            th.index_select(self._act_table, 0, index,
                            out=self._actions_buf)
        else:
            # option #2 : compute from strides.
            raise ValueError('compute branch is not currently functional.')
            self._actions_buf.fill_(0)
            index = th.div(index, cfg.num_bin, rounding_mode='floor')
            value = (index % cfg.num_bin)
            value = self.bias[None, :] + value * self.scale[None, :]
            value = value.to(self._actions_buf.dtype)
            # actions_buf[i][index[i,j]] = value[i, j]
            # self._actions_buf.scatter_(-1, index[..., None], value)
            self._actions_buf[th.arange(len(index)), index] = value
        return self._actions_buf

    @property
    def action_space(self):
        return self._act_space


def main():
    num_env: int = 729
    device: str = 'cuda:0'

    class DummyEnv(EnvIface):

        @property
        def device(self):
            return 'cuda:0'

        @property
        def action_space(self):
            return spaces.Box(-0.2, +0.2, (6,))

        @property
        def observation_space(self):
            pass

        def reset(self):
            pass

        def reset_indexed(self):
            pass

        def setup(self):
            pass

        def step(self):
            pass
    env = DummyEnv()
    wrap = TabulateAction(
        TabulateAction.Config(
            center=False,
            lookup=True
        ),
        env)

    # action = th.randint(0, wrap.action_space.n,
    #                     size=(num_env,), device=device)
    action = th.arange(num_env, device=device) % wrap.action_space.n
    print(wrap._wrap_act(action))
    pass


if __name__ == '__main__':
    main()
