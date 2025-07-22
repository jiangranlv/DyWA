#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi

from dataclasses import dataclass
from util.config import ConfigBase
from pathlib import Path
from functools import partial
from typing import Mapping
from tempfile import TemporaryDirectory

import torch as th
import numpy as np
import pickle

from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv
from util.path import ensure_directory
from util.torch_util import dcn
from models.common import map_struct

from env.env.wrap.normalize_env import NormalizeEnv


class LogEpisodes(WrapperEnv):
    """
    Log episodes from env.
    """

    @dataclass
    class Config(ConfigBase):
        log_dir: str = '/tmp/pkm/log-episodes'

    def __init__(self, cfg: Config, env: EnvIface):
        super().__init__(env)
        self._log_dir: Path = ensure_directory(cfg.log_dir)
        self._out_count: int = 0
        self._episodes = [[] for _ in range(env.num_env)]

    def _export_episode(self, prefix: str, ep):
        dirname = ensure_directory(self._log_dir / prefix)
        fname = F'{self._out_count:05d}.pkl'
        filename: str = dirname / fname
        with open(str(filename), 'wb') as fp:
            pickle.dump(ep, fp)
        self._out_count += 1

    def _do_step(self, act, obs, rew, done, info):
        assert ('success' in info)

        # Copy obs/rew and transfer to cpu.
        copy_fn = partial(map_struct,
                          op=lambda src, dst: dcn(src).copy(),
                          dst=None,
                          base_cls=th.Tensor)

        act = copy_fn(act)

        # FIXME: the below code will _only_ work
        # if there are no additional transformations until
        # reaching this... i.e. the pathway from NormalizeEnv
        # to LogFailures does not modify `obs`.
        n_env = self.unwrap(target=NormalizeEnv)
        if isinstance(n_env, NormalizeEnv):
            obs = n_env.normalizer.unnormalize_obs(obs)

        obs = copy_fn(obs)
        rew = copy_fn(rew)

        # Split data across episodes.
        # NOTE: we don't track `info`.
        if isinstance(obs, Mapping):
            for i in range(self.num_env):
                self._episodes[i].append((act[i],
                                          {k: v[i] for k, v in obs.items()},
                                          rew[i]))
        else:
            for ep, a, o, r in zip(self._episodes, act, obs, rew):
                ep.append((a, o, r))

        # Export failed/successful episodes
        done_mask = dcn(done)
        succ_mask = dcn(info['success'])
        fail_mask = (~succ_mask) & done_mask
        fail_ids = np.argwhere(~succ_mask & done_mask).ravel()
        for i in fail_ids:
            if len(self._episodes[i]) > 0:
                self._export_episode('fail', self._episodes[i])

        succ_ids = np.argwhere(succ_mask & done_mask).ravel()
        for i in succ_ids:
            if len(self._episodes[i]) > 0:
                self._export_episode('succ', self._episodes[i])

        # Clear the episodes.
        done_ids = np.argwhere(done_mask).ravel()
        for i in done_ids:
            self._episodes[i] = []

    def step(self, actions: th.Tensor):
        out = self.env.step(actions)
        obs, rew, done, info = out
        self._do_step(actions, obs, rew, done, info)
        return out


def main():
    import random
    from env.random_env import RandomEnv
    from models.rl.v2.ppo_config import DomainConfig
    cfg = DomainConfig(
        num_env=4,
        num_obs=4,
        num_act=1)
    env = RandomEnv(cfg)

    with TemporaryDirectory() as tmpdir:
        env = LogFailures(LogFailures.Config(log_dir=tmpdir), env)
        for _ in range(1000):
            action = th.as_tensor(np.stack(
                [env.action_space.sample()
                 for _ in range(env.num_env)]),
                device=env.device)
            env.step(action)

        episodes = (list(Path(tmpdir).glob('*.pkl')))
        if len(episodes) > 0:
            episode = random.choice(episodes)
            with open(episode, 'rb') as fp:
                ep = pickle.load(fp)
                act, obs, rew = zip(*ep)
                act = np.stack(act)
                obs = np.stack(obs)
                rew = np.stack(rew)
                print(act.shape)
                print(obs.shape)
                print(rew.shape)
            print(ep)


if __name__ == '__main__':
    main()
