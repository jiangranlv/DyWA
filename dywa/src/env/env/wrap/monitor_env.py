#!/usr/bin/env python3

from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv

from typing import Optional
from dataclasses import dataclass
from util.config import ConfigBase

import torch as th
from torch.utils.tensorboard import SummaryWriter


class MonitorEnv(WrapperEnv):
    """
    Monitor Env. logs the following statistics:

    Average:
    > [approx] avg. episode length.
    > reward.
    > [approx] discounted returns.
    > [approx] undiscounted returns.

    Total:
    > number of interactions (mul. by num_env)
    > number of episodes.
    > number of successful episodes.

    WARN: to avoid annoyingnesses with empty tensors,
    we do not apply masking.
    """

    @dataclass
    class Config(ConfigBase):
        gamma: float = 0.99
        log_period: int = 1024
        verbose: bool = True
        num_env_record: int = 0

    def __init__(self,
                 cfg: Config,
                 env: EnvIface):
        super().__init__(env)
        self.cfg = cfg
        N: int = env.num_env

        self.returns = th.zeros(N, dtype=th.float,
                                device=env.device)
        self.discounted_returns = th.zeros(N, dtype=th.float,
                                           device=env.device)
        self.cumulative_reward = th.zeros(N, dtype=th.float,
                                          device=env.device)
        self.discounted_cumulative_reward = th.zeros(N, dtype=th.float,
                                                     device=env.device)
        self.episode_step = th.zeros(N, dtype=th.int32,
                                     device=env.device)
        self.true_cumulative_return = th.zeros(N, dtype=th.float,
                                               device=env.device)
        self.episode_count: int = 0
        self.success_count: int = 0
        self.step_count: int = 0
        self.last_episode_count: int = 0
        self.last_success_count: int = 0

        self.num_env_record = cfg.num_env_record
        self.batch_episode_count = th.zeros(self.num_env_record, dtype=th.int32,
                                            device=env.device)
        self.batch_success_count = th.zeros(self.num_env_record, dtype=th.int32,
                                            device=env.device)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def setup(self):
        return self.env.setup()

    def reset_indexed(self, *args, **kwds) -> th.Tensor:
        return self.env.reset_indexed(*args, **kwds)

    def reset(self) -> th.Tensor:
        return self.env.reset()

    def _add_scalar(self, k, v, step: Optional[int] = None):
        if step is None:
            step = self.step_count
        if self.writer is not None:
            return self.writer.add_scalar(k, v, global_step=step)
        elif self.cfg.verbose:
            print(F'{k}={v}')

    def _log(self):
        self._add_scalar('env/eplen',
                         self.episode_step.float().mean().item())
        self._add_scalar('env/return',
                         self.returns.mean().item())
        self._add_scalar('env/discounted_return',
                         self.discounted_returns.mean().item())
        self._add_scalar('env/num_interactions',
                         self.step_count * self.num_env)
        self._add_scalar('env/num_steps', self.step_count)
        self._add_scalar('env/num_episodes', self.episode_count)
        self._add_scalar('env/num_success', self.success_count)

        # Add "smooth" success rate
        suc_rate = (0.0 if self.episode_count <= 0
                    else self.success_count / self.episode_count)
        self._add_scalar('env/suc_rate', suc_rate)

        # Add Batch success rate
        for i in range(self.num_env_record):
            if self.batch_episode_count[i] > 0:
                self._add_scalar(f'batch/suc_rate_{i}',
                                 (self.batch_success_count[i] / self.batch_episode_count[i]).float().item())

        # Add "current" success rate
        d_suc = (self.success_count - self.last_success_count)
        d_eps = (self.episode_count - self.last_episode_count)
        current_suc_rate = (0.0 if d_eps <= 0 else d_suc / d_eps)
        self._add_scalar('env/cur_suc_rate', current_suc_rate)

        # Add episode returns.
        self._add_scalar('env/avg_episode_return',
                         (self.cumulative_reward.sum() / d_eps).item())
        steps = self.cfg.log_period * self.num_env
        self._add_scalar(
            'env/avg_reward',
            (self.cumulative_reward.sum() / steps).item())
        self._add_scalar(
            'env/avg_episode_discounted_return',
            (self.discounted_cumulative_reward.sum() / d_eps).item())
        self._add_scalar(
            'env/avg_true_return',
            (self.true_cumulative_return.sum() / d_eps).item())

        # Reset counts and statistics.
        self.last_success_count = self.success_count
        self.last_episode_count = self.episode_count

        self.cumulative_reward.fill_(0)
        self.discounted_cumulative_reward.fill_(0)
        self.true_cumulative_return.fill_(0)

    def step(self, action: th.Tensor):
        cfg = self.cfg
        obs, rew, done, info = self.env.step(action)

        with th.no_grad():
            resetf = (~done).float()

            # Update episodic statistics.
            self.cumulative_reward.add_(rew)
            self.discounted_cumulative_reward.mul_(cfg.gamma).add_(rew)
            self.true_cumulative_return.add_(
                rew.mul(th.pow(cfg.gamma, self.episode_step))
            )

            # update returns.
            self.returns.add_(rew)
            self.discounted_returns.mul_(cfg.gamma).add_(rew)

            # update episode steps.
            self.episode_step.add_(1)
            # track total number of episodes.
            self.episode_count += done.sum().item()
            self.batch_episode_count += done[:self.num_env_record]
            if 'success' in info:
                self.success_count += info['success'].sum().item()
                self.batch_success_count += info['success'][:self.num_env_record]
            self.step_count += 1

            if (self.step_count % cfg.log_period == 0):
                self._log()

            # Reset cumulative statistics.
            self.returns.mul_(resetf)
            self.discounted_returns.mul_(resetf)
            # self.episode_step.masked_fill_(done, 0)
            self.episode_step[done] = 0

            # consistent = (done == self.env.buffers['done']).all()
            # if not consistent.item():
            #    raise ValueError('incons')
            # print(self.episode_step.float().max(),
            #       self.episode_step.float().mean(),
            #       self.env.buffers['step'].float().max(),
            #       self.env.buffers['step'].float().mean(),
            #       )

        return obs, rew, done.clone(), info

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)
