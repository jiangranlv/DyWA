#!/usr/bin/env python3

from typing import Mapping
from isaacgym import gymapi, gymtorch

from dataclasses import dataclass
from util.config import ConfigBase

from env.env.base import EnvIface
from env.push_env import PushEnv

import torch as th
import time
import numpy as np

from pytorch3d.transforms import quaternion_to_matrix

from env.util import get_mass_properties
from env.env.wrap.zero_out_obs import ZeroOutObs
from env.robot.virtual_poker import VirtualPoker
from gym import spaces

from env.env.wrap.base import (
    WrapperEnv,
    ObservationWrapper,
    ActionWrapper,
    add_obs_field
)


class AddObjectName(ObservationWrapper):
    """
    Add the full object state into the observation.
    """

    def __init__(self, env: EnvIface, key: str = 'name'):
        super().__init__(env, self._wrap_obs)
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             spaces.Text(max_length=256)
                                             )
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        names = self.scene.cur_names
        return self._update_fn(obs, names)


class AddObjectNameToInfo(WrapperEnv):
    """
    Add the full object state into the observation.
    """

    def __init__(self,
                 env: EnvIface,
                 key: str = 'object_name'):
        super().__init__(env)
        self.__key = key

    def step(self, *args, **kwds):
        obs, rew, done, info = self.env.step(*args, **kwds)
        if info is not None:
            assert (isinstance(info, Mapping))
            info = {**info, self.__key: self.scene.cur_names}
        else:
            info = {self.__key: self.scene.cur_names}
        return (obs, rew, done, info)


class AddObjectState(ObservationWrapper):
    """
    Add the full object state into the observation.
    """

    def __init__(self, env: EnvIface, key: str = 'state'):
        super().__init__(env, self._wrap_obs)
        obs_space, update_fn = add_obs_field(env.observation_space,
                                             key,
                                             spaces.Box(-np.inf, np.inf, (13,))
                                             )
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        obj_ids = self.scene.cur_ids.long()
        state = self.tensors['root'][obj_ids, :]
        return self._update_fn(obs, state)


class AddGoalFromPushTask(ObservationWrapper):
    """
    Add the current goal from the push task into the observation.
    """

    def __init__(self,
                 env: EnvIface,
                 key: str = 'goal',
                 dim: int = 3,
                 add_thresh: bool = False):
        super().__init__(env, self._wrap_obs)
        if add_thresh:
            dim = dim + 3
        self._add_thresh = add_thresh
        obs_space, update_fn = add_obs_field(
            env.observation_space, key, spaces.Box(-np.inf, np.inf, (dim,)))
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        if self._add_thresh:
            task = self.task
            full_goal_spec = th.cat(
                [task.goal, task.goal_radius_samples[..., None],
                 task.goal_angle_samples[..., None],
                 task.max_speed_samples[..., None]],
                dim=-1)
            return self._update_fn(obs, full_goal_spec)
        else:
            return self._update_fn(obs, self.task.goal)


class AddGoalThreshFromPushTask(ObservationWrapper):
    """
    Add the current goal thresholds from the
    push task into the observation as well.
    Currently, includes
    > goal radius
    > goal angle
    > max speed
    """

    def __init__(self,
                 env: EnvIface,
                 key: str = 'goal_thresh',
                 dim: int = 3):
        super().__init__(env, self._wrap_obs)
        obs_space, update_fn = add_obs_field(
            env.observation_space, key, spaces.Box(-np.inf, np.inf, (dim,)))
        self._obs_space = obs_space
        self._update_fn = update_fn

    @property
    def observation_space(self):
        return self._obs_space

    def _wrap_obs(self, obs):
        task = self.task
        thresh = th.stack([task.goal_radius_samples,
                           task.goal_angle_samples,
                           task.max_speed_samples],
                          dim=-1)
        return self._update_fn(obs, thresh)


class PrependBodyId(ActionWrapper):
    """
    Automatically prepend the controlled object ID to the env action.
    """

    def __init__(self, env: EnvIface):
        super().__init__(env, self._wrap_act)
        if isinstance(self.robot, VirtualPoker):
            self._act_space = spaces.Box(
                env.action_space.low[1:],
                env.action_space.high[1:])
        else:
            self._act_space = env.action_space

    @property
    def action_space(self):
        return self._act_space

    def _wrap_act(self, action: th.Tensor) -> th.Tensor:
        if action is None:
            return None
        body_indices = self.scene.body_ids.to(
            action.device)
        return th.cat([body_indices[..., None], action], dim=-1)


def main():
    import numpy as np
    from env.push_env import ManualAgent
    from matplotlib import pyplot as plt

    th.cuda.set_device('cuda:3')

    num_env: int = 1
    env = PushEnv(PushEnv.Config(num_env=num_env,
                                 use_viewer=False,
                                 draw_task_goal=True,
                                 draw_obj_pos_2d=True,
                                 draw_force=False,

                                 compute_device_id=3,
                                 graphics_device_id=3,
                                 th_device='cuda:3'))
    env = AddGoalFromPushTask(env)
    env = PrependBodyId(env)
    # env = TrainEnv(TrainEnv.Config(num_env=num_env,
    #                                use_viewer=True,
    #                                draw_task_goal=True,
    #                                draw_obj_pos_2d=True,
    #                                draw_force=True
    #                                ))
    env.reset()
    agent = ManualAgent(num_env, env)

    rews = []

    plt.ion()  # hmm
    fig, ax = plt.subplots()
    ln, = ax.plot([], [])
    for _ in range(4096):
        action = agent()
        # print(action)
        _, rew, _, _ = env.step(action[..., 1:])
        rews.append(rew.mean().item())
        ln.set_data(np.arange(len(rews)), rews)
        try:
            ax.set_xlim(-1, len(rews) + 1)
            # ax.set_ylim(np.min(rews), np.max(rews))
            ax.set_ylim(-0.04, 0.04)
        except Exception as e:
            print(e)
        # time.sleep(0.001)
        ln.figure.canvas.flush_events()
        plt.pause(0.001)


if __name__ == '__main__':
    main()
