#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
import pickle

from typing import Optional, Mapping
from dataclasses import dataclass, replace
from models.common import map_struct, map_tensor
from env.env.wrap.base import (ObservationWrapper,
                                   add_obs_field,
                                   WrapperEnv
                                   )
from env.env.wrap.record_viewer import RecordViewer
from env.env.wrap.nvdr_record_viewer import NvdrRecordViewer
from env.env.wrap.nvdr_record_episode import NvdrRecordEpisode
from envs.cube_env_wrappers import CountCategoricalSuccess

import torch as th
import torch.nn as nn
import numpy as np
import einops

from util.torch_util import dcn
from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from env.util import (
    set_seed, draw_sphere,
    draw_cloud_with_sphere,
    draw_patch_with_cvxhull
)
from env.env.wrap.normalize_env import NormalizeEnv
from env.env.wrap.draw_patch_attn import DrawPatchAttention

from icecream import ic
from gym import spaces

from train_ppo_arm import (
    Config as TrainConfig,
    load_agent,
    load_env)

from envs.cube_env_wrappers import DrawPatchCenter
import cv2
from models.rl.v4.rppo import (
    RecurrentPPO as RecurrentPPOv4,
    get_action_distribution,
    STATE_KEY
)


@dataclass
class Config(TrainConfig):
    sample_action: bool = True

    use_record_viewer: bool = False
    record_viewer: RecordViewer.Config = RecordViewer.Config()

    draw_debug_lines: bool = True
    log_categorical_results: bool = False

    force_vel: Optional[float] = 0.1
    force_rad: Optional[float] = 0.1
    force_ang: Optional[float] = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.force_vel is not None:
            self.use_tune_goal_speed = False
            self.env.task.max_speed = self.force_vel
        if self.force_rad is not None:
            self.use_tune_goal_radius = False
            self.env.task.goal_radius = self.force_rad
        if self.force_ang is not None:
            self.use_tune_goal_radius = False
            self.env.task.goal_angle = self.force_ang


class TeacherEnv(WrapperEnv):
    """
    ~DAgger
    """

    def __init__(self, env, agent,
                 add_action: bool = True,
                 add_state: bool = True):
        super().__init__(env)
        self.agent = agent

        self._update_action = None
        if add_action:
            obs_space = env.observation_space
            obs_space, self._update_action = add_obs_field(
                env.observation_space,
                'teacher_action',
                env.action_space)

        self._update_state = None
        if add_state:
            obs_space, self._update_state = add_obs_field(
                env.observation_space,
                'teacher_state',
                env.action_space)
        self._obs_space = obs_space
        self.__hidden = None
        self.__done = None

    @property
    def observation_space(self):
        return self._obs_space

    def __get_action(self, state):
        agent = self.agent
        dist = get_action_distribution(
            state,
            agent.actor_net,
            agent.domain_cfg.discrete, agent.cfg.tanh_xfm,
            aux={}, stable_normal=agent.stable_normal)
        actn = dist.sample()
        return actn

    def reset(self):
        obs = self.env.reset()

        with th.inference_mode():
            # Initialize the agent and self states
            _, _, hidden, done = self.agent.init(obs)
            self.__hidden = hidden
            self.__done = done

            # immediately return the action that
            # best suits this observation
            next_action = self.__get_action(self.__hidden[STATE_KEY])

        if self._update_action is not None:
            obs = self._update_action(obs, next_action)
        if self._update_state is not None:
            obs = self._update_state(obs, self.__hidden[STATE_KEY])
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)  # super().step(action)

        with th.inference_mode():
            _, self.__hidden = self.agent.state_net(
                self.__hidden, action, obs)
            next_action = self.__get_action(
                self.__hidden[STATE_KEY])

            # Also reset agent's state
            keep = (~done)[..., None]
            map_tensor(self.__hidden,
                       lambda src, _: src.mul_(keep))

        if self._update_action is not None:
            obs = self._update_action(obs, next_action)
        if self._update_state is not None:
            obs = self._update_state(obs, self.__hidden[STATE_KEY])

        return obs, rew, done, info


@hydra_cli(config_name='show')
def main(cfg: Config):
    ic.configureOutput(includeContext=True)
    cfg = recursive_replace_map(cfg, {'finalize': True})
    ic(cfg)

    # Maybe it's related to jit
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path, writer = None, None
    _ = set_seed(cfg.env.seed)
    if (cfg.use_nvdr_record_episode or cfg.use_nvdr_record_viewer):
        cfg.env.track_debug_lines = True

    env = load_env(cfg, path, writer, freeze_env=True,
                   check_viewer=False
                   )

    # Update cfg elements from `env`.
    obs_space = map_struct(
        env.observation_space,
        lambda src, _: src.shape,
        base_cls=spaces.Box,
        dict_cls=(Mapping, spaces.Dict)
    )
    if cfg.state_net_blocklist is not None:
        for key in cfg.state_net_blocklist:
            obs_space.pop(key, None)
    dim_act = (
        env.action_space.shape if isinstance(
            env.action_space,
            spaces.Box) else env.action_space.n)
    cfg = replace(cfg, net=replace(cfg.net,
                                   obs_space=obs_space,
                                   act_space=dim_act
                                   ))
    agent = load_agent(cfg, env, None, None)
    # agent.eval()
    ic(agent)
    env = TeacherEnv(env, agent)

    obs = env.reset()
    for _ in range(1000):
        obs, rew, done, info = env.step(obs['teacher_action'].clone())


if __name__ == '__main__':
    main()
