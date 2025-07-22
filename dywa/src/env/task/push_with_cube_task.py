#!/usr/bin/env python3

from abc import ABC, abstractmethod

import time
from typing import Tuple, Iterable, Optional, Union, Dict
from dataclasses import dataclass, field
from util.config import ConfigBase
import numpy as np
import cv2
from tqdm.auto import tqdm
from pathlib import Path
from cho_util.math import transform as tx


from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
import torch as th
import einops

from env.scene.base import SceneBase
from env.scene.tabletop_scene import TableTopScene
from env.scene.tabletop_with_cube_scene import TableTopWithCubeScene
from env.scene.tabletop_with_object_scene import TableTopWithObjectScene
from env.robot.base import RobotBase

from env.robot.virtual_poker import VirtualPoker
from env.robot.fe_gripper import FEGripper
from env.robot.cube_poker import CubePoker
from env.robot.object_poker import ObjectPoker
from env.robot.ur5_fe import UR5FE, matrix_from_quaternion
from env.robot.franka import Franka

from env.env.help.with_nvdr_camera import WithNvdrCamera

# from env.task.null_task import NullTask
from env.task.push_task import PushTask
from env.env.base import (EnvBase)
from env.common import (
    get_default_sim_params,
    create_camera
)

from env.util import get_mass_properties
from util.vis.flow import flow_image
from util.torch_util import dcn
from util.config import recursive_replace_map
from util.torch_util import dot

from icecream import ic
import nvtx


def copy_obs(obs):
    if isinstance(obs, th.Tensor):
        return obs.detach().clone()
    elif isinstance(obs, dict):
        return {k: copy_obs(v) for k, v in obs.items()}
    else:
        raise ValueError('no idea.')


@dataclass
class RewardConfig(ConfigBase):
    # Reward coefficients
    reach_coef: float = 1.0
    reach_eps: float = 1e-1
    task_coef: float = 0.0
    dist_coef: float = 2.0
    time_coef: float = 0.02
    pot_coef: float = 1.0

    # Termination terms
    succ_coef: float = +100.0
    fail_coef: float = -10.0

    reward_type: str = 'pot'


def distance_to_line(p: th.Tensor, a: th.Tensor, b: th.Tensor) -> th.Tensor:
    # unit vector in the direction of the line
    u = b - a
    u /= th.linalg.norm(u, dim=-1, keepdim=True)
    d = (p - a)
    s = d - (dot(d, u)[..., None] * u)
    return th.linalg.norm(s, dim=-1)


@th.jit.script
def potential(
        cube_pos: th.Tensor,
        goal_pos: th.Tensor,
        obj_pos: th.Tensor,

        reach_coef: float,
        goal_coef: float) -> th.Tensor:
    # smaller the better
    dist_near = (-th.linalg.norm(
        obj_pos[..., :3] - cube_pos[..., :3],
        dim=-1))
    # smaller the better
    dist_goal = (-th.linalg.norm(
        obj_pos[..., :3] - goal_pos[..., :3],
        dim=-1))
    return (reach_coef * dist_near + goal_coef * dist_goal)


def pot_rew(s0: Dict[str, th.Tensor], s1: Dict[str, th.Tensor],
            reach_coef: float,
            goal_coef: float
            ) -> th.Tensor:
    """ potential-based reward. """
    c0 = s0['cube_state'][..., :3]
    g0 = s0['goal'][..., :3]
    o0 = s0['object_state'][..., :3]

    c1 = s1['cube_state'][..., :3]
    g1 = s1['goal'][..., :3]
    o1 = s1['object_state'][..., :3]
    p1 = potential(c1, g1, o1, reach_coef, goal_coef)
    p0 = potential(c0, g0, o0, reach_coef, goal_coef)
    return (p1 - p0)


class PushWithCubeTask(PushTask):
    @dataclass
    class Config(PushTask.Config):
        reward: RewardConfig = RewardConfig()

    def __init__(self, cfg: Config, writer=None):
        super().__init__(cfg, writer=writer)
        self.cfg = cfg
        self._has_prev: th.Tensor = None
        self._prev_state: th.Tensor = None
        self.k_w: th.Tensor = None

    def setup(self, env: 'EnvBase'):
        super().setup(env)

        cfg = self.cfg
        device: th.device = th.device(env.device)
        self.device = device

        # Previous values cache
        self._has_prev: th.Tensor = th.zeros(
            (env.num_env,), dtype=th.bool,
            device=device)
        self._prev_state: th.Tensor = None

    def reset(self, env: 'EnvBase', indices: Iterable[int]):
        if indices is None:
            indices = th.arange(env.cfg.num_env,
                                dtype=th.int32,
                                device=env.cfg.th_device)
        else:
            indices = th.as_tensor(indices,
                                   dtype=th.long,
                                   device=env.cfg.th_device)
            self._has_prev[indices] = 0
        return super().reset(env, indices)

    def compute_feedback(self,
                         env: 'EnvBase',
                         obs: th.Tensor,
                         action: th.Tensor):
        cfg = self.cfg

        # Compute the basics from PushTask.
        reward, done, info = super().compute_feedback(env, obs, action)
        info['task_reward'] = reward

        # Query relevant quantities.
        # FIXME: maybe try to avoid duplicating these
        # lookups and code...???
        goal = self.goal
        obj_ids = env.scene.cur_ids.long()
        obj_state = env.tensors['root'][obj_ids, :]
        cube_state = env.tensors['root'][env.robot.actor_ids.long()]
        state = {
            'goal': goal,
            'object_state': obj_state,
            'cube_state': cube_state
        }

        # Add another termination condition:
        # check if cube is OOB and also terminate episode.
        with nvtx.annotate("cube_oob"):
            cube_pos = cube_state[..., :3]
            cube_oob = th.logical_or(
                (cube_pos < self.ws_lo).any(dim=-1),
                (cube_pos >= self.ws_hi).any(dim=-1))
            done = th.logical_or(done, cube_oob)

        # potential-based reward
        # (requires prev_state)
        if self._prev_state is None:
            reward = reward * 0
        else:
            pot_reward = pot_rew(
                self._prev_state,
                state,
                cfg.reward.reach_coef,
                cfg.reward.dist_coef)

            reward = (
                cfg.reward.pot_coef * pot_reward
                - cfg.reward.time_coef
            )
        reward[~self._has_prev] = 0.0
        self._has_prev.copy_(~done)
        info['reward/pot'] = reward

        if abs(cfg.reward.fail_coef) > 0:
            fail = done & (~info['success'])
            reward.masked_fill_(fail,
                                -abs(cfg.reward.fail_coef))
        if abs(cfg.reward.succ_coef) > 0:
            reward.masked_fill_(info['success'],
                                cfg.reward.succ_coef)

        self._prev_state = copy_obs(state)

        return (reward, done, info)
