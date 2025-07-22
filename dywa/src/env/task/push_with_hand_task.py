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
from env.robot.object_poker import ObjectPoker
from env.robot.ur5_fe import UR5FE, matrix_from_quaternion
from env.robot.franka import Franka

from env.env.help.with_nvdr_camera import WithNvdrCamera

# from env.task.null_task import NullTask
from env.task.push_task import PushTask, potential
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


def pot_rew(s0: Dict[str, th.Tensor],
            s1: Dict[str, th.Tensor]) -> th.Tensor:
    """ potential-based reward. """
    c0 = s0['hand_state'][..., :3]
    o0 = s0['object_state'][..., :3]
    c1 = s1['hand_state'][..., :3]
    o1 = s1['object_state'][..., :3]
    p1 = potential(o1, c1)  # neg. distance
    p0 = potential(o0, c0)  # neg. distance
    return (p1 - p0)


class PushWithHandTask(PushTask):
    @dataclass
    class Config(PushTask.Config):
        hand_obj_pot_coef: float = 1.0
        oob_fail_coef: Optional[float] = None
        max_pot_rew: float = 0.5

        def __post_init__(self):
            if self.oob_fail_coef is None:
                self.oob_fail_coef = self.fail_coef

    def __init__(self, cfg: Config, writer=None):
        super().__init__(cfg, writer=writer)
        self.cfg = cfg
        self._has_prev: th.Tensor = None
        self._prev_state: th.Tensor = None

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

    def is_hand_oob(self, hand: th.Tensor):
        # assert self.ws_lo.all_same()
        ws_lo = self.ws_lo[0]
        ws_hi = self.ws_hi[0]

        assert (ws_lo.shape[-1] == 3)
        return th.logical_or(
            (hand < ws_lo).any(dim=-1),
            (hand >= ws_hi).any(dim=-1))

    def compute_reward_from_trajectory(
            self, traj: th.Tensor, extra: th.Tensor,
            goal: th.Tensor,
            rewd: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Compute reward from trajectory.
        `traj` should be buffered by one?
        (Assumes TxNxD layout)
        """
        rewd = super().compute_reward_from_trajectory(traj, extra, goal, rewd)
        cfg = self.cfg
        if not cfg.sparse_reward:
            prv_obj = traj[:-1]
            nxt_obj = traj[1:]

            prv_hand = extra[:-1]
            nxt_hand = extra[1:]

            pot0 = potential(prv_obj[..., :3], prv_hand[..., :3])
            pot1 = potential(nxt_obj[..., :3], nxt_hand[..., :3])
            pot_reward = pot1 - pot0
            rewd[1:] += cfg.hand_obj_pot_coef * pot_reward.clamp_(
                -cfg.max_pot_rew, +cfg.max_pot_rew)
        return rewd

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
        hand_state = env.tensors['root'][env.robot.actor_ids.long()]
        state = {
            'goal': goal,
            'object_state': obj_state,
            'hand_state': hand_state
        }

        # Add another termination condition:
        # check if cube is OOB and also terminate episode.
        with nvtx.annotate("hand_oob"):
            hand_pos = hand_state[..., :3]
            hand_oob = th.logical_or(
                (hand_pos < self.ws_lo).any(dim=-1),
                (hand_pos >= self.ws_hi).any(dim=-1))
            new_fail = (hand_oob & ~done)
            done = th.logical_or(done, hand_oob)
            # Add additional failure condition.
            reward -= new_fail.float() * cfg.oob_fail_coef

        # Add potential-based reward based on hand-object distance.
        # (requires prev_state)
        if cfg.use_potential:
            if self._prev_state is not None:
                pot_reward = pot_rew(
                    self._prev_state,
                    state)
                pot_reward *= (self._has_prev).float()
                reward = reward + cfg.hand_obj_pot_coef * pot_reward.clamp_(
                    -cfg.max_pot_rew, +cfg.max_pot_rew)

        else:
            hand_dist = th.norm(
                state['object_state'][:, 0: 3] -
                state['hand_state'][:, 0: 3],
                p=2, dim=-1)
            print(cfg.hand_obj_pot_coef * 1 / (hand_dist + cfg.epsilon))
            reward += cfg.hand_obj_pot_coef * 1 / (hand_dist + cfg.epsilon)
        self._has_prev.copy_(~done)
        info['reward/pot'] = reward
        self._prev_state = copy_obs(state)
        return (reward, done, info)
