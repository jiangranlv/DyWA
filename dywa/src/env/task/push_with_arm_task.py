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

from env.common import quat_rotate

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


@th.jit.script
def compute_card_reward(
    action_gains: Optional[th.Tensor],
    object_dist_weight: Tuple[float, float],
    epsilon: Tuple[float, float],
    object_goal_poses_buf: th.Tensor,
    # mean_energy: th.Tensor,
    object_state: th.Tensor,
    gripper_state: th.Tensor,
    # size: Tuple[float, float, float],
    use_inductive_reward: bool,
    # use_energy_reward: bool,
    kpt: th.Tensor,
    pcd: Optional[th.Tensor],
    left_tip_state: th.Tensor,
    right_tip_state: th.Tensor,
) -> th.Tensor:
    # distance from each finger to the centroid of the object, shape (N, 3).
    # gripper_state = th.mean(gripper_state, 1)
    if pcd is None:
        curr_norms = th.norm(gripper_state[..., 0:3] - object_state[..., 0:3],
                            p=2, dim=-1)
    else:
        cur_pcd = quat_rotate(object_state[..., None, 3:7], pcd)\
                            + object_state[..., None, 0:3]

        left_min, _ = th.min(th.norm(left_tip_state[..., None, 0:3]-cur_pcd,
                                p=2, dim=-1), dim = -1)
        right_min, _ = th.min(th.norm(right_tip_state[..., None, 0:3]-cur_pcd,
                                p=2, dim=-1), dim = -1)
        curr_norms = th.minimum(left_min, right_min)
    # residual = th.norm(obs_buf[:, :7], 2, -1)

    q_src = object_state[..., None, 3:7]
    q_dst = object_goal_poses_buf[..., None, 3:7]
    t_src = object_state[..., None, 0:3]
    t_dst = object_goal_poses_buf[..., None, 0:3]
    object_keypoints = quat_rotate(q_src, kpt) + t_src
    goal_keypoints = quat_rotate(q_dst, kpt) + t_dst

    # Reward for object distance
    # object_keypoints = gen_keypoints(pose=object_state[:, 0:7], size=size)
    # goal_keypoints = gen_keypoints(
    #     pose=object_goal_poses_buf[:, 0:7], size=size)
    delta = object_keypoints - goal_keypoints
    dist = th.norm(delta, p=2, dim=-1)
    object_dist_reward = th.sum(
        (object_dist_weight[0] / (dist + epsilon[0])), -1)
    obj_reward = object_dist_reward
    # ic(th.median(obj_reward))

    # if use_energy_reward:
    #     total_reward = obj_reward - 0.01 * mean_energy
    # else:
    total_reward = obj_reward

    if use_inductive_reward:
        reach_reward = 0.03 / (curr_norms + epsilon[0])
        # ic(th.median(reach_reward))
        total_reward += reach_reward

    if action_gains is not None:
        last_action = th.norm(action_gains, 2, -1)
        # ic(th.median(last_action))
        total_reward -= 1.0 * last_action  # - 0.5 * residual

    return total_reward


def copy_obs(obs):
    if isinstance(obs, th.Tensor):
        return obs.detach().clone()
    elif isinstance(obs, dict):
        return {k: copy_obs(v) for k, v in obs.items()}
    else:
        raise ValueError('no idea.')


def log_potential(err: th.Tensor,
                  k_1: float = 0.3 * 0.5,
                  k_2: float = 7.1):
    # err = th.linalg.norm(src - dst, dim=-1)
    pot = -k_1 * th.log(k_2 * err + 1)
    return pot


def exp_potential(err: th.Tensor,
                  k_1: float = 0.3 * 0.5,
                  k_2: float = 7.1,
                  gamma: float = 0.99):
    # err = th.linalg.norm(src - dst, dim=-1)
    return k_1 * th.pow(gamma, k_2 * err)


def pot_rew(s0: Dict[str, th.Tensor],
            s1: Dict[str, th.Tensor],
            use_log_reward: bool = False,
            use_exp_reward: bool = False,
            k_1: Optional[float] = None,
            k_2: Optional[float] = None,
            gamma: float = 0.999
            ) -> th.Tensor:
    """ potential-based reward. """
    if 'nearest' in s1:
        err1 = s1['nearest']
        err0 = s0['nearest']
    else:
    # c0 = s0['hand_state'][..., :3]
        c0 = s0['tip_state'][..., :3]
        o0 = s0['object_state'][..., :3]
        # c1 = s1['hand_state'][..., :3]
        c1 = s1['tip_state'][..., :3]
        o1 = s1['object_state'][..., :3]
        err1 =th.linalg.norm(c1 - o1, dim=-1)
        err0 =th.linalg.norm(c0 - o0, dim=-1)
    if use_log_reward:
        p1 = log_potential(err1, k_1, k_2)  # neg. distance
        p0 = log_potential(err0, k_1, k_2)  # neg. distance
    elif use_exp_reward:
        p1 = exp_potential(err1, k_1, k_2, gamma)  # neg. distance
        p0 = exp_potential(err0, k_1, k_2, gamma)  # neg. distance
    else:
        if 'nearest' in s1:
            raise NotImplementedError("no poitential for nearest inducing")
        p1 = potential(o1, c1)  # neg. distance
        p0 = potential(o0, c0)  # neg. distance
    # return (p1 - p0)
    return (gamma * p1 - p0)
    # return (p1 - p0)


class PushWithArmTask(PushTask):
    @dataclass
    class Config(PushTask.Config):
        hand_obj_pot_coef: float = 1.0
        # NOTE: unused, only here for compatibility
        oob_fail_coef: Optional[float] = None
        max_pot_rew: float = 0.5
        # Potential reward magnitude, relative to
        # the object<->goal potential.
        rel_pot_rew_scale: float = 0.5
        regularize_coef: float = 0.
        crm_override: bool = False
        nearest_induce: bool =False

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
        self.regularize = env.robot.cfg.regularize

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
        if not cfg.crm_override:
            # ic(th.median(reward))
            pass
        info['task_reward'] = reward

        # Query relevant quantities.
        # FIXME: maybe try to avoid duplicating these
        # lookups and code...???
        goal = self.goal
        obj_ids = env.scene.cur_ids.long()
        obj_state = env.tensors['root'][obj_ids, :]

        # FIXME: shouldn't this be the tip??
        # body_indices = env.robot.ee_body_indices.long()
        body_indices = env.robot.ee_body_indices.long()
        hand_state = env.tensors['body'][body_indices]

        tip_indices = env.robot.tip_body_indices.long()
        tip_state = env.tensors['body'][tip_indices]

        if cfg.nearest_induce:
            left_finger_tool_indices = env.robot.left_finger_tool_indices.long()
            right_finger_tool_indices = env.robot.right_finger_tool_indices.long()

            left_finger_tip = env.tensors['body'][left_finger_tool_indices]
            right_finger_tip = env.tensors['body'][right_finger_tool_indices]
        else:
            left_finger_tip = tip_state
            right_finger_tip = tip_state

        state = {
            'goal': goal,
            'object_state': obj_state,
            'hand_state': hand_state,
            'tip_state': tip_state,
        }
        if cfg.crm_override:
            reward = 1000.0 * info['success'].float() + compute_card_reward(
                None,
                (0.02, 0.05),
                (0.02, 0.02),
                state['goal'],
                # env.robot.mean_energy,
                state['object_state'],
                state['hand_state'],
                True,
                # <DEFAULT>
                # False,
                env.scene.cur_bboxes,
                env.scene.cur_cloud if cfg.nearest_induce else None, 
                left_finger_tip,
                right_finger_tip)
        else:
            if not cfg.sparse_reward:
                if cfg.use_potential:
                    if cfg.nearest_induce:
                        cur_pcd = quat_rotate(obj_state[..., None, 3:7], env.scene.cur_cloud)\
                                            + obj_state[..., None, 0:3]

                        left_min, _ = th.min(th.norm(left_finger_tip[..., None, 0:3]-cur_pcd,
                                                p=2, dim=-1), dim = -1)
                        right_min, _ = th.min(th.norm(right_finger_tip[..., None, 0:3]-cur_pcd,
                                                p=2, dim=-1), dim = -1)
                        err = th.minimum(left_min, right_min)
                        state['nearest'] = err
                    # Add potential-based reward based on hand-object distance.
                    # (requires prev_state)
                    if self._prev_state is not None:
                        pot_reward = pot_rew(
                            self._prev_state,
                            state,
                            cfg.use_log_reward,
                            cfg.use_exp_reward,
                            cfg.k_1 * cfg.rel_pot_rew_scale,
                            cfg.k_2,
                            cfg.gamma
                        )
                        # In the `linear` case, pot_rew is not premultiplied
                        # with `hand_obj_pot_coef`.
                        if not (cfg.use_log_reward or cfg.use_exp_reward):
                            pot_reward *= cfg.hand_obj_pot_coef
                        reach_reward = (
                            pot_reward
                            .mul_(self._has_prev)
                            .clamp_(-cfg.max_pot_rew, +cfg.max_pot_rew)
                        )
                        # ic(th.median(reach_reward))
                        reward = reward + reach_reward

                else:
                    # CRM reward
                    hand_dist = th.norm(
                        obj_state[..., 0: 3] - hand_state[..., 0: 3],
                        p=2, dim=-1)
                    reach_reward = cfg.hand_obj_pot_coef * \
                        1 / (hand_dist + cfg.epsilon)
                    # ic(th.median(reach_reward))
                    reward = (reward + reach_reward)

            # FIXME: wrong
            info['reward/pot'] = reward

        if self.regularize == 'action':
            energy_penalty = cfg.regularize_coef * env.robot.energy
        elif self.regularize in ('energy', 'torque'):
            energy_penalty = cfg.regularize_coef * (
                env.robot.energy / env.cfg.action_period)
        else:
            energy_penalty = th.zeros_like(reward)
        info['reward/energy_penalty'] = energy_penalty
        reward -= energy_penalty

        # just naively subtract by avg base reward
        # reward -= 0.7

        self._has_prev.copy_(~done)
        self._prev_state = copy_obs(state)
        return (reward, done, info)
