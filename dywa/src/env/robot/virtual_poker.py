#!/usr/bin/env python3

from dataclasses import dataclass
from util.config import ConfigBase
from typing import Optional, Dict

from isaacgym import gymtorch
from isaacgym import gymapi
import torch as th
import einops
import numpy as np

from env.env.base import EnvBase
from env.robot.base import RobotBase
from gym import spaces

import nvtx


class VirtualPoker(RobotBase):
    """
    Robot without a body that
    applies an arbitrary "poking" force to the
    target object at the specified position, direction, and magnitude.
    """

    @dataclass
    class Config(ConfigBase):
        at_com: bool = False
        direct_wrench: bool = True

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.forces: th.Tensor = None
        self.force_pos: th.Tensor = None
        self.torques: th.Tensor = None
        self.empty_id: th.Tensor = None

    @property
    def action_space(self):
        inf = float('inf')
        # NOTE: it's float but the first
        # element is considered as an index.
        # return spaces.Box(-inf, +inf, (7,))
        # return spaces.Box(-2.0, +2.0, (7,))

        # C, the scaling factor for torque magnitudes,
        # is determined based on:
        # np.cbrt(np.linalg.det(I)) / m
        C: float = 0.06889
        S_F: float = 10.0
        S_T: float = C * S_F
        lo = np.asarray([0, -S_F, -S_F, -S_F, -S_T, -S_T, -S_T])
        hi = np.asarray([np.inf, +S_F, +S_F, +S_F, +S_T, +S_T, +S_T])
        return spaces.Box(lo, hi)

    def create_assets(self, gym, sim, counts: Optional[Dict[str, int]] = None):
        if counts is not None:
            counts['body'] = 0
            counts['shape'] = 0
        return {}

    def create_actors(self, gym, sim, env, env_id: int):
        return {}

    def reset(self, gym, sim, env, env_id):
        return self.empty_id, None, None

    def setup(self, env: 'EnvBase'):
        """ extra domain-specific setup """
        self.num_envs = env.cfg.num_env
        self.device = env.cfg.th_device

        if self.num_envs > 0:
            self.num_bodies = env.gym.get_env_rigid_body_count(
                env.envs[0])
        else:
            self.num_bodies = 0

        self.forces = th.zeros((self.num_envs, self.num_bodies, 3),
                               device=self.device,
                               dtype=th.float32)

        self.force_pos = th.zeros((self.num_envs, self.num_bodies, 3),
                                  device=self.device, dtype=th.float32)
        if self.cfg.direct_wrench:
            self.torques = th.zeros((self.num_envs, self.num_bodies, 3),
                                    device=self.device,
                                    dtype=th.float32)
        self.empty_id: th.Tensor = th.empty(
            (0,), device=self.device,
            dtype=th.int32)

    def step_controller(self, gym, sim, env):
        # raise NotImplementedError()
        return

    @nvtx.annotate("VirtualPoker.apply_actions")
    def apply_actions(self, gym, sim, env,
                      actions: th.Tensor,
                      done=None):
        """
        Actions is an implicitly structured tensor
        with the following elements:
        A = (N, (1+3+3)) where A[i \\in n_{env}]
          = (body_index, force_vector, force_point)
        """
        if actions is None:
            return

        # TODO:
        # I don't really think introspection into
        # env.buffers() is the ideal solution...
        # if env.buffers['done'] is not None:
        #     actions = actions[~env.buffers['done']]
        # if len(actions) <= 0:
        #     return
        with nvtx.annotate("A"):
            # actions[env.buffers['done'], 1:] = 0
            actions[..., 1:] *= ~env.buffers['done'][..., None]

        with nvtx.annotate("B"):
            # Parse actions.
            body_indices = actions[..., 0]
            idx = body_indices.long().to(
                device=self.device)
            env_ids = th.arange(self.num_envs,
                                dtype=th.long,
                                device=self.device)

        with nvtx.annotate("C"):
            # indirect wrench from force + point
            if self.cfg.direct_wrench:
                with nvtx.annotate("D"):
                    force_action = actions[..., 1:4]
                    torque_action = actions[..., 4:7]

                with nvtx.annotate("E"):
                    forces = self.forces
                    forces.fill_(0)
                    torques = self.torques
                    torques.fill_(0)

                with nvtx.annotate("F"):
                    forces[env_ids, idx] = (
                        force_action
                    )
                    torques[env_ids, idx] = (
                        torque_action
                    )

                with nvtx.annotate("G"):
                    # Directly apply wrench
                    out = gym.apply_rigid_body_force_tensors(
                        sim,
                        gymtorch.unwrap_tensor(forces),
                        gymtorch.unwrap_tensor(torques),
                        gymapi.ENV_SPACE
                    )
                return out
            else:
                force_vector = actions[..., 1:4]
                force_point = actions[..., 4:7]

                # Format to gym-expected forms.
                # N = len(actions)

                forces = self.forces
                force_positions = self.force_pos

                forces.fill_(0)
                forces[env_ids, idx] = force_vector.to(self.device)
                force_positions.fill_(0)
                force_positions[:, idx] = force_point.to(self.device)
                # print('forces', forces)

                # Apply actions.
                if self.cfg.at_com:
                    return gym.apply_rigid_body_force_tensors(
                        sim,
                        gymtorch.unwrap_tensor(forces),
                        None,
                        gymapi.ENV_SPACE  # ??
                    )
                else:
                    return gym.apply_rigid_body_force_at_pos_tensors(
                        sim,
                        gymtorch.unwrap_tensor(forces),
                        gymtorch.unwrap_tensor(force_positions),
                        gymapi.ENV_SPACE  # ??
                    )
