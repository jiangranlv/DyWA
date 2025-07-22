#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import torch_utils

from typing import Tuple, Iterable, List, Optional, Dict, Union
import math
import nvtx
import pkg_resources
from dataclasses import dataclass
from util.config import ConfigBase
import numpy as np
import torch as th
import torch.nn.functional as F
import einops
from gym import spaces
from icecream import ic

from env.env.base import EnvBase
from env.robot.base import RobotBase
from env.robot.franka_util import (
    CartesianControlError,
    JointControlError,
    CheckHandObjectCollision,
    IKController,
    find_actor_indices,
    find_actor_handles,
    solve_ik_from_contact,
    CartesianImpedanceController)
from env.common import apply_domain_randomization
from util.math_util import matrix_from_quaternion
from util.torch_util import dcn


class Franka(RobotBase):
    @dataclass
    class Config(ConfigBase):
        # cube_dims: Tuple[float, float, float] = (0.08, 0.08, 0.08)
        # cube_dims: Tuple[float, float, float] = (0.045, 0.045, 0.045)
        # apply_mask: bool = False

        # What are reasonably 'random' joint initializations?
        # I can think of four:
        # Option#0 - home position
        # Option#1 - uniformly sample joint cfgs
        # Option#2 - discretely sample from "valid" cfgs
        # Option#3 - kinematics-based euclidean-ish sampling
        asset_root: str = pkg_resources.resource_filename('data', 'assets')
        robot_file: str = 'franka_description/robots/franka_panda.urdf'
        open_hand_robot_file: str = 'franka_description/robots/franka_panda_open_hand.urdf'

        # randomize_init_joints: bool = False
        # if `sample`, use one of the pre-sampled configurations
        init_type: str = 'home'

        # 1. joint position control
        # 2. joint velocity control
        # 3. cartesian position control; numerical IK
        # 4. cartesian position control; analytic IK (unsupported)
        # 5. cartesian impedance;
        # (jpos, jvel, cpos_n, cpos_a, CI)
        ee_frame: str = 'tool'
        ctrl_mode: str = 'CI'

        gain: str = 'variable'  # or fixed
        # or action magnitude or torque or None
        regularize: Optional[str] = 'energy'

        # (use isaacgym built in pos or vel controller or effort controller)
        use_effort: bool = True
        target_type: str = 'rel'  # or 'abs'
        # Numerical IK damping factor.
        damping: float = 0.05
        rot_type: str = 'axis_angle'
        KP_pos: float = 10.0  # 1.0
        # KP_pos: float = 30.0
        # KP_ori: float = 50.
        KP_ori: float = 100.0  # 1.0#0.3
        # KP_ori: float = 5.
        KD_pos: float = math.sqrt(KP_pos)  # * 2
        KD_ori: float = math.sqrt(KP_ori)
        # KD_pos: float = 0.0
        # KD_ori: float = 0.0
        VISCOUS_FRICTION: float = 0.0
        keepout_radius: float = 0.3

        max_pos: float = 0.1  # 0.1m / timestep(=0.04s)
        max_ori: float = 0.5  # 0.5rad (~30deg) / timestep(=0.04s)

        lin_vel_damping: float = 1.0
        ang_vel_damping: float = 5.0
        max_lin_vel: float = 2.0
        max_ang_vel: float = 6.28

        accumulate: bool = True
        lock_orn: bool = False

        ws_bound: Optional[List[List[float]]] = (
            [-0.3, -0.4, 0.4],  # min
            [+0.3, +0.4, 0.8]  # max
        )

        track_object: bool = False
        obj_margin: float = max_pos
        use_open_hand: bool = False
        compute_wrench: bool = False

        crm_override: bool = False
        hack_open_hand:bool = False
        base_height: str = 'ground'  # or 'table'
        clip_bound: bool = False

        init_ik_prob: float = 0.5
        disable_table_collision: bool = False

        add_tip_sensor: bool = False
        add_control_noise: bool = False
        control_noise_mag: float = 0.03

        box_min: Tuple[float, ...] = (-0.3, -0.4636, -0.2,
                                      -2.7432, -0.3335, 1.5269, -np.pi/2)  # 0.3816)
        box_max: Tuple[float, ...] = (0.3, 0.5432, 0.2,
                                      -1.5237, 0.3335, 2.5744, np.pi/2)  # 1.3914)

        # NOTE: friction parameters here are
        # applied to left-right fingers of panda FE gripper.
        default_hand_friction: float = 1.5
        default_body_friction: float = 0.1
        restitution: float = 0.5
        randomize_hand_friction: bool = True
        min_hand_friction: float = 1.0
        max_hand_friction: float = 1.2
        max_control_delay: Optional[int] = None

    def __init__(self, cfg: Config):
        self.cfg = cfg
        ic(cfg)
        self.n_bodies: int = None
        self.n_dofs: int = None
        self.dof_limits: Tuple[np.ndarray, np.ndarray] = None
        self.assets = {}
        self.valid_qs: Optional[th.Tensor] = None
        self.q_lo: th.Tensor = None
        self.q_hi: th.Tensor = None
        self._first = True
        self.q_home: th.Tensor = None

        # NOTE: decided to ALWAYS
        # assume that `robot_radius` references
        # `panda_hand`.
        # if cfg.ee_frame == 'tool':
        #     # round up from 0.1562...
        #     self.robot_radius: float = 0.16
        # elif cfg.ee_frame == 'hand':
        #     # round up from 0.1162...
        #     self.robot_radius: float = 0.12
        # else:
        #     raise ValueError(F'Unknown ee_frame = panda_{cfg.ee_frame}')
        self.robot_radius: float = 0.12

        self.pose_error: Union[CartesianControlError, JointControlError] = None
        if (not cfg.clip_bound) and cfg.track_object:
            raise ValueError(
                'clip_bound should be true to enable track_object')
        self._check_col = None
        self.delay_counter = None # counter for delay

    def setup(self, env: 'EnvBase'):
        # FIXME: introspection!
        cfg = self.cfg

        self.num_env = env.cfg.num_env
        self.device = th.device(env.cfg.th_device)

        accumulate = (
            self.cfg.accumulate and
            self.cfg.target_type == 'rel'
        )

        # Workspace boundary.
        # Ignored in case `track_object` is true.
        self.ws_bound = None
        # cfg.ws_bound = None
        if cfg.ws_bound is not None:
            self.ws_bound = th.as_tensor(cfg.ws_bound,
                                         dtype=th.float,
                                         device=self.device)
        if cfg.ctrl_mode in ['osc', 'CI']:
           self.pose_error = CartesianControlError(
               self.cfg.accumulate,
               target_shape=(self.num_env, 7),
               dtype=th.float,
               device=self.device,
               pos_bound=self.ws_bound if cfg.clip_bound else None)
        else:
            # FIXME: pose_error -> joint_error
            self.pose_error = JointControlError(
                self.cfg.accumulate,
                target_shape=(self.num_env, 7),
                dtype=th.float,
                device=self.device,
                pos_bound=self.ws_bound if cfg.clip_bound else None)

        self.eff_limits = th.as_tensor(self.eff_limits,
                                       dtype=th.float,
                                       device=self.device)

        if cfg.ctrl_mode in ['osc', 'CI']:
            self.controller = CartesianImpedanceController(
                cfg.KP_pos,
                cfg.KD_pos,
                cfg.KP_ori,
                cfg.KD_ori,
                self.eff_limits,
                device=self.device,
                OSC=cfg.ctrl_mode == 'osc')
        else:
            self.controller = IKController(
                cfg.KP_pos,
                self.eff_limits,
                num_env=env.num_env,
                device=self.device
            )

        if cfg.regularize is not None:
            self.energy = th.zeros(self.num_env, dtype=th.float,
                                   device=self.device)

        self.indices = th.as_tensor(
            find_actor_indices(env.gym, env.envs, 'robot'),
            dtype=th.int32, device=self.device)
        self.handles = find_actor_handles(env.gym, env.envs, 'robot')

        self.hand_ids = [
            env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                f'panda_{cfg.ee_frame}',
                gymapi.IndexDomain.DOMAIN_SIM
            ) for i in range(self.num_env)
        ]
        self.hand_ids = th.as_tensor(
            self.hand_ids,
            dtype=th.long,
            device=self.device
        )

        self._control = th.zeros((self.num_env, self.n_dofs),
                                 dtype=th.float, device=self.device)
        self._target = th.zeros((self.num_env, 7),
                                dtype=th.float, device=self.device)
        self.ee_wrench = th.zeros((self.num_env, 6),
                                  dtype=th.float, device=self.device)
        self.q_lo = th.as_tensor(self.dof_limits[0],
                                 dtype=th.float, device=self.device)
        self.q_hi = th.as_tensor(self.dof_limits[1],
                                 dtype=th.float, device=self.device)
        if self.cfg.init_type == 'sample':
            valid_qs = np.load('/tmp/qs.npy')
            self.valid_qs = th.as_tensor(valid_qs,
                                         dtype=th.float32, device=self.device)
            self.valid_qs = (self.valid_qs + np.pi) % (2 * np.pi) - np.pi
        elif self.cfg.init_type == 'home':
            if cfg.base_height == 'ground':
                self.q_home = th.tensor(
                    [0.0, 0.0, 0.0, -0.9425, 0.0, 1.1205, 0.0],
                    device=self.device)
            elif cfg.base_height == 'table':
                self.q_home = 0.5 * (self.q_lo + self.q_hi)
                # self.q_home =th.tensor(
                #     [-0.0122, -0.1095,  0.0562, -2.5737,\
                #      -0.0196,  2.4479,  0.7756],
                #      device=self.device
                # )
            else:
                raise KeyError(F'Unknown base_height = {cfg.base_height}')
        elif self.cfg.init_type == 'ik-test':
            import pickle
            with open('/input/pre_contacts.pkl', 'rb') as fp:
                contacts = pickle.load(fp)
            self.ik_configs = th.as_tensor([np.random.permutation(
                contacts[k])
                for k in env.scene.keys],
                device=env.device)
            self.cursor = th.zeros(self.ik_configs.shape[0], dtype=th.long,
                                   device=env.device)
            if cfg.base_height == 'ground':
                self.q_home = th.tensor(
                    [0.0, 0.0, 0.0, -0.9425, 0.0, 1.1205, 0.0],
                    device=self.device)
            elif cfg.base_height == 'table':
                self.q_home = 0.5 * (self.q_lo + self.q_hi)
        elif self.cfg.init_type == 'ik':
            if cfg.base_height == 'ground':
                self.q_home = th.tensor(
                    [0.0, 0.0, 0.0, -0.9425, 0.0, 1.1205, 0.0],
                    device=self.device)
            elif cfg.base_height == 'table':
                self.q_home = 0.5 * (self.q_lo + self.q_hi)
        elif self.cfg.init_type == 'easy':
            # TODO: make
            self.q_easy = th.as_tensor(
                [0.6634681, 0.42946462, 0.19089655, -2.15512631, -0.1472046,
                 2.57276871, 2.53247449, 0.012, 0.012],
                dtype=th.float, device=self.device)
        elif self.cfg.init_type == 'mvp0':
            self.q_mvp0 = th.as_tensor(
                [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035],
                dtype=th.float, device=self.device)
        elif self.cfg.init_type == 'box_sample':
            self.box_min = th.as_tensor(cfg.box_min,
                                        dtype=th.float, device=self.device)
            self.box_max = th.as_tensor(cfg.box_max,
                                        dtype=th.float, device=self.device)
        self._first = True

        # Acquire jacobian
        # FIXME: why here? well, it's because it's kinda
        # hard for env to know a prior what the name for the `robot`
        # should be...right? Hmmm....

        self._jacobian = gymtorch.wrap_tensor(
            env.gym.acquire_jacobian_tensor(
                env.sim, 'robot'))
        _mm = env.gym.acquire_mass_matrix_tensor(env.sim, "robot")

        EE_INDEX = self.franka_link_dict[f'panda_{cfg.ee_frame}']
        self.j_eef = self._jacobian[:, EE_INDEX - 1, :, :7]
        self.lmbda = th.eye(6, dtype=th.float,
                            device=self.device) * (self.cfg.damping**2)
        self.mm = gymtorch.wrap_tensor(_mm)
        self.mm = self.mm[:, :(EE_INDEX - 1), :(EE_INDEX - 1)]

        base_body_indices = []
        ee_body_indices = []
        tip_body_indices = []
        if env.task.cfg.nearest_induce:
            right_finger_tool_indices = []
            left_finger_tool_indices = []
            right_finger_indices = []
            left_finger_indices = []
        for i in range(self.num_env):
            base_idx = env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                'panda_link0',
                gymapi.DOMAIN_SIM
            )
            base_body_indices.append(base_idx)

            ee_idx = env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                # 'tool_tip',
                # 'wrist_3_link',
                'panda_hand',
                gymapi.DOMAIN_SIM
            )
            ee_body_indices.append(ee_idx)

            tip_idx = env.gym.find_actor_rigid_body_index(
                env.envs[i],
                self.handles[i],
                'panda_tool',
                gymapi.DOMAIN_SIM
            )
            tip_body_indices.append(tip_idx)
            if env.task.cfg.nearest_induce:
                left_finger_tool_idx = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    self.handles[i],
                    'panda_left_tool',
                    gymapi.DOMAIN_SIM
                )
                left_finger_tool_indices.append(left_finger_tool_idx)
                right_finger_tool_idx = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    self.handles[i],
                    'panda_right_tool',
                    gymapi.DOMAIN_SIM
                )
                right_finger_tool_indices.append(right_finger_tool_idx)

                left_finger_idx = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    self.handles[i],
                    'panda_leftfinger',
                    gymapi.DOMAIN_SIM
                )
                left_finger_indices.append(left_finger_idx)
                right_finger_idx = env.gym.find_actor_rigid_body_index(
                    env.envs[i],
                    self.handles[i],
                    'panda_rightfinger',
                    gymapi.DOMAIN_SIM
                )
                right_finger_indices.append(right_finger_idx)

        self.base_body_indices = th.as_tensor(
            base_body_indices,
            dtype=th.int32,
            device=self.device)

        self.ee_body_indices = th.as_tensor(
            ee_body_indices,
            dtype=th.int32,
            device=self.device)

        self.tip_body_indices = th.as_tensor(
            tip_body_indices,
            dtype=th.int32,
            device=self.device)

        self._hack_base_offset = th.as_tensor(
            [0.4, 0, -0.4],
            dtype=th.float32,
            device=self.device)
        if env.task.cfg.nearest_induce:
            self.left_finger_tool_indices = th.as_tensor(
                left_finger_tool_indices,
                dtype=th.int32,
                device=self.device
            )
            self.right_finger_tool_indices = th.as_tensor(
                right_finger_tool_indices,
                dtype=th.int32,
                device=self.device
            )
            self.left_finger_indices = th.as_tensor(
                left_finger_indices,
                dtype=th.int32,
                device=self.device
            )
            self.right_finger_indices = th.as_tensor(
                right_finger_indices,
                dtype=th.int32,
                device=self.device
            )
        self._check_col = CheckHandObjectCollision(env)
        self.cur_hand_friction = th.full((self.num_env,),
                                         cfg.default_hand_friction,
                                         dtype=th.float,
                                         device=self.device)
        if cfg.max_control_delay is not None:
            self.delay_counter = th.zeros((self.num_env,),
                                          dtype=th.long,
                                          device=self.device)

    def create_assets(self, gym, sim, counts: Optional[Dict[str, int]] = None):
        cfg = self.cfg
        asset_options = gymapi.AssetOptions()
        # asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        asset_options.enable_gyroscopic_forces = True

        asset_options.vhacd_enabled = False
        asset_options.convex_decomposition_from_submeshes = True

        # asset_options.override_inertia = True
        # asset_options.override_com = True
        # asset_options.linear_damping = cfg.lin_vel_damping
        # asset_options.angular_damping = cfg.ang_vel_damping
        # asset_options.max_linear_velocity = cfg.max_lin_vel
        # asset_options.max_angular_velocity = cfg.max_ang_vel

        if cfg.use_open_hand:
            robot_asset = gym.load_urdf(sim,
                                        cfg.asset_root,
                                        cfg.open_hand_robot_file,
                                        asset_options)
        elif cfg.crm_override:
            robot_asset_options = gymapi.AssetOptions()
            robot_asset_options.flip_visual_attachments = True
            robot_asset_options.fix_base_link = True
            robot_asset_options.collapse_fixed_joints = False
            robot_asset_options.disable_gravity = True
            robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
            robot_asset_options.thickness = 0.001
            if cfg.hack_open_hand:
                robot_file = "crm-panda/robots/franka_panda_fixed_finger_open.urdf"
            else:
                robot_file = "crm-panda/robots/franka_panda_fixed_finger.urdf"
            # if cfg.init_type =='ik':
            #     robot_file = cfg.robot_file.replace('franka_panda', 'franka_panda_no_coll')
            print(F'load {cfg.asset_root} - {robot_file}')
            robot_asset = gym.load_urdf(
                sim, cfg.asset_root,
                robot_file,
                robot_asset_options)
        else:
            if cfg.init_type == 'ik':
                # robot_file = cfg.robot_file.replace('franka_panda', 'franka_panda_no_coll')
                robot_file = cfg.robot_file
            else:
                robot_file = cfg.robot_file
            print(F'load {cfg.asset_root} - {cfg.robot_file}')
            robot_asset = gym.load_urdf(sim,
                                        cfg.asset_root,
                                        robot_file,
                                        asset_options)

        robot_props = gym.get_asset_rigid_shape_properties(robot_asset)
        left_finger_handle = gym.find_asset_rigid_body_index(
            robot_asset, "panda_leftfinger")
        right_finger_handle = gym.find_asset_rigid_body_index(
            robot_asset, "panda_rightfinger")
        hand_handle = gym.find_asset_rigid_body_index(
            robot_asset, "panda_hand")

        shape_indices = gym.get_asset_rigid_body_shape_indices(robot_asset)
        sil = shape_indices[left_finger_handle]
        sir = shape_indices[right_finger_handle]

        finger_shape_indices = (
            list(range(sil.start, sil.start + sil.count))
            + list(range(sir.start, sir.start + sir.count))
        )
        self.__finger_shape_indices = finger_shape_indices

        cnt = robot_asset
        for i, p in enumerate(robot_props):
            if i in finger_shape_indices:
                p.friction = cfg.default_hand_friction
            else:
                p.friction = cfg.default_body_friction
            p.restitution = cfg.restitution

            if i in [left_finger_handle, right_finger_handle,
                     hand_handle]:
                if i == hand_handle:
                    # Pass through the object
                    print('pass through the object')
                    p.filter |= 0b0110
                    #              ^----[arm]
                    #               ^---[object]
                    #                ^--[table]
                else:
                    # Hits the table and object
                    p.filter |= 0b0100
            else:
                # ARM
                if cfg.disable_table_collision:
                    # Pass through the table and the object
                    print('arm is 0b0111')
                    p.filter |= 0b0111
                else:
                    # Hits the table
                    p.filter |= 0b0100

        gym.set_asset_rigid_shape_properties(robot_asset, robot_props)
        # Cache some properties.
        self.n_bodies = gym.get_asset_rigid_body_count(robot_asset)
        self.n_dofs = gym.get_asset_dof_count(robot_asset)
        dof_props = gym.get_asset_dof_properties(robot_asset)
        dof_lo = []
        dof_hi = []
        vel_lo = []
        vel_hi = []
        eff_hi = []
        for i in range(self.n_dofs):
            dof_lo.append(dof_props['lower'][i])
            dof_hi.append(dof_props['upper'][i])
            vel_lo.append(-dof_props['velocity'][i])
            vel_hi.append(dof_props['velocity'][i])
            eff_hi.append(dof_props['effort'][i])
        self.dof_limits = (
            np.asarray(dof_lo), np.asarray(dof_hi)
        )
        self.eff_limits = np.asarray(eff_hi)

        if cfg.target_type == 'rel':
            if cfg.ctrl_mode == 'jpos':
                self.action_space = spaces.Box(*self.dof_limits)
            elif cfg.ctrl_mode == 'jvel':
                self.action_space = spaces.Box(
                    np.asarray(vel_lo), np.asarray(vel_hi))
            elif cfg.ctrl_mode in ('cpos_n', 'cpos_a', 'CI', 'osc'):
                if cfg.rot_type == 'axis_angle':
                    min_bound = [-cfg.max_pos, -cfg.max_pos, -cfg.max_pos,
                                 -cfg.max_ori, -cfg.max_ori, -cfg.max_ori]
                    max_bound = [cfg.max_pos, cfg.max_pos, cfg.max_pos,
                                 cfg.max_ori, cfg.max_ori, cfg.max_ori]
                    if cfg.lock_orn:
                        min_bound = min_bound[:3]
                        max_bound = max_bound[:3]
                    if cfg.gain == 'variable':
                        if cfg.ctrl_mode in ('CI', 'osc'):
                            min_bound += [10.] * 6
                            min_bound += [0.] * 6
                            max_bound += [cfg.KP_pos] * 3
                            max_bound += [cfg.KP_ori] * 3
                            max_bound += [2.] * 6
                        else:
                            min_bound += [10.] * 7
                            min_bound += [0.1] * 7
                            max_bound += [cfg.KP_pos] * 7
                            max_bound += [2.] * 7
                    self.action_space = spaces.Box(
                        np.asarray(min_bound),
                        np.asarray(max_bound),
                    )
                else:
                    min_bound = [-0.2, -0.2, -0.2, -1, -1, -1, -1]
                    max_bound = [+0.2, +0.2, +0.2, +1, +1, +1, +1]
                    if cfg.lock_orn:
                        min_bound = min_bound[:3]
                        max_bound = max_bound[:3]
                    self.action_space = spaces.Box(
                        np.asarray(min_bound),
                        np.asarray(max_bound),
                    )
        else:
            if cfg.ctrl_mode == 'jpos':
                self.action_space = spaces.Box(*self.dof_limits)
            else:
                raise ValueError(F'invalid target type = {cfg.target_type}')

        # ic(gym.get_asset_joint_names(robot_asset))
        # ic('body count', self.n_bodies)
        # ic('body names', gym.get_asset_rigid_body_names(robot_asset))
        # for i in range(self.n_dofs):
        #     ic(gym.get_asset_actuator_name(robot_asset, i))
        self.franka_link_dict = gym.get_asset_rigid_body_dict(robot_asset)
        self.hand_idx = gym.find_asset_rigid_body_index(
            robot_asset, "panda_hand")

        if False:
            # sensor with constraint
            sensor_props = gymapi.ForceSensorProperties()
            sensor_props.enable_forward_dynamics_forces = True
            sensor_props.enable_constraint_solver_forces = False
            sensor_props.use_world_frame = True
            sensor_pose = gymapi.Transform(gymapi.Vec3(0., 0.0, 0.0))
            sensor_idx = gym.create_asset_force_sensor(
                robot_asset, self.hand_idx, sensor_pose, sensor_props)

            # sensor with forward_dynamics_forces
            sensor_props.enable_forward_dynamics_forces = False
            sensor_props.enable_constraint_solver_forces = True
            sensor_idx = gym.create_asset_force_sensor(
                robot_asset, self.hand_idx, sensor_pose, sensor_props)

        if cfg.add_tip_sensor:
            sensor_options = gymapi.ForceSensorProperties()
            sensor_options.enable_forward_dynamics_forces = False  # for example gravity
            sensor_options.enable_constraint_solver_forces = True  # for example contacts
            # report forces in world frame (easier to get vertical components)
            sensor_options.use_world_frame = True

            _hand_index = gym.find_asset_rigid_body_index(robot_asset,
                                                          'panda_hand')
            _lf_index = gym.find_asset_rigid_body_index(robot_asset,
                                                        'panda_leftfinger')
            _rf_index = gym.find_asset_rigid_body_index(robot_asset,
                                                        'panda_rightfinger')
            gym.create_asset_force_sensor(
                robot_asset,
                _hand_index,
                gymapi.Transform(),
                sensor_options)
            gym.create_asset_force_sensor(
                robot_asset, _lf_index, gymapi.Transform(), sensor_options)
            gym.create_asset_force_sensor(
                robot_asset, _rf_index, gymapi.Transform(), sensor_options)

        if counts is not None:
            body_count = gym.get_asset_rigid_body_count(robot_asset)
            shape_count = gym.get_asset_rigid_shape_count(robot_asset)
            counts['body'] = body_count
            counts['shape'] = shape_count

        self.assets = {'robot': robot_asset}
        return dict(self.assets)

    def create_actors(self, gym, sim, env, env_id: int):
        cfg = self.cfg

        robot = gym.create_actor(env,
                                 self.assets['robot'],
                                 gymapi.Transform(),
                                 'robot',
                                 env_id,
                                 0b0100
                                 )

        if cfg.disable_table_collision:
            left_finger_handle = gym.find_actor_rigid_body_index(
                env,
                robot, "panda_leftfinger",
                gymapi.IndexDomain.DOMAIN_ACTOR)
            right_finger_handle = gym.find_actor_rigid_body_index(
                env,
                robot, "panda_rightfinger",
                gymapi.IndexDomain.DOMAIN_ACTOR)
            hand_handle = gym.find_actor_rigid_body_index(
                env,
                robot, "panda_hand",
                gymapi.IndexDomain.DOMAIN_ACTOR)

            index_range = gym.get_actor_rigid_body_shape_indices(
                env, robot)
            shape_props = gym.get_actor_rigid_shape_properties(
                env, robot)
            for i, p in enumerate(shape_props):
                # Disable table collision
                if cfg.disable_table_collision:
                    filter_mask = 0b0111
                else:
                    filter_mask = 0b0100

                if cfg.disable_table_collision:
                    for h in [left_finger_handle,
                              right_finger_handle,
                              hand_handle]:
                        r = index_range[h]
                        if r.start <= i and (i - r.start) < r.count:
                            if h == hand_handle:
                                filter_mask = 0b0111
                            else:
                                filter_mask = 0b0100

                p.filter = filter_mask
            gym.set_actor_rigid_shape_properties(
                env, robot, shape_props)

        # Configure the controller.
        robot_dof_props = gym.get_asset_dof_properties(
            self.assets['robot'])

        CTRL_MODES = {
            'jpos': gymapi.DOF_MODE_POS,
            'jvel': gymapi.DOF_MODE_VEL,
            'cpos_n': gymapi.DOF_MODE_POS
        }
        sysid_friction = [
            0.00174,
            0.01,
            7.5e-09,
            2.72e-07,
            0.39 * 0.2,
            0.12,
            0.9]
        sysid_damping = [2.12, 2.3, 1.29, 2.8, 0.194 * 1.5, 0.3, 0.46]
        sysid_armature = [0.192, 0.54, 0.128, 0.172, 0.15, 0.08, 0.06]
        if self.cfg.use_effort:
            ctrl_mode = gymapi.DOF_MODE_EFFORT
        else:
            ctrl_mode = CTRL_MODES[self.cfg.ctrl_mode]
        # sysid_damping = [2.12, 2.3, 1.29, 2.8, 0.194, 0.3, 0.46]
        # sysid_armature = [0.192, 0.54, 0.128, 0.172, 5.26e-09, 0.08, 0.06]

        for i in range(self.n_dofs):
            # robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            # Apparently this is what ACRONYM expects!
            robot_dof_props['driveMode'][i] = ctrl_mode
            if i < 7:
                if ctrl_mode == gymapi.DOF_MODE_POS:
                    robot_dof_props['stiffness'][i] = 300.0
                else:
                    robot_dof_props['stiffness'][i] = 0.0  # 5000
                    # robot_dof_props['damping'][i] = self.cfg.VISCOUS_FRICTION
                    robot_dof_props['friction'][i] = sysid_friction[i]
                    robot_dof_props['damping'][i] = sysid_damping[i]
                    robot_dof_props['armature'][i] = sysid_armature[i]
            else:
                robot_dof_props['damping'][i] = 1e2
                robot_dof_props['friction'][i] = 1e3
                robot_dof_props['armature'][i] = 1e2

        gym.set_actor_dof_properties(env,
                                     robot, robot_dof_props)

        return {'robot': robot}

    def reset(self, gym, sim, env, env_id) -> th.Tensor:
        """ Reset the _intrinsic states_ of the robot. """
        # if not self._first:
        #     return [], None
        cfg = self.cfg
        qpos = None
        qvel = None
        if env_id is None:
            env_id = th.arange(self.num_env,
                               dtype=th.int32,
                               device=self.device)

        indices = self.indices[env_id.long()]
        # indices = env_id
        # I = indices.long()
        I = env_id.long()

        if self._first:
            iii = indices.long()
            root_tensor = env.tensors['root']
            # zero out first
            root_tensor[iii, 0] = (
                env.scene.table_pos[..., 0]
                - 0.5 * env.scene.table_dims[..., 0]
                - cfg.keepout_radius
            )
            if cfg.base_height == 'ground':
                root_tensor[iii, 2] = 0.0
            elif cfg.base_height == 'table':
                root_tensor[iii, 2] = env.scene.table_dims[..., 2]
            else:
                raise KeyError(F'Unknown base_height = {cfg.base_height}')

            # unit quaternion
            root_tensor[iii, 6] = 1
        self._first = False

        if cfg.randomize_hand_friction:
            num_reset: int = len(I)
            self.cur_hand_friction[I] = (th.empty((num_reset,),
                                                  dtype=th.float,
                                                  device=self.device)
                                         .uniform_(
                                             cfg.min_hand_friction,
                                             cfg.max_hand_friction)
                                         )
            hand_friction = dcn(self.cur_hand_friction)
            for i in dcn(env_id):
                # NOTE: we randomize & the friction _first_,
                # commit to `cur_hand_friction`,
                # then apply that value during apply_domain_randomization.
                dr_params = apply_domain_randomization(
                    gym,
                    env.envs[i],
                    self.handles[i],
                    enable_friction=True,
                    min_friction=hand_friction[i],
                    max_friction=hand_friction[i],
                    target_shape_indices=self.__finger_shape_indices)

        dof_tensor = env.tensors['dof']
        if True:
            # Initialize the joint positions.
            if cfg.init_type == 'zero':
                dof_tensor[I, ..., 0] = 0.0  # pos (zero?)
            elif cfg.init_type == 'sample':
                sample_indices = th.randint(self.valid_qs.shape[0],
                                            size=(len(I),))
                dof_tensor[I, ..., 0] = self.valid_qs[sample_indices.long()]
            elif cfg.init_type == 'home':
                dof_tensor[I, :, 0] = self.q_home
            elif cfg.init_type == 'easy':
                dof_tensor[I, :, 0] = self.q_easy
            elif cfg.init_type == 'ik-test':
                index = th.randperm(len(I), device=self.device)
                num_reset_ik = int(len(I) * cfg.init_ik_prob)
                reset_w_ik, _ = I[index[:num_reset_ik]].sort()
                reset_wo_ik, _ = I[index[num_reset_ik:]].sort()
                obj_ids = env.scene.cur_ids.long()[reset_w_ik]
                n_obj_pos = self.ik_configs[reset_w_ik,
                                            self.cursor[reset_w_ik], :7]
                n_robot_pos = self.ik_configs[reset_w_ik,
                                              self.cursor[reset_w_ik], 7:]
                env.tensors['root'][obj_ids, :7] = n_obj_pos

                dof_tensor[reset_wo_ik, :, 0] = self.q_home
                dof_tensor[reset_w_ik, :, 0] = n_robot_pos

                self.cursor[reset_w_ik] = (self.cursor[reset_w_ik] + 1) % 1000

            elif cfg.init_type == 'ik':
                if True:
                    # I = env_id
                    # iii = robot indices
                    assert (env.scene.cfg.load_normal)
                    index = th.randperm(len(I), device=self.device)
                    num_reset_ik = int(len(I) * cfg.init_ik_prob)
                    reset_w_ik, _ = I[index[:num_reset_ik]].sort()
                    reset_wo_ik, _ = I[index[num_reset_ik:]].sort()

                    iii = indices.long()
                    iiii = iii[index[:num_reset_ik]]

                    root_tensor = env.tensors['root']
                    T_b = einops.repeat(
                        th.eye(4, device=env.device),
                        '... -> n ...', n=num_reset_ik).contiguous()
                    # print(root_tensor.shape, iii.shape, T_b.shape)
                    T_b[..., : 3, : 3] = matrix_from_quaternion(
                        root_tensor[iiii, 3: 7])
                    T_b[..., :3, 3] = root_tensor[iiii, :3]

                    # if len(reset_wo_ik)>0:
                    #     dof_tensor[reset_wo_ik, :, 0] = self.q_home
                    #     pass
                    dof_tensor[I, :, 0] = self.q_home[None]
                    if len(reset_w_ik) > 0:
                        q_ik, suc = solve_ik_from_contact(
                            env, reset_w_ik,
                            self.q_home, T_b,
                            self._check_col,
                            num_reset_ik,
                            # offset=0.023,
                            offset=0.02,
                            yaw_random=True,
                            # cone_max_theta=0.0,
                            cone_max_theta=math.radians(30.0),
                            # Oversample contacts
                            # and IK solutions by 32x.
                            query_multiplier=128
                        )
                        dof_tensor[reset_w_ik, :, 0] = q_ik
                        # which envs sucessfully got their IK solution?
                        self._ik_suc = reset_w_ik[suc]
                    else:
                        self._ik_suc = None
            elif cfg.init_type == 'box_sample':
                samples = th.rand(len(I), 7, device=self.device)
                diff = self.box_max - self.box_min
                noise = th.randn(len(I), 7, device=self.device) * 0.03
                qs = diff[None] * samples + self.box_min + noise
                dof_tensor[I, ..., 0] = qs

            # Currently, we always initialize velocity to zero.
            dof_tensor[I, ..., 1] = 0.0

        self._first = False

        if self.cfg.ctrl_mode == 'jvel':
            # Currently, we always initialize control velocity to zero as well.
            self._control[I] = 0.0
            qvel = self._control

        elif self.cfg.ctrl_mode == 'jpos':
            self._control[I] = dof_tensor[I, ..., 0]
            qpos = self._control
        elif self.cfg.ctrl_mode == 'cpos_n':
            self._control[I] = 0.
        elif self.cfg.ctrl_mode == 'CI':
            self._control[I] = 0.
        elif self.cfg.ctrl_mode == 'osc':
            self._control[I] = 0.
        else:
            raise KeyError('Unknown ctrl_mode')

        # FIXME: cannot reset with `dof_tensor`,
        # we need to wait until the hand_state becomes
        # valid again.
        # self.pose_error.reset(dof_tensor[..., :7], I)
        self.pose_error.clear(I)

        # reset ee_wrench to zero
        self.ee_wrench[I, :] = 0.
        if cfg.regularize is not None:
            self.energy[I] = 0.

        return indices, qpos, qvel

    @nvtx.annotate("step_controller")
    def step_controller(self, gym, sim, env):
        cfg = self.cfg
        # actions = self._target

        # TODO: think if this exactly what we want.
        # do we need _indexed() form in this case??
        indices = self.indices
        j_eef = self.j_eef
        # body_tensor = env.tensors['body'].clone().view(self.num_env, -1, 13)
        # ori = body_tensor[:,self.hand_idx, 3:7]
        # j_eef = get_analytic_jacobian(ori,self.j_eef,self.num_env,self.device)
        j_eef_T = th.transpose(j_eef, 1, 2)

        if cfg.track_object:
            assert (cfg.clip_bound)
            with nvtx.annotate("track_object"):
                # FIXME: hardcoded introspection !!!
                # obj_ids = env.scene.body_ids.long()
                # obj_pos = env.tensors['body'][obj_ids, ..., :3]
                obj_ids = env.scene.cur_ids.long()
                obj_pos = env.tensors['root'][obj_ids, :3]
                obj_rad = env.scene.cur_radii
                # NOTE: object-specific clipping bounds
                obj_bound = th.stack(
                    [obj_pos - obj_rad[..., None] - cfg.obj_margin,
                     obj_pos + obj_rad[..., None] + cfg.obj_margin],
                    dim=-2)
                self.pose_error.update_pos_bound(obj_bound)

        cache = {}
        if self.cfg.ctrl_mode == 'cpos_n':
            # Solve damped least squares.
            hand_state = env.tensors['body'][self.hand_ids]
            if self.delay_counter is not None:
                # update gain and joint target for envs where delay is finished
                update_indices = th.argwhere(self.delay_counter == 0).squeeze(-1)
                q_pos = env.tensors['dof'][..., 0]
                self.pose_error.update(
                    hand_state,
                    q_pos= q_pos,
                    action = None,
                    j_eef=j_eef,
                    relative=(cfg.target_type == 'rel'),
                    indices=update_indices,
                    update_pose=False,
                    recompute_orn=True
                )
                self.controller.update_gain(self.gains,
                                                indices=update_indices)
            b = self.pose_error(
                hand_state,
                env.tensors['dof'][..., 0],
                j_eef
            )
            self._control[..., :7] = self.controller(
                self.mm,
                env.tensors['dof'][..., 1], b
            )
        elif self.cfg.ctrl_mode == 'CI' or self.cfg.ctrl_mode == 'osc':
            with nvtx.annotate("OSC"):
                hand_state = env.tensors['body'][self.hand_ids]
                # [1] Relative
                if self.cfg.target_type == 'rel':
                    b = self.pose_error(hand_state[..., :7],
                                        axa=True)
                else:
                    raise ValueError('`abs` technically supported now')
                assert (self.cfg.use_effort)
                hand_vel = hand_state[..., 7:]
                self._control[..., :7] = self.controller(
                    self.mm,
                    j_eef,
                    hand_vel,
                    b,
                    cache=cache)

        elif self.cfg.ctrl_mode == 'jvel':
            self._control[...] = self._target
            # NOTE: assumes rank of action == 1
            if self.cfg.use_effort:
                jvel = env.tensors['dof'][..., 1]
                self._control[...] = self.cfg.KD_pos * (self._control - jvel)
                self.ee_wrench = self.j_eef
            else:
                gym.set_dof_velocity_target_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(self._control),
                    gymtorch.unwrap_tensor(indices),
                    len(indices)
                )
        elif self.cfg.ctrl_mode == 'jpos':
            self._control[...] = self._target
            if self.cfg.use_effort:
                jpos = env.tensors['dof'][..., 0]
                jvel = env.tensors['dof'][..., 1]
                self._control[...] = self.cfg.KP_pos * (
                    self._control - jpos) - self.cfg.KD_pos * (jvel)
                self.ee_wrench = self.j_eef
            else:
                gym.set_dof_position_target_tensor_indexed(
                    sim, gymtorch.unwrap_tensor(self._control),
                    gymtorch.unwrap_tensor(indices),
                    len(indices)
                )

        if self.cfg.use_effort:
            # NOTE: be default we zero-out
            # possible actuation forces against the fingers.
            self._control[:, 7:] = 0.

            if self.cfg.add_control_noise:
                samples = th.randn_like(self._control[:, :7])
                self._control[:, :7] += self.cfg.control_noise_mag * samples

            with nvtx.annotate("compute_wrench"):
                if self.cfg.compute_wrench:
                    # A = j_eef @ j_eef_T
                    # b = th.einsum('...ij,...j->...i', j_eef, self._control[:, :7])
                    if 'm_eef_inv' in cache:
                        m_eef_inv = cache['m_eef_inv']
                    else:
                        # mm_inv = th.inverse(self.mm)
                        m_eef_inv = j_eef @ th.linalg.solve(self.mm, j_eef_T)

                    # m_eef = th.inverse(m_eef_inv)
                    A = m_eef_inv
                    # b = (j_eef @ mm_inv @
                    #      self._control[:, :7].unsqueeze(-1)).squeeze(-1)
                    # b = j_eef @ th.linalg.solve(self.mm, self._control[:, :7])
                    b = th.einsum('...ij, ...j -> ...i', j_eef,
                                  th.linalg.solve(self.mm, self._control[:, :7]))
                    x = th.linalg.solve(A, b)
                    self.ee_wrench = x

            if cfg.regularize in ('energy', 'torque'):
                current_energy = (
                    self._control[..., :7]
                    if cfg.regularize == 'torque' else self._control
                    [..., :7] * env.tensors['dof'][..., :7, 1])
                self.energy += th.abs(current_energy).sum(dim=-1)

            if True:
                with nvtx.annotate("apply_effort"):
                    gym.set_dof_actuation_force_tensor_indexed(
                        sim, gymtorch.unwrap_tensor(self._control),
                        gymtorch.unwrap_tensor(indices),
                        len(indices)
                    )
            if self.delay_counter is not None:
                self.delay_counter -= 1

    @nvtx.annotate("Franka.apply_actions")
    def apply_actions(self, gym, sim, env, actions,
                      done=None):
        """ Set the actuation targets for the simulator. """
        if actions is None:
            print('actions is None.')
            return
        cfg = self.cfg
        if self.delay_counter is not None:
            self.delay_counter[:] = th.randint(1, self.cfg.max_control_delay,
                                            (self.num_env, ),
                                            device=self.device)
        if cfg.regularize == 'action':
            # scale = [1/cfg.max_pos, 1/cfg.max_pos, 1/cfg.max_pos,
            #          1/cfg.max_ori, 1/cfg.max_ori, 1/cfg.max_ori]
            # scale = th.as_tensor(scale, dtype=actions.dtype, device=actions.device)
            # # scale = (1/cfg.KP_pos) * th.ones_like(actions[..., -7:-14])
            self.energy[:] = th.linalg.norm(
                env.prev_action[..., -7: -14],
                ord=2, dim=-1)
            # self.energy[:] = th.linalg.norm(actions[..., :6] * scale, ord=2, dim=-1)
        elif cfg.regularize is not None:
            self.energy[:] = 0

        if cfg.ctrl_mode in ('CI', 'osc'):
            state = env.tensors['body'][self.hand_ids]
            self.pose_error.update(state, actions * (~done[..., None]),
                                   relative=(cfg.target_type == 'rel'))
            if cfg.lock_orn:
                self.pose_error.target[..., 3:7].fill_(0)
                self.pose_error.target[..., 4] = 1.0
            if cfg.gain == 'variable':
                self.controller.update_gain(actions[..., -12:])
        elif cfg.ctrl_mode in ('cpos_n', 'cpos_a'):
            q_pos = env.tensors['dof'][..., 0]
            state = env.tensors['body'][self.hand_ids, ..., :7]
            if self.delay_counter is not None:
                ## only update cartesian pose in here
                self.pose_error.pose_error.update(
                    state, actions, relative=(cfg.target_type == 'rel')
                )
                self.gains = actions[..., -14:].clone()
                #update gain and target for envs where delay is zero
                update_indices = th.argwhere(self.delay_counter == 0).squeeze(-1)
                self.pose_error.update(state, q_pos, actions, self.j_eef,
                                    relative=(cfg.target_type == 'rel'),
                                    indices=update_indices,
                                    update_pose=False)
                self.controller.update_gain(actions[..., -14:],
                                            indices=update_indices)
            else:
                self.pose_error.update(state, q_pos, actions, self.j_eef,
                                    relative=(cfg.target_type == 'rel'))
                if cfg.lock_orn:
                    self.pose_error.pose_error.target[..., 3:7].fill_(0)
                    self.pose_error.pose_error.target[..., 4] = 1.0
                if cfg.gain == 'variable':
                    self.controller.update_gain(actions[..., -14:])
        else:
            # ++ JOINT ++
            if cfg.target_type == 'rel':
                actions = actions + env.tensors['dof'][..., 0]
            if done is not None:
                self._target = actions * (~done[..., None])
            else:
                self._target = actions.copy()
