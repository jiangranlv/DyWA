#!/usr/bin/env python3

from abc import ABC, abstractmethod

import time
from typing import Tuple, Iterable, Optional, Union
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
from env.scene.tabletop_with_multi_object_scene import TableTopWithMultiObjectScene
from env.robot.base import RobotBase

from env.robot.virtual_poker import VirtualPoker
from env.robot.fe_gripper import FEGripper
from env.robot.cube_poker import CubePoker
from env.robot.object_poker import ObjectPoker
from env.robot.ur5_fe import UR5FE, matrix_from_quaternion
from env.robot.franka import Franka
from env.robot.push_stick import PushStick

from env.env.help.with_nvdr_camera import WithNvdrCamera

# from env.task.null_task import NullTask
from env.task.push_task import PushTask
from env.task.push_with_cube_task import PushWithCubeTask
from env.task.push_with_hand_task import PushWithHandTask
from env.env.base import EnvBase
from env.common import (
    get_default_sim_params,
    create_camera
)

from env.util import get_mass_properties
from util.vis.flow import flow_image
from util.torch_util import dcn
from util.config import recursive_replace_map

from icecream import ic
import nvtx


class PushEnv(EnvBase):
    """
    Tabletop environment.
    """

    @dataclass
    class Config(EnvBase.Config):
        # SCENES
        single_object_scene: TableTopWithObjectScene.Config = TableTopWithObjectScene.Config()
        multi_object_scene: TableTopWithMultiObjectScene.Config = TableTopWithMultiObjectScene.Config()
        which_scene: str = 'single_object'

        # ROBOTS
        virtual_poker: VirtualPoker.Config = VirtualPoker.Config()
        cube_poker: CubePoker.Config = CubePoker.Config()
        ur5_fe: UR5FE.Config = UR5FE.Config()
        object_poker: ObjectPoker.Config = ObjectPoker.Config()
        franka: Franka.Config = Franka.Config()
        fe_gripper: FEGripper.Config = FEGripper.Config()
        push_stick: PushStick.Config = PushStick.Config()
        which_robot: str = 'virtual'

        # TASKS
        # task: NullTask.Config = NullTask.Config()
        # task: PushWithCubeTask.Config = PushWithCubeTask.Config()
        task: PushWithHandTask.Config = PushWithHandTask.Config()

        draw_task_goal: bool = False
        draw_obj_pos_2d: bool = False
        draw_obj_path: bool = False
        draw_force: bool = False
        draw_obj_vel: bool = False

        render: bool = False

    def __init__(self, cfg: Config,
                 writer=None,
                 task_cls=None):
        if cfg.which_scene == 'multi_object':
            scene = TableTopWithMultiObjectScene(cfg.multi_object_scene)
            self.scene_cfg = cfg.multi_object_scene
        else:
            scene = TableTopWithObjectScene(cfg.single_object_scene)
            self.scene_cfg = cfg.single_object_scene

        self.robot_cfg = None
        if cfg.which_robot == 'cube':
            self.robot_cfg = cfg.cube_poker
        elif cfg.which_robot == 'object':
            self.robot_cfg = cfg.object_poker
        elif cfg.which_robot == 'ur5_fe':
            self.robot_cfg = cfg.ur5_fe
        elif cfg.which_robot == 'franka':
            self.robot_cfg = cfg.franka
        elif cfg.which_robot == 'virtual':
            self.robot_cfg = cfg.virtual_poker
        elif cfg.which_robot == 'fe_gripper':
            self.robot_cfg = cfg.fe_gripper
        elif cfg.which_robot == 'push_stick':
            self.robot_cfg = cfg.push_stick
        else:
            raise KeyError(F'Unknown robot = {cfg.which_robot}')

        if cfg.which_robot == 'cube':
            robot = CubePoker(self.robot_cfg)
        elif cfg.which_robot == 'object':
            robot = ObjectPoker(self.robot_cfg)
        elif cfg.which_robot == 'ur5_fe':
            robot = UR5FE(self.robot_cfg)
        elif cfg.which_robot == 'franka':
            robot = Franka(self.robot_cfg)
        elif cfg.which_robot == 'fe_gripper':
            robot = FEGripper(self.robot_cfg)
        elif cfg.which_robot == 'push_stick':
            robot = PushStick(self.robot_cfg)
        else:
            robot = VirtualPoker(self.robot_cfg)
        # robot = FEGripper(self.robot_cfg)

        # FIXME: brittle multiplexing between
        # pushtask and PushWithCubeTask!!!
        if task_cls is None:
            if cfg.which_robot == 'cube':
                task = PushWithCubeTask(cfg.task, writer=writer)
            elif cfg.which_robot == 'fe_gripper':
                task = PushWithHandTask(cfg.task, writer=writer)
            else:
                task = PushTask(cfg.task, writer=writer)
        else:
            task = task_cls(cfg.task, writer=writer)
        super().__init__(cfg,
                         scene, robot, task)

        self.configure()
        # self.setup()
        # self.gym.prepare_sim(self.sim)
        # self.refresh_tensors()
        # self.reset()

        self.__path = []
        self._actions_path = []

    @property
    def action_space(self):
        return self.robot.action_space

    @property
    def observation_space(self):
        return None

    def _draw_obj_pos_2d(self):
        cfg = self.cfg
        gym = self.gym

        obj_ids = self.scene.cur_ids.to(
            self.cfg.th_device)
        # pos_2d = dcn(self.tensors['root'][
        #     obj_ids.long(), :2])
        pos_3d = dcn(self.tensors['root'][
            obj_ids.long(), :3])

        for i in range(cfg.num_env):
            verts = np.zeros(shape=(2, 3), dtype=np.float32)
            verts[:, :3] = pos_3d[i]
            verts[0, 2] += - 0.2
            verts[1, 2] += + 0.2

            gym.add_lines(self.viewer,
                          self.envs[i],
                          1,
                          verts[None],
                          np.asarray([0, 0, 1], dtype=np.float32))

    def _draw_obj_path(self):
        cfg = self.cfg
        gym = self.gym

        path = [np.stack(p) if len(p) > 1 else
                np.empty((0, 3), dtype=np.float)
                for p in self.__path]
        # actions_path = np.stack(self._actions_path, axis=0)  # NxTx3
        actions_path = [np.stack(p) if len(p) > 1 else
                        np.empty((0, 6), dtype=np.float)
                        for p in self._actions_path]

        # Get mass related properties
        # with which to scale the force vectors
        # I guess? :P:P:P
        masses = np.zeros(cfg.num_env)
        for i in range(cfg.num_env):
            prop = gym.get_actor_rigid_body_properties(
                self.envs[i],
                self.scene.cur_handles[i].item()
            )
            masses[i] = prop[0].mass

        scale_vector = cfg.dt / masses  # N
        # force_vector = actions_path[..., 1:4]  # NXTX3
        # force_vector = scale_vector[..., None, None] * dcn(force_vector)

        for i in range(cfg.num_env):
            points = path[i]
            force_vector = scale_vector[i] * dcn(
                actions_path[i][..., 1:4])

            if len(points) <= 0:
                continue
            if len(force_vector) <= 0:
                continue

            lines = np.stack(
                [points[:-1], points[1:]],
                axis=1)  # Nx2x3

            colors = np.zeros_like(lines[:, 0])
            colors[..., 2] = 1.0  # blue
            gym.add_lines(self.viewer,
                          self.envs[i],
                          len(lines), lines,
                          colors)

            force_lines = np.stack([
                points[:len(force_vector)],
                points[:len(force_vector)] + force_vector], axis=1)
            colors = np.zeros_like(force_lines[:, 0])
            colors[..., 1] = 1.0  # green
            gym.add_lines(self.viewer,
                          self.envs[i],
                          len(force_lines),
                          force_lines.astype(np.float32),
                          colors.astype(np.float32))

    def _draw_task_goal(self):
        # Just for debugging
        cfg = self.cfg
        gym = self.gym
        if self.viewer is None:
            return

        goals = dcn(self.task.goal)

        circle_verts = np.zeros(shape=(1, 128, 3),
                                dtype=np.float32)
        angles = np.linspace(-np.pi, np.pi, num=128)[None, :]
        circle_verts[..., 0] = np.cos(angles)
        circle_verts[..., 1] = np.sin(angles)
        circle_verts = circle_verts * dcn(
            self.task.goal_radius_samples)[:, None, None]

        for i in range(cfg.num_env):
            verts = np.zeros(shape=(2, 3), dtype=np.float32)
            verts[:, :3] = goals[i, ..., :3]
            verts[0, 2] -= 0.3
            verts[1, 2] += 0.3

            gym.add_lines(self.viewer,
                          self.envs[i],
                          1,
                          verts[None],
                          np.asarray([1, 0, 0], dtype=np.float32))

            # circle indicating goal region
            points = np.zeros(shape=(128, 3), dtype=np.float32)
            points[:, :3] = goals[None, i, ..., :3] + circle_verts[i]
            verts = np.stack(
                [points[:-1], points[1:]],
                axis=-2)
            colors = np.zeros_like(verts[..., 0, :])
            colors[..., 0] = 1.0
            gym.add_lines(self.viewer,
                          self.envs[i],
                          verts.shape[0],
                          verts,
                          colors
                          )

    def _draw_obj_vel(self):
        scale: float = 4.0
        cfg = self.cfg
        gym = self.gym

        obj_ids = self.scene.cur_ids.long()
        obj_state = self.tensors['root'][obj_ids, :]
        obj_pos = dcn(obj_state[..., 0:3])
        obj_vel = dcn(obj_state[..., 7:10])

        goal = dcn(self.task.goal)
        pos_err = goal[..., :3] - obj_pos

        for i in range(cfg.num_env):
            # yellow = object velocity
            verts = np.zeros(shape=(2, 3), dtype=np.float32)
            verts[0] = obj_pos[i]
            # verts[1] = obj_pos[i] + scale * cfg.dt * obj_vel[i]
            verts[1] = obj_pos[i] + pos_err[i]

            gym.add_lines(self.viewer,
                          self.envs[i],
                          1,
                          verts[None],
                          np.asarray([1, 1, 0], dtype=np.float32)
                          )

    def _draw_force(self):
        if self._actions is None:
            return
        if not hasattr(self.robot_cfg, 'at_com'):
            return
        cfg = self.cfg
        gym = self.gym

        actions = self._actions
        body_indices = actions[..., 0]
        force_vector = actions[..., -6:-3]

        # Get mass related properties
        # with which to scale the force vectors
        # I guess? :P:P:P
        masses = np.zeros(cfg.num_env)
        for i in range(cfg.num_env):
            prop = gym.get_actor_rigid_body_properties(
                self.envs[i],
                self.scene.cur_handles[i].item()
            )
            masses[i] = prop[0].mass

        if self.robot_cfg.at_com or self.robot_cfg.direct_wrench:
            # FIXME: bit of a hack, might not always work
            T = self.tensors['root'].reshape(cfg.num_env, -1, 13)
            pos_3d = T[th.arange(cfg.num_env).long(),
                       body_indices.long(), :3]
            force_point = dcn(pos_3d)
        else:
            force_point = actions[..., 4:7]
            force_point = dcn(force_point)

        scale_vector = cfg.dt / masses
        force_vector = scale_vector[..., None] * dcn(force_vector)
        measured_force_vector = scale_vector[..., None] * dcn(
            self.tensors['force_sensor'][..., :3])

        for i in range(cfg.num_env):
            # green = applied(desired) force
            verts = np.zeros(shape=(2, 3), dtype=np.float32)
            verts[0] = force_point[i]
            verts[1] = force_point[i] + 1.0 * force_vector[i]
            gym.add_lines(self.viewer,
                          self.envs[i],
                          1,
                          verts[None],
                          np.asarray([0, 1, 0], dtype=np.float32)
                          )

            # magenta = measured force
            verts[0] = force_point[i]
            verts[1] = force_point[i] + 1.0 * measured_force_vector[i]
            gym.add_lines(self.viewer,
                          self.envs[i],
                          1,
                          verts[None],
                          np.asarray([1, 0, 1], dtype=np.float32)
                          )

    @nvtx.annotate('PushEnv.step()', color="cyan")
    def step(self, *args, **kwds):
        with nvtx.annotate('PushEnv.step.A()'):
            out = super().step(*args, **kwds)

        with nvtx.annotate('PushEnv.step.B()'):
            # FIXME: super fragile fix
            if (self.viewer is not None) or hasattr(self.gym, 'lines'):
                self.gym.clear_lines(self.viewer)

            if self.viewer is not None:
                if self.cfg.draw_task_goal:
                   self._draw_task_goal()
                if self.cfg.draw_obj_pos_2d:
                    self._draw_obj_pos_2d()
                if self.cfg.draw_obj_path:
                    obj_ids = self.scene.cur_ids.to(self.cfg.th_device)
                    pos = dcn(self.tensors['root'][obj_ids.long(), :3])
                    for i in range(self.cfg.num_env):
                        self.__path[i].append(pos[i])
                    self._draw_obj_path()
                if self.cfg.draw_force:
                    self._draw_force()
                if self.cfg.draw_obj_vel:
                    self._draw_obj_vel()

        with nvtx.annotate('PushEnv.step.C()'):
            return out

    def create_assets(self):
        cfg = self.cfg
        gym = self.gym
        sim = self.sim

        outputs = super().create_assets()

        if False:
            assets = {}
            obj_assets = {}
            object_files = list(
                Path('/opt/datasets/acronym/urdf/').glob('*.urdf'))
            num_load: int = 32
            for filename in tqdm(object_files, desc='load_object_assets'):
                if num_load <= 0:
                    break
                num_load -= 1
                asset_options = gymapi.AssetOptions()
                asset_options.disable_gravity = False
                asset_options.fix_base_link = False
                asset_options.vhacd_enabled = False
                asset_options.thickness = 0.0001  # ????
                asset_options.convex_decomposition_from_submeshes = True

                # asset_options.armature = 0.01
                # asset_options.thickness = 0.0
                # asset_options.linear_damping = 1.0
                # asset_options.angular_damping = 0.0

                # Somewhat non-standard choices...
                # asset_options.density = 0.1  # kg/m^3
                # asset_options.density = 1.0  # kg/m^3
                asset_options.density = 1.0  # kg/m^3
                asset_options.override_com = False
                asset_options.override_inertia = False
                obj_asset = gym.load_urdf(sim,
                                          str(Path(filename).parent),
                                          str(Path(filename).name),
                                          asset_options)
                obj_assets[filename] = obj_asset
            self._obj_keys = list(obj_assets.keys())
            assets['meta'] = obj_assets

            outputs.update(assets)
        return outputs

    def create_envs(self):
        # Create all the default envs
        outputs = super().create_envs()

        cfg = self.cfg
        gym = self.gym
        sim = self.sim
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        gym.add_ground(sim, plane_params)

        if False:
            envs = outputs[0]
            # Create a meta-env to hold objects?
            cfg = self.cfg
            gym = self.gym
            sim = self.sim

            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            gym.add_ground(sim, plane_params)

            meta_env = self.gym.create_env(
                self.sim, gymapi.Vec3(*2 * cfg.env_bound_lower),
                gymapi.Vec3(*2 * cfg.env_bound_upper),
                int(np.sqrt(cfg.num_env)))
            for obj_id in range(cfg.num_env):
                k = np.random.choice(self._obj_keys)
                pose = gymapi.Transform()
                pose.p = (
                    gym.get_env_origin(envs[obj_id % cfg.num_env]) -
                    gym.get_env_origin(meta_env)
                )
                pose.p.z += 1
                # pose.p.x = obj_id * 10
                object_actor = gym.create_actor(
                    meta_env,
                    self.assets['meta'][k],
                    pose,
                    F'object-{obj_id}',
                    # cfg.num_env,
                    -1,
                    0)

        return outputs

    def setup(self):
        """
        * load assets.
        * allocate buffers related to {scene, robot, task}.
        """
        return super().setup()

    @nvtx.annotate('PushEnv.reset_indexed()', color="orange")
    def reset_indexed(self, indices: Optional[Iterable[int]] = None):
        cfg = self.cfg
        if cfg.draw_obj_path:
            if indices is not None:
                for i in indices:
                    self.__path[i] = []
                    self._actions_path[i] = []
            else:
                self.__path = [[] for _ in range(cfg.num_env)]
                self._actions_path = [[] for _ in range(cfg.num_env)]
        return super().reset_indexed(indices)

    def reset(self):
        return super().reset()

    def apply_actions(self, actions):
        cfg = self.cfg
        if self.cfg.draw_force:
            self._actions = actions

            if self.cfg.draw_obj_path:
                a = dcn(actions)
                for i in range(cfg.num_env):
                    self._actions_path[i].append(a[i])

        return self.robot.apply_actions(
            self.gym, self.sim, self,
            actions, done=self.buffers['done'])

    def compute_feedback(self, *args, **kwds):
        return super().compute_feedback(*args, **kwds)

    @nvtx.annotate('PushEnv.compute_observations()', color="blue")
    def compute_observations(self):
        # self.gym.fetch_results(self.sim, True)

        if False:
            if self.cfg.render:
                # ^^ TODO: needed?
                self.gym.step_graphics(self.sim)
                # ^^ TODO: needed?
                self.gym.render_all_camera_sensors(self.sim)

                # Access and convert all image-related tensors.
                # TODO: the image access and related rendering utilities
                # should only be conditioned on image-based environments.
                self.gym.start_access_image_tensors(self.sim)
                color_tensors = [s['tensors']['color']
                                 for s in self.sensors['scene']]
                depth_tensors = [s['tensors']['depth']
                                 for s in self.sensors['scene']]
                label_tensors = [s['tensors']['label']
                                 for s in self.sensors['scene']]
                th.stack(color_tensors, out=self.buffers['color'])
                th.stack(depth_tensors, out=self.buffers['depth'])
                th.stack(label_tensors, out=self.buffers['label'])
                self.gym.end_access_image_tensors(self.sim)

        # return {'color': self.buffers['color'],
        #         'depth': self.buffers['depth'],
        #         'label': self.buffers['label']}
        return {}


class RandomGripperAgent:
    def __init__(self, num_env: int, device: str):
        self.num_env = num_env
        self.device = device

    def __call__(self):
        """ random gripper velocity... ? """
        vel_targets = 0.1 * th.rand(size=(self.num_env, 2),
                                    device=self.device) - 0.05
        return vel_targets


class RandomVirtualPushAgent:
    def __init__(self, num_env: int, body_indices: Iterable[int]):
        self.num_env = num_env
        self.body_indices = th.as_tensor(list(body_indices))

    def __call__(self):
        # body_indices = actions[..., 0]
        # force_vector = actions[..., 1:4]
        # force_point = actions[..., 4:7]
        force_vector = 100.0 * th.randn((self.num_env, 3))
        # TODO: sample a feasible contact point
        # on the target object(how?)
        force_point = th.randn((self.num_env, 3))
        actions = th.cat([
            self.body_indices[..., None],
            force_vector,
            force_point], dim=-1)
        return actions


class RandomVirtualPushAgentV2:
    def __init__(self, num_env: int, env: EnvBase):
        self.num_env = num_env
        self.env = env

    def __call__(self):
        # body_indices = self.env.scene.cur_ids
        body_indices = self.env.scene.body_ids.cuda()
        force_vector = 2.0 * th.randn((self.num_env, 3)).cuda()
        force_point = th.randn((self.num_env, 3)).cuda()
        actions = th.cat([
            body_indices[..., None],
            force_vector,
            force_point], dim=-1)
        return actions


class MAV2:
    def __init__(self, num_env: int, env: EnvBase):
        self.num_env = num_env
        self.env = env

    def __call__(self):
        # CONSTATNTS
        DT: float = self.env.cfg.dt
        MASS: float = 0.170
        MAX_WRENCH: float = +2.0
        A: float = MAX_WRENCH / MASS

        # QUERIES
        env = self.env
        body_ids = env.scene.body_ids.cuda()
        object_ids = env.scene.cur_ids
        table_ids = env.scene.table_body_ids
        s = env.tensors['root'][object_ids.long()]  # N x 3
        goal = env.task.goal

        goal_2d = goal[..., :2]
        obj_pos = s[..., :3]
        obj_vel = s[..., 7:13]

        # Initial velocity
        x1 = th.zeros_like(s[..., 7:13])  # NX6
        x1[..., :2] = goal_2d - obj_pos[..., :2]

        v0 = obj_vel
        q = th.sqrt(A * th.abs(x1) + v0**2 / 2)
        k1 = -(v0 - q) / A
        k2 = q / A
        wrench = th.sign(x1) * th.where(
            k1 > 0,
            +MAX_WRENCH,
            -MAX_WRENCH)
        actions = th.cat(
            [
                body_ids[..., None],
                wrench], dim=-1)
        return actions


class ManualAgent:
    def __init__(self, num_env: int, env: EnvBase):
        self.num_env = num_env
        self.env = env

    def __call__(self):
        env = self.env

        body_ids = env.scene.body_ids.cuda()
        object_ids = env.scene.cur_ids
        table_ids = env.scene.table_body_ids
        pos_3d = env.tensors['root'][object_ids.long(), :3]  # N x 3
        pos_2d = pos_3d[:, :2]

        pos_error = env.task.goal[..., :2] - pos_2d
        distance = th.linalg.norm(pos_error, dim=-1)

        force_vector = th.zeros((self.num_env, 3),
                                dtype=th.float32).cuda()
        # force_vector.normal_()
        force_vector[:, :2] = 0.1 * (pos_error / distance[..., None])

        if env.robot_cfg.direct_wrench:
            # interpreted as `torque`
            force_point = 0 * pos_3d
        else:
            # interpreted as force appliation point
            force_point = pos_3d

        actions = th.cat([
            body_ids[..., None],
            # object_ids.cuda()[...,None],
            force_vector,
            force_point], dim=-1)
        return actions


def main():
    from env.env.wrap.with_ig_camera import WithCameraWrapper
    GID: int = 0
    QQ: int = 2
    num_env: int = QQ * QQ
    cfg = PushEnv.Config(num_env=num_env,
                         use_viewer=False,
                         draw_task_goal=False,
                         draw_obj_pos_2d=False,
                         draw_obj_path=False,
                         draw_force=False,
                         which_robot='ur5_fe',
                         graphics_device_id=GID)
    cfg = recursive_replace_map(cfg, {
        'task.timeout': 128,
        # just for testing DR
        'scene.use_dr_on_setup': True,
        'ur5_fe.ctrl_mode': 'jvel',
        'ur5_fe.target_type': 'abs',
    })
    env = PushEnv(cfg)
    if True:
        env = WithCameraWrapper(
            recursive_replace_map(WithCameraWrapper.Config(), {
                'camera.num_env': num_env,
                # 'camera.device': cfg.th_device,
                'camera.device': F'cuda:{GID}',
                'camera.pos': (-0.8, -0.2, 1.0),
                'camera.width': 256,
                'camera.height': 256,
                'camera.use_color': True,
                'camera.use_depth': True,
                'camera.use_flow': True,
            }),
            env)
    env.setup()
    env.gym.prepare_sim(env.sim)
    env.refresh_tensors()
    env.reset()

    obs = env.reset()

    if True:
        cam = WithNvdrCamera(num_env,
                             device=F'cuda:{GID}')
        object_body_id = env.scene.body_ids  # DOMAIN_ENV
        robot_body_ids = env.robot.link_body_indices  # DOMAIN_ENV
        ic(object_body_id)
        ic(robot_body_ids)
        body_tensors = env.tensors['body']
        BT = body_tensors.reshape(num_env, -1, 13)
        body_poses_7dof = BT[:, 1:16, ..., :7]
        body_poses_4x4 = th.zeros(
            body_poses_7dof.shape[:-1] + (4, 4),
            dtype=th.float32,
            device=F'cuda:{GID}')
        body_poses_4x4[..., 3, 3] = 1
        body_poses_4x4[..., :3, 3] = body_poses_7dof[..., :3]
        matrix_from_quaternion(body_poses_7dof[..., 3:7],
                               body_poses_4x4[..., :3, :3])
        out = cam(body_poses_4x4)

    if False:
        color = obs['images']['color'].detach().cpu().numpy()
        mosaic = einops.rearrange(color, '(bh bw) h w c -> (bh h) (bw w) c',
                                  bh=QQ, bw=QQ)
        cv2.namedWindow('mosaic', cv2.WINDOW_NORMAL)
        cv2.imshow('mosaic', mosaic)
        cv2.waitKey(1)

    # TODO:
    # this type of body id acquisition is a little...
    # you know, ad-hoc.
    # print(env.gym.get_actor_rigid_body_dict(env.envs[0],
    #         env.actors['scene'][0]['cube']))
    if False:
        body_indices = [env.gym.find_actor_rigid_body_index(
            env.envs[i],
            env.actors['scene'][i]['cube'],
            'box', gymapi.IndexDomain.DOMAIN_ENV
        ) for i in range(num_env)]
        # agent = RandomVirtualPushAgent(num_env,
        #                                body_indices)
    # agent = RandomVirtualPushAgentV2(num_env, env)
    # agent = RandomGripperAgent(num_env,
    #                            env.cfg.th_device)
    # agent = ManualAgent(num_env, env)
    # agent = MAV2(num_env, env)

    # while True:
    # for i in range(10000):

    # action = 0.1 * th.randn(size=(num_env, 8),
    #                         dtype=th.float32,
    #                         device=env.device)
    for i in tqdm(range(10000)):
        # action = None
        action = 2.0 * th.randn(size=(num_env, 8),
                                dtype=th.float32,
                                device=env.device)
        # action = agent()
        # if (i % 100) != 0:
        #     if isinstance(agent, RandomVirtualPushAgent):
        #         action[..., 1:4] *= 0.0
        #     else:
        #         # action[...] *= 0.0
        #         action = None
        # else:
        #     # print(action)
        #     pass

        obs, rew, done, info = env.step(action)
        # print('rew', rew)
        # print('done', done)

        if True:
            # Custom renderer!!
            matrix_from_quaternion(
                body_poses_7dof[..., 3:7],
                body_poses_4x4[..., :3, :3])
            mask, depth = cam(body_poses_4x4)

            depth = dcn(depth / depth.max())
            depth = depth[..., ::-1, :]  # hack!
            mosaic = einops.rearrange(
                depth, '(bh bw) h w -> (bh h) (bw w)', bh=QQ, bw=QQ)
            cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
            cv2.imshow('depth', mosaic)
            cv2.waitKey(1)
        else:
            # built-in renderer
            if True:
                flow = obs['images']['flow']

                # u = flow[..., ::2]
                # v = flow[..., 1::2]
                # print('u', u.shape)
                # print('v', v.shape)
                # print('flow', flow.shape, flow.dtype)
                # u = flow[:,1::2, 1::2]
                # flow = einops.rearrange(
                # flow, '... (h q1) (w q2) c -> ... (q1 h) (q2 w) c', q1=2,
                # q2=2)

                # u = flow[..., 0]
                # v = flow[..., 1]

                def _normalize(x):
                    return (x - x.min()) / (x.max() - x.min())

                # for tt, nn in zip((u, v), 'uv'):
                # flow_image = dcn(_normalize(tt)).astype(np.float32)
                flo = flow_image(flow / 2**15)
                mosaic = dcn(
                    _normalize(
                        einops.rearrange(
                            flo,
                            '(bh bw) h w c-> (bh h) (bw w) c',
                            bh=QQ,
                            bw=QQ))).astype(
                    np.float32)
                cv2.namedWindow(F'mosaic', cv2.WINDOW_NORMAL)
                cv2.imshow(F'mosaic', mosaic)
                cv2.waitKey(1)

            if True:
                color = dcn(obs['images']['color'])
                mosaic = einops.rearrange(
                    color, '(bh bw) h w c -> (bh h) (bw w) c', bh=QQ, bw=QQ)
                cv2.namedWindow('mosaic-color', cv2.WINDOW_NORMAL)
                cv2.imshow('mosaic-color', mosaic)
                cv2.waitKey(1)
            # time.sleep(0.01)
        # break


if __name__ == '__main__':
    main()
