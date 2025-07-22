#!/usr/bin/env python3

from abc import ABC, abstractmethod, abstractproperty

from contextlib import contextmanager
from typing import (
    Tuple, Optional, Iterable,
    Dict, Any, List, Callable)
from collections import defaultdict
from dataclasses import dataclass, field

from isaacgym import gymtorch
from isaacgym import gymapi

import numpy as np
import torch as th
from gym import spaces

from tqdm.auto import tqdm
from util.config import ConfigBase

from env.env.iface import EnvIface
from env.common import set_vulkan_device
from env.scene.base import SceneBase
from env.robot.base import RobotBase
from env.task.base import TaskBase
from env.common import (
    get_default_sim_params,
    aggregate
)
from env.env.wrap.with_ig_camera import WithCamera
from env.env.wrap.record_traj import TrajRecorder
_ENGINES = {'physx': gymapi.SIM_PHYSX, 'flex': gymapi.SIM_FLEX}

import nvtx

from icecream import ic


class EpisodeMonitor(EnvIface):
    def __init__(self, env: EnvIface):
        super().__init__()
        self.env = env

        self._num_iterate: int = 0
        self._num_episode: int = 0
        self._num_success: int = 0
        self._log_perstep: int = 512

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def setup(self):
        return self.env.setup()

    def reset(self):
        return self.env.reset()

    def reset_indexed(self, indices: Optional[th.Tensor] = None):
        return self.env.reset_indexed(indices)

    def step(self, actions: th.Tensor):
        obs, rew, done, info = self.env.step(actions)
        # assert ('success' in info)
        self._num_episode += done.sum()
        self._num_success += (done & info['success']).sum()
        self._num_iterate += 1
        if (self._num_iterate % self._log_perstep) == 0:
            ic(self._num_success / self._num_episode)
        return (obs, rew, done, info)

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)


class MonitoredGym:
    def __init__(self, gym):
        # self.gym = gym
        object.__setattr__(self, 'gym', gym)

    def __getattr__(self, name):
        # if name == 'set_actor_root_state_tensor_indexed':
        #     print('here!')
        print(F'gym.{name}')
        return getattr(self.gym, name)

    def __setattr__(self, name, value):
        return setattr(self.gym, name, value)


class TrackDebugLines:
    def __init__(self, gym, env):
        object.__setattr__(self, 'gym', gym)
        object.__setattr__(self, '_env', env)
        object.__setattr__(self, 'lines', {})

    def clear_lines(self, viewer):
        # print('-clear_lines-')
        self.lines.clear()
        if viewer is not None:
            return self.gym.clear_lines(viewer)

    def _store_lines(self, env_id: int, num_lines, vertices, colors):
        if env_id not in self.lines:
            self.lines[env_id] = []

        vertices = vertices.reshape(-1)
        colors = colors.reshape(-1)

        if colors.dtype == gymapi.Vec3:
            colors = np.asarray([[c['x'], c['y'], c['z']] for c in colors],
                                dtype=np.float32)

        if vertices.dtype == gymapi.Vec3:
            vertices = np.asarray([[v['x'], v['y'], v['z']] for v in vertices],
                                  dtype=np.float32)
        vertices = vertices.reshape(-1, 2, 3)
        colors = colors.reshape(-1, 3)
        colors = np.broadcast_to(colors, vertices[..., 0, :].shape)
        self.lines[env_id].append((num_lines, vertices.copy(), colors.copy()))

    def add_lines(self,
                  viewer,
                  env,
                  num_lines: int,
                  vertices,
                  colors):
        env_id: int = self._env.envs.index(env)
        self._store_lines(env_id, num_lines, vertices, colors)
        if viewer is not None:
            return self.gym.add_lines(viewer, env, num_lines, vertices, colors)

    def __getattr__(self, name):
        return getattr(self.gym, name)

    def __setattr__(self, name, value):
        return setattr(self.gym, name, value)


class TrackObjectAssets:
    def __init__(self, gym):
        object.__setattr__(self, 'gym', gym)
        object.__setattr__(self, 'asset_args', {})
        object.__setattr__(self, 'actor_args', {})

    def create_box(self, sim: 'Sim', width: float, height: float, depth: float,
                   options: 'AssetOptions' = None):
        asset = self.gym.create_box(sim, width, height, depth, options)
        self.asset_args[asset] = {
            'width': width,
            'height': height,
            'depth': depth,
            'options': options
        }
        return asset

    def load_urdf(self, sim: 'Sim', rootpath: str, filename: str,
                  options: 'AssetOptions' = None):
        asset = self.gym.load_urdf(sim, rootpath, filename, options)
        self.asset_args[asset] = {'rootpath': rootpath,
                                  'filename': filename,
                                  'options': options}
        return asset

    def create_actor(self,
                     env: 'Env',
                     asset: 'Asset',
                     pose: 'Transform',
                     name: str,
                     group: int = -1,
                     filter: int = -1,
                     segmentationId: int = 0):
        actor = self.gym.create_actor(
            env, asset,
            pose, name,
            group, filter,
            segmentationId)
        self.actor_args[(env, actor)] = self.asset_args[asset]
        return actor

    def __getattr__(self, name):
        # if name == 'load_urdf':
        #     return self._load_urdf
        # elif name == 'create_actor':
        #     return self._create_actor
        return getattr(self.gym, name)

    def __setattr__(self, name, value):
        return setattr(self.gym, name, value)


class EnvBase(EnvIface):
    @dataclass
    class Config(ConfigBase):
        seed: int = 0

        compute_device_id: int = 0
        graphics_device_id: int = 0

        # Device to remap torch tensors to.
        # I'm assuming this usually has to be
        # identical to `compute_device_id`.
        th_device: Optional[str] = None
        physics_engine: str = 'physx'

        env_bound_lower: Tuple[float, float, float] = (-1, -1, -1)
        env_bound_upper: Tuple[float, float, float] = (+1, +1, +1)

        env_margin_scale: float = 1.0
        env_margin: float = 0.0

        num_env: int = 1
        action_period: int = 1
        # max_num_steps: int = 1000

        use_viewer: bool = False
        sync_viewer: bool = True

        dt: float = (1.0 / 240.0)
        substeps: int = 2
        pos_iter: int = 8
        vel_iter: int = 1
        collect_contact: str = 'never'

        env_create_pbar: bool = True

        viewer_camera_origin: Optional[
            Tuple[float, float, float]] = None
        viewer_camera_target: Optional[
            Tuple[float, float, float]] = None

        use_isaac_camera: bool = False
        isaac_camera: WithCamera.Config = WithCamera.Config()
        use_aggregate: bool = False
        aggregate_type: str = 'env'

        monitor_api_calls: bool = False
        track_object_assets: bool = True
        track_debug_lines: bool = False

        max_body_count: Optional[int] = None
        max_shape_count: Optional[int] = None
        reset_order: str = 'robot+scene'

        use_traj_recorder: bool = False

        def __post_init__(self):
            # This is probably correct.
            if self.th_device is None:
                self.th_device = F'cuda:{self.compute_device_id}'

    def __init__(self,
                 cfg: Config,
                 scene: SceneBase,
                 robot: RobotBase,
                 task: TaskBase):
        self.cfg = cfg

        self.gym = None
        self.sim = None
        self._device = None
        self.rng = None

        self.scene = scene
        self.robot = robot
        self.task = task

        self.assets = {}
        self.actors = {}
        self.sensors = {}
        self.descriptors = {}
        self.tensors = {}
        self.buffers = {'done': None}

        if cfg.use_isaac_camera:
             self.isaac_camera = WithCamera(cfg.isaac_camera)
        else:
             self.isaac_camera = None

        # NOTE: ad-hoc modification to
        # include keyboard callbacks.
        self._event_cb = defaultdict(list)
        self._step_phys: bool = True

        self._sim_cb = defaultdict(list)

        self.step_idx = 0
        
        if cfg.use_isaac_camera:
            self.isaac_camera.save_img_p0.start()
            self.isaac_camera.save_img_p1.start()
            self.isaac_camera.save_img_p2.start()

        if cfg.use_traj_recorder:
            self.traj_recorder = TrajRecorder(self.num_env)

    def on_key(self, key, cb):
        """ register callback on key. """
        if not hasattr(self, 'viewer'):
            return
        if self.viewer is None:
            return
        self.gym.subscribe_viewer_keyboard_event(
            self.viewer,
            key,
            str(key))
        self._event_cb[str(key)].append(cb)

    def on_mouse(self, mouse, cb):
        """ register callback on key. """
        if not hasattr(self, 'viewer'):
            return
        if self.viewer is None:
            return
        self.gym.subscribe_viewer_mouse_event(
            self.viewer,
            mouse,
            str(mouse))
        self._event_cb[str(mouse)].append(cb)

    def configure(self):
        cfg = self.cfg

        # Vulkan Device
        #self.set_vulkan_device()

        # FIXME: currently this RNG isn't
        # really used anywhere, and there are
        # concerns regarding mismatched RNG
        # between maybe the one from
        # - the simulator (isaac gym)
        # - numpy
        # - random.choice
        #  - torch
        # RNG.
        self.rng: np.random.Generator = None
        # self.seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        # Default torch device.
        self._device = th.device(cfg.th_device)

        # Simulation.
        sim_params = get_default_sim_params(dt=cfg.dt,
                                            substeps=cfg.substeps,
                                            pos_iter=cfg.pos_iter,
                                            vel_iter=cfg.vel_iter,
                                            collect_contact=cfg.collect_contact
                                            )
        self.gym = gymapi.acquire_gym()

        if cfg.monitor_api_calls:
            self.gym = MonitoredGym(self.gym)

        if cfg.track_object_assets:
            self.gym = TrackObjectAssets(self.gym)

        if cfg.track_debug_lines:
            self.gym = TrackDebugLines(self.gym, self)

        self.sim = self.gym.create_sim(
            cfg.compute_device_id,
            cfg.graphics_device_id,
            _ENGINES.get(cfg.physics_engine),
            sim_params)

        # Environments container.
        self.envs = []
        self.buffers['done'] = th.ones(size=(cfg.num_env,),
                                       dtype=bool,
                                       device=self.device)

    @property
    def device(self):
        return self._device

    @property
    def num_env(self):
        return self.cfg.num_env

    @property
    def timeout(self):
        return self.task.timeout

    def setup(self):
        cfg = self.cfg

        self.assets.update(self.create_assets())

        # Create envs.
        # > create actors
        # > create sensors ?
        # self.actors['scene'] = []
        # self.actors['robot'] = []
        # self.sensors['scene'] = []
        outputs = self.create_envs()
        (envs, actors, sensors) = outputs
        self.envs = envs
        self.actors = actors
        self.sensors = sensors

        # Simulation states et al. -> data arrays
        outputs = self.acquire_tensors()
        (descriptors, tensors) = outputs
        self.descriptors.update(descriptors)
        self.tensors.update(tensors)
        self.buffers['step'] = th.zeros(
            (cfg.num_env,), dtype=th.int,
            device=self.cfg.th_device)
        # self.allocate_buffers()

        # TODO:
        # this step is really awkward...
        # self.robot.setup(self)
        self.robot.setup(self)
        self.scene.setup(self)
        self.task.setup(self)

        # Additionally, camera
        # this sets up (allocates) camera-image data buffers.
        if self.cfg.use_isaac_camera:
            self.isaac_camera.setup(self)

        if cfg.use_viewer:
            self.viewer = self.gym.create_viewer(self.sim,
                                                 gymapi.CameraProperties())
            if (cfg.viewer_camera_origin is not None and
                    cfg.viewer_camera_target is not None and
                    len(self.envs) > 0):
                ic(cfg.viewer_camera_origin)
                ic(cfg.viewer_camera_target)
                self.gym.viewer_camera_look_at(
                    self.viewer,
                    self.envs[0],
                    gymapi.Vec3(*cfg.viewer_camera_origin),
                    gymapi.Vec3(*cfg.viewer_camera_target)
                )
        else:
            self.viewer = None

    @abstractmethod
    def apply_actions(self, actions):
        pass

    @abstractmethod
    def compute_observations(self):
        pass

    @abstractmethod
    def compute_feedback(self,
                         obs: th.Tensor,
                         action: th.Tensor):
        """
        NOTE: technically, the complete
        specification of the feedback (~reward) function
        should be (prev_state, action, next_state)
        """
        return self.task.compute_feedback(self, obs, action)

    ###@nvtx.annotate("EnvBase.refresh_tensors()", color="pink")
    def refresh_tensors(self):
        cfg = self.cfg
        sim = self.sim
        gym = self.gym

        # [1] Refresh robot related tensors.
        gym.refresh_dof_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_force_tensor(sim)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_jacobian_tensors(sim)
        gym.refresh_mass_matrix_tensors(sim)
        gym.refresh_force_sensor_tensor(sim)

    def set_vulkan_device(self):
        set_vulkan_device()

    ###@nvtx.annotate("EnvBase.reset_indexed()", color="brown")
    def reset_indexed(self,
                      indices: Optional[Iterable[int]] = None):
        """ Reset specific environments. """
        cfg = self.cfg
        gym = self.gym
        sim = self.sim

        if cfg.reset_order == 'robot+scene':
            # [1] populate the tensors
            # self.refresh_tensors()
            (id_robot, qpos, qvel) = self.robot.reset(
                self.gym, self.sim, self, indices)
            # if len(id_robot) > 0:
            #     ic(F'set id_robot = {id_robot}')
            #     id_robot = id_robot.to(dtype=th.int32,device=self.device).contiguous()
            #     out = self.gym.set_actor_root_state_tensor_indexed(
            #         self.sim, gymtorch.unwrap_tensor(self.tensors['root']),
            #         gymtorch.unwrap_tensor(id_robot),
            #         len(id_robot)
            #     )
            id_scene = self.scene.reset(self.gym, self.sim, self, indices)
            # id_scene = id_scene.to(dtype=th.int32,device=self.device).contiguous()
            # out = self.gym.set_actor_root_state_tensor_indexed(
            #     self.sim, gymtorch.unwrap_tensor(self.tensors['root']),
            #     gymtorch.unwrap_tensor(id_scene),
            #     len(id_scene)
            # )
        else:
            id_scene = self.scene.reset(self.gym, self.sim, self, indices)
            (id_robot, qpos, qvel) = self.robot.reset(
                self.gym, self.sim, self, indices)

        # [1] commit the `dof_state` tensors (usually robot joints)
        if len(id_robot) > 0 and ('dof' in self.tensors):
            gym.set_dof_state_tensor_indexed(
                sim, gymtorch.unwrap_tensor(self.tensors['dof']),
                gymtorch.unwrap_tensor(id_robot),
                len(id_robot)
            )

        # NOTE: len(id_robot) cannot be <=0.
        if qpos is not None and len(id_robot) > 0:
            gym.set_dof_position_target_tensor_indexed(
                sim, gymtorch.unwrap_tensor(qpos),
                gymtorch.unwrap_tensor(id_robot),
                len(id_robot)
            )

        # NOTE: len(id_robot) cannot be <=0.
        if qvel is not None and len(id_robot) > 0:
            gym.set_dof_velocity_target_tensor_indexed(
                sim, gymtorch.unwrap_tensor(qvel),
                gymtorch.unwrap_tensor(id_robot),
                len(id_robot)
            )

        # # [2] commit the `root_state` tensors.
        if True:
            if len(id_robot) <= 0:
                set_ids = id_scene
            else:
                # NOTE: is the sort needed?
                set_ids, _ = th.sort(th.cat([id_scene, id_robot], dim=0))
                # set_ids = th.cat([id_scene, id_robot], dim=0)

            set_ids = set_ids.to(dtype=th.int32, device=self.device)
            # ic(set_ids, id_scene, id_robot)
            # ic('pre', self.tensors['root'][set_ids.long()])
            # ic(set_ids, set_ids.shape, set_ids.dtype, set_ids.device)
            # ic(self.tensors['root'].shape,
            #    self.tensors['root'].dtype,
            #    self.tensors['root'].device)

            if len(set_ids) > 0:
                out = self.gym.set_actor_root_state_tensor_indexed(
                    self.sim, gymtorch.unwrap_tensor(self.tensors['root']),
                    gymtorch.unwrap_tensor(set_ids),
                    len(set_ids)
                )

        self.task.reset(self, indices)

        if self.cfg.use_isaac_camera:
             self.isaac_camera.reset(gym, self, indices)

        if indices is not None:
            self.buffers['step'][indices] = 0
            self.buffers['done'][indices] = 0
        else:
            self.buffers['step'].fill_(0)
            self.buffers['done'].fill_(0)
        return

    ###@nvtx.annotate("EnvBase.reset()", color="teal")
    def reset(self):
        """ Reset everything. """
        set_ids = self.reset_indexed()
        # self.buffers['step'].fill_(0)

        # TODO: try to reduce code duplication with step()?
        # TODO: is it necessary to call render() here?
        # self.render()
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.refresh_tensors()
        # ic('post', self.tensors['root'][set_ids.long()])
        return self.compute_observations()

    ###@nvtx.annotate("EnvBase.step()", color="green")
    def step(self, actions):
        """
        Generic step function.
        Runs the following routines in sequence:
            pre->phys->post
        """
        # _ALWAYS_ clear rendering... is this ok?
        # if self.viewer is not None:
        #    self.gym.clear_lines(self.viewer)

        cfg = self.cfg

        with nvtx.annotate("pre_physics()"):
            self.pre_physics_step(actions)

        with nvtx.annotate("physics()"):
            self.physics_step()
        
        if cfg.use_isaac_camera:
            cam_out = self.isaac_camera.step(self)
            image_lists = self.isaac_camera.save_images()
            self.isaac_camera.enqueue_images_mp(image_lists)

        #t1 = time.time()
        # Process callbacks
        if (hasattr(self, 'viewer')
                and (self.viewer is not None)
                and len(self._event_cb) > 0):
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action not in self._event_cb:
                    continue
                if evt.value <= 0:  # ???
                    continue
                cbs = self._event_cb[evt.action]
                for cb in cbs:
                    cb()
        with nvtx.annotate("post_physics()"):
            out = self.post_physics_step(actions)
            #t2 = time.time()
            #print(f'Other works cost {t2-t1} secs.')
        return out

    ###@nvtx.annotate("EnvBase.pre_physics_step()", color="purple")
    def pre_physics_step(self, actions):
        """
        * Apply actions from policy.
        """
        with nvtx.annotate("buffers[done]"):
            done = self.buffers['done']

        with nvtx.annotate("apply_actions"):
            # NOTE:
            # apply_actions() should _not_ apply
            # any actions for which `done`=True
            # since those environments will be reset here.
            self.apply_actions(actions)
            # self.robot.step_controller(self.gym,
            #                            self.sim, self)

        with nvtx.annotate("reset_indexed"):
            done_indices = th.argwhere(done).ravel()
            self.reset_indexed(done_indices)

        # Reset Option #1: blind reset
        # self.reset_indexed(th.argwhere(done).ravel())
        # Reset Option #2: apply control flow
        # if (done is not None) and done.sum() > 0:

    @nvtx.annotate("EnvBase.physics_step()", color="yellow")
    def physics_step(self):
        """
        * Run physics simulation.
        """
        cfg = self.cfg

        # Step physics N times.
        if True:
            for i in range(cfg.action_period - 1):
                self.render()
                self.robot.step_controller(self.gym, self.sim, self)
                with nvtx.annotate("gym.simulate"):
                    if self._step_phys:
                        self.gym.simulate(self.sim)
                        if self.cfg.use_traj_recorder:
                            self.traj_recorder.record(self.gym, self.sim, goal= self.task.goal)
                # Controller requires tensor update.
                self.refresh_tensors()
            self.render()
            self.robot.step_controller(self.gym, self.sim, self)
            with nvtx.annotate("gym.simulate"):
                if self._step_phys:
                    self.gym.simulate(self.sim)
                    self.step_idx += 1
                    if self.cfg.use_traj_recorder:
                        self.traj_recorder.record(self.gym, self.sim, goal= self.task.goal)
                    #self.gym.end_access_image_tensors(self.sim)

        else:
            self.render()
            self.gym.simulate(self.sim)

        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

    @nvtx.annotate("EnvBase.post_physics_step()", color="magenta")
    def post_physics_step(self, action):
        """
        Compute returned values.

        * refresh tensors.
        * compute rewards/feedback from task.
        * compute whether some environments have terminated.
        * reset terminated environments.
        * compute observations from sensors.
        """
        # Tensors are refreeshed at the physics step
        self.refresh_tensors()

        self.buffers['step'] += 1

        # First compute the observations.
        obs = self.compute_observations()
        rew, done, info = self.compute_feedback(obs, action)
        self.buffers['done'][...] = done

        return (obs, rew, done, info)

    @nvtx.annotate("render")
    def render(self):
        cfg = self.cfg
        if self.viewer is None:
            return

        if self.gym.query_viewer_has_closed(self.viewer):
            print('viewer has closed')
            return

        # TODO:
        # process keyboard events here.

        if self.device != 'cpu':
            self.gym.fetch_results(self.sim, True)

        if (cfg.use_viewer and cfg.sync_viewer):
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        else:
            self.gym.poll_viewer_events(self.viewer)

    @nvtx.annotate("create_assets")
    def create_assets(self) -> Dict[str, Any]:
        cfg = self.cfg
        sim = self.sim
        gym = self.gym
        # Create assets.
        assets = {}
        counts = {'scene': {},
                  'robot': {}}
        assets['scene'] = self.scene.create_assets(gym, sim, self,
                                                   counts=counts['scene'])
        assets['robot'] = self.robot.create_assets(gym, sim,
                                                   counts=counts['robot'])
        assets['task'] = self.task.create_assets(gym, sim)

        # FIXME: temporary hack
        self.__counts = counts

        return assets

    @nvtx.annotate("create_envs")
    def create_envs(self):
        cfg = self.cfg

        envs = []
        actors = {}
        sensors = {}
        for env_id in tqdm(range(cfg.num_env),
                           disable=not (cfg.env_create_pbar),
                           desc='create env'):
            # bodies, shapes - ?
            outputs = self.create_env(env_id)
            (env_i, actors_i, sensors_i) = outputs
            envs.append(env_i)
            for k, v in actors_i.items():
                if k not in actors:
                    actors[k] = []
                actors[k].append(v)
            for k, v in sensors_i.items():
                if k not in sensors:
                    sensors[k] = []
                sensors[k].append(v)
        return (envs, actors, sensors)

    def create_env(self, env_id: int) -> Tuple[List, Dict, Dict]:
        cfg = self.cfg
        gym = self.gym
        sim = self.sim

        # [1] Create env
        s = cfg.env_margin_scale
        b = cfg.env_margin
        lo = np.asarray(cfg.env_bound_lower)
        hi = np.asarray(cfg.env_bound_upper)
        env = gym.create_env(self.sim,
                             gymapi.Vec3(*(s * lo - b)),
                             gymapi.Vec3(*(s * hi + b)),
                             int(np.sqrt(cfg.num_env)))

        # HACK
        use_agg_all = (cfg.use_aggregate and cfg.aggregate_type == 'env')
        use_agg_each = (cfg.use_aggregate and cfg.aggregate_type == 'each')

        # Resolve `max_{body,shape}_count` arguments.
        # First try from config args, then fallback to auto compute.
        max_body_count = cfg.max_body_count
        max_shape_count = cfg.max_shape_count
        if max_body_count is None:
            max_body_count = (
                self.__counts['scene']['body'] +
                self.__counts['robot']['body'])
        if max_shape_count is None:
            max_shape_count = (
                self.__counts['scene']['shape'] +
                self.__counts['robot']['shape'])

        actors = {}
        with aggregate(gym, env, max_body_count, max_shape_count, True,
                       use=use_agg_all):
            with aggregate(gym, env, self.__counts['scene']['body'],
                           self.__counts['scene']['shape'], True,
                           use=use_agg_each):
                actors['scene'] = self.scene.create_actors(
                    gym, sim, env, env_id)

            with aggregate(gym, env, self.__counts['robot']['body'],
                           self.__counts['robot']['shape'], True,
                           use=use_agg_each):
                actors['robot'] = self.robot.create_actors(
                    gym, sim, env, env_id)
                # try:
                #    actors['robot'] = self.robot.create_actors(
                #        gym, sim, env, env_id)
                # except Exception:
                #    actors['robot'] = self.robot.create_actors(
                #        gym, sim, env, env_id, scene=self.scene)

        # VALIDATION
        if use_agg_all:
            shape_count = 0
            for i in range(gym.get_actor_count(env)):
                actor = gym.get_actor_handle(env, i)
                shape_count += gym.get_actor_rigid_shape_count(env, actor)
            body_count = gym.get_env_rigid_body_count(env)
            if body_count > max_body_count:
                raise ValueError(F'{body_count} > {max_body_count}')
            if shape_count > max_shape_count:
                raise ValueError(F'{shape_count} > {max_shape_count}')

        sensors = {}
        sensors['scene'] = self.scene.create_sensors(gym, sim, env, env_id)
        sensors['task'] = self.task.create_sensors(gym, sim, env, env_id)

        # NOTE:
        # Is it "fair" to give special treatment to cameras?
        # camera, tensors = self.camera.create_camera(gym, sim, env)
        # sensors['camera'] = {'camera': camera, 'tensors': tensors}

        # self.sensors['robot'].append(
        #     self.robot.create_sensors(gym, sim, env, i)
        # )
        # self.actors['task'].append(
        # self.task.create_actors(...)
        # )
        return (env, actors, sensors)

    def acquire_tensors(self):
        cfg = self.cfg
        gym = self.gym
        sim = self.sim

        # 1. Acquire gym torch tensor descriptors.
        descriptors = {}
        descriptors['root'] = gym.acquire_actor_root_state_tensor(sim)
        descriptors['body'] = gym.acquire_rigid_body_state_tensor(sim)
        descriptors['dof'] = gym.acquire_dof_state_tensor(sim)
        descriptors['net_contact'] = gym.acquire_net_contact_force_tensor(sim)
        descriptors['force_sensor'] = gym.acquire_force_sensor_tensor(sim)

        # 2. Wrap tensor descriptors as torch tensors.
        # NOTE: 13 = (pos(3)+orn(4)+lin_vel(3)+ang_vel(3)
        tensors = {}
        tensors['root'] = gymtorch.wrap_tensor(
            descriptors['root']
        )
        tensors['body'] = gymtorch.wrap_tensor(
            descriptors['body']
        )
        # .view(cfg.num_env, -1, 13)
        tensors['net_contact'] = gymtorch.wrap_tensor(
            descriptors['net_contact'])
        tensors['force_sensor'] = gymtorch.wrap_tensor(
            descriptors['force_sensor'])
        # .view(cfg.num_env, -1, 3)

        # TODO: figure this out
        dof_state = gymtorch.wrap_tensor(descriptors['dof'])
        if dof_state is not None:
            dof_state = dof_state.view(
                cfg.num_env, -1, 2)
            tensors['dof'] = dof_state

        # 3. Also save subsections for convenience.
        # dof_pos = dof_state[..., 0]
        # dof_vel = dof_state[..., 1]

        # contact_force = gymtorch.wrap_tensor(net_contact_force_tensor).view(
        #    cfg.num_env, -1, 3)
        return descriptors, tensors