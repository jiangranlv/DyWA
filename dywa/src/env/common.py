#!/usr/bin/env python3

import os
import subprocess
from typing import Optional, Tuple, Dict, Iterable, Any, List
import itertools
from functools import partial
import numpy as np
import logging
from contextlib import contextmanager

from isaacgym import gymtorch
from isaacgym import gymapi
import torch as th

from util.math_util import quat_rotate


def set_actor_friction(gym, env, actor_handle, friction: float):
    prop = gym.get_actor_rigid_shape_properties(
        env,
        actor_handle
    )
    for p in prop:
        p.friction = friction
        p.torsion_friction = friction
        p.rolling_friction = friction
    return gym.set_actor_rigid_shape_properties(env,
                                                actor_handle,
                                                prop)


def set_actor_restitution(gym, env, actor_handle, restitution: float):
    prop = gym.get_actor_rigid_shape_properties(
        env,
        actor_handle
    )
    for p in prop:
        p.restitution = restitution
    return gym.set_actor_rigid_shape_properties(env,
                                                actor_handle,
                                                prop)


def apply_domain_randomization(gym, env: int, actor_handle: int,

                               enable_mass: bool = False,
                               min_mass: float = 0.01,
                               max_mass: float = 2.0,
                               use_mass_set: bool = False,
                               mass_set: Optional[Tuple[float, ...]] = None,

                               change_scale: bool = False,
                               min_scale: float = 0.5,
                               max_scale: float = 1.5,
                               radius: Optional[float] = None,

                               enable_friction: bool = True,
                               min_friction: float = 0.0,
                               max_friction: float = 2.0,

                               enable_restitution: bool = False,
                               min_restitution: float = 0.0,
                               max_restitution: float = 0.2,

                               target_shape_indices: Optional[Tuple[int, ...]] = None
                               ):
    out = {}

    # NOTE: `scale` apparently does not work for now.
    # NOTE: actually, `scale` WILL NEVER WORK.
    # https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/domain_randomization.md#domain-randomization-dictionary
    # scale = np.random.uniform(0.8, 1.2)  # + 1.0
    if change_scale:
        assert (radius is not None)
        target_size = np.random.uniform(min_scale, max_scale)
        scale = target_size / radius
        gym.set_actor_scale(env, actor_handle, scale)
        out['scale'] = scale

    if enable_mass:
        prop = gym.get_actor_rigid_body_properties(env, actor_handle)
        if use_mass_set:
            assert ((mass_set is not None))
            mass = np.random.choice(mass_set)
        else:
            mass = np.random.uniform(min_mass, max_mass)
        for p in prop:
            p.mass = mass
        suc = gym.set_actor_rigid_body_properties(
            env,
            actor_handle, prop,
            True)  # recompute inertia!
        out['mass'] = mass

    prop = gym.get_actor_rigid_shape_properties(
        env,
        actor_handle
    )

    # TODO: Switch for firction DR and its range should be added as config
    enable_shape_prop: bool = (
        enable_friction or enable_restitution)

    if enable_shape_prop:
        if enable_friction:
            friction = np.random.uniform(min_friction, max_friction)
            out['friction'] = friction

        if enable_restitution:
            restitution = np.random.uniform(min_restitution, max_restitution)
            out['restitution'] = restitution

        if target_shape_indices is None:
            target_shape_indices = range(len(prop))

        for i in target_shape_indices:
            if enable_friction:
                prop[i].friction = friction

            if enable_restitution:
                prop[i].restitution = restitution

        suc0 = gym.set_actor_rigid_shape_properties(env,
                                                    actor_handle,
                                                    prop)
    else:
        out['friction'] = 1.0

    # print('suc0', suc0)

    # NOTE: `scale` apparently does not work for now.
    # suc = gym.set_actor_scale(
    #     env,
    #     actor_handle,
    #     scale)
    # print('scale, suc', suc) # ==> True?
    return out


def set_vulkan_device():
    """
    Set vulkan device? Not sure what's happening.
    But without this, we cannot wrap to torch tensors.
    """
    logging.warn('_configure_env() overrides'
                 + '`MESA_VK_DEVICE_SELECT` variable.')
    # Mysterious process of configuring vulkan.
    # Based on the following answer:
    # https://forums.developer.nvidia.com/t/create-camera-sensor-fail-on-buffer/186529/5
    vk_info = subprocess.run(
        'vulkaninfo',
        env=dict(
            MESA_VK_DEVICE_SELECT='list'),
        check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL).stderr
    vk_info = vk_info.decode('utf-8').split('\n')
    vk_info = [s for s in vk_info if 'discrete GPU' in s]
    vk_device = vk_info[-1].split(': ', 1)[1].split(' "', 1)[0]
    print('vk_device', vk_device)
    os.environ['MESA_VK_DEVICE_SELECT'] = vk_device


def get_default_sim_params(dt: float = 1.0 / 240.0,
                           substeps: int = 2,
                           pos_iter: int = 8,
                           vel_iter: int = 0,
                           collect_contact: str = 'never') -> gymapi.SimParams:
    """ Get default simulation parameters. """
    params = gymapi.SimParams()
    params.up_axis = gymapi.UP_AXIS_Z
    # TODO: Consider also randomizing gravity?
    params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    # params.dt = 1.0 / 60.0
    params.dt = dt
    # params.dt = 1.0 / 481.0
    # params.dt = 1.0 / 961.0
    # params.dt = 0.0
    # params.dt = 1.0 / 1920.0
    # params.dt = 1.0 / 9600.0
    # params.dt = 1.0 / 1200
    # params.dt = 1.0 / 9600
    # params.dt = 1.0 / 2400
    # params.dt = 1.0 / 1920
    params.substeps = substeps
    params.use_gpu_pipeline = True  # ?

    # params.physx.solver_type = 0 # PGS
    params.physx.solver_type = 1  # TGS

    # params.physx.num_position_iterations = 12
    params.physx.num_position_iterations = pos_iter
    params.physx.num_velocity_iterations = vel_iter
    params.physx.rest_offset = 0.0
    params.physx.contact_offset = 0.001

    if collect_contact == 'never':
        params.physx.contact_collection = gymapi.ContactCollection.CC_NEVER
    elif collect_contact == 'last':
        params.physx.contact_collection = gymapi.ContactCollection.CC_LAST_SUBSTEP
    elif collect_contact == 'all':
        params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
    else:
        raise ValueError(F'unknown `collect-contact` = {collect_contact}')

    params.physx.friction_offset_threshold = 0.001
    params.physx.friction_correlation_distance = 0.0005
    params.physx.bounce_threshold_velocity = (
        2 * 9.81 * dt / substeps)
    params.physx.num_threads = 16

    # params.physx.max_gpu_contact_pairs = 1024 * 1024 * 16
    params.physx.max_gpu_contact_pairs = 1024 * 1024 * 8
    params.physx.default_buffer_size_multiplier = 8.0  # default 2

    # NOTE: possibility of improving sim. stability
    # params.physx.max_depenetration_velocity = 100.0
    params.physx.max_depenetration_velocity = 10.0
    params.physx.use_gpu = True  # ?

    # Set FleX-specific parameters
    params.flex.solver_type = 5
    params.flex.num_outer_iterations = 10
    params.flex.num_inner_iterations = 200
    params.flex.relaxation = 0.75
    params.flex.warm_start = 0.8
    params.flex.deterministic_mode = True
    # Set contact parameters
    params.flex.shape_collision_distance = 5e-4
    params.flex.contact_regularization = 1.0e-6
    params.flex.shape_collision_margin = 1.0e-4
    # params.flex.dynamic_friction = self.friction

    return params


def create_camera(height: int, width: int,
                  gym, sim, env):
    """ Create cameras. """
    prop = gymapi.CameraProperties()
    # prop.near_plane = 0.01
    # prop.far_plane = 5.0
    prop.height = height
    prop.width = width
    prop.enable_tensors = True
    # prop.use_collision_geometry = True
    prop.use_collision_geometry = False
    # print(prop.near_plane)
    # print(prop.far_plane)
    camera = gym.create_camera_sensor(env, prop)

    # TODO: figure out where to place the cameras,
    # in order to properly render the scene(s).
    # gym.set_camera_location(camera, env,
    #                         gymapi.Vec3(0, 0.1, 0.4),
    #                         gymapi.Vec3(0.5, 0.0, 0.2))

    # Acquire RGB/D tensors.
    color_tensor_descriptor = gym.get_camera_image_gpu_tensor(
        sim, env, camera, gymapi.IMAGE_COLOR
    )
    color_tensor = gymtorch.wrap_tensor(
        color_tensor_descriptor)

    depth_tensor_descriptor = gym.get_camera_image_gpu_tensor(
        sim, env, camera, gymapi.IMAGE_DEPTH
    )
    depth_tensor = gymtorch.wrap_tensor(depth_tensor_descriptor)

    label_tensor_descriptor = gym.get_camera_image_gpu_tensor(
        sim, env, camera, gymapi.IMAGE_SEGMENTATION
    )
    label_tensor = gymtorch.wrap_tensor(label_tensor_descriptor)

    # NOTE: this assert fails if
    # your GPU device configuration is botched for
    # whatever reason!
    assert (color_tensor is not None)
    assert (depth_tensor is not None)
    assert (label_tensor is not None)
    tensors = {'color': color_tensor,
               'depth': depth_tensor,
               'label': label_tensor}
    return (camera, tensors)


def TTT(*x, device: Optional[th.device] = None):
    return th.as_tensor(x,
                        device=device,
                        dtype=th.float32)


def draw_bbox(gym, viewer, env, txn, rxn, scale,
              device: Optional[th.device] = None):
    lines = []
    for fixed in itertools.product([-1, +1], repeat=3 - 1):
        for axis in range(3):
            i0 = list(fixed)
            i0.insert(axis, -1)
            i1 = list(fixed)
            i1.insert(axis, +1)

            i0 = th.as_tensor(i0, dtype=th.float32)
            i1 = th.as_tensor(i1, dtype=th.float32)
            p0 = txn + quat_rotate(rxn, scale * i0)
            p1 = txn + quat_rotate(rxn, scale * i1)
            line = th.cat([p0, p1])
            lines.append(line)
    lines = th.stack(lines, dim=0)
    return gym.add_lines(viewer, env, len(lines),
                         lines.detach().cpu().numpy(),
                         th.ones_like(lines[..., :3]).detach().cpu().numpy())


def draw_axis(gym, viewer, env, txn, rxn, device: Optional[th.device] = None):
    _TTT = partial(TTT, device=device)
    points = th.stack([
        # X-axis
        txn,
        txn + quat_rotate(rxn, _TTT(1, 0, 0)),
        # Y-axis
        txn,
        txn + quat_rotate(rxn, _TTT(0, 1, 0)),
        # Z-axis
        txn,
        txn + quat_rotate(rxn, _TTT(0, 0, 1)),
    ])
    return gym.add_lines(viewer, env,
                         3, points.detach().cpu().numpy(),
                         th.eye(3, device=device,
                                dtype=th.float32).detach().cpu().numpy()
                         )


def unproject(gym, sim, envs, cams, depth_image: th.Tensor,
              reshape: bool = True) -> th.Tensor:
    n: int = len(envs)

    # Build and maintain NDC coordinates.
    H, W = depth_image.shape[1], depth_image.shape[2]
    IJ = th.cartesian_prod(
        th.arange(H, device=depth_image.device),
        th.arange(W, device=depth_image.device))
    X = IJ[..., 1].float().sub_(W / 2).neg_().div_(W)
    Y = IJ[..., 0].float().sub_(H / 2).div_(H)
    XY = th.stack([X, Y], dim=-1).view(H, W, 2)

    # Lookup camera-related matrices.
    P = np.empty((n, 4, 4), dtype=np.float32)
    V = np.empty((n, 4, 4), dtype=np.float32)
    for i in range(n):
        P[i] = np.asarray(gym.get_camera_proj_matrix(sim, envs[i], cams[i]))
        V[i] = np.asarray(gym.get_camera_view_matrix(sim, envs[i], cams[i]))

    # FIXME: This routine assumes that all cameras
    # have identical intrinsic parameters.
    fu = 2.0 / P[0][0, 0]
    fv = 2.0 / P[0][1, 1]

    Vi = np.linalg.inv(V)

    # NOTE: this is not _actually_ the
    # inverse of the projection matrix.
    # It's the version that we use, since
    # we get `depth_image` already world coordinates.
    Pi = np.asarray([
        [fu, 0, 0, 0],
        [0, fv, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]],
        dtype=np.float32)
    PVi = np.einsum('ij,...jk -> ...ik', Pi, Vi)
    PVi = th.as_tensor(PVi, device=depth_image.device)

    # Populate initial NDC-like coordinates.
    points = th.empty(depth_image.shape[:3] + (4,),
                      dtype=th.float32,
                      device=depth_image.device)
    th.multiply(depth_image[..., None], XY[None],
                out=points[..., :2])
    points[..., 2] = depth_image
    points[..., 3] = 1

    # Apply inverse projection/view transforms.
    points = th.einsum('n...i,nij->n...j', points, PVi)
    if reshape:
        return points[..., :3].reshape(n, -1, 3)
    else:
        return points[..., :3]


@contextmanager
def aggregate(gym, env, *args, **kwds):
    use: bool = kwds.pop('use', True)
    if not use:
        yield
        return

    try:
        gym.begin_aggregate(env, *args, **kwds)
        yield
    finally:
        gym.end_aggregate(env)
