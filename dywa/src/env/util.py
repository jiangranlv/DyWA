#!/usr/bin/env python3

from typing import Optional, Tuple
import itertools

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

import torch as th
import numpy as np
import open3d as o3d
import trimesh
from cho_util.math import transform as tx

from util.torch_util import dcn, set_seed
from util.math_util import (adjoint_matrix, skew_matrix)

import os
import random


def from_vec3(v) -> np.ndarray:
    return np.asarray([v.x, v.y, v.z])


def from_mat3(m) -> np.ndarray:
    return np.asarray([
        from_vec3(m.x),
        from_vec3(m.y),
        from_vec3(m.z)
    ])


def draw_one_inertia_box(
        gym, viewer, env_handle,
        pose: th.Tensor,
        mass: th.Tensor,
        com: th.Tensor,
        inertia: th.Tensor,
        color=None):
    # Principal moment + orientation.
    # Ensure that determinant is positive.
    w, v = np.linalg.eigh(inertia)
    if np.linalg.det(v) < 0:
        w[[0, 1]] = w[[1, 0]]
        v[:, [0, 1]] = v[:, [1, 0]]

    # Convert to quaternion.
    q = tx.rotation.quaternion.from_matrix(v)

    # Convert to box parameters.
    box_pose = gymapi.Transform()
    box_pose.p = gymapi.Vec3(*com)
    box_pose.r = gymapi.Quat(*q)
    bx = np.sqrt(6 * (w[1] + w[2] - w[0]) / mass)
    by = np.sqrt(6 * (w[2] + w[0] - w[1]) / mass)
    bz = np.sqrt(6 * (w[0] + w[1] - w[2]) / mass)
    box_geom = gymutil.WireframeBoxGeometry(
        bx, by, bz,
        pose=box_pose,
        color=color
    )

    xfm = dcn(pose[..., :7])
    txn = xfm[..., :3]
    rxn = xfm[..., 3:7]

    obj_pose = gymapi.Transform()
    obj_pose.p = gymapi.Vec3(*txn)
    obj_pose.r = gymapi.Quat(*rxn)

    return (gymutil.draw_lines(
        box_geom,
        gym,
        viewer,
        env_handle,
        obj_pose
    ))


def draw_bbox(gym, viewer, env_handle,
              pose: th.Tensor,
              bbox: th.Tensor,
              color=None):

    # indices = th.tensor([0,7],device=bbox.device)

    bbox_geom = gymutil.TrimeshBBoxGeometry(bbox, color=color)
    xfm = dcn(pose[..., :7])
    txn = xfm[..., :3]
    rxn = xfm[..., 3:7]
    # print(txn, th.mean(bbox,-2), bbox, bbox_geom.vertices())
    obj_pose = gymapi.Transform()
    obj_pose.p = gymapi.Vec3(*txn)
    obj_pose.r = gymapi.Quat(*rxn)
    colors = list(itertools.product([0, 1], repeat=3))
    for i, point in enumerate(bbox):
        point_pose = gymapi.Transform()
        point_pose.p = gymapi.Vec3(*point)
        ball_geom = gymutil.WireframeSphereGeometry(
            radius=0.01, pose=point_pose, color=colors[i])
        gymutil.draw_lines(
            ball_geom, gym, viewer, env_handle, None)
    return (gymutil.draw_lines(
        bbox_geom,
        gym,
        viewer,
        env_handle,
        None
    ))


def draw_keypoints(
        gym, viewer,
        env_handle,
        bbox: th.Tensor,
        alpha: Optional[np.ndarray] = None,
        min_alpha: Optional[float] = 0.5
):

    colors = list(itertools.product([0, 1], repeat=3))
    for i, point in enumerate(bbox):
        point_pose = gymapi.Transform()
        point_pose.p = gymapi.Vec3(*point)
        ball_geom = None
        if alpha is not None:
            # print(colors[i], type(colors[i]))
            # wtf = tuple([c * alpha for c in colors[i]])
            # print(wtf, type(wtf))
            if (min_alpha is None) or (alpha[i] >= min_alpha):
                ball_geom = gymutil.WireframeSphereGeometry(
                    radius=0.01, pose=point_pose,
                    color=tuple([c * alpha[i] for c in colors[i]])
                )
        else:
            ball_geom = gymutil.WireframeSphereGeometry(
                radius=0.01, pose=point_pose, color=colors[i])
        if ball_geom is not None:
            gymutil.draw_lines(
                ball_geom, gym, viewer, env_handle, None)


def get_mass_properties(gym, envs: list, handles: list
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return mass properties of handles as a numpy array.
    NOTE: For now, this routine is only valid for rigid-bodies.
    """
    N: int = len(envs)
    masses = np.zeros(N)
    inertias = np.zeros((N, 3, 3))

    for i in range(N):
        prop = gym.get_actor_rigid_body_properties(
            envs[i], handles[i]
        )
        # assert (len(prop) == 1)
        if len(prop) == 1:
            masses[i] = prop[0].mass
            inertias[i] = from_mat3(prop[0].inertia)
        else:
            masses[i] = 0.0
            for j in range(len(prop)):
                masses[i] += prop[j].mass
                # FOR NOW I HAVE NO IDEA
                inertias[i] = float('nan')
    return (masses, inertias)


def draw_patch_with_cvxhull(gym, viewer, patch, env, **kwds):
    """
    ...
    patch: (S,P,3)
    """
    color = kwds.pop('color', None)
    patch = dcn(patch)

    S, P, _ = patch.shape

    for j in range(S):
        points = patch[j]
        c = tuple(color[j])
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        edges = trimesh.PointCloud(points).convex_hull.edges
        for p1, p2 in points[edges]:  # N, 2, 3
            gymutil.draw_line(gymapi.Vec3(*p1), gymapi.Vec3(*p2),
                              gymapi.Vec3(*c), gym, viewer, env)


def draw_cloud_with_sphere(gym, viewer, cloud, env,
                           **kwds):
    num_lats = kwds.pop('num_lats', 4)
    num_lons = kwds.pop('num_lons', 4)
    radius = kwds.pop('radius', 0.01)
    color = kwds.pop('color', None)

    if color is None:
        color = [None] * len(cloud)
    elif len(color.shape) == 1:
        color = [color] * len(cloud)

    for c, k in zip(cloud, color):
        draw_sphere(gym, viewer, env,
                    num_lats=num_lats,
                    num_lons=num_lons,
                    pos=c,
                    radius=radius,
                    color=tuple(k),
                    **kwds)


def draw_cloud_with_ray(gym,
                        viewer,
                        cloud,
                        env,
                        **kwds):
    num_lats = kwds.pop('num_lats', 4)
    num_lons = kwds.pop('num_lons', 4)
    radius = kwds.pop('radius', 0.01)
    color = kwds.pop('color', None)
    eye = kwds.pop('eye', None)

    if color is None:
        color = [None] * len(cloud)
    elif len(color.shape) == 1:
        color = [color] * len(cloud)

    for c, k in zip(cloud, color):
        gymutil.draw_line(
            gymapi.Vec3(*eye),
            gymapi.Vec3(*c),
            gymapi.Vec3(*k),
            gym, viewer, env)


def draw_cloud_with_nn(gym, viewer, cloud, env):
    """
    Draw point clouds, where the
    `lines` are created from connected k-neighbors
    from the label image.
    """
    from pytorch3d.ops import knn_points
    _, _, nn = knn_points(
        cloud[None],
        cloud[None],
        K=2,
        return_nn=True,
        return_sorted=True)

    source = cloud.reshape(-1, 3)
    target = nn[:, :, 1].reshape(-1, 3)

    # origin = gym.get_env_origin(env)
    # origin = th.as_tensor([origin.x, origin.y, origin.z],
    #                       dtype=th.float32, device=source.device)
    # source = source - origin
    # target = target - origin
    lines = th.cat([source, target], dim=-1)
    gym.add_lines(
        viewer, env,
        len(lines),
        lines.detach().cpu().numpy(),
        th.ones_like(lines[..., : 3]).detach().cpu().numpy()
    )


def draw_cloud_with_mask(gym, viewer, cloud, env,
                         mask: th.Tensor,
                         color=None):
    """
    Draw point clouds, where the
    `lines` are created from connected 4-neighbors
    from the mask image.
    """
    src = mask[:-1, :-1]
    dst_i = mask[1:, :-1]
    dst_j = mask[:-1, 1:]

    msk_i = (src & dst_i)
    msk_j = (src & dst_j)

    src_i_points = cloud[:-1, :-1][msk_i]
    dst_i_points = cloud[1:, :-1][msk_i]

    src_j_points = cloud[:-1, :-1][msk_j]
    dst_j_points = cloud[:-1, 1:][msk_j]

    i_lines = th.stack(
        [src_i_points, dst_i_points],
        dim=1)
    j_lines = th.stack(
        [src_j_points, dst_j_points],
        dim=1)
    lines = th.cat([i_lines, j_lines], dim=0)
    colors = th.ones_like(lines[:, 0, :3]).detach().cpu().numpy()

    if color is not None:
        colors[:] = color

    gym.add_lines(
        viewer, env,
        len(lines),
        lines.detach().cpu().numpy(),
        colors
    )


def draw_sphere(gym, viewer, env, pose=None, **kwds):
    pose = kwds.pop('pose', None)

    if not isinstance(pose, gymapi.Transform):
        pos = kwds.pop('pos')
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*pos)

    ball_geom = gymutil.WireframeSphereGeometry(**kwds)
    return gymutil.draw_lines(
        ball_geom,
        gym,
        viewer,
        env,
        pose)


def draw_cloud(gym, viewer, cloud, env,
               label: Optional[th.Tensor] = None,
               color: Optional[Tuple[float, float, float]] = None):
    if label is not None:
        return draw_cloud_with_mask(gym, viewer, cloud, env, label,
                                    color=color)
    else:
        return draw_cloud_with_nn(gym, viewer, cloud, env)


@th.jit.script
def compute_wrench(
        pose0: th.Tensor,
        twist0: th.Tensor,
        twist1: th.Tensor,
        dt: float,
        inertia: th.Tensor,
        mass: th.Tensor):
    """
    Assumes [w;v] convention for twists.

    pose0   -> world-frame initial pose of object at t=0
    twist0  -> world-frame twist of object CoM at t=0
    twist1  -> world-frame twist of object CoM at t=1
    dt      -> timestep size
    inertia -> principal moment of inertia, 3x3
    """
    w0, v0 = twist0[..., :3], twist0[..., 3:]
    # w1, v1 = twist1[..., :3], twist1[..., 3:]

    accel = (twist1 - twist0) / dt

    # Generalized body-frame inertia
    Gb = th.zeros(
        inertia.shape[:-2] + (6, 6),
        dtype=inertia.dtype,
        device=inertia.device)
    Gb[..., :3, :3] = inertia
    Gb[..., 3, 3] = mass
    Gb[..., 4, 4] = mass
    Gb[..., 5, 5] = mass

    A = adjoint_matrix(pose0)
    # Gw = A.T @ Gb @ A
    Gw = th.einsum('...ij, ...ik, ...kl -> ...jl', A, Gb, A)

    ad_vw = th.zeros_like(Gb)
    W0 = skew_matrix(w0)
    ad_vw[..., :3, :3] = W0
    ad_vw[..., 3:, :3] = skew_matrix(v0)
    ad_vw[..., 3:, 3:] = W0

    # ad_vw = np.block([
    #     [skew_matrix(w0), np.zeros((3, 3))],
    #     [skew_matrix(v0), skew_matrix(w0)]
    # ])
    lhs = th.einsum('...ij, ...i -> ...j', Gw, accel)
    rhs = th.einsum('...ij, ...ik, ...k -> ...j',
                    ad_vw, Gw, twist0)
    return lhs - rhs
    # return Gw @ accel - ad_vw.T @ Gw @ twist0

# refer to
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/utils/utils.py
