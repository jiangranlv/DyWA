from typing import Tuple, Iterable, List, Optional, Dict, Union

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import torch_utils

import numpy as np
import torch as th
import torch.nn.functional as F
import einops
from scipy.spatial.transform import Rotation as R
import pkg_resources


import trimesh
from pytorch3d.ops import knn_points
from yourdfpy import URDF

from util.math_util import (quat_from_axa, quat_inverse,
                                quat_multiply, quat_rotate,
                                matrix_from_quaternion,
                                apply_pose_tq,
                                matrix_from_pose
                                )
from env.env.help.with_nvdr_camera import get_link_mesh
from env.robot.franka_kin import franka_ik, franka_fk

from env.util import (
    draw_cloud_with_sphere,
)

from pdb import set_trace as bp

import nvtx
from icecream import ic


def align_vectors(a, b, eps: float = 0.00001):
    """
    Return q: rotate(q, a) == b
    """
    dot = th.einsum('...j, ...j->...', a, b)
    parallel = (dot > (1 - eps))
    opposite = (dot < (-1 + eps))

    not_alligend = th.logical_not(th.logical_or(parallel, opposite))
    cross = th.cross(a, b, dim=-1)
    # sin(\theta) = 2 sin(0.5*theta) cos(0.5*theta)
    # 1 + cos(\theta) # = 2 cos^2(0.5*theta)
    out = th.cat([cross, (1 + dot)[..., None]], dim=-1)
    out /= (1e-6 + out.norm(p=2, dim=-1, keepdim=True))

    # Handle aligned cases.
    out[parallel] = th.as_tensor((0, 0, 0, 1),
                                 dtype=out.dtype,
                                 device=out.device)
    out[opposite] = th.as_tensor((1, 0, 0, 0),
                                 dtype=out.dtype,
                                 device=out.device)

    return out


def check_collision(ee_pose: th.Tensor, ee_cloud,
                    obj_pose: th.Tensor, obj_cloud,
                    brute: bool = False,
                    eps: float = 1e-3):
    """
    ee_pose:   [N,7]
    ee_cloud:  [M,3], M=number of points in ee mesh
    obj_pose:  [N,7]
    obj_cloud: [N,O,3], O=number of points in obj mesh
    brute: compute distance as min(NxM) instead of KNN lookup
    eps: contact threshold; if distance is less than eps, we consider it as collision.
    """
    cloud_ee = apply_pose_tq(ee_pose[..., None, :], ee_cloud[None])
    cloud_obj = apply_pose_tq(obj_pose[..., None, :], obj_cloud)

    query, target = cloud_ee, cloud_obj
    if query.shape[-2] > target.shape[-2]:
        query, target = target, query
    # Will this be faster than brute min(NxM) query?
    if brute:
        delta = query[..., None, :, :]
        - target[..., :, None, :]
        dists = th.linalg.norm(delta, dim=-1)
        dists = (dists.reshape(*query.shape[:-2], -1)
                 .min(dim=-1)
                 .values)
    else:
        dists, _, _ = knn_points(query, target, K=1,
                                 return_nn=False,
                                 return_sorted=False)
        dists = dists.squeeze_(dim=-1).min(dim=-1).values
    return (dists < eps), dists


class CheckHandObjectCollision:

    def __init__(self, env):
        self.env = env
        self._hand_cloud = self._load_hand_cloud()
        self._load_hand_surface()

    def _load_hand(self, simplified_finger=False):
        asset_root: str = pkg_resources.resource_filename('data', 'assets')
        urdf = URDF.load(
            F'{asset_root}/crm-panda/robots/franka_panda_fixed_finger.urdf',
            # '../../../src/data/assets/franka_description/robots/franka_panda.urdf',
            load_meshes=False,
            load_collision_meshes=True,
            build_collision_scene_graph=True)
        scene = urdf.collision_scene
        lf_xfm, _ = scene.graph.get('panda_leftfinger',
                                    'panda_tool')
        rf_xfm, _ = scene.graph.get('panda_rightfinger',
                                    'panda_tool')
        hand_xfm, _ = scene.graph.get('panda_hand',
                                      'panda_tool')

        # print(lf_xfm)
        # print(hand_xfm)
        # raise ValueError('stop')

        hand = scene.geometry['hand.obj'].apply_transform(hand_xfm)
        if simplified_finger:
            lfinger = trimesh.load_mesh(
                F'{asset_root}/crm-panda/meshes/collision/simple_finger.obj')
            rfinger = lfinger.copy()
            # rotate its finger
            T = np.eye(4)
            rot = R.from_rotvec(np.pi * np.array([0, 0, 1]))
            T[:3, :3] = rot.as_matrix()
            rfinger = rfinger.apply_transform(T)
            lf = lfinger.apply_transform(lf_xfm)
            rf = rfinger.apply_transform(rf_xfm)
        else:
            lf = scene.geometry['finger.obj'].apply_transform(lf_xfm)
            rf = scene.geometry['finger.obj_1'].apply_transform(rf_xfm)
        return hand, lf, rf

    def _load_hand_surface(self):
        env = self.env
        # hand, lf, rf = self._load_hand(True)
        hand, lf, rf = self._load_hand(False)

        hand_bbox = hand.bounding_box
        hand_normals = hand_bbox.face_normals
        h_points = hand_bbox.triangles_center
        # vertices[hand_bbox.faces].mean(axis=-2)
        # h_points = hand_bbox.face_centers
        # hand_offsets=np.einsum('...ij,...ij->...i',
        #                     hand_normals, h_points[:,0])

        self.hand_normals = th.as_tensor(hand_normals,
                                         dtype=th.float,
                                         device=env.device)
        self.h_points = th.as_tensor(h_points,
                                     dtype=th.float,
                                     device=env.device)
        # self.hand_offsets = th.as_tensor(hand_offsets,
        #                               dtype=th.float,
        #                               device=env.device)

        lf_normals = lf.face_normals
        lf_points = lf.triangles_center
        # lf_offsets=np.einsum('...ij,...ij->...i',
        #                      lf_normals, lf_points[:,0])
        self.lf_normals = th.as_tensor(lf_normals,
                                       dtype=th.float,
                                       device=env.device)
        self.lf_points = th.as_tensor(lf_points,
                                      dtype=th.float,
                                      device=env.device)
        # self.lf_offsets = th.as_tensor(lf_offsets,
        #                               dtype=th.float,
        #                               device=env.device)

        rf_normals = rf.face_normals
        rf_points = rf.triangles_center
        # rf_offsets=np.einsum('...ij,...ij->...i',
        #                      rf_normals, rf_points[:,0])
        self.rf_normals = th.as_tensor(rf_normals,
                                       dtype=th.float,
                                       device=env.device)
        self.rf_points = th.as_tensor(rf_points,
                                      dtype=th.float,
                                      device=env.device)
        # self.rf_offsets = th.as_tensor(rf_offsets,
        #                               dtype=th.float,
        #                               device=env.device)

    def _load_hand_cloud(self):
        env = self.env
        # FIXME:
        # [1] load from cache.
        # [2] combine with finger point cloud.
        if True:
            hand, lf, rf = self._load_hand()
            hand_cloud, _ = trimesh.sample.sample_surface(
                trimesh.util.concatenate([hand, lf, rf]), 1024)
            # trimesh.PointCloud(hand_cloud).export('/tmp/hand-with-fingers.obj')
            hand_cloud = th.as_tensor(hand_cloud,
                                      dtype=th.float,
                                      device=env.device)
        else:
            urdf = URDF.load(
                '../../../src/data/assets/crm-panda/robots/franka_panda_fixed_finger.urdf',
                load_meshes=False,
                load_collision_meshes=False)
            hand_cloud = th.as_tensor(
                get_link_mesh(urdf, urdf.link_map['panda_hand'], use_col=True,
                              alt_path='../../../src/data/assets/'
                              ).vertices,
                dtype=th.float,
                device=env.device)
        return hand_cloud

    def check_collision(self, ee_pose: th.Tensor,
                        obj_pose: th.Tensor, obj_cloud,
                        table_height: float = 0.4,
                        eps: float = 1e-6,
                        debug: bool = False):

        cloud_obj = apply_pose_tq(obj_pose[..., None, :], obj_cloud)
        env = self.env

        # hand object collision
        batch = ee_pose.shape[0]
        hand_normals = quat_rotate(ee_pose[..., None, 3:],
                                   self.hand_normals[None, None])
        hand_point = apply_pose_tq(ee_pose[..., None, :],
                                   self.h_points[None, None])
        hand_offset = th.einsum('...ij,...ij->...i',
                                hand_normals, hand_point)
        _dot = th.einsum('...ij,...kj->...ki',
                         hand_normals, cloud_obj)
        _dot -= hand_offset.unsqueeze(-2)
        hand_object_collide = (_dot < eps).all(-1).any(-1)

        # lf object collision
        lf_normals = quat_rotate(ee_pose[..., None, 3:],
                                 self.lf_normals[None])
        lf_points = apply_pose_tq(ee_pose[..., None, :],
                                  self.lf_points[None, :])

        lf_offset = th.einsum('...ij,...ij->...i',
                              lf_normals, lf_points)
        _dot = th.einsum('...ij,...kj->...ki',
                         lf_normals, cloud_obj)
        _dot -= lf_offset.unsqueeze(-2)
        lf_object_collide = (_dot < eps).all(-1).any(-1)
        hand_object_collide[lf_object_collide] = True

        # rf object collision
        rf_normals = quat_rotate(ee_pose[..., None, 3:],
                                 self.rf_normals[None])
        rf_points = apply_pose_tq(ee_pose[..., None, :],
                                  self.rf_points[None, :,])
        rf_offset = th.einsum('...ij,...ij->...i',
                              rf_normals, rf_points)
        _dot = th.einsum('...ij,...kj->...ki',
                         rf_normals, cloud_obj)
        _dot -= rf_offset.unsqueeze(-2)
        rf_object_collide = (_dot < eps).all(-1).any(-1)
        hand_object_collide[rf_object_collide] = True

        z = th.cat([hand_point[..., -1], lf_points[..., -1],
                   rf_points[..., -1]], -1)
        min_z, _ = z.min(-1)
        hand_object_collide[min_z < table_height + eps] = True
        if debug:
            for i in range(env.num_env):
                draw_cloud_with_sphere(
                    env.gym, env.viewer, cloud_obj[i, :: 8, :],
                    env.envs[i],
                    color=(0, 0, 1))
                draw_cloud_with_sphere(
                    env.gym, env.viewer, hand_point[i, :, :],
                    env.envs[i],
                    color=(0, 1, 0))
                draw_cloud_with_sphere(
                    env.gym, env.viewer, lf_points[i, :, :],
                    env.envs[i],
                    color=(1, 0, 0))
                draw_cloud_with_sphere(
                    env.gym, env.viewer, rf_points[i, :, :],
                    env.envs[i],
                    color=(1, 1, 0))
                print("hand_object: ", hand_object_collide)
                print("lf_object: ", lf_object_collide)
                print("rf_object: ", rf_object_collide)
                print("table_hand: ", min_z < table_height + eps)

        return hand_object_collide

    def __call__(self,
                 hand_state: Optional[th.Tensor] = None,
                 obj_state: Optional[th.Tensor] = None,
                 obj_cloud: Optional[th.Tensor] = None,
                 index: Optional[th.Tensor] = None
                 ):
        env = self.env
        hand_cloud = self._hand_cloud

        if hand_state is None:
            # Automatically lookup `hand_state`
            hand_ids = env.robot.ee_body_indices.long()
            hand_state = env.tensors['body'][hand_ids, :]

        if obj_state is None:
            # Automatically lookup `obj_state`
            obj_state = env.tensors['root'][env.scene.cur_ids.long()]

        if obj_cloud is None:
            obj_cloud = env.scene.cur_cloud
        if index is not None:
            col, dist = check_collision(hand_state[index], hand_cloud,
                                        obj_state[index], obj_cloud[index])
        else:
            # col, dist = check_collision(hand_state, hand_cloud,
            #                             obj_state, obj_cloud, eps=0.00000001)
            col = self.check_collision(hand_state, obj_state,
                                       obj_cloud)
        # print('dist', dist)
        return col


def sample_gripper_poses(num_samples,
                         obj_pose: th.Tensor,
                         obj_cloud: th.Tensor,
                         obj_normal: th.Tensor,
                         cone_max_theta: float,
                         offset: float,
                         randomize_yaw: bool):
    device = obj_pose.device
    num_env: int = obj_pose.shape[0]
    num_point: int = obj_cloud.shape[-2]

    point_idx = th.randint(0,
                           num_point,
                           size=(num_env, num_samples,),
                           device=device).long()
    sample_cloud = th.take_along_dim(obj_cloud,
                                     point_idx[..., None], -2)
    sample_normals = th.take_along_dim(obj_normal,
                                       point_idx[..., None], -2)

    z_axis = th.zeros((num_env, num_samples, 3),
                      device=device)
    z_axis[..., -1] = 1

    if cone_max_theta > 0:
        theta = th.tensor(cone_max_theta, device=device)
        min_z = th.cos(theta)
        delta = 1 - th.cos(theta)
        z = min_z + delta * th.rand((num_env, num_samples,),
                                    device=device)
        axis = th.stack([th.zeros_like(z),
                         th.sqrt(1 - z**2), z], dim=-1)
    else:
        axis = z_axis.clone()

    sample_cloud += sample_normals * offset

    # N, S,
    sample_cloud = apply_pose_tq(obj_pose[..., None, 0:7],
                                 sample_cloud)
    sample_normals = quat_rotate(obj_pose[..., None, 3:7],
                                 sample_normals.reshape(sample_cloud.shape))

    EE_ori = align_vectors(axis, -sample_normals)
    delta = quat_rotate(EE_ori, axis) + sample_normals

    if randomize_yaw:
        random_yaw = 2 * np.pi * th.rand((num_samples, 1), device=device)
        yaw_quat = quat_from_axa(z_axis * random_yaw)
        EE_ori = quat_multiply(EE_ori, yaw_quat)
    # get EE
    # print(sampled_points.shape)
    # print(EE_ori.shape)
    # print(obj_pose.shape)
    EE_pose = th.cat([sample_cloud, EE_ori], axis=-1)
    return (EE_pose)


def solve_ik_from_contact(env, env_ids: th.Tensor,
                          default_pose: th.Tensor,
                          T_b: th.Tensor,
                          collision_checker,
                          num_queries: Optional[int] = None,
                          offset: float = 0.005,
                          yaw_random: bool = True,
                          cone_max_theta: float = 30.,
                          out: Optional[th.Tensor] = None,
                          query_multiplier: int = 1
                          ):
    if num_queries is None:
        num_queries = len(env_ids)
    num_samples = num_queries * query_multiplier

    if out is None:
        out = th.zeros(num_queries, 7, dtype=th.float,
                       device=env.device)
        indices = th.arange(num_queries, device=env.device)
    else:
        indices = env_ids

    if num_queries == 0:
        print('returning since theres nothing to do')
        return None, None

    obj_ids = env.scene.cur_ids.long()[env_ids]
    obj_pose = env.tensors['root'][obj_ids, :]

    # Get contact point from point set
    point_sets = env.scene.cur_cloud[env_ids]
    point_normals = env.scene.cur_normal[env_ids]

    EE_pos = sample_gripper_poses(query_multiplier,
                                  obj_pose,
                                  point_sets,
                                  point_normals,
                                  cone_max_theta,
                                  offset,
                                  yaw_random)

    # Filter to collision-free EE poses.
    is_col = collision_checker(EE_pos, obj_pose[:, None],
                               point_sets[:, None])
    if False:
        is_col = is_col.min(dim=-1)
        EE_pos = th.take_along_dim(EE_pos,
                                   is_col.indices[:, None, ..., None],
                                   dim=-2).squeeze(dim=-2)
        is_col = is_col.values
    else:
        # preferentially sample collision-free poses.
        # msk = th.bernoulli(self.p_mask.expand(seq.shape[:-1]))
        # print('is_col', is_col.shape)
        col_free_indices = th.multinomial(1.0 - is_col.float() + 1e-6,
                                          query_multiplier,
                                          True)
        EE_pos = th.take_along_dim(EE_pos,
                                   col_free_indices[..., None],
                                   dim=-2)
        is_col = th.take_along_dim(is_col,
                                   col_free_indices,
                                   dim=-1)
        # is_col = is_col.values

    # print(is_col)
    # print(th.argwhere(~is_col))
    # print('mean collision ratio', is_col.float().mean())

    # Compute relative pose as homogeneous matrix, as IK input
    T_tip = matrix_from_pose(EE_pos[..., 0:3],
                             EE_pos[..., 3:7])
    # print(T_tip.shape, T_b.shape) # 8,16,4,4 / 8,4,4
    T_rel = (th.linalg.inv(T_b)[:, None] @ T_tip)
    # print('T_rel', T_rel.shape)
    # print('is_col', is_col.shape) # 8,16
    # print('mean post-filter collision ratio', is_col.float().mean())

    # out.copy_(default_pose[None])
    out[:] = default_pose[None]
    has_any_sol = th.zeros_like(out[..., 0],
                                dtype=th.bool)

    if not th.all(is_col).item():
        if True:
            # Simplified route
            ik_sol = franka_ik(T_rel.reshape(-1, 4, 4), width=0.0,
                               n_iter=32).reshape(*T_rel.shape[:-2], -1)
            has_ik = ik_sol[..., -1].bool()
            has_sol = (~is_col & has_ik)
            has_any_sol = has_sol.any(dim=-1)

            a = ik_sol[..., :7]
            ab = th.multinomial(
                has_sol.float() + 1e-6,
                1,
                True).squeeze(
                dim=-1)
            b = th.take_along_dim(
                a, ab[..., None, None], dim=-2).squeeze(dim=-2)
            c = b[has_any_sol]
            out[has_any_sol] = c
            # inds = th.argwhere(~is_col)[valid_ik]
            # print(inds)
            # out[indices[:,None].expand_as(is_col)[~is_col][valid_ik]] = (
            #         ik_sol[valid_ik, ..., :7]
            #         )
        else:
            # (8,16,4,4) -> (Q, 4, 4)
            ik_sol = franka_ik(T_rel[~is_col],
                               width=0.0,
                               n_iter=32)
            valid = franka_fk((ik_sol[..., :7] + th.pi) % (2 * th.pi) - th.pi)
            # ik_sol[..., 7] -= th.pi/2 + th.pi/4 # incorporate 45-deg rotation

            # print('T_rel target', T_rel[~is_col])
            # print('valid', valid)

            # print(ik_sol.shape)
            # print(out.shape)
            # print(indices[~is_col].shape, indices[~is_col])
            # invalid_ik = (ik_sol[..., -1] == 0)
            valid_ik = (
                (ik_sol[..., -1] == 1)
                # & (th.abs(T_rel[~is_col] - valid) < 1e-3).all(dim=-1).all(dim=-1)
            )
            # print('got_ik', valid_ik)
            # print(T_rel[~is_col][0])
            # print(valid[0])
            # print(th.abs(T_rel[~is_col] - valid).reshape(*T_rel.shape[:-2],-1).sum(dim=-1))

            # if True:
            #     delta = th.linalg.inv(valid[0][...,:3,:3]) @ T_rel[~is_col][0][...,:3,:3]
            #     print(delta)

            # print(indices[:,None].expand_as(is_col)[~is_col][valid_ik])
            # print(th.unique(indices[:,None].expand_as(is_col)[~is_col][valid_ik]))
            inds = th.argwhere(~is_col)[valid_ik]
            print(inds)
            out[indices[:, None].expand_as(is_col)[~is_col][valid_ik]] = (
                ik_sol[valid_ik, ..., :7]
            )
        # print('out', out)

        # N, M -> I
        # th.argwhere(~is_col)[valid_ik]

        # print('ik valid ratio', valid_ik.float().mean())
        # ik_sol[invalid_ik,:7] = default_pose
        # print(is_col.shape)
        # print(is_col
        # print(indices[~is_col].shape) # doesnotwork
        # print(valid_ik.shape)
        # is_col => (8, 16)
        # out[indices[~is_col][valid_ik], :] = (
        #        ik_sol[valid_ik, ..., :7]
        # )
    out = (out + th.pi) % (2 * th.pi) - th.pi
    return out, has_any_sol


@th.jit.script
def orientation_error_osc(q_src: th.Tensor, q_dst: th.Tensor):
    s1, c1 = q_src[..., :3], q_src[..., 3:4]
    s2, c2 = q_dst[..., :3], q_dst[..., 3:4]
    return -(s1 * c2 - s2 * c1 + th.cross(s2, s1, dim=-1))


def find_actor_indices(gym, envs: Iterable, name: str,
                       domain=gymapi.IndexDomain.DOMAIN_SIM) -> List[int]:
    return [gym.find_actor_index(envs[i], name, domain)
            for i in range(len(envs))]


def find_actor_handles(gym, envs: Iterable, name: str) -> List[int]:
    return [gym.find_actor_handle(envs[i], name)
            for i in range(len(envs))]


class CartesianControlError:
    def __init__(self,
                 accumulate: bool,
                 target_shape: Optional[Tuple[int, ...]] = None,
                 dtype=th.float,
                 device=None,
                 clip_pos: Optional[float] = 0.1,
                 clip_orn: Optional[float] = 0.1,
                 pos_bound: th.Tensor = None,
                 local: bool = False
                 ):
        if local:
            raise ValueError('local is broken :(')
        self.accumulate = accumulate
        self.target = None

        if self.accumulate:
            self.target = th.zeros(target_shape,
                                   dtype=dtype,
                                   device=device)
            self.hasref = th.zeros(target_shape[:1],
                                   dtype=bool,
                                   device=device)
        self.clip_pos = clip_pos
        self.clip_orn = clip_orn
        self.pos_bound = pos_bound
        self.local = local

    def update_pos_bound(self, pos_bound: th.Tensor):
        self.pos_bound = pos_bound

    def clear(self, indices: th.Tensor):
        if not self.accumulate:
            return
        self.hasref[indices] = False

    def reset(self,
              state: th.Tensor,
              indices: th.Tensor,
              apply_index: bool = True
              ):
        if not self.accumulate:
            return

        if apply_index:
            if indices.dtype == th.bool:
                # self.target = th.where(indices, state, self.target)
                # self.target.masked_scatter_(indices, state)
                self.target[indices] = state[indices]
            else:
                self.target[indices] = state[indices]
        else:
            self.target[indices] = state

    @nvtx.annotate("Error.update")
    def update(self,
               state: th.Tensor,
               action: th.Tensor,
               relative: bool):
        if not relative:
            self.target = action
            return

        if self.accumulate:
            # Reset target with references.
            with nvtx.annotate("reset_target"):
                # self.target.masked_scatter_(~self.hasref[..., None], state)
                mask = (~self.hasref[..., None]).expand(self.target.shape)
                self.target[mask] = state[..., :7][mask]
                self.hasref.fill_(True)

            with nvtx.annotate("convert_quat"):
                if (action.shape[-1] == 6 or action.shape[-1] == 18 or
                        action.shape[-1] == 20):
                    # quat = axisaToquat(action[..., 3:6], action.device)
                    quat = quat_from_axa(action[..., 3:6])
                else:
                    quat = action[..., 3:7]

            # Compose transform to `target`.
            with nvtx.annotate("apply_delta"):
                self.target[..., 0:3] += action[..., 0:3]
                if quat.shape[-1] == 4:
                    if self.local:
                        self.target[..., 3:7] = quat_multiply(
                            self.target[..., 3:7], quat)
                    else:
                        self.target[..., 3:7] = quat_multiply(
                            quat, self.target[..., 3:7])

        else:
            # We directly save the `action.
            self.target = state.clone()
            self.target[..., 0:3] += action[..., 0:3]
            if (action.shape[-1] == 6 or action.shape[-1] == 18
                or action.shape[-1] == 20):
                # quat = axisaToquat(action[..., 3:6], action.device)
                quat = quat_from_axa(action[..., 3:6])
            else:
                quat = action[..., 3:7]

            if quat.shape[-1] >= 4:
                if self.local:
                    self.target[..., 3:7] = quat_multiply(
                        self.target[..., 3:7], quat)
                else:
                    self.target[..., 3:7] = quat_multiply(
                        quat, self.target[..., 3:7])
        if self.clip_pos is not None:
            # First, conditionally clamp target to workspace.
            with nvtx.annotate("clip_pos"):
                if self.pos_bound is not None:
                    self.target[..., :3].clamp_(self.pos_bound[..., 0, :3],
                                                self.pos_bound[..., 1, :3])
                # Clamp translational offset from ee state.
                self.target[..., :3].clamp_(state[..., :3] - self.clip_pos,
                                            state[..., :3] + self.clip_pos)

        if self.clip_orn is not None:
            # clamp rotational offset.
            with nvtx.annotate("clip_orn"):
                if self.local:
                    d_qxn = quat_multiply(quat_inverse(state[..., 3:7]),
                                          self.target[..., 3:7])
                else:
                    d_qxn = quat_multiply(self.target[..., 3:7],
                                          quat_inverse(state[..., 3:7]))
                d_axa = axis_angle_from_quat(d_qxn)
                angle = th.abs(
                    (th.linalg.norm(d_axa, dim=-1, keepdim=True) + th.pi) %
                    (2 * th.pi) - th.pi)
                oob_mask = (angle > self.clip_orn).squeeze(dim=-1)
                d_axa[oob_mask] = (d_axa * self.clip_orn / angle)[oob_mask]

                if self.local:
                    self.target[..., 3:7] = quat_multiply(state[..., 3:7],
                                                          quat_from_axa(d_axa))
                else:
                    self.target[..., 3:7] = quat_multiply(quat_from_axa(d_axa),
                                                          state[..., 3:7])

    @nvtx.annotate("CartesianControlError.error")
    def __call__(self, state: th.Tensor,
                 axa: bool = True):
        # if not self.accumulate:
        #     dt = self.target[..., :3] - state[..., :3]
        #     if axa:
        #         dr = -(state[:,-4:]*self.target[...,-1:]-self.target[...,-4:]*state[:,-1:]+\
        #                 th.cross(self.target[...,-4:], state[...,-4:],-1))
        #     else:
        #         dr = quat_multiply(self.target[..., 3:7],
        #                     quat_inverse(state[..., 3:7]))
        #     return th.cat([dt, dr], dim=-1)
        # else:
        #     dt = self.target[..., :3] - state[..., :3]
        #     dq = quat_multiply(self.target[..., 3:7],
        #                     quat_inverse(state[..., 3:7]))
        #     # print(self.target.shape)
        #     # print(state.shape)
        #     # print('dt', dt)
        #     # print('dq', dq)
        #     if axa:
        #         dr = axis_angle_from_quat(dq)
        #     else:
        #         dr = dq
        dt = self.target[..., :3] - state[..., :3]
        if axa:
            if False:  # not self.local:
                # "shortcut" only enabled for
                # AXA + non-local path
                dr = orientation_error_osc(
                    state[..., 3: 7],
                    self.target[..., 3: 7]
                )
            else:
                if self.local:
                    # R dR = R'
                    # dR = R^{-1} R'
                    dq = quat_multiply(quat_inverse(state[..., 3:7]),
                                       self.target[..., 3:7])
                else:
                    # dR R = R'
                    # dR = R' R^{-1}
                    dq = quat_multiply(self.target[..., 3:7],
                                       quat_inverse(state[..., 3:7]))
                dr = axis_angle_from_quat(dq)
            # dr = -(state[..., 3:6] * self.target[..., -1:] - self.target[..., 3:6]
            #        * state[:, -1:] + th.cross(self.target[..., 3:6], state[..., 3:6], -1))
        else:
            dr = quat_multiply(self.target[..., 3:7],
                               quat_inverse(state[..., 3:7]))
        return th.cat([dt, dr], dim=-1)


class JointControlError:
    def __init__(self,
                 accumulate: bool,
                 target_shape: Optional[Tuple[int, ...]] = None,
                 dtype=th.float,
                 device=None,
                 clip_pos: Optional[float] = 0.1,
                 clip_orn: Optional[float] = 0.1,
                 pos_bound: th.Tensor = None,
                 local: bool = False
                 ):
        self.pose_error = CartesianControlError(accumulate,
                                                target_shape,
                                                dtype,
                                                device,
                                                clip_pos,
                                                clip_orn,
                                                pos_bound,
                                                local)
        # self.pos_bound = pos_bound
        self.local = local
        self.device = device
        self.clip_pos = clip_pos
        self.q_target = None

        self._lmda = 0.01 * th.eye(6,
                                  dtype=th.float,
                                  device=device)

    @property
    def pos_bound(self):
        return self.pose_error.pos_bound

    def update_pos_bound(self, pos_bound: th.Tensor):
        return self.pose_error.update_pos_bound(pos_bound)

    def clear(self, index):
        return self.pose_error.clear(index)

    @nvtx.annotate("get_delta_dof_pos")
    def _get_delta_dof_pos(self, delta_pose, ik_method, jacobian):
        """Get delta Franka DOF position from delta pose using specified IK method."""
        # References:
        # 1) https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
        # 2)
        # https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
        # (p. 47)

        if ik_method == 'pinv':  # Jacobian pseudoinverse
            k_val = 1.0
            jacobian_pinv = th.linalg.pinv(jacobian)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'trans':  # Jacobian transpose
            k_val = 1.0
            jacobian_T = th.transpose(jacobian, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        elif ik_method == 'dls':  # damped least squares (Levenberg-Marquardt)
            delta_dof_pos = dls_ik(jacobian, delta_pose, self._lmda)
            # lambda_val = 0.1
            # jacobian_T = th.transpose(jacobian, dim0=1, dim1=2)
            # lambda_matrix = (
            #     lambda_val ** 2) * th.eye(n=jacobian.shape[1], device=self.device)
            # A = jacobian @ jacobian_T + lambda_matrix
            # rhs = th.linalg.solve(A, delta_pose)
            # delta_dof_pos = th.einsum('...jp,...p->...j', jacobian_T, rhs)
        elif ik_method == 'svd':  # adaptive SVD
            k_val = 1.0
            U, S, Vh = th.linalg.svd(jacobian)
            S_inv = 1. / S
            min_singular_value = 1.0e-5
            S_inv = th.where(
                S > min_singular_value,
                S_inv,
                th.zeros_like(S_inv))
            jacobian_pinv = th.transpose(Vh, dim0=1, dim1=2)[
                :, :, :6] @ th.diag_embed(S_inv) @ th.transpose(U, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)

        return delta_dof_pos

    @nvtx.annotate("IKController.update()")
    def update(self,
               state: th.Tensor,
               q_pos: th.Tensor,
               action: th.Tensor,
               j_eef,
               relative: bool = None,
               indices: Optional[th.Tensor] = None,
               update_pose: bool = True,
               recompute_orn: bool = False
               ):
        if indices is None:
            indices = th.arange(len(action),
                                dtype = th.long,
                                device = action.device)
        if update_pose:
            self.pose_error.update(state,
                                action,
                                relative)
        if self.q_target is None:
            self.q_target = q_pos.clone()
        if len(indices):
            if not recompute_orn:
                clamped_error = action[indices, :6].clone()
                clamped_error[:, :3] = self.pose_error(state)[indices, :3]
            else:
               clamped_error = self.pose_error(state)[indices]
        
            self.q_target[indices] = self._get_delta_dof_pos(
                clamped_error, 'dls', j_eef[indices]) + q_pos[indices]

    @property
    def target(self):
        return self.q_target
        # return self.pose_error.target

    def __call__(self, state, q_pos: th.Tensor, j_eef):
        # delta_pose = self.pose_error(state)
        # return self._get_delta_dof_pos(delta_pose, 'dls', j_eef)
        return self.q_target - q_pos


def _matmul(a, b):
    return th.einsum('...ij,...j->...i', a, b)


class CartesianImpedanceController:
    """
    Maybe correct
    """

    def __init__(self,
                 kp_pos: th.Tensor,
                 kd_pos: th.Tensor,
                 kp_orn: th.Tensor,
                 kd_orn: th.Tensor,
                 max_u: th.Tensor,
                 device=None,
                 OSC=True):
        self.kp_pos = kp_pos
        self.kd_pos = kd_pos
        self.kp_orn = kp_orn
        self.kd_orn = kd_orn
        self.max_u = max_u
        self.device = device
        self.OSC = OSC

    def update_gain(self, gains):
        self.kp_pos = gains[:, :3]
        self.kp_orn = gains[:, 3:6]
        self.kd_pos = gains[:, 6:9] * th.sqrt(self.kp_pos)
        self.kd_orn = gains[:, 9:] * th.sqrt(self.kp_orn)

    def __call__(self,
                 mm: th.Tensor,
                 J_e: th.Tensor,
                 ee_vel: th.Tensor,
                 pose_error: th.Tensor,
                 gains: Optional[th.Tensor] = None,
                 cache: Optional[Dict[str, th.Tensor]] = None
                 ):

        # Convert error -> control target.
        # We consider positional/orientation error differently.
        xddt = th.zeros_like(pose_error)
        xddt[:, :3] = (self.kp_pos *
                       pose_error[:, :3] -
                       self.kd_pos *
                       ee_vel[:, :3])
        xddt[:, 3:] = (self.kp_orn *
                       pose_error[:, 3:] -
                       self.kd_orn *
                       ee_vel[:, 3:])
        J_e_T = th.transpose(J_e, 1, 2)

        if self.OSC:
            # MM is PSD, so we can use cholesky_solve().
            mm_L = th.linalg.cholesky(mm)
            # mm_inv = th.cholesky_inverse(mm)
            # m_eef_inv = J_e @ mm_inv @ J_e_T
            m_eef_inv = J_e @ th.cholesky_solve(J_e_T, mm_L)
            if cache is not None:
                cache['m_eef_inv'] = m_eef_inv

            # if True:
            #     ii = th.linalg.inv(m_eef_inv)
            #     print(th.linalg.eigvals(ii))

            if True:
                wrench = th.linalg.solve(m_eef_inv, xddt)
            else:
                lhs = J_e @ th.linalg.inv(mm_L) @ J_e_T
                wrench = (th.linalg.inv(lhs) @ xddt[..., None]).squeeze(dim=-1)

        else:
            wrench = xddt

        # u = J_e^{T}@(J_e@MM^{-1}@J_e^T)^{-1} @ xddt
        u = th.einsum(
            '...ij, ...j -> ...i', J_e_T, wrench)

        # Clip the values to be within valid effort range
        u.clamp_(-self.max_u, self.max_u)
        return u


class IKController:
    """
    Maybe correct
    """

    def __init__(self,
                 kp: th.Tensor,
                 max_u: th.Tensor,
                 num_env,
                 device=None):
        self.kp = th.full((num_env, 7), kp,
                          dtype=th.float,
                          device=device)
        self.kd =  th.full((num_env, 7), np.sqrt(kp),
                          dtype=th.float,
                          device=device)
        self.max_u = max_u
        self.device = device

    def update_gain(self, gains,
                    indices = None):
        if indices is None:
            self.kp = gains[:, :7]
            self.kd = gains[:, 7:] * th.sqrt(self.kp)
        else:
            if len(indices)>0:
                self.kp[indices, :] = gains[indices, :7]
                self.kd[indices, :] = (gains[indices, 7:] *
                                        th.sqrt(self.kp[indices, :]))

    def __call__(self,
                 mm: th.Tensor,
                 q_vel: th.Tensor,
                 joint_error: th.Tensor,
                 cache: Optional[Dict[str, th.Tensor]] = None
                 ):
        u = (self.kp * joint_error - self.kd * q_vel)
        # .unsqueeze(-1)
        # u = (mm @ u).squeeze(-1)

        # Clip the values to be within valid effort range
        u.clamp_(-self.max_u, self.max_u)
        return u


@th.jit.script
def dls_ik(J: th.Tensor, p: th.Tensor, lmda: th.Tensor) -> th.Tensor:
    J_T = th.swapaxes(J, -2, -1)
    A = (J @ J_T).add_(lmda)
    # u = th.linalg.cholesky(A)
    # return (J_T @ th.cholesky_solve(p, u.unsqueeze(-1))).squeeze(-1)
    return (J_T @ th.linalg.solve(A, p)[..., None])[..., 0]


def get_analytic_jacobian(
        fingertip_quat, fingertip_jacobian, num_envs, device):
    """Convert geometric Jacobian to analytic Jacobian."""
    # Reference: https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2018/RD_HS2018script.pdf
    # NOTE: Gym returns world-space geometric Jacobians by default

    batch = num_envs

    # Overview:
    # x = [x_p; x_r]
    # From eq. 2.189 and 2.192, x_dot = J_a @ q_dot = (E_inv @ J_g) @ q_dot
    # From eq. 2.191, E = block(E_p, E_r); thus, E_inv = block(E_p_inv, E_r_inv)
    # Eq. 2.12 gives an expression for E_p_inv
    # Eq. 2.107 gives an expression for E_r_inv

    # Compute E_inv_top (i.e., [E_p_inv, 0])
    I = th.eye(3, device=device)
    E_p_inv = I.repeat((batch, 1)).reshape(batch, 3, 3)
    E_inv_top = th.cat(
        (E_p_inv, th.zeros((batch, 3, 3), device=device)), dim=2)

    # Compute E_inv_bottom (i.e., [0, E_r_inv])
    fingertip_axis_angle = axis_angle_from_quat(fingertip_quat)
    fingertip_axis_angle_cross = get_skew_symm_matrix(
        fingertip_axis_angle, device=device)
    fingertip_angle = th.linalg.vector_norm(fingertip_axis_angle, dim=1)
    factor_1 = 1 / (fingertip_angle ** 2)
    factor_2 = 1 - fingertip_angle * 0.5 * th.sin(
        fingertip_angle) / (1 - th.cos(fingertip_angle))
    factor_3 = factor_1 * factor_2
    E_r_inv = I - 1 * 0.5 * fingertip_axis_angle_cross + (
        fingertip_axis_angle_cross @ fingertip_axis_angle_cross) * factor_3.unsqueeze(-1).repeat((1, 3 * 3)).reshape((batch, 3, 3))
    E_inv_bottom = th.cat(
        (th.zeros((batch, 3, 3), device=device), E_r_inv), dim=2)

    E_inv = th.cat(
        (E_inv_top.reshape((batch, 3 * 6)),
         E_inv_bottom.reshape((batch, 3 * 6))),
        dim=1).reshape(
        (batch, 6, 6))

    J_a = E_inv @ fingertip_jacobian

    return J_a


def get_skew_symm_matrix(vec, device):
    """Convert vector to skew-symmetric matrix."""
    # Reference:
    # https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication

    batch = vec.shape[0]
    I = th.eye(3, device=device)
    skew_symm = th.transpose(
        th.cross(
            vec.repeat((1, 3)).reshape((batch * 3, 3)),
            I.repeat((batch, 1))).reshape(batch, 3, 3),
        dim0=1, dim1=2)

    return skew_symm


def translate_along_local_z(pos, quat, offset, device):
    """Translate global body position along local Z-axis and express in global coordinates."""

    num_vecs = pos.shape[0]
    offset_vec = offset * th.tensor([0.0, 0.0, 1.0],
                                    device=device).repeat((num_vecs, 1))
    _, translated_pos = torch_utils.tf_combine(q1=quat, t1=pos, q2=th.tensor(
        [0.0, 0.0, 0.0, 1.0], device=device).repeat((num_vecs, 1)), t2=offset_vec)

    return translated_pos


def axis_angle_from_euler(euler):
    """Convert tensor of Euler angles to tensor of axis-angles."""

    quat = torch_utils.quat_from_euler_xyz(
        roll=euler[:, 0],
        pitch=euler[:, 1],
        yaw=euler[:, 2])
    quat = quat * th.sign(quat[:, 3]).unsqueeze(-1)  # smaller rotation
    axis_angle = axis_angle_from_quat(quat)

    return axis_angle


def axis_angle_from_quat(quat, eps=1.0e-6):
    """Convert tensor of quaternions to tensor of axis-angles."""
    # Reference:
    # https://github.com/facebookresearch/pytorch3d/blob/bee31c48d3d36a8ea268f9835663c52ff4a476ec/pytorch3d/transforms/rotation_conversions.py#L516-L544

    if True:
        axis = F.normalize(quat[..., 0:3])

        half_angle = th.acos(quat[..., 3:].clamp(-1.0, +1.0))
        angle = (2.0 * half_angle + th.pi) % (2 * th.pi) - th.pi
        return axis * angle

    if False:
        # axis = quat[..., :3] # sin(h/2) * axis
        # w = quat[..., 3:]
        h = 2.0 * th.acos(quat[..., 3:].clamp(-1.0, +1.0)) / th.pi
        return quat[..., :3] / th.sinc(h)

        # scale factor: h/2 / cos(h/2)

        # [1]
        # norm(quat[..., :3]) = sin(h/2)

        # out[..., :3] = uvec(quat[..., :3])
        # out[..., 3:] = 2 * th.arccos(np.clip(w, -1.0, 1.0))
        # return quat[..., :3] * (2 * th.acos(w.clamp(-1.0, +1.0))) / th.linalg.norm(
        #         quat[..., :3], dim=-1, keepdim=True))
    else:
        mag = th.linalg.norm(quat[..., 0:3], dim=-1,
                             keepdim=True)
        half_angle = th.atan2(mag, quat[..., 3])

        axis = F.normalize(quat[..., 0:3])

        # print('half-angle', half_angle)
        # angle = 2.0 * half_angle
        # print('angle', angle)
        # th.sinc(half_angle / th.pi)  = sin(half_angle) / half_angle
        sin_half_angle_over_angle = th.where(th.abs(angle) > eps,
                                             th.sin(half_angle) / angle,
                                             1 / 2 - angle ** 2.0 / 48)
        # sin_half_angle_over_angle = 0.5 * th.sinc(half_angle / th.pi)
        # print('sin_half', sin_half_angle_over_angle)
        axis_angle = quat[:, 0:3] / sin_half_angle_over_angle

    return axis_angle


def axisaToquat(axisA: th.Tensor, device):

    num_rotations = axisA.shape[0]
    angle = th.norm(axisA, dim=-1)
    small_angle = (angle <= 1e-3)
    large_angle = ~small_angle

    scale = th.empty((num_rotations,), device=device, dtype=th.float)
    scale[small_angle] = (0.5 - angle[small_angle] ** 2 / 48 +
                          angle[small_angle] ** 4 / 3840)
    scale[large_angle] = (th.sin(angle[large_angle] / 2) /
                          angle[large_angle])
    quat = th.empty((num_rotations, 4), device=device, dtype=th.float)
    quat[:, :3] = scale[:, None] * axisA
    quat[:, -1] = th.cos(angle / 2)
    return quat
