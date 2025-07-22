#!/usr/bin/env python3

from isaacgym import gymapi

import pkg_resources
from typing import (
    Tuple, Optional, Dict,
    Iterable, List)
import trimesh
from dataclasses import dataclass, replace
from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper
from env.env.help.with_nvdr_camera import (
    BoxArg, camera_matrices, get_link_mesh)
import pickle

import torch as th
import numpy as np
from gym import spaces
from einops import rearrange

from env.env.base import TrackObjectAssets
from env.robot.cube_poker import CubePoker
from env.env.help.urdf_geom import (
    load_hull_objects,
    pack_shapes
)
from env.env.wrap.base import add_obs_field
from env.env.wrap.raycast_util import vectorized_raycast
from models.common import merge_shapes
from util.path import ensure_directory
from util.vis.img import tile_images
from util.torch_util import (
    dcn,
    masked_sample
)
from util.config import ConfigBase

from env.util import draw_cloud_with_mask
from util.math_util import (
    matrix_from_quaternion,
    apply_pose_tq,
    quat_rotate
)

import nvtx
from icecream import ic
# from dgl.ops import gather_mm

_COUNT: int = 0
_DELAY: int = 10

_DEBUG_UNPROJECT: bool = False

KEY_DEPTH: str = 'depth'
KEY_COLOR: str = 'color'
KEY_LABEL: str = 'label'
KEY_FLOW: str = 'flow'
KEY_RAW: str = 'raw'
KEY_CLOUD: str = 'cloud'

# maximum number of planes in a
# convex hull "part"
MAX_PLANE: int = 64


import torch as th


def flatten_varlen(x):
    idx = []
    offset: int = 0
    for y in x:
        # idx.extend(len(idx) + np.arange(len(y)))
        idx.append(offset + np.arange(len(y)))
        offset += len(y)
    out = sum(x, [])
    return out, idx


def invert_index(a_from_b):
    """
    invert a unique index mapping.
    No guarantees about outputs when a_from_b contains duplicates.
    """
    b_from_a = np.arange(a_from_b.max() + 1)
    b_from_a[a_from_b] = np.arange(a_from_b.shape[-1])
    return b_from_a


def compose_index(a_from_b, b_from_c):
    return a_from_b[b_from_c]


def convert_hull(hull: trimesh.Trimesh,
                 max_num_planes: Optional[int] = None) -> th.Tensor:
    normals = hull.face_normals
    centers = hull.triangles_center

    if max_num_planes is not None:
        pad_count = (max_num_planes - normals.shape[-2])
        normals = np.pad(
            normals, ((0, pad_count), (0, 0)),
            mode='edge')
        centers = np.pad(centers,
                         ((0, pad_count), (0, 0)),
                         mode='edge')
    # hull_eqn = np.concatenate([normals, centers], axis=-1)
    hull_eqn = np.concatenate([centers, normals], axis=-1)
    # tri_normal = th.as_tensor(normals)
    # tri_origin = th.as_tensor(centers)
    # hull_eqn = th.cat([tri_normal, tri_origin], dim=-1)
    return hull_eqn


def pad_and_convert_hulls(hulls: List[trimesh.Trimesh]):
    """
    Convert hulls into a set of padded hull equations.
    Output: ~HxPx4 Tensor~

    where
    * H = number of hulls,
    * P = number of planes (padded to same length)
    # 4 = hull equation dimensions
    """
    max_num_planes = max([len(hull.faces) for hull in hulls])

    hull_eqns = []
    for hull in hulls:
        hull_eqn = th.as_tensor(
            convert_hull(hull, max_num_planes)
        )
        hull_eqns.append(hull_eqn)
    return th.stack(hull_eqns, dim=0).to(dtype=th.float)


def ray_hull_intersection(
        tri_origin: Optional[th.Tensor],
        tri_normal: th.Tensor,
        ray_origin: th.Tensor,
        ray_vector: th.Tensor,
        tri_offset: Optional[th.Tensor] = None,
        eps: float = 1e-6,

        hull_axis: Optional[int] = 1,

        # a single hull is split into multiple parts
        part_axis: Optional[int] = None
):

    # Distance from origin to triangle plane
    # Technically (and practically) precomputable
    if tri_offset is not None:
        d = tri_offset
    else:
        d = th.einsum('...fi, ...fi -> ...f',
                      tri_normal, tri_origin)

    # Denominator
    # ic(tri_normal.shape)
    # ic(ray_vector.shape)
    vd = th.einsum('...fi, ...ri -> ...fr',
                   tri_normal, ray_vector)
    # Numerator
    vn = (d[..., None] - th.einsum('...fi, ...ri -> ...fr', tri_normal,
                                   ray_origin))
    # Distance from ray origin to surface
    t = vn / vd

    # We consider some faces with invalid normals to be at infinity
    t[(th.linalg.norm(tri_normal, axis=-1) <= eps)] = th.inf

    # Data buffers for front/back-face reasoning
    front = (vd < 0.0)

    # Resolve which face was hit
    # (we don't consider backfaces)
    t_lo = th.where(front, t, th.as_tensor(-th.inf, dtype=t.dtype,
                    device=t.device))  # used against t_near
    t_hi = th.where(front, th.as_tensor(+th.inf, dtype=t.dtype,
                    device=t.device), t)  # used against t_far
    t_near = th.max(t_lo, axis=-2).values
    t_far = th.min(t_hi, axis=-2).values

    # In case `hull_axis` is not None,
    # we reduce across multiple hulls
    if hull_axis is not None:
        # 16, 19, 512, 3
        t_near = th.min(t_near, axis=hull_axis).values
        t_far = th.max(t_far, axis=hull_axis).values

    # Check if ray is valid
    valid = (t_near > 0.0) & (t_far + eps >= t_near)  # & th.isfinite(t_near)
    return t_near, valid

# def scatter_raycast():
#    hulls:th.Tensor = None # M hulls x P planes x 6
#    env_hulls = gather_mm(
#            hulls, # M hulls x P planes x 2 x 3(4) => (MxPx2, 4)
#            body_poses, # N envs x B bodies x 4x4 => (NxB, 4, 4)
#            hull_indices, # select which hull: (R,)
#            body_indices, # select which body pose
#            ) # output:  (R, 4)
#    gather_mm(
#            env_hulls,
#            rays,
#
#    # th.segment_reduce(env_hulls,


def asset_key(asset_args):

    if 'width' in asset_args:
        # BOX
        extents = (
            float(dcn(asset_args['width'])),
            float(dcn(asset_args['height'])),
            float(dcn(asset_args['depth']))
        )
        box_arg = BoxArg(extents)
        # NOTE: exact check here
        # might be quite problematic due to
        # floating-point mismatch...
        # if is_new_asset:
        #     object_urdfs.append(box_arg)
        # if box_arg not in object_urdfs:
        #    is_new_asset = True
        #    object_urdfs.append(box_arg)
        key = (box_arg)
        return key

    if 'filename' in asset_args:
        # URDF
        asset_root = asset_args['rootpath']
        asset_file = asset_args['filename']
        asset_path = F'{asset_root}/{asset_file}'

        # NOTE: technically quite slow
        # maybe use ordereddict instead? :)
        # if is_new_asset:
        #     object_urdfs.append(asset_path)
        # if asset_path not in object_urdfs:
        #    is_new_asset = True
        #    object_urdfs.append(asset_path)
        key = (asset_path)
        return key
    return ValueError(F'Invalid asset_args = {asset_args}')


def get_body_index_map(gym, env,
                       object_urdfs,
                       urdf_link_map,
                       geom_index):
    num_actor = gym.get_actor_count(env)
    num_body: int = gym.get_env_rigid_body_count(env)
    geom_from_body = [None for _ in range(num_body)]
    # num_geom = 14  # ???
    # body_from_geom = [None for _ in range(num_geom)]

    for i in range(num_actor):
        actor_handle = gym.get_actor_handle(env, i)
        asset = gym.get_actor_asset(env, actor_handle)
        body_count = gym.get_actor_rigid_body_count(env,
                                                    actor_handle)
        asset_args = gym.actor_args[env, actor_handle]

        # Parse asset args / key
        is_new_asset: bool = False
        key = asset_key(asset_args)
        if key not in object_urdfs:
            is_new_asset = True
            object_urdfs.append(key)

        # Register new asset
        if is_new_asset:
            body_names = gym.get_asset_rigid_body_names(
                asset)
            urdf_link_map[key] = body_names

            # NOTE: we assume that
            # the geometries are loaded in the order
            # of `body_name`.
            # FIXME: is using len(geom_index) safe??
            for body_name in body_names:
                # print(key, body_name)
                if (key, body_name) not in geom_index:
                    geom_index[(key, body_name)] = len(geom_index)

        # Map body names to body indices
        # ultimately to loaded geometry index
        body_names = urdf_link_map[key]
        for ii in range(body_count):
            body_name: str = body_names[ii]
            body_index = gym.find_actor_rigid_body_index(
                env,
                actor_handle,
                body_name,
                gymapi.IndexDomain.DOMAIN_ENV
            )
            # ic(key, body_name, body_index)
            # sel = geom_from_body
            # need: body_from_hull = body_from_link x link_from_hull
            geom_from_body[body_index] = geom_index[(key, body_name)]
            # body_from_geom[geom_index[(key, body_name)]] = body_index
    return geom_from_body

    # We require that all bodies in `geom_from_body` are associated
    # with a valid geometry index.
    # assert ((None not in geom_from_body))
    # ic(geom_from_body)
    # raise ValueError('stop')
    # return np.asarray(geom_from_body)
    # body_from_geom = invert_index(np.asarray(geom_from_body))
    # ic(body_from_geom)
    # return np.asarray(body_from_geom)


def get_hacky_blocklist_for_arm_links():
    asset_root: str = pkg_resources.resource_filename(
        'data', 'assets')
    # FIXME: hardcoded `franka_file` ... may not work in general
    # franka_file: str = 'franka_description/robots/franka_panda_fixed_finger.urdf'
    franka_file: str = 'crm-panda/robots/franka_panda_fixed_finger.urdf'
    franka_path = F'{asset_root}/{franka_file}'
    ur5_fe_file: str = 'ur5-fe/robot.urdf'
    ur5_fe_path = F'{asset_root}/{ur5_fe_file}'
    table_file: str = 'table-with-wall/table-with-wall-small.urdf'
    table_path = F'{asset_root}/{table_file}'
    blocklist = {
        franka_path: [
            'panda_link0', 'panda_link1', 'panda_link2',
            'panda_link3', 'panda_link4', 'panda_link5',
            'panda_link6', 'panda_link7',
            # 'panda_hand',
            'panda_tool',
            # 'panda_leftfinger',
            # 'panda_rightfinger'
        ],
        ur5_fe_path: [
            'base_link',
            'shoulder_link',
            'upper_arm_link',
            'forearm_link',
            'wrist_1_link',
            'wrist_2_link',
            'wrist_3_link',
            'ee_link',
            'base',
            'tool0',
            'tool_tip',
            'world',
            'rotated_base_link',
            # 'panda_hand',
            # 'panda_leftfinger',
            # 'panda_rightfinger'
        ],
        table_path: ['base_link']
    }
    print(F'hide_arm : blocklist = {blocklist}')
    return blocklist


class RaycastCameraWrapper(ObservationWrapper):
    """
    Raycast from camera to a set of object points,
    and take the argmin among chulls.
    """

    @dataclass
    class Config(ConfigBase):
        # Camera extrinsics parameterization
        # eye: Tuple[float, float, float] = (0.5, 0.0, 1.5)
        eye: Tuple[float, float, float] = (0.362, 0.0, 0.747)
        at: Tuple[float, float, float] = (0.0, 0.0, 0.55)
        up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
        hide_arm: bool = False
        key_cloud: str = 'cloud'
        cloud_size: int = 512

    def __init__(self, env: EnvIface, cfg: Config):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg

        self.__aux = {}
        object_urdfs = []
        urdf_link_map = {}
        # Map urdf file + body --> actual geometry index
        geom_index: Dict[Tuple[str, str], int] = {}

        # Map body index to corresponding geometry index
        # FIXME: we assume same number of bodies per env,
        # for now
        # body_from_geoms = []
        geom_from_body = []  # N x Z[B->G]
        for e in env.envs:
            num_actors = env.gym.get_actor_count(e)
            num_bodies: int = env.gym.get_env_rigid_body_count(e)
            geom_from_body_i = get_body_index_map(env.gym, e,
                                                  object_urdfs,
                                                  urdf_link_map,
                                                  geom_index,
                                                  # body_from_geom_dict
                                                  )
            geom_from_body.append(geom_from_body_i)
            # body_from_geoms.append(body_from_geom)
            # break
            # "geom" is assumed to be an array
            # organized in the order of object_urdfs -> urdf_link_map
            # in other words, it's a flattened list of links (with geom) in
            # urdfs
        geom_from_body = np.asarray(geom_from_body)
        self.geom_from_body = geom_from_body

        # dict -> array
        # body_from_geom = np.zeros(max(body_from_geom_dict.keys()) + 1,
        #                          dtype=np.int32)
        # for k, v in body_from_geom_dict.items():
        #    body_from_geom[k] = v
        # raise ValueError('stop')

        if cfg.hide_arm:
            blocklist = get_hacky_blocklist_for_arm_links()
        else:
            blocklist = None

        # NOTE: trying to automatically figure out
        # the number of bodies in the environment !!
        # To be more rigorous, we could sweep through all envs and
        # ensure they all have the same number of bodies...
        self.num_body = env.gym.get_env_rigid_body_count(env.envs[0])

        # num_poses = (self.num_body + 1 if cfg.use_ground else self.num_body)
        # self.body_poses_4x4 = th.zeros(
        #    (env.num_env, num_poses, 4, 4),
        #    dtype=th.float32,
        #    device=env.device)
        # self.body_poses_4x4[:] = th.eye(4,
        #                                dtype=th.float, device=env.device)

        # [1] get object / robot urdf files
        # [2] load corresponding link-maps
        hullss = load_hull_objects(
            object_urdfs=object_urdfs,
            urdf_link_map=urdf_link_map,
            load_ground=False,
            blocklist=blocklist
        )  # L x (h_l x {trimesh})
        # ic(hullss)
        # print(len(hullss), len(object_urdfs)) # 13, 3
        # print([len(urdf_link_map.get(urdf)) for urdf in object_urdfs]) # 1,1,12 ??
        # for hulls in hullss:
        #     print(len(hulls))

        self.__hulls, self.__body_hull_ids = flatten_varlen(hullss)
        # ic(self.__body_hull_ids)  # List[List[int]]
        self.__hulls = [th.as_tensor(convert_hull(h),
                                     dtype=th.float,
                                     device=self.device)
                        for h in self.__hulls]
        self.__env_body_ids = []
        # hulls: (num_env_links) X (num_hulls_per_link) x trimesh

        # [1] build env-specific hulls list
        self.hulls = []  # N hulls
        self.hull_sizes = []
        self.pose_ids = []

        self.hull_sizes = []
        self.num_hulls = []

        # for env_id, geoms in enumerate(self.geom_from_body):
        #     for body_index, geom_index in enumerate(geoms):

        for env_id, geoms in enumerate(self.geom_from_body):
            env_hulls = []
            env_body_ids = []
            env_hull_sizes = []
            # self.__env_body_ids[env_id]=[]
            self.__env_body_ids.append([])  # [env_id]=[]
            for body_index, geom_index in enumerate(geoms):
                # hulls = hullss[geom_index]
                # hulls = self.__hulls[geom_index]

                # hmm.... might be dangerous
                # if len(hulls) <= 0:
                #    continue

                # hull_sizes = [len(hull.face_normals)
                #              for hull in hulls]
                # hulls = [convert_hull(hull,
                #                      # max_num_planes
                #                      None
                #                      )
                #         for hull in hulls]
                # if len(hulls) <= 0:
                #    continue
                # hulls = np.asarray(hulls, dtype=np.float32)
                # hulls = np.concatenate(hulls, axis=0)

                # pad_count = max_num_hulls - hulls.shape[0]
                # if len(hulls) < 0:
                # continue
                # ic(hulls.shape) # empty?
                # hulls = np.pad(hulls, ( (0, pad_count), (0,0), (0,0)),
                # mode='edge')
                # env_hulls.extend(hulls)
                # env_hull_sizes.extend(hull_sizes)
                # env_body_ids.extend([geom_index] * len(hulls))
                # env_body_ids.extend([body_index] * len(hulls))
                self.__env_body_ids[env_id].append([body_index, geom_index])
            # env_hulls = np.asarray(env_hulls, dtype=np.float32)
            # env_hulls = np.concatenate(env_hulls, axis=0)
            # self.hulls.append(env_hulls)
            # self.hull_sizes.append(env_hull_sizes)
            # self.pose_ids.append(env_body_ids)
        # max_num_hulls_per_env = max([h.shape[0] for h in self.hulls])
        # self.hulls = [np.pad(h,
        #                      ((0, max_num_hulls_per_env - len(h)),
        #                          (0, 0),
        #                          (0, 0)), mode='edge')
        #               for h in self.hulls]
        # self.pose_ids = [np.pad(h,
        #                         ((0, max_num_hulls_per_env - len(h))),
        #                         mode='edge')
        #                  for h in self.pose_ids]

        # self.hulls = np.asarray(self.hulls, dtype=np.float32)
        # self.pose_ids = np.asarray(self.pose_ids, dtype=np.int32)

        # self.hulls = th.as_tensor(self.hulls,
        #                           dtype=th.float,
        #                           device=self.device)
        # self.pose_ids = th.as_tensor(self.pose_ids,
        #                              dtype=th.long,
        #                              device=self.device)

        if False:
            if True:
                if False:
                    hull_flat, hull_from_geom = pack_shapes(hullss)
                    hull_from_geom = np.asarray(hull_from_geom)
                    # hull_from_geom = invert_index(geom_from_hull)
                    hulls = pad_and_convert_hulls(hull_flat)  # => H x P x 4
                    self.hulls = hulls.to(device=self.device,
                                          dtype=th.float32)
                    # This means hulls per body I guess
                    # self.hulls = self.hulls[self.body_from_hull]
                    # print(self.geom_from_body.shape)  # 16, 14
                    # print(self.hulls.shape)  # 220, 512, 6
                    self.hulls = self.hulls[hull_from_geom[self.geom_from_body],
                                            :, :]  # (16, 14, 512, 6)
                else:
                    # hull_flat, hull_from_geom = pack_shapes(hullss)
                    # hulls = pad_and_convert_hulls(hull_flat)  # => H x P x 4
                    # hullss: geom_index -> hulls

                    max_num_hulls = 0
                    max_num_planes = 0
                    for hulls in hullss:
                        max_num_hulls = max(max_num_hulls, len(hulls))
                        for hull in hulls:
                            # ic(len(hull.faces))
                            max_num_planes = max(
                                max_num_planes, len(hull.faces))

                    # [1] build env-specific hulls list
                    self.hulls = []
                    self.lens = []
                    self.pose_ids = []
                    for env_id, geoms in enumerate(self.geom_from_body):
                        env_hulls = []
                        env_body_ids = []
                        env_lens = []
                        for body_index, geom_index in enumerate(geoms):
                            hulls = hullss[geom_index]

                            # hmm.... might be dangerous
                            if len(hulls) < 0:
                                continue

                            hulls = [convert_hull(hull,
                                                  # max_num_planes
                                                  None
                                                  )
                                     for hull in hulls]
                            # hulls = np.asarray(hulls, dtype=np.float32)
                            lens = [len(hull.face_normals)
                                    for hull in hulls]
                            hulls = np.concatenate(hulls, axis=0)

                            # pad_count = max_num_hulls - hulls.shape[0]
                            # if len(hulls) < 0:
                            #    continue
                            # ic(hulls.shape) # empty?
                            # hulls = np.pad(hulls, ( (0, pad_count), (0,0), (0,0)),
                            #               mode='edge')
                            env_hulls.extend(hulls)
                            env_lens.extend(lens)
                            # env_body_ids.extend([geom_index] * len(hulls))
                            env_body_ids.extend([body_index] * len(hulls))
                        # env_hulls = np.asarray(env_hulls, dtype=np.float32)
                        env_hulls = np.concatenate(env_hulls, axis=0)
                        self.hulls.append(env_hulls)
                        self.lens.append(env_lens)
                        self.pose_ids.append(env_body_ids)
                    max_num_hulls_per_env = max(
                        [h.shape[0] for h in self.hulls])
                    self.hulls = [np.pad(h,
                                         ((0, max_num_hulls_per_env - len(h)),
                                          (0, 0),
                                          (0, 0)), mode='edge')
                                  for h in self.hulls]
                    self.pose_ids = [np.pad(h,
                                            ((0, max_num_hulls_per_env - len(h))),
                                            mode='edge')
                                     for h in self.pose_ids]

                    self.hulls = np.asarray(self.hulls, dtype=np.float32)
                    self.pose_ids = np.asarray(self.pose_ids, dtype=np.int32)

                    self.hulls = th.as_tensor(self.hulls,
                                              dtype=th.float,
                                              device=self.device)
                    self.pose_ids = th.as_tensor(self.pose_ids,
                                                 dtype=th.long,
                                                 device=self.device)

                    # expand via selection:
                    # basically, repeat link poses by N times
                    # for each hull that the link has
                    # poses: body_index -> body_pose
                    # poses[body_from_geom] : geom_index -> geom_pose
                    # map of "which hull corresponds to which body"?
                    # poses -> hull_poses

                    # expand via selection:
                    # hulls: hull_index -> hull
                    # env_hulls: env_index -> body_index -> hull

                    # hulls -> env_hulls
                    # env_hulls = hull[hull_from_body @ body_from_env_id]

            else:
                pass
                # max_num_hulls = 0
                # max_num_planes = 0
                # for hulls in hullss:
                #    max_num_hulls = max(max_num_hulls, len(hulls))
                #    for hull in hulls:
                #        max_num_planes = max(max_num_planes, len(hull.faces))

                # self.hulls = np.zeros((self.num_env, max_num_hulls, max_num_planes, 6),
                #                 dtype=np.float32)
                # geom_from_hull = []
                # geom_index = 0
                # for i, hulls in enumerate(hullss):
                #    for j, hull in enumerate(hulls):
                #        self.hulls[i, j] = convert_hull(hull, max_num_planes)
                #        geom_from_hull.append(i)
                # self.hulls[?, i, len(hulls):]= self.hulls[?, i,
                # len(hulls)-1:len(hulls)])

                # self.hulls = th.as_tensor(self.hulls,
                #                          dtype=th.float,
                #                          device=self.device)
                # geom_from_hull = th.as_tensor(np.asarray(geom_from_hull),
                #                              dtype=th.long,
                #                              device=self.device)

            # ic(body_from_geom)
            # self.body_from_hull = compose_index(
            #    body_from_geom, geom_from_hull)
            # self.body_from_hull = th.as_tensor(self.body_from_hull,
            #                                   dtype=th.long,
            #                                   device=self.device)

        # Setup observation space...
        self._make_dict = False
        self._obs_space, self._update_obs = add_obs_field(
            env.observation_space, cfg.key_cloud, spaces.Box(
                -float('inf'), +float('inf'), (cfg.cloud_size, 3), dtype=np.float32))

    def __prepare_indices(self,
                          base_hulls,
                          body_hull_ids,
                          env_body_ids,
                          ray_origin,
                          ray_vector,
                          body_poses: th.Tensor):
        all_hull_ids = [[body_hull_ids[body_id]
                        for _, body_id in env_body_ids[env_id]]
                        for env_id in range(num_env)]
        env_hull_ids = [np.concatenate(h) for h in all_hull_ids]
        all_hull_ids_flat = np.concatenate(env_hull_ids)
        all_hulls = th.cat([base_hulls[h]
                            for h in all_hull_ids_flat])

    @property
    def observation_space(self):
        return self._obs_space

    def raycast(self, poses: th.Tensor):
        cfg = self.cfg
        #                                            # should be more like 64, 10, 512, 6
        # print(self.hulls.shape, self.hulls.dtype)  # 40,512,6

        # 64,40,1,7 ; 64,40,512,6
        # tri_normal = quat_rotate(poses[..., None, 3:7],
        #                          self.hulls[None, ..., :3])
        # tri_origin = apply_pose_tq(poses[..., None, :],
        #                            self.hulls[None, ..., 3:])
        tri_normal = quat_rotate(poses[..., None, 3:7],
                                 self.hulls[..., :3])
        tri_origin = apply_pose_tq(poses[..., None, :],
                                   self.hulls[..., 3:])
        eye = th.as_tensor(cfg.eye, dtype=th.float,
                           device=self.device)[None]

        # Object point cloud
        cur_obj_pose = self.tensors['root'][
            self.scene.cur_ids.long(), :7]
        vertices = apply_pose_tq(cur_obj_pose[..., None, :],
                                 self.scene.cur_cloud)

        ray_origin = eye
        ray_vector = vertices - eye
        ray_length = th.linalg.norm(ray_vector, dim=-1, keepdim=True)
        ray_vector /= ray_length
        # ic(tri_origin.shape)
        # ic(tri_normal.shape)
        # ic(ray_origin.shape)
        # ic(ray_vector.shape)
        distance, hit = ray_hull_intersection(
            # hull
            tri_origin,
            tri_normal,
            # ray
            ray_origin[..., None, :].expand(ray_vector.shape)[:, None],
            ray_vector[:, None],
        )
        # distance[~hit] = th.inf
        # cloud = ray_origin[..., None, :] + ray_vector * distance[..., None]
        cloud = vertices

        # valid = hit

        # Apply object mask
        delta = (distance - ray_length.squeeze(dim=-1))[hit]
        # ic(delta.min(), delta.max(), delta.mean(), delta.std())
        valid = hit

        # index = th.argmin(distance, dim=1, keepdim=True)

        # cloud = (
        #    ray_origin[..., None, :]
        #    + (ray_vector * th.take_along_dim(
        #        distance, index, dim=1).squeeze(dim=1)[..., None])
        # )
        # valid = th.take_along_dim(hit,
        #                          index, dim=1).squeeze(dim=1)
        return cloud, valid

    @nvtx.annotate("RaycastCameraWrapper._wrap_obs")
    def _wrap_obs(self, obs) -> th.Tensor:
        cfg = self.cfg
        body_tensors = self.env.tensors['body']
        body_tensors = body_tensors.reshape(self.num_env, -1, 13)
        # ic(body_tensors.shape)  # 4, 14, 13
        body_poses_7dof = body_tensors[:, :self.num_body, ..., :7]

        # N x H x P x 4
        # where N = num_env
        # H = num_hulls
        # P = hull size (number of planes)
        # 4 = hull equation dimensionality; (3+1) for normal + offset
        # print(self.body_from_hull)
        # hull_poses_7dof = body_poses_7dof[:, self.body_from_hull]
        # all_cloud, hit = self.raycast(hull_poses_7dof)
        # ic(self.pose_ids.shape)
        # hull_poses = th.take_along_dim(body_poses_7dof,
        #                                self.pose_ids[..., None], dim=1)
        # all_cloud, hit = self.raycast(hull_poses)

        eye = th.as_tensor(cfg.eye, dtype=th.float,
                           device=self.device)[None]

        # Compute ray
        with nvtx.annotate("compute_ray"):
            cur_obj_pose = self.tensors['root'][
                self.scene.cur_ids.long(), :7]
            vertices = apply_pose_tq(cur_obj_pose[..., None, :],
                                     self.scene.cur_cloud)
            ray_origin = eye
            ray_vector = vertices - eye
            ray_length = th.linalg.norm(ray_vector, dim=-1, keepdim=True)
            ray_vector /= ray_length
        # ic(self.__hulls)
        # ic(self.__body_hull_ids)
        # ic(self.__env_body_ids)

        with nvtx.annotate("vectorized_raycast"):
            with th.inference_mode():
                distance = vectorized_raycast(
                    self.__hulls,
                    self.__body_hull_ids,
                    self.__env_body_ids,
                    ray_origin[..., None, :].expand(ray_vector.shape),
                    ray_vector,
                    body_poses_7dof,
                    aux=self.__aux)

        with nvtx.annotate("make_and_sample_cloud"):
            # print(ray_length, distance)
            # hit = (th.abs(ray_length.squeeze(dim=-1) - distance) <= 1e-3)
            hit = th.isfinite(distance)

            cloud = ray_origin[..., None, :] + ray_vector * distance[..., None]

            # print(all_cloud.mean())
            # print(hit.float().mean())

            # cloud =
            cloud = masked_sample(cloud, hit, eps=1e-6,
                                  num_samples=cfg.cloud_size)
        return self._update_obs(obs, cloud)
