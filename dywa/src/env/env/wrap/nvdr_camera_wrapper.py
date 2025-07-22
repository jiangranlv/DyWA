#!/usr/bin/env python3

from isaacgym import gymapi

import pkg_resources
from typing import Tuple, Optional, Dict
from dataclasses import dataclass, replace
from env.env.iface import EnvIface
from env.env.wrap.base import ObservationWrapper
from env.env.help.with_nvdr_camera import (
    WithNvdrCamera, BoxArg, camera_matrices)
import pickle

import torch as th
import numpy as np
from gym import spaces
from einops import rearrange

from env.env.base import TrackObjectAssets
from env.robot.cube_poker import CubePoker
from models.common import merge_shapes
from util.path import ensure_directory
from util.vis.img import tile_images
from util.torch_util import dcn
from util.config import ConfigBase

from env.util import draw_cloud_with_mask
from util.math_util import matrix_from_quaternion

import nvtx
from icecream import ic

_COUNT: int = 0
_DELAY: int = 10

_DEBUG_UNPROJECT: bool = False

KEY_DEPTH: str = 'depth'
KEY_COLOR: str = 'color'
KEY_LABEL: str = 'label'
KEY_FLOW: str = 'flow'
KEY_RAW: str = 'raw'
KEY_CLOUD: str = 'partial_cloud'


def debug_image(cam_out,
                save: bool = False,
                show: bool = False
                ):
    from util.torch_util import dcn
    from util.vis.img import digitize_image, normalize_image
    import cv2
    # cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
    # cv2.imshow('depth', digitize_image(dcn(cam_out['depth'][0])).astype(
    #     np.uint8))

    depth = cam_out[KEY_DEPTH]
    if depth.shape[-3] != 1:
        depth = depth.unsqueeze(-3)
    depth = rearrange(depth, '... c h w -> (...) h w c')
    grid = tile_images(depth)
    grid = (grid - grid.min()) / (grid.max() - grid.min())

    if save:
        global _COUNT
        out_dir = ensure_directory('/tmp/docker/depth')
        cv2.imwrite(F'{out_dir}/depth-{_COUNT:04d}.png',
                    (255 * normalize_image(dcn(grid))).astype(np.uint8)
                    )
        _COUNT += 1
    if show:
        # hmm..........................>?
        # FIXME:
        # Why the hell do we need [::-1,::-1]???
        grid = dcn(grid)[..., ::-1, ::-1]
        cv2.namedWindow('depth', cv2.WINDOW_NORMAL)
        cv2.imshow(
            'depth',
            (255 *
                normalize_image(
                    dcn(grid))).astype(
                np.uint8))
        cv2.waitKey(_DELAY)


@nvtx.annotate("unproject")
def unproject(
        depth_img: th.Tensor,
        cfg: Optional[WithNvdrCamera.Config] = None,
        T_cam: Optional[th.Tensor] = None,
        tan_half_fov: Optional[th.Tensor] = None
):
    device = depth_img.device

    # 0. Convert depth img to NDC point img

    # 0.1. Build and maintain NDC coordinates.
    # TODO: may need to reverse H ("axis 0")
    # to match bottom-to-top scanline convention.
    H, W = depth_img.shape[-2], depth_img.shape[-1]
    # IJ = th.cartesian_prod(
    #     th.arange(H, device=device),
    #     th.arange(W, device=device))
    # X = IJ[..., 1].float().sub_(W / 2).div_(W / 2)
    # Y = IJ[..., 0].float().sub_(H / 2).neg_().div_(H / 2)
    # XY = th.stack([X, Y], dim=0).view(2, H, W)

    if tan_half_fov is None:
        x = th.tan(0.5 * th.as_tensor(cfg.fov, device=device))
        XY = rearrange(th.cartesian_prod(
            th.linspace(+x, -x, depth_img.shape[-2], device=device),
            th.linspace(-x, +x, depth_img.shape[-1], device=device)
        ).view(-1, H, W, 2), '... h w c -> ... c h w')
        # Here we actually end up with XY = (1,H,W,2)
    else:
        x = tan_half_fov
        grid = rearrange(th.cartesian_prod(
            th.linspace(+1, -1, depth_img.shape[-2], device=device),
            th.linspace(-1, +1, depth_img.shape[-1], device=device)
        ).view(-1, H, W, 2), '... h w c -> ... c h w')
        XY = x[:, None, None, None] * grid
        # Here we actually end up with XY = (N,H,W,2)

    # 0.2. Allocate & concatenate everything into a single image.
    point_img = th.empty(
        merge_shapes(depth_img.shape[:-3], 4, depth_img.shape[-2:]),
        dtype=th.float32, device=device)
    # print(depth_img.shape)
    # print(XY.shape)
    # print(point_img.shape)
    # point_img[:, 0:2] = XY[None]  # 4,1,64,64 * 1,2,64,64
    point_img[:, 0:2] = th.flip(XY, dims=(-3,))
    point_img[:, 2:3] = 1
    # point_img[:, 0:2] = XY[None]  # 4,1,64,64 * 1,2,64,64
    # point_img[:, 2:3] = 0.01
    # point_img[:, 3] = 0.01
    point_img[:, 3:] = th.reciprocal(depth_img)

    # 1. Get camera matrices...
    # NOTE: T_cam is a world->view coord. transform
    if T_cam is None:
        print('T_cam is None so... :)')
        T_cam, _ = camera_matrices(cfg,
                                   depth_img.shape[0],
                                   depth_img.device)
    # print('T_ndc', T_ndc)
    # FIXME: numerically unstable but convenient
    # there are better ways to invert a homogeneous
    # transform matrix ...
    # T = th.einsum('...ij, ...jk -> ...ik', T_ndc, T_cam)
    # T = T_cam
    # Ti = th.linalg.inv(T)
    # Ti = (R^{T}, -R^{T} X)

    # 2. unproject the points ...
    # point_img = th.einsum('nij,nj...->ni...', Ti, point_img)
    # point_img[:, :3] = (th.einsum('nji, nj... -> ni...',
    #                               T_cam[..., :3, :3], point_img[:, :3, ...])
    #                     - th.einsum('nji, nj, n... -> ni...',
    #                                 T_cam[..., :3, :3],
    #                                 T_cam[..., :3, 3],
    #                                 point_img[:, 3]))

    # x' = h^{-1}(R^T.x - R^T.t.h)
    return (
        # h^{-1}(R^T.x)
        th.einsum(
            'nji, nj... -> ni...',
            T_cam[..., : 3, : 3],
            point_img[:, :3, ...])
        .div_(point_img[:, 3:])
        # - R^{T}.t
        .sub_(th.einsum('nji, nj -> ni',
                        T_cam[..., :3, :3],
                        T_cam[..., :3, 3])[..., None, None])

    )
    # (point_img[:, :3]
    #  .copy_(th.einsum('nji, nj... -> ni...',
    #                   T_cam[..., :3, :3], point_img[:, :3, ...]))
    #  .sub_(th.einsum('nji, nj, n... -> ni...',
    #                  T_cam[..., :3, :3],
    #                  T_cam[..., :3, 3],
    #                  point_img[:, 3])))
    # return point_img[:, :3].div_(point_img[:, 3:])

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


class NvdrCameraWrapper(ObservationWrapper):

    @dataclass
    class Config(WithNvdrCamera.Config):
        img_size: Tuple[int, int] = (224, 224)

        eye: Tuple[float, float, float] = (0.54, 0.0, 0.9)
        at: Tuple[float, float, float] = (-0.2, 0.0, 0.4)
        up: Tuple[float, float, float] = (0.0, 0.0, 1.0)

        # NOTE:
        # `fast_depth` is disabled for two reasons:
        # first, I'm not sure if `fast_depth` is actually faster.
        # second, I'm not sure if the output is always correct.
        fast_depth: bool = False

        use_depth: bool = False
        use_flow: bool = False
        use_label: bool = False
        use_color: bool = False

        use_cloud: bool = False
        resample_cloud: bool = True
        cloud_type: str = 'object'
        cloud_size: int = 512

        object_id: int = -1
        robot_id: int = -1

        use_robot: bool = False
        use_ground: bool = False
        use_table: bool = False

        use_col: bool = True

        debug_image: Optional[str] = None

        hide_arm: bool = True

        mimic_table_removal: bool = False
        table_param: Optional[Tuple[float, ...]] = (0., 0., 1., -0.4)
        eps_for_removal: float = 0.01

        key_depth: str = KEY_DEPTH
        key_color: str = KEY_COLOR
        key_label: str = KEY_LABEL
        key_flow: str = KEY_FLOW
        key_raw: str = KEY_RAW
        key_cloud: str = KEY_CLOUD

    def __init__(self,
                 env: EnvIface,
                 cfg: Config):
        super().__init__(env, self._wrap_obs)
        self.cfg = cfg
        # assert (isinstance(env.gym, TrackObjectAssets))

        assets = []
        num_env: int = env.num_env
        # We need to get back the object mask.

        if self.env.scene.tmpdir is not None:
            new_asset_root = self.env.scene.tmpdir
            self.set_camera(new_asset_root)
        else:
            self.set_camera()

        # NOTE: trying to automatically figure out
        # the number of bodies in the environment !!
        self.num_body = env.gym.get_env_rigid_body_count(env.envs[0])

        num_poses = (self.num_body + 1 if cfg.use_ground else self.num_body)
        self.body_poses_4x4 = th.zeros(
            (num_env, num_poses, 4, 4),
            dtype=th.float32,
            device=env.device)
        self.body_poses_4x4[:] = th.eye(4,
                                        dtype=th.float, device=env.device)

        # Depth ranges from 0 (no data) to 10.
        depth_space = spaces.Box(
            0.0, self.cfg.z_far, merge_shapes(1, cfg.img_size),
            dtype=np.float32)

        self._make_dict = False

        if cfg.mimic_table_removal:
            table_param = th.tensor(cfg.table_param,
                                         dtype=th.float,
                                         device=env.device)
            self.table_param = table_param[None, :, None].repeat(num_env, 1, 1)

        if isinstance(env.observation_space, spaces.Dict):
            # let's try to prevent nested mappings, if possible...
            obs_space = dict(env.observation_space.spaces)
        else:
            self._make_dict = True
            self._skip_raw = False
            if env.observation_space is None:
                self._skip_raw = True
                obs_space = {}
            else:
                obs_space = {
                    cfg.key_raw: env.observation_space
                }

        if cfg.use_depth:
            obs_space[cfg.key_depth] = depth_space

        if cfg.use_flow:
            obs_space[cfg.key_flow] = spaces.Box(-1.0, +1.0,
                                                 merge_shapes(cfg.img_size, 2),
                                                 dtype=np.float32
                                                 )
        if cfg.use_label:
            # NOTE; background=0 and all other
            # objects are offset by 1 I guess.
            # TODO: do we need to remap the indices?
            obs_space[cfg.key_label] = spaces.Box(
                0, self.num_body, merge_shapes(1, cfg.img_size), dtype=np.int32)

        if cfg.use_color:
            # NOTE; background=0 and all other
            # objects are offset by 1 I guess.
            # TODO: do we need to remap the indices?
            obs_space[cfg.key_color] = spaces.Box(0, self.num_body,
                                                  merge_shapes(3, cfg.img_size),
                                                  dtype=np.float32)

        if cfg.use_cloud:
            # NOTE; background=0 and all other
            # objects are offset by 1 I guess.
            # TODO: do we need to remap the indices?
            obs_space[cfg.key_cloud] = spaces.Box(0, self.num_body,
                                                  (cfg.cloud_size, 3),
                                                  dtype=np.float32)

        self._obs_space = spaces.Dict(obs_space)

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def device(self):
        return self.env.device

    @property
    def timeout(self):
        return self.env.timeout

    @property
    def num_env(self):
        return self.env.num_env
    
    def set_camera(self, 
                   new_root: Optional[str] = None):
        cfg = self.cfg
        object_urdfs = []
        urdf_link_map = {}
        offsets = []
        geom_index: Dict[Tuple[str, str], int] = {}

        sel = []
        # urdf_index = {}
        for e in self.env.envs:
            num_actors = self.env.gym.get_actor_count(e)
            num_bodies: int = self.env.gym.get_env_rigid_body_count(e)
            sel_i = [None for _ in range(num_bodies)]

            for i in range(num_actors):
                actor_handle =self.env.gym.get_actor_handle(e, i)
                asset = self.env.gym.get_actor_asset(e, actor_handle)
                body_count = self.env.gym.get_actor_rigid_body_count(e,
                                                                actor_handle)

                asset_args = self.env.gym.actor_args[e, actor_handle]

                is_new_asset: bool = False
                # if asset not in assets:
                #    assets.append(asset)
                #    is_new_asset = True

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
                    if box_arg not in object_urdfs:
                        is_new_asset = True
                        object_urdfs.append(box_arg)
                    key = (box_arg)

                if 'filename' in asset_args:
                    # URDF
                    asset_root = asset_args['rootpath'] \
                        if new_root is None or \
                        'robots' in asset_args['filename'] else new_root
                    asset_file = asset_args['filename']
                    asset_path = F'{asset_root}/{asset_file}'

                    # NOTE: technically quite slow
                    # maybe use ordereddict instead? :)
                    # if is_new_asset:
                    #     object_urdfs.append(asset_path)
                    if asset_path not in object_urdfs:
                        is_new_asset = True
                        object_urdfs.append(asset_path)
                    key = (asset_path)

                if is_new_asset:
                    body_names = self.env.gym.get_asset_rigid_body_names(
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

                body_names = urdf_link_map[key]
                for ii in range(body_count):
                    body_name: str = body_names[ii]
                    body_index = self.env.gym.find_actor_rigid_body_index(
                        e,
                        actor_handle,
                        body_name,
                        gymapi.IndexDomain.DOMAIN_ENV
                    )
                    sel_i[body_index] = geom_index[(key, body_name)]
            assert (None not in sel_i)
            sel.append(sel_i)

        if cfg.use_ground:
            for s in sel:
                s.append(len(object_urdfs))

        # print(object_urdfs)
        # print(body_counts)
        # print(sel)
        # print('actor args', env.gym.actor_args)
        # raise ValueError('stop.')
        use_cube = False
        cube_dims = None
        use_table = False
        use_robot = False
        if cfg.hide_arm:
            blocklist = get_hacky_blocklist_for_arm_links()
        else:
            blocklist = None

        num_env: int = self.env.num_env

        if (cfg.use_cloud) and (cfg.cloud_type == 'object'):
            cfg = replace(cfg, use_label=True)

        self.cam = WithNvdrCamera(cfg,
                                  num_env,
                                  device=self.env.device,
                                  shape=cfg.img_size,
                                  sel=np.asarray(sel, dtype=np.int32),
                                  object_urdfs=object_urdfs,
                                  urdf_link_map=urdf_link_map,
                                  blocklist=blocklist)

    def setup(self):
        out = super().setup()
        # == setup ==
        # jit compile. stuff
        self._wrap_obs(None)
        return out

    @nvtx.annotate("NvdrCameraWrapper._wrap_obs")
    def _wrap_obs(self, obs) -> th.Tensor:
        cfg = self.cfg

        body_tensors = self.env.tensors['body']
        BT = body_tensors.reshape(self.num_env, -1, 13)
        body_poses_7dof = BT[:, :self.num_body, ..., :7]
        self.body_poses_4x4[..., :self.num_body,
                            :3, 3] = body_poses_7dof[..., :3]
        matrix_from_quaternion(
            body_poses_7dof[..., 3: 7],
            self.body_poses_4x4[..., : self.num_body, : 3, : 3])

        if cfg.track_object:
            env = self.env
            obj_ids = env.scene.cur_ids.long()
            cur_pose = env.tensors['root'][obj_ids, :7]
            cam_out = self.cam(self.body_poses_4x4,
                               targets=cur_pose[..., :3],
                               radii=env.scene.cur_radii)
        else:
            cam_out = self.cam(self.body_poses_4x4)

        # NOTE: uncomment this part
        # for ad-hoc visualization.
        if cfg.debug_image is not None:
            if cfg.debug_image == 'nan':
                ic(th.isfinite(cam_out['depth']).float().mean())
            else:
                debug_image(cam_out,
                            save=(cfg.debug_image == 'save'),
                            show=(cfg.debug_image == 'show')
                            )

        if self._make_dict:
            if self._skip_raw:
                out = {}
            else:
                out = {cfg.key_raw: obs}
        else:
            out = dict(obs)

        # ADD DEPTH (always)
        if cfg.use_depth:
            depth = cam_out['depth']
            # depth = th.fliplr(depth)#[..., ::-1, ::-1]
            # depth = th.flipud(depth)#[..., ::-1, ::-1]
            # if depth.shape[-3] != 1:
            depth = depth.unsqueeze(-3)
            # FIXME: rename `img` -> `depth`
            out[cfg.key_depth] = depth

        # ADD FLOW
        if cfg.use_flow:
            out[cfg.key_flow] = cam_out['flow']

        # ADD SEGMENTATION MASK (LABEL)
        if cfg.use_label:
            seg = cam_out['label']
            if seg.shape[-3] != 1:
                seg = seg.unsqueeze(-3)
            out[cfg.key_label] = seg

        if cfg.use_color:
            rgb = cam_out['color']
            out[cfg.key_color] = rearrange(rgb,
                                           '... h w c -> ... c h w')

        if cfg.use_cloud:
            if cfg.track_object:
                cloud = unproject(depth,
                                  cfg=self.cfg,
                                  # T_cam=self.cam.renderer.T_cam
                                  T_cam=self.cam.renderer.T_pos,
                                  tan_half_fov=th.reciprocal(
                                      self.cam.renderer.T_ndc[..., 0, 0]
                                  ))
            else:
                cloud = unproject(depth, self.cfg)
            with nvtx.annotate("resample_cloud"):
                if cfg.resample_cloud:
                    cloud = rearrange(cloud, '... c h w -> ... (h w) c')
                    if cfg.cloud_type == 'visible':
                        has_data = depth > 0
                    elif cfg.cloud_type == 'all':
                        has_data = th.ones_like(depth)
                    elif cfg.cloud_type == 'object':
                        has_data = (cam_out['label'] == cfg.object_id)
                    elif cfg.cloud_type in ('robot+object', 'object+robot'):
                        has_data = th.logical_or(
                            (cam_out['label'] == cfg.object_id),
                            (cam_out['label'] == cfg.robot_id)
                        )
                    
                    has_data = has_data.reshape(has_data.shape[0], -1)
                    if cfg.mimic_table_removal:
                        is_table = (th.bmm(cloud, self.table_param[:, :3]) 
                                    + self.table_param[:, 3:]) <= cfg.eps_for_removal
                        has_data = th.logical_and(has_data,
                                                  th.logical_not(is_table).squeeze(-1))
                    # FIXME:
                    # 1e-6 (hackily) prevents sum=0...
                    # prob = (mask.float() + (depth > 0).reshape(mask.shape).float()
                    #         * 1e-9) / (mask.sum(keepdim=True, dim=(-1,)) + 1e-9)
                    EPS: float = 1e-6
                    prob = has_data.float().add_(EPS)
                    prob = prob.div_(prob.sum(keepdim=True, dim=-1))
                    # if th.isnan(prob).any():
                    #    raise ValueError('prob is nan')
                    # if (prob < 0).any():
                    #    raise ValueError('prob is neg')
                    # if (prob == 0).all(dim=-1).any():
                    #    raise ValueError('prob is all-zero')

                    # FIXME: throws ValueError in the off case
                    # where `(depth<=0).all() i.e. no valid point exists
                    # try:
                    indices1 = th.multinomial(prob, num_samples=cfg.cloud_size,
                                              replacement=True)
                    # except RuntimeError:
                    #    with open('/tmp/prob.pkl', 'wb') as fp:
                    #        pickle.dump(prob, fp)
                    #    raise
                    # indices1 = th.argwhere(prob)[..., 1]
                    # indices0 = th.arange(prob.shape[0],
                    #                      dtype=indices1.dtype,
                    #                      device=indices1.device)
                    # ic(cloud[indices0[:, None], indices1].shape)  # 4096,512,3
                    # ic(mask.shape)  # 4096,1024
                    cloud = th.take_along_dim(
                        cloud, indices1[..., None], dim=-2)

                    # Where no points are available, we set to zero instead of
                    # potentially dangerous `NaN` propagation.
                    cloud.masked_fill_((~has_data.any(dim=-1))
                                       [..., None, None], 0.0)

            out[cfg.key_cloud] = cloud
            
        if _DEBUG_UNPROJECT:
            stride: int = 1

            depth = cam_out['depth'][..., ::stride, ::stride]
            if depth.shape[-3] != 1:
                depth = depth.unsqueeze(-3)

            # 1. unproject depth img -> point cloud
            cloud = unproject(depth, self.cfg)
            cloud_hwc = rearrange(cloud, '... c h w -> ... h w c')

            # 2. render the point cloud to the viewer
            # with gym.[...] calls.
            mask = (depth > 0).squeeze(-3)
            # mask = th.ones_like(depth > 0).bool().squeeze(-3)

            self.gym.clear_lines(self.viewer)
            for i in range(self.num_env):
                draw_cloud_with_mask(self.gym, self.viewer,
                                     cloud_hwc[i], self.envs[i],
                                     mask[i], color=(1, 0, 0))

        return out
