#!/usr/bin/env python3

from typing import Tuple, Optional
import torch as th
from icecream import ic


class GetBbox:
    def __init__(self, mode: str = 'corner'):
        self.mode = mode

    def reset(self, env: 'NvdrCameraWrapper'):
        self.env = env

    def __call__(self, img_shape: Tuple[int, int],
                 center: Optional[th.Tensor] = None,
                 radius: Optional[th.Tensor] = None
                 ):

        env = self.env

        # Query object properties.
        # NOTE: when `center`/`radius` args are
        # not populated, we default to the legacy
        # behavior of cropping the "object".
        if center is None:
            center = env.tensors['root'][env.scene.cur_ids.long(), ..., :3]
        if radius is None:
            radius = env.scene.cur_radii

        # Query camera transforms.
        T_cam = env.cam.renderer.T_cam  # Nx4x4
        T_pos = env.cam.renderer.T_pos  # world-to-camera
        uvec = T_pos[..., 1, :3]  # this corresponds to Y (=="I")
        vvec = T_pos[..., 0, :3]  # this corresponds to X (=="J")

        if self.mode == 'corner':
            corners = th.empty((center.shape[0], 4, 4),
                               dtype=th.float,
                               device=env.device)
            corners[..., 3] = 1
            # tl, tr, br, bl
            r = radius[..., None]
            corners[..., 0, :3] = center + r * (+uvec - vvec)
            corners[..., 1, :3] = center + r * (+uvec + vvec)
            corners[..., 2, :3] = center + r * (-uvec + vvec)
            corners[..., 3, :3] = center + r * (-uvec - vvec)

        elif self.mode == 'minmax':
            # Compute corner coordinates.
            corners = th.empty((center.shape[0], 2, 4),
                               dtype=th.float,
                               device=env.device)
            corners[..., 3] = 1
            corners[..., 0, :3] = center + radius[..., None] * (-uvec - vvec)
            corners[..., 1, :3] = center + radius[..., None] * (+uvec + vvec)

        # Apply projection.
        points = th.einsum('...ij,...pj->...pi', T_cam, corners)
        xy = points[..., :2]
        z = points[..., 2:3]

        # NDC -> pixel coord
        xy.div_(z).add_(1.0).mul_(0.5)
        xy[..., 0].mul_(img_shape[-2])  # h?
        xy[..., 1].mul_(img_shape[-1])  # w
        xy = xy.to(th.int64)
        return xy
