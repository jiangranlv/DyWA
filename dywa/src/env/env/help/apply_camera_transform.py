#!/usr/bin/env python3

from typing import Tuple

import torch as th
import cv2


class ApplyCameraTransform:
    def __init__(self):
        pass

    def reset(self, env: 'NvdrRecordViewer'):
        self.env = env

    def __call__(self, point: th.Tensor,
                 img_size: Tuple[int, int]):
        """
        point: (N, P, 3)
        """
        env = self.env

        # Get ndc transform
        T_cam = env.img_env.cam.renderer.T_cam  # Nx4x4

        p3 = point if isinstance(
            point, th.Tensor) else th.as_tensor(
            point, dtype=T_cam.dtype, device=T_cam.device)
        p4 = th.empty((*point.shape[:-1], 4),
                      dtype=T_cam.dtype,
                      device=T_cam.device)
        p4[..., :3] = p3
        p4[..., 3] = 1

        p_cam = th.einsum('...ij,...pj->...pi', T_cam, p4)
        p_xy = p_cam[..., :2]
        p_z = p_cam[..., 2:3]
        p_xy.div_(p_z).add_(1.0).mul_(0.5)
        p_xy[..., 0].mul_(img_size[0])  # H=y
        p_xy[..., 1].mul_(img_size[1])  # W=x
        p_xy = p_xy.to(th.int32)
        # p_xy = th.flip(p_xy, (-1,))
        return p_xy
