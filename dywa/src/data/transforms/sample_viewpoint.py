#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Tuple, Optional, Union
import torch as th
import numpy as np

from pytorch3d.renderer import look_at_view_transform


def ndc_projection_matrix(fov: th.Tensor,
                          z_near: float, z_far: float,
                          out: th.Tensor = None) -> th.Tensor:
    """ Compute NDC projection matrix in the
    convention of nvdiffrast.

    Args:
        fov: Field of view (radians)
        z_near: Near-plane of camera frustum.
        z_far:  Far-plane of camera frustum.
        out: Optional output tensor, if buffer already allocated.

    Returns:
        NDC projection matrix; fov.shape + (4, 4)
    """
    # Compute pixel-space focal length.
    x = th.tan(0.5 * fov) * z_near

    # Allocate projection matrix.
    P = th.zeros(fov.shape + (4, 4),
                 out=out, dtype=fov.dtype, device=fov.device)

    # Populate projection matrix.
    P[..., 0, 0] = z_near / x
    P[..., 1, 1] = z_near / x
    P[..., 2, 2] = (z_far + z_near) / (z_far - z_near)
    P[..., 2, 3] = -(2 * z_far * z_near) / (z_far - z_near)
    P[..., 3, 2] = 1.0

    return P


class SampleCamera:
    """
    Sample camera intrinsics and extrinsics.
    """

    @dataclass(frozen=True)
    class Config:
        min_fov: float = np.deg2rad(10.0)
        max_fov: float = np.deg2rad(80.0)
        z_near: float = 0.01
        z_far: float = 100.0

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self,
                 radius: th.Tensor,
                 center: th.Tensor,
                 size: Union[int, Tuple[int, ...]]):
        cfg = self.cfg
        # By default, we'll propagate the input tensor device
        # to our outputs.
        device = radius.device

        # Field of view
        if isinstance(size, int):
            fov = th.empty((size,), device=device, dtype=th.float32).uniform_(
                self.cfg.min_fov, self.cfg.max_fov)
        else:
            fov = th.empty(size, device=device, dtype=th.float32).uniform_(
                self.cfg.min_fov, self.cfg.max_fov)

        # fov -> NDC projection
        P_ndc = ndc_projection_matrix(fov, cfg.z_near, cfg.z_far)

        # NOTE: automatically figure out
        # a decent viewpoint location based on the camera frustum.
        # We try to position the camera somewhere `reasonable`,
        # so that we get many useful points.
        viewpoint = sample_viewpoint(radius, center, fov, cfg.z_near,
                                     cfg.z_far)

        # Lookat()
        R, T = look_at_view_transform(
            eye=viewpoint.to(device),
            at=center.to(device),
            device=device,
            up=((1,0,0),))
        return (R, T, P_ndc, viewpoint, fov)


class SampleViewpoint:
    """
    Functor wrapper around sample_viewpoint.
    """

    def __call__(self,
                 radius: th.Tensor,
                 center: th.Tensor,
                 fov: th.Tensor,
                 z_near: th.Tensor,
                 z_far: th.Tensor) -> th.Tensor:
        return sample_viewpoint(radius, center, fov, z_near, z_far)


def sample_viewpoint(radius: float,
                     center: Tuple[float, float, float],
                     fov: float,
                     z_near: float,
                     z_far: float):
    """
    Sample a viewpoint around the center such that it fits the
    bounding sphere of the object within the camera field of view.

    Args:
        radius: Object radius. shape=[...]
        center: Object center. shape=[...,3]
        fov: Camera FoV. We assume vfov=hfov. shape=[...]
        z_near: Camera near-clipping distance. shape=[...]
        z_far: Camera far-clipping distance. shape=[...]
    Returns:
        Camera viewpoints. shape=[...,3]
    """
    # Radius of the bounding box.

    # Map (1.0 x radius) -> (1.0~2.0) x radius.
    # scale = (1.0 + th.rand(size=(batch_size,), device=device))
    scale = 1.0
    distance = (radius * scale) / th.tan(0.5 * fov)

    # NOTE:
    # `delta` is the sum of the following:
    # - cfg.znear : distance from camera position to near-plane.
    # - radius    : offset to object boundary from object center.
    # - distance  : the distance between object boundary and
    #             the near-plane of the camera frustum.
    # Geometrically, delta is arranged as the distance from the
    # center of the object to the position of the camera, as follows:
    # [center] -<radius>- [boundary]
    # [boundary] -<distance>- [near-plane]
    # [near-plane] -<znear>- [camera]
    delta = (z_near + radius + distance)

    # WARN: randomness is NOT controlled when
    # sampling viewpoints.
    # WARN: th.randn(device='cpu') results in
    # a different value from th.randn(device='cuda')
    # even with the same random seed!
    offset = th.randn(size=(*delta.shape, 3), device=delta.device)
    offset = offset * delta[..., None] / th.linalg.norm(
        offset, dim=-1, keepdim=True)
    viewpoint = offset + center
    return viewpoint
