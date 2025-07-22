#!/usr/bin/env python3

from typing import Optional
import torch as th
import numpy as np
from einops import repeat


def sample_yaw(n: int, dtype=th.float, device=None):
    h = th.empty((n,), dtype=dtype, device=device).uniform_(
        -th.pi / 2, +th.pi / 2)
    z = th.sin(h)
    w = th.cos(h)
    q_z = th.stack([0 * z, 0 * z, z, w], dim=-1)
    return q_z


def sample_box_xy(
        bound: th.Tensor,
        z: float,
        n: Optional[int] = None,
        out: Optional[th.Tensor] = None
):
    if n is None:
        n = bound.shape[0]
    scale = bound[..., 1, :] - bound[..., 0, :]
    point_xy = bound[..., 0, :] + scale * th.rand((n, bound.shape[-1]),
                                                  dtype=bound.dtype,
                                                  device=bound.device)
    if out is None:
        out = th.zeros((*point_xy.shape[:-1], 3),
                       dtype=bound.dtype,
                       device=bound.device)
    out[..., :2] = point_xy[..., :2]
    out[..., 2] = z
    return out


def ray_box_intersection(
        origin: th.Tensor,
        raydir: th.Tensor,
        box: th.Tensor):
    c = origin
    d = raydir
    b = box  # .reshape(..., 2, 2)
    i = 1 / d
    s = (i < 0).to(dtype=th.long)
    txmax = (b[th.arange(b.shape[0]),
               1 - s[..., 0], 0] - c[..., 0]) * i[..., 0]
    tymax = (b[th.arange(b.shape[0]),
               1 - s[..., 1], 1] - c[..., 1]) * i[..., 1]
    tmax = th.minimum(tymax, txmax)
    return c + d * tmax[..., None]


def sample_goal(
        bound: th.Tensor,
        center: th.Tensor,
        radius: float,
        z: float,
        eps: float = 1e-6,
        out: Optional[th.Tensor] = None
):
    """
    Sample points outside of the goal radius.
    Unlike rejection sampling, this method is not iterative,
    but at the cost of a slightly distorted distribution
    and potentially infeasible goals outside of the table boundary
    due to projection.
    """
    scale = (bound[..., 1, :] - bound[..., 0, :])
    p = bound[..., 0, :] + scale * th.rand_like(center)
    d = p - center
    r = th.linalg.norm(d, dim=-1,
                       keepdim=True)
    raydir = d.div_(r + eps)
    dst = ray_box_intersection(center, raydir, bound)

    # Explicitly sampled points at the
    # exterior of the goal.
    ext = th.lerp(center + radius * raydir, dst,
                  th.rand_like(center[..., :1])
                  )
    if out is None:
        out = th.zeros((*center.shape[:-1], 3),
                       dtype=th.float,
                       device=center.device)
    out[..., :2] = th.where(((0 * (r > radius)).bool()), p, ext)
    out[..., 2] = z
    return out


def sample_goal_v2(
        bound: th.Tensor,
        center: th.Tensor,
        radius: float,
        z: Optional[float] = None,
        eps: float = 1e-6,
        out: Optional[th.Tensor] = None,
        num_samples: int = 4):
    """
    Sample points outside of the goal radius.
    Non-iterative rejection sampling,
    via oversampling by a factor of `num_samples`.
    """
    scale = (bound[..., 1, :] - bound[..., 0, :])
    p = bound[..., 0, :] + scale * th.rand(
        ((num_samples,) + center.shape),
        dtype=center.dtype,
        device=center.device)
    d = p - center
    r2 = th.einsum('...i,...i->...', d, d)
    mask = (r2 > radius**2)  # 1=bad, 0=good
    if out is None:
        dim = (2 if z is None else 3)
        out = th.zeros((*center.shape[:-1], dim),
                       dtype=th.float,
                       device=center.device)
    out[..., :2] = p[
        th.argmax(mask.float(), dim=0),
        th.arange(center.shape[0])
    ]
    if (z is not None) and out.shape[-1] > 2:
        out[..., 2] = z
    return out


def main():
    from util.torch_util import dcn
    from matplotlib import pyplot as plt
    device: str = 'cpu'
    num_env: int = 4096
    bound = repeat(th.as_tensor([[-0.3, -0.2], [0.4, 0.5]]),
                   '... -> n ...', n=num_env)
    bound = bound.to(device=device)
    scale = bound[..., 1, :] - bound[..., 0, :]
    # obj_pos = bound[..., 0, :] + th.rand((num_env, 2)) * scale
    obj_pos = bound[..., 0, :] + th.rand((1, 2), device=device) * scale
    obj_pos = obj_pos.to(device=device)
    goal_radius: float = 0.1
    goals = sample_goal(
        bound,
        obj_pos,
        goal_radius, z=0.0)

    goals_np = dcn(goals)
    plt.plot(goals_np[..., 0], goals_np[..., 1], '*')
    plt.axis('equal')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
