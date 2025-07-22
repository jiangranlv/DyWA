#!/usr/bin/env python3

from typing import Optional, Tuple

import numpy as np

import torch as th

from models.common import merge_shapes


def align_vectors(a, b, eps: float = 0.00001):
    """
    Return q: rotate(q, a) == b
    """
    dot = th.einsum('...j, ...j->...', a, b)
    parallel = (dot > (1 - eps))
    opposite = (dot < (-1 + eps))

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


def quat_conjugate(q: th.Tensor):
    return th.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def random_rotation_matrix(n: int, *args, **kwds) -> th.Tensor:
    shape = merge_shapes(n, (3, 3))
    m = th.randn(size=shape, *args, **kwds)
    Q = th.linalg.qr(m, mode='complete')[0]
    return Q


def random_yaw_quaternion(shape: Tuple[int, ...], **kwds):
    yaw = th.pi * th.rand(shape, **kwds) - 0.5 * th.pi
    c, s = th.cos(yaw), th.sin(yaw)
    z = 0 * c
    qxn = th.stack([z, z, s, c], dim=-1)
    return qxn


def apply_pose(q: th.Tensor, t: th.Tensor, x: th.Tensor):
    return quat_rotate(q, x) + t


def apply_pose_tq(tq: th.Tensor, x: th.Tensor):
    return apply_pose(tq[..., 3:7], tq[..., 0:3], x)


def compose_pose_tq(tq0: th.Tensor, tq1: th.Tensor):
    t0, q0 = tq0[..., 0:3], tq0[..., 3:7]
    t1, q1 = tq1[..., 0:3], tq1[..., 3:7]
    return th.cat([
        quat_rotate(q0, t1) + t0,
        quat_multiply(q0, q1)
    ], dim=-1)


def invert_pose_tq(tq: th.Tensor) -> th.Tensor:
    out = th.empty_like(tq)
    t, q = tq[..., 0:3], tq[..., 3:7]
    qi = quat_inverse(q, out=out[..., 3:7])
    out[..., 0:3] = -quat_rotate(qi, t)
    return out


def invert_transform(T: th.Tensor) -> th.Tensor:
    out = T.clone()
    # R.T
    out[..., :3, :3] = T[..., :3, :3].swapaxes(-1, -2)
    # -R.T@x
    out[..., :3, 3] = - out[..., :3, :3] @ T[..., :3, 3]
    return out


@th.jit.script
def _sqrt_positive_part(x: th.Tensor) -> th.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = th.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = th.sqrt(x[positive_mask])
    return ret


@th.jit.script
def matrix_to_quaternion(matrix: th.Tensor) -> th.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = th.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        th.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of x, y, z, w
    quat_by_xyzw = th.stack(
        [th.stack(
            [m21 - m12, m02 - m20, m10 - m01, q_abs[..., 0] ** 2],
            dim=-1),
         th.stack(
             [q_abs[..., 1] ** 2, m10 + m01, m02 + m20, m21 - m12],
             dim=-1),
         th.stack(
             [m10 + m01, q_abs[..., 2] ** 2, m12 + m21, m02 - m20],
             dim=-1),
         th.stack(
             [m20 + m02, m21 + m12, q_abs[..., 3] ** 2, m10 - m01],
             dim=-1),],
        dim=-2,)

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = th.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_xyzw / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        th.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


# @th.jit.script
def _matrix_from_quaternion(x: th.Tensor, out: th.Tensor):
    # qx, qy, qz, qw = [x[..., i] for i in range(4)]
    qx = x[..., 0]
    qy = x[..., 1]
    qz = x[..., 2]
    qw = x[..., 3]

    tx = 2.0 * qx
    ty = 2.0 * qy
    tz = 2.0 * qz
    twx = tx * qw
    twy = ty * qw
    twz = tz * qw
    txx = tx * qx
    txy = ty * qx
    txz = tz * qx
    tyy = ty * qy
    tyz = tz * qy
    tzz = tz * qz

    # outr = out.reshape(-1, 9)
    # th.stack([
    #     1.0 - (tyy + tzz),
    #     txy - twx,
    #     txz + twy,
    #     txy + twz,
    #     1.0 - (txx + tzz),
    #     tyz - twx,
    #     txz - twy,
    #     tyz + twx,
    #     1.0 - (txx + tyy)], dim=-1, out=outr)
    out[..., 0, 0] = 1.0 - (tyy + tzz)
    out[..., 0, 1] = txy - twz
    out[..., 0, 2] = txz + twy
    out[..., 1, 0] = txy + twz
    out[..., 1, 1] = 1.0 - (txx + tzz)
    out[..., 1, 2] = tyz - twx
    out[..., 2, 0] = txz - twy
    out[..., 2, 1] = tyz + twx
    out[..., 2, 2] = 1.0 - (txx + tyy)


def matrix_from_quaternion(
        x: th.Tensor, out: Optional[th.Tensor] = None) -> th.Tensor:
    if out is None:
        out = th.zeros(size=x.shape[:-1] + (3, 3),
                       dtype=x.dtype, device=x.device)
    _matrix_from_quaternion(x, out)
    return out


@th.jit.script
def _quaternion_from_matrix(x: th.Tensor, out: th.Tensor):
    # parse input
    m00, m01, m02 = [x[..., 0, i] for i in range(3)]
    m10, m11, m12 = [x[..., 1, i] for i in range(3)]
    m20, m21, m22 = [x[..., 2, i] for i in range(3)]
    th.stack([1. + m00 - m11 - m22,
              1. - m00 + m11 - m22,
              1. - m00 - m11 + m22,
              1. + m00 + m11 + m22], dim=-1,
             out=out)
    out.clamp_min_(0.0).sqrt_().mul_(0.5)
    out[..., 0].copysign_(m21 - m12)
    out[..., 1].copysign_(m02 - m20)
    out[..., 2].copysign_(m10 - m01)
    return out


def quaternion_from_matrix(
        x: th.Tensor, out: Optional[th.Tensor] = None) -> th.Tensor:
    if out is None:
        out = th.zeros(size=x.shape[:-2] + (4,),
                       dtype=x.dtype,
                       device=x.device)
    _quaternion_from_matrix(x, out)
    return out

def quaternion_to_6D(x: th.Tensor):
    rot_mat = matrix_from_quaternion(x)
    out = rot_mat[..., :, :2].reshape(*rot_mat.shape[:-2], -1)
    return out


def rot6d_to_matrix(rot6d: th.Tensor) -> th.Tensor:
    x = rot6d[..., [0, 2, 4]]
    y = rot6d[..., [1, 3, 5]]
    x = x / th.norm(x, dim=-1, keepdim=True)
    y = y - th.sum(x*y, dim=-1, keepdim=True) * x
    y = y / th.norm(y, dim=-1, keepdim=True)
    z = th.cross(x,y, dim=-1)    
    R = th.stack((x, y, z), dim=-1)
    return R

def pose7d_to_matrix(pose7d: th.Tensor):
    pos = pose7d[:, :3]
    quat = pose7d[:, 3:]
    # quat = quat / quat.norm(dim=1, keepdim=True)
    matrix = th.eye(4).unsqueeze(0).repeat(pose7d.shape[0], 1, 1).to(pose7d.device)
    matrix[:, :3, 3] = pos
    matrix[:, :3, :3] = matrix_from_quaternion(quat)
    return matrix

def pose9d_to_matrix(pose9d: th.Tensor):
    pos = pose9d[:, :3]
    rot6d = pose9d[:, 3:]
    matrix = th.eye(4).unsqueeze(0).repeat(pose9d.shape[0], 1, 1).to(pose9d.device)
    matrix[:, :3, 3] = pos
    matrix[:, :3, :3] = rot6d_to_matrix(rot6d)
    return matrix

def matrix_to_pose9d(matrix:th.Tensor):
    pos = matrix[:, :3, 3]
    rot = matrix[:, :3, :3]
    rot6d = rot[..., :, :2].reshape(*rot.shape[:-2], -1)
    out = th.concat([pos, rot6d], dim = -1)
    return out

@th.jit.script
def _matrix_from_pose(p: th.Tensor, q: th.Tensor,
                      out: th.Tensor):
    _matrix_from_quaternion(q, out[..., :3, :3])
    out[..., :3, 3] = p
    out[..., 3, 3] = 1
    return out


def matrix_from_pose(p: th.Tensor, q: th.Tensor,
                     out: Optional[th.Tensor] = None):
    batch_shape = th.broadcast_shapes(p.shape[:-1], q.shape[:-1])
    if out is None:
        out = th.zeros(*batch_shape, 4, 4,
                       dtype=p.dtype,
                       device=p.device)
    _matrix_from_pose(p, q, out)
    return out


@th.jit.script
def quat_rotate_legacy(q: th.Tensor, x: th.Tensor) -> th.Tensor:
    qx, qy, qz, qw = th.unbind(q, dim=-1)
    vx, vy, vz = th.unbind(x, dim=-1)

    x0 = qw * vx + qy * vz - qz * vy
    x1 = qx * vx + qy * vy + qz * vz
    x2 = qw * vz + qx * vy - qy * vx
    x3 = qw * vy - qx * vz + qz * vx

    return th.stack([
        qw * x0 + qx * x1 + qy * x2 - qz * x3,
        qw * x3 - qx * x2 + qy * x1 + qz * x0,
        qw * x2 + qx * x3 - qy * x0 + qz * x1], dim=-1)


# @th.jit.script
def quat_rotate(q: th.Tensor, x: th.Tensor) -> th.Tensor:
    q_ax = q[..., 0:3]
    t = 2.0 * th.cross(q_ax, x, dim=-1)
    return x + q[..., 3:4] * t + th.cross(q_ax, t, dim=-1)

# @th.jit.script


def random_quat(size, *args, **kwds) -> th.Tensor:
    x0 = th.rand(size, *args, **kwds)
    theta1 = (2.0 * np.pi) * th.rand(size, *args, **kwds)
    theta2 = (2.0 * np.pi) * th.rand(size, *args, **kwds)
    r1 = th.sqrt(1.0 - x0)
    r2 = th.sqrt(x0)
    return th.stack((r1 * th.sin(theta1), r1 * th.cos(theta1),
                     r2 * th.sin(theta2), r2 * th.cos(theta2)), dim=-1)


@th.jit.script
def quat_rotate_as_matrix(q: th.Tensor, x: th.Tensor) -> th.Tensor:
    R = matrix_from_quaternion(q)
    return th.einsum('...ij, ...j -> ...i', R, x)


def quat_inverse(q: th.Tensor, out: Optional[th.Tensor] = None) -> th.Tensor:
    if out is None:
        out = q.clone()
    out.copy_(q)
    out[..., 3] = -out[..., 3]
    return out


def quat_multiply(q1: th.Tensor, q2: th.Tensor,
                  out: Optional[th.Tensor] = None) -> th.Tensor:
    x1, y1, z1, w1 = th.unbind(q1, dim=-1)
    x2, y2, z2, w2 = th.unbind(q2, dim=-1)
    if out is None:
        out = th.empty_like(q1)
    out[...] = th.stack([
        x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2,
        -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2,
        x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2,
        -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2], dim=-1)
    return out

@th.jit.script
def sinc(x:th.Tensor, eps:float=1e-9):
    return th.where(th.abs(x)<eps, th.ones_like(x), th.sin(x)/x)

@th.jit.script
def quat_from_axa(x: th.Tensor, eps: float = 1e-9):
    angle = x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps, max=None)
    theta = (0.5 * angle)
    # xyz = 0.5 * x * th.sinc(theta / np.pi)
    xyz = 0.5 * x * sinc(theta)
    w = theta.cos()
    return th.cat([xyz, w], dim=-1)


@th.jit.script
def axa_from_quat(x: th.Tensor, eps: float = 1e-9):
    # q = sin(h/2)*u, cos(h/2)
    axis = x[..., :3]
    sin_half = th.linalg.norm(axis, dim=-1, keepdim=True)
    angle = 2.0 * th.asin(sin_half)
    return th.where(sin_half < eps,
                    th.zeros_like(axis),
                    axis.div_(sin_half).mul_(angle))


def quat_diff_rad(a: th.Tensor, b: th.Tensor) -> th.Tensor:
    """
    Get the difference in radians between two quaternions.
    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    dq = quat_multiply(a, b_conj)
    # opt1 = 2 * th.acos(th.abs(dq[..., -1]).clamp_max(1.0))
    # opt2 = 2.0 * th.asin(th.clamp(th.norm(dq[..., 0:3], p=2, dim=-1),
    #                               max=1.0))
    # s = th.norm(dq[..., 0:3], p=2, dim=-1)
    sin_half = th.linalg.norm(dq[..., :3],
                              dim=-1)
    angle = 2.0 * th.asin(sin_half)
    # c = dq[..., 3]
    # opt3 = 2.0 * th.atan2(s, c)
    # normalize
    # opt3 = (opt3 + th.pi) % (2 * th.pi) - th.pi
    # return opt3
    return th.abs(angle)


@th.jit.script
def skew_matrix(x: th.Tensor):
    """ matrix representation of cross product """
    out = th.zeros(x.shape[:-1] + (3, 3),
                   dtype=x.dtype,
                   device=x.device)
    # out = np.zeros((3, 3), dtype=x.dtype)
    out[..., 0, 1] = -x[..., 2]
    out[..., 0, 2] = x[..., 1]
    out[..., 1, 0] = x[..., 2]
    out[..., 1, 2] = -x[..., 0]
    out[..., 2, 0] = -x[..., 1]
    out[..., 2, 1] = x[..., 0]

    return out


@th.jit.script
def orientation_error(desired: th.Tensor, current: th.Tensor):
    # refer to
    # Operational SpaceControl: A Theoreticaland EmpiricalComparison
    # skew_desired = skew_matrix(desired[:, :3])
    return -(current[:, :3] * desired[:, -1:] - desired[:, :3]
             * current[:, -1:] + th.cross(desired[:, :3], current[:, :3], -1))
    # th.einsum('Bij,Bj->Bj',skew_desired, current[:,:3]))


@th.jit.script
def axisaToquat(axisA: th.Tensor):

    num_rotations = axisA.shape[0]
    device = axisA.device
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


@th.jit.script
def adjoint_matrix(T: th.Tensor):
    """
    6x6 adjoint representation
    NOTE: assume [v; w] ordering?
    """
    out = th.zeros(
        T.shape[:-2] + (6, 6),
        dtype=T.dtype,
        device=T.device)

    R = T[..., :3, :3]
    p = T[..., :3, 3]
    P = skew_matrix(p)

    out[..., :3, :3] = R
    out[..., 3:, :3] = P @ R
    out[..., 3:, 3:] = R
    return out

    # return np.block([
    #     [R, np.zeros((3, 3))],
    #     [P @ R, R]
    # ])


@th.jit.script
def _invert_transform(T: th.Tensor, out: th.Tensor) -> th.Tensor:
    R = T[..., :3, :3]
    t = T[..., :3, 3:]

    out[..., :3, :3] = R.swapaxes(-1, -2)
    out[..., :3, 3:] = -th.einsum('...ba,...bc->...ac', R, t)
    out[..., 3, 3] = 1
    return out


def invert_transform(T: th.Tensor, out: Optional[th.Tensor] = None):
    if out is None:
        out = th.zeros_like(T)
    return _invert_transform(T, out)


def check_time_for_quat_rotate():
    import time
    q = th.nn.functional.normalize(th.rand(size=(1024, 1, 4)))
    x = th.randn(size=(1024, 512, 3))

    ts = []
    for i in range(2):
        if i > 0:
            ts.append(time.time())
        x2 = quat_rotate_legacy(q, x)
        if i > 0:
            ts.append(time.time())
        x3 = quat_rotate(q, x)
        if i > 0:
            ts.append(time.time())
        x4 = quat_rotate_as_matrix(q, x)
        if i > 0:
            ts.append(time.time())
        print((x2 - x3).mean())
        print((x3 - x4).mean())
    print(np.diff(ts))
    # print('x2', x2)
    # print('x3', x3)


def check_m2q_q2m():
    quat = th.randn(size=(1024, 4))
    quat /= th.linalg.norm(quat, dim=-1, keepdim=True)
    quat2 = quaternion_from_matrix(matrix_from_quaternion(quat))
    delta = quat_multiply(quat_inverse(quat), quat2)
    angle = th.linalg.norm(axa_from_quat(delta), dim=-1)
    print(angle.min(), angle.max())
    print(quat[0], quat2[0])


def main():
    check_m2q_q2m()


if __name__ == '__main__':
    main()
