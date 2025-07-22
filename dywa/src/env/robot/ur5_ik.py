#!/usr/bin/env python3

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from typing import Optional
import einops

import torch as th

if False:
    # JIT
    from torch.utils.cpp_extension import load
    from util.path import get_path
    d = get_path('../../../')

    ur5_ik = load(
        name='ur5_ik_cuda',
        sources=[
            F'{d}/ur5_kin_cuda.cpp',
            F'{d}/ur5_kin_cuda_kernel.cu'],
        verbose=True)
else:
    # COMPILED
    from cxx.ur5_kin_cuda import forward as ur5_ik


def ur5_fk(q: th.Tensor, T: Optional[th.Tensor] = None) -> th.Tensor:
    if T is None:
        T = th.zeros(q.shape[:-1] + (4, 4), dtype=q.dtype,
                     device=q.device)
    d1 = 0.089159
    a2 = -0.42500
    a3 = -0.39225
    d4 = 0.10915
    d5 = 0.09465
    d6 = 0.0823

    s = th.sin(q)
    c = th.cos(q)

    c1, c2, c3, c4, c5, c6 = th.unbind(c, dim=-1)
    s1, s2, s3, s4, s5, s6 = th.unbind(s, dim=-1)

    q23 = q[..., 1] + q[..., 2]
    q234 = q23 + q[..., 3]
    s23 = th.sin(q23)
    c23 = th.cos(q23)
    s234 = th.sin(q234)
    c234 = th.cos(q234)

    T[..., 0, 0] = c234 * c1 * s5 - c5 * s1
    T[..., 0, 1] = c6 * (s1 * s5 + c234 * c1
                         * c5) - s234 * c1 * s6
    T[..., 0, 2] = -s6 * (s1 * s5 + c234 * c1
                          * c5) - s234 * c1 * c6
    T[..., 0, 3] = (d6 * c234 * c1 * s5 -
                    a3 * c23 * c1 -
                    a2 * c1 * c2
                    - d6 * c5 * s1
                    - d5 * s234 * c1 -
                    d4 * s1)

    T[..., 1, 0] = c1 * c5 + c234 * s1 * s5
    T[..., 1, 1] = -c6 * (c1 * s5 - c234 * c5 * s1
                          ) - s234 * s1 * s6
    T[..., 1, 2] = s6 * (c1 * s5 - c234 * c5 * s1
                         ) - s234 * c6 * s1
    T[..., 1, 3] = (d6 * (c1 * c5 + c234 * s1 *
                          s5) + d4 * c1 - a3 * c23 *
                    s1 - a2 * c2 * s1 - d5 *
                    s234 * s1)

    T[..., 2, 0] = -s234 * s5
    T[..., 2, 1] = -c234 * s6 - s234 * c5 * c6
    T[..., 2, 2] = s234 * c5 * s6 - c234 * c6
    T[..., 2, 3] = (d1 + a3 * s23 + a2 * s2 - d5 * (c23 * c4 - s23 * s4) -
                    d6 * s5 * (c23 * s4 + s23 * c4))

    T[..., 3, 0] = 0
    T[..., 3, 1] = 0
    T[..., 3, 2] = 0
    T[..., 3, 3] = 1
    return T


def main():
    # help(ur5_ik.forward)
    T = th.eye(4, dtype=th.float32, device='cuda:0')
    T[0, 3] = 0.6
    T[2, 3] = 0.2
    T = einops.repeat(T, '... -> d ...', d=1).contiguous()
    q0 = th.zeros((1, 6), dtype=th.float32, device=T.device)
    q1 = th.empty_like(q0)
    print(T.shape)
    print('q0', q0.shape)
    # with th.no_grad():
    q = ur5_ik(T, q0, q1)  # .contiguous().detach().clone()
    T2 = ur5_fk(q)
    print(T, T2)
    print(q1.grad)
    print(q1)
    print(th.isnan(q1).sum())
    print(q.data)
    q2 = th.as_tensor(q)
    q3 = q2.detach()
    q4 = q3.clone()  # issue
    print(q2)
    # print(q.detach().clone().numpy())
    # time.sleep(1000)
    print(type(q))
    print(q.dtype)
    print(q.shape)
    print(q.layout)
    print(q)


if __name__ == '__main__':
    main()


# @th.jit.script
# def ur5_ik(T: th.Tensor, q: th.Tensor) -> int:
#     # UR5 parameters.
#     d1 = 0.089159
#     a2 = -0.42500
#     a3 = -0.39225
#     d4 = 0.10915
#     d5 = 0.09465
#     d6 = 0.0823
#     PI = math.pi
#     ZERO_THRESH = 0.00000001
#     q6_des = 0.0

#     num_sols = 0
#     T02 = -T[..., 0]
#     T00 = T[..., 1]
#     T01 = T[..., 2]
#     T03 = -T[..., 3]
#     T12 = -T[..., 4]
#     T10 = T[..., 5]
#     T11 = T[..., 6]
#     T13 = -T[..., 7]
#     T22 = T[..., 8]
#     T20 = -T[..., 9]
#     T21 = -T[..., 10]
#     T23 = T[..., 11]

#     # shoulder rotate joint (q1)
#     q1 = [0, 0]
#     A = d6 * T12 - T13
#     B = d6 * T02 - T03
#     R = A * A + B * B
#     if abs(A) < ZERO_THRESH:
#         div = -d4 / B if abs(abs(d4) - abs(B)
#                              ) >= ZERO_THRESH else -th.sign(d4) * th.sign(B)
#         arcsin = th.asin(div)
#         if abs(arcsin) < ZERO_THRESH:
#             arcsin = 0.0
#         if arcsin < 0.0:
#             q1[0] = arcsin + 2.0 * PI
#         else:
#             q1[0] = arcsin
#         q1[1] = PI - arcsin
#     elif abs(B) < ZERO_THRESH:
#         div = d4 / A if abs(abs(d4) - abs(A)
#                             ) >= ZERO_THRESH else th.sign(d4) * th.sign(A)
#         arccos = th.acos(div)
#         q1[0] = arccos
#         q1[1] = 2.0 * PI - arccos
#     elif d4 * d4 > R:
#         return num_sols
#     else:
#         arccos = th.acos(d4 / th.sqrt(R))
#         arctan = th.atan2(-B, A)
#         pos = arccos + arctan
#         neg = -arccos + arctan
#         if abs(pos) < ZERO_THRESH:
#             pos = 0.0
#         if abs(neg) < ZERO_THRESH:
#             neg = 0.0
#         if pos >= 0.0:
#             q1[0] = pos
#         else:
#             q1[0] = 2.0 * PI + pos
#         if neg >= 0.0:
#             q1[1] = neg
#         else:
#             q1[1] = 2.0 * PI + neg
