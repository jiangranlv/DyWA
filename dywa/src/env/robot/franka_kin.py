#!/usr/bin/env python3

from typing import Optional
from functools import reduce
from itertools import accumulate

import torch as th
import einops

from cxx.franka_kin_cuda import forward as _franka_ik_fwd
# from imm_gym.cxx.franka_kin_cuda import forward as _franka_ik_fwd


def matrix_from_dh(dh_base: th.Tensor, q: th.Tensor,
                   T_out: Optional[th.Tensor] = None):
    # Define Transformation matrix based on DH params
    alpha, a, d = th.unbind(dh_base, dim=-1)
    cq, sq = th.cos(q), th.sin(q)
    ca, sa = th.cos(alpha), th.sin(alpha)
    z, o = th.zeros_like(q), th.ones_like(q)

    #if T_out is None:
    #    T_out = th.empty(*q.shape[:-1], 4, 4,
    #                     dtype=q.dtype,
    #                     device=q.device)
    #out_buf = T_out.reshape(*q.shape[:-1], 16)
    T = th.stack([cq, -sq, z, a,
                  sq * ca, cq * ca, -sa, -sa * d,
                  sq * sa, cq * sa, ca, ca * d,
                  z, z, z, o], dim=-1)#, out=out_buf)
    T = T.reshape(*dh_base.shape[:-1], 4, 4)
    return T


def franka_fk(q: th.Tensor,
              return_intermediate: bool = False,
              tool_frame: bool = True
              ):
    # Partial DH paramters without joint values
    DH_BASE = th.as_tensor([[0, 0, 0.333],
                            [-th.pi / 2, 0, 0],
                            [th.pi / 2, 0, 0.316],
                            [th.pi / 2, 0.0825, 0],
                            [-th.pi / 2, -0.0825, 0.384],
                            [th.pi / 2, 0, 0],
                            # NOTE:
                            # `0.107` accounts for flange offset;
                            # `0.1034` accounts for gripper offset.
                            [th.pi / 2, 0.088,
                             (0.107 + 0.1034 if tool_frame else 0.0)],
                            ],
                           dtype=q.dtype,
                           device=q.device)
    qs = q.reshape(-1, q.shape[-1])
    DHS = einops.repeat(DH_BASE, '... -> n ...', n=qs.shape[0])
    TS = matrix_from_dh(DHS, qs)

    if return_intermediate:
        return list(accumulate(TS.unbind(dim=-3), th.bmm))
    else:
        T = reduce(th.bmm, TS.unbind(dim=-3))

        # Account for rotational offset ...
        pi_4 = th.pi / 4
        T[..., :3, :3] = T[..., :3, :3] @ th.as_tensor([
            [pi_4, pi_4, 0],
            [-pi_4, pi_4, 0],
            [0, 0, 1.0000000]],
            dtype=T.dtype,
            device=T.device)
        return T


def franka_ik(T_tip: th.Tensor,
              q_out: Optional[th.Tensor] = None,
              width: float = 0.08,
              n_iter: int = 100) -> th.Tensor:
    if q_out is None:
        # 7(arm joints) + 2(gripper) + 1(success flag)
        q_out = th.empty(*T_tip.shape[:-2], 10,
                         dtype=T_tip.dtype,
                         device=T_tip.device)
    _franka_ik_fwd(T_tip, q_out, width, n_iter)
    return q_out


def test_cycle():
    q = th.empty((4, 7), device='cuda:0').uniform_(-th.pi, +th.pi)
    T = franka_fk(q)
    q2 = franka_ik(T)
    T2 = franka_fk(q2[..., :7])
    q3 = franka_ik(T2)
    T3 = franka_fk(q3[..., :7])
    print('T')
    print(T)
    print('T2')
    print(T2)
    print('T3')
    print(T3)

    print('diffs')
    # print(T @ th.linalg.inv(T2))
    # print(T2 @ th.linalg.inv(T))
    print(th.linalg.inv(T2) @ T)
    print(th.linalg.inv(T) @ T2)

    # print(T2 @ th.linalg.inv(T3))
    # print(T3 @ th.linalg.inv(T2))
    print(th.linalg.inv(T3) @ T2)
    print(th.linalg.inv(T2) @ T3)


def main():
    q = th.empty((1, 7), device='cuda:0').uniform_(-th.pi, +th.pi)
    T = franka_fk(q)
    Ts = franka_fk(q, return_intermediate=True)
    print(T)
    print(Ts)


if __name__ == '__main__':
    main()
