#!/usr/bin/env python3

import torch as th


def hsv2rgb_torch(hsv: th.Tensor) -> th.Tensor:
    hsv_h, hsv_s, hsv_l = hsv[..., 0:1], hsv[..., 1:2], hsv[..., 2:3]
    _c = hsv_l * hsv_s
    _x = _c * (- th.abs(hsv_h * 6. % 2. - 1) + 1.)
    _m = hsv_l - _c
    _o = th.zeros_like(_c)
    idx = (hsv_h * 6.).type(th.uint8)
    idx = (idx % 6).expand(-1, -1, -1, 3)
    rgb = th.empty_like(hsv)
    rgb[idx == 0] = th.cat([_c, _x, _o], dim=-1)[idx == 0]
    rgb[idx == 1] = th.cat([_x, _c, _o], dim=-1)[idx == 1]
    rgb[idx == 2] = th.cat([_o, _c, _x], dim=-1)[idx == 2]
    rgb[idx == 3] = th.cat([_o, _x, _c], dim=-1)[idx == 3]
    rgb[idx == 4] = th.cat([_x, _o, _c], dim=-1)[idx == 4]
    rgb[idx == 5] = th.cat([_c, _o, _x], dim=-1)[idx == 5]
    rgb += _m
    return rgb


def flow_image(flow: th.Tensor, eps: float = 1e-6):
    axis: int = -1
    flo_ang = th.atan2(flow[..., 1], flow[..., 0])
    h = (flo_ang + th.pi) / (2.0 * th.pi)  # 0 ~ 1
    flo_mag = th.linalg.norm(flow, dim=axis)  # ...HW
    smax = th.amax(flo_mag, dim=(-2, -1), keepdim=True)  # N11
    s = flo_mag / (smax + eps)
    # Value is always one.
    v = th.ones_like(h)
    # NOTE: hsv axis here needs to be the last dimension.
    hsv = th.stack([h, s, v], dim=-1)
    rgb = hsv2rgb_torch(hsv)
    img = rgb
    return img
