#!/usr/bin/env python3

from typing import Callable

import torch as th
import numpy as np
import cv2
import einops

from util.torch_util import dcn
from env.env.wrap.base import WrapperEnv

from icecream import ic

from env.util import (
    draw_cloud_with_sphere,
    draw_patch_with_cvxhull
)


def apply_colormap(value: th.Tensor, cmap: th.Tensor,
                   normalize: bool = True):
    if normalize:
        # map value to colormap index.
        vmin, vmax = value.min(), value.max()
        # bound = (value - vmin) / (vmax - vmin) * th.arange(
        #         cmap.shape[0], dtype=value.dtype, device=value.device)
        # )
        # th.bucketize(value, bound, right = True)
        value = (value - vmin) / (vmax - vmin)
    index = ((value * cmap.shape[0])
             .to(dtype=th.long)
             .clamp_(0, cmap.shape[0] - 1))
    return cmap[index]


def colorize_attention(
        attn: th.Tensor,
        cmap: th.Tensor,
        normalize: bool = True):
    if normalize:
        attn = attn / attn.max(dim=-1, keepdims=True).values
    attn[attn < 0.5] = 0.0
    colors = apply_colormap(attn, cmap)
    return colors


class DrawPatchAttention(WrapperEnv):
    def __init__(self, env,
                 patch_attn_fn: Callable,
                 style: str = 'star',
                 check_viewer: bool = True,

                 dilate: float = 1.1
                 ):
        super().__init__(env)
        # SAVE SOME PARAMS
        self.__dilate = dilate

        # CREATE COLORMAP FROM CV2
        CMAP_VIRIDIS = (
            cv2.applyColorMap(
                np.arange(255, dtype=np.uint8),
                colormap=cv2.COLORMAP_VIRIDIS)[..., :: -1] / 255.0)
        # MOVE TO DEVICE
        self.__colormap = th.as_tensor(CMAP_VIRIDIS,
                                       dtype=th.float32,
                                       device=self.device)
        self.__patch_attn_fn = patch_attn_fn
        self.style = style

    @property
    def patch_attn_fn(self):
        return self.__patch_attn_fn

    def __draw_star(self, patch, patch_color):
        point_color = einops.repeat(patch_color, '... d -> ... p d',
                                    p=patch.shape[-2])
        # lines connecting patch center to
        # the members of the patch
        center = patch.mean(dim=-2, keepdim=True).expand(patch.shape)
        # dilate by a bit for better visibility
        member = center + self.__dilate * (patch - center)
        lines = th.cat([member, patch], dim=-1)
        lines = einops.rearrange(lines, '... s p d -> ... (s p) d')
        lines = dcn(lines)
        point_color = dcn(point_color)
        for i in range(self.num_env):
            self.gym.add_lines(
                self.viewer, self.envs[i],
                len(lines[i]),
                lines[i],
                point_color[i].reshape(-1, 3),
            )

    def __draw_cloud(self, patch, patch_color):
        # point cloud
        point_color = einops.repeat(patch_color, '... d -> ... p d',
                                    p=patch.shape[-2])
        patch = dcn(patch)
        point_color = dcn(point_color)
        for i in range(self.num_env):
            draw_cloud_with_sphere(self.gym, self.viewer,
                                   patch[i].reshape(-1, 3),
                                   self.envs[i],
                                   color=point_color[i].reshape(-1, 3),
                                   radius=0.002)

    def __draw_hull(self, patch, patch_color):
        # edges of patch hull
        patch = dcn(patch)
        patch_color = dcn(patch_color)
        for i in range(self.num_env):
            draw_patch_with_cvxhull(self.gym, self.viewer,
                                    patch[i],
                                    self.envs[i],
                                    color=patch_color[i].reshape(-1, 3))

    def __draw(self, obs):
        # == draw ==
        # we assume patch_attn_fn returns a pair of
        # (patch: A[..., S, P, 3])
        # (attn : A[..., S])
        patch, attn = self.__patch_attn_fn(obs)
        ic(patch.shape)
        ic(attn.shape)
        patch_color = colorize_attention(attn, self.__colormap,
                                         normalize=True)  # B, S, D

        if self.style == 'star':
            self.__draw_star(patch, patch_color)
        elif self.style == 'cloud':
            self.__draw_cloud(patch, patch_color)
        elif self.style == 'hull':
            self.__draw_hull(patch, patch_color)

    def step(self, *args, **kwds):
        # == rollout ==
        obs, rew, done, info = super().step(*args, **kwds)
        if (not self.check_viewer) or (self.viewer is not None):
            self.__draw(obs)
        return obs, rew, done, info
