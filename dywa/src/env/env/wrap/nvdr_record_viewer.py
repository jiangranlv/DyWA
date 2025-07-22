#!/usr/bin/env python3

from typing import Tuple, Optional, Callable

from isaacgym import gymtorch
from isaacgym import gymapi
from pathlib import Path

from dataclasses import dataclass
from util.config import ConfigBase

from einops import rearrange
import numpy as np
import torch as th

from util.path import ensure_directory

from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv
from env.env.wrap.nvdr_camera_wrapper import NvdrCameraWrapper
from env.env.help.apply_camera_transform import ApplyCameraTransform
from env.robot.cube_poker import CubePoker
from env.robot.ur5_fe import UR5FE

from util.torch_util import dcn
from util.vis.img import tile_images, to_hwc
import cv2


class NvdrRecordViewer(WrapperEnv):
    """
    Wrapper to record images from gym viewer.
    """

    @dataclass
    class Config(ConfigBase):
        record_dir: str = '/tmp/docker/record'
        record_reward: bool = True
        img_size: Tuple[int, int] = (224, 224)

        # cam_eye: Tuple[float, float, float] = (0.0, 0.0, 1.2)
        # cam_at: Tuple[float, float, float] = (0.0, 0.0, 0.4)
        cam_eye: Tuple[float, float, float] = (0.54, 0.0, 0.9)
        cam_at: Tuple[float, float, float] = (-0.2, 0.0, 0.4)

        # NOTE: render `visual mesh` for objects, by default.
        use_col: bool = True

    def __init__(self, cfg: Config, env: EnvIface,
                 on_step: Optional[Callable[[th.Tensor], None]] = None,
                 **kwds):
        super().__init__(env)
        self.cfg = cfg
        self.img_env = NvdrCameraWrapper(
            env,
            NvdrCameraWrapper.Config(
                img_size=cfg.img_size,

                eye=cfg.cam_eye,
                at=cfg.cam_at,

                use_flow=False,
                use_label=False,
                use_color=True,

                use_col=cfg.use_col,
                **kwds
            )
        )
        self._record_dir: Path = ensure_directory(
            cfg.record_dir)

        # Minimize the likelihood of colliding
        # with other properties...
        self._record_step: int = 0
        self._on_step = on_step

        self.project = ApplyCameraTransform()
        self.project.reset(self)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def __get_lines(self, lines) -> Tuple[np.ndarray, np.ndarray]:
        vss = []
        css = []
        nss = []
        for env_id in sorted(lines.keys()):
            # self.lines[env_id].append((num_lines, vertices.copy(), colors.copy()))
            vs = []
            cs = []
            ns = []
            for line in lines[env_id]:
                n, v, c = line
                vs.append(v)
                cs.append(c)
                ns.append(n)
            vss.append(np.concatenate(vs, axis=0).reshape(-1, 6))
            css.append(np.concatenate(cs, axis=0).reshape(-1, 3))
            nss.append(ns)

        if len(vss) <= 0:
            vss = np.empty((self.env.num_env, 0, 6))
            css = np.empty((self.env.num_env, 0, 3))
            nss = np.empty((self.env.num_env, 0))
        else:
            vss = np.stack(vss, axis=0)
            css = np.stack(css, axis=0)
            nss = np.stack(nss, axis=0)
        return (vss, css, nss)

    def __draw_lines(self, rgb_imgs_np: np.ndarray):
        if not hasattr(self.env.gym, 'lines'):
            return

        # Draw all debug lines on our image as well
        vertices, colors, counts = self.__get_lines(self.env.gym.lines)
        vertices = vertices.reshape(self.env.num_env, -1, 3)
        vertices = dcn(self.project(vertices, self.cfg.img_size))
        lines = vertices.reshape(self.env.num_env, -1, 2, 2)
        lines_np = dcn(lines)

        def _to_cvpt(x):
            return tuple(int(e) for e in x)

        for i in range(self.env.num_env):
            for l in range(lines_np.shape[1]):
                color = (255 *
                         (colors[i, l]).clip(0.0, 1.0)).astype(np.uint8)
                cv2.line(
                    rgb_imgs_np[i],
                    _to_cvpt(lines_np[i, l, 0]),
                    _to_cvpt(lines_np[i, l, 1]),
                    _to_cvpt(color),
                    thickness=1
                )

    def _record(self, obs):
        # Bypass...
        alt = self.img_env._wrap_obs(obs)
        rgb_imgs = alt['color']  # NCHW
        if self._on_step is not None:
            self._on_step(rgb_imgs)

        # [1] CHW->HWC, [2] Torch->Numpy
        rgb_imgs_np = dcn(to_hwc(rgb_imgs))
        # [3] FP32 -> UINT8
        if rgb_imgs_np.dtype != np.uint8:
            rgb_imgs_np = (255 * rgb_imgs_np.clip(0.0, 1.0)).astype(np.uint8)
        # [4] DRAW LINES
        self.__draw_lines(rgb_imgs_np)

        # [5] RGB->BGR, [6] TILE
        rgb_imgs_np = tile_images(rgb_imgs_np[..., ::-1])

        filename: str = str(self._record_dir / F'{self._record_step:04d}.png')
        # RGB->BGR
        cv2.imwrite(filename, rgb_imgs_np)
        self._record_step += 1

    def step(self, *args, **kwds):
        # Step original env.
        obs, rew, done, info = self.env.step(*args, **kwds)
        # Record.
        self._record(obs)
        # Return original outputs.
        return (obs, rew, done, info)
