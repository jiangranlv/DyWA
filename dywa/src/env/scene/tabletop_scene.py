#!/usr/bin/env python3

from typing import Tuple
from dataclasses import dataclass
from util.config import ConfigBase

from isaacgym import gymtorch
from isaacgym import gymapi

from env.scene.base import SceneBase


class TableTopScene(SceneBase):
    @dataclass
    class Config(ConfigBase):
        table_dims: Tuple[float, float, float] = (0.4, 1.0, 0.4)
        table_pos: Tuple[float, float, float] = (0.0, 0.0, 0.2)
        table_orn: Tuple[float, float, float, float] = (0, 0, 0, 1)

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def reset(self, gym, sim, env):
        # TODO: consider (is it worth it?)
        # domain-randomization on:
        # * table pose
        # * table dimensions
        # * table texture (color)
        return

    def create_actors(self, gym, sim, env,
                      env_id: int):
        cfg = self.cfg
        # Spawn table.
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(*cfg.table_pos)
        table_pose.r = gymapi.Quat(*cfg.table_orn)
        table_actor = gym.create_actor(
            env, self.assets['table'],
            table_pose, F'table', env_id, 0)
        return {'table': table_actor}

    def create_assets(self, gym, sim):
        cfg = self.cfg
        # Create table asset.
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        table_asset = gym.create_box(sim,
                                     *cfg.table_dims,
                                     asset_options)
        self.assets = {'table': table_asset}
        return dict(self.assets)
