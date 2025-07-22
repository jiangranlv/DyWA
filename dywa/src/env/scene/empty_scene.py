#!/usr/bin/env python3

from env.scene.base import SceneBase


class EmptyScene(SceneBase):
    def reset(self, gym, sim, env):
        return

    def create_actors(self, gym, sim, env):
        return {}

    def create_assets(self, gym, sim, env):
        return {}
