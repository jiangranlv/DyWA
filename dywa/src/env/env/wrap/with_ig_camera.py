#!/usr/bin/env python3

from env.env.wrap.base import ObservationWrapper
from env.env.help.with_camera import WithCamera

from dataclasses import dataclass
from typing import Optional, Iterable

from gym import spaces


class WithCameraWrapper(ObservationWrapper):
    """
    Env wrapper to add camera at each env.
    """

    @dataclass
    class Config:
        camera: WithCamera.Config = WithCamera.Config()

    def __init__(self, cfg: Config, env):
        super().__init__(env, self._wrap_obs)
        # NOTE: this hides env.cfg
        # but it should be OK as long as
        # no one tries to "introspect" the cfg variable
        self.cfg = cfg
        self.camera = WithCamera(cfg.camera)
        # ic(env.observation_space)
        # ic(self.camera.observation_space)
        obs_dict = {
            'raw_obs': env.observation_space,
            'images': self.camera.observation_space
        }
        if env.observation_space is None:
            obs_dict.pop('raw_obs')
        self.obs_space = spaces.Dict(obs_dict)
        self.__first = True

    @property
    def observation_space(self):
        return self.obs_space

    def setup(self):
        out = super().setup()
        self.camera.setup(self)
        return out

    def reset_indexed(self, indices: Optional[Iterable[int]] = None):
        # out = self.env.reset_indexed(indices)
        out = super().reset_indexed(indices)
        if self.__first:
            self.camera.reset(self.gym, self.sim,
                              self, indices)
            self.__first = False
        return out

    def reset(self):
        # Hmmm....
        # TODO: potentially care about
        # the output of this thing
        _ = self.reset_indexed()

        # FIXME: remove code duplication between
        # this function and EnvBase.reset()!!
        # This part achieves the following two goals:
        # 1. simulate (commit reset results)
        # 2. compute and return observations.
        self.gym.simulate(self.sim)
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        self.refresh_tensors()
        obs = self.compute_observations()
        return self._wrap_obs(obs)

    def _wrap_obs(self, obs):
        images = self.camera.step(self)
        out = {
            'images': images,
            'raw_obs': obs
        }
        if obs is None:
            out.pop('raw_obs')
        return out
