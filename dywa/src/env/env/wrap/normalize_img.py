#!/usr/bin/env python3

from typing import Tuple

from torchvision.transforms import Normalize
from data.transforms.common import WrapDict

from env.env.wrap.base import ObservationWrapper


class NormalizeImg(ObservationWrapper):
    def __init__(self, env,
                 img_mean: Tuple[float, ...],
                 img_std: Tuple[float, ...],
                 key: str):
        super().__init__(env, self._wrap_obs)
        self.normalize_img = WrapDict(Normalize(
            img_mean, img_std), key, key)

    @property
    def observation_space(self):
        # FIXME: technically speaking, bounds here should be
        # updated to account for normalization.
        return self.env.observation_space

    def _wrap_obs(self, x):
        return self.normalize_img(x)
