#!/usr/bin/env python3

from typing import Optional
from env.env.help.get_bbox import GetBbox
import torch as th
from kornia.geometry.transform.crop2d import crop_and_resize


class CropBbox:
    def __init__(self, overwrite: bool = False):
        self.bbox = GetBbox(mode='corner')
        self.overwrite = overwrite

    def reset(self, env: 'NvdrRecordViewer'):
        self.env = env
        self.bbox.reset(env)

    def __call__(self, img: th.Tensor,
                 center: Optional[th.Tensor] = None,
                 radius: Optional[th.Tensor] = None
                 ):
        bbox = self.bbox(img.shape,
                         center, radius)
        crop = crop_and_resize(img, bbox,
                               img.shape[-2:])
        # JUST FOR TEMPORARY VISUALIZATION
        if self.overwrite:
            img[...] = crop
        return {'bbox': bbox, 'crop': crop}
