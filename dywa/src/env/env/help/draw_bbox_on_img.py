#!/usr/bin/env python3

from env.env.help.get_bbox import GetBbox
import torch as th
from torchvision.utils import draw_bounding_boxes


class DrawBboxOnImg:
    def __init__(self):
        self.bbox = GetBbox(mode='minmax')

    def reset(self, env: 'NvdrRecordViewer'):
        self.env = env
        self.bbox.reset(env.img_env)

    def __call__(self, img: th.Tensor):
        bbox = self.bbox(img.shape)
        img = th.unbind(img, dim=0)
        for i in range(self.env.num_env):
            input_img = (255 * img[i]).to(dtype=th.uint8)
            bbox_img = draw_bounding_boxes(
                input_img, bbox[None, i, [0, 0, 1, 1], [0, 1, 0, 1]],
                colors='red')
            img[i][...] = (bbox_img / 255.0).to(dtype=img[i].dtype,
                                                device=img[i].device)
