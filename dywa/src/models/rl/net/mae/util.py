#!/usr/bin/env python3

from typing import Optional

import numpy as np
import torch as th
from einops import repeat
import cv2

from util.vis.img import (normalize_image, tile_images)



def show_attention_overlay(
        attn: np.ndarray,
        image: np.ndarray,
        patch_size: Optional[int] = None):
    """
    Visualize attention overlay on the source image.
    """

    # Try to figure out patch dimensions by
    # assuming square grid layout.
    seq_length: int = attn.shape[-1]
    num_patch: int = int(np.sqrt(seq_length))
    if patch_size is None:
        patch_size = image.shape[-2] // num_patch

    for head in range(attn.shape[-2]):
        amap = attn[:, head].reshape(attn.shape[0], num_patch, num_patch, 1)
        amap = repeat(amap, '... h w c -> ... (h n) (w m) c',
                      n=patch_size, m=patch_size)
        green = normalize_image(tile_images(image.squeeze(axis=-3)[..., None]))
        red = normalize_image(tile_images(amap))
        blue = np.zeros_like(red)
        rgb = np.concatenate([blue, green, red], axis=-1)
        key: str = F'attn-head#{head}'
        cv2.namedWindow(key, cv2.WINDOW_NORMAL)
        cv2.imshow(key, rgb)
        cv2.waitKey(1)
