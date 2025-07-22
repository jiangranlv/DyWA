#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
import einops


class ImageToPatchSeq:
    """
    Convert a "full" image into a sequence of (padded) patches.
    """

    @dataclass(frozen=True)
    class Config:
        patch_size: int = 1

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, img: np.ndarray):
        cfg = self.cfg

        # -> numpy
        img = np.asarray(img)

        # -> pad
        h, w = img.shape[:2]
        dh = (cfg.patch_size - h) % cfg.patch_size
        dw = (cfg.patch_size - w) % cfg.patch_size
        patches = np.pad(img, ((0, dh), (0, dw)))

        # -> to seq
        seqs = einops.rearrange(
            patches,
            '(nh ph) (nw pw) -> (nh nw) (ph pw)',
            ph=cfg.patch_size,
            pw=cfg.patch_size)

        # -> to float in range (-0.5, 0.5)
        seqs = seqs.astype(np.float32) * float(1.0 / 255) - 0.5
        return seqs
