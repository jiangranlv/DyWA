#!/usr/bin/env python3

from typing import List
from torchvision.transforms import Normalize
from torchvision.transforms.functional import normalize
import numpy as np


def unnormalize(
        tensor,
        mean: List[float],
        std: List[float],
        inplace: bool = False):
    """ Inverse of VF.normalize() with identical args."""
    return normalize(tensor, -mean / std, std, inplace)


class UnNormalize(Normalize):
    """ Inverse of Normalize() with identical args."""

    def __init__(self, mean, std, inplace):
        offset = -np.divide(mean, std)
        scale  = np.reciprocal(std)
        super().__init__(offset, scale, inplace)


def main():
    pass
