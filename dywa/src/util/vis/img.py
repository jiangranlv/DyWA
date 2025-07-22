#!/usr/bin/env python3

from typing import Optional, Dict

import torch as th
import numpy as np
from einops import rearrange


def to_hwc(x):
    return rearrange(x, '... c h w -> ... h w c')


def to_chw(x):
    return rearrange(x, '... h w c -> ... c h w')


def normalize_image(x: np.ndarray):
    maxx = x.max()
    minx = x.min()
    return (x - minx) / (maxx - minx)


def digitize_image(x: np.ndarray, num_bins: int = 255,
                   aux: Optional[Dict[str, np.ndarray]] = None):
    """ slow, to only use for visualization . """
    bins = np.quantile(x, np.linspace(0, 1, num_bins))
    if aux is not None:
        aux['bins'] = bins
    i = np.searchsorted(bins, x, side='left')
    return i


def tile_images(x: np.ndarray):
    """
    Tile input images to a nearest square grid.
    """
    assert (len(x.shape) == 4)  # NHWC
    n: int = x.shape[0]
    dim = int(np.ceil(np.sqrt(x.shape[0])))
    shape = (dim * dim,) + tuple(x.shape[1:])

    # Create output buffer as a grid.
    if isinstance(x, th.Tensor):
        grid = th.zeros(shape,
                        device=x.device,
                        dtype=x.dtype)
    else:
        grid = np.zeros(shape, dtype=x.dtype)

    # Copy input to grid.
    grid[:x.shape[0]] = x

    # Just for convenience, we'll do per-image normalization.
    # vmin = th.amin(grid, dim=(1, 2, 3), keepdim=True)
    # vmax = th.amax(grid, dim=(1, 2, 3), keepdim=True)
    # grid = (grid - vmin) / (vmax - vmin)

    # Apply tiling.
    grid = rearrange(grid, '(d1 d2) h w c -> (d1 h) (d2 w) c',
                     d1=dim, d2=dim)
    return grid


def test_tile_images():
    x = np.zeros((7, 64, 64, 3))
    y = tile_images(x)
    print(y.shape)

    x = th.zeros((7, 64, 64, 3))
    y = tile_images(x)
    print(y.shape)


def plot_digitize_image():
    from matplotlib import pyplot as plt
    x = np.random.normal(size=(48, 48))
    # d = digitize_image(x, 256).astype(np.uint8)
    # d = normalize_image(x)

    # plt.hist(d.ravel())
    # plt.show()

    fig, ax = plt.subplot_mosaic([['a', 'b']])
    ax['a'].imshow(x)
    # ax['b'].imshow(d)
    plt.show()


def main():
    test_tile_images()


if __name__ == '__main__':
    main()
