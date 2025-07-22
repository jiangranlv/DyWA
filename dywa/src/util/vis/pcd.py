#!/usr/bin/env python3

import copy

import einops
import torch as th
import numpy as np
import open3d as o3d

from util.torch_util import dcn
from util.vis.img import digitize_image

from icecream import ic


def colorize_pca(embed: th.Tensor,
                 num_color: int = 3) -> np.ndarray:
    """
    Args:
        embed: (..., D) patchwise embeddings

    Returns:
        color: (..., C) colorization based on PCA
    """

    # Colorize embeddings
    # via three-component PCA
    zz = embed.reshape(-1, embed.shape[-1])
    u, s, v = th.pca_lowrank(zz, 3)
    vec = (zz - zz.mean(dim=0, keepdim=True)) @ v  # ..., 3
    values = vec.reshape(*embed.shape[:-1], -1)

    # Convert to colorized range.
    values = dcn(values)
    colors = np.stack([digitize_image(values[..., i], num_bins=255) / 255.0
                       for i in range(values.shape[-1])], axis=-1)
    return colors


def colorize_embedding(patch: th.Tensor,
                       embed: th.Tensor,
                       num_color: int = 3):
    # patch given as (num_patch, patch_size, 3)
    # `embed` given as (num_patch, embed_size)
    colors = colorize_pca(embed)
    colors = einops.repeat(colors,
                           '... s d -> ... (s p) d',
                           p=patch.shape[-2])
    coords = patch.reshape(*patch.shape[:-3], -1, 3)
    return (coords, colors)


def draw_colored_clouds(coords, colors,
                        as_sphere: bool = False,
                        draw: bool = True, **kwds):
    aux = kwds.pop('aux', {})
    offset_scale: float = kwds.pop('offset_scale', 0.5)

    s = None
    if as_sphere:
        s = o3d.geometry.TriangleMesh.create_sphere(
            radius=kwds.pop('radius', 0.01),
            resolution=kwds.pop('resolution', 4))

    geoms = []
    aux['offsets'] = []

    # We assume batched input
    d: int = int(max(1, np.ceil(np.sqrt(len(coords)))))
    for i in range(len(coords)):
        offset = (offset_scale * (i // d), offset_scale * (i % d), 0)
        # offset = (0, 0, 0)
        if as_sphere:
            # Show each point as sphere.
            balls = [copy.deepcopy(s).translate(x + offset)
                     for x in dcn(coords[i])]
            [b.paint_uniform_color(c)
                for (b, c) in zip(balls, dcn(colors[i]))]
            geoms.extend(balls)
            aux['offsets'].extend([offset for _ in balls])
        else:
            # Show as point cloud.
            # (in this mode, each point looks like a box)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                dcn(coords[i]) + offset
            )
            pcd.colors = o3d.utility.Vector3dVector(dcn(colors[i]))
            geoms.append(pcd)
            aux['offsets'].append(offset)

    if draw:
        o3d.visualization.draw(
            geoms,
            show_skybox=False,
            point_size=8,
            bg_color=(0, 0, 0, 1.0)
        )
    return geoms


def main():
    coords = np.random.normal(size=(1, 64, 3))
    colors = np.random.uniform(size=(1, 64, 3))
    draw_colored_clouds(coords, colors)


if __name__ == '__main__':
    main()
