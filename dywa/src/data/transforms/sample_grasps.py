#!/usr/bin/env python3

from typing import Dict, Iterable, Tuple, Optional
import torch as th
import numpy as np
import einops
from cho_util.math import transform as tx


def generate_views(
        N: int,
        phi: float = (np.sqrt(5) - 1) / 2,
        center: Tuple[float, float, float] = (0, 0, 0),
        radius: float = 1.0):
    """
    Generate spherical views at regular angular intervals.

    NOTE: Taken from GraspNetAPI.
    """
    center = np.asanyarray(center, dtype=np.float32)
    idxs = np.arange(N, dtype=np.float32)
    Z = (2 * idxs + 1) / N - 1
    X = np.sqrt(1 - Z**2) * np.cos(2 * idxs * np.pi * phi)
    Y = np.sqrt(1 - Z**2) * np.sin(2 * idxs * np.pi * phi)
    views = np.stack([X, Y, Z], axis=1)
    views = radius * np.array(views) + center
    return views


def batch_viewpoint_params_to_matrix(
        batch_towards: np.ndarray,
        batch_angle: np.ndarray) -> np.ndarray:
    """ NOTE: Taken from GraspNetAPI """
    axis_x = batch_towards
    ones = np.ones(axis_x.shape[0], dtype=axis_x.dtype)
    zeros = np.zeros(axis_x.shape[0], dtype=axis_x.dtype)
    axis_y = np.stack([-axis_x[:, 1], axis_x[:, 0], zeros], axis=-1)
    mask_y = (np.linalg.norm(axis_y, axis=-1) == 0)
    axis_y[mask_y] = np.array([0, 1, 0])
    axis_x /= np.linalg.norm(axis_x, axis=-1, keepdims=True)
    axis_y /= np.linalg.norm(axis_y, axis=-1, keepdims=True)
    axis_z = np.cross(axis_x, axis_y)
    sin = np.sin(batch_angle)
    cos = np.cos(batch_angle)
    R1 = np.stack([ones, zeros, zeros,
                   zeros, cos, -sin,
                   zeros, sin, cos],
                  axis=-1)
    R1 = R1.reshape([-1, 3, 3])
    R2 = np.stack([axis_x, axis_y, axis_z], axis=-1)
    R = np.matmul(R2, R1)
    return R.astype(np.float32)


class FilterGrasps:
    def __init__(self, max_mu: float = 0.4):
        self.max_mu = max_mu

    def __call__(self,
                 collision: np.ndarray,
                 scores: np.ndarray) -> np.ndarray:
        pos_mask = np.logical_and(
            ~collision, np.logical_and(
                scores >= 0,
                scores < self.max_mu)
        )
        indices = np.argwhere(pos_mask)
        return indices


class FormatGrasps:
    def __init__(self):
        self.views = generate_views(300)

    def __call__(self,
                 centers: np.ndarray,
                 offsets: np.ndarray,
                 indices: Optional[np.ndarray] = None):
        # By default, we query `all` indices.
        # TODO: super inefficient !
        if indices is None:
            i, j, k, l = offsets.shape[:4]
            indices = np.mgrid[:i, :j, :k, :l]  # 4 x ...
            indices = einops.rearrange(indices,
                                       'd ... -> (...) d')

        center = centers[indices[:, 0]]
        offset = offsets[
            indices[:, 0],
            indices[:, 1],
            indices[:, 2],
            indices[:, 3],
            :]

        # NOTE: I'm not really a big fan of this
        # view-vector shenanigans. Would much rather have
        # quaternions instead.
        # view_vector = self.views[indices[:, 1]]
        rotation_matrix = batch_viewpoint_params_to_matrix(
            -self.views[indices[:, 1]], offset[:, 0])
        quaternion = tx.rotation.quaternion.from_matrix(rotation_matrix)
        return np.concatenate([
            center,  # x,y,z
            quaternion,  # qx,qy,qz,qw
            offset[..., 1:]  # depth, width
            # TODO:
            # consider including metadata:
            # collision label and friction coefficient.
        ], axis=-1)


class QuaternionFromViewVector:
    def __call__(self, view_vector, roll: float):
        R = batch_viewpoint_params_to_matrix(
            -view_vector, roll
            - data['pos_grasps'][:, -3:],
            data['pos_grasps'][:, 3])
        q = tx.rotation.quaternion.from_matrix(R)
        return q


class SampleGrasps:
    """
    Sample positive and negative grasps from
    the dataset.
    """

    def __init__(self, n_pos: int, n_neg: int,
                 max_mu: float = 0.4):
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.max_mu = max_mu
        # NOTE:
        # (1) `300` is taken from GraspNetAPI convention.
        # (2) The way the view-vectors are generated replicates
        # the scheme from the graspnet1B dataset.
        self.views = generate_views(300)
        # self.format = FormatGrasps()

    def _pack_grasp_params(self,
                           inputs: Dict[str, th.Tensor],
                           indices: Iterable[int]) -> th.Tensor:
        """ Pack grasp parameters from selected indices. """
        center = inputs['grasp_center'][indices[:, 0]]
        offset = inputs['offsets'][
            indices[:, 0],
            indices[:, 1],
            indices[:, 2],
            indices[:, 3], :]
        view_vector = self.views[indices[:, 1]]
        return np.concatenate([center, offset, view_vector], axis=-1)

    def __call__(self, inputs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        collision = inputs['collision']

        # Compute positive and negative grasp indices.
        neg_mask = np.logical_or(
            collision, np.logical_or(
                inputs['scores'] < 0,
                inputs['scores'] > self.max_mu))

        outputs = dict(inputs)

        if self.n_pos > 0:
            pos = np.argwhere(~neg_mask)
            pos_idxs = pos[np.random.choice(
                len(pos),
                size=self.n_pos)]
            # Pack respective grasp parameteres.
            pos_grasps = self._pack_grasp_params(inputs, pos_idxs)
            # Dump into outputs.
            outputs['pos_grasps'] = pos_grasps

        if self.n_neg > 0:
            neg = np.argwhere(neg_mask)
            neg_idxs = neg[np.random.choice(
                len(neg),
                size=self.n_neg)]
            # Pack respective grasp parameteres.
            neg_grasps = self._pack_grasp_params(inputs, neg_idxs)
            # Dump into outputs.
            outputs['neg_grasps'] = neg_grasps
        return outputs
