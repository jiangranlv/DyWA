#!/usr/bin/env python3

from typing import (Dict, Union, Tuple, Optional, Any, Iterable)
import numpy as np
import torch as th
import open3d as o3d

from scipy.ndimage import affine_transform
from data.util import bbox_nd
from cho_util.math.rotation import matrix


def voxelize(points: np.ndarray,
             grid_shape: Optional[Tuple[int, int, int]],
             voxel_dims: Tuple[float, float, float],
             min_bound: Tuple[float, float, float],
             out: Optional[np.ndarray] = None) -> np.ndarray:
    """Voxelize point cloud based on voxel configs."""
    if out is None:
        out = np.zeros(grid_shape, dtype=np.float32)
    index = ((points - min_bound) * np.reciprocal(voxel_dims)).astype(
        np.int32)
    in_bound_mask = np.logical_and(
        np.all(index >= 0, axis=-1), np.all(index < grid_shape, axis=-1))
    index = index[in_bound_mask]
    out[index[..., 0], index[..., 1], index[..., 2]] = 1  # ==> error ?
    return out


def voxelize_mesh_o3d(verts: np.ndarray, faces: np.ndarray,
                      grid_shape: Optional[Tuple[int, int, int]],
                      voxel_dims: Union[float, Tuple[float, float, float]],
                      min_bound: Tuple[float, float, float],
                      out: Optional[np.ndarray] = None) -> np.ndarray:
    if out is None:
        out = np.zeros(grid_shape, dtype=np.float32)

    # (0) Match Open3d config.
    # Open3D requires that `voxel_dims` be uniform,
    # so we could either (1) stretch the vertices to create an equivalent effect,
    # or (2) take the maximum voxel size to keep the scaling uniform.
    # cuda_voxelizer takes option#2, so we follow this convention.
    if isinstance(voxel_dims, Iterable):
        voxel_dim = np.max(voxel_dims)
        # verts = verts * (voxel_dim / voxel_dims)
        # min_bound = min_bound * (voxel_dim / voxel_dims)
    else:
        voxel_dim = voxel_dims

    # (1) Create Open3D TriMesh.
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # (2) Project to VoxelGrid via Open3D API.
    # NOTE: in Open3D:
    # grid_shape_ = int(round((max_bound - min_bound) / voxel_dims))
    # so this _should_ yield the correct grid shape.
    max_bound = np.multiply(voxel_dim, grid_shape) + min_bound
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_dim, min_bound + 0.5 * voxel_dim, max_bound + 0.5 * voxel_dim)

    for voxel in voxel_grid.get_voxels():
        out[tuple(voxel.grid_index)] = 1
    return out


class ConfigureVoxelGrid:
    def __init__(self,
                 grid_shape: Tuple[int, int, int],
                 key_in: str = 'full'):
        self.grid_shape = grid_shape
        self.key_in = key_in

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = dict(inputs)

        ref_cloud = inputs[self.key_in]
        if isinstance(ref_cloud, th.Tensor):
            ref_cloud = ref_cloud.detach().cpu().numpy()
        scale = np.ptp(ref_cloud, axis=0)
        min_bound = np.min(ref_cloud, axis=0)
        voxel_dims = scale / self.grid_shape
        # option # 1 = stretch to non-uniform voxel size
        # pass

        # option # 2 = keep uniform voxel sizes.
        # This convention is identical to the cuda_voxelizer
        # convention.
        voxel_dim = np.max(voxel_dims)
        min_bound -= 0.5 * (np.max(scale) - scale)
        voxel_dims = np.full(3, voxel_dim)

        outputs['grid_shape'] = self.grid_shape
        outputs['voxel_dims'] = voxel_dims
        outputs['min_bound'] = min_bound
        return outputs


class Voxelize:
    """Voxelize a point cloud input.  """

    def __init__(
            self, keys: Iterable[str] = ('full', 'part'),
            suffix: str = 'voxel'):
        self.keys = keys
        self.suffix = suffix

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract metadata ...
        # TODO: metadata keys should be configurable.
        grid_shape = inputs['grid_shape']
        voxel_dims = inputs['voxel_dims']
        min_bound = inputs['min_bound']

        # TODO: Consider avoiding recreation of memory array
        # every time this function is called.
        # TODO: Consider randomizing voxel-grid parameters
        # in terms of their dimensions, min bound, etc.
        outputs = dict(inputs)
        for key in self.keys:
            cloud = inputs[key]
            if isinstance(cloud, th.Tensor):
                cloud = cloud.detach().cpu().numpy()
            outputs[F'{key}/{self.suffix}'] = voxelize(
                cloud, grid_shape,
                voxel_dims, min_bound)
        return outputs


class VoxelizeMesh:
    """Voxelize a trimesh input.  """

    def __init__(
            self,
            out_key: str,
            vert_key: Optional[str] = None,
            face_key: Optional[str] = None,
            mesh_key: Optional[str] = None,
    ):
        assert((mesh_key is not None) or
               (vert_key is not None and face_key is not None))
        self.vert_key = vert_key
        self.face_key = face_key
        self.mesh_key = mesh_key
        self.out_key = out_key

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Extract metadata ...
        # TODO: metadata keys should be configurable.
        grid_shape = inputs['grid_shape']
        voxel_dims = inputs['voxel_dims']
        min_bound = inputs['min_bound']

        # TODO: Consider avoiding recreation of memory array
        # every time this function is called.
        # TODO: Consider randomizing voxel-grid parameters
        # in terms of their dimensions, min bound, etc.
        outputs = dict(inputs)
        if self.mesh_key is not None:
            mesh = inputs[self.mesh_key]
            outputs[self.out_key] = voxelize_mesh_o3d(
                mesh.vertices, mesh.faces,
                grid_shape,
                voxel_dims,
                min_bound)
        else:
            outputs[self.out_key] = voxelize_mesh_o3d(
                inputs[self.vert_key],
                inputs[self.face_key],
                grid_shape,
                voxel_dims,
                min_bound)
        return outputs


class AugmentFlip:
    """
    Augment voxelized data with with
    90-deg rotations and flipping for each axis.
    """

    def __init__(self, keys: Optional[Iterable[str]] = (
            'full/voxel', 'part/voxel')):
        assert(len(keys) > 0)
        self.keys = keys

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = dict(inputs)

        # (1) Flip.
        for axis in range(3):
            if np.random.uniform() < 0.5:
                continue
            for key in self.keys:
                outputs[key] = np.flip(
                    outputs[key], axis=axis)

        # (2) Rotate.
        for axes in [(0, 1), (1, 2), (2, 0)]:
            if np.random.uniform() < 0.5:
                continue
            for key in self.keys:
                outputs[key] = np.flip(
                    outputs[key], axis=axis)

        # (3) Make contiguous.
        for key in self.keys:
            outputs[key] = np.ascontiguousarray(
                outputs[key])

        return outputs


class AugmentAffine:
    """
    Augment voxelized data by applying an arbitrary affine transform.
    The affine transform is applied about the center of the object.
    """

    def __init__(self, max_shift: int = 3,
                 keys: Optional[Iterable[str]] = ('full/voxel', 'part/voxel')):
        """
        Args:
            max_shift: Maximum allowed translation for each axis.
            keys: The keys to which to apply the affine augmentations.
                  The first entry is used for determining the center.
        """
        assert(len(keys) > 0)
        self.max_shift = max_shift
        self.keys = keys
        # self.rng = np.random.default_rng(0)

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Sample random affine transform.
        # NOTE: `rxn`, `txn` are applied in the
        # `output` coordinates, where the rotation
        # is applied about the center. In other words,
        # V0[x0] == V1[R @ (x1-c) + c + t].

        # FIXME: `rxn` should be generated with the
        # RNG with a controlled seed.
        rxn = matrix.random()
        ctr = np.multiply(0.5, inputs[self.keys[0]].shape)
        txn = ctr - rxn @ ctr + np.random.uniform(-self.max_shift,
                                                  self.max_shift,
                                                  size=3).astype(np.int32)
        outputs = dict(inputs)
        for key in self.keys:
            outputs[key] = affine_transform(
                inputs[key], rxn, txn, order=0,
                output_shape=inputs[key].shape)
        # TODO: additionally return the applied transform parameters.
        return outputs
