"""This script is the differentiable renderer for Deep3DFaceRecon_pytorch
    Attention, antialiasing step is missing in current version.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable, Optional
import logging
import numpy as np
import torch as th
import torch.nn as nn
import nvdiffrast.torch as dr
import einops

from data.transforms.sample_viewpoint import SampleCamera

USE_UNPROJECTION = False


def bboxes(verts: Iterable[th.Tensor]) -> th.Tensor:
    """ Compute bounding boxes of a list of point clouds. """
    out = th.empty(
        size=(len(verts), 2, 3),
        dtype=th.float32,
        device=verts[0].device)
    for i, v in enumerate(verts):
        out[i, 0] = v.min(dim=0).values
        out[i, 1] = v.max(dim=0).values
    return out


class MeshRenderer(nn.Module):
    """
    Mesh renderer with `nvdiffrast` backend.
    """

    @dataclass(frozen=True)
    class Config:
        batch_size: Optional[int] = None

        # Camera projection parameters.
        aspect: float = 1.0
        fov: float = np.deg2rad(15.0)
        znear: float = 0.01
        # TODO: `zfar` should technically
        # depend on the mesh scaling...
        zfar: float = 100.0
        # Output image shape parameters.
        shape: Tuple[int, int] = (128, 128)

        return_image: bool = False
        return_cloud: bool = True

        # Randomization parameters.
        min_fov: float = np.deg2rad(10.0)
        max_fov: float = np.deg2rad(80.0)

    def __init__(self, cfg: Config):
        super(MeshRenderer, self).__init__()
        self.cfg = cfg
        self.shape = cfg.shape
        self.glctx = None

        self.sample_camera = SampleCamera(
            SampleCamera.Config(
                min_fov=cfg.min_fov,
                max_fov=cfg.max_fov,
                z_near=cfg.znear,
                z_far=cfg.zfar))

    def _parse_inputs(self, inputs: Dict[str, th.Tensor]):
        # mesh? -> (verts, faces)
        if 'mesh' in inputs:
            mesh = inputs['mesh']
            verts = [m.vertices for m in mesh]
            faces = [m.faces for m in mesh]
        else:
            verts = inputs['verts']
            faces = inputs['faces']

        # Convert to List[th.Tensor].
        if isinstance(verts[0], th.Tensor):
            device = verts[0].device
            verts = list(verts)
            faces = list(faces)
        else:
            device = th.device(inputs.get('device',
                                          th.device('cuda')))
            verts = [
                th.as_tensor(
                    v,
                    dtype=th.float32,
                    device=device) for v in verts]
            faces = [
                th.as_tensor(
                    f,
                    dtype=th.int32,
                    device=device) for f in faces]
        verts0 = list(verts)
        return (verts, faces, verts0, device)

    def _to_ndc(self, batch_size: int,
                verts: List[th.Tensor],
                R, T, ndc_proj,
                device):
        """ Converts `verts` to NDC coordinates. """
        # TODO: preallocate instead?
        # verts_ndc = [None for _ in range(batch_size)]
        num_verts = sum(len(v) for v in verts)

        verts_ndc = th.empty(size=(num_verts, 4),
                             dtype=th.float32, device=device)
        verts_inv_y_list = [None for _ in verts]

        offset: int = 0
        # TODO: this loop can be vectorized
        # in case we're running in instance mode,
        # or if all vertices happen to be the same size.
        for i in range(batch_size):
            # Lookup the corresponding vertex set.
            v = verts[i]

            # Apply the transform (to camera frame)
            # NOTE: beware the convention here!
            # v' = v@R+T.
            v_cam = th.cat([
                v @ R[i] + T[i],
                th.ones([*v.shape[:-1], 1], device=device)],
                dim=-1)

            # NOTE: this step (inverting the y-axis)
            # is needed because the
            # memory order in `nvdiffrast` is bottom-up,
            # rather than the top-down scanline order!
            # NOTE: isn't this an in-place operation?
            # Because of this, it would probably make nvdiffrast
            # "not _really_ differentiable" according to pytorch.
            v_cam[..., 1] = -v_cam[..., 1]
            verts_inv_y_list[i] = v_cam

            th.matmul(v_cam, ndc_proj[i].t(),
                      out=verts_ndc[offset: offset + len(v_cam)])
            offset += len(v_cam)
        return (verts_inv_y_list, verts_ndc)

    def forward(self, inputs: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        """
        Args:
            `inputs` is a dictionary with the following fields:
            [Range Mode]
                verts     : List of vertices,  (B, N(i), 3), i=0:B
                faces     : List of triangles, (B, M(i), 3), i=0:B
                viewpoint : optional, size (B, 3), camera position.
            [Instance Mode]
                verts     : List of vertices,  (1, N, 3)
                faces     : List of triangles, (1, M, 3)
                viewpoint : optional, size (B, 3), camera position.
                num_viewpoint: optional, number of camera positions.

                Either `viewpoint` or `num_viewpoint` has to be
                supplied as input, to trigger instance mode.

            [Optional]
                device   : render device. See `_parse_inputs`.

        Returns:
            Dictionary with the following fields:
            mask               : th.tensor, size (B, 1, H, W)
            depth              : th.tensor, size (B, 1, H, W)

            visible_points : List of tensors, (B, N(i), 3), i=0:B

        Optional outputs enabled with `return_image=True`:
            mask      : mask image, (B,H,W)
            depth     : depth image, (B,H,W)
            viewpoint : camera position, (B,3)
            center    : object center, B(3)
        """
        cfg = self.cfg

        # Parse inputs.
        (verts, faces, verts0, device) = self._parse_inputs(inputs)

        # range_mode is triggered when `faces` is provided
        # as a list over batches. Otherwise, the rasterization
        # operates in instance mode.
        if len(faces) > 1:
            # multiple `meshes'
            num_viewpoints = len(faces)
        else:
            # one mesh

            # 1) directly from input.
            num_viewpoints = inputs.get('num_viewpoints', None)

            # 2) infer from `viewpoints`.
            if num_viewpoints is None:
                viewpoints = inputs.get('viewpoints', None)
                if viewpoints is not None:
                    num_viewpoints = len(viewpoints)

            # 3) default to 1.
            if num_viewpoints is None:
                num_viewpoints = 1

        instance_mode: bool = (len(faces) == 1 and num_viewpoints > 1)
        range_mode = (not instance_mode)

        # Compute metadata.
        boxes = bboxes(verts)
        if range_mode:
            batch_size: int = len(verts)
        else:
            batch_size: int = num_viewpoints

        radius = 0.5 * th.linalg.norm(
            boxes[..., 1, :] - boxes[..., 0, :],
            dim=-1)
        center = 0.5 * (boxes[..., 0, :] + boxes[..., 1, :])

        # Construct the camera transforms.
        R, T, ndc_proj, viewpoint, fov = self.sample_camera(radius, center,
                                                            num_viewpoints)
        if range_mode:
            out = self._to_ndc(batch_size, verts, R, T, ndc_proj,
                               device)
        else:
            b_verts = einops.repeat(verts[0], '... d -> b ... d',
                                    b=batch_size)
            out = self._to_ndc(batch_size, b_verts, R, T, ndc_proj,
                               device)
        verts_inv_y_list, verts_ndc = out

        # (4) If required, initialize OpenGL context.
        # By default, the context is initialized in the same device
        # as the inputs (verts/faces).
        if self.glctx is None:
            self.glctx = dr.RasterizeGLContext(device=device)
            logging.debug(
                "create glctx on device cuda:%d" % device.index)

        ranges = None
        if range_mode:
            num_faces = th.tensor([f.shape[0]
                                   for f in faces],
                                  device='cpu').unsqueeze(1)
            off_faces = th.cumsum(num_faces, dim=0) - num_faces
            # NOTE: `ranges` has to be on the
            # cpu, for use in in rasterize().
            ranges = th.cat([off_faces, num_faces], axis=1).to(
                dtype=th.int32, device='cpu')
            if 'faces_packed' in inputs:
                faces = inputs['faces_packed']
            else:
                # Apply vertex offsets to each face index.
                num_verts = [len(v) for v in verts]
                off_verts = np.cumsum(num_verts) - num_verts
                faces = th.cat([faces[i] + off_verts[i]
                                for i in range(len(faces))], dim=0)
        else:
            # raise ValueError('instance_mode is not currently supported.')
            verts_ndc = verts_ndc.view(batch_size, -1, 4)
            faces = faces[0]

        # For range_mode, vertex: [B*N, 4], faces: [B*M, 3].
        # for instance_mode, vertex: [B, N, 4], faces: [M, 3]
        faces = faces.to(th.int32).contiguous()
        rast_out, _ = dr.rasterize(
            self.glctx, verts_ndc.contiguous(),
            faces, resolution=self.shape,
            ranges=ranges)

        # Which pixels in the image are `occupied`?
        mask = (rast_out[..., 3] > 0)

        outputs = dict(inputs)
        outputs['viewpoint'] = viewpoint

        # Also add camera extrinsics.
        outputs['center'] = center
        outputs['rotation'] = R
        outputs['translation'] = T

        if cfg.return_image:
            # Compute depth image.
            verts_inv_y = th.cat(verts_inv_y_list, dim=0)
            depth, _ = dr.interpolate(
                verts_inv_y.reshape([-1, 4])[..., 2]
                .unsqueeze(1).contiguous(),
                rast_out, faces)
            depth = depth.squeeze(3)
            depth = mask * depth

            # Return images: depth image and visibility mask.
            outputs['mask'] = mask
            outputs['depth'] = depth

        if cfg.return_cloud:
            # -- Process mesh features, e.g. texture. --
            if USE_UNPROJECTION:
                # outputs['visible_points'] = unproject(depth)
                # points = [(p_ - T_) @ R_.T
                # for (p_, T_, R_) in zip(points, T, R)]
                raise NotImplementedError('Unprojection not implemented;')
            else:
                # Use interpolation for computing point cloud.
                if range_mode:
                    feats = th.cat(verts0, dim=0)
                    coord_image, _ = dr.interpolate(feats, rast_out, faces)
                else:
                    feats = einops.repeat(verts0[0], '... d -> b ... d',
                                          b=batch_size).contiguous()
                    coord_image, _ = dr.interpolate(feats, rast_out, faces)
                # coord_image = coord_image.permute(0, 3, 1, 2)
                points = [i[m] for (i, m) in zip(coord_image, mask)]
                outputs['visible_points'] = points

        return outputs
