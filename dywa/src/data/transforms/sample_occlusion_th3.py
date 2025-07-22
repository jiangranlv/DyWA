#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union
import torch as th
import numpy as np
import open3d as o3d

from pytorch3d.io.obj_io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    look_at_view_transform
)


class SampleOcclusionFromRender:
    """
    Sample occlusion from unprojecting a depth-rendered mesh.
    Uses pytorch3d as the rendering backend.
    """
    DEFAULT_KEY_MAP = {k: k for k in (
        ('mesh', 'verts', 'faces', 'viewpoint', 'visible_points'))}

    @dataclass(frozen=True)
    class Config:
        # Camera projection parameters.
        aspect: float = 1.0
        fov: float = 60.0
        znear: float = 1.0
        zfar: float = 5.0

        # Output image shape parameters.
        shape: Tuple[int, int] = (256, 256)

        # NOTE: currently unused.
        # Optionally samples a subset of the visible points
        # to a fixed number.
        num_pts: Optional[int] = None
        # NOTE: @see th3.RasterizationSettings.
        # Generally best to keep as default(0.0)
        blur_radius: float = 0.0
        # NOTE: @see th3.RasterizationSettings.
        # Generally best to keep as default(1)
        faces_per_pixel: int = 1

    def __init__(self,
                 cfg: Config,
                 key_map: Optional[Dict[str, str]] = None,
                 device: Union[th.device, str, None] = 'cpu'):
        self.cfg = cfg
        self.device = device

        self.key_map = dict(SampleOcclusionFromRender.DEFAULT_KEY_MAP)
        if key_map is not None:
            self.key_map.update(key_map)
        self.key_mesh = self.key_map['mesh']
        self.key_verts = self.key_map['verts']
        self.key_faces = self.key_map['faces']
        self.key_viewpoint = self.key_map['viewpoint']
        self.key_visible_points = self.key_map['visible_points']

        raster_settings = RasterizationSettings(
            image_size=cfg.shape,
            blur_radius=cfg.blur_radius,
            faces_per_pixel=cfg.faces_per_pixel,
        )
        self.cameras = FoVPerspectiveCameras(
            znear=cfg.znear,
            zfar=cfg.zfar,
            aspect_ratio=cfg.aspect,
            fov=cfg.fov,
            degrees=True,
            device=self.device).to(device)
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=raster_settings
        ).to(self.device)

        # NDC coordinates buffer.
        # TODO: for whatever reason, it has to
        # go from 1 to -1. figure out why this is the case.
        I, J = th.meshgrid(
            th.linspace(1, -1, cfg.shape[0]),
            th.linspace(1, -1, cfg.shape[1]),
            indexing='ij')
        # TODO: figure out if (I,J) is correct or (J,I) is correct.
        self.coord = th.stack([J, I], dim=-1).to(self.device)

    def __call__(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        cfg = self.cfg

        # Construct or retrieve torch3d meshes.
        if 'mesh' in inputs:
            meshes = inputs['mesh']
        else:
            verts = [
                th.as_tensor(x, device=self.device)
                for x in inputs[self.key_verts]]
            faces = [
                th.as_tensor(x, device=self.device)
                for x in inputs[self.key_faces]]
            meshes = Meshes(verts=verts, faces=faces)

        # Construct the camera transform.
        batch_size: int = len(meshes)
        viewpoint = inputs.get(self.key_viewpoint, None)

        boxes = meshes.get_bounding_boxes()
        center = 0.5 * (boxes[..., 0] + boxes[..., 1])

        # NOTE: automatically figure out
        # a decent viewpoint location based on the camera frustum.
        # We try to position the camera somewhere `reasonable`,
        # so that we get many useful points.
        if viewpoint is None:
            radius = th.linalg.norm(
                boxes[..., 1] - boxes[..., 0],
                dim=-1) * 0.5
            distance = radius / np.tan(np.deg2rad(0.5 * cfg.fov))
            # WARN: randomness is NOT controlled when
            # sampling viewpoints.
            # WARN: th.randn(device='cpu') results in
            # a different value from th.randn(device='cuda')
            # even with the same random seed!
            viewpoint = th.randn(size=(batch_size, 3),
                                 device=self.device)
            viewpoint = viewpoint * (cfg.znear + radius + distance)[..., None] / th.linalg.norm(
                viewpoint, dim=-1, keepdim=True)
            viewpoint = viewpoint + center

        # TODO: what should `viewpoint` parameter output,
        # when `at` != (0,0,0)? In this case, just returning `viewpoint`
        # will not be a sufficient specification for the camera.
        R, T = look_at_view_transform(
            eye=viewpoint.to(self.device),
            at=center.to(self.device),
            device=self.device)

        # Rasterize the mesh.
        depth = self.rasterizer(meshes,
                                R=R, T=T).zbuf  # BxHxW I guess?
        points = th.cat((self.coord[None].expand(batch_size, -1, -1, -1),
                         depth), dim=-1)

        # Unproject the points on the depth-image
        # back to point cloud in world coordinates.
        shape = points.shape
        points = self.cameras.unproject_points(points.view(
            batch_size, -1, 3), R=R, T=T,
            world_coordinates=True)
        points = points.view(shape)
        # TODO: consider optionally
        # converting back to a numpy representation.
        # points = points.numpy()

        # Select only the visible points.
        # NOTE: we apply mask at each batch index, and return
        # as a list, since the number of visible points vary per each mesh.
        mask = (depth > -1)
        points = [p[m] for (p, m) in zip(points, mask.squeeze(-1))]
        outputs = dict(inputs)
        outputs[self.key_visible_points] = points
        return outputs


def main():
    device: str = 'cpu'
    transform = SampleOcclusionFromRender(
        SampleOcclusionFromRender.Config(),
        device=device)

    objs = [
        '/opt/datasets/GSO/ScannedObjects/30_CONSTRUCTION_SET/meshes/model.obj',
        '/opt/datasets/GSO/ScannedObjects/Weisshai_Great_White_Shark/meshes/model.obj'
    ]
    index: int = 0

    meshes = load_objs_as_meshes(objs, device=device)
    outputs = transform({'mesh': meshes})
    # print(outputs['part'].shape)

    full_mesh = o3d.geometry.TriangleMesh()
    # a = meshes.verts_packed()[0].detach().cpu().numpy()
    # print(a.dtype)
    # print(a.shape)
    full_mesh.vertices = o3d.utility.Vector3dVector(
        meshes.verts_list()[index].detach().cpu().numpy())
    full_mesh.triangles = o3d.utility.Vector3iVector(
        meshes.faces_list()[index].detach().cpu().numpy())

    part_cloud = o3d.geometry.PointCloud()
    part_cloud.points = o3d.utility.Vector3dVector(
        outputs['part'][index].reshape(-1, 3).detach().cpu().numpy())
    print(full_mesh.get_axis_aligned_bounding_box())
    print(part_cloud.get_axis_aligned_bounding_box())
    # o3d.visualization.draw([full_mesh, part_cloud])
    # o3d.visualization.draw([full_mesh])


if __name__ == '__main__':
    from util.profile_app import Profiler
    with Profiler(Profiler.Config(
            cprofile=True)).profile():
        main()
