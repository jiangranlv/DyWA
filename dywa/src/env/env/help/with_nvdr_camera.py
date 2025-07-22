#!/usr/bin/env python3

from typing import Tuple, Optional, Iterable, Dict, List, Union
from dataclasses import dataclass
from util.config import ConfigBase

import os
import pkg_resources
from pathlib import Path
from yourdfpy import URDF
from yourdfpy.urdf import apply_visual_color
import trimesh
import numpy as np
import torch as th
import einops

from pytorch3d.renderer import look_at_view_transform
from env.env.help.render_nvdr import NvdrRenderer
from util.torch_util import dcn
from data.transforms.io_xfm import scene_to_mesh

from icecream import ic
# Need:
# > all rigid body mesh vertices and faces
# > selector indices for which object(s) are active for each scene
# > camera pose and NDC transform; whether it should be inverted

ASSET_ROOT = pkg_resources.resource_filename('data', 'assets')
OBJECT_URDF = '/opt/datasets/acronym/urdf2/Speaker_64058330533509d1d747b49524a1246e_0.003949258269301651.urdf'
# OBJECT_URDF = '/opt/datasets/acronym/urdf2/Desktop_fd7b14f3e5c58b2e84da0a0f266ade28_0.004044317286833961.urdf'
ROBOT_URDF = F'{ASSET_ROOT}/ur5-fe/robot.urdf'

UR5_LINK_NAMES = [
    'base_link',
    'base',
    'shoulder_link',
    'upper_arm_link',
    'forearm_link',
    'wrist_1_link',
    'wrist_2_link',
    'wrist_3_link',
    'ee_link',
    'tool0',
    'panda_hand',
    'panda_leftfinger',
    'panda_rightfinger',
    'tool_tip']


@dataclass(frozen=True)
class BoxArg:
    extents: Tuple[float, float, float]
    color: Optional[Tuple[float, float, float]] = None

    def __eq__(self, other):
        EPS: float = 1e-6
        if not isinstance(other, BoxArg):
            return False
        # FIXME: ad-hoc floating-point precision
        # approximate equality check
        for i in range(3):
            if abs(self.extents[i] - other.extents[i]) >= EPS:
                return False
        return True


def pack_tensors(x, dtype, device):
    elem_counts = [0 if xx is None else len(xx) for xx in x]
    x = np.concatenate(x, axis=0)
    x = th.as_tensor(x, dtype=dtype, device=device)
    ends = np.cumsum(elem_counts)
    begs = ends - elem_counts
    ranges = np.stack([begs, ends], axis=-1).astype(np.int32)
    return (x, ranges)


def ndc_projection_matrix(fov: th.Tensor,
                          z_near: float, z_far: float,
                          out: th.Tensor = None,
                          aspect: float = 1.0) -> th.Tensor:
    """Compute NDC projection matrix in the convention of nvdiffrast.

    Args:
        fov: Field of view (radians)
        z_near: Near-plane of camera frustum.
        z_far:  Far-plane of camera frustum.
        out: Optional output tensor, if buffer already allocated.
        aspect : width/height (e.g. 16:9 = 1.777...)

    Returns:
        NDC projection matrix; fov.shape + (4, 4)
    """
    # Compute pixel-space focal length.
    x = th.reciprocal(th.tan(0.5 * fov))

    # Allocate projection matrix.
    P = th.zeros(fov.shape + (4, 4),
                 out=out, dtype=fov.dtype, device=fov.device)

    # Populate projection matrix.
    P[..., 0, 0] = x
    # P[..., 1, 1] = -z_near / x
    # P[..., 1, 1] = -x  # NOTE: minus to take care of y-axis flip
    P[..., 1, 1] = -x * aspect
    P[..., 2, 2] = +(z_far + z_near) / (z_far - z_near)  # why not minus?
    P[..., 2, 3] = -(2 * z_far * z_near) / (z_far - z_near)
    P[..., 3, 2] = +1.0  # why not minus?

    return P


def camera_matrices(cfg: 'WithNvdrCamera.Config',
                    num_env: int,
                    device: str,
                    at: Optional[th.Tensor] = None,
                    fov: Optional[th.Tensor] = None
                    ):
    if at is None:
        at = th.as_tensor(cfg.at, dtype=th.float,
                          device=device)
        at = einops.repeat(at, '... -> n ...',
                           n=num_env)

    if fov is None:
        fov = th.as_tensor(cfg.fov,
                           device=device, dtype=th.float)
        fov = einops.repeat(fov, '... -> n ...',
                            n=num_env)

    # [7] Configure camera extrinsics.
    # T_cam_1 = np.eye(4)

    # FIXME: maybe look_at isn't _exactly_ what we want
    R_T, X = look_at_view_transform(
        eye=th.as_tensor(cfg.eye, device=device)[None],
        at=at,
        up=(cfg.up,),
        device=device)
    T_cam = th.zeros(*R_T.shape[:-2], 4, 4,
                     dtype=R_T.dtype,
                     device=R_T.device)
    T_cam[..., :3, :3] = R_T.swapaxes(-1, -2)
    T_cam[..., :3, 3] = X
    T_cam[..., 3, 3] = 1
    # T_cam_1[:3, :3] = dcn(R_T.reshape(3, 3).T)
    # T_cam_1[:3, 3] = dcn(T.reshape(3,))
    # T_cam = einops.repeat(T_cam_1, '... -> n ...', n=num_env)
    # T_cam = th.as_tensor(T_cam, device=device, dtype=th.float32)

    # [8] Configure camera intrinsics (NDC projection)
    T_ndc = ndc_projection_matrix(fov,
                                  cfg.z_near, cfg.z_far).contiguous()
    # T_ndc = einops.repeat(T_ndc, '... -> n ...', n=num_env).contiguous()
    # true "camera transform" = NDC @

    return (T_cam, T_ndc)


def scene_to_mesh_with_visual_hack(m: trimesh.Scene) -> trimesh.Trimesh:
    vs = []
    fs = []
    ns = []
    cs = []
    nv: int = 0
    for n in m.graph.nodes_geometry:
        # Lookup geometry and apply transforms.
        xfm, name = m.graph[n]
        part = m.geometry[name]
        part.apply_transform(xfm)
        # Concatenate.
        vs.append(part.vertices)
        fs.append(part.faces + nv)
        nv += len(part.vertices)
        ns.append(part.vertex_normals)

        # Try to figure out what the color should be.
        if isinstance(part.visual, trimesh.visual.TextureVisuals):
            if part.visual.uv is not None:
                # create ColorVisuals from result
                pv = part.visual
                colors = pv.material.to_color(pv.uv)
                vis = trimesh.visual.color.ColorVisuals(
                    mesh=pv.mesh,
                    vertex_colors=colors)
                cs.append(
                    vis.vertex_colors[..., :3]
                )
            else:
                if isinstance(part.visual.material,
                              trimesh.visual.material.SimpleMaterial):
                    cs.append(
                        part.visual.material.diffuse[:3] * np.ones_like(part.vertices)
                    )
                elif isinstance(part.visual.material,
                                trimesh.visual.material.PBRMaterial):
                    material = part.visual.material.to_simple()
                    cs.append(
                        material.diffuse[:3] * np.ones_like(part.vertices)
                    )
                else:
                    raise ValueError(
                        F'Unknown material type = {part.visual.material}')
        elif isinstance(part.visual, trimesh.visual.ColorVisuals):
            cs.append(part.visual.vertex_colors[..., :3])
    vs = np.concatenate(vs, axis=0)
    fs = np.concatenate(fs, axis=0)
    ns = np.concatenate(ns, axis=0)
    cs = np.concatenate(cs, axis=0)
    m = trimesh.Trimesh(vertices=vs,
                        faces=fs,
                        vertex_normals=ns,
                        vertex_colors=cs)
    return m


def get_link_mesh(urdf, link, use_col: bool = True,
                  alt_path: Optional[Union[List[str], str]] = None,
                  cat: bool = True, scale: Optional[Tuple[float, ...]] = None):
    if isinstance(alt_path, str) or isinstance(alt_path, os.PathLike):
        alt_path = [alt_path]
    if alt_path is None:
        alt_path = []

    geometries = link.collisions if use_col else link.visuals
    visuals = link.visuals

    if len(geometries) == 0:
        return None
    meshes = []
    for g in geometries:
        if g.geometry.mesh is None:
            continue

        f = g.geometry.mesh.filename
        if 'package://' in f:
            f = f.replace('package://', '')

        for root in [''] + alt_path:
            f2 = F'{root}/{f}'
            # print(F'try =  {f2}')
            if Path(f2).is_file():
                f = f2
                break
        else:
            raise FileNotFoundError(
                F'{g.geometry.mesh.filename} not found from paths = {alt_path}')

        m = trimesh.load(f, skip_materials=False)

        # >> TRY TO LOAD A "DECENT" visual mesh.
        if cat and isinstance(m, trimesh.Scene):
            if not use_col:
                m = scene_to_mesh_with_visual_hack(m)
            else:
                m = scene_to_mesh(m)

        # NOTE: also fill in potentially missing colors
        # from URDF material specifications, if available.
        if not use_col:
            apply_visual_color(m, g, urdf._material_map)
        else:
            if len(geometries) == 1 and len(visuals) == 1:
                apply_visual_color(m, visuals[0], urdf._material_map)

        pose = g.origin

        if pose is None:
            pose = np.eye(4)
        # pose = np.linalg.inv(pose)

        # if (not use_col) and (visuals[0].origin is not None):
        #     pose = np.linalg.inv(visuals[0].origin)
        #     print(visuals[0].origin)
        #     print('pose', pose)

        if scale is None:
             scale = g.geometry.mesh.scale
        if scale is not None:
            # print(F'apply scale = {g.geometry.mesh.scale}')
            S = np.eye(4)
            S[:3, :3] = np.diag(scale)
            pose = pose @ S
        m.apply_transform(pose)
        meshes.append(m)
    if len(meshes) == 0:
        return None
    if cat:
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = [meshes]
    return mesh


def load_objects(
        # ==robot==
        # load_robot: bool,
        # load_cube: bool,
        # ==scene(objects)
        object_urdfs: Optional[Iterable[str]] = None,
        # ==table is ignored
        # cube_dims: Optional[Tuple[float, float, float]] = None,
        # load_table: bool = True,
        use_col: bool = False,
        urdf_link_map: Optional[Dict[str, List[str]]] = None,
        load_ground: bool = True,
        blocklist: Optional[Dict[str, Iterable[str]]] = None,
        cat: bool = True,
        scale: Optional[Dict[str, List[float]]] = None 
):
    if blocklist is None:
        blocklist = {}

    if object_urdfs is None:
        object_urdfs = [OBJECT_URDF]

    vertices = []
    faces = []
    normals = []
    colors = []

    for object_urdf in object_urdfs:
        skip_link = frozenset(blocklist.get(object_urdf, []))

        if isinstance(object_urdf, BoxArg):
            # Custom box
            obj = trimesh.creation.box(
                extents=dcn(object_urdf.extents),
            )
            obj = obj.subdivide()
            vertices.append(obj.vertices)
            faces.append(obj.faces)
            normals.append(obj.vertex_normals)
            if object_urdf.color is not None:
                colors.append(
                    obj.visual.vertex_colors[..., :3] / 255.0
                )
            else:
                colors.append(np.ones_like(obj.vertices))
        else:
            # URDF Object(s)
            obj = URDF.load(
                object_urdf,
                load_meshes=False,
                load_collision_meshes=False)

            if len(obj.link_map) > 1:
                # multibody
                link_names = urdf_link_map[object_urdf]
                offset: int = len(vertices)
                vertices += [np.empty((0, 3), dtype=np.float32)
                             for _ in link_names]
                faces += [np.empty((0, 3), dtype=np.int32)
                          for _ in link_names]
                normals += [np.empty((0, 3), dtype=np.float32)
                            for _ in link_names]
                colors += [np.empty((0, 3), dtype=np.float32)
                           for _ in link_names]
                for l, v in obj.link_map.items():
                    if (l in skip_link):
                        print(F'skipping link={l} due to blocklist.')
                        continue
                    s = None
                    if scale is not None:
                        s = scale.get(l, None)
                    m = get_link_mesh(
                        obj, v, use_col=use_col, alt_path=[
                            Path(object_urdf).parent,
                            Path(object_urdf).parent.parent,
                            Path(object_urdf).parent.parent.parent,
                        ], scale=s)

                    # skip.
                    if (m is None):
                        print(F'link mesh for {l} was None.')
                        continue

                    # prefix = 'col' if use_col else 'vis'
                    # m.export(F'/tmp/docker/{prefix}-{l}.obj')

                    # populate.
                    i = link_names.index(l) + offset
                    vertices[i] = m.vertices
                    faces[i] = m.faces
                    normals[i] = m.vertex_normals
                    try:
                        c = m.visual.to_color().vertex_colors[..., :3]
                    except AttributeError:
                        c = m.visual.vertex_colors[..., :3]
                    colors[i] = (c / 255.0).astype(np.float32)

            else:
                # single rigid body
                for l, v in obj.link_map.items():
                    # FIXME: be careful about `use_col` here.
                    m = get_link_mesh(obj, v,
                                      use_col=use_col,
                                      alt_path=Path(object_urdf).parent,
                                      cat=cat)
                    if m is None:
                        continue

                    vertices.append(m.vertices)
                    faces.append(m.faces)
                    normals.append(m.vertex_normals)
                    try:
                        c = m.visual.to_color().vertex_colors[..., :3]
                    except AttributeError:
                        c = m.visual.vertex_colors[..., :3]
                    colors.append(
                        (c / 255.0).astype(np.float32)
                    )

    if load_ground:
        ground = trimesh.creation.box(
            extents=(20.0, 20.0, 0.001)
        )
        vertices.append(ground.vertices)
        faces.append(ground.faces)
        normals.append(ground.vertex_normals)
        colors.append(np.ones_like(ground.vertices))

    out = (vertices, faces, normals, colors)
    return out


class   WithNvdrCamera:

    @dataclass
    class Config(ConfigBase):
        # camera position
        eye: Tuple[float, float, float] = (0.5, 0.0, 1.5)
        # camera target (for lookat)
        at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
        # camera field of view
        fov: float = float(np.deg2rad(90.0))
        aspect: float = 1.0
        # camera near plane
        z_near: float = 0.001
        # camera far plane
        z_far: float = 10.0

        ctx_type: str = 'cuda'
        # ctx_type: str = 'gl'
        use_color: bool = False
        use_depth: bool = True
        use_flow: bool = False
        use_label: bool = False
        fast_depth: bool = True

        # use_robot: bool = False
        # use_cube: bool = False
        # FIXME: avoid hardcoding `cube_dims` ??
        # cube_dims: Optional[Tuple[float, float, float]] = None
        # use_table: bool = False
        use_ground: bool = False

        # Use URDF collision mesh for rendering
        # (as opposed to visual mesh)
        # > for depth, flow --> use_col is fine
        # > for RGB --> use_col is problematic
        #               due to the artifacts from convex decomposition.
        use_col: bool = True
        track_object: bool = False
        link_scale_factor: Optional[Dict[str, List[float]]] = None 

    def __init__(self,
                 cfg: Config,
                 num_env: int,
                 shape: Tuple[int, int] = (224, 224),
                 device: str = 'cuda:1',

                 sel: Optional[np.ndarray] = None,
                 object_urdfs: Optional[Iterable[str]] = None,
                 urdf_link_map: Optional[Dict[str, List[str]]] = None,
                 blocklist: Optional[Dict[str, List[str]]] = None
                 ):
        """
        Args:
            sel:          Which objects are selected per each env.
            object_urdfs: The total list of object files.
                          I think right now the indices in `sel`
                          are offset by one compared to the
                          actual "index" into the object_urdfs list
                          due to the presence of `table` which is
                          loaded first.
        """
        self.cfg = cfg
        self.num_env = num_env
        self.shape = shape
        self.device = device

        # [1] load objects.
        (v, f, n, c) = load_objects(
            # cfg.use_robot,
            # cfg.use_cube,
            object_urdfs=object_urdfs,
            # cube_dims=cfg.cube_dims,
            # load_table=cfg.use_table,
            use_col=cfg.use_col,
            urdf_link_map=urdf_link_map,
            load_ground=cfg.use_ground,
            blocklist=blocklist,
            scale=cfg.link_scale_factor
        )

        # [3] Pack tensor lists.
        # NOTE: (tensor, ranges) where tensor is th.Tensor
        # but ranges is an np.ndarray.
        v, v_ranges = pack_tensors(v, dtype=th.float32, device=device)
        f, f_ranges = pack_tensors(f, dtype=th.int32, device=device)
        n, _ = pack_tensors(n, dtype=th.float32, device=device)
        c, _ = pack_tensors(c, dtype=th.float32, device=device)

        # [6] Selector matrix determines
        # which object was loaded per each environment.
        # By default, we assume all objects are loaded for every object.
        if sel is None:
            sel = einops.repeat(np.arange(len(v_ranges)),
                                'v -> n v', n=num_env)
        else:
            pass

        # [7] Configure camera extrinsics.
        T_cam, T_ndc = camera_matrices(cfg, num_env, device)

        # [10] Finally, reset the renderer !!
        renderer = NvdrRenderer(shape, device,
                                ctx_type=cfg.ctx_type,
                                use_shader=cfg.use_color,
                                use_depth=cfg.use_depth,
                                use_color=cfg.use_color,
                                use_flow=cfg.use_flow,
                                use_label=cfg.use_label,
                                fast_depth=cfg.fast_depth)
        renderer.reset(v, f, sel,
                       T_cam, T_ndc,
                       v_ranges, f_ranges, False,
                       obj_normals=n,
                       obj_colors=c)
        self.renderer = renderer

    def __call__(self,
                 poses: th.Tensor,
                 targets: Optional[th.Tensor] = None,
                 radii: Optional[th.Tensor] = None):

        cfg = self.cfg
        if cfg.track_object:
            assert (targets is not None)
            eye = th.as_tensor(cfg.eye, device=self.device)[None]
            distance = th.linalg.norm(targets - eye, dim=-1)
            # ic(radii.shape, distance.shape)
            fov = 2.0 * th.arctan2(radii, distance).abs()
            T_cam, T_ndc = camera_matrices(cfg,
                                           self.num_env,
                                           poses.device,
                                           at=targets,
                                           fov=fov)
            # Configure camera extrinsics.
            self.renderer.reset_camera(camera_poses=T_cam,
                                       ndc_transforms=T_ndc,
                                       inv_camera=False)

        with th.no_grad():
            return self.renderer(poses)


def main():
    cam = WithNvdrCamera(WithNvdrCamera.Config(), 1)


if __name__ == '__main__':
    main()
