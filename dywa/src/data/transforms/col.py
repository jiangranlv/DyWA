#!/usr/bin/env python3

from collections import defaultdict
from typing import Optional, List, Dict, Iterable, Mapping
from pathlib import Path
from functools import partial
from yourdfpy import URDF, filename_handler_magic
from scipy.optimize import linprog
import logging

import numpy as np
import torch
import torch as th
import einops
import pytorch_kinematics as pk
import pytorch_volumetric as pv
import coacd

import open3d as o3d

import copy
import trimesh
from tqdm.auto import tqdm
from icecream import ic
from tempfile import TemporaryDirectory

from util.torch_util import dcn
from data.transforms.io_xfm import scene_to_mesh

from env.robot.franka_kin import franka_fk

from pytorch3d._C import point_face_array_dist_forward

try:
    import pymeshlab
except ImportError:
    print('pymeshlab import failed. Mesh simplification disabled...')
    pymeshlab = None
pymeshlab = None


@th.jit.script
def point_line_distance(p: th.Tensor,
                        l: th.Tensor,
                        eps: float = 1e-6):
    """
    Arg:
        p: point cloud of shape (..., P, 3)
        l: line of shape (..., L, 2, 3)
    Return:
        out: _squared_ distance of shape (..., P, L)
    """
    v0, v1 = l.unbind(dim=-2)
    # v0 = (..., L, 3)
    v1v0 = v1 - v0
    # LAYOUT: (..., P, L, 3)
    pv0 = p[..., :, None, :] - v0[..., None, :, :]
    t_bot = th.einsum('...d, ...d -> ...', v1v0, v1v0)  # LD -> L
    t_bot = t_bot[..., None, :]
    t_top = th.einsum('...pld, ...ld -> ...pl',
                      pv0, v1v0)
    tt = (t_top / t_bot).masked_fill_(
        t_bot < eps, 0).clamp_(0, 1)

    p_proj = v0[..., None, :, :] + tt[..., None] * v1v0[..., None, :, :]
    diff = p[..., :, None, :] - p_proj
    dist = th.einsum('...d, ...d -> ...', diff, diff)
    return dist


@th.jit.script
def barycentric_coords(p: th.Tensor, tri: th.Tensor,
                       eps: float = 1e-6):
    """
    Arg:
        p: Point Cloud of shape (..., P, 3)
        t: Triangles of shape (..., T, 3, 3)

    Return:
        b: Barycentric coords of shape (..., P, T, 3)
    """
    v0, v1, v2 = th.unbind(tri, dim=-2)  # T3
    p0 = v1 - v0  # T3
    p1 = v2 - v0  # T3
    p2 = p[..., :, None, :] - v0[..., None, :, :]  # PT3

    d00 = th.einsum('...d, ...d -> ...', p0, p0)
    d01 = th.einsum('...d, ...d -> ...', p0, p1)
    d11 = th.einsum('...d, ...d -> ...', p1, p1)
    d20 = th.einsum('...ptd, ...td -> ...pt', p2, p0)
    d21 = th.einsum('...ptd, ...td -> ...pt', p2, p1)

    denom = d00 * d11 - d01 * d01 + 1e-18  # T
    denom = denom[..., :, None, :]
    w1 = (d11[..., None, :] * d20 - d01[..., None, :] * d21) / denom
    w2 = (d00[..., None, :] * d21 - d01[..., None, :] * d20) / denom
    w0 = 1.0 - w1 - w2
    return th.stack([w0, w1, w2], dim=-1)


@th.jit.script
def is_inside_triangle(p: th.Tensor, tri: th.Tensor,
                       eps: float = 1e-6):
    """
    Arg:
        x: Point Cloud of shape (..., P, 3)
        t: Triangles of shape (..., T, 3, 3)

    Return:
        in: Whether each point is inside each triangle; (..., P, T)
    """
    c = barycentric_coords(p, tri, eps)
    out = ((0 <= c) & (c <= 1)).all(dim=-1)
    return out


@th.jit.script
def is_inside_barycentric_coords_fused(
        p: th.Tensor, tri: th.Tensor):
    v0, v1, v2 = th.unbind(tri, dim=-2)  # T3
    p0 = v1 - v0  # T3
    p1 = v2 - v0  # T3
    p2 = p[..., :, None, :] - v0[..., None, :, :]  # PT3

    d00 = th.einsum('...d, ...d -> ...', p0, p0)
    d01 = th.einsum('...d, ...d -> ...', p0, p1)
    d11 = th.einsum('...d, ...d -> ...', p1, p1)
    d20 = th.einsum('...ptd, ...td -> ...pt', p2, p0)
    d21 = th.einsum('...ptd, ...td -> ...pt', p2, p1)

    denom = d00 * d11 - d01 * d01
    denom = denom[..., :, None, :]
    k1 = (d11[..., None, :] * d20 - d01[..., None, :] * d21)
    k2 = (d00[..., None, :] * d21 - d01[..., None, :] * d20)

    return (
        (0 <= k1) &
        (0 <= k2) &
        (k1 + k2 <= denom)
    )


@th.jit.script
def point_triangle_distance(x: th.Tensor,
                            tri: th.Tensor,
                            eps: float = 1e-8):
    """
    Arg:
        x: Point Cloud of shape (..., P, 3)
        tri: Triangles of shape (..., T, 3, 3)

    Return:
        d: _squared_ distance between points and triangles.
    """
    v0, v1, v2 = tri.unbind(dim=-2)  # T3
    n = th.linalg.cross(v2 - v0, v1 - v0)  # Can be cached if needed
    mag_n = th.linalg.norm(n, dim=-1, keepdim=True) + eps
    n.div_(mag_n)

    # LAYOUT: (..., P, T, 3)
    v0x = v0[..., None, :, :] - x[..., :, None, :]
    t = th.einsum('...ptd, ...td -> ...pt', v0x, n)

    inside = is_inside_barycentric_coords_fused(x, tri)
    edge_indices = th.as_tensor([0, 1, 1, 2, 2, 0],
                                dtype=th.long,
                                device=tri.device)
    lines = th.index_select(tri, -2, edge_indices)  # 6 3
    lines = lines.reshape(tri.shape[:-3] + (-1, 2, 3))
    pld = point_line_distance(x, lines, eps)
    pld = pld.reshape(pld.shape[:-1] + (-1, 3)).amin(dim=-1)
    d = th.where(inside & (mag_n.swapaxes(-1, -2) > eps),
                 th.square(t),
                 pld)
    return d


def convex_radius(m):
    return np.linalg.norm(m.vertices, axis=-1).max()


def franka_link_transforms(q: th.Tensor) -> th.Tensor:
    # Obtain transforms from forward kinematics.
    # FIXME: because of this, THIS ROUTINE
    # ONLY WORKS FOR FRANKA PANDA !!
    transforms = franka_fk(q,
                           return_intermediate=True,
                           tool_frame=False)  # ..., 4, 4
    # == Apply transform to convexes ==
    # FIXME: not necessarily the most efficient operation
    identity = transforms[0] * 0 + th.eye(
        4,
        dtype=transforms[0].dtype,
        device=transforms[0].device)
    transforms.insert(0, identity)
    transforms = th.stack(transforms, dim=-3)
    transforms = transforms.reshape(
        *q.shape[:-1],
        *transforms.shape[-3:])
    return transforms


def _pad_hulls(
        hulls: Iterable[th.Tensor],
        n: Optional[int] = None) -> th.Tensor:
    if n is None:
        n: int = max([len(h) for h in hulls])
    out = []
    for h in hulls:
        p = np.empty((n, h.shape[-1]),
                     dtype=np.float32)
        p[:len(h)] = h
        p[len(h):] = h[-1:]
        out.append(p)
    return out


def _simplify_mesh(input_parts: List[trimesh.Trimesh],
                   min_face_count: int = 4,
                   max_face_count: int = 128):
    if pymeshlab is None:
        return input_parts
    output_parts = []
    with TemporaryDirectory() as tmpdir:
        for i, p in enumerate(input_parts):
            # Export each part.
            dst_obj = F'{tmpdir}/p{i:03d}-pre.obj'
            p.export(dst_obj)

            # simplify part with meshlab
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(dst_obj)
            max_fc = int((max_face_count if isinstance(max_face_count, int)
                          else max_face_count[i]))
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=max(min_face_count, max_fc),
                preserveboundary=True,
                preservenormal=True,
                preservetopology=True)
            out_file = F'{tmpdir}/p{i:03d}-post.obj'
            ms.save_current_mesh(out_file)
            mesh = trimesh.load(out_file, file_type='obj')
            output_parts.append(mesh)
    return output_parts


def apply_coacd(mesh):
    max_concavity: float = 0.04
    max_convex_hull: int = 4
    preprocess: bool = True
    resolution: int = 2048
    mcts_max_depth: int = 3
    mcts_iterations: int = 256
    mcts_nodes: int = 64

    cmesh = coacd.Mesh()
    cmesh.vertices = mesh.vertices
    cmesh.indices = mesh.faces

    parts = coacd.run_coacd(
        cmesh,
        threshold=max_concavity,
        max_convex_hull=max_convex_hull,
        resolution=resolution,
        mcts_max_depth=mcts_max_depth,
        mcts_iterations=mcts_iterations,
        mcts_nodes=mcts_nodes,
        preprocess=preprocess
    )

    mesh_parts = [
        trimesh.Trimesh(
            np.asanyarray(p.vertices),
            np.asanyarray(p.indices).reshape((-1, 3))
        ) for p in parts
    ]
    return mesh_parts


def load_acd_obj(filename: str, **kwds):
    return trimesh.load(filename,
                        split_object=True,
                        group_material=False,
                        skip_texture=True,
                        skip_materials=True,
                        **kwds)


def apply_transform_to_hull(T, hull):
    hull = einops.rearrange(
        hull,
        '... (two d) -> ... two d',
        two=2)
    out = einops.einsum(
        T[..., :3, :3], hull,
        '... n m, h two m -> ... h two n')
    out = einops.rearrange(out, '... two n -> ... (two n)')
    out[..., 0:3].add_(T[..., None, :3, 3])
    return out


def is_in_hull(hull: th.Tensor, points: th.Tensor, tol: float = 1e-8):
    """
    hull:   array(..., T, 6), N=num_triangles
    points: array(..., P, 3), M=num_points
    """
    origin = hull[..., 0:3]
    normal = hull[..., 3:6]
    offset = th.einsum('...td, ...td -> ...t', origin, normal)
    upper = (offset + tol)[..., :, None]  # ...t -> ...t1
    delta = th.einsum('...td, ...pd -> ...tp',
                      normal, points)
    return th.all(delta <= upper, dim=-2)


def build_rigid_body_chain(urdf: URDF, col: bool = True):
    """Pre-compute the chain of links that share a common rigid body
    transform."""

    # Logic to compute that chain
    root_links = list(set([urdf.base_link] +
                          [j.child for j in urdf.actuated_joints]))
    ss = [len(urdf._successors(link)) for link in root_links]
    indices = np.argsort(ss)[::-1]
    chain = [root_links[i] for i in indices]

    # Logic to compute child links
    visited = set()
    sublinks = {}
    for c in chain[::-1]:
        s = list(urdf._successors(c))
        sublinks[c] = set(s).difference(visited)
        visited.update(s)

    # Compute (fixed) relative transforms w.r.t. root frames
    rel_xfms = {k: np.eye(4) for k in urdf.link_map.keys()}
    for root_link, child_links in sublinks.items():
        for child_link in child_links:
            T = urdf.get_transform(child_link, root_link,
                                   collision_geometry=col)
            print(F'{root_link} <- {child_link}')
            rel_xfms[child_link] = T

    roots = {}
    for root, subs in sublinks.items():
        for sub in subs:
            roots[sub] = root
    return (chain, roots, rel_xfms)


class IsInRobotPV:
    """pytorch-volumetric backend; sort of broken."""

    def __init__(self, urdf_file: str, device: str = 'cpu'):
        with open(urdf_file, 'r') as fp:
            chain = pk.build_serial_chain_from_urdf(
                fp.read(),
                # "panda_tool",
                "panda_rightfinger",
                geom_type='visual').to(
                device=device)
        path_prefix = str(Path(urdf_file).parent.resolve())
        sdf = pv.RobotSDF(
            chain,
            path_prefix=path_prefix,
            link_sdf_cls=pv.cache_link_sdf_factory(
                resolution=0.02,
                padding=1.0,
                device=device),
            geom_type='visual')
        self.__sdf = sdf

    def __call__(self, q: np.ndarray, x: np.ndarray,
                 tol: Optional[float] = 1e-3,
                 aux=None):
        q = th.as_tensor(q, device=self.__sdf.device)
        self.__sdf.set_joint_configuration(q)
        d, g = self.__sdf(x)
        if tol is None:
            return d
        else:
            return (d <= tol)


class IsInRobot:
    """Check if a given set of points `x` belong to a robot at a given
    configuration q."""

    def __init__(self,
                 urdf_file: str,
                 device: str = 'cpu',
                 assume_cvx: bool = True,
                 prune: Optional[bool] = None,
                 keep_last: Optional[int] = None,
                 num_faces: int = 64,
                 col_root: Optional[str] = None
                 ):
        self._device = device
        self._assume_cvx = assume_cvx

        self.prune = False
        if prune is None:
            prune = (device == 'cpu')
            self.prune = prune
        else:
            self.prune = prune

        (self._urdf, self._chain, self._roots,
         self._cvx_list, self._rad_list
         ) = self._load_urdf(urdf_file)
        if keep_last is None:
            keep_last = self.last_cvx_counts(
                self._chain, self._roots, col_root)
        self.keep_last = keep_last
        self._cvx_list = [[self._convert_cvx(cvx) for cvx in cvxs]
                          for cvxs in self._cvx_list]
        max_cvx: int = max([max([len(c) for c in v]) for v in self._cvx_list])
        self._cvx_list = [_pad_hulls(c, max_cvx) for c in self._cvx_list]
        self._cvx_list = [[th.as_tensor(cvx, dtype=th.float, device=device)
                           for cvx in cvxs] for cvxs in self._cvx_list]
        self._link_index = np.concatenate(
            [[i] * len(cvxs) for (i, cvxs) in enumerate(self._cvx_list)])
        cvxs = th.cat([th.stack(cvxs, dim=0)
                      for cvxs in self._cvx_list], dim=0)
        self._cvxs = cvxs
        if self.prune:
            self._rad_list = np.concatenate(self._rad_list)

    def last_cvx_counts(self, chain: Iterable[str],
                        roots: Mapping[str, str],
                        col_root: Optional[str]):
        """
        Compute the number of convex counts
        that occur at or after `col_root`.
        """
        if col_root is None:
            return None
        counts = defaultdict(int)
        for k, v in roots.items():
            counts[v] += 1

        start_index = chain.index(col_root)
        keep_last = 0
        for link_name in chain[start_index:]:
            keep_last += counts[link_name]
        return keep_last

    def _coacd(self, mesh):
        # max_concavity: float = 0.04
        max_concavity: float = 0.01
        max_convex_hull: int = 4
        preprocess: bool = True
        resolution: int = 2048
        mcts_max_depth: int = 3
        mcts_iterations: int = 256
        mcts_nodes: int = 64

        cmesh = coacd.Mesh()
        cmesh.vertices = mesh.vertices
        cmesh.indices = mesh.faces

        parts = coacd.run_coacd(
            cmesh,
            threshold=max_concavity,
            max_convex_hull=max_convex_hull,
            resolution=resolution,
            mcts_max_depth=mcts_max_depth,
            mcts_iterations=mcts_iterations,
            mcts_nodes=mcts_nodes,
            preprocess=preprocess
        )

        mesh_parts = [
            trimesh.Trimesh(
                np.asanyarray(p.vertices),
                np.asanyarray(p.indices).reshape((-1, 3))
            ) for p in parts
        ]

        return _simplify_mesh(mesh_parts)

    def _load_urdf(self, urdf_file: str):
        filename_handler = partial(
            filename_handler_magic,
            dir=Path(urdf_file).parent
        )
        urdf = URDF.load(urdf_file,
                         build_scene_graph=True,
                         build_collision_scene_graph=True,
                         load_meshes=False,
                         load_collision_meshes=False,
                         force_mesh=False,
                         force_collision_mesh=False)

        chain, roots, rel_xfms = build_rigid_body_chain(
            urdf, col=True)

        cvx_list = [[] for _ in chain]
        rad_list = [[] for _ in chain]

        for link_name, link in tqdm(urdf.link_map.items()):
            cvxs = []
            for col in link.collisions:
                g = col.geometry
                assert (g.box is None)
                assert (g.cylinder is None)
                assert (g.sphere is None)
                if g.mesh is None:
                    continue

                mesh_file: str = filename_handler(g.mesh.filename)
                cvx = load_acd_obj(mesh_file)

                if isinstance(cvx, trimesh.Scene):
                    scene = cvx
                    parts = []
                    for node_name in scene.graph.nodes_geometry:
                        (transform, geometry_name) = scene.graph[node_name]
                        mesh = scene.geometry[geometry_name]
                        # assert (trimesh.convex.is_convex(mesh))
                        if not trimesh.convex.is_convex(mesh):
                            logging.warn(
                                F'part {node_name} for {link_name} was not convex')
                            mesh = mesh.convex_hull
                        part = copy.deepcopy(mesh).apply_transform(transform)
                        parts.append(part)
                    cvxs.extend(parts)
                else:
                    if trimesh.convex.is_convex(cvx):
                        cvxs.extend([cvx])
                    else:
                        # print(F'{mesh_file} is not convex !')
                        if self._assume_cvx:
                            cvxs.extend([cvx.convex_hull])
                        else:
                            cvxs.extend(self._coacd(cvx))

            # TODO: is this necessary ?
            if False:
                cvxs = _simplify_mesh(cvxs,
                                      max_face_count=64)

            index = chain.index(roots[link_name])
            T = rel_xfms[link_name]
            if g.mesh.scale is not None:
                S = np.eye(4)
                S[:3,:3] *= np.asarray(g.mesh.scale)
                S = np.asarray(S, dtype=np.float32)
                T = T @ S
            if col.origin is not None:
                T = T @ col.origin
            [cvx.apply_transform(T) for cvx in cvxs]
            cvx_list[index].extend(cvxs)
            if self.prune:
                rad_list[index].extend([convex_radius(m) for m in cvxs])

        return (urdf, chain, roots, cvx_list, rad_list)

    def _convert_cvx(self, cvx: trimesh.Trimesh):
        origin = cvx.triangles_center
        normal = cvx.face_normals
        params = np.concatenate([origin, normal], axis=-1)
        return params

    def _get_transforms(self, q: th.Tensor,
                        repeat: bool = True):
        # Obtain transforms from forward kinematics.
        # FIXME: because of this, THIS ROUTINE
        # ONLY WORKS FOR FRANKA PANDA !!
        transforms = franka_link_transforms(q)
        if repeat:
            transforms = transforms[..., self._link_index, :, :]
        return transforms

    def __call__(self, q: th.Tensor, x: th.Tensor, tol: float = 1e-6,
                 transforms: Optional[th.Tensor] = None,
                 aux: Optional[Dict[str, th.Tensor]] = None,
                 reduce: bool = True) -> th.Tensor:
        """
        q: [(...), 7]
        x: [(...)', 3]
        """
        if transforms is None:
            # `q` is ignored in this case :)
            transforms = self._get_transforms(q)

        cvxs = einops.rearrange(self._cvxs,
                                'h t (two j) -> h t two j',
                                two=2)
        # Apply rotation.
        cvxs = einops.einsum(
            transforms[..., :3, :3],
            cvxs,
            '... h i j, h t two j -> ... h t two i')

        # Apply translation (only for `origin` element).
        cvxs[..., :, :, 0, :].add_(transforms[..., None, :3, 3])
        cvxs = einops.rearrange(cvxs, '... two j -> ... (two j)')

        # Test whether the given point is within the convex hull.
        if self.prune:
            inside = th.zeros_like(x[..., 0], dtype=bool)
            for t, r, c in zip(
                    th.unbind(transforms, dim=-3),
                    self._rad_list,
                    th.unbind(cvxs, dim=-3)):
                dd = x - t[..., :3, 3]
                ixr = (th.linalg.norm(dd, dim=-1) >= r)
                if ixr.all():
                    print('skip')
                    continue
                h = is_in_hull(c, x)
                inside |= h
        else:
            if self.keep_last is not None:
                inside = is_in_hull(
                    cvxs[..., -self.keep_last:, :, :],
                    x[..., None, :, :],
                    tol=tol)
                if reduce:
                    inside = inside.any(dim=-2)
            else:
                inside = is_in_hull(
                    cvxs, x[..., None, :, :],
                    tol=tol)
                if reduce:
                    inside = inside.any(dim=-2)
        if aux is not None:
            aux['cvxs'] = cvxs
        return inside


class IsOnRobot:
    def __init__(self,
                 urdf_file: str,
                 # device: str = 'cpu',
                 # assume_cvx: bool = True
                 ):
        (self._urdf, self._chain, self._mesh_list) = (
            self._build_geoms(urdf_file)
        )
        self._mesh_list = [
            m.as_open3d for m in self._mesh_list if (
                m is not None)]
        self._mesh_list = [o3d.t.geometry.TriangleMesh.from_legacy(m)
                           for m in self._mesh_list]

    def _build_geoms(self, urdf_file: str):
        filename_handler = partial(
            filename_handler_magic,
            dir=Path(urdf_file).parent
        )
        col: bool = True

        urdf = URDF.load(urdf_file,
                         build_scene_graph=True,
                         build_collision_scene_graph=True,
                         load_meshes=False,
                         load_collision_meshes=False,
                         force_mesh=False,
                         force_collision_mesh=False)

        chain, roots, rel_xfms = build_rigid_body_chain(urdf, col=col)

        mesh_list = [None for _ in chain]
        for link_name, link in tqdm(urdf.link_map.items()):
            meshes = []
            link_shapes = link.collisions if col else link.visuals
            for shape in link_shapes:
                g = shape.geometry
                assert (g.box is None)
                assert (g.cylinder is None)
                assert (g.sphere is None)
                if g.mesh is None:
                    continue

                mesh_file: str = filename_handler(g.mesh.filename)
                mesh = trimesh.load(
                    mesh_file,
                    ignore_broken=False,
                    force="mesh",
                    skip_materials=True,
                )
                meshes.append(mesh)
            mesh = trimesh.util.concatenate(meshes)
            if mesh == []:
                # print(link_name, meshes, mesh)
                continue

            # if link_name != 'panda_hand':
            #    continue

            # Hmm...
            # meshes = _simplify_mesh(meshes)

            # Pre-apply transforms w.r.t. root link.
            index = chain.index(roots[link_name])
            T = rel_xfms[link_name]
            # print(link_name, T, mesh)
            if g.mesh.scale is not None:
                T = T @ trimesh.transformations.scale_matrix(g.mesh.scale)
            if shape.origin is not None:
                T = T @ shape.origin
            if mesh_list[index] is not None:
                mesh_list[index] = trimesh.util.concatenate([
                    mesh_list[index], mesh.apply_transform(T)])
            else:
                mesh_list[index] = mesh.apply_transform(T)
            # mesh_list[index] = mesh
        # print(mesh_list)

        return (urdf, chain, mesh_list)

    def __call__(self,
                 q: np.ndarray,
                 x: np.ndarray,
                 tol: Optional[float] = 1e-6):
        # Apply transform.
        self._urdf.update_cfg(q)
        mesh_list = copy.deepcopy(self._mesh_list)
        transforms = [self._urdf.get_transform(link, collision_geometry=True)
                      for link in self._chain]
        for m, T in zip(mesh_list, transforms):
            if m is None:
                continue
            m.transform(T)

        # Generate accelerated spatial data structure for distance computation.
        scene = o3d.t.geometry.RaycastingScene()
        for m in mesh_list:
            if m is None:
                continue
            scene.add_triangles(m)
        # o3d.visualization.draw(mesh_list)

        # Compute distance to mesh, and prune by max distance.
        dis = np.asarray(scene.compute_distance(x))
        if tol is None:
            return dis
        return (dis <= tol)


class DistanceToRobot:
    """
    **Unsigned** distance to the robot in a given configuration to a given point.
    """

    def __init__(self,
                 urdf_file: str,
                 device: str = 'cpu'):
        (self._urdf, self._chain, self._mesh_list) = (
            self._build_geoms(urdf_file)
        )

        # verts = [m.vertices for m in self._mesh_list]
        # faces = [m.faces    for m in self._mesh_list]
        self._mesh_list = _simplify_mesh(self._mesh_list, max_face_count=256)
        triangles = [m.triangles for m in self._mesh_list]
        max_tris = max([len(t) for t in triangles])
        triangles = [
            np.pad(
                t, ((0, max_tris - len(t)),
                    (0, 0),
                    (0, 0)),
                mode='edge') for t in triangles]
        triangles = np.stack(triangles, axis=0)
        self.triangles = th.as_tensor(triangles,
                                      dtype=th.float,
                                      device=device)

    def _build_geoms(self, urdf_file: str):
        filename_handler = partial(
            filename_handler_magic,
            dir=Path(urdf_file).parent
        )
        col: bool = True

        urdf = URDF.load(urdf_file,
                         build_scene_graph=True,
                         build_collision_scene_graph=True,
                         load_meshes=False,
                         load_collision_meshes=False,
                         force_mesh=False,
                         force_collision_mesh=False)

        chain, roots, rel_xfms = build_rigid_body_chain(urdf, col=col)

        mesh_list = [None for _ in chain]
        for link_name, link in tqdm(urdf.link_map.items()):
            meshes = []
            link_shapes = link.collisions if col else link.visuals
            for shape in link_shapes:
                g = shape.geometry
                assert (g.box is None)
                assert (g.cylinder is None)
                assert (g.sphere is None)
                if g.mesh is None:
                    continue

                mesh_file: str = filename_handler(g.mesh.filename)
                mesh = trimesh.load(
                    mesh_file,
                    ignore_broken=False,
                    force="mesh",
                    skip_materials=True,
                )
                meshes.append(mesh)
            mesh = trimesh.util.concatenate(meshes)
            if mesh == []:
                continue

            # Hmm...
            # meshes = _simplify_mesh(meshes, max_face_count=256)

            # Pre-apply transforms w.r.t. root link.
            index = chain.index(roots[link_name])
            T = rel_xfms[link_name]
            if g.mesh.scale is not None:
                T = T @ trimesh.transformations.scale_matrix(g.mesh.scale)
            if shape.origin is not None:
                T = T @ shape.origin
            if mesh_list[index] is not None:
                mesh_list[index] = trimesh.util.concatenate([
                    mesh_list[index], mesh.apply_transform(T)])
            else:
                mesh_list[index] = mesh.apply_transform(T)

        mesh_list = _simplify_mesh(mesh_list,
                                   max_face_count=256)

        return (urdf, chain, mesh_list)

    def __call__(self,
                 q: th.Tensor,
                 x: th.Tensor,
                 sqrt: bool = True,
                 aux=None):

        # Apply joint transforms to triangles.
        transforms = franka_link_transforms(q)  # (..., H, 4, 4)

        # h = num hulls; t = num triangles; q = 3
        triangles = th.einsum('...hij, htqj -> ...htqi',
                              transforms[..., :3, :3],
                              self.triangles).add_(
            transforms[..., None, None, :3, 3])
        triangles = einops.rearrange(triangles,
                                     '... h t q i -> ... (h t) q i')

        dis = point_triangle_distance(x, triangles)
        dis = dis.amin(dim=-1)
        if sqrt:
            dis.sqrt_()
        return dis


class SignedDistanceToRobot:
    def __init__(self, urdf_file: str, device: str = 'cpu'):
        self.occ = IsInRobot(urdf_file,
                             device=device,
                             prune=False)
        self.udf = DistanceToRobot(urdf_file, device=device)

    def __call__(self, q: th.Tensor, x: th.Tensor):
        o = self.occ(q, x, tol=0.0)  # occupancy
        d = self.udf(q, x, sqrt=True)  # distance
        s = (2 * o.float() - 1)  # sign
        return s * d


def vertices_from_half_spaces(eqn):
    from scipy.spatial import HalfspaceIntersection
    try:
        hs = HalfspaceIntersection(eqn, eqn[..., 0:3].mean(axis=0))
    except BaseException:
        try:
            ctr = chebyshev_center(eqn)
            hs = HalfspaceIntersection(eqn, ctr)
        except BaseException:
            return None
    vs = hs.intersections
    return vs


def chebyshev_center(eqn):
    norm_vector = np.reshape(np.linalg.norm(
        eqn[:, :-1], axis=1), (eqn.shape[0], 1))
    c = np.zeros((eqn.shape[1],))
    c[-1] = -1
    A = np.hstack((eqn[:, :-1], norm_vector))
    b = - eqn[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    return res.x[:-1]


def test_chull():
    from scipy.spatial import HalfspaceIntersection
    cone = trimesh.creation.cone(0.3, 0.5, 8)
    origin = cone.triangles_center
    normal = cone.face_normals

    eqn = np.concatenate(
        [normal, -np.einsum('...i,...i->...', origin, normal)[..., None]],
        axis=-1)
    # hs = HalfspaceIntersection(eqn, origin.mean(axis=0))
    hs = HalfspaceIntersection(eqn, chebyshev_center(eqn))
    vs = hs.intersections
    trimesh.Scene(
        [cone, trimesh.PointCloud(vs),
         trimesh.PointCloud(
             chebyshev_center(eqn)[None],
             colors=(255, 0, 0))]).show()


def test_ptd():
    B: int = 2
    P: int = 3
    T: int = 4

    tri = th.randn(B, T, 3, 3)
    # tri[...] = tri[:,:1] # degeneracy 1
    # tri.fill_(0)  # degeneracy 2
    pcd = th.randn(B, P, 3)
    dis = point_triangle_distance(pcd, tri)  # should be B,P,T
    # == 2, 5, 7
    dis2 = th.stack([point_face_array_dist_forward(pcd[i], tri[i], 1e-6)
                     for i in range(B)])
    print('dis = ours; dis2 = meta')
    ic(dis, dis2)
    delta = (dis - dis2)
    ic(delta)
    ic(delta.min(), delta.max())


def test_ptd2():
    device: str = 'cpu'
    # mesh = trimesh.creation.cone(0.4, 0.3)
    mesh = trimesh.creation.box(extents=(0.4, 0.3, 0.2))
    pcd, _ = trimesh.sample.sample_surface(mesh, 4096)
    pcd += 0.05 * np.random.normal(size=pcd.shape)
    tri = mesh.triangles
    pcd = th.as_tensor(pcd, device=device)
    tri = th.as_tensor(tri, device=device)
    ic(pcd.shape, tri.shape)
    # effectively, (1,4096,3) | (1, 12, 3, 3)
    dis = point_triangle_distance(pcd[None], tri[None])
    ic(dis.shape)
    dis = dis.amin(dim=-1)
    ic(dis.shape, pcd.shape)
    trimesh.PointCloud(pcd[dis[0] < 0.02 * 0.02]).show()


def test_dtr():
    from util.path import get_path
    urdf_path = get_path(
        'assets/franka_description/robots/franka_panda_2.urdf')
    device: str = 'cuda:0'
    batch_size: int = 3
    cloud_size: int = 5
    distance_to_robot = DistanceToRobot(urdf_path,
                                        device=device)

    q = th.empty((batch_size, 7), device=device).uniform_(-np.pi, np.pi)
    x = th.empty((batch_size, cloud_size, 3), device=device).normal_(0.0, 3.0)
    d = distance_to_robot(q, x)
    print(d)


def main():
    # == configure ==
    # urdf_path = get_path(
    #     'assets/franka_description/robots/franka_panda_2.urdf')
    urdf_path = '/tmp/franka_panda_simple/robot.urdf'
    query_type: str = 'random'
    num_env: int = 1  # only used for random queries
    cloud_size: int = 4096  # only used for random queries
    device: str = 'cuda:1'
    tol: float = 0.02

    is_in_robot = None
    is_on_robot = None
    dist_to_robot = None

    # is_in_robot = IsInRobotPV(urdf_path, device=device)
    is_in_robot = IsInRobot(urdf_path,
                            device=device,
                            prune=False,
                            keep_last=None,
                            col_root='panda_link6')
    # dist_to_robot = DistanceToRobot(urdf_path, device=device)
    # is_on_robot = IsOnRobot(urdf_pat

    # == sampling query points ==
    if query_type == 'random':
        q = th.empty((num_env, 7), device=device).uniform_(-np.pi, np.pi)
        x = th.cartesian_prod(
            th.linspace(-0.9, 0.9, 32),
            th.linspace(-0.9, 0.9, 32),
            th.linspace(-0.4, 1.2, 32)).to(device)
        if isinstance(is_in_robot, IsInRobot):
            x = einops.repeat(x, '... -> n ...',
                              n=num_env).contiguous()
        if isinstance(dist_to_robot, DistanceToRobot):
            x = einops.repeat(x, '... -> n ...',
                              n=num_env).contiguous()
    elif query_type == 'surface':
        q = np.random.uniform(-np.pi, np.pi, size=(7,))
        urdf = URDF.load(urdf_path,
                         build_collision_scene_graph=True,
                         load_collision_meshes=True)
        urdf.update_cfg(q)
        x, *_ = trimesh.sample.sample_surface(
            scene_to_mesh(urdf.collision_scene),
            count=cloud_size)
        x = np.random.normal(loc=x, scale=0.01)

        if is_in_robot is not None:
            q = th.as_tensor(q, device=device,
                             dtype=th.float)
            x = th.as_tensor(x, device=device,
                             dtype=th.float)

        # Add batch dimension, for convenience
        q = q[None]
        x = x[None]
    else:
        raise ValueError(F'Unknown query_type= {query_type}')

    aux = {}
    if dist_to_robot is not None:
        d = dist_to_robot(q, x,
                          sqrt=False, aux=aux)
        # ic(d.min(), d.max())
        o = (d <= tol * tol)
        ic(q.shape)
        ic(x.shape)
        ic(o.shape)
        ic(o.float().mean())
    elif is_in_robot is not None:
        # o = (is_in_robot(q, x, tol=tol, aux=aux) < 0)
        # o = o[th.arange(o.shape[0]), th.arange(o.shape[1])]
        o = is_in_robot(q, x)
        ic(q.shape)
        ic(x.shape)
        ic(o.shape)
    elif is_on_robot is not None:
        o = is_on_robot(dcn(q)[0], dcn(x)[0].astype(np.float32), tol=tol)
        o = o[None]

    s = []

    if 'cvxs' in aux:
        for cvx in dcn(aux['cvxs']):
            eqn = np.concatenate(
                [cvx[..., 3:6],
                 -np.einsum('...i,...i->...', cvx[..., 0:3], cvx[..., 3:6])[..., None]],
                axis=-1)
            vs = vertices_from_half_spaces(eqn)
            s.append(trimesh.PointCloud(vs))

    for oo, xx, qq in zip(dcn(o), dcn(x), dcn(q)):
        s = trimesh.Scene(s)
        urdf = URDF.load(urdf_path,
                         build_collision_scene_graph=True,
                         load_collision_meshes=True)
        urdf.update_cfg(qq)
        s.add_geometry(scene_to_mesh(urdf.collision_scene))
        s.add_geometry(trimesh.creation.axis())
        if isinstance(is_in_robot, IsInRobot):
            src = xx
        elif isinstance(dist_to_robot, DistanceToRobot):
            src = xx
        else:
            src = x
        ic(src.shape, oo.shape)
        cloud = trimesh.PointCloud(src,  # [oo]
                                   colors=np.where(oo[..., None],
                                                   (255, 0, 0),
                                                   (0, 0, 255))
                                   )
        s.add_geometry(cloud)
        s.show()


if __name__ == '__main__':
    main()
