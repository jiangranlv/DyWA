#!/usr/bin/env python3

from typing import (
    Optional, Iterable, Dict,
    List, Union, Callable)

import copy
import os
from pathlib import Path
from yourdfpy import URDF
from yourdfpy.urdf import apply_visual_color
import trimesh
import numpy as np

from data.transforms.io_xfm import scene_to_mesh

from icecream import ic


def _sanitize_alt_path(
        alt_path: Optional[Union[List[str], str]] = None) -> List[str]:
    if isinstance(alt_path, str) or isinstance(alt_path, os.PathLike):
        alt_path = [alt_path]
    if alt_path is None:
        alt_path = []
    alt_path = [''] + alt_path
    return alt_path


def _resolve_mesh_path(f: str,
                       alt_path: List[str]):
    if 'package://' in f:
        f = f.replace('package://', '')
    for root in [''] + alt_path:
        f2 = F'{root}/{f}'
        if Path(f2).is_file():
            return f2
    raise FileNotFoundError(
        F'{f} not found from paths = {alt_path}')


def _scene_to_mesh_with_visual_hack(m: trimesh.Scene) -> trimesh.Trimesh:
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
                        part.visual.material.diffuse[:3]
                        * np.ones_like(part.vertices)
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
                  cat: bool = True,
                  acd: bool = True):
    alt_path = _sanitize_alt_path(alt_path)

    geometries = link.collisions if use_col else link.visuals
    visuals = link.visuals

    if len(geometries) == 0:
        return None
    meshes = []
    for g in geometries:
        if g.geometry.mesh is None:
            continue

        f = _resolve_mesh_path(
            g.geometry.mesh.filename,
            alt_path=alt_path)

        if acd:
            assert (use_col)
            assert (not cat)
            m = trimesh.load(f,
                             split_object=True,
                             group_material=False,
                             skip_texture=True,
                             skip_materials=True)
        else:
            m = trimesh.load(f, skip_materials=False)

        # >> TRY TO LOAD A "DECENT" visual mesh.
        if cat and isinstance(m, trimesh.Scene):
            if not use_col:
                m = _scene_to_mesh_with_visual_hack(m)
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

        if g.geometry.mesh.scale is not None:
            # print(F'apply scale = {g.geometry.mesh.scale}')
            S = np.eye(4)
            S[:3, :3] = np.diag(g.geometry.mesh.scale)
            pose = pose @ S
        m.apply_transform(pose)
        meshes.append(m)
    if len(meshes) == 0:
        return None
    if cat:
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = meshes
    return mesh


def load_objects(
        object_urdfs: Optional[Iterable[str]] = None,
        use_col: bool = False,
        urdf_link_map: Optional[Dict[str, List[str]]] = None,
        load_ground: bool = True,
        blocklist: Optional[Dict[str, Iterable[str]]] = None,
        cat: bool = True,
        acd: bool = False,
        on_mesh: Callable[[int, trimesh.Trimesh], None] = None):
    if blocklist is None:
        blocklist = {}
    offset: int = 0
    for object_urdf in object_urdfs:
        skip_link = frozenset(blocklist.get(object_urdf, []))

        # URDF Object(s)
        obj = URDF.load(
            object_urdf,
            load_meshes=False,
            load_collision_meshes=False
        )

        # multibody
        link_names = urdf_link_map[object_urdf]
        for link_name, link in obj.link_map.items():
            if (link_name in skip_link):
                print(F'skipping link={link_name} due to blocklist.')
                # raise ValueError('.')
                continue
            m = get_link_mesh(
                obj, link, use_col=use_col, alt_path=[
                    Path(object_urdf).parent,
                    Path(object_urdf).parent.parent,
                    Path(object_urdf).parent.parent.parent,
                ], cat=cat, acd=acd)

            # skip.
            if (m is None):
                print(F'link mesh for {link_name} was None.')
                continue

            # populate.
            i = link_names.index(link_name) + offset
            on_mesh(i, m,
                    urdf=object_urdf,
                    link=link_name)
        offset += len(link_names)

    if load_ground:
        ground = trimesh.creation.box(
            extents=(20.0, 20.0, 0.001)
        )
        on_mesh(offset, ground)
    return


def load_rendering_objects(*args, **kwds):
    class OnMesh:
        def __init__(self):
            self.vertices = []
            self.faces = []
            self.normals = []
            self.colors = []

        def __call__(self, i: int, m: trimesh.Trimesh, **kwds):
            # Allocate missing parts
            if i >= len(self.vertices):
                num_insert = (i + 1) - len(self.vertices)
                self.vertices += [np.empty((0, 3), dtype=np.float32)
                                  for _ in range(num_insert)]
                self.faces += [np.empty((0, 3), dtype=np.int32)
                               for _ in range(num_insert)]
                self.normals += [np.empty((0, 3), dtype=np.float32)
                                 for _ in range(num_insert)]
                self.colors += [np.empty((0, 3), dtype=np.float32)
                                for _ in range(num_insert)]
            # Assign to specified index
            self.vertices[i] = m.vertices
            self.faces[i] = m.faces
            self.normals[i] = m.vertex_normals
            try:
                c = m.visual.to_color().vertex_colors[..., :3]
            except AttributeError:
                c = m.visual.vertex_colors[..., :3]
            self.colors[i] = (c / 255.0).astype(np.float32)

    on_mesh = OnMesh()
    load_objects(*args, **kwds, on_mesh=on_mesh)
    out = (on_mesh.vertices,
           on_mesh.faces,
           on_mesh.normals,
           on_mesh.color)
    return out


def load_hull_objects(object_urdfs: Optional[Iterable[str]] = None,
                      urdf_link_map: Optional[Dict[str, List[str]]] = None,
                      load_ground: bool = True,
                      blocklist: Optional[Dict[str, Iterable[str]]] = None):
    """
    Returns:
        hulls: (num_env_links) X (num_hulls_per_link) x trimesh
    """
    class OnMesh:
        def __init__(self):
            self.hulls = []

        def __call__(self, i: int, m: trimesh.Trimesh, **kwds):
            assert (isinstance(m, list))
            # m = result of get_link_mesh()
            link_meshes = m
            for scene in link_meshes:
                if not isinstance(scene, trimesh.Scene):
                    scene = trimesh.Scene(scene)
                for node_name in scene.graph.nodes_geometry:
                    transform, geom_name = scene.graph[node_name]
                    mesh = copy.deepcopy(scene.geometry[geom_name])
                    mesh.apply_transform(transform)

                    # append_to_element_if()
                    num_insert = (i + 1) - len(self.hulls)
                    if num_insert > 0:
                        self.hulls.extend([[] for _ in range(num_insert)])
                    ic(node_name, kwds.get('urdf'), kwds.get('link'), i)
                    self.hulls[i].append(mesh)
    on_mesh = OnMesh()
    load_objects(object_urdfs, True, urdf_link_map, False,
                 blocklist=blocklist,
                 cat=False, acd=True, on_mesh=on_mesh)
    return on_mesh.hulls


def pack_shapes(hullss):
    # link_from_hull = []
    hull_from_link = []
    for link_index, hulls in enumerate(hullss):
        ic(link_index, len(hulls))
        hull_from_link.append(len(hull_from_link) + np.arange(len(hulls)))
        # link_from_hull.extend([link_index] * len(hulls))
    hulls = sum(hullss, [])
    # return hulls, link_from_hull
    return hulls, hull_from_link


def test_load_hulls():
    # urdf_file = (
    #     '../../../data/assets/franka_description/robots/franka_panda.urdf'
    # )
    urdf_file = (
        # '/input/DGN/meta-v8/urdf/core-bottle-1071fa4cddb2da2fc8724d5673a063a6-0.060.urdf'
        '/input/DGN/meta-v8/urdf/ddg-ycb_011_banana-0.120.urdf'
    )
    urdf_link_map = URDF.load(urdf_file, load_meshes=False).link_map
    urdfs = [urdf_file]
    link_maps = {urdf_file: list(urdf_link_map.keys())}
    hullss = load_hull_objects(urdfs, link_maps)
    trimesh.Scene(hullss).show()
    # for hulls in hullss:
    #     for hull in hulls:
    #         hull.show()


def test_pack_hulls():
    franka_file = (
        '../../../data/assets/franka_description/robots/franka_panda.urdf'
    )
    object_file = (
        '/input/DGN/meta-v8/urdf/core-bottle-1071fa4cddb2da2fc8724d5673a063a6-0.060.urdf'
    )
    # [1] `urdfs` should be in order of actor
    urdfs = [franka_file, object_file]
    link_maps = {k: list(URDF.load(k, load_meshes=False).link_map.keys())
                 for k in urdfs}
    hullss = load_hull_objects(urdfs, link_maps)

    # Create an example, might not exactly match IG order
    link_body_indices = []
    for u in urdfs:
        for l in link_maps.get(u):
            link_body_indices.append(len(link_body_indices))

    # map body indices to hull indices
    hull_flat, hull_body_indices = pack_shapes(hullss)
    # hull_body_indices = []
    # hull_flat = []
    # for body_index, hulls in zip(link_body_indices, hullss):
    #     for hull in hulls:
    #         hull_body_indices.append(body_index)
    #         hull_flat.append(hull)
    print(hull_body_indices)
    print([h.faces.shape for h in hull_flat])
    hull_poses = body_poses[:, hull_body_indices, :]

    # laid out as { body_index(?): geometries_per_each_body } I think
    print(len(hullss), sum(len(l) for l in link_maps.values()))
    # assert (len(hullss) == len(urdfs))
    # for h, u in zip(hullss, urdfs):
    #     assert (len(h) == len(link_maps.get(u)))

    # for hulls in hullss:
    #     for hull in hulls:
    #         hull.show()


def main():
    test_load_hulls()


if __name__ == '__main__':
    main()
