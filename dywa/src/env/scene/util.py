#!/usr/bin/env python3

import pickle
from typing import Tuple
import numpy as np
import re
from data.transforms.io_xfm import load_mesh
from functools import partial
import torch as th
from typing import Optional, Callable
import einops
import copy
from tqdm.auto import tqdm

import trimesh
from pathlib import Path
import gdown
from cho_util.math import transform as tx
from yourdfpy import URDF
from util.torch_util import dcn
from util.math_util import matrix_from_pose
from data.transforms.io_xfm import scene_to_mesh

import pyglet
from icecream import ic
from trimesh.viewer.windowed import SceneViewer

import shapely
from scipy.spatial import ConvexHull


def box_inertia(dims: np.ndarray) -> np.ndarray:
    sq = np.square(dims)
    return (1.0 / 12.0) * np.diag(sq.sum() - sq)


def create_bin(
        table_dims: Tuple[float, float, float],
        wall_width: float,
        wall_height: float):
    # Base
    xw, yw, zw = (
        0.5 * (table_dims[0] - wall_width),
        0.5 * (table_dims[1] - wall_width),
        0.5 * (table_dims[2] + wall_height)
    )

    mb = trimesh.creation.box(table_dims)
    # wall along x-axis
    mx = trimesh.creation.box((table_dims[0], wall_width, wall_height))
    # wall along y-axis
    my = trimesh.creation.box((wall_width, table_dims[1], wall_height))
    BOX_LINK_TEMPLATE: str = '''
        <link name="{name}">
            <inertial>
                <mass value="{mass}"/>
                <origin xyz="0 0 0"/>
                <inertia ixx="{ixx:.03g}" ixy="{ixy:.03g}" ixz="{ixz:.03g}"
                iyy="{iyy:.03g}" iyz="{iyz:.03g}" izz="{izz:.03g}"/>
            </inertial>
            <visual>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <geometry>
                    <box size="{s[0]} {s[1]} {s[2]}" />
                </geometry>
            </visual>
            <collision>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <geometry>
                    <box size="{s[0]} {s[1]} {s[2]}" />
                </geometry>
            </collision>
        </link>
    '''

    def _box(name, dims, density: float = 300.0):
        mass: float = density * float(np.prod(dims))
        I = mass * box_inertia(table_dims)
        return BOX_LINK_TEMPLATE.format(
            name=name,
            mass=mass,
            s=dims,
            ixx=I[0, 0],
            ixy=I[0, 1],
            ixz=I[0, 2],
            iyy=I[1, 1],
            iyz=I[1, 2],
            izz=I[2, 2]
        )

    out: str = '''<robot name="robot">
        {base}
        {wall_px}
        {wall_nx}
        {wall_py}
        {wall_ny}
        <joint name="base_px" type="fixed">
            <origin xyz="0 {dy} {dz}" rpy="0 0 0"/>
            <parent link="base"/>
            <child link="wall_px"/>
        </joint>

        <joint name="base_nx" type="fixed">
            <origin xyz="0 -{dy} {dz}" rpy="0 0 0"/>
            <parent link="base"/>
            <child link="wall_nx"/>
        </joint>

        <joint name="base_py" type="fixed">
            <origin xyz="{dx} 0 {dz}" rpy="0 0 0"/>
            <parent link="base"/>
            <child link="wall_py"/>
        </joint>

        <joint name="base_ny" type="fixed">
            <origin xyz="-{dx} 0 {dz}" rpy="0 0 0"/>
            <parent link="base"/>
            <child link="wall_ny"/>
        </joint>
    </robot>
    '''.format(
        base=_box('base', table_dims),
        wall_px=_box('wall_px', (table_dims[0], wall_width, wall_height)),
        wall_nx=_box('wall_nx', (table_dims[0], wall_width, wall_height)),
        wall_py=_box('wall_py', (wall_width, table_dims[1], wall_height)),
        wall_ny=_box('wall_ny', (wall_width, table_dims[1], wall_height)),
        dx=xw,
        dy=yw,
        dz=zw)
    # with open('/tmp/xxx.urdf', 'w') as fp:
    #     fp.write(out)
    # URDF.load('/tmp/xxx.urdf').show()
    return out


def rejection_sample(size: int,
                     sample_fn: Callable[[int], np.ndarray],
                     accept_fn: Callable[[np.ndarray], np.ndarray],
                     max_oversample_ratio: float = 4.0,
                     verbose: bool = False):
    remain: int = size
    numer: int = 0
    denom: int = 0

    acceptance_ratio: float = 1.0

    out = []
    with tqdm(total=size, disable=(not verbose), desc='rej_sample') as pbar:
        while remain > 0:
            num_sample = remain * min(max_oversample_ratio,
                                      1.0 / acceptance_ratio)

            samples = sample_fn(size)
            accept = accept_fn(samples)
            good = samples[accept]
            num_accept = len(good)

            out.append(samples)
            remain -= num_accept
            pbar.update(num_accept)

            numer += num_accept
            denom += num_sample

            acceptance_ratio = numer / denom
    return np.concatenate(out, axis=0)[:size]


class SV(SceneViewer):
    def __init__(self, *args, **kwds):
        self.__on_key = kwds.pop('on_key')
        super().__init__(*args, **kwds)

    def on_key_press(self, symbol, modifiers):
        super().on_key_press(symbol, modifiers)
        if self.__on_key is not None:
            self.__on_key(pyglet.window.key._key_names[symbol],
                          modifiers)


def _show_stable_pose(
        mesh: trimesh.Trimesh,
        pose: np.ndarray,
        add_table: bool = True,
        on_key: Optional[Callable[[str, int], None]] = None
):

    if isinstance(mesh, trimesh.Scene):
        # scene = copy.deepcopy(mesh)
        scene = mesh  # hmm...
    else:
        scene = trimesh.Scene([mesh])

    # Shorthand for torch tensor conversion
    T = partial(th.as_tensor,
                dtype=th.float,
                device='cpu')

    # FIXME: hardcoded table
    table = trimesh.creation.box((0.4, 0.5, 0.4))
    table_xfm = np.eye(4)
    table_xfm[2, 3] = +0.2

    if add_table:
        m = dcn(matrix_from_pose(T(pose[..., :3]),
                                 T(pose[..., 3:7])))
        scene.apply_transform(m)
        # v.color= (0,1,0)
        scene.add_geometry(table, transform=table_xfm)
        for k, v in scene.geometry.items():
            v.visual.face_colors = np.random.randint(128, 255, size=3,
                                                     dtype=np.uint8)
        scene.add_geometry(trimesh.creation.axis())

        R = tx.rotation.matrix.from_euler([+np.pi / 4, 0, np.pi / 4])
        T_cam = np.eye(4)
        T_cam[..., :3, :3] = R
        camera_xfm = trimesh.scene.cameras.look_at(scene.bounds,
                                                   fov=np.deg2rad(90),
                                                   distance=1.0,
                                                   center=(0, 0, 0.4),
                                                   rotation=T_cam)
        scene.camera_transform = camera_xfm

    viewer = SV(scene, on_key=on_key)


def _support_polygon(v: np.ndarray,
                     com: Optional[Tuple[float, ...]] = None,
                     eps: float = 1e-2):
    z_min = v[..., 2].min()
    z_max = v[..., 2].max()

    msk = (v[..., 2] <= (z_min + eps))
    if com is None:
        com = v.mean(axis=0)

    v_msk = v[msk]
    try:
        hull = ConvexHull(v_msk[..., :2])
    except:
        return None
    return v_msk[hull.vertices]


def _is_stable(v: np.ndarray,
               com: Optional[Tuple[float, ...]] = None,
               eps: float = 1e-2,
               min_angle: float = np.deg2rad(15)
               ):
    if com is None:
        com = v.mean(axis=0)

    points = _support_polygon(v, com, eps)
    if points is None:
        return False
    poly = shapely.Polygon(points[..., :2])

    com_xy = com[..., :2]

    d_xy = poly.distance(shapely.Point(com_xy))
    if d_xy != 0.0:
        return False

    d_xy = poly.boundary.distance(shapely.Point(com_xy))
    z = com[..., 2] - v[..., 2].min()
    ang = np.arctan2(d_xy, z)
    return (ang >= min_angle)


def _show_stable_poses(
        obj_set, scene: bool = True,
        on_key: Optional[Callable[[str, str, int],
                                  None]] = None,
        per_obj: int = 1):
    for key in obj_set.keys():
        T = partial(th.as_tensor,
                    dtype=th.float,
                    device='cpu')

        # FIXME: hardcoded table
        table = trimesh.creation.box((0.4, 0.5, 0.4))
        table_xfm = np.eye(4)
        table_xfm[2, 3] = +0.2

        poses = obj_set.pose(key)
        ic(key, poses.shape)
        if poses is None:
            print(F'Skipping poseless {key}')
            continue

        for i, state in zip(range(per_obj), iter(poses)):
            obj = URDF.load(obj_set.urdf(key))

            # Apply pose
            m = dcn(matrix_from_pose(T(state[..., :3]),
                                     T(state[..., 3:7])))

            # mmm = scene_to_mesh(obj.scene)
            # mmm.apply_transform(m)
            # mmm.show()

            # print(_is_stable(mmm.vertices @ m[...,:3,:3].T + m[..., :3, 3],
            # mmm.center_mass @ m[...,:3,:3].T + m[..., :3, 3]))

            obj.scene.apply_transform(m)

            # Add table
            if scene:
                obj.scene.add_geometry(table, transform=table_xfm)

            # Colorize
            for k, v in obj.scene.geometry.items():
                v.visual.face_colors = np.random.randint(128, 255, size=3,
                                                         dtype=np.uint8)

            # Add pose axes
            pose_axis = trimesh.creation.axis().apply_scale(0.2)
            pose_axis.apply_transform(m)
            obj.scene.add_geometry(pose_axis)

            # Add origin axes
            obj.scene.add_geometry(trimesh.creation.axis())

            # Configure camera
            R = tx.rotation.matrix.from_euler([+np.pi / 4, 0, np.pi / 4])
            T_cam = np.eye(4)
            T_cam[..., :3, :3] = R
            camera_xfm = trimesh.scene.cameras.look_at(obj.scene.bounds,
                                                       fov=np.deg2rad(90),
                                                       distance=1.0,
                                                       center=(0, 0, 0.4),
                                                       rotation=T_cam)
            obj.scene.camera_transform = camera_xfm

            # Show and wait for key
            print(F'key = {key}')
            if on_key is not None:
                _on_key = partial(on_key, key)
            else:
                _on_key = None
            viewer = SV(obj.scene, on_key=_on_key)


def mesh_from_urdf(urdf: str):
    return URDF.load(urdf, force_collision_mesh=True).scene


def stat_from_mesh(mesh: trimesh.Trimesh):
    # Get some mesh-related statistics.
    entry = {}
    try:
        aabb = mesh.bounds
        entry['aabb'] = np.asarray(aabb, dtype=np.float32).tolist()

        corners = trimesh.bounds.corners(mesh.bounds)
        entry['bbox'] = corners.tolist()

        entry['volume'] = mesh.volume

        obb = mesh.bounding_box_oriented
        entry['obb'] = (
            np.asarray(obb.transform).tolist(),
            np.asarray(obb.extents).tolist())
        single = mesh.dump(concatenate=True)
        radius = np.linalg.norm(single.vertices, axis=-1).max()
        entry['radius'] = radius
        entry['num_faces'] = len(single.faces)
        entry['num_verts'] = len(single.vertices)
    except ValueError as e:
        entry = {}
        print(F'skipping {mesh} due to {e}')
    return entry


def lookup_normal(mesh: trimesh.Trimesh,
                  point: np.ndarray):
    _, _, face_idx = trimesh.proximity.closest_point(mesh, point)
    return mesh.face_normals[face_idx]


def stat_from_urdf(urdf: str):
    scene = mesh_from_urdf(urdf)
    return stat_from_mesh(scene)


def sample_stable_poses(hull: trimesh.Trimesh,
                        # xy_bound:
                        height: float = 0.4,
                        num_samples: int = 16,
                        multiplier: int = 8):
    xfms, probs = trimesh.poses.compute_stable_poses(
        hull, n_samples=(num_samples * multiplier))
    xfms = xfms[np.argsort(probs)[-num_samples:]]
    R = tx.rotation_from_matrix(xfms)
    x = tx.translation_from_matrix(xfms)
    # x[..., :2] =
    x[..., 2] += height
    q = tx.rotation.quaternion.from_matrix(R)
    pose = np.concatenate([x, q], axis=-1)
    return pose


def align_vectors(a: np.ndarray, b: np.ndarray):
    # find the SVD of the two vectors
    a_ = einops.rearrange(a, '... d -> (...) one d', one=1)
    b_ = einops.rearrange(b, '... d -> (...) one d', one=1)
    a_u = np.linalg.svd(a_)[0]
    b_u = np.linalg.svd(b_)[0]

    if np.linalg.det(a_u) < 0:
        a_u[:, -1] *= -1.0

    if np.linalg.det(b_u) < 0:
        b_u[:, -1] *= -1.0

    # put rotation into homogeneous transformation
    return b_u.dot(a_u).T


def align_z(a: np.ndarray, eps: float = 1e-6):
    """ R@a=z """
    c = a[..., 2]
    ki = (1.0 + c)
    # R = np.eye(3) + np.hat
    x, y, z = a[..., 0], a[..., 1], a[..., 2]
    R = np.stack([
        -x * x + ki,
        -x * y,
        -x * ki,

        -x * y,
        -y * y + ki,
        -y * ki,

        x * ki,
        y * ki,
        -x * x - y * y + ki
    ], axis=-1)
    with np.errstate(divide='ignore'):
        out = np.where(c[..., None] <= -1.0 + eps,  # (...)
                       -np.eye(3).ravel(),
                       R / ki[..., None])
    return out.reshape(*a.shape[:-1], 3, 3)


def plane_transform(normal: np.ndarray,
                    origin: Optional[np.ndarray] = None,
                    eps: float = 1e-6):
    rxn = align_z(normal, eps=eps)
    txn = None
    if origin is not None:
        txn = -np.einsum('...ij, ...j -> ...i', rxn, origin)
    return (rxn, txn)


def sample_stable_poses_fast(
        hull: trimesh.Trimesh,
        com: np.ndarray,
        height: float = 0.4,
        target: int = 32):
    """ approximate (static) stable poses """

    origin = hull.triangles_center
    normal = hull.face_normals
    proj_dists = np.einsum('ij,ij->i', normal,
                           com - origin)
    proj_coms = com - np.einsum('i,ij->ij',
                                proj_dists, hull.face_normals)
    barys = trimesh.triangles.points_to_barycentric(
        hull.triangles, proj_coms)

    # Also add "heuristic" filter
    mask = (
        np.all(barys > 0, axis=1)
        & np.all(barys < 1, axis=1)
        & (np.sum(barys, axis=1) <= 1)
    )

    if mask.sum() > target:
        weights = mask * hull.area_faces
        index = np.random.choice(len(weights),
                                 size=target,
                                 p=weights / weights.sum(),
                                 replace=False)
    else:
        # index = np.argwhere(mask)
        index = np.where(mask)[0]

    if len(index) <= 0:
        return None

    # stable_faces = hull.faces[stable_index]
    rxn, txn = plane_transform(-normal[index], origin[index])

    # rxn, txn = plane_transform(-normal, origin)
    # com_ = th.einsum('fij, j -> fi', rxn, com) + txn
    # tri_ = th.einsum('fij, ftj -> fti', rxn, hull.triangles) + txn
    # is_stable = (com_[...,2] > 0) & (
    #        com[..., None, :2] - hull.triangles[..., :, :2]
    # txn

    # apply stability mask.
    # rxn = rxn[is_stable]
    # txn = txn[is_stable]

    # Format and return results.
    qxn = tx.rotation.quaternion.from_matrix(rxn)
    txn[..., 2] += height
    pose = np.concatenate([txn, qxn], axis=-1)
    if np.any(np.isnan(pose)):
        raise ValueError('NANI!?')
    return pose


def _get_gdrive_file_id(url: str) -> str:
    """
    Get unique file identifier from Gdrive share URL.
    FIXME: super fragile
    """
    regex = r"https://drive\.google\.com/file/d/([-\w]+)"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None


def _load_meta(filename: str, url: str, binary: bool = True,
               load_fn=pickle.load):
    if not Path(filename).is_file():
        # Try to fix URL if applicable.
        file_id = _get_gdrive_file_id(url)
        if file_id is not None:
            url = F'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, filename, quiet=False)
    flag = 'rb' if binary else 'r'
    with open(filename, flag) as fp:
        out = load_fn(fp)
    return out


def load_pkl(s: str):
    if not Path(s).is_file():
        return None
    with open(s, 'rb') as fp:
        return pickle.load(fp)


def load_npy(s: str):
    if not Path(s).is_file():
        return None
    try:
        return np.load(s)
    except FileNotFoundError:
        return None


def load_glb(s: str):
    if not Path(s).is_file():
        return None
    try:
        return load_mesh(s, as_mesh=True)
    except ValueError:
        return None


def main():
    out = create_bin((0.4, 0.5, 0.4), 0.02, 0.02)
    print(out)


if __name__ == '__main__':
    main()
