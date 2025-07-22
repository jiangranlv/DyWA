#!/usr/bin/env python3

import trimesh
from typing import Tuple
import numpy as np
import open3d as o3d

from cho_util.math import transform as tx
from PIL import Image
import io


def scene_to_img(scene: trimesh.Scene,
                 img_size: int = 64
                 ):
    # Set "decent" camera transform.
    R = tx.rotation.matrix.from_euler([0,0,0])
    T_cam = np.eye(4)
    T_cam[..., :3, :3] = R
    camera_xfm = trimesh.scene.cameras.look_at(
        scene.bounds,
        fov=np.deg2rad(90),
        distance=3.0 * scene.bounding_sphere.primitive.radius,
        center=scene.centroid,
        rotation=T_cam)
    scene.camera_transform = camera_xfm

    # Render image.
    data = scene.save_image(resolution=(img_size, img_size),)
    image = np.asarray(Image.open(io.BytesIO(data)))
    return image


def box_mesh(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0, 0, 0],
                         [width, 0, 0],
                         [0, 0, depth],
                         [width, 0, depth],
                         [0, height, 0],
                         [width, height, 0],
                         [0, height, depth],
                         [width, height, depth]])
    vertices[:, 0] += dx
    vertices[:, 1] += dy
    vertices[:, 2] += dz
    triangles = np.array([[4, 7, 5], [4, 6, 7], [0, 2, 4], [2, 6, 4],
                          [0, 1, 2], [1, 3, 2], [1, 5, 7], [1, 7, 3],
                          [2, 3, 7], [2, 7, 6], [0, 4, 1], [1, 4, 5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box


def gripper_mesh_acronym(
        color: Tuple[int, int, int] = (0, 0, 255),
        tube_radius: float = 0.001, sections: int = 6):
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(radius=0.002, sections=sections, segment=[
        [0, 0, 0], [0, 0, 6.59999996e-02]])
    cb2 = trimesh.creation.cylinder(
        radius=0.002, sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02],
                 [4.100000e-02, 0, 6.59999996e-02]],)

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def gripper_mesh(center, R, width, depth, score=1,
                 scale: float = 1.0,
                 offset: float = None,
                 depth_base: float = 0.02):
    """
    Gripper mesh.
    (From graspnetAPI.utils.utils.py#L453)

    Args:
        center: numpy array of (3,), target point as gripper center
        R: numpy array of (3,3), rotation matrix of gripper
        width: float, gripper width
        score: float, grasp quality score

    Returns:
        open3d.geometry.TriangleMesh

    Author: chenxi-wang
    """
    x, y, z = center
    height = 0.1 * width
    finger_width = 0.01 * width
    depth_base = depth_base * scale
    tail_length = depth_base
    finger_length = 0.5 * depth + offset - tail_length

    delta = (offset - tail_length)

    color_r = score  # red for high score
    color_b = 1 - score  # blue for low score
    color_g = 0
    left = box_mesh(
        finger_length + finger_width,
        finger_width,
        height)
    right = box_mesh(
        finger_length + finger_width,
        finger_width,
        height)
    bottom = box_mesh(finger_width, width, height)
    tail = box_mesh(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= finger_width + delta
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= finger_width + delta
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + delta
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + delta
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate(
        [left_points, right_points, bottom_points, tail_points],
        axis=0)
    # vertices = np.dot(R, vertices.T).T + center
    vertices = vertices @ R.T + center
    triangles = np.concatenate(
        [left_triangles, right_triangles, bottom_triangles, tail_triangles],
        axis=0)
    colors = np.array([[color_r, color_g, color_b]
                       for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    # gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper


def surface_points_from_sdf(
        sdf_points: np.ndarray,
        sdf_values: np.ndarray,
        k: int = 9):
    """Estimate surface points based on the SDF."""
    # NOTE:
    pcd_query = o3d.geometry.PointCloud()
    pcd_query.points = o3d.utility.Vector3dVector(sdf_points)
    knn_query = o3d.geometry.KDTreeFlann(pcd_query)

    p_surf = []

    # NOTE: estimate the gradient from the samples of
    # a scalar field by solving an overdetermined linear system
    # ==> dx@g=dd.
    # This is probably not the most efficient/accurate method,
    # but it's only used for validation anyway.
    for i, p in enumerate(sdf_points):
        _, indices, distances = knn_query.search_knn_vector_3d(p, k)
        p_src = p
        p_dst = sdf_points[indices]
        A = p_dst - p_src  # Nx3
        b = sdf_values[indices] - sdf_values[i]
        g = np.linalg.lstsq(A, b)[0]
        g /= np.linalg.norm(g, axis=-1, keepdims=True)
        p_surf.append(sdf_points[i] - g * sdf_values[i])
    p_surf = np.asanyarray(p_surf, dtype=np.float32)
    return p_surf
