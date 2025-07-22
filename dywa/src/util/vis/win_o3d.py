#!/usr/bin/env python3

import trimesh
from typing import Optional, Tuple, Dict
import numpy as np
import time
import torch as th
import open3d as o3d
import open3d.visualization.gui as gui
import logging

try:
    import pytorch3d as th3
    import pytorch3d.ops
except ImportError:
    logging.warn('torch3d does not seem to work.')

from cho_util.math import transform as tx


def o3d_sphere_from_point(point: np.ndarray,
                          color: Tuple[float, ...] = (1.0, 0.0, 0.0),
                          radius: float = 0.01):
    out = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    out.translate(point)
    out.paint_uniform_color(color)
    return out


def o3d_cloud_from_cloud(point: np.ndarray,
                         color: Tuple[float, ...] = (1.0, 0.0, 0.0)):
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(point)
    if hasattr(color, 'shape'):
        out.colors = o3d.utility.Vector3dVector(color)
    else:
        out.paint_uniform_color(color)
    return out


def o3d_frame_from_pose(pose: np.ndarray):
    out = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    out.rotate(tx.rotation.matrix.from_quaternion(
        pose[..., 3:7]))
    out.translate(pose[..., 0:3])
    return out


class AutoWindow:
    def __init__(self, **kwds):
        app = gui.Application.instance
        app.initialize()
        vis = Window()
        # Setup key callback.
        self.state = {'next': False, 'index': 0}
        vis.set_on_key(self.on_key)
        self.vis = vis

    def on_key(self, key) -> bool:
        if key == gui.KeyName.SPACE:
            self.state['next'] = True
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def tick(self):
        gui.Application.instance.run_one_tick()

    def wait(self, dt: float = 2.0 / 128):
        while not self.state['next']:
            self.tick()
            time.sleep(dt)
        self.state['next'] = False


class Window:
    def __init__(self, **kwds):
        title: str = kwds.pop('title', 'Window')
        shape: Tuple[int, int] = kwds.pop('shape', (500, 1000))

        # [1] window
        self.window = gui.Application.instance.create_window(
            title, shape[1], shape[0])
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        # [2] widget
        self.widget = gui.SceneWidget()
        self.widget.scene = o3d.visualization.rendering.Open3DScene(
            self.window.renderer)
        self.window.add_child(self.widget)

        # [2.1] Panel (image?)
        if True:
            em = self.window.theme.font_size
            margin = 0.5 * em
            self.panel = gui.Vert(0.5 * em, gui.Margins(margin))
            self.panel.add_child(gui.Label("Color image"))
            self.color_widget = gui.ImageWidget()
            self.panel.add_child(self.color_widget)
            self.panel.add_child(gui.Label("Depth image"))
            self.depth_widget = gui.ImageWidget()
            self.panel.add_child(self.depth_widget)
        self.window.add_child(self.panel)

        # [2.5] Event handlers
        self._key_cb = None
        self.widget.set_on_key(self._on_key)

        # [4] state
        self.is_done = False

        # [5] extra cache for geometries
        self._geoms = {}

    def get_image(self):
        return gui.Application.instance.render_to_image(
            self.widget.scene, 1000, 500)

    def add_cloud(self, key: str, cloud, **kwds):
        color = kwds.pop('color', (1, 0, 0, 1))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.paint_uniform_color(color[:3])
        alpha: float = (color[3] if len(color) >= 4 else 1)
        return self.add_geometry(key, pcd,
                                 color=(1, 1, 1, alpha),
                                 **kwds)

    def add_cloud_pair(self, key: str, c0, c1, **kwds):
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(c0)
        src.paint_uniform_color([1, 1, 1])

        dst = o3d.geometry.PointCloud()
        dst.points = o3d.utility.Vector3dVector(c1)
        dst.paint_uniform_color([1, 1, 1])

        n = len(c0)
        ls = (o3d.geometry.LineSet
              .create_from_point_cloud_correspondences(
                  src, dst, list(zip(range(n), range(n)))))
        # `src` = cyan
        g0 = self.add_geometry(F'{key}.src', src, color=(0, 1, 1, 1),
                               point_size=6)
        # `dst` = yellow
        g1 = self.add_geometry(F'{key}.dst', dst, color=(1, 1, 0, 1),
                               point_size=6)
        # line = magenta
        g2 = self.add_geometry(F'{key}.edge',
                               ls, color=(1, 0, 1, 1))
        return (g0, g1, g2)

    def add_voxel_grid(self, key: str, voxel, **kwds):
        color = kwds.pop('color', (1, 0, 0, 1))
        thresh = kwds.pop('thresh', 0.5)
        voxel_dims = kwds.pop('voxel_dims', 1.0)
        min_bound = kwds.pop('min_bound', 0.0)
        # alpha: float = (color[3] if len(color) >= 4 else 1)

        # voxel to mesh

        # NOTE: cubify() is a really weird operation
        # that requires ZYX ordering...
        voxel = th.as_tensor(
            voxel,
            dtype=th.float32,
            device='cuda').permute(
            2,
            1,
            0)[None]
        # print('voxel', th.count_nonzero(voxel>0))
        # voxel.permute(2, 1, 0)[None]
        # spans (-1, 1) in maximal axis
        mesh_th = th3.ops.cubify(voxel, thresh=thresh)
        verts = mesh_th.verts_list()[0].detach().cpu().numpy()
        # print('scale', voxel_dims * voxel.shape[-3:])
        # print('min_bound', min_bound.shape)
        # print('voxel_dims', voxel_dims)
        # print('verts', verts)
        verts = (min_bound + (verts + 1.0) * np.multiply(0.5,
                 voxel_dims) * voxel.shape[-3:])
        faces = mesh_th.faces_list()[0].detach().cpu().numpy()

        # th3 mesh to o3d mesh
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        # res0 = self.add_geometry(key, mesh_o3d,
        #                         color=color,
        #                         **kwds)
        res0 = None
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(
            mesh_o3d)
        res1 = self.add_geometry(key + '/wire',
                                 line_set, **kwds)
        return (res0, res1)

    def add_axes(self, key: str, pose: Optional[np.ndarray] = None, **kwds):
        geom = o3d.geometry.TriangleMesh.create_coordinate_frame(
            kwds.pop('size', None))
        if pose is not None:
            geom.transform(pose)
        return self.add_geometry(key, geom, **kwds)

    def remove_geometry(self, key: str):
        if self.widget.scene.has_geometry(key):
            self.widget.scene.remove_geometry(key)
        return self._geoms.pop(key, None)

    def clear_geometry(self):
        keys = list(self._geoms.keys())
        for key in keys:
            self.remove_geometry(key)

    def add_geometry(
            self, key: str, geometry, material=None,
            color: Optional[Tuple[float, float, float, float]] = None, **kwds):
        setup_camera: bool = kwds.pop('setup_camera', False)

        def _add_geometry():

            # Configure material.
            if material is None:
                m = o3d.visualization.rendering.MaterialRecord()
                if color is None:
                    c = (1.0, 0.0, 0.0, 1.0)
                else:
                    c = color
                m.base_color = c
                m.shader = kwds.pop('shader', 'defaultLitTransparency')
                # m.shader = "defaultUnlitTransparency"
                m.point_size = kwds.pop('point_size', m.point_size)
            else:
                m = material

            # add or replace geometry.
            if self.widget.scene.has_geometry(key):
                self.widget.scene.remove_geometry(key)
            self.widget.scene.add_geometry(key, geometry, m)
            self._geoms[key] = geometry
            # self.widget.scene.modify_geometry_material(key, m)
            if setup_camera:
                self.widget.setup_camera(60.0,
                                         self.widget.scene.bounding_box,
                                         (0, 0, 0))
            self.widget.scene.set_background((1, 1, 1, 1))

        gui.Application.instance.post_to_main_thread(
            self.window, _add_geometry)
        
    def add_image(self, color=None, depth=None):
        color_image = None
        depth_image = None
        if color is not None:
            color_image = o3d.geometry.Image(color)
        if depth is not None:
            depth_image = o3d.geometry.Image(depth)

        def _add_image():
            if color_image is not None:
                self.color_widget.update_image(color_image)
            if depth_image is not None:
                self.depth_widget.update_image(depth_image)
            # no clue abt this
            # self.widget.scene.set_background([1, 1, 1, 1], rgb_frame)
        gui.Application.instance.post_to_main_thread(
            self.window, _add_image)

    @property
    def geometries(self):
        return self._geoms

    def set_on_key(self, key_cb):
        self._key_cb = key_cb

    def step(self, delay: float = 0.0):
        gui.Application.instance.run_one_tick()
        if delay > 0.0:
            time.sleep(delay)

    def _on_key(self, key_event) -> int:
        if self._key_cb is not None:
            if key_event.type == gui.KeyEvent.UP:
                return self._key_cb(key_event.key)
        return gui.Widget.EventCallbackResult.IGNORED

    def _on_layout(self, layout_context):
        if True:
            content_rect = self.window.content_rect
            panel_width = 15 * layout_context.theme.font_size  # 15 ems wide
            self.widget.frame = gui.Rect(content_rect.x, content_rect.y,
                                        content_rect.width - panel_width,
                                        content_rect.height)
            self.panel.frame = gui.Rect(self.widget.frame.get_right(),
                                        content_rect.y, panel_width,
                                    content_rect.height)
        else:
            rect = self.window.content_rect
            width = 15 * layout_context.theme.font_size
            self.widget.frame = gui.Rect(
                rect.x, rect.y,
                rect.width - width,
                rect.height)

    def _on_close(self):
        self.is_done = True
        return True

    def _update(self):
        pass

    def _on_update(self):
        gui.Application.instance.post_to_main_thread(
            self.window, self._update)


def main():
    win = AutoWindow()
    vis = win.vis
    vis.add_cloud('cloud', np.random.normal(size=(512, 3)))
    win.wait()


if __name__ == '__main__':
    main()
