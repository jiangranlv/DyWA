#!/usr/bin/env python3

from typing import Tuple, List, Optional, Iterable
import numpy as np
import pickle

import torch as th
import nvdiffrast.torch as dr
import einops
# import trimesh
from data.transforms.io_xfm import load_mesh
from data.transforms.sample_viewpoint import SampleCamera
from util.torch_util import dcn
import time
from matplotlib import pyplot as plt

import nvtx

import opt_einsum as oe

from icecream import ic

# TODO: automatically configure `PART_SIZE`.
# (optimize by some tradeoff between memory<->compute)
# PART_SIZE: int = 3072
# PART_SIZE: int = 4000
# PART_SIZE: int = 5
PART_SIZE: int = 1250
# PART_SIZE: int = 10000
from util.torch_util import dot


def invert_index(sel: np.ndarray):
    # sel is a mapping _from_
    # the env-logical index to _actual vectex buffer_
    # since f_range v_range for each env. already applies `sel`,
    # we actually need to lookup the pose for the _inverted_ index
    # if env 0 uses object [2, 5], then
    # then self.pi that pointed to [2,2] should point to [0,0] instead
    # in this case, we need a mapping that works like:
    # let sel[i] -> (2, 5) // i-th env uses object (2, 5)
    # sel_inv[i] should look like (_, _, 0, _, _, 1, _, ...)

    # sel_inv[2] -> 0
    # sel_inv[5] -> 1
    # in oth
    pass


def pad_to_array(x: Iterable, dtype=None,
                 debug_export: bool = False):
    if isinstance(x[0], np.ndarray):
        dtype = x[0].dtype
    dim_1 = max(len(row) for row in x)
    out = np.zeros((len(x), dim_1),
                   dtype=dtype)
    for i in range(len(x)):
        out[i, :len(x[i])] = x[i]

    if debug_export:
        print('== debug-export ==')
        with open('/tmp/pad-to-array-debug.pkl', 'wb') as fp:
            pickle.dump(x, fp)

    return out


@th.jit.script
def depth_from_zbuf(d: th.Tensor, n: float, f: float) -> th.Tensor:
    s = (f - n) / 2.0
    b = (f + n) / 2.0
    depth = -d * s + b
    return depth


def split_vert_body_parts(
        v_ranges: np.ndarray,
        part_size: int = PART_SIZE) -> Tuple[int, np.ndarray]:
    v_counts = v_ranges[..., 1] - v_ranges[..., 0]
    v_nexts = np.cumsum(v_counts)
    v_starts = v_nexts - v_counts
    num_parts = (v_counts + part_size - 1) // part_size

    p_starts = []
    p_ends = []
    p_indices = []
    for obj_id, num_part in enumerate(num_parts):
        if num_part == 0:
            continue
        p_indices.extend(np.full(num_part, obj_id, dtype=np.int32))
        p_beg = np.arange(v_starts[obj_id], v_nexts[obj_id], part_size)
        p_starts.extend(p_beg)
        p_ends.extend(
            np.minimum(p_beg + part_size, v_nexts[obj_id])
        )
    p_ranges = np.stack([p_starts, p_ends], axis=-1)
    p_indices = p_indices
    return (p_ranges, p_indices)


def build_index_map(
        part_size: int,
        selectors: np.ndarray,
        v_ranges: np.ndarray,
        f_ranges: np.ndarray,
        p_ranges: np.ndarray,
        p_indices: np.ndarray,
        ofs: np.ndarray,
        n_part_per_scene: int):
    n_scene: int = selectors.shape[0]
    n_obj: int = selectors.shape[1]

    vi_src = []
    vi_dst = []
    fi_src = []
    delta = []

    # p_base: int = 0

    for i in range(n_scene):
        p_base = i * part_size * n_part_per_scene
        for j in range(n_obj):
            object_index = selectors[i, j]

            # << DEAL WITH FACES >>
            fr = f_ranges[object_index]
            fi0, fi1 = int(fr[..., 0]), int(fr[..., 1])
            nf = fi1 - fi0
            fi_src.append(np.arange(fi0, fi1))

            # Query the faces that correspond to this object.
            fis = ofs[fi0: fi1]

            # << DEAL WITH VERTEX PARTITIONS >>
            part_indices = np.argwhere(p_indices
                                       == object_index).ravel()
            fd = np.zeros((nf, 3), dtype=np.int32)
            v00 = v_ranges[object_index, 0]

            # previously:
            # v' = vs[fs + object_offset]
            # now:
            # v' = vs'[fs']

            f_base: int = p_base
            for k in part_indices:
                pr = p_ranges[k]
                vi0, vi1 = int(pr[0]), int(pr[1])
                nv = vi1 - vi0

                # move vertices
                vi_dst.append(p_base + np.arange(nv))
                vi_src.append(np.arange(vi0, vi1))
                # apply part-based vertex offset to faces
                # fd[fis >= vi0] += part_size
                mask = np.logical_and(
                    (vi0 - v00) <= fis,
                    fis < (vi1 - v00)
                    # vi0 <= fis,
                    # fis < vi1
                )
                fd[mask] = f_base
                # fd[mask] = p_base
                # in other words, fs[...] = ofs[fi0:fi1] + p_base
                p_base += part_size
                # f_base += part_size - nv

            # << COMMIT DELTA >>
            delta.append(fd)

    vi_src = np.concatenate(vi_src, axis=0)
    vi_dst = np.concatenate(vi_dst, axis=0)
    fi_src = np.concatenate(fi_src, axis=0)
    delta = np.concatenate(delta, axis=0)
    return (vi_src, vi_dst, fi_src, delta)


# def ndc_projection_matrix(
#        fov: th.Tensor,
#        z_near: float, z_far: float,
#        rhs: bool = False,
#        out: th.Tensor = None) -> th.Tensor:
#    """ Compute NDC projection matrix in the
#    convention of nvdiffrast.
#
#    Args:
#        fov: Field of view (radians)
#        z_near: Near-plane of camera frustum.
#        z_far:  Far-plane of camera frustum.
#        out: Optional output tensor, if buffer already allocated.
#
#    Returns:
#        NDC projection matrix; fov.shape + (4, 4)
#    """
#    # Compute pixel-space focal length.
#    x = th.tan(0.5 * fov) * z_near
#
#    # Allocate projection matrix.
#    P = th.zeros(fov.shape + (4, 4),
#                 out=out, dtype=fov.dtype, device=fov.device)
#
#    # Populate projection matrix.
#    P[..., 0, 0] = z_near / x
#
#    # NOTE: we invert y coords. here
#    # _solely_ for more convenient
#    # depth/color-image transfer
#    # during rasterize().
#    P[..., 1, 1] = -z_near / x
#
#    P[..., 2, 2] = (z_far + z_near) / (z_far - z_near)
#
#    # NOTE:
#    # we _could_ simply transpose this, but
#    # we can avoid discontiguity by just
#    # dealing with the rhs/lhs conventions here.
#    if rhs:
#        # RHS version; x@P
#        P[..., 2, 3] = -(2 * z_far * z_near) / (z_far - z_near)
#        P[..., 3, 2] = 1.0
#    else:
#        # LHS version; P@x
#        P[..., 3, 2] = -(2 * z_far * z_near) / (z_far - z_near)
#        P[..., 2, 3] = 1.0
#    return P


# @th.jit.script
def simple_phong(
        normals: th.Tensor,
        light_dir: th.Tensor,
        ambient_color: th.Tensor,
        diffuse_color: th.Tensor,
        k_a: float = 0.2,
        k_d: float = 1.0):
    # NOV4
    N = normals[..., :3]
    L = light_dir
    # what the fuck are you doing???
    # dot1 = th.einsum('...i, ...i -> ...', N, L)
    # print('N', N.shape) # NUM_ENV X NUM_PART X NUM_VERT X 3
    # print('L', L.shape)
    # lambertian = th.tensordot(N, L, dims=[[-1], [-1]]).clamp_min_(0.0)
    lambertian = dot(N, L).clamp_min_(0.0)
    # print('lambertian', lambertian.shape)
    # dot=dot1
    # lambertian = dot.clamp_min_(0.0)
    # ambient and diffuse only!

    # color = (k_a * ambient_color +
    #          k_d * lambertian[..., None] * diffuse_color)
    color = (
        (k_d * lambertian[..., None] * diffuse_color)
        + (k_a * ambient_color)
    )
    return color


def shade_phong(
        normals: th.Tensor,
        light_dir: th.Tensor,

        ambient_color: th.Tensor,
        diffuse_color: th.Tensor,
        specular_color: th.Tensor,

        vertices: Optional[th.Tensor] = None,

        # ambient
        k_a: float = 1.0,
        # diffuse
        k_d: float = 1.0,
        # specular
        k_s: float = 1.0,
        k_shiny: float = 1.0
):
    # Assuming isotropic scaling and
    # premultiplied modelview matrix (=normalMat),
    # this is valid.
    N = normals[..., :3]

    # Assuming precomputed light direction,
    # this is valid.
    # For instance, point light:
    # vec3 L = normalize(lightPos - vertPos)
    # For instance, directional light:
    # L = broadcast_to_(L, N)
    L = light_dir

    # We assume N, L are compatible shapes.
    # Depending on `l`,
    # L is either global light or point light.
    ddd = th.einsum('...i, ...i -> ...', N, L)

    # Reflected light vector

    # NOTE:
    # R=[...] needs to come before lambertian
    # since it uses ddd which is identical tensor
    # as lambertian.
    if k_s >= 0:
        # R = reflect(-L, N)
        R = -L + 2.0 * ddd[..., None] * N
    lambertian = ddd.clamp_min_(0.0)

    if k_s > 0:
        # Vector to viewer... hmm
        # `vertPos` is the position of the vertex
        # in camera coordinates (not NDC!)
        # how do we handle this without excess
        # memory consumption / duplicate compute?
        # V = normalize(-vertices)
        V = (-vertices[..., :3]) / th.linalg.norm(vertices[..., :3],
                                                  dim=-1, keepdim=True)

        # Compute the specular term
        spec_angle = dot(R, V).clamp_min_(0.0)
        specular = th.pow(spec_angle, k_shiny)
        specular[lambertian <= 0.0] = 0

        color = (k_a * ambient_color +
                 k_d * lambertian[..., None] * diffuse_color +
                 k_s * specular[..., None] * specular_color)
    else:
        # ambient and diffuse only
        color = (k_a * ambient_color +
                 k_d * lambertian[..., None] * diffuse_color)
    return color

# def nvdr_phong(
#         normals: th.Tensor,
#         light_dir: th.Tensor,
#         ambient_color: th.Tensor,
#         diffuse_color: th.Tensor,
#         specular_color: th.Tensor,
#         k_a: float = 1.0,
#         k_d: float = 1.0):
#     Ia = 0.5
#     Is = 1.0
#     a = 128.0

#     N = normalize((nMat * vec4(aNorm, 1.0)).xyz)

#     # diffuse
#     vec3 lightPos = vec3(0.0, 0.0, 10.0)
#     vec3 L = normalize(lightPos - wcVert)

#     diff = dot(N, L).clamp_min_(0.0)
#     vec3 Ka = vec3(1.0, 0.0, 0.0)
#     vec3 cdiff = diff*Ka*Ia

#     # specular
#     R = reflect(-L, N)
#     V = normalize(-wcVert)
#     float spec = pow(max(dot(R, V), 0.0), a)
#     cspec = spec*specular_color*Is

#     # final color
#     color = ambient_color + cdiff + cspec
#     return color


class NvdrRenderer:
    """
    Batch Renderer with NVDiffRast backend.
    """

    def __init__(self,
                 shape: Tuple[int, int],
                 device: 'cuda:1',
                 ctx_type: str = 'gl',
                 use_shader: bool = False,
                 antialias: bool = False,

                 use_depth: bool = True,
                 use_color: bool = True,
                 use_flow: bool = True,
                 use_label: bool = True,
                 fast_depth: bool = True,
                 split_parts: bool = True,
                 apply_mask: bool = False
                 ):
        self.shape = shape
        self.device = device
        self.use_shader = use_shader
        self.antialias = antialias
        self.use_depth = use_depth
        self.use_color = use_color
        self.use_flow = use_flow
        self.use_label = use_label
        self.fast_depth = fast_depth
        self.split_parts = split_parts
        self.apply_mask = apply_mask
        # NOTE:
        # `cuda` context is almost always slower
        # than the LG context.
        if ctx_type == 'gl':
            self.glctx = dr.RasterizeGLContext(
                output_db=False, device=device)
        elif ctx_type == 'cuda':
            self.glctx = dr.RasterizeCudaContext(
                device=device)
        else:
            raise ValueError(F'Unknown ctx_type = {ctx_type}')

        self.ovs: th.Tensor = None
        self.ofs: th.Tensor = None
        self.sel: th.Tensor = None
        self.T_cam: th.Tensor = None
        self.vr: th.Tensor = None
        self.fr: th.Tensor = None

        self.vs: th.Tensor = None
        self.fs: th.Tensor = None
        self.ns: th.Tensor = None
        self.cs: th.Tensor = None

        # NDC cache
        self.uv: th.Tensor = None

    def reset(self,
              obj_vertices: th.Tensor,
              obj_faces: th.Tensor,
              selectors: th.Tensor,
              camera_poses: th.Tensor,
              ndc_transforms: th.Tensor,
              v_ranges: th.Tensor,
              f_ranges: th.Tensor,
              inv_camera: bool,
              # normals is optional,
              # used for RGB shading.
              obj_normals: Optional[th.Tensor] = None,
              obj_colors: Optional[th.Tensor] = None
              ):
        # These are generally expected to
        # _never change_ throughout the lifetime of this class.
        v_ranges = dcn(v_ranges)
        f_ranges = dcn(f_ranges)
        self.vr = v_ranges
        self.fr = f_ranges

        if self.split_parts:
            p_ranges, p_indices = split_vert_body_parts(v_ranges,
                                                        part_size=PART_SIZE)

            self.pr = p_ranges
            # `p_indices`[i] is the object index
            # corresponding to the ith part.
            # we can use it like part_indices = obj_poses[:, p_indices]
            self.pi = p_indices

        self.ovs = obj_vertices
        self.ofs = obj_faces
        self.ons = obj_normals
        self.ocs = obj_colors

        # Which objects should get rendered?
        selectors = dcn(selectors)
        # `sel` maps Z^{num_object_per_env} -> Z^{num_object_types}
        self.sel = selectors

        # What are the camera intrinsics and
        # extrinsics?
        if inv_camera:
            # NOTE: numerically unstable but convenient
            camera_poses = th.linalg.inv(camera_poses)
        self.T_ndc = ndc_transforms
        self.T_pos = camera_poses
        self.T_cam = self.reset_camera(
            camera_poses, ndc_transforms,
            inv_camera=False)

        # Cache selected vertices and faces.
        # maxv, imap, vs, fs
        self.reset_selectors(selectors)

        N = selectors.shape[0]
        # P = num_parts = len(self.pr)
        P = num_parts_per_env = self.ppi.shape[1]

        # m = num envs
        self.oe_expr = oe.contract_expression(
            "mij,msjk,msvk->msvi",
            (N, 4, 4),
            (N, P, 4, 4),
            (N, P, PART_SIZE, 4),
            optimize='optimal'
        )

        # DEFAULT LIGHTS
        if True:
            self.light = th.randn((N, 1, 1, 3),
                                  dtype=th.float32,
                                  device=self.device)
            # z dir has to be pointing down
            self.light[..., 2].abs_().neg_()

            # self.light[..., :3] = [0.3, 0.7, -1.5]

            # self.light = -th.as_tensor(
            #     [0.3, 0.7, 1.5], dtype=th.float32,
            #     device=self.device)
        else:
            self.light = -th.as_tensor(
                [0.3, 0.7, 1.5], dtype=th.float32,
                device=self.device)
        self.light /= th.linalg.norm(self.light, dim=-1, keepdim=True)

        # DEFAULT COLORS
        # Ambient color (color of the light ??)
        self.c_amb = th.rand((N, 1, 1, 3),
                             dtype=th.float32,
                             device=self.device)
        self.c_amb[..., 0] = 1.0
        self.c_amb[..., 1] = 1.0
        self.c_amb[..., 2] = 1.0

        # Specular color
        self.c_spc = th.as_tensor(
            [0.0, 0.0, 1.0], dtype=th.float32,
            device=self.device)

    def reset_selectors(self, selectors: th.Tensor):
        """
        Reset selectors, which are indicator indices
        for which objects or parts are used for each scene.
        """
        selectors = dcn(selectors)
        self.sel = selectors
        # print(selectors[0])

        # Compute "ranges" describing the
        # face index boundary between scenes.
        scene_face_ranges = self.fr[self.sel].sum(axis=1)
        n_faces_per_scene = (scene_face_ranges[..., 1]
                             - scene_face_ranges[..., 0])
        end_faces = np.cumsum(n_faces_per_scene, axis=0)
        off_faces = end_faces - n_faces_per_scene
        ranges = np.stack([off_faces, n_faces_per_scene], axis=-1)
        # print('ranges', ranges)
        ranges = th.as_tensor(ranges,
                              dtype=th.int32, device='cpu').contiguous()
        self.ranges = ranges

        # Compute selector-related stats.
        if False:
            v_index = self.vr[self.sel]
            n_verts_per_obj = (v_index[..., 1] -
                               v_index[..., 0])  # .sum(dim=-1)
            self.maxv = int(n_verts_per_obj.max())
            density = n_verts_per_obj.mean() / self.maxv
            # print(dcn(n_verts_per_obj[0]))
            # print(n_verts_per_obj[:,0], self.maxv)
            print(F'vertex density = {density}')  # 13%!?

        # Allocate vs and index maps.
        n_scene: int = selectors.shape[0]
        n_obj: int = selectors.shape[1]
        # n_part: int = len(self.pr)
        # n_part:int = self.ppi.shape[1]

        # We're going to build this thing manually
        # for sanity's sake...
        self.ppi = []
        # self.ppi2 = []
        for i in range(n_scene):
            ppii = []
            # let's say object poses correspond to object (5, 2)
            # if object 5 corresponds to part (7,8)
            # and object 2 corresponds to part (2,)
            # poses[0] = poses[object=5]
            # the ppii should look like [

            # n_part here is the total number of object parts,
            # not "per-scene".
            n_part: int = len(self.pr)
            for c, object_id in enumerate(self.sel[i]):
                # part_ids = self.pi[object_id]
                for part_id in range(n_part):
                    if self.pi[part_id] == object_id:
                        # ppii.append(part_id)
                        ppii.append(c)
                        # map from
            self.ppi.append(ppii)
        self.ppi = pad_to_array(self.ppi,
                                dtype=np.int32)
        n_part_per_scene: int = self.ppi.shape[1]

        # Build index map.
        # The index map stays the same as long as the
        # selectors are identical.
        imap = build_index_map(PART_SIZE, selectors,
                               self.vr, self.fr, self.pr, self.pi,
                               dcn(self.ofs),
                               n_part_per_scene)
        imap = [th.as_tensor(t, device=self.device) for t in imap]
        vi_src, vi_dst, fi_src, delta = imap

        # Packed list of selected verts and faces.
        # Verts and faces

        # 1. Allocate and populate vs.
        # vs = th.zeros((n_scene, n_part, PART_SIZE, 4),
        #               dtype=th.float32,
        #               device=self.device)
        vs = th.zeros((n_scene, n_part_per_scene, PART_SIZE, 4),
                      dtype=th.float32,
                      device=self.device)
        vs[..., 3] = 1
        vsr = vs.reshape(-1, vs.shape[-1])
        vsr[vi_dst, ..., :3] = self.ovs[vi_src]

        # 2. Create fs.
        fs = (self.ofs[fi_src] + delta)
        self.vs = vs
        self.fs = fs

        # 3. allocate and populate ns.
        # TODO: should ns also be rotated et al.,
        # in the same way that vertices are??
        if self.use_shader:
            if self.ons is not None:
                ns = th.zeros((n_scene, n_part_per_scene, PART_SIZE, 3),
                              dtype=th.float32,
                              device=self.device)
                nsr = ns.reshape(-1, ns.shape[-1])
                nsr[vi_dst, ..., :3] = self.ons[vi_src][..., :3]
                self.ns = ns

            if self.ocs is not None:
                cs = th.zeros((n_scene, n_part_per_scene, PART_SIZE, 3),
                              dtype=th.float32,
                              device=self.device)
                csr = cs.reshape(-1, 3)
                csr[vi_dst, ..., :3] = self.ocs[vi_src]
                # csr[vi_dst, ..., :3] = self.ons[vi_src][...,:3]
                self.cs = cs
        else:
            self.ns = None
            self.cs = None

        if self.use_label:
            # map part index to vertex-wise
            # object labels.
            ls = th.zeros(
                (n_scene, n_part_per_scene, PART_SIZE, 1),
                dtype=th.float32,
                device=self.device)
            # NOTE: offset by `1` to indicate that
            # `0` corresponds to the background.
            # print(ls.shape) # (4, 2, 10000, 1)
            # print(self.ppi.shape) # (4,2)
            ls[...] = 1 + th.as_tensor(
                # self.pi,
                self.ppi,
                dtype=ls.dtype,
                device=ls.device
            )[:, :, None, None]
            self.ls = ls

    def reset_camera(self,
                     camera_poses: Optional[th.Tensor],
                     ndc_transforms: Optional[th.Tensor] = None,
                     inv_camera: bool = False,
                     indices: Optional[th.Tensor] = None) -> th.Tensor:
        if indices is None:
            indices = Ellipsis

        # Populate missing transforms from members.
        # And also sync the members to match the inputs.
        if camera_poses is None:
            camera_poses = self.T_pos[indices]
        else:
            self.T_pos[indices] = camera_poses

        if ndc_transforms is None:
            ndc_transforms = self.T_ndc[indices]
        else:
            self.T_ndc[indices] = ndc_transforms

        # Optionally convert to camera_from_world transform .
        if inv_camera:
            # NOTE: numerically unstable but convenient
            camera_poses = th.linalg.inv(camera_poses)

        # Pre-multiply the ndc projection and extrinsics transforms.
        if self.T_cam is None:
            self.T_cam = th.zeros_like(self.T_pos)
            self.T_cam[:] = th.eye(4)

        self.T_cam[indices] = th.einsum('...ij,...jk->...ik',
                                        ndc_transforms,
                                        camera_poses)

        return self.T_cam

    @nvtx.annotate("Renderer()")
    def __call__(self, obj_poses: th.Tensor,
                 return_vs: bool = False):
        if True:
            with nvtx.annotate("tile_poses()"):
                # obj_poses looks like N X (selected) objects
                # for i_src, i_dst in box_assignments:
                #     box_poses[i_dst] = obj_poses[i_src]

                # obj_poses  : [num_scene X num_obj X  4x4]
                # part_poses : [num_scene X num_part X  4x4]
                # pi_true = [self.pi]

                # if env[0] uses object (5, 2)
                # and object 2 maps to part index (3, 4)
                # and object 5 maps to part index (7,)
                # what we need is (0, 1, 1, ?, ?)
                # in other words, a mapping from object to object count

                # `sel` maps Z^{num_object_per_env} -> Z^{num_object_types}

                # `obj_poses` maps from Z^{num_object_per_env} to
                # SE(3) pose of the object.

                # `pi` maps from Z^{num_parts} -> Z^{num_object_types}
                # however, we _actually_ need a mapping from
                # Z^{num_parts} -> Z^{num_object_per_env}
                # pi = self.sel_inv[self.pi]

                # part_poses = obj_poses[:, pi]
                # part_poses = obj_poses[th.arange(num_scene), pi]
                n_scene: int = self.sel.shape[0]
                part_poses = obj_poses[
                    th.arange(n_scene)[:, None],
                    # pi
                    self.ppi
                ]
                # "4,2,4,4"
                # print('part_poses', part_poses.shape)

        with nvtx.annotate("transform()"):
            # Apply camera and world transforms
            # to the packed list of objects.
            # NDC, modelviewprojection
            # TODO: consider pre-allocating `vs`.
            if True:
                if False:
                    # pytorch default
                    # ndc.cam.obj.v
                    # compute complexity:
                    # mul: MXSXVXI
                    # add: MXSXVXI
                    vs = th.einsum('mij,msjk,msvk->msvi',
                                   self.T_cam,
                                   obj_poses,
                                   self.vs).contiguous()
                else:
                    with nvtx.annotate("vs"):
                        # T_cam: (
                        # print(self.T_cam.shape)
                        # print(part_poses.shape)
                        # print(self.vs.shape)
                        vs = self.oe_expr(self.T_cam,
                                          part_poses,
                                          self.vs)
                        # "4,16,600,4"

                    if self.use_shader:
                        with nvtx.annotate("ns"):
                            ns = self.oe_expr(self.T_pos[..., :3, :3],
                                              part_poses[..., :3, :3],
                                              self.ns)

                        # self.cs: (n_scene, n_part, part_size, 4)
                        with nvtx.annotate("cs"):
                            cs = simple_phong(
                                ns,
                                self.light,
                                self.c_amb,
                                self.cs
                            )
                            # c_spc = th.ones_like(self.c_amb)
                            # c_spc[..., 1] = 0
                            # c_spc[..., 2] = 0
                            # cs = shade_phong(ns,
                            #                  self.light,
                            #                  self.c_amb,
                            #                  self.cs,
                            #                  c_spc,
                            #                  vs,
                            #                  0.2,
                            #                  1.0,
                            #                  0.2)

                # zs is useful for depth image computations.
                zs = vs[..., 3]
            else:
                # m_ij, msjk -> msik
                T = th.matmul(self.T_cam[:, None], obj_poses,
                              out=self.T)
                # msik,msvk->msvi
                # ms_ik,msvk_->msvi
                vs = th.matmul(T[:, :, None],
                               self.vs[..., None],
                               out=self.vs_ndc[..., None]).squeeze(
                    dim=-1)
                zs = vs[..., 3]

        # 1. Rasterize.
        with nvtx.annotate("rasterize()"):
            # qq = vs.reshape(-1, 4)
            # di = qq[..., :3] / qq[..., 3:]
            # print(di.min(), di.max())
            vsr=vs.reshape(-1, 4)
            rast_out, _ = dr.rasterize(
                self.glctx,
                vsr,
                self.fs,
                resolution=self.shape,
                ranges=self.ranges)

        if False:
            # compute color and depth together.
            with nvtx.annotate("depth_and_color()"):
                mask = (rast_out[..., 3] > 0)
                czs = th.concat([cs, zs[..., None]], dim=-1)
                # czs = vs
                # czs[..., :3] = cs
                color_depth, _ = dr.interpolate(
                    czs.reshape(-1, 4), rast_out, self.fs)
                color_depth *= mask[..., None]
                color = color_depth[..., :3]
                depth = color_depth[..., 3]
                out = [mask, depth, color]
        else:
            out = {}
            if self.apply_mask:
                mask = (rast_out[..., 3] > 0)
                out['mask'] = mask

            if self.use_depth:
                if self.fast_depth:
                    # directly use z buffer
                    with nvtx.annotate("depth_from_zbuf()"):
                        z_near = 0.001
                        z_far = 10.0
                        r = rast_out[..., 2]
                        numer = 2.0 * z_near * z_far
                        denom = (z_far + z_near - r * (z_far - z_near))
                        # denom = z_far * (1 - r) + z_near * (1 + r)
                        depth = (numer / denom)

                        # depth = depth_from_zbuf(
                        #     rast_out[..., 2],
                        #     z_near, z_far)
                        if self.apply_mask:
                            depth *= mask
                        out['depth'] = depth
                else:
                    # 2. Compute depth image.
                    with nvtx.annotate("interpolate_depth()"):
                        depth, _ = dr.interpolate(
                            zs.reshape(-1, 1).contiguous(),
                            rast_out, self.fs)
                        depth = depth.squeeze(-1)
                        if self.apply_mask:
                            depth *= mask
                    out['depth'] = depth

            # 3. Compute color image.
            if self.use_color:
                with nvtx.annotate("color()"):
                    color, _ = dr.interpolate(
                        cs.reshape(-1, 3),
                        rast_out, self.fs)

                    if self.antialias:
                        color = dr.antialias(color,
                                             rast_out,
                                             vs.reshape(-1, 4),
                                             self.fs)
                    if self.apply_mask:
                        color *= mask[..., None]
                    out['color'] = color

            # 4. Compute optical flow image. How?
            # flow = NDC(t=0) - NDC(t=-1)
            if self.use_flow:
                uv = vs[..., :2]
                if self.uv is None:
                    # flow image is "zero" by default
                    # TODO: might need to accept `done` as argument
                    # to prevent artifacts at episode boundaries
                    flow = th.zeros(
                        (len(self.ranges),) + self.shape + (2,),
                        dtype=th.float32,  # I guess
                        device=self.device)
                else:
                    d_uv = uv - self.uv
                    with nvtx.annotate("color()"):
                        flow, _ = dr.interpolate(
                            d_uv.reshape(-1, 2),
                            rast_out, self.fs)
                out['flow'] = flow
                # Maybe this is correct, maybe not.
                # maybe it needs to be inverted; etc. etc.
                self.uv = uv

            if self.use_label:
                with nvtx.annotate("label()"):
                    label, _ = dr.interpolate(
                        self.ls.reshape(-1, 1).contiguous(),
                        rast_out, self.fs)
                    label = label.squeeze(-1)
                    if self.apply_mask:
                        label *= mask[..., None]
                    out['label'] = label

        if return_vs:
            out['vs'] = vs

        return out


def main():
    # build_index_map()
    pass


if __name__ == '__main__':
    main()
