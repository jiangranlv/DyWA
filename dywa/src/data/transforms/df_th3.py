#!/usr/bin/env python3

from typing import List, Optional, Dict, Tuple

import torch as th
import pytorch3d as th3
import pytorch3d.utils
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import point_face_distance


class DistanceTransform:
    """ (Unsigned) Distance Transform.
    Measures the distance between the mesh and query points.
    Assumes batched inputs for the meshes and queries.
    """
    KEYS: Tuple[str, ...] = ('verts', 'faces', 'query', 'distance')

    def __init__(self, key_map: Optional[Dict[str, str]] = None):
        if key_map is None:
            key_map = {k: k for k in DistanceTransform.KEYS}
        self.key_map = key_map

        # Parse keys
        self.key_v = self.key_map['verts']
        self.key_f = self.key_map['faces']
        self.key_q = self.key_map['query']
        self.key_d = self.key_map['distance']
        self.key_g = self.key_map.get('grad', None)

    def __call__(self, inputs):
        verts: List[th.Tensor] = inputs[self.key_v]  # mesh vertices
        # mesh triangle face indices
        faces: List[th.Tensor] = inputs[self.key_f]
        queries: List[th.Tensor] = inputs[self.key_q]  # query points

        # Build data structures.
        meshes = Meshes(verts, faces)
        pcls = Pointclouds(queries)
        # assert(pcls.device == meshes.device)

        # packed representation for pointclouds
        points = pcls.points_packed()  # (P, 3)
        points_first_idx = pcls.cloud_to_packed_first_idx()
        max_points = pcls.num_points_per_cloud().max().item()

        # packed representation for faces
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()

        tris = verts_packed[faces_packed]  # (T, 3, 3)
        tris_first_idx = meshes.mesh_to_faces_packed_first_idx()

        # point to face distance: shape (P,)
        outputs = dict(inputs)
        if self.key_g is not None:
            with th.enable_grad():
                points_wg = points.clone().requires_grad_(True)
                distance = th.sqrt_(point_face_distance(
                    points_wg, points_first_idx, tris,
                    tris_first_idx, max_points
                ))
                g = th.autograd.grad(distance, points_wg,
                                     grad_outputs=th.ones_like(distance))
                outputs[self.key_g] = g[0]
        else:
            distance = th.sqrt_(point_face_distance(
                points, points_first_idx, tris,
                tris_first_idx, max_points
            ))
        outputs[self.key_d] = distance
        return outputs


def main():
    device: str = 'cuda'
    device = th.device(device)
    m = th3.utils.ico_sphere(1, device=device)
    transform = DistanceTransform()
    outputs = transform({
        'verts': m.verts_list(),
        'faces': m.faces_list(),
        'points': th.randn(size=(1, 128, 3), device=device)
    })
    print(outputs['distance'].shape)


if __name__ == '__main__':
    main()
