#!/usr/bin/env python3

import numpy as np
import torch as th
import einops
from typing import Optional, Dict

from dgl.ops import segment_mm, segment_reduce
from util.math_util import matrix_from_pose
from util.torch_util import dcn

from icecream import ic
import nvtx
from torch_scatter import segment_min_csr, segment_max_csr

BLOCK_SIZE: int = 32

# @th.jit.script


def reduce_hits(t, front, data_mask, block_bound, num_hulls):
    # Count number of blocks occupied per hull.
    # num_blocks = (seg_lens + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Fill in invalid data
    with nvtx.annotate("find-ignore"):
        ignore = th.isnan(t) | ~data_mask[..., None]
    with nvtx.annotate("fill_invalid-hi"):
        # t_hi = th.masked_fill(t, front | ignore, th.inf)
        t_hi = th.where(front | ignore,
                        th.as_tensor(th.inf, dtype=t.dtype,
                                     device=t.device), t)
    with nvtx.annotate("fill_invalid-lo"):
        # t_lo = th.masked_fill(t, ~front | ignore, -th.inf)
        t_lo = th.where(~front | ignore,
                        th.as_tensor(-th.inf, dtype=t.dtype,
                                     device=t.device), t)

    # reduction within hull
    with nvtx.annotate("reduce_intra_hull"):
        t_lo = t_lo.view(-1, *t_lo.shape[2:])
        t_hi = t_hi.view(-1, *t_hi.shape[2:])
        # t_near = th.segment_reduce(t_lo, 'max',
        #                           lengths=num_blocks * BLOCK_SIZE)
        # t_far = th.segment_reduce(t_hi, 'min',
        #                          lengths=num_blocks * BLOCK_SIZE)
        t_near = segment_max_csr(t_lo,
                                 # th.cumsum(num_blocks * BLOCK_SIZE)
                                 block_bound)[0]
        t_far = segment_min_csr(t_hi,
                                # th.cumsum(num_blocks * BLOCK_SIZE)
                                block_bound)[0]
        # ic(t_near.shape)
        # ic(t_far.shape)  # 27116,512 ?
        # ic(num_hulls.sum()) # 27117???

    # reduction across hulls
    with nvtx.annotate("reduce_inter_hull"):
        valid = (t_near > 0.0) & (t_far + 1e-6 >= t_near)
        t_near.masked_fill_(~valid, th.inf)
        min_dist = th.segment_reduce(t_near, 'min', lengths=num_hulls)

    return min_dist


def pad_and_split_array(x: th.Tensor,
                        seg_len: th.Tensor,
                        block_size: int = BLOCK_SIZE):
    input_index: int = 0
    # output_index: int = 0

    # integer space ceil()
    num_blocks = th.sum((seg_len + block_size - 1) // block_size)

    y = th.zeros((num_blocks, block_size, *x.shape[1:]),
                 dtype=x.dtype,
                 device=x.device)
    m = th.zeros((num_blocks, block_size,),
                 dtype=bool,
                 device=x.device)
    h = th.zeros((num_blocks, block_size,),
                 dtype=th.long,
                 device=x.device)
    qs = []
    block_index = 0
    for q, s in enumerate(seg_len):
        for i in range(0, s, block_size):
            d = min(block_size, s - i)
            y[block_index, :d] = x[input_index:input_index + d]
            m[block_index, :d] = 1
            h[block_index, :d] = q
            input_index += d
            # output_index += block_size
            block_index += 1
            qs.append(q)
    qs = th.as_tensor(np.asarray(qs, dtype=np.int32),
                      dtype=th.long,
                      device=x.device)
    return y, m, qs, h.reshape(-1)


# def block_segment_mm(block_size:int=64):
#     gather(x, block
#     num_blocks = (seglen + block_size - 1) // block_size

@nvtx.annotate("hull_param_to_coords")
def hull_param_to_coords(h: th.Tensor):
    coords = th.zeros((*h.shape[:-1], 2, 4),
                      dtype=h.dtype,
                      device=h.device)
    # origin is a point
    coords[..., 0, :3] = h[..., :3]
    coords[..., 0, 3] = 1

    # normal is a vector
    coords[..., 1, :3] = h[..., 3:6]
    coords[..., 1, 3] = 0
    return coords


@nvtx.annotate("vectorized_raycast")
def vectorized_raycast(
        base_hulls,
        body_hull_ids,
        env_body_ids,
        ray_origin,
        ray_vector,
        body_poses: th.Tensor,

        # all_hulls: Optional[th.Tensor] = None
        # all_pose_ids: Optional[th.Tensor] = None
        aux: Dict[str, th.Tensor] = None
):
    num_env = len(env_body_ids)

    all_hulls = aux.get('all_hulls', None)
    all_hull_ids_flat = aux.get('all_hull_ids_flat', None)

    with nvtx.annotate("all_hulls"):
        if all_hulls is None:
            # ic(len(body_hull_ids))
            # ic(env_body_ids)
            all_hull_ids = [[body_hull_ids[body_id]
                            for _, body_id in env_body_ids[env_id]]
                            for env_id in range(num_env)]
            # print(all_hull_ids)
            # print([h.shape for h in all_hull_ids])
            env_hull_ids = [np.concatenate(h) for h in all_hull_ids]
            # env_hull_ids = [h for h in all_hull_ids]
            all_hull_ids_flat = np.concatenate(env_hull_ids)

            all_hulls = th.cat([base_hulls[h]
                                for h in all_hull_ids_flat])
            aux['all_hulls'] = all_hulls
            aux['all_hull_ids_flat'] = all_hull_ids_flat
            aux['env_hull_ids'] = env_hull_ids
        all_hulls = aux.get('all_hulls', None)
        all_hull_ids_flat = aux.get('all_hull_ids_flat', None)
        env_hull_ids = aux.get('env_hull_ids', None)

    # Indexing and segment logic on the CPU
    with nvtx.annotate("segment_lengths"):
        if 'num_hulls' not in aux:
            # seg_lens = number of triangles per hull
            seg_lens = [len(base_hulls[h]) for h in all_hull_ids_flat]
            num_hulls = [len(e) for e in env_hull_ids]

            num_hulls_cpu = th.as_tensor(num_hulls,
                                         device='cpu',
                                         dtype=th.long)
            num_hulls = num_hulls_cpu.to(ray_origin.device)

            seg_lens_cpu = th.as_tensor(seg_lens,
                                        device='cpu',
                                        dtype=th.long)
            ic(seg_lens_cpu)  # maybe a better block size
            seg_lens = seg_lens_cpu.to(ray_origin.device)

            aux['num_hulls'] = num_hulls
            aux['num_hulls_cpu'] = num_hulls_cpu
            aux['seg_lens'] = seg_lens
            aux['seg_lens_cpu'] = seg_lens_cpu

        num_hulls = aux.get('num_hulls')
        num_hulls_cpu = aux.get('num_hulls_cpu')
        seg_lens = aux.get('seg_lens')
        seg_lens_cpu = aux.get('seg_lens_cpu')

        # Doesn't really need "segment_reduce" but anyways...
        # num_tri_per_env = segment_reduce(
        #     num_hulls_cpu,
        #     seg_lens_cpu.float(),
        #     'sum').long().cpu()
        # num_tri_per_env = th.segment_reduce(
        #     seg_lens_cpu.float(),
        #     'sum',
        #     lengths=num_hulls_cpu,
        # ).long().cpu()

    # Somehow retrieve `all_poses` from somewhere
    with nvtx.annotate("all_poses"):
        all_pose_ids = aux.get('all_pose_ids', None)
        if all_pose_ids is None:
            all_pose_ids = []
            for env_id in range(num_env):
                for (pose_index, body_id) in env_body_ids[env_id]:
                    for _ in range(len(body_hull_ids[body_id])):
                        all_pose_ids.append([env_id, pose_index])
            all_pose_ids = np.stack(all_pose_ids, axis=0)
            all_pose_ids = th.as_tensor(all_pose_ids,
                                        dtype=th.long,
                                        device=ray_origin.device)
            aux['all_pose_ids'] = all_pose_ids
            all_poses = body_poses[all_pose_ids[..., 0],
                                   all_pose_ids[..., 1]]
            # all_poses = [th.cat([
            #    # Duplicate `body_pose` by number of hulls per body
            #    einops.repeat(body_poses[env_id, pose_index], '... -> h ...',
            #                  h=len(body_hull_ids[body_id]))
            #    for (pose_index, body_id) in env_body_ids[env_id]])
            #    for env_id in range(num_env)]
            # all_poses = th.cat(all_poses, axis=0)
        else:
            all_poses = body_poses[all_pose_ids[..., 0],
                                   all_pose_ids[..., 1]]

    # 2, 11, 17
    # num_envs X num_body_per_env X body_sizes
    # this seems correct...?

    with nvtx.annotate("convert_pose"):
        all_poses_h = matrix_from_pose(all_poses[..., :3],
                                       all_poses[..., 3:7]).swapaxes(-1, -2)

    data_mask = None
    with nvtx.annotate("transform_hull"):
        if True:
            if 'hull_blocks_h' not in aux:
                all_hulls_h = hull_param_to_coords(all_hulls)
                hull_blocks_h, data_mask, q, hs = pad_and_split_array(
                    all_hulls_h, seg_lens_cpu, block_size=BLOCK_SIZE)
                aux['hull_blocks_h'] = hull_blocks_h
                aux['data_mask'] = data_mask
                aux['q'] = q
                aux['hs'] = hs

            hull_blocks_h = aux.get('hull_blocks_h')
            data_mask = aux.get('data_mask')
            q = aux.get('q')

            # by the way,
            # s = number of blocks
            # b = block size
            # t = two
            # i = data dimension (3/4)
            transformed_hull_blocks_h = th.einsum('...sbti, ...sij -> ...sbtj',
                                                  hull_blocks_h,
                                                  all_poses_h[q])
            # print(transformed_hull_blocks_h.shape, data_mask.shape)
            # transformed_hull_blocks_h = transformed_hull_blocks_h[data_mask]
            origins_h = transformed_hull_blocks_h[..., 0, :3]
            normals_h = transformed_hull_blocks_h[..., 1, :3]

            # origins_h2 = segment_mm(all_hulls_h[:, 0],
            #                         all_poses_h,
            #                         seg_lens_cpu)
            # normals_h2 = segment_mm(all_hulls_h[:, 1],
            #                         all_poses_h,
            #                         seg_lens_cpu)
            # print( (origins_h[...,:3] - origins_h2[...,:3]).std())
            # print( (normals_h[...,:3] - normals_h2[...,:3]).std())
        else:
            # print(all_hulls_h.shape) # 8160,2,4 (not ok)
            # print(all_poses_h.shape) # 229,4,4 (ok)
            # print(seg_lens)
            if False:
                origins_h = segment_mm(all_hulls_h[:, 0],
                                       all_poses_h,
                                       seg_lens_cpu)
                normals_h = segment_mm(all_hulls_h[:, 1],
                                       all_poses_h,
                                       seg_lens_cpu)
            else:
                # print(all_hulls_h[:, 0].shape)  # 3144842, 4
                # print(all_poses_h.shape)  # 54217 , 4, 4
                origins_h = th.einsum('...fi, ...ij -> ...fj',
                                      all_hulls_h[:, 0],
                                      all_poses_h)
                normals_h = th.einsum('...fi, ...ij -> ...fj',
                                      all_hulls_h[:, 1],
                                      all_poses_h)
    # origins_h = hulls_h[:, 0]
    # normals_h = hulls_h[:, 1]

    with nvtx.annotate("compute_offset"):
        d = th.einsum('...fi, ...fi -> ...f',
                      normals_h[..., :3],
                      origins_h[..., :3])

    if data_mask is None:
        with nvtx.annotate("compute_denom"):
            # T = th.as_tensor
            vd = segment_mm(
                normals_h[..., :3],
                ray_vector.swapaxes(-1, -2),
                num_tri_per_env)

        with nvtx.annotate("compute_numer"):
            # Numerator...
            vn = d[..., None] - segment_mm(
                normals_h[..., :3],
                ray_origin.swapaxes(-1, -2),
                num_tri_per_env)
    else:
        # Denominator
        # ic(all_poses_h.shape)  # 229,4,4
        # ic(normals_h.shape)  # 306,64, 3
        # ic(ray_vector.shape)  # 16,512,3

        if 'q2' not in aux:
            block_size = BLOCK_SIZE
            num_blocks_per_env = th.segment_reduce(
                ((seg_lens_cpu + block_size - 1) // block_size).float(),
                'sum',
                lengths=num_hulls_cpu
            ).long().cpu()  # e.g. (5,5,3,...)
            num_blocks_per_env = dcn(num_blocks_per_env)
            repeats = []
            for index, count in enumerate(num_blocks_per_env):
                repeats += [index] * count
            repeats = np.asarray(repeats)
            q2 = th.as_tensor(repeats,
                              dtype=th.long,
                              device=ray_origin.device)
            aux['q2'] = q2
        q2 = aux.get('q2')
        # ic(normals_h.shape, ray_vector[q2].shape)

        # Denominator
        with nvtx.annotate("vd"):
            vd = th.einsum('...sbi, ...sri -> ...sbr',
                           normals_h[..., :3], ray_vector[q2])
        # Numerator
        with nvtx.annotate("vn"):
            vn = (d[..., None] - th.einsum('...sbi, ...sri -> ...sbr',
                                           normals_h[..., :3], ray_origin[q2]))

    with nvtx.annotate("compute_distance"):
        t = vn / vd
        # print('t', t.shape, data_mask.shape,
        #       data_mask.dtype)
        # t = t[data_mask]

    with nvtx.annotate("hit_reasoning"):
        front = (vd < 0.0)
        if 'block_bound' not in aux:
            num_blocks = (seg_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
            block_bound = th.zeros(
                len(num_blocks) + 1,
                dtype=th.long,
                device=num_blocks.device)
            th.cumsum(num_blocks * BLOCK_SIZE, 0,
                      dtype=th.long, out=block_bound[1:])
            aux['block_bound'] = block_bound
        block_bound = aux.get('block_bound')
        return reduce_hits(t, front, data_mask, block_bound, num_hulls)

        # with nvtx.annotate("masked_fill"):
        #    nan_mask = th.isnan(t)
        #    t_lo = th.where(~front | nan_mask,
        #                    th.as_tensor(-th.inf,
        #                                 dtype=t.dtype,
        #                                 device=t.device),
        #                    t)  # used against t_near
        #    t_hi = th.where(front | nan_mask,
        #                    th.as_tensor(+th.inf,
        #                                 dtype=t.dtype,
        #                                 device=t.device),
        #                    t)  # used against t_far

        with nvtx.annotate("masked_fill"):
            nan_mask = th.isnan(t)
            t_lo = th.masked_fill(t, ~front | nan_mask, -th.inf)
            t_hi = th.masked_fill(t, front | nan_mask, th.inf)

        # t_near = segment_reduce(seg_lens, t_lo, 'max')
        # t_far = segment_reduce(seg_lens, t_hi, 'min')

        # t_near = th.segment_reduce(t_lo, 'max', lengths=seg_lens)
        # t_far = th.segment_reduce(t_hi, 'min', lengths=seg_lens)

        if False:
            with nvtx.annotate("index_reduce"):
                with nvtx.annotate("rearrange"):
                    t_lo_flat = einops.rearrange(t_lo,
                                                 '... s b r -> ... (s b) r')
                    t_hi_flat = einops.rearrange(t_hi,
                                                 '... s b r -> ... (s b) r')
                    hull_index = aux.get('hs')

                # sbr
                with nvtx.annotate("create-dummy-zeros"):
                    dummy = th.zeros((len(seg_lens),
                                      t_lo_flat.shape[-1]),
                                     dtype=th.float,
                                     device=hull_index.device)
                with nvtx.annotate("index_reduce_inner"):
                    t_near = th.index_reduce(
                        dummy, 0, hull_index, t_lo_flat, 'amax',
                        include_self=False)
                    t_far = th.index_reduce(
                        dummy, 0, hull_index, t_hi_flat, 'amin',
                        include_self=False)  # 229, 512 ?
                # t_near = th.scatter_reduce(dummy, 0, hull_index, t_lo, 'amax',
                #                          include_self=False)
                # t_far = th.scatter_reduce(dummy, 0, hull_index, t_hi, 'amin',
                #                         include_self=False) # 229, 512 ?
                # ic(t_near.shape)
                # ic(t_far.shape)
                # th.index_reduce(t_lo,0,hull_index,source?,'max')
        elif False:
            with nvtx.annotate("apply_mask"):
                t_lo = t_lo[data_mask]
                t_hi = t_hi[data_mask]
            with nvtx.annotate("segment_reduce"):
                t_near = th.segment_reduce(t_lo, 'max', lengths=seg_lens)
                t_far = th.segment_reduce(t_hi, 'min', lengths=seg_lens)
        else:
            num_blocks = (seg_lens + BLOCK_SIZE - 1) // BLOCK_SIZE
            t_near = th.segment_reduce(t_lo, 'max', lengths=num_blocks)
            t_near = t_near.max(dim=1).values
            t_far = th.segment_reduce(t_hi, 'min', lengths=num_blocks)
            t_far = t_far.min(dim=1).values
            print(t_near.shape)  # 9, 512
            print(t_far.shape)  # 9, 512

        # print(t_near, t_near2)
        # print(t_far, t_far2)

        # [1] distance
        dist = t_near
        valid = (t_near > 0.0) & (t_far + 1e-6 >= t_near)

    # [2] reduction over bodies
    with nvtx.annotate("compute_min"):
        dist[~valid] = th.inf
        min_dist = segment_reduce(num_hulls, dist, 'min')
        # min_dist = th.segment_reduce(dist, 'min', lengths=num_hulls)
    return min_dist
