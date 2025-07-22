#!/usr/bin/env python3

from omegaconf import OmegaConf
import torch as th
import itertools
import sys
import pickle
import json
from pathlib import Path
from env.scene.util import _show_stable_pose
from env.scene.filter_object_set import FilteredObjectSet, FilterDims
from env.scene.dgn_object_set import DGNObjectSet
from env.scene.acronym_object_set import AcronymObjectSet
from tqdm.auto import tqdm
import numpy as np
from util.torch_util import dcn
from util.path import ensure_directory
from pytorch3d.ops.sample_farthest_points import sample_farthest_points
from yourdfpy import URDF

from icecream import ic

sys.path.append('../../../../scripts/toy/point_mae/point2vec/point2vec/')
from point2vec.models import Point2Vec

from util.vis.mesh import scene_to_img
import cv2


def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = itertools.cycle(itertools.islice(nexts, pending))


def main():
    filter_complex = True
    filter_dims = (0.05, 0.15, 2.5)  # ???
    # filter_pose_count = (20, 40)
    filter_pose_count = None
    max_vertex_count: int = 8192
    # max_chull_count: int = 128
    max_chull_count: int = 16
    # key_file: str = '/tmp/dgn-keys.json'
    # key_file: str = '/tmp/dgn-keys-v2.json'
    # tmp_file: str = '/tmp/tmp-dgn-keys.json'
    key_file: str = '/tmp/acr-keys.json'
    tmp_file: str = '/tmp/tmp-acr-keys.json'

    dataset = AcronymObjectSet(AcronymObjectSet.Config(
        data_path=F'/input/ACRONYM/meta-v0/'
    ))

    # dataset = DGNObjectSet(DGNObjectSet.Config(
    #     data_path='/input/DGN/meta-v8/',
    # ))

    # if True:
    #    with open('/tmp/all-dgn-keys.json', 'r') as fp:
    #        keys = json.load(fp)
    #    dataset = FilteredObjectSet(dataset, keys=keys)

    # Filter by size, and remove degenerate mesh
    keys = dataset.keys()

    if True:
        need_attr = ['num_verts']
        for attr in need_attr:
            query = getattr(dataset, attr)
            fkeys = []
            for key in keys:
                try:
                    if query(key) is None:
                        continue
                except KeyError:
                    continue
                fkeys.append(key)
            keys = fkeys

    if filter_complex:
        keys = [key for key in keys if
                (dataset.num_verts(key) < max_vertex_count and
                    dataset.num_hulls(key) < max_chull_count)]

    if filter_dims is not None:
        d_min, d_max, r_max = filter_dims
        f = FilterDims(d_min, d_max, r_max)
        keys = [key for key in keys if f(dataset, key)]

    if filter_pose_count is not None:
        pmin, pmax = filter_pose_count
        keys = [key for key in keys if
                dataset.pose(key) is not None and
                pmin <= dataset.pose(key).shape[0] and
                dataset.pose(key).shape[0] < pmax]

    # Round-robin across different classes
    if False:
        cls_map = {}
        icls_map = {}
        for key in keys:
            src, x = key.split('-', 1)
            x, scale = x.rsplit('-', 1)
            a = x.split('-', 1)
            if len(a) == 1:
                # cls = x
                cls = 'misc'
                obj_id = x
            else:
                cls, obj_id = a
            cls_map[key] = cls

            if cls not in icls_map:
                icls_map[cls] = []
            icls_map[cls].append(key)

        for k, l in icls_map.items():
            np.random.shuffle(l)

        keys = []
        for key in roundrobin(*icls_map.values()):
            keys.append(key)
        ic(len(keys))

    # Filter by number of available stable poses
    if False:
        fkey = []
        for key in keys:
            try:
                pose = dataset.pose(key)
            except KeyError:
                continue

            # "pose unavailable"
            if pose is None:
                continue
            num_pose = pose.shape[0]

            # "too unstable"
            if num_pose < 20:
                continue

            # "too stable"
            if num_pose > 40:
                continue
            fkey.append(key)
        keys = fkey
    ic(len(keys))

    dataset = FilteredObjectSet(dataset, keys=keys)

    if False:
        ensure_directory('/tmp/img')
        for key in tqdm(dataset.keys()):
            img = scene_to_img(URDF.load(dataset.urdf(key)).scene)
            cv2.imshow('img', img)
            # cv2.imwrite(
            # F'/tmp/img/{key}.png', cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
            cv2.waitKey(0)

    if False:
        cfg = OmegaConf.load(
            '../../../../scripts/toy/point_mae/point2vec/shapenet.yaml')
        model = Point2Vec(**OmegaConf.to_container(cfg.model))
        ckpt = th.load(
            '../../../../scripts/toy/point_mae/point2vec/pre_point2vec-epoch.799-step.64800.ckpt')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        model.to(device='cuda:0')
        model.eval()

        emb_dir = ensure_directory('/tmp/dgn-emb')

        for key in tqdm(dataset.keys()):
            cloud = dataset.cloud(key).astype(
                np.float32)
            cloud = th.as_tensor(cloud, device='cuda:0')
            cloud = cloud[None]

            # Compute embdding
            cloud -= cloud.mean(dim=-2)
            cloud /= th.linalg.norm(cloud, dim=-1).max(dim=-1).values
            embeddings, centers = model.tokenizer(cloud)
            pos = model.positional_encoding(centers)
            x = model.student(embeddings, pos).last_hidden_state
            x = th.cat([x.max(dim=1).values, x.mean(dim=1)], dim=-1)
            x = x.squeeze(0)

            with open(str(emb_dir / F'{key}.pkl'), 'wb') as fp:
                pickle.dump(x, fp)

    # "First-pass" filter based on embedding-based FPS(??)
    if False:
        if True:
            emb_dir = ensure_directory('/tmp/dgn-emb')
            embs = {}
            for key in dataset.keys():
                with open(str(emb_dir / F'{key}.pkl'), 'rb') as fp:
                    embs[key] = pickle.load(fp)

            keys = sorted(list(embs.keys()))
            embs = [embs[k] for k in keys]
            # emb_keys, emb_values = zip(*embs.items())
            # emb_keys = np.argsort(emb_keys)

            embs = th.stack(embs,
                            # dtype=th.float32,
                            # device='cuda:0'
                            ).to(
                dtype=th.float32,
                device='cuda:0')
            _, indices = sample_farthest_points(embs[None], K=200,)
            indices = dcn(indices.squeeze(0))
            sel_keys = [keys[i] for i in indices]
            with open(tmp_file, 'w') as fp:
                pickle.dump(sel_keys, fp)
        else:
            with open(tmp_file, 'r') as fp:
                sel_keys = pickle.load(fp)
        dataset = FilteredObjectSet(dataset, keys=sel_keys)

    sel_keys = []
    if Path(key_file).is_file():
        with open(key_file, 'r') as fp:
            sel_keys = json.load(fp)
    ic(len(sel_keys))

    try:
        for key in dataset.keys():
            def _on_key(key_char, mod):
                if key_char == 'Y':
                    print(F'append = {key}')
                    sel_keys.append(key)
                if key_char == 'V':
                    print(F'{key} = key')
            poses = dataset.pose(key)
            if poses is None or len(poses) <= 0:
                continue
            if key in sel_keys:
                continue
            _show_stable_pose(URDF.load(dataset.urdf(key)).scene,
                              next(iter(poses)),
                              True,
                              _on_key)
    finally:
        with open(key_file, 'w') as fp:
            json.dump(sel_keys, fp)


def _filter_unique():
    with open('/tmp/all-dgn-keys.json', 'r') as fp:
        keys = json.load(fp)
    with open('/tmp/all-dgn-uniq-keys.json', 'w') as fp:
        json.dump(np.unique(keys).tolist(), fp)


if __name__ == '__main__':
    main()
