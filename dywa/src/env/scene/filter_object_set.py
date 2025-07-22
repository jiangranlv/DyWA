#!/usr/bin/env python3

import numpy as np
from typing import Callable, Optional, Tuple, Iterable
from functools import cached_property
from env.scene.object_set import ObjectSet


def dataset_hasattr(object_set: ObjectSet, attr: str, key: str):
    if not hasattr(object_set, attr):
        raise KeyError(F'Unknown key = {key}')
    query = getattr(object_set, attr)
    try:
        if query(key) is None:
            return False
    except KeyError:
        return False
    return True


def apply_filter_hasattr(
        object_set: ObjectSet,
        attrs: Iterable[str],
        keys: Optional[Iterable[str]] = None
) -> Iterable[str]:

    if keys is None:
        keys = list(object_set.keys())

    for attr in attrs:
        query = getattr(object_set, attr)
        for key in keys:
            try:
                if query(key) is None:
                    continue
            except KeyError:
                continue
            fkeys.append(key)
        keys = fkeys

    return keys


class FilteredObjectSet(ObjectSet):
    def __init__(self,
                 base: ObjectSet,
                 filter_fn: Optional[Callable[[str], bool]] = None,
                 keys: Optional[Tuple[str, ...]] = None):
        self.__base = base
        if keys is None:
            keys = [k for k in base.keys() if filter_fn(k)]
        self.__keys = keys

    def __len__(self):
        return len(self.__keys)

    def keys(self):
        return self.__keys

    @cached_property
    def poses(self):
        return {k: self.__base.pose(k)
                for k in self.keys}

    @cached_property
    def codes(self):
        return {k: self.__base.code(k)
                for k in self.keys}

    @cached_property
    def clouds(self):
        return {k: self.__base.cloud(k)
                for k in self.keys}

    @cached_property
    def normals(self):
        return {k: self.__base.normal(k)
                for k in self.keys}

    @cached_property
    def hulls(self):
        return {k: self.__base.hull(k)
                for k in self.keys}

    def aabb(self, key: str): return self.__base.aabb(key)
    def bbox(self, key: str): return self.__base.bbox(key)
    def cloud(self, key: str): return self.__base.cloud(key)
    def code(self, key: str): return self.__base.code(key)
    def hull(self, key: str): return self.__base.hull(key)
    def label(self, key: str): return self.__base.label(key)
    def normal(self, key: str): return self.__base.normal(key)
    def num_faces(self, key: str): return self.__base.num_faces(key)
    def num_hulls(self, key: str): return self.__base.num_hulls(key)
    def num_verts(self, key: str): return self.__base.num_verts(key)
    def obb(self, key: str): return self.__base.obb(key)
    def pose(self, key: str): return self.__base.pose(key)
    def radius(self, key: str): return self.__base.radius(key)
    def urdf(self, key: str): return self.__base.urdf(key)
    def volume(self, key: str): return self.__base.volume(key)
    def predefined_goal(self, key: str): return self.__base.predefined_goal(key)


class FilterDims:
    """ Avoid thin or flat geometries. """

    def __init__(self,
                 d_min: float,
                 d_max: float,
                 r_max: float):
        self.d_min = d_min
        self.d_max = d_max
        self.r_max = r_max

    def __extent_obb(self, obj_set: ObjectSet, key: str):
        _, extent = obj_set.obb(key)
        return extent

    def __extent_bbox(self, obj_set: ObjectSet, key: str):
        bbox = obj_set.bbox(key)
        extent = bbox.max(axis=0) - bbox.min(axis=0)
        return extent

    def __extent_pca(self, obj_set: ObjectSet, key: str):
        cloud = obj_set.cloud(key)
        extent = (cloud @ np.linalg.eig(np.cov(cloud.T))[1]).ptp(axis=0)
        return extent

    def __call__(self, obj_set: ObjectSet, key: str) -> bool:
        extent = self.__extent_obb(obj_set, key)
        vmin, vmax = np.min(extent), np.max(extent)
        return (vmin >= self.d_min and
                vmax < self.d_max and
                vmax < vmin * self.r_max)
