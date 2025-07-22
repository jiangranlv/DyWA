#!/usr/bin/env python3

from itertools import chain
from typing import Callable, Optional, Tuple, Iterable
from functools import cached_property
from env.scene.object_set import ObjectSet


class CombinedObjectSet(ObjectSet):
    def __init__(self, base: Iterable[ObjectSet]):
        self.__bases = list(base)

        self.__bmap = {}
        for (i, b) in enumerate(base):
            # FIXME: watch out for key collision
            for k in b.keys():
                self.__bmap[k] = i
        self.__keys = list(self.__bmap.keys())

    def keys(self):
        return self.__keys

    def _base(self, key: str): return self.__bases[self.__bmap[key]]
    def aabb(self, key: str): return self._base(key).aabb(key)
    def bbox(self, key: str): return self._base(key).bbox(key)
    def cloud(self, key: str): return self._base(key).cloud(key)
    def code(self, key: str): return self._base(key).code(key)
    def hull(self, key: str): return self._base(key).hull(key)
    def label(self, key: str): return self._base(key).label(key)
    def normal(self, key: str): return self._base(key).normal(key)
    def num_faces(self, key: str): return self._base(key).num_faces(key)
    def num_hulls(self, key: str): return self._base(key).num_hulls(key)
    def num_verts(self, key: str): return self._base(key).num_verts(key)
    def obb(self, key: str): return self._base(key).obb(key)
    def pose(self, key: str): return self._base(key).pose(key)
    def radius(self, key: str): return self._base(key).radius(key)
    def urdf(self, key: str): return self._base(key).urdf(key)
    def volume(self, key: str): return self._base(key).volume(key)
