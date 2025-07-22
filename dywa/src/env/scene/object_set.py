#!/usr/bin/env python3

from typing import List, Tuple
from abc import ABC, abstractmethod
import numpy as np
import trimesh


class ObjectSet(ABC):
    def __init__(self):
        pass

    def __len__(self):
        return len(self.keys())

    @abstractmethod
    def keys(self) -> List[str]:
        """ keys for indexing this object set """
        pass

    @abstractmethod
    def label(self, key: str) -> str:
        """ Category of this object """
        pass

    @abstractmethod
    def urdf(self, key: str) -> str:
        """ Robot URDF files """
        pass

    @abstractmethod
    def pose(self, key: str) -> np.ndarray:
        """ available stable poses on a flat plane. """
        pass

    @abstractmethod
    def code(self, key: str) -> np.ndarray:
        """ Shape embdding, assumed to be computed in canonical pose """
        pass

    @abstractmethod
    def cloud(self, key: str) -> np.ndarray:
        """ Surface point cloud """
        pass

    @abstractmethod
    def normal(self, key: str) -> np.ndarray:
        """ Per-point normals corresponding to surface point cloud. """
        pass

    @abstractmethod
    def bbox(self, key: str) -> np.ndarray:
        """ AABB corners """
        pass

    @abstractmethod
    def aabb(self, key: str) -> np.ndarray:
        """ AABB bounds """
        pass

    @abstractmethod
    def obb(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oriented bounding box bounds.
        For now, obb is returned as (4x4 transform, scale)
        """
        pass

    @abstractmethod
    def hull(self, key: str) -> trimesh.Trimesh:
        """ Convex hull """
        pass

    @abstractmethod
    def radius(self, key: str) -> float:
        """ Radius """
        pass

    @abstractmethod
    def volume(self, key: str) -> float:
        """ Volume """
        pass

    @abstractmethod
    def num_verts(self, key: str) -> float:
        """ Num vertices in mesh """
        pass

    @abstractmethod
    def num_faces(self, key: str) -> float:
        """ Num triangle faces """
        pass

    @abstractmethod
    def num_hulls(self, key: str) -> float:
        """ Num convex hulls """
        pass
