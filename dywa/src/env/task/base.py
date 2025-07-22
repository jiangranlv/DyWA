#!/usr/bin/env python3

from typing import Tuple, Dict
from abc import ABC, abstractmethod, abstractproperty


class TaskBase(ABC):

    @abstractproperty
    def timeout(self):
        pass

    @abstractmethod
    def create_assets(self, *args, **kwds):
        return {}

    @abstractmethod
    def create_actors(self, *args, **kwds):
        return {}

    @abstractmethod
    def create_sensors(self, *args, **kwds):
        return {}

    @abstractmethod
    def compute_feedback(self, *args, **kwds) -> Tuple[float, bool, Dict]:
        pass
