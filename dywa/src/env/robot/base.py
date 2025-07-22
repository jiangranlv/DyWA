#!/usr/bin/env python3

from abc import ABC, abstractmethod


class RobotBase(ABC):

    @abstractmethod
    def create_assets(self):
        pass

    @abstractmethod
    def create_actors(self):
        pass

    @abstractmethod
    def reset(self, gym, sim, env):
        """ Reset the _intrinsic states_ of the robot. """
        pass

    @abstractmethod
    def step_controller(self, gym, sim, env):
        pass

    @abstractmethod
    def apply_actions(self, gym, sim, env, actions):
        """ Set the actuation targets for the simulator. """
        pass
