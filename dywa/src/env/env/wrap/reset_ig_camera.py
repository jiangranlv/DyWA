#!/usr/bin/env python3

from isaacgym import gymapi

from typing import Tuple

from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv


def reset_ig_camera(env: EnvIface,
                    target: Tuple[float, float, float] = (0.0, 0.0, 0.4),
                    offset: Tuple[float, float, float] = (2.0, 2.0, 3.0)):
    if env.viewer is None:
        return
    target = gymapi.Vec3(target[0], target[1], target[2])

    source = gymapi.Vec3()
    source.x = target.x + offset[0]
    source.y = target.y + offset[1]
    source.z = target.z + offset[2]
    if False:
        # This code does not actually work due to IG bug
        camera = env.gym.get_viewer_camera_handle(env.viewer)
        env.gym.set_camera_location(camera,
                                    env.envs[0],
                                    source,
                                    target)
    else:
        env.gym.viewer_camera_look_at(env.viewer,
                                      env.envs[0],
                                      source, target)
