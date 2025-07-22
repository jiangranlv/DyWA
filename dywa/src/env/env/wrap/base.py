#!/usr/bin/env python3

from abc import abstractproperty

from typing import Optional, Iterable, Callable, Tuple

from env.env.iface import EnvIface

import torch as th
from gym import spaces
from functools import partial


def _create_new(old, new, key: str):
    return {key: new}


def _update_as_child(old, new, key: str):
    return {**old, key: new}


def _update_as_sibling(old, new, key: str):
    return {'raw': old, key: new}

def _remove_key(old, key: str):
    """
    Remove the specified key from the old dictionary.
    """
    if key in old:
        return {k: v for k, v in old.items() if k != key}
    return old  # 如果 key 不在字典中，返回原字典


def add_obs_field(
        old_space: spaces.Space,
        key: str,
        item: spaces.Space,
        key_old: str = 'raw'
) -> Tuple[spaces.Dict, Callable]:
    """
    add a field to an observation space.
    """
    if old_space is None:
        obs_space = spaces.Dict({key: item})
        update_fn = partial(_create_new, key=key)
        return obs_space, update_fn

    make_dict: bool = (not isinstance(old_space, spaces.Dict))
    if make_dict:
        # Create a new dictionary-based observation space.
        obs_space = spaces.Dict({
            key_old: old_space,
            key: item
        })
        update_fn = partial(_update_as_sibling, key=key)
    else:
        # Let's try to prevent nested mappings, if possible...
        obs_space = spaces.Dict({
            **old_space.spaces,
            key: item,
        })
        update_fn = partial(_update_as_child, key=key)

    return obs_space, update_fn

def remove_obs_field(
        old_space: spaces.Space,
        key: str
) -> Tuple[spaces.Space, Callable]:
    """
    Remove a field from an observation space.
    """
    if old_space is None or key not in old_space.spaces:
        return old_space, lambda obs: obs  # 如果没有该字段，返回原空间和无操作函数

    # Create a new dictionary-based observation space without the specified key
    new_spaces = {k: v for k, v in old_space.spaces.items() if k != key}
    
    # Handle the case where we may need to return a dictionary
    if len(new_spaces) == 0:
        obs_space = None  # If no fields remain, return None
        update_fn = lambda obs: obs  # No updates needed
    else:
        obs_space = spaces.Dict(new_spaces)
        update_fn = partial(_remove_key, key=key)

    return obs_space, update_fn



class WrapperEnv(EnvIface):
    def __init__(self, env: EnvIface):
        super().__init__()
        self.env = env

    @property
    def num_env(self):
        return self.env.num_env

    @property
    def device(self):
        return self.env.device

    @property
    def timeout(self):
        return self.env.timeout

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def setup(self):
        return self.env.setup()

    def reset(self):
        return self.env.reset()

    def step(self, actions: th.Tensor):
        return self.env.step(actions)

    # def pre_physics_step(self, actions: th.Tensor):
    #     return self.env.pre_physics_step(actions)

    # def physics_step(self):
    #     return self.env.physics_step()

    # def post_physics_step(self):
    #     return self.env.post_physics_step()

    def reset_indexed(self, indices: Optional[Iterable[int]] = None):
        return self.env.reset_indexed(indices)

    def apply_actions(self, actions: th.Tensor):
        """ (act) -> () """
        return self.env.apply_actions(actions)

    def compute_observations(self):
        """ () -> (obs) """
        return self.env.compute_observations()

    def compute_feedback(self):
        """ (obs,act) -> (rew,done,info) """
        return self.env.compute_feedback()

    def unwrap(self, target=None):
        if isinstance(self, target):
            return self
        env = self.env
        if (target is not None) and isinstance(env, target):
            return env
        if isinstance(env, WrapperEnv):
            return env.unwrap(target)
        return env

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)


class ObservationWrapper(WrapperEnv):
    def __init__(self, env: EnvIface,
                 obs_wrapper: Callable[[th.Tensor], th.Tensor]):
        super().__init__(env)
        self.env = env
        self.obs_wrapper = obs_wrapper

    @abstractproperty
    def observation_space(self):
        return

    @property
    def action_space(self):
        return self.env.action_space

    def setup(self):
        return self.env.setup()

    def reset(self):
        return self.obs_wrapper(self.env.reset())

    def reset_indexed(self, indices: Optional[th.Tensor]):
        return self.env.reset_indexed(indices)

    def step(self, actions: th.Tensor):
        obs, rew, done, info = self.env.step(actions)
        w_obs = self.obs_wrapper(obs)
        return (w_obs, rew, done, info)

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)


class ActionWrapper(WrapperEnv):
    def __init__(self, env: EnvIface,
                 act_wrapper: Callable[[th.Tensor], th.Tensor]):
        super().__init__(env)
        self.env = env
        self.act_wrapper = act_wrapper

    @property
    def observation_space(self):
        return self.env.observation_space

    @abstractproperty
    def action_space(self):
        pass

    def setup(self):
        return self.env.setup()

    def reset(self):
        return self.env.reset()

    def reset_indexed(self, indices: Optional[th.Tensor]):
        return self.env.reset_indexed(indices)

    def step(self, actions: th.Tensor):
        w_actions = self.act_wrapper(actions)
        return self.env.step(w_actions)

    def __getattr__(self, attr):
        """ delegate missing attributes to self.env. """
        return getattr(self.env, attr)
