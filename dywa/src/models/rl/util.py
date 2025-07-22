#!/usr/bin/env python3

from typing import Optional
from tqdm.auto import tqdm

import numpy as np

import torch as th


from util.torch_util import (
    dcn)


def explained_variance(y_pred: th.Tensor, y_true: th.Tensor) -> th.Tensor:
    v_pred = th.var(y_true)
    return 1.0 - th.var(y_true - y_pred) / v_pred


def mixed_reset(env,
                num_env: int,
                device: th.device,
                timeout: Optional[int] = None,
                num_steps: Optional[int] = None
                ):
    """
    Set N envs to start from non-init states
    at different stages of progression.

    Args:
        env:       Vectorized environments to reset.
        num_env:   Total number of environments.
        device:    The device that envs would accept.
        timeout:   Maximum number of timesteps per episode.
        num_steps: Number of steps to run the resets for.
    """
    if num_steps is None:
        if timeout is None:
            num_steps = num_env
        else:
            num_steps = timeout
    # Number of envs to reset per step.
    n_reset: int = max(1, num_env // num_steps) ### env / 300
    # The Reset interval.
    stride: int = max(1, num_steps // num_env) ### 300 / env
    for i in tqdm(range(num_steps), desc='mixed_reset'):
        if i % stride == 0:
            indices = th.randint(num_env, size=(n_reset,),
                                 device=device) ### 随机选一些reset
            # FIXME:
            # `reset_indexed` results in double resets,
            # so we indirectly work around this by
            # overwriting the `done` buffer instead.
            # env.reset_indexed(indices=indices)
            if hasattr(env, 'buffers'):
                env.buffers['done'][indices] = 1

        if hasattr(env, 'action_space'):
            actions = np.stack([env.action_space.sample()
                                for _ in range(num_env)])
            actions = th.as_tensor(actions, device=env.device)
        else:
            actions = None
        obs, _, _, _ = env.step(actions)
    return obs


def test_mixed_reset():
    from matplotlib import pyplot as plt

    class DummyEnv:
        def __init__(self, num_env: int):
            self.index = th.zeros((num_env,),
                                  dtype=th.int32)
            self.reset = None
            self.buffers = {'done': th.zeros((num_env,),
                                             dtype=th.bool)}

        def reset_indexed(self, indices):
            self.reset = indices

        def step(self, action):
            self.index += 1
            self.index[self.buffers['done']] = 0
            self.buffers['done'] = (self.index >= timeout)
            return None, None, self.buffers['done'], {}

    num_env: int = 1024
    # num_steps: int = 128
    num_steps: int = None
    # timeout: int = 512
    # timeout: int = None
    timeout: int = 256

    env = DummyEnv(num_env)
    obs = mixed_reset(env,
                      num_env,
                      th.device('cpu'),
                      timeout,
                      num_steps)
    print(dcn(env.index))
    plt.hist(dcn(env.index), bins=64)
    plt.show()


def main():
    test_mixed_reset()


if __name__ == '__main__':
    main()
