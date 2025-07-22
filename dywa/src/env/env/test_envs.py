#!/usr/bin/env python3

from typing import Optional
from env.env.base import EnvIface
import torch as th

from icecream import ic


class A(EnvIface):
    def __init__(self,
                 num_env: int,
                 device: str):
        super().__init__()
        self.num_env = num_env
        self.device = device
        self.obs = th.zeros((num_env, 1),
                            dtype=th.float32,
                            device=self.device)
        self.index = th.zeros((num_env,),
                              dtype=th.int32,
                              device=self.device)
        self.done = th.zeros((num_env,),
                             dtype=th.bool,
                             device=self.device)

    def setup(self):
        pass

    def reset(self):
        self.index.fill_(0)
        self.done.fill_(0)
        return self.obs

    def reset_indexed(self, indices: Optional[th.Tensor]):
        self.index[indices] = 0
        self.done[indices] = False

    def step(self, actions: th.Tensor):
        # step()
        self.index += 1
        # reset()
        self.reset_indexed(self.done)
        # compute_obs()
        self.done[...] = (self.index >= 1)
        rew = th.ones((self.num_env,),
                      dtype=th.float, device=self.device)
        info = {}
        return self.obs, rew, self.done, info


class B(EnvIface):
    def __init__(self,
                 num_env: int,
                 device: str):
        super().__init__()
        self.num_env = num_env
        self.device = device
        self.obs = th.zeros((num_env, 1),
                            dtype=th.float32,
                            device=self.device)
        self.index = th.zeros((num_env,),
                              dtype=th.int32,
                              device=self.device)
        self.done = th.zeros((num_env,),
                             dtype=th.bool,
                             device=self.device)

    def setup(self):
        pass

    def _randomize_obs(self, indices: Optional[th.Tensor] = None):
        if indices is not None:
            self.obs = th.randint(2,
                                  size=(self.num_env, 1),
                                  dtype=th.int32,
                                  device=self.device).float()
        else:
            self.obs[indices] = th.randint(2,
                                           size=(self.num_env, 1),
                                           dtype=th.int32,
                                           device=self.device).float()[indices]

    def reset(self):
        self.index.fill_(0)
        self.done.fill_(0)
        self._randomize_obs()

    def reset_indexed(self, indices: Optional[th.Tensor]):
        self.index[indices] = 0
        self.done[indices] = False
        self._randomize_obs(indices)

    def step(self, actions: th.Tensor):
        # step()
        self.index += 1
        obs0 = self.obs.detach().clone()
        self._randomize_obs()
        # reset()
        self.reset_indexed(self.done)

        # compute_feedback()
        self.done[...] = (self.index >= 1)
        # rew = (self.obs - obs0).squeeze(-1).detach().clone()

        # rew = V(t) - gamma * V(t')
        def _v(obs):
            return obs.squeeze(-1).detach().clone()
        # rew = (obs0 - self.obs).squeeze(-1).detach().clone()
        rew = _v(obs0) - _v(self.obs)

        info = {}
        return self.obs, rew, self.done, info


class C(EnvIface):
    def __init__(self,
                 num_env: int,
                 device: str):
        super().__init__()
        self.num_env = num_env
        self.device = device
        self.obs = th.zeros((num_env, 1),
                            dtype=th.float32,
                            device=self.device)
        self.index = th.zeros((num_env,),
                              dtype=th.int32,
                              device=self.device)
        self.done = th.zeros((num_env,),
                             dtype=th.bool,
                             device=self.device)

    def setup(self):
        pass

    def reset(self):
        self.index.fill_(0)
        self.done.fill_(0)

    def reset_indexed(self, indices: Optional[th.Tensor]):
        self.index[indices] = 0
        self.done[indices] = False

    def step(self, actions: th.Tensor):
        # step()
        self.index += 1

        # reset()
        self.reset_indexed(self.done)

        # compute_feedback()
        self.done[...] = (self.index >= 1)
        if actions is not None:
            sel = th.argmax(actions, dim=-1)
            rew = sel.float()
            # ic(rew)
        else:
            rew = th.zeros((self.num_env,),
                           dtype=th.float,
                           device=self.device)

        info = {}
        return self.obs, rew, self.done, info


class D(EnvIface):
    def __init__(self,
                 num_env: int,
                 device: str):
        super().__init__()
        self.num_env = num_env
        self.device = device
        self.obs = th.zeros((num_env, 1),
                            dtype=th.float32,
                            device=self.device)
        self.index = th.zeros((num_env,),
                              dtype=th.int32,
                              device=self.device)
        self.done = th.zeros((num_env,),
                             dtype=th.bool,
                             device=self.device)

    def setup(self):
        pass

    def _randomize_obs(self, indices: Optional[th.Tensor] = None):
        if indices is not None:
            rnd = th.randint(2,
                             size=(self.num_env, 1),
                             dtype=th.int32,
                             device=self.device).float()
            self.obs[indices] = rnd[indices]
        else:
            self.obs = th.randint(2,
                                  size=(self.num_env, 1),
                                  dtype=th.int32,
                                  device=self.device).float()
        self.obs = self.obs.to(dtype=th.float)

    def reset(self):
        self.index.fill_(0)
        self.done.fill_(0)
        self._randomize_obs()
        return self.obs

    def reset_indexed(self, indices: Optional[th.Tensor]):
        self.index[indices] = 0
        self._randomize_obs(indices)
        self.done[indices] = False

    def step(self, actions: th.Tensor):
        # step()
        if actions is not None:
            # det = (actions > 0.0)
            det = th.argmax(actions, dim=-1)
            # print(det.max(), det.min())
            det = det.to(dtype=th.bool)
            # print('obs', self.obs.shape)
            # print('det', det.shape)
            # action = "flip obs"
            # print(self.done[0].ravel())
            # print(self.obs[0].ravel())
            # print(det[0].ravel())
            self.obs[det] = 1.0 - self.obs[det]
            # print(self.obs[0].ravel())
            # so the correct action is:
            # obs=0 -> act=1
            # obs=1 -> act=0
        self.index += 1

        # reset()
        self.reset_indexed(self.done)

        # compute_feedback()
        self.done[...] = (self.index >= 1)
        if actions is not None:
            rew = (self.done.float()) * self.obs.float().squeeze(-1)
        else:
            rew = th.zeros((self.num_env,),
                           dtype=th.float,
                           device=self.device)

        info = {}
        return self.obs, rew, self.done, info


class E(EnvIface):
    def __init__(self,
                 num_env: int,
                 device: str):
        super().__init__()
        self.num_env = num_env
        self.device = device
        self.obs = th.zeros((num_env, 1),
                            dtype=th.float32,
                            device=self.device)
        self.index = th.zeros((num_env,),
                              dtype=th.int32,
                              device=self.device)
        self.done = th.zeros((num_env,),
                             dtype=th.bool,
                             device=self.device)

    def setup(self):
        pass

    def reset(self):
        self.index.fill_(0)
        self.done.fill_(0)
        self.obs.fill_(0)

    def reset_indexed(self, indices: Optional[th.Tensor]):
        self.index[indices] = 0
        self.done[indices] = False
        self.obs[indices] = 0

    def step(self, actions: th.Tensor):
        # step()
        if actions is not None:
            print(self.done[0].ravel())
            print(self.obs[0].ravel())
            print(actions[0].ravel())
            self.obs += actions
            print(self.obs[0].ravel())
        self.index += 1

        # reset()
        self.reset_indexed(self.done)

        # compute_feedback()
        self.done[...] = (self.index >= 1)
        if actions is not None:
            rew = th.greater(self.obs, 0.5).float().squeeze(-1)
        else:
            rew = th.zeros((self.num_env,),
                           dtype=th.float,
                           device=self.device)

        info = {}
        return self.obs, rew, self.done, info
