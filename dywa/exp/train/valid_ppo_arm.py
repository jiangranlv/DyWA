#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil

from typing import Optional, Mapping
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, replace
from models.common import map_struct
from env.env.wrap.base import WrapperEnv
from env.env.wrap.draw_inertia_box import DrawInertiaBox
from env.env.wrap.record_viewer import RecordViewer
from env.env.wrap.nvdr_record_viewer import NvdrRecordViewer
from env.env.wrap.nvdr_record_episode import NvdrRecordEpisode
from envs.cube_env_wrappers import DrawGoalPose, DrawObjPose
from env.env.wrap.monitor_env import MonitorEnv

import torch as th
import numpy as np
import cv2

from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from env.util import set_seed
from util.torch_util import dcn

from icecream import ic
from gym import spaces
import pickle

from train_ppo_arm import (
    Config as TrainConfig,
    load_agent,
    load_env)


class TrackPerObjectSucRate(WrapperEnv):
    def __init__(self, env):
        super().__init__(env)

        self._success_count = th.zeros(env.num_env,
                                       dtype=th.long,
                                       device=env.device)
        self._episode_count = th.zeros(env.num_env,
                                       dtype=th.long,
                                       device=env.device)

    def step(self, *args, **kwds):
        obs, rew, done, info = super().step(*args, **kwds)
        with th.no_grad():
            self._episode_count += done
            if 'success' in info:
                self._success_count += info['success']
        return (obs, rew, done, info)


@dataclass
class Config(TrainConfig):
    force_vel: Optional[float] = 0.1
    force_rad: Optional[float] = 0.1
    force_ang: Optional[float] = 0.1

    def __post_init__(self):
        super().__post_init__()
        if self.force_vel is not None:
            self.use_tune_goal_speed = False
            self.env.task.max_speed = self.force_vel
        if self.force_rad is not None:
            self.use_tune_goal_radius = False
            self.env.task.goal_radius = self.force_rad
        if self.force_ang is not None:
            self.use_tune_goal_radius = False
            self.env.task.goal_angle = self.force_ang


def main(cfg: Config):
    cfg = recursive_replace_map(cfg, {'finalize': True})
    # Configure CUDA/CuDNN
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    th.backends.cudnn.benchmark = True
    seed = set_seed(cfg.env.seed)
    if (cfg.use_nvdr_record_episode or cfg.use_nvdr_record_viewer):
        cfg.env.track_debug_lines = True
    env = load_env(cfg, None, None,
                   freeze_env=False,
                   check_viewer=False)

    if cfg.use_nvdr_record_episode:
        env = NvdrRecordEpisode(cfg.nvdr_record_episode, env,
                                hide_arm=False)
    if cfg.eval_track_per_obj_suc_rate:
        env = TrackPerObjectSucRate(env)

    obs_space = map_struct(
        env.observation_space,
        lambda src, _: src.shape,
        base_cls=spaces.Box,
        dict_cls=(Mapping, spaces.Dict)
    )
    if cfg.state_net_blocklist is not None:
        for key in cfg.state_net_blocklist:
            obs_space.pop(key, None)
    dim_act = (
        env.action_space.shape if isinstance(
            env.action_space,
            spaces.Box) else env.action_space.n)
    cfg = replace(cfg, net=replace(cfg.net,
                                   obs_space=obs_space,
                                   act_space=dim_act,
                                   # FIXME: consier restoring this arg
                                   # at some point if needed
                                   # hide_action=cfg.hide_action
                                   ))
    agent = load_agent(cfg, env, None, None)
    agent.eval()

    # == EVAL ==
    with th.cuda.amp.autocast(enabled=cfg.use_amp):
        for (act, obs, rew, done, info) in agent.test(sample=cfg.sample_action,
                                                      steps=cfg.eval_step):
            pass

    # == RETURN ==
    monitor_env = env.unwrap(target=MonitorEnv)
    episode_count = monitor_env.episode_count
    success_count = monitor_env.success_count
    out = {
        'scalar': {
            'valid/episode_count': episode_count,
            'valid/success_count': success_count,
            'valid/suc_rate': (
                (success_count / episode_count)
                if (episode_count > 0) else 0),
        }
    }

    # == COLLECT STATISTICS FROM PER_CATEGORY ==
    track_env = env.unwrap(target=TrackPerObjectSucRate)
    if isinstance(track_env, TrackPerObjectSucRate):
        object_success_count = defaultdict(lambda: 0)
        object_episode_count = defaultdict(lambda: 0)
        for n, s, e in zip(
                env.scene.cur_names,
                dcn(track_env._success_count),
                dcn(track_env._episode_count)):
            object_success_count[n] += s
            object_episode_count[n] += e

        suc_rates = []
        object_success_rates = {}
        for k in object_success_count.keys():
            if object_episode_count[k] > 0:
                suc_rate = object_success_count[k] / object_episode_count[k]
            else:
                suc_rate = 0.0
            object_success_rates[k] = suc_rate
            suc_rates.append(suc_rate)
        balanced_suc_rate = float(np.mean(suc_rates))
        with open('/tmp/per-cat-suc.pkl', 'wb') as fp:
            pickle.dump(object_success_rates, fp)
        out['scalar']['valid/balanced_suc_rate'] = balanced_suc_rate
        ic(suc_rates)
        ic(balanced_suc_rate)

    if cfg.use_nvdr_record_viewer:
        filenames = sorted(
            Path(cfg.nvdr_record_viewer.record_dir).glob('*.png'))
        rgb_images = [cv2.imread(str(x))[..., ::-1] for x in filenames]
        vid_array = np.stack(rgb_images, axis=0)[None]
        out['video'] = {
            'envs': str(cfg.nvdr_record_viewer.record_dir)
            # 'envs': vid_array
        }

    if cfg.use_nvdr_record_episode:
        out['video'] = {
            'episode': str(cfg.nvdr_record_episode.record_dir)
        }
    return out


if __name__ == '__main__':
    @hydra_cli(config_name='show')
    def _main(cfg: Config):
        print(main(cfg))
    _main()
