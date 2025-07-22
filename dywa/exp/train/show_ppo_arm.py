#!/usr/bin/env python3

import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float32

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
import pickle
import yaml

from typing import Optional, Mapping
from dataclasses import dataclass, replace, asdict
from models.common import map_struct
from env.env.wrap.record_viewer import RecordViewer
from env.env.wrap.nvdr_record_viewer import NvdrRecordViewer
from env.env.wrap.nvdr_record_episode import NvdrRecordEpisode
from env.env.wrap.log_episodes import LogEpisodes
from envs.cube_env_wrappers import CountCategoricalSuccess, ScenarioTest
from env.env.wrap.reset_ig_camera import reset_ig_camera
from env.env.wrap.draw_clouds import DrawClouds
import torch as th
import numpy as np
import einops
from env.arm_env import OBS_BOUND_MAP
from util.torch_util import dcn
from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from env.util import (
    set_seed, draw_sphere,
    draw_cloud_with_sphere,
    draw_patch_with_cvxhull
)
from env.env.wrap.normalize_env import NormalizeEnv
from env.env.wrap.draw_patch_attn import DrawPatchAttention
from env.env.wrap.plotly_wrapper import PlotlyPointCloudRecorder
from models.common import CountDormantNeurons
from util.path import ensure_directory
from train.ckpt import save_ckpt
from omegaconf import OmegaConf

from icecream import ic
from gym import spaces

from train_ppo_arm import (
    Config as TrainConfig,
    setup as setup_logging,
    load_agent,
    load_env)

from envs.cube_env_wrappers import DrawPatchCenter, ICPEmbObs
import cv2


@dataclass
class Config(TrainConfig):
    sample_action: bool = True

    use_record_viewer: bool = False
    record_viewer: RecordViewer.Config = RecordViewer.Config()

    use_log_episode: bool = False
    log_episode: LogEpisodes.Config = LogEpisodes.Config()

    draw_debug_lines: bool = False
    draw_pose: bool = True
    log_categorical_results: bool = False

    force_vel: Optional[float] = 0.1
    force_rad: Optional[float] = 0.1
    force_ang: Optional[float] = 0.1

    count_dormant_neurons: bool = False
    draw_patch_attn: bool = False
    sync_frame_time: bool = False
    test_scenario: bool = False

    export_cfg_dir: Optional[str] = None


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


@hydra_cli(config_name='show')
def main(cfg: Config):
    ic.configureOutput(includeContext=True)
    cfg = recursive_replace_map(cfg, {'finalize': True})
    ic(cfg)

    # Maybe it's related to jit
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    # path, writer = None, None
    path = setup_logging(cfg)
    _ = set_seed(cfg.env.seed)
    if (cfg.use_nvdr_record_episode or cfg.use_nvdr_record_viewer):
        cfg.env.track_debug_lines = True
    cfg, env = load_env(cfg, path, freeze_env=True,
                   check_viewer=False
                   )
    # env = DrawClouds(env, check_viewer=True,
    #                  cloud_key='cloud')
    if cfg.use_nvdr_record_episode:
        env = NvdrRecordEpisode(cfg.nvdr_record_episode, env,
                                hide_arm=False)

    if cfg.use_log_episode:
        env = LogEpisodes(cfg.log_episode, env)

    if cfg.log_categorical_results:
        env = CountCategoricalSuccess(env)

    if cfg.use_record_viewer:
        env = RecordViewer(cfg.record_viewer, env)
    if cfg.test_scenario:
        env = ScenarioTest(env)


    # Update cfg elements from `env`.
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
                                   act_space=dim_act
                                   ))
    agent = load_agent(cfg, env, None, None)

    if cfg.count_dormant_neurons:
        class OnCount:
            def __init__(self):
                self.__count = 0

            def __call__(self, stats):
                stats = {k: [dcn(x) for x in v] for k, v in stats.items()}
                with open(F'/tmp/redo-rand-{self.__count:05d}.pkl', 'wb') as fp:
                    pickle.dump(stats, fp)
                stats = {k: np.mean(v) for k, v in stats.items()}
                ic(stats)
                self.__count += 1
        count_dormant = CountDormantNeurons(OnCount())
        count_dormant.hook(agent)

    if cfg.draw_patch_attn:
        draw_env = env.unwrap(target=DrawPatchAttention)
        if isinstance(draw_env, DrawPatchAttention):
            draw_env.patch_attn_fn.register(
                agent.state_net.feature_encoders['cloud']
            )
        else:
            raise ValueError('failed to unwrap')

    reset_ig_camera(env,
                    offset=(1.0, 0.0, 0.5)
                    )
    agent.eval()
    ic(agent)

    if cfg.export_cfg_dir is not None:
        cfg_dir = ensure_directory(cfg.export_cfg_dir)
        
        # Save policy.
        policy = agent.actor_net
        OmegaConf.save(policy.cfg, F'{cfg_dir}/policy.yaml')
        save_ckpt(dict(policy=policy),
                  F'{cfg_dir}/policy.ckpt')

        # Save state encoder.
        state_net = agent.state_net
        # OmegaConf.save(cfg.net.state, F'{cfg_dir}/state.yaml')
        with open(F'{cfg_dir}/state.pkl', 'wb') as fp:
            pickle.dump(cfg.net.state, fp)
        save_ckpt(dict(state=state_net),
                  F'{cfg_dir}/state.ckpt')
        
        if cfg.use_icp_obs:
            icp = env.unwrap(target = ICPEmbObs).encoder
            OmegaConf.save(icp.cfg, F'{cfg_dir}/icp.yaml')

        # Save normalizer.
        normalizer = env.unwrap(target=NormalizeEnv).normalizer
        norm_cfg = normalizer.cfg
        if norm_cfg.stats['cloud'] == None:
            stats = norm_cfg.stats
            stats['cloud'] = OBS_BOUND_MAP.get('cloud')  
            norm_cfg = replace(
                norm_cfg, stats = stats
            )
        OmegaConf.save(norm_cfg, F'{cfg_dir}/normalizer.yaml')
        save_ckpt(dict(normalizer=normalizer),
                  F'{cfg_dir}/normalize.ckpt')

    try:

        cfg_dict = asdict(cfg)
        with open(str(path.dir / "cfg.yaml"), 'w') as file:
            yaml.dump(cfg_dict, file, default_flow_style=False)

        for (act, obs, rew, done, info) in agent.test(
                sample=cfg.sample_action, steps=1024):
            if cfg.sync_frame_time:
                env.gym.sync_frame_time(env.sim)
    finally:
        if cfg.log_categorical_results:
            env.unwrap(target=CountCategoricalSuccess).save()


if __name__ == '__main__':
    main()
