#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
import pickle
import yaml
from typing import Optional, Mapping, Dict, Any
from dataclasses import dataclass, replace, asdict
from env.env.wrap.plotly_wrapper import PlotlyPointCloudRecorder
from torch.utils.tensorboard import SummaryWriter
from models.common import map_struct, map_tensor, transfer
from env.env.wrap.base import (ObservationWrapper,
                                   add_obs_field,
                                   WrapperEnv
                                   )
from omegaconf import OmegaConf
from env.env.wrap.record_viewer import RecordViewer
# from env.env.wrap.nvdr_record_viewer import NvdrRecordViewer
# from env.env.wrap.nvdr_record_episode import NvdrRecordEpisode
from envs.cube_env_wrappers import (CountCategoricalSuccess,
                                    ScenarioTest
                                    )

from env.env.wrap.reset_ig_camera import reset_ig_camera

import torch as th
import torch.nn as nn
import numpy as np
import einops

from util.torch_util import dcn
from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from util.path import ensure_directory
from env.util import (
    set_seed, draw_sphere,
    draw_cloud_with_sphere,
    draw_patch_with_cvxhull
)
from train.ckpt import (
    last_ckpt, step_from_ckpt,
    save_ckpt
)
from models.cloud.point_mae import (
    subsample
)
from env.env.wrap.normalize_env import NormalizeEnv
from env.env.wrap.draw_patch_attn import DrawPatchAttention
from env.env.wrap.draw_clouds import DrawClouds

from icecream import ic
from gym import spaces

from train_ppo_arm import (
    Config as TrainConfig,
    AddTensorboardWriter,
    setup as setup_logging,
    load_agent,
    load_env)

from train.wandb import with_wandb, WandbConfig
from env.env.help.with_camera import WithCamera

from envs.cube_env_wrappers import DrawPatchCenter
import cv2
from models.rl.v6.ppo import (
    PPO,
    get_action_distribution,
    STATE_KEY
)
from rma import RMAEnv
from train_rma import (
    Config as RMAConfig,
    #  save_images,
    setup_rma_env_v2,
    DAggerTrainerEnv
)
from distill import StudentAgentRMA
# from distill_off import StudentAgentRMA
from tqdm.auto import tqdm
from pathlib import Path
from env.env.wrap.log_episodes import LogEpisodes
from train.hf_hub import (upload_ckpt, HfConfig, GroupConfig)
from util.math_util import (apply_pose_tq,
                                quat_multiply,
                                quat_inverse)
from env.arm_env import OBS_SPACE_MAP
import matplotlib.pyplot as plt
import os
import copy


@dataclass
class Config(RMAConfig):
    # sample_action: bool = True

    train_rma: bool = False
    log_categorical_results: bool = False
    use_log_episode: bool = False
    draw_debug_lines: bool = False
    log_episode: LogEpisodes.Config = LogEpisodes.Config()
    sync_frame_time: bool = False
    test_scenario: bool = False

    student: StudentAgentRMA.StudentAgentRMAConfig = (
        # StudentAgentRMA.StudentAgentRMAConfig()
        recursive_replace_map(
            StudentAgentRMA.StudentAgentRMAConfig(), {
                'shapes': {
                    'goal': 9,
                    'hand_state': 9,
                    'robot_state': 14,
                    'previous_action': 20,
                },
                # 'model.agg_type': 'xfm',
                # 'model.agg_type': 'gru',
                # 'model.rnn_layer': 1,
                "rnn_arch": "mlp",
                'max_delay_steps': 0,
                'without_teacher': False,
                # 'update_period': 16,
                # 'buffer_size': 64,
                "horizon": 1
            })
    )

    # rma_env: RMAEnv.Config = RMAEnv.Config(
    #     is_train=False,
    #     save_data=False,
    # )

    test_step: int = 4000
    export_cfg_dir: Optional[str] = None


    def __post_init__(self):
        super().__post_init__()
        if self.env.use_viewer:
            self.sync_frame_time = True


def update_net_cfg(base_net_cfg,
                   env,
                   blocklist=None,
                   allowlist=None):
    obs_space = map_struct(
        env.observation_space,
        lambda src, _: src.shape,
        base_cls=spaces.Box,
        dict_cls=(Mapping, spaces.Dict)
    )
    if allowlist is not None:
        for key in list(obs_space.keys()):
            if key in allowlist:
                continue
            obs_space.pop(key, None)

    if blocklist is not None:
        for key in blocklist:
            obs_space.pop(key, None)
    print('obs_space', obs_space)
    print('base_net', base_net_cfg)
    dim_act = (
        env.action_space.shape if isinstance(
            env.action_space,
            spaces.Box) else env.action_space.n)
    return replace(base_net_cfg,
                   obs_space=obs_space,
                   act_space=dim_act)


def get_config_path():
    if Path('/home/user/mambaforge').exists():
        return '/home/user/mambaforge/envs/genom/lib/python3.8/site-packages/data/cfg/'
    else:
        return '../../src/data/cfg/'

@hydra_cli(
    config_path=get_config_path(),
    config_name='show')
def main(cfg: Config):
    th.backends.cudnn.benchmark = True

    # cfg.train_rma = {}

    ic.configureOutput(includeContext=True)
    cfg.project = 'rma'
    cfg = recursive_replace_map(cfg, {'finalize': True})
    ic(cfg)

    # Maybe it's related to jit
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    writer = SummaryWriter(path.tb_train)
    # writer=None
    # path, writer = None, None
    _ = set_seed(cfg.env.seed)

    cfg, env = load_env(cfg, path, freeze_env=True,
                        check_viewer=False
                        )
    env.unwrap(target=AddTensorboardWriter).set_writer(writer)
    
    if cfg.use_log_episode:
        env = LogEpisodes(cfg.log_episode, env)

    if cfg.log_categorical_results:
        env = CountCategoricalSuccess(env)

    if cfg.test_scenario:
        env = ScenarioTest(env)

    # Update cfg elements from `env`.
    if not cfg.train_student_policy:
        cfg = replace(cfg, net=update_net_cfg(cfg.net, env,
                                              cfg.state_net_blocklist))
        # load teacher
        teacher_agent = load_agent(cfg, env, None, None)
        teacher_agent.eval()
        ic(teacher_agent)

    student_cfg = replace(
        cfg.student,
        batch_size=env.num_env
    )
    ic(student_cfg)
    student = StudentAgentRMA(student_cfg,
                              writer=writer,
                              device=env.device).to(env.device)
    
    if cfg.load_student is not None:
        ckpt = last_ckpt(cfg.load_student)
        try:
            student.load(ckpt, strict=False)
        except BaseException as e:
            xfer_dict = th.load(ckpt, map_location='cpu')
            transfer(
                student, xfer_dict,
                freeze=True,
                verbose=True,
                prefix_map={
                    'aggregator': 'aggregator.cell'
                }
            )
            raise ValueError
            exit(0)
        
            
    student.eval()

    ic(student)
    cameras = None
    # if cfg.ig_renderer is not None:
    #     cameras = WithCamera(cfg.ig_renderer)

    # env = DrawClouds(env, check_viewer=True,
    #                  cloud_key='cloud',
    #                  stride= 8)

    # env = setup_rma_env(env, cfg, student_cfg,
    #                     obs_space, agent, student,
    #                     cameras)

    state_model = (student if cfg.train_student_policy
                   else teacher_agent)
    env = setup_rma_env_v2(cfg, env,
                           state_model,
                           # student,
                           # state_size=state_size,
                           state_size=128,
                           is_student=(cfg.train_student_policy),
                           dagger=cfg.dagger)

    ### 必须load env
    if cfg.load_environment is not None:
        env_ckpt = Path(cfg.load_environment)
    else:
        env_ckpt = Path(cfg.load_student).parent / '../stat/env-last.ckpt'
    if env_ckpt.is_file():
        env.load(env_ckpt, strict=True)
    else:
        raise ValueError
        env_ckpt = cfg.load_student + '_stat'
        env.load(last_ckpt(env_ckpt))

    if cfg.plot_pc:
        env = PlotlyPointCloudRecorder(cfg.plotly_pc_recorder, env)

    # reset_ig_camera(env,
    #                 offset=(1.0, 0.0, 0.5)
    #                 )

    # state = env.reset()
    prefix: str = F'{cfg.name}'

    def export_cfg():
        if cfg.export_cfg_dir is None:
            return
        cfg_dir = ensure_directory(cfg.export_cfg_dir)

        # Save policy.
        policy = None
        if cfg.train_student_policy:
            state_net = student_policy.state_net
            policy = student_policy.actor_net
        else:
            if not cfg.dagger:
                policy = teacher_agent.actor_net
        if not cfg.dagger:
            OmegaConf.save(policy.cfg, F'{cfg_dir}/policy.yaml')
            save_ckpt(dict(policy=policy),
                      F'{cfg_dir}/policy.ckpt')
        # Save state net.
        if cfg.train_student_policy:
            with open(F'{cfg_dir}/state.pkl', 'wb') as fp:
                pickle.dump(state_net, fp)

        # Save student.
        OmegaConf.save(student.cfg, F'{cfg_dir}/student.yaml')
        save_ckpt(dict(student=student),
                  F'{cfg_dir}/student.ckpt')

        # Save normalizer.
        normalizer = env.unwrap(target=NormalizeEnv).normalizer
        OmegaConf.save(normalizer.cfg, F'{cfg_dir}/normalizer.yaml')
        save_ckpt(dict(normalizer=normalizer),
                  F'{cfg_dir}/normalize.ckpt')

    if cfg.train_student_policy:
        env.reset()
        student_policy_cfg = replace(
            cfg, net=update_net_cfg(
                cfg.net, env, allowlist=['student_state']),
            # NOTE: we explicitly disable student policy loading.
            # load_ckpt=None,
            # transfer_ckpt=None
        )
        student_policy = load_agent(student_policy_cfg,
                                    env,
                                    path,
                                    writer)
        if hasattr(env, 'save'):
            stat_ckpt = last_ckpt(cfg.load_ckpt + "_stat")
            print(F'Also loading env stats from {stat_ckpt}')
            env.load(stat_ckpt,
                     strict=False)
        export_cfg()

        ic(student_policy)
        th.cuda.empty_cache()
        with th.cuda.amp.autocast(enabled=cfg.use_amp):
            for step in student_policy.test():
                pass

    else:
        export_cfg()

        if cfg.dagger:
            env = DAggerTrainerEnv(
                # DAggerTrainerEnv.Config(
                #     alpha0=1.0,
                #     alpha1=1.0, 
                #     anneal_step=1),
                cfg.dagger_train_env,
                env, student)
            env.reset()

        cfg_dict = asdict(cfg)
        with open(str(path.dir / "cfg.yaml"), 'w') as file:
            yaml.dump(cfg_dict, file, default_flow_style=False)

        try:
            with th.no_grad():
                for step in tqdm(range(cfg.test_step), desc=prefix):
                    aux = {}
                    if cfg.dagger:
                        env.step()
                    else:
                        state, done = env.step(step, state, done, aux=aux)
                    if cfg.sync_frame_time:
                        env.gym.sync_frame_time(env.sim)

                    # if cfg.save_image:
                    #     images = aux.get('image', None)
                    #     save_images(images, step)

        finally:
            if cfg.log_categorical_results:
                env.unwrap(target=CountCategoricalSuccess).save()


if __name__ == '__main__':
    main()
