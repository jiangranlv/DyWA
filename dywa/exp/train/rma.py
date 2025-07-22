#!/usr/bin/env python3

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
import pickle

from typing import Optional, Mapping, Dict, Any
from dataclasses import dataclass, replace
from models.common import map_struct, map_tensor
from env.env.wrap.base import (ObservationWrapper,
                                   add_obs_field,
                                   WrapperEnv
                                   )
from env.env.wrap.record_viewer import RecordViewer
from env.env.wrap.nvdr_record_viewer import NvdrRecordViewer
from env.env.wrap.nvdr_record_episode import NvdrRecordEpisode
from envs.cube_env_wrappers import CountCategoricalSuccess

import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
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

from envs.cube_env_wrappers import (
    DrawPatchCenter,
    ICPEmbObs
)

import cv2
from models.rl.v6.ppo import (
    PPO,
    get_action_distribution,
    STATE_KEY
)
from distill import StudentAgentRMA
# from distill_off import StudentAgentRMA
from tqdm.auto import tqdm
from pathlib import Path
from train.hf_hub import (upload_ckpt, HfConfig, GroupConfig)
from util.math_util import (apply_pose_tq,
                                quat_multiply,
                                quat_inverse,
                                matrix_from_quaternion)
from env.arm_env import OBS_SPACE_MAP
import matplotlib.pyplot as plt
import os
import copy


class RMAEnv(WrapperEnv):
    """
    ~DAgger
    """
    @dataclass
    class Config:
        is_train: bool = False
        save_data: bool = False
        noise_mag: float = 0.005
        noise_type: str = 'additive'  # 'additive' or 'scaling'
        num_multicamera: int = 0
        use_gpcd: bool = False
        pose_type: str = 'goal'  # or offset (from initial)
        goal_pose_noise: float = 0.005
        cloud_size: int = 512
        use_neg_icp: bool = False

    def __init__(self, cfg: Config,
                 env, teacher,
                 student, cameras=None):
        super().__init__(env)
        self.cfg = cfg
        self.teacher = teacher
        self.student = student
        if cfg.is_train:
            self.student.train()
        else:
            self.student.eval()
        self.cameras = cameras
        obs_space = env.observation_space
        self._obs_space = copy.deepcopy(obs_space)
        if cfg.use_gpcd:
            obs_space, self._update_gpcd = add_obs_field(
                obs_space,
                'gpcd',
                OBS_SPACE_MAP.get('cloud')
            )
        obs_space, self._update_pose_target = add_obs_field(
            obs_space,
            'pose_target',
            OBS_SPACE_MAP.get('pose')
        )
        _, self._update_state = add_obs_field(
            obs_space,
            'teacher_state',
            env.action_space)

        if cfg.use_neg_icp:
            _, self._update_neg_state = add_obs_field(
                obs_space,
                'neg_teacher_state',
                env.action_space)

    @property
    def observation_space(self):
        return self._obs_space

    def get_action(self, state, deterministic=True):
        if deterministic:
            mu, _ = self.teacher.actor_net(state.detach().clone())
            actn = mu
        else:
            agent = self.teacher
            dist = get_action_distribution(
                state,
                agent.actor_net,
                agent.domain_cfg.discrete, agent.cfg.tanh_xfm,
                aux={}, stable_normal=agent.stable_normal)
            actn = dist.sample()
        return actn

    def reset(self):
        obs = self.env.reset()
        if self.cameras is not None:
            self.cameras.setup(self.env)

        # Initialize the teacher and student states.
        with th.no_grad():

            _, _, hidden, done = self.teacher.init(obs)
            self.hidden = hidden

            # Create partial-cloud observation.
            if self.cfg.num_multicamera > 0:
                cloud = obs['partial_cloud']
                clouds = [cloud]
                for i in range(1, self.cfg.num_multicamera + 1):
                    clouds.append(obs[f'partial_cloud_{i}'])
                cloud = th.cat(clouds, -2)
                cloud = subsample(cloud, n=self.cfg.cloud_size)
                obs['partial_cloud'] = cloud

            # Contrastive...
            if self.cfg.use_neg_icp:
                obs_neg = {**obs, 'icp_emb': obs['icp_emb_neg']}
                _, _, hidden_neg, _ = self.teacher.init(obs_neg)
                self.hidden_neg = hidden_neg

            if 'icp_emb' in obs:
                obs.pop('icp_emb')

            if 'icp_emb_neg' in obs:
                obs.pop('icp_emb_neg')

            norm = self.unwrap(target=NormalizeEnv)
            un_obs = norm.normalizer.unnormalize_obs(obs)
            if self.cfg.use_gpcd:
                self.gpcd = apply_pose_tq(un_obs['goal'][..., None, :],
                                          un_obs['partial_cloud']
                                          )
                t_obs = {}
                t_obs['partial_cloud'] = self.gpcd.clone()
                t_gpcd = norm.normalizer.normalize_obs(t_obs)['partial_cloud']
                obs = self._update_gpcd(obs, t_gpcd)
            if self.cfg.pose_type == 'goal':
                obs = self._update_pose_target(obs, un_obs['goal'])
            elif self.cfg.pose_type == 'offset':
                cur_obj = un_obs['object_state']
                self.init_pose = cur_obj[..., :7].clone()
                offset = th.zeros_like(self.init_pose)
                offset[..., -1] = 1
                obs = self._update_pose_target(obs, offset)
            obs = self._update_state(obs, self.hidden[STATE_KEY])
            if self.cfg.use_neg_icp:
                obs = self._update_neg_state(obs, self.hidden_neg[STATE_KEY])
        state = self.student.reset(obs)

        return state, done

    def on_step(self,
                step: int,
                obs: Dict[str, th.Tensor],
                actn: th.Tensor,
                aux: Optional[Dict[str, Any]] = None):
        cfg = self.cfg

        # (1) Optionally output visual observations for debugging.
        if aux is not None:
            if self.cameras is not None:
                images = self.cameras.step(self.env)
                aux['image'] = images

        # (2) Optionally export data for inspection.
        if cfg.save_data:
            norm = self.unwrap(target=NormalizeEnv)
            with open(F'/tmp/test/sim-data-{step:04d}.pkl', 'wb') as fp:
                t_obs = copy.deepcopy(obs)
                t_obs.pop('teacher_state')
                t_obs.pop('pose_target')
                export_data = norm.normalizer.unnormalize_obs(t_obs)
                export_data = {k: dcn(v) for k, v in export_data.items()}
                export_data['action'] = dcn(norm._wrap_act(actn))
                if 'pose' in aux:
                    export_data['pred_goal'] = dcn(aux['pose'])
                pickle.dump(export_data, fp)

    def step(self, step, state, done, aux=None):
        cfg = self.cfg
        prev_done = done
        # (2.1) Reset states where done=True in the previous step.
        # (This also sets `need_goal` flag for the student.)
        with th.no_grad():
            keep = (~done)[..., None]
            map_tensor(self.hidden,
                       lambda src, _: src.mul_(keep))
            if cfg.use_neg_icp:
                map_tensor(self.hidden_neg,
                        lambda src, _: src.mul_(keep))
            self.student.reset_state(done)

        # (2) Compute the action, and execute in the environment.
        with th.no_grad():
            actn = self.get_action(state, deterministic=False)
            obs, rew, done, info = self.env.step(actn)

        # (3) Update the hidden states based on the observations.
        with th.no_grad():
            # (3.1) Update teacher states.
            _, self.hidden = self.teacher.state_net(
                self.hidden, actn, obs)
            if cfg.use_neg_icp:
                obs_neg = {**obs, 'icp_emb': obs['icp_emb_neg']}
                _, self.hidden_neg = self.teacher.state_net(
                    self.hidden_neg, actn, obs_neg)
            if 'icp_emb' in obs:
                obs.pop('icp_emb')
            if 'icp_emb_neg' in obs:
                obs.pop('icp_emb_neg')
            # (3.2) Add noise to partial cloud observations.
            # (only applicable for the student, since teacher
            # does not receive `partial_cloud` inputs)
            if cfg.noise_mag > 0:
                cloud = obs['partial_cloud']
                if cfg.num_multicamera > 0:
                    clouds = [cloud]
                    for i in range(1, cfg.num_multicamera + 1):
                        clouds.append(obs[f'partial_cloud_{i}'])
                    cloud = th.cat(clouds, -2)
                    cloud = subsample(cloud, n=self.cfg.cloud_size)
                noise = cfg.noise_mag * th.randn(
                    (*cloud.shape[:-1], 3),
                    dtype=cloud.dtype, device=cloud.device
                )
                # random gaussian noise
                if self.cfg.noise_type == 'additive':
                    obs['partial_cloud'] = cloud + noise
                # noise proportional to distance
                elif self.cfg.noise_type == 'scaling':
                    obs['partial_cloud'] = cloud * (1 + noise)
                else:
                    raise ValueError(
                        f"{self.cfg.noise_type} is not a proper noise type")

            # (3.3) Add teacher states to observation inputs.
            norm = self.unwrap(target=NormalizeEnv)
            un_obs = norm.normalizer.unnormalize_obs(obs)
            if self.cfg.use_gpcd:
                self.gpcd[prev_done] = apply_pose_tq(
                    un_obs['goal'][prev_done, None, :],
                    un_obs['partial_cloud'][prev_done])
                t_obs = {}
                t_obs['partial_cloud'] = self.gpcd.clone()
                t_gpcd = norm.normalizer.normalize_obs(t_obs)['partial_cloud']
                obs = self._update_gpcd(obs, t_gpcd)
            if cfg.pose_type == 'goal':
                obs = self._update_pose_target(obs, un_obs['goal'])
            elif cfg.pose_type == 'offset':
                cur_obj = un_obs['object_state']
                self.init_pose[prev_done] = cur_obj[prev_done, :7]
                pos_offset = cur_obj[..., :3] - self.init_pose[..., :3]
                rot_offset = quat_multiply(
                    cur_obj[..., 3:7], quat_inverse(self.init_pose[..., 3:7])
                )
                offset = th.cat([pos_offset, rot_offset], -1)
                obs = self._update_pose_target(obs, offset)
            obs = self._update_state(obs, self.hidden[STATE_KEY])
            if cfg.use_neg_icp:
                obs = self._update_neg_state(obs, self.hidden_neg[STATE_KEY])

        goal = obs['goal'].clone()
        obs['goal'][:] = goal + cfg.goal_pose_noise*th.randn(
            *goal.shape,
            dtype=goal.dtype, device=goal.device)

        state = self.student(obs, step, done)
        # Post-step hook for optional processes, like debugging
        # and exporting data.
        # reset hand state before log
        self.on_step(step, obs, actn, aux=aux)
        return state, done


@dataclass
class Config(TrainConfig):
    # sample_action: bool = True

    train_rma: bool = True

    use_record_viewer: bool = False
    record_viewer: RecordViewer.Config = RecordViewer.Config()

    load_student: Optional[str] = None

    save_image: bool = False

    # noise maginitude for cloud
    noise_mag: float = 0.005
    noise_type: str = 'additive'  # additive or scaling

    force_vel: Optional[float] = 0.1
    force_rad: Optional[float] = 0.1
    force_ang: Optional[float] = 0.1

    ig_renderer: Optional[WithCamera.Config] = None
    # WithCamera.Config(
    #     width = 640,
    #     height = 480,
    #     use_color = True,
    #     use_depth = True,
    #     use_label = True,
    #     use_transform = True,
    #     # rot = (0.6698814, 0.6700356, -0.2279932, -0.2243441),
    #     #rot = (-0.63242199,  0.6348932 ,  0.31515148, -0.31246215),
    #     rot = (3.82683432e-01,  5.65713056e-17, -9.23879533e-01,  2.34326020e-17),
    #     # rot = (2.58819045e-01,  5.91458986e-17, -9.65925826e-01,  1.58480958e-17),
    #     # pos = (0.96596838, 0.0149411, 0.43651931),
    #     pos = (0.96596838, 0., 0.43651931),
    #     fov = 55.44601 # Degree
    # )
    student: StudentAgentRMA.StudentAgentRMAConfig = StudentAgentRMA.StudentAgentRMAConfig()

    rma_env: RMAEnv.Config = RMAEnv.Config(
        is_train=True,
        save_data=False,
    )

    train_step: int = 1000000
    save_step: int = 10000
    anneal_step: int = 10000

    use_neg_icp: bool = False
    neg_icp_obs: ICPEmbObs.Config = ICPEmbObs.Config()

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
        self.rma_env = replace(
            self.rma_env,
            is_train=self.train_rma
        )
        if self.multiple_cameras:
            self.rma_env = replace(
                self.rma_env,
                num_multicamera=len(self.camera_eyes)
            )
        # self.student = replace(
        #     self.student,
        #     model=replace(
        #         self.student.model,
        #         use_gpcd=self.rma_env.use_gpcd
        #     )
        # )


def save_images(images, step: int):
    if images is not None:
        n: int = images.shape[0]
        for i in range(n):
            out_dir = ensure_directory(F'/tmp/images/{i}')

            cv2.imwrite(
                f'{out_dir}/rgb_{step}.png',
                cv2.cvtColor(images['color'][i][..., :3].detach(
                ).cpu().numpy(), cv2.COLOR_RGB2BGR)
            )
            np.save(
                f'{out_dir}/segmentation_{step}.npy',
                images['label'][i].detach().cpu().numpy())
            np.save(
                f'{out_dir}/depth_{step}.npy',
                images['depth'][i].detach().cpu().numpy())


def setup_rma_env(env, cfg,
                  student_cfg,
                  obs_space,
                  agent,
                  student,
                  cameras):

    return RMAEnv(cfg.rma_env, env, agent, student, cameras)


@hydra_cli(
    config_path='../../../src/data/cfg/',
    # config_path='/home/user/mambaforge/envs/genom/lib/python3.8/site-packages/data/cfg/',
    config_name='train_rl')
@with_wandb
def main(cfg: Config):
    th.backends.cudnn.benchmark = True

    ic.configureOutput(includeContext=True)
    cfg.project = 'rma'
    cfg = recursive_replace_map(cfg, {'finalize': True})
    ic(cfg)

    # Maybe it's related to jit
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    writer = SummaryWriter(path.tb_train)
    _ = set_seed(cfg.env.seed)
    if (cfg.use_nvdr_record_episode or cfg.use_nvdr_record_viewer):
        cfg.env.track_debug_lines = True
    cfg, env = load_env(cfg, path, freeze_env=True,
                   check_viewer=False
                   )
    env.unwrap(target=AddTensorboardWriter).set_writer(writer)

    if cfg.use_neg_icp:
        env = ICPEmbObs(env, cfg.neg_icp_obs,
                        'icp_emb_neg',
                        'partial_cloud')

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
    agent.eval()

    ic(agent)
    student_cfg = replace(
        cfg.student,
        batch_size=env.num_env
    )
    student = StudentAgentRMA(student_cfg,
                              writer=writer,
                              device=env.device).to(env.device)

    if cfg.load_student is not None:
        student.load(cfg.load_student)
        env_ckpt = Path(cfg.load_student).parent / '../stat/env-last.ckpt'
        env.load(env_ckpt)

    ic(student)
    cameras = None
    if cfg.ig_renderer is not None:
        cameras = WithCamera(cfg.ig_renderer)

    # env = DrawClouds(env, check_viewer=True,
    #                  cloud_key='cloud',
    #                  stride= 8)
    env = setup_rma_env(env, cfg, student_cfg,
                        obs_space, agent, student,
                        cameras)
    state, done = env.reset()
    prefix: str = F'{cfg.name}@{path.dir}'
    try:
        for step in tqdm(range(cfg.train_step), desc=prefix):
            aux = {}
            state, done = env.step(step, state, done, aux=aux)

            if cfg.save_image:
                images = aux.get('image', None)
                save_images(images, step)

            if step % cfg.save_step == 0:
                student.save(path.ckpt / F'step-{step:06d}.ckpt')
                if hasattr(env, 'save'):
                    env.save(path.stat / 'env-{step:06d}.ckpt')

            # if step % cfg.anneal_step == 0:
            #     if step > 0:
            #         student.alpha *= 0.95
            #     writer.add_scalar('alpha', student.alpha,
            #                       global_step=step)

    finally:
        student.save(path.ckpt / 'last.ckpt')
        if hasattr(env, 'save'):
            env.save(path.stat / 'env-last.ckpt')

        if cfg.use_hfhub and (cfg.hf_repo_id is not None):
            upload_ckpt(
                cfg.hf_repo_id,
                (path.ckpt / 'last.ckpt'),
                cfg.name)
            upload_ckpt(
                cfg.hf_repo_id,
                (path.stat / 'env-last.ckpt'),
                cfg.name + '_stat')


if __name__ == '__main__':
    main()
