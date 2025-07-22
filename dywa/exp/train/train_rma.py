#!/usr/bin/env python3


import isaacgym
import yaml
from typing import Optional, Mapping, List
from dataclasses import dataclass, replace, asdict
from models.common import map_struct, map_tensor
from env.env.wrap.base import (
    EnvIface,
    WrapperEnv)
from env.env.wrap.record_viewer import RecordViewer
from env.env.wrap.plotly_wrapper import PlotlyPointCloudRecorder
import torch as th
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from env.util import (
    set_seed
)

from icecream import ic
from gym import spaces

from train_ppo_arm import (
    Config as TrainConfig,
    AddTensorboardWriter,
    setup as setup_logging,
    load_agent,
    load_env)

from train.wandb import with_wandb
from train.ckpt import last_ckpt, step_from_ckpt


from models.rl.v6.ppo import (
    get_action_distribution
)
from models.rl.util import mixed_reset
from distill import StudentAgentRMA
# from distill_off import StudentAgentRMA
from tqdm.auto import tqdm
from pathlib import Path
from train.hf_hub import upload_ckpt

from rma_env import (
    AddInitialGoalCloud,
    AddRelGoalCloud,
    ShuffleCloud,
    setup_rma_env_v2,
    CombineCloud,
    PerturbCloud,
    PerturbGoal
)


class RMATrainerEnv(WrapperEnv):

    @dataclass
    class Config:
        agent_type: str = 'teacher'
        deterministic_action: bool = False

    def __init__(self,
                 cfg: Config,
                 env: EnvIface, student):
        super().__init__(env)
        self.cfg = cfg
        self.student = student
        self.__step: int = 0
        self.__state = None

    def reset(self):
        obs = super().reset()
        obs = mixed_reset(self.env,
                          self.num_env,
                          self.device,
                          self.timeout,
                          self.timeout)
        self.__state = self.student.reset(obs)
        done = th.zeros((self.num_env,),
                        dtype=bool,
                        device=self.device)
        self.student.reset_state(done)
        return obs

    def get_action(self, state, deterministic=True):
        cfg = self.cfg

        # Select agent.
        if cfg.agent_type == 'teacher':
            agent = self.teacher
        elif cfg.agent_type == 'student':
            agent = self.student
        # else, maybe use some mixture
        # of actions from student<->teacher

        if deterministic:
            mu, _ = agent.actor_net(state.detach().clone())
            actn = mu
        else:
            dist = get_action_distribution(
                state,
                agent.actor_net,
                agent.domain_cfg.discrete,
                agent.cfg.tanh_xfm,
                aux={},
                stable_normal=agent.stable_normal)
            actn = dist.sample()
        return actn

    def step(self, actions: Optional[th.Tensor] = None):
        """
        Automatically select action from agent,
        unless `actions` is supplied
        """
        cfg = self.cfg

        # Get student's action based on previous state.
        if actions is None:
            actions = self.get_action(self.__state,
                                      deterministic=cfg.deterministic_action)

        # Step environment with the action.
        obs, rew, done, info = super().step(actions)
        self.__state = self.student(obs, self.__step, done)

        # (2.1) Reset states where done=True.
        # (This also sets `need_goal` flag for the student.)
        with th.no_grad():
            keep = (~done)[..., None]
            map_tensor(self.__state,
                       lambda src, _: src.mul_(keep))
            self.student.reset_state(done)
        self.__step += 1
        return obs, rew, done, info


class DAggerTrainerEnv(WrapperEnv):

    @dataclass
    class Config:
        agent_type: str = 'anneal'
        deterministic_action: bool = False

        alpha0: float = 0.0
        alpha1: float = 1.0
        anneal_step: int = 32768

    def __init__(self,
                 cfg: Config,
                 env: EnvIface,
                 student):
        super().__init__(env)
        self.cfg = cfg
        self.student = student
        self.student_module = student.module if isinstance(student, nn.DataParallel) else student
        self.__step: int = 0
        self.__next_student_action = None
        self.__next_action = None

        self.last_obs = None


    def reset(self):
        obs = super().reset()
        obs = mixed_reset(self.env,
                          self.num_env,
                          self.device,
                          self.timeout,
                          self.timeout)
        self.__next_action = obs.get('teacher_action')
        self.__next_student_action = self.student_module.reset(obs)
        done = th.zeros((self.num_env,),
                        dtype=bool,
                        device=self.device)
        self.student_module.reset_state(done)

        self.last_obs = obs
        
        return obs

    def get_action(self, next_act, deterministic=False):
        cfg = self.cfg

        # Select agent.
        if cfg.agent_type == 'teacher':
            actor = self.teacher.actor_net
        elif cfg.agent_type == 'student':
            actor = self.student.actor_net
        elif cfg.agent_type == 'anneal':
            # alpha=0 => teacher
            # alpha=1 => student
            alpha = (cfg.alpha0 + (cfg.alpha1 - cfg.alpha0)
                     * (self.__step / cfg.anneal_step))
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.add_scalar('log/dagger_alpha',
                                       alpha,
                                       global_step=self.__step)

            def actor(x: th.Tensor, aux=None):
                mask = (th.rand(x.shape[0],
                                device=x.device) > alpha)
                mu_t, ls_t = self.__next_action.unbind(dim=-2)
                # mu_s, ls_s = self.student.actor_net(x)
                mu_s, ls_s = next_act.unbind(dim=-2)
                # self.student.project(x)
                mu = th.where(mask[..., None], mu_t, mu_s.to(mu_t.dtype))
                ls = th.where(mask[..., None], ls_t, ls_s.to(ls_t.dtype))  # ?
                return mu, ls

        elif cfg.agent_type == 'anneal_linear':
            # alpha=0 => teacher
            # alpha=1 => student
            alpha = min((cfg.alpha0 + (cfg.alpha1 - cfg.alpha0)
                     * (self.__step / cfg.anneal_step)), 
                     1)
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.add_scalar('log/dagger_alpha',
                                       alpha,
                                       global_step=self.__step)

            def actor(x: th.Tensor, aux=None):

                mu_t, ls_t = self.__next_action.unbind(dim=-2)
                mu_s, ls_s = next_act.unbind(dim=-2)

                mu = alpha * mu_s + (1 - alpha) * mu_t
                ls = alpha * ls_s + (1 - alpha) * ls_t
                return mu, ls
        else:
            raise NotImplementedError
        

        # else, maybe use some mixture
        # of actions from student<->teacher

        if deterministic:
            mu, _ = actor(next_act)
            actn = mu
        else:
            dist = get_action_distribution(
                next_act,
                actor,
                # self.teacher.domain_cfg.discrete,
                False, False,
                # self.teacher.cfg.tanh_xfm,
                aux={},
                # stable_normal=self.teacher.stable_normal
                stable_normal=False
                )
            actn = dist.sample()
        return actn

    def step(self, actions: Optional[th.Tensor] = None):
        """
        Automatically select action from agent,
        unless `actions` is supplied
        """
        cfg = self.cfg

        self.__next_action = self.last_obs.get('teacher_action')
        self.__next_student_action, aux = self.student.get_output(self.last_obs)

        # Get student's action based on previous state.
        if actions is None:
            actions = self.get_action(self.__next_student_action,
                                      deterministic=cfg.deterministic_action)

        # Step environment with the action.
        obs, rew, done, info = super().step(actions)
        self.last_obs['rel_goal_gt'] = obs['rel_goal_gt']

        # Backward
        self.student.update_policy(self.last_obs, self.__step, self.__next_student_action, aux)
        self.last_obs = obs

        # print(obs['partial_cloud'][0][0], obs['initial_cloud'][0][0], obs['goal_cloud'][0][0], done[0])
        # un_obs = self.env.normalizer.unnormalize_obs({'rel_goal':obs['rel_goal'],
        #                                                        'partial_cloud': obs['partial_cloud'],
        #                                                        'goal_cloud': obs['goal_cloud'],
        #                                                        })
        
        # print(un_obs['partial_cloud'][0].mean(dim=0), un_obs['goal_cloud'][0].mean(dim=0), un_obs['goal_cloud'][0].mean(dim=0) - un_obs['partial_cloud'][0].mean(dim=0), un_obs['rel_goal'][0][:3], done[0])

        # (2.1) Reset states where done=True.
        # (This also sets `need_goal` flag for the student.)
        with th.no_grad():
            keep = (~done)[..., None, None]
            map_tensor(self.__next_student_action,
                       lambda src, _: src.mul_(keep))
            self.student_module.reset_state(done)
        self.__step += 1
        return obs, rew, done, info


@dataclass
class Config(TrainConfig):
    train_rma: bool = True

    use_record_viewer: bool = False
    record_viewer: RecordViewer.Config = RecordViewer.Config()

    # noise maginitude for cloud
    combine_cloud: CombineCloud.Config = CombineCloud.Config(
        src_keys=('partial_cloud', 'partial_cloud_1', 'partial_cloud_2'),
        dst_key='partial_cloud',
        cloud_size=512,
        combine_method="uniform",
    )
    perturb_cloud: PerturbCloud.Config = PerturbCloud.Config(
        noise_mag=0.005,
        noise_type='additive'
    )
    perturb_goal: PerturbGoal.Config = PerturbGoal.Config(
        noise_mag=0.0,
    )

    force_vel: Optional[float] = 0.1
    force_rad: Optional[float] = 0.1
    force_ang: Optional[float] = 0.1

    student: StudentAgentRMA.StudentAgentRMAConfig = (
        StudentAgentRMA.StudentAgentRMAConfig()
    )
    load_student: Optional[str] = None
    load_environment: Optional[str]=None

    rma_train_env: RMATrainerEnv.Config = RMATrainerEnv.Config()
    dagger_train_env: DAggerTrainerEnv.Config = DAggerTrainerEnv.Config()

    train_step: int = 1000000
    save_step: int = 10000
    anneal_step: int = 10000

    train_student_policy: bool = False
    dagger: bool = False

    # use_neg_icp: bool = False
    # neg_icp_obs: ICPEmbObs.Config = ICPEmbObs.Config()
    use_goal_cloud: bool = False
    goal_cloud_type: Optional[str] = None
    rel_goal_cloud: AddRelGoalCloud.Config = AddRelGoalCloud.Config()

    add_init_cloud: bool = False
    add_init_rel_goal: bool = False
    initial_goal_cloud: AddInitialGoalCloud.Config = AddInitialGoalCloud.Config()

    use_partial_cloud: bool = False

    use_shuffle_cloud: bool = False
    shuffle_cloud:ShuffleCloud.Config = ShuffleCloud.Config()

    dp_gpus: Optional[List[int]] = None
    add_teacher_state: bool = False 

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

        if self.multiple_cameras:
            src_keys = ['partial_cloud']
            src_keys.extend([
                F'partial_cloud_{i+1}' for i in range(len(self.camera_eyes))])
            self.combine_cloud = replace(
                self.combine_cloud,
                src_keys=src_keys)

        if self.dagger:
            self.student.estimate_level = 'action'

        self.student.use_gpcd = self.use_goal_cloud


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
    config_path=get_config_path(), config_name='train_rl')
@with_wandb
def main(cfg: Config):
    '''
    student policy enter this
    '''
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

    # if cfg.use_neg_icp:
    #     env = ICPEmbObs(env, cfg.neg_icp_obs,
    #                     'icp_emb_neg',
    #                     'partial_cloud')

    # Update cfg elements from `env`.
    if not cfg.train_student_policy:
        cfg = replace(cfg, net=update_net_cfg(cfg.net, env,
                                              cfg.state_net_blocklist))
        # load teacher
        teacher_agent = load_agent(cfg, env, None, None)
        teacher_agent.eval()
        ic(teacher_agent)
    else:
        teacher_agent = None

    student_cfg = replace(
        cfg.student,
        batch_size=env.num_env
    )
    if cfg.train_student_policy:
        student_cfg = replace(student_cfg,
                              without_teacher=True)
    student = StudentAgentRMA(student_cfg,
                              writer=writer,
                              device=env.device)
    
    student = nn.DataParallel(student, cfg.dp_gpus) if cfg.dp_gpus is not None else student
    student = student.to(env.device)

    if cfg.load_student is not None:
        student.load(cfg.load_student, strict=True)
        

    ic(student)

    state_model = (student if cfg.train_student_policy
                   else teacher_agent)
    state_size = (student.cfg.state_size if cfg.train_student_policy
                  else cfg.net.policy.dim_state)
    env = setup_rma_env_v2(cfg, env, state_model,
                           # state_size=state_size,
                           state_size=128,
                           is_student=(cfg.train_student_policy),
                           dagger=cfg.dagger)
    
    if cfg.load_environment is not None:
        env_ckpt = Path(cfg.load_environment)
        env.load(env_ckpt, strict=True)

    # else:
    #     env_ckpt = Path(cfg.load_student).parent / '../stat/env-last.ckpt'
    # try:
        # env.load(env_ckpt, strict=True)
    # except FileNotFoundError:
    #     stat_ckpt = last_ckpt(cfg.load_ckpt + "_stat",
    #                           key=step_from_ckpt)
    #     print(F'Also loading env stats from {stat_ckpt}')
    #     env.load(stat_ckpt, strict=False)

    if cfg.plot_pc:
        env = PlotlyPointCloudRecorder(cfg.plotly_pc_recorder, env)

    prefix: str = F'{cfg.name}@{cfg.global_device}'
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
        ic(student_policy)
        try:
            th.cuda.empty_cache()
            with th.cuda.amp.autocast(enabled=cfg.use_amp):
                for step in student_policy.learn(
                        name=F'{cfg.name}@{path.dir}'):
                    pass
        finally:
            # Dump final checkpoints.
            student_policy.save(path.ckpt / 'last.ckpt')
            if hasattr(env, 'save'):
                env.save(path.stat / 'env-last.ckpt')

            # Finally, upload the trained model to huggingface model hub.
            if cfg.use_hfhub and (cfg.hf_repo_id is not None):
                upload_ckpt(
                    cfg.hf_repo_id,
                    (path.ckpt / 'last.ckpt'),
                    cfg.name)
                upload_ckpt(
                    cfg.hf_repo_id,
                    (path.stat / 'env-last.ckpt'),
                    cfg.name + '_stat')
    else:
        # == train student via RMA ==
        student.train()
        # student.eval()
        if cfg.dagger:
            train_env = DAggerTrainerEnv(
                cfg.dagger_train_env,
                env, student)
        else:
            train_env = RMATrainerEnv(cfg.rma_train_env,
                                      env, student)
        train_env.reset()

        # ==== save config and scripts === #
        cfg_dict = asdict(cfg)
        with open(str(path.stat.parent / "cfg.yaml"), 'w') as file:
            yaml.dump(cfg_dict, file, default_flow_style=False)

        student_module = student.module if isinstance(student, nn.DataParallel) else student
        try:
            for step in tqdm(range(cfg.train_step), desc=prefix, mininterval=5):
                train_env.step(None)

                if step % cfg.save_step == 0:
                    student_module.save(path.ckpt / F'step-{step:06d}.ckpt')
                    if hasattr(env, 'save'):
                        env.save(path.stat / F'env-{step:06d}.ckpt')

        finally:
            student_module.save(path.ckpt / 'last.ckpt')
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
