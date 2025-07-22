#!/usr/bin/env python3

import numpy as np
if not hasattr(np, 'float'):
    np.float = np.float32

from isaacgym import gymtorch

from typing import Optional, Dict, Union, Mapping, Tuple, List, Any, Iterable
from dataclasses import dataclass, InitVar, replace
from pathlib import Path
from copy import deepcopy
from gym import spaces
import tempfile
import multiprocessing as mp
import json
import cv2
from functools import partial

import numpy as np
import torch as th
import einops
from torch.utils.tensorboard import SummaryWriter

from models.common import (transfer, map_struct)
# from models.rl.v4.rppo import (
#     RecurrentPPO as PPO)
from models.rl.v6.ppo import PPO
from models.rl.generic_state_encoder import (
    MLPStateEncoder)
from models.rl.nets import (
    VNet,
    PiNet,
    CategoricalPiNet,
    MLPFwdBwdDynLossNet
)

# env + general wrappers
# FIXME: ArmEnv _looks_ like a class, but it's
# actually PushEnv + wrapper.
from env.arm_env import (ArmEnv, ArmEnvConfig,
                             OBS_BOUND_MAP, _identity_bound)
from env.env.wrap.base import WrapperEnv
from env.env.wrap.normalize_env import NormalizeEnv
from env.env.wrap.monitor_env import MonitorEnv
from env.env.wrap.adaptive_domain_tuner import MultiplyScalarAdaptiveDomainTuner
from env.env.wrap.nvdr_camera_wrapper import NvdrCameraWrapper
from env.env.wrap.popdict import PopDict
from env.env.wrap.mvp_wrapper import MvpWrapper
from env.env.wrap.normalize_img import NormalizeImg
from env.env.wrap.tabulate_action import TabulateAction
from env.env.wrap.nvdr_record_viewer import NvdrRecordViewer
from env.env.wrap.nvdr_record_episode import NvdrRecordEpisode
from env.util import set_seed

from util.config import (ConfigBase, recursive_replace_map)
from util.hydra_cli import hydra_cli
from util.path import RunPath, ensure_directory
from train.ckpt import last_ckpt, step_from_ckpt
from train.hf_hub import (upload_ckpt, HfConfig, GroupConfig)
from train.wandb import with_wandb, WandbConfig
from train.util import (
    assert_committed
)
# domain-specific wrappers
from envs.push_env_wrappers import (
    AddGoalThreshFromPushTask

)
from envs.cube_env_wrappers import (
    # AddInitialCloud,
    # AddInitialRelGoal,
    AddObjectMass,
    AddPhysParams,
    AddPrevArmWrench,
    AddPrevAction,
    AddWrenchPenalty,
    AddObjectEmbedding,
    AddObjectKeypoint,
    AddObjectFullCloud,
    AddFingerFullCloud,
    AddApproxTouchFlag,
    AddTouchCount,
    AddSuccessAsObs,
    AddTrackingReward,
    CopyObs,
    QuatToDCM,
    QuatTo6D,
    RelGoal,
    AbsAndRelGoal,
    AddAbsGoal,
    Phase2Training,
    P2VembObs,
    ICPEmbObs,
    PNEmbObs
)


# == drawing/debugging wrappers ==
from env.env.wrap.draw_bbox_kpt import DrawGoalBBoxKeypoint, DrawObjectBBoxKeypoint
from env.env.wrap.draw_inertia_box import DrawInertiaBox
from env.env.wrap.draw_clouds import DrawClouds
from env.env.wrap.draw_patch_attn import DrawPatchAttention
from envs.cube_env_wrappers import (DrawGoalPose,
                                    DrawObjPose,
                                    DrawTargetPose,
                                    DrawPosBound,
                                    DrawDebugLines)
from env.env.wrap.plotly_wrapper import PlotlyPointCloudRecorder

import nvtx
from icecream import ic


def to_pod(x: np.ndarray) -> List[float]:
    return [float(e) for e in x]


@dataclass
class PolicyConfig(ConfigBase):
    """ Actor-Critic policy configuration. """
    actor: PiNet.Config = PiNet.Config()
    value: VNet.Config = VNet.Config()

    dim_state: InitVar[Optional[int]] = None
    dim_act: InitVar[Optional[int]] = None

    def __post_init__(self,
                      dim_state: Optional[int] = None,
                      dim_act: Optional[int] = None):
        if dim_state is not None:
            self.actor = replace(self.actor, dim_feat=dim_state)
            self.value = replace(self.value, dim_feat=dim_state)
        if dim_act is not None:
            self.actor = replace(self.actor, dim_act=dim_act)


@dataclass
class NetworkConfig(ConfigBase):
    """ Overall network configuration. """
    state: MLPStateEncoder.Config = MLPStateEncoder.Config()
    policy: PolicyConfig = PolicyConfig()

    obs_space: InitVar[Union[int, Dict[str, int], None]] = None
    act_space: InitVar[Optional[int]] = None

    def __post_init__(self, obs_space=None, act_space=None):
        self.state = replace(self.state,
                             obs_space=obs_space,
                             act_space=act_space)
        try:
            if isinstance(act_space, Iterable) and len(act_space) == 1:
                act_space = act_space[0]
            policy = replace(self.policy,
                             dim_state=self.state.state.dim_out,
                             dim_act=act_space)
            self.policy = policy
        except AttributeError:
            pass


@dataclass
class Config(WandbConfig, HfConfig, GroupConfig, ConfigBase):
    # WandbConfig parts
    project: str = 'arm-ppo'
    use_wandb: bool = True

    # HfConfig (huggingface) parts
    hf_repo_id: Optional[str] = 'corn/corn-/arm'
    use_hfhub: bool = True

    # General experiment / logging
    force_commit: bool = False
    description: str = ''
    path: RunPath.Config = RunPath.Config(root='/tmp/pkm/ppo-arm/')

    env: ArmEnvConfig = ArmEnvConfig(which_robot='franka')
    agent: PPO.Config = PPO.Config()

    # State/Policy network configurations
    net: NetworkConfig = NetworkConfig()

    # Loading / continuing from prevous runs
    load_ckpt: Optional[str] = None
    transfer_ckpt: Optional[str] = None
    freeze_transferred: bool = True

    global_device: Optional[str] = None

    # VISION CONFIG
    use_camera: bool = False
    camera: NvdrCameraWrapper.Config = NvdrCameraWrapper.Config(
        use_depth=True,
        use_col=True,
        ctx_type='cuda',
        # == D435 config(?) ==
        # aspect=8.0 / 5.0,
        # img_size=(480,848)
        # z_near for the physical camera
        # is actually pretty large!
        # z_near=0.195
        # Horizontal Field of View	69.4	91.2
        # Vertical Field of View	42.5	65.5
    )

    # Convert img into MVP-pretrained embeddings
    use_mvp: bool = False

    remove_state: bool = False
    remove_robot_state: bool = False
    remove_all_state: bool = False

    # Determines which inputs, even if they remain
    # in the observation dict, are not processed
    # by the state representation network.
    state_net_blocklist: Optional[List[str]] = None
    # FIXME: remove `hide_action`:
    # legacy config from train_ppo_hand.py
    hide_action: Optional[bool] = True

    add_object_mass: bool = False
    add_object_embedding: bool = False
    add_phys_params: bool = False
    add_keypoint: bool = False
    add_object_full_cloud: bool = False
    add_goal_full_cloud: bool = False
    add_finger_full_cloud: bool = False
    add_prev_wrench: bool = True
    add_prev_action: bool = True
    zero_out_prev_action: bool = False
    add_goal_thresh: bool = False
    add_wrench_penalty: bool = False
    wrench_penalty_coef: float = 1e-4
    add_touch_flag: bool = False

    add_touch_count: bool = False
    min_touch_force: float = 5e-2
    min_touch_speed: float = 1e-3
    add_success: bool = False
    add_tracking_reward: bool = False

    # ==<CURRICULUM>==
    use_tune_init_pos: bool = False
    tune_init_pos_scale: MultiplyScalarAdaptiveDomainTuner.Config = MultiplyScalarAdaptiveDomainTuner.Config(
        step=1.05, easy=0.1, hard=1.0)

    use_tune_goal_radius: bool = False
    tune_goal_radius: MultiplyScalarAdaptiveDomainTuner.Config = MultiplyScalarAdaptiveDomainTuner.Config(
        step=0.95, easy=0.5, hard=0.05)

    use_tune_goal_speed: bool = False
    tune_goal_speed: MultiplyScalarAdaptiveDomainTuner.Config = MultiplyScalarAdaptiveDomainTuner.Config(
        step=0.95, easy=4.0, hard=0.1)

    use_tune_goal_angle: bool = False
    tune_goal_angle: MultiplyScalarAdaptiveDomainTuner.Config = MultiplyScalarAdaptiveDomainTuner.Config(
        step=0.95, easy=1.57, hard=0.05)

    use_tune_pot_gamma: bool = False
    tune_pot_gamma: MultiplyScalarAdaptiveDomainTuner.Config = MultiplyScalarAdaptiveDomainTuner.Config(
        step=0.999, easy=1.00, hard=0.99, step_down=1.001,
        metric='return',
        target_lower=0.0,
        target_upper=0.0)
    force_vel: Optional[float] = None
    force_rad: Optional[float] = None
    force_ang: Optional[float] = None
    # ==</CURRICULUM>==

    use_tabulate: bool = False
    tabulate: TabulateAction.Config = TabulateAction.Config(
        num_bin=3
    )
    use_norm: bool = True
    normalizer: NormalizeEnv.Config = NormalizeEnv.Config()

    # Convert some observations into
    # alternative forms...
    use_dcm: bool = False
    use_rel_goal: bool = False
    use_6d_rel_goal: bool = False
    add_abs_goal: bool = False

    ## gt
    add_gt_list: Optional[List[str]] = None

    use_monitor: bool = True
    monitor: MonitorEnv.Config = MonitorEnv.Config()

    # == camera config ==
    use_nvdr_record_episode: bool = False
    nvdr_record_episode: NvdrRecordEpisode.Config = NvdrRecordEpisode.Config()
    use_nvdr_record_viewer: bool = False
    nvdr_record_viewer: NvdrRecordViewer.Config = NvdrRecordViewer.Config(
        img_size=(128, 128)
    )
    normalize_img: bool = True
    img_mean: float = 0.4
    img_std: float = 0.2
    cloud_only: bool = False
    multiple_cameras: bool = False
    camera_eyes: Tuple[Any] = (
        (-0.238, 0.388, 0.694), (-0.408, -0.328, 0.706)
    )

    # == "special" training configs
    # add auxiliary dynamics netweork+loss
    add_dyn_aux: bool = False
    # automatic mixed-precision(FP16) training
    use_amp: bool = False
    # DataParallel training across multiple devices
    parallel: Optional[Tuple[int, ...]] = None

    # == periodic validation configs ==
    sample_action: bool = False
    eval_period: int = -1
    eval_step: int = 256
    eval_num_env: int = 16
    eval_record: bool = True
    eval_device: str = 'cuda:0'
    eval_track_per_obj_suc_rate: bool = False

    draw_debug_lines: bool = False
    draw_pose: bool = False
    draw_patch_attn: bool = False

    finalize: bool = False
    parallel: Optional[Tuple[int, ...]] = None

    is_phase2: bool = False
    phase2: Phase2Training.Config = Phase2Training.Config()

    use_p2v: bool = False
    use_icp_obs: bool = False
    use_pn_obs: bool = False

    p2v: P2VembObs.Config = P2VembObs.Config()
    icp_obs: ICPEmbObs.Config = ICPEmbObs.Config()
    pn_obs: PNEmbObs.Config = PNEmbObs.Config()

    plot_pc: bool = False
    plotly_pc_recorder: PlotlyPointCloudRecorder.Config = PlotlyPointCloudRecorder.Config()

    def __post_init__(self):
        if self.name is None:
            self.group = F'{self.machine}-{self.env_name}-{self.model_name}-{self.tag}'
            self.name = F'{self.group}-{self.env.seed:06d}'
        if not self.finalize:
            return
        # WARNING: VERY HAZARDOUS
        use_dr_on_setup = self.env.single_object_scene.use_dr_on_setup | self.is_phase2
        use_dr = self.env.single_object_scene.use_dr | self.is_phase2
        self.env = recursive_replace_map(
            self.env, {'franka.compute_wrench': self.add_prev_wrench,
                       'franka.add_control_noise': self.is_phase2,
                       'single_object_scene.use_dr_on_setup': use_dr_on_setup,
                       'single_object_scene.use_dr': use_dr,
                       })

        if self.global_device is not None:
            dev_id: int = int(str(self.global_device).split(':')[-1])
            self.env = recursive_replace_map(self.env, {
                'graphics_device_id': (dev_id if self.env.use_viewer else -1),
                'compute_device_id': dev_id,
                'th_device': self.global_device,
            })
            self.agent = recursive_replace_map(self.agent, {
                'device': self.global_device})

        if self.force_vel is not None:
            self.use_tune_goal_speed = False
            self.env.task.max_speed = self.force_vel
        if self.force_rad is not None:
            self.use_tune_goal_radius = False
            self.env.task.goal_radius = self.force_rad
        if self.force_ang is not None:
            self.use_tune_goal_angle = False
            self.env.task.goal_angle = self.force_ang


def setup(cfg: Config):
    # Maybe it's related to jit
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    th.backends.cudnn.benchmark = True

    commit_hash = assert_committed(force_commit=cfg.force_commit)
    path = RunPath(cfg.path)
    print(F'run = {path.dir}')
    return path


class AddTensorboardWriter(WrapperEnv):
    def __init__(self, env):
        super().__init__(env)
        self._writer = None

    def set_writer(self, w):
        self._writer = w

    @property
    def writer(self):
        return self._writer


def load_env(cfg: Config, path,
             freeze_env: bool = False, **kwds):
    env = ArmEnv(cfg.env)
    env.setup()
    env.gym.prepare_sim(env.sim)
    env.refresh_tensors()
    env.reset()

    env = AddTensorboardWriter(env)

    obs_bound = None
    if cfg.use_norm:
        obs_bound = {}

        # Populate `obs_bound` with defaults
        # from `ArmEnv`.
        obs_bound['goal'] = OBS_BOUND_MAP.get(cfg.env.goal_type)
        obs_bound['object_state'] = OBS_BOUND_MAP.get(
            cfg.env.object_state_type)
        obs_bound['hand_state'] = OBS_BOUND_MAP.get(cfg.env.hand_state_type)
        obs_bound['robot_state'] = OBS_BOUND_MAP.get(cfg.env.robot_state_type)

        if cfg.normalizer.norm.stats is not None:
            obs_bound.update(deepcopy(cfg.normalizer.norm.stats))

    print(obs_bound)

    def __update_obs_bound(key, value, obs_bound,
                           overwrite: bool = True):
        if not cfg.use_norm:
            return
        if value is None:
            obs_bound.pop(key, None)

        if key in obs_bound:
            if overwrite:
                print(F'\t WARN: key = {key} already in obs_bound !')
            else:
                raise ValueError(F'key = {key} already in obs_bound !')

        obs_bound[key] = value
    update_obs_bound = partial(__update_obs_bound, obs_bound=obs_bound)

    if cfg.env.task.use_pose_goal:
        if cfg.add_goal_full_cloud:
            update_obs_bound('goal',
                             OBS_BOUND_MAP.get('cloud'))
        else:
            update_obs_bound('goal',
                             OBS_BOUND_MAP.get(cfg.env.goal_type))

    # Crude check for mutual exclusion
    # Determines what type of privileged "state" information
    # the policy will receive, as observation.
    assert (
        np.count_nonzero(
            [cfg.remove_state, cfg.remove_robot_state, cfg.remove_all_state])
        <= 1)
    if cfg.remove_state:
        env = PopDict(env, ['object_state'])
        update_obs_bound('object_state', None)
    elif cfg.remove_robot_state:
        env = PopDict(env, ['hand_state'])
        update_obs_bound('hand_state', None)
    elif cfg.remove_all_state:
        env = PopDict(env, ['hand_state', 'object_state'])
        update_obs_bound('hand_state', None)
        update_obs_bound('object_state', None)

    if cfg.add_object_mass:
        env = AddObjectMass(env, 'object_mass')
        update_obs_bound('object_mass', OBS_BOUND_MAP.get('mass'))

    if cfg.add_phys_params:
        env = AddPhysParams(env, 'phys_params')
        update_obs_bound('phys_params', OBS_BOUND_MAP.get('phys_params'))

    if cfg.add_object_embedding:
        env = AddObjectEmbedding(env, 'object_embedding')
        update_obs_bound('object_embedding', OBS_BOUND_MAP.get('embedding'))

    if cfg.add_keypoint:
        env = AddObjectKeypoint(env, 'object_keypoint')
        update_obs_bound('object_keypoint', OBS_BOUND_MAP.get('keypoint'))

    if cfg.add_object_full_cloud:
        # mutually exclusive w.r.t. `use_cloud`
        # i.e. the partial point cloud coming from
        # the camera.
        # assert (cfg.camera.use_cloud is False)
        goal_key = None
        if cfg.add_goal_full_cloud:
            goal_key = 'goal'
        env = AddObjectFullCloud(env,
                                 'cloud',
                                 goal_key=goal_key)
        update_obs_bound('cloud', OBS_BOUND_MAP.get('cloud'))
        if goal_key is not None:
            update_obs_bound(goal_key, OBS_BOUND_MAP.get('cloud'))

    if cfg.add_finger_full_cloud:
        env = AddFingerFullCloud(env, 'finger_cloud')
        update_obs_bound('finger_cloud', OBS_BOUND_MAP.get('cloud'))

    if cfg.add_goal_thresh:
        env = AddGoalThreshFromPushTask(env,
                                        key='goal_thresh',
                                        dim=3)
        update_obs_bound('goal_thresh', _identity_bound(3))

    if cfg.add_prev_wrench:
        env = AddPrevArmWrench(env, 'previous_wrench')
        update_obs_bound('previous_wrench', OBS_BOUND_MAP.get('wrench'))

    if cfg.add_prev_action:
        env = AddPrevAction(env, 'previous_action',
                            zero_out=cfg.zero_out_prev_action)
        update_obs_bound('previous_action', _identity_bound(
            env.observation_space['previous_action'].shape
        ))

    if cfg.add_wrench_penalty:
        env = AddWrenchPenalty(env,
                               cfg.wrench_penalty_coef,
                               key='env/wrench_cost')
    if cfg.add_touch_flag:
        env = AddApproxTouchFlag(env,
                                 key='touch',
                                 min_force=cfg.min_touch_force,
                                 min_speed=cfg.min_touch_speed)

    if cfg.add_touch_count:
        assert (cfg.add_touch_flag)
        env = AddTouchCount(env, key='touch_count')
        update_obs_bound('touch_count', _identity_bound(
            env.observation_space['touch_count'].shape
        ))

    if cfg.add_success:
        env = AddSuccessAsObs(env, key='success')
        update_obs_bound('success', _identity_bound(()))

    if cfg.use_camera:
        prev_space_keys = deepcopy(list(env.observation_space.keys()))
        env = NvdrCameraWrapper(
            env, cfg.camera
        )

        for k in env.observation_space.keys():
            if k in prev_space_keys:
                continue
            obs_shape = env.observation_space[k].shape
            # if k in cfg.normalizer.obs_shape:
            #     obs_shape = cfg.normalizer.obs_shape[k]
            print(k, obs_shape)
            if 'cloud' in k:
                update_obs_bound(k, OBS_BOUND_MAP.get('cloud'))
            else:
                update_obs_bound(k, _identity_bound(obs_shape[-1:]))

        if cfg.multiple_cameras:
            camera = deepcopy(cfg.camera)
            camera = replace(
                camera, use_label=False
            )
            for i, eye in enumerate(cfg.camera_eyes):
                cloud_key = f'partial_cloud_{i+1}'
                new_camera = replace(
                    camera, eye=eye
                )
                new_camera = replace(
                    new_camera, key_cloud=cloud_key
                )
                env = NvdrCameraWrapper(
                    env, new_camera
                )
                update_obs_bound(cloud_key, OBS_BOUND_MAP.get('cloud'))

        if cfg.normalize_img:
            env = NormalizeImg(env, cfg.img_mean, cfg.img_std,
                               key='depth')
            # After normalization, it (should) map to (0.0, 1.0)
            update_obs_bound('depth', (0.0, 1.0))

        if cfg.cloud_only:
            env = PopDict(env, ['depth', 'label'])
            update_obs_bound('depth', None)
            update_obs_bound('label', None)

    if cfg.use_mvp:
        assert (cfg.use_camera)
        env = MvpWrapper(env)
        raise ValueError(
            'MVPWrapper does not currently configure a proper obs space.'
        )

    if cfg.add_tracking_reward:
        env = AddTrackingReward(env, 1e-4)

    # == curriculum ==
    if cfg.use_tune_init_pos:
        def get_init_pos_scale():
            return env.scene._pos_scale

        def set_init_pos_scale(s: float):
            env.scene._pos_scale = s
        env = MultiplyScalarAdaptiveDomainTuner(cfg.tune_init_pos_scale,
                                                env,
                                                get_init_pos_scale,
                                                set_init_pos_scale,
                                                key='env/init_pos_scale')

    if cfg.use_tune_goal_radius:
        def get_goal_rad():
            return env.task.goal_radius

        def set_goal_rad(s: float):
            env.task.goal_radius = s
        env = MultiplyScalarAdaptiveDomainTuner(cfg.tune_goal_radius,
                                                env,
                                                get_goal_rad,
                                                set_goal_rad,
                                                key='env/goal_radius')

    if cfg.use_tune_goal_speed:
        def get_goal_speed():
            return env.task.max_speed

        def set_goal_speed(s: float):
            env.task.max_speed = s
        env = MultiplyScalarAdaptiveDomainTuner(cfg.tune_goal_speed,
                                                env,
                                                get_goal_speed,
                                                set_goal_speed,
                                                key='env/max_speed')

    if cfg.use_tune_goal_angle:
        def get_goal_ang():
            return env.task.goal_angle

        def set_goal_ang(s: float):
            env.task.goal_angle = s
        env = MultiplyScalarAdaptiveDomainTuner(cfg.tune_goal_angle,
                                                env,
                                                get_goal_ang,
                                                set_goal_ang,
                                                key='env/goal_angle')

    if cfg.use_tune_pot_gamma:
        def get_pot_gamma():
            return env.task.gamma

        def set_pot_gamma(s: float):
            env.task.gamma = s
        env = MultiplyScalarAdaptiveDomainTuner(cfg.tune_pot_gamma,
                                                env,
                                                get_pot_gamma,
                                                set_pot_gamma,
                                                key='env/pot_gamma')

    if cfg.use_tabulate:
        env = TabulateAction(cfg.tabulate, env)

    if cfg.use_dcm:
        env = QuatToDCM(env, {
            'goal': 3,
            'hand_state': 3,
            'object_state': 3
        })
        raise ValueError(
            'DCM (directional cosine matrix) conversions are '
            'currently disabled due to complex integration '
            'with obs_bound.')

    # Use relative goal between current object pose
    # and the goal pose, instead of absolute goal.
    if cfg.use_rel_goal:
        env = RelGoal(env, 'goal',
                      use_6d=cfg.use_6d_rel_goal)
        if cfg.use_6d_rel_goal:
            update_obs_bound('goal',
                             OBS_BOUND_MAP.get('relpose6d'))
        else:
            update_obs_bound('goal',
                             OBS_BOUND_MAP.get('relpose'))
        if cfg.add_abs_goal:
            env = AddAbsGoal(env, 'goal',
                         use_6d=cfg.use_6d_rel_goal)
            if cfg.use_6d_rel_goal:
                update_obs_bound('goal',
                                OBS_BOUND_MAP.get('relpose6d'))
                update_obs_bound('abs_goal',
                                OBS_BOUND_MAP.get('pose6d'))
            else:
                update_obs_bound('goal',
                                OBS_BOUND_MAP.get('relpose'))
                update_obs_bound('abs_goal',
                                OBS_BOUND_MAP.get('pose'))

    else:
        env = AbsAndRelGoal(env, use_6d = cfg.use_6d_rel_goal)
        if cfg.use_6d_rel_goal:
            update_obs_bound('rel_goal',
                             OBS_BOUND_MAP.get('relpose6d'))
            update_obs_bound('abs_goal',
                             OBS_BOUND_MAP.get('pose6d'))
        else:
            update_obs_bound('rel_goal',
                             OBS_BOUND_MAP.get('relpose'))
            update_obs_bound('abs_goal',
                             OBS_BOUND_MAP.get('pose'))
        # if cfg.use_6d_rel_goal:
        #     env = QuatTo6D(env, {
        #     'goal': 3,
        #     # 'hand_state': 3,
        #     # 'object_state': 3
        # })

    ### add gt for training auxiliary tasks
    if cfg.add_gt_list:
        for key in cfg.add_gt_list:
            env = CopyObs(env, key, F'{key}_gt', space=env.observation_space[key])

    if cfg.is_phase2:
        env = Phase2Training(cfg.phase2, env)

    # == DRAW, LOG, RECORD ==
    if cfg.draw_pose:
        check_viewer = kwds.pop('check_viewer', True)
        env = DrawGoalPose(env,
                           check_viewer=check_viewer)
        env = DrawObjPose(env,
                          check_viewer=check_viewer)
    elif cfg.draw_debug_lines:
        check_viewer = kwds.pop('check_viewer', True)
        env = DrawDebugLines(DrawDebugLines.Config(
            draw_workspace=kwds.pop('draw_workspace', False),
            draw_wrench_target=kwds.pop('draw_wrench_target', False),
            draw_cube_action=kwds.pop('draw_hand_action', False)
        ), env,
            check_viewer=check_viewer)
        # NOTE: blocklist=0 indicates the table;
        # blocklist=2 indicates the robot. Basically,
        # only draw the inertia-box for the object.
        env = DrawInertiaBox(env, blocklist=[0, 2],
                             check_viewer=check_viewer)
        env = DrawObjectBBoxKeypoint(env)
        env = DrawGoalBBoxKeypoint(env)
        env = DrawGoalPose(env,
                           check_viewer=check_viewer)
        env = DrawObjPose(env,
                          check_viewer=check_viewer)
        # Some alternative visualizations are available below;
        # [1] draw the goal as a "pose" frame axes
        # env = DrawTargetPose(env,
        #                      check_viewer=check_viewer)
        # [2] Draw franka EE boundary
        if cfg.env.franka.track_object:
            env = DrawPosBound(env,
                               check_viewer=check_viewer)
        # [3] Draw input point cloud observations as spheres.
        # Should usually be prevented, so check_viewer=True
        # env = DrawClouds(env, check_viewer=True, stride=8,
        #     cloud_key='partial_cloud', # or 'cloud'
        #     style='ray')
        if cfg.draw_patch_attn:
            class PatchAttentionFromPPV5:
                """
                Retrieve patchified point cloud and attention values
                from PointPatchV5FeatNet.
                """

                def __init__(self):
                    # self.__net = agent.state_net.feature_encoders['cloud']
                    self.__net = None

                def register(self, net):
                    self.__net = net

                def __call__(self, obs):
                    ravel_index = self.__net._patch_index.reshape(
                        *obs['cloud'].shape[:-2], -1, 1)
                    patch = th.take_along_dim(
                        # B, N, D
                        obs['cloud'],
                        # B, (S, P), 1
                        ravel_index,
                        dim=-2
                    ).reshape(*self.__net._patch_index.shape,
                              obs['cloud'].shape[-1])
                    attn = self.__net._patch_attn
                    # ic(attn)

                    # Only include parts that correspond to
                    # point patches
                    # ic('pre',attn.shape)
                    # attn = attn[..., 1:, :]
                    attn = attn[..., :, 1:]
                    # ic('post',attn.shape)

                    # max among heads
                    # attn = attn.max(dim=-2).values
                    # head zero
                    attn = attn[..., 2, :]

                    return (patch, attn)
            env = DrawPatchAttention(env, PatchAttentionFromPPV5(),
                                     dilate=1.2,
                                     style='cloud')

    if cfg.use_nvdr_record_viewer:
        env = NvdrRecordViewer(cfg.nvdr_record_viewer,
                               env,
                               hide_arm=False)

    # == MONITOR PERFORMANCE ==
    if cfg.use_monitor:
        env = MonitorEnv(cfg.monitor, env)

    # == Normalize environment ==
    # normalization must come after
    # the monitoring code, since it
    # overwrites env statistics.
    if cfg.use_norm:
        cfg = recursive_replace_map(cfg,
                                    {'normalizer.norm.stats': obs_bound})
        env = NormalizeEnv(cfg.normalizer, env, path)

        if cfg.load_ckpt is not None:
            ckpt_path = Path(cfg.load_ckpt)

            if ckpt_path.is_file():
                # Try to select stats from matching timestep.
                step = ckpt_path.stem.split('-')[-1]

                def ckpt_key(ckpt_file):
                    return (step in str(ckpt_file.stem).rsplit('-')[-1])
                stat_dir = ckpt_path.parent / '../stat/'
            else:
                # Find the latest checkpoint.
                ckpt_key = step_from_ckpt
                stat_dir = ckpt_path / '../stat'

            if stat_dir.is_dir():
                stat_ckpt = last_ckpt(stat_dir, key=ckpt_key)
                print(F'Also loading env stats from {stat_ckpt}')
                env.load(stat_ckpt,
                         strict=False)

                # we'll freeze env stats by default, if loading from ckpt.
                if freeze_env:
                    env.normalizer.eval()
            else:
                stat_ckpt = last_ckpt(cfg.load_ckpt + "_stat", key=ckpt_key)
                print(F'Also loading env stats from {stat_ckpt}')
                env.load(stat_ckpt,
                         strict=False)

    if cfg.use_p2v:
        env = P2VembObs(env, cfg.p2v)
        env = PopDict(env, ['cloud'])
        update_obs_bound('cloud', None)

    if cfg.use_icp_obs:
        env = ICPEmbObs(env, cfg.icp_obs)
        env = PopDict(env, ['cloud'])
        update_obs_bound('cloud', None)

    if cfg.use_pn_obs:
        env = PNEmbObs(env, cfg.pn_obs)
        env = PopDict(env, ['cloud'])
        update_obs_bound('cloud', None)
    return cfg, env


def load_agent(cfg, env, path, writer):
    device = env.device
    ic(cfg)

    # FIXME: We currently disable MLPStateEncoder from
    # receiving previous_action implicitly; it has to be
    # included in the observations explicitly.
    cfg.net.state.state.dim_act = 0
    state_net = MLPStateEncoder.from_config(cfg.net.state)

    # Create policy/value networks.
    # FIXME: introspection into cfg.dim_out
    dim_state = state_net.state_aggregator.cfg.dim_out
    if isinstance(env.action_space, spaces.Discrete):
        actor_net = CategoricalPiNet(cfg.net.policy.actor).to(device)
    else:
        actor_net = PiNet(cfg.net.policy.actor).to(device)
    value_net = VNet(cfg.net.policy.value).to(device)

    # Add extra networks (Usually for regularization,
    # auxiliary losses, or learning extra models)
    extra_nets = None
    if cfg.add_dyn_aux:
        trans_net_cfg = MLPFwdBwdDynLossNet.Config(
            dim_state=dim_state,
            dim_act=cfg.net.policy.actor.dim_act,
            dim_hidden=(128,),
        )
        trans_net = MLPFwdBwdDynLossNet(trans_net_cfg).to(device)
        extra_nets = {'trans_net': trans_net}

    agent = PPO(
        cfg.agent,
        env,
        state_net,
        actor_net,
        value_net,
        path,
        writer,
        extra_nets=extra_nets
    ).to(device)

    if cfg.transfer_ckpt is not None:
        ckpt = last_ckpt(cfg.transfer_ckpt, key=step_from_ckpt)
        xfer_dict = th.load(ckpt, map_location='cpu')
        keys = transfer(agent,
                        xfer_dict['self'],
                        freeze=cfg.freeze_transferred,
                        substrs=[
                            # 'state_net.feature_encoders',
                            # 'state_net.feature_aggregators'
                            'state_net'
                        ],
                        # prefix_map={
                        #     'state_net.feature_encoders.state':
                        #     'state_net.feature_encoders.object_state',
                        #     'state_net.feature_aggregators.state':
                        #     'state_net.feature_aggregators.object_state',
                        # },
                        verbose=True)
        print(keys)

    if cfg.load_ckpt is not None:
        ckpt: str = last_ckpt(cfg.load_ckpt, key=step_from_ckpt)
        print(F'Load agent from {ckpt}')
        agent.load(last_ckpt(cfg.load_ckpt, key=step_from_ckpt),
                   strict=True)
    return agent


def eval_agent_inner(cfg: Config, return_dict):
    # [1] Silence outputs during validation.
    import sys
    import os
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    # [2] Import & run validation.
    from valid_ppo_arm import main
    return_dict.update(main(cfg))


def eval_agent(cfg: Config, env, agent: PPO):
    # subprocess.check_output('python3 valid_ppo_hand.py ++run=')

    manager = mp.Manager()
    return_dict = manager.dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save agent ckpt for validation.
        ckpt_dir = ensure_directory(F'{tmpdir}/ckpt')
        stat_dir = ensure_directory(F'{tmpdir}/stat')

        agent_ckpt = str(ckpt_dir / 'last.ckpt')
        env_ckpt = str(stat_dir / 'env-last.ckpt')

        env.save(env_ckpt)
        agent.save(agent_ckpt)

        # Override cfg.
        # FIXME:
        # Hardcoded target_domain coefs
        # shold potentially be
        # tune_goal_speed.hard...
        # etc.
        cfg = recursive_replace_map(cfg, {
            'load_ckpt': str(ckpt_dir),
            'force_vel': 0.1,
            'force_rad': 0.05,
            'force_ang': 0.1,
            'env.num_env': cfg.eval_num_env,
            'env.use_viewer': False,
            'env.single_object_scene.num_object_types': (
                cfg.env.single_object_scene.num_object_types),
            'monitor.verbose': False,
            'draw_debug_lines': True,
            'use_nvdr_record_viewer': cfg.eval_record,
            'nvdr_record_viewer.record_dir': F'{tmpdir}/record',
            'env.task.mode': 'valid',
            'env.single_object_scene.mode': 'valid',
            'env.single_object_scene.num_valid_poses': 4,
            'global_device': cfg.eval_device,
            'eval_track_per_obj_suc_rate': True
        })
        ctx = mp.get_context('spawn')
        # Run.
        proc = ctx.Process(
            target=eval_agent_inner,
            args=(cfg, return_dict),
        )
        proc.start()
        proc.join()

        return_dict = dict(return_dict)

        if 'video' in return_dict:
            replaced = {}
            for k, v in return_dict['video'].items():
                if isinstance(v, str):
                    video_dir = v
                    assert Path(video_dir).is_dir()
                    filenames = sorted(Path(video_dir).glob('*.png'))
                    rgb_images = [cv2.imread(str(x))[..., ::-1]
                                  for x in filenames]
                    vid_array = np.stack(rgb_images, axis=0)
                    v = th.as_tensor(vid_array[None])
                    v = einops.rearrange(v, 'n t h w c -> n t c h w')
                replaced[k] = v
            return_dict['video'] = replaced
    return return_dict


@with_wandb
def inner_main(cfg: Config, env, path):
    """
    Basically it's the same as main(),
    but we commit the config _after_ finalizing.
    """
    commit_hash = assert_committed(force_commit=cfg.force_commit)
    writer = SummaryWriter(path.tb_train)
    writer.add_text('meta/commit-hash',
                    str(commit_hash),
                    global_step=0)
    env.unwrap(target=AddTensorboardWriter).set_writer(writer)
    agent = load_agent(cfg, env, path, writer)

    # Enable DataParallel() for subset of modules.
    if (cfg.parallel is not None) and (th.cuda.device_count() > 1):
        count: int = th.cuda.device_count()
        device_ids = list(cfg.parallel)
        # FIXME: hardcoded DataParallel processing only for
        # `img` feature
        if 'img' in agent.state_net.feature_encoders:
            agent.state_net.feature_encoders['img'] = th.nn.DataParallel(
                agent.state_net.feature_encoders['img'], device_ids)
    ic(agent)

    def __eval(step: int):
        logs = eval_agent(cfg, env, agent)
        log_kwds = {'video': {'fps': 20.0}}
        # == generic log() ==
        for log_type, log in logs.items():
            for tag, value in log.items():
                write = getattr(writer, F'add_{log_type}')
                write(tag, value, global_step=step,
                      **log_kwds.get(log_type, {}))

    try:
        th.cuda.empty_cache()
        with th.cuda.amp.autocast(enabled=cfg.use_amp):
            for step in agent.learn(name=F'{cfg.name}@{path.dir}'):
                # Periodically run validation.
                if (cfg.eval_period > 0) and (step % cfg.eval_period) == 0:
                    th.cuda.empty_cache()
                    __eval(step)
    finally:
        # Dump final checkpoints.
        agent.save(path.ckpt / 'last.ckpt')
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


@hydra_cli(
    config_path='../../src/data/cfg/',
    # config_path='/home/user/mambaforge/envs/genom/lib/python3.8/site-packages/data/cfg/',
    config_name='train_rl')
def main(cfg: Config):
    ic.configureOutput(includeContext=True)
    cfg = recursive_replace_map(cfg, {'finalize': True})

    # path, writer = setup(cfg)
    path = setup(cfg)
    seed = set_seed(cfg.env.seed)
    cfg, env = load_env(cfg, path)

    # Save object names... useful for debugging
    if True:
        with open(F'{path.stat}/obj_names.json', 'w') as fp:
            json.dump(env.scene.cur_names, fp)

    # Update `cfg` elements from `env`.
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
        env.action_space.shape[0] if isinstance(
            env.action_space,
            spaces.Box) else env.action_space.n)
    cfg = replace(cfg, net=replace(cfg.net,
                                   obs_space=obs_space,
                                   act_space=dim_act,
                                   ))
    return inner_main(cfg, env, path)


if __name__ == '__main__':
    main()
