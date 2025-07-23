#!/usr/bin/env python3

from typing import Tuple, Dict, Optional, List, Union

from dataclasses import dataclass, fields, replace
from functools import partial

import torch as th
import torch.nn as nn

import pdb

from util.torch_util import dcn
from models.rl.net.base import FeatureBase
from models.common import (
    attention,
    MultiHeadLinear,
    grad_step,
    MLP,
    map_tensor,
    SingleGRU, SingleLSTM, DeepGRU
)
from pathlib import Path
from train.ckpt import save_ckpt, load_ckpt, last_ckpt
from util.path import ensure_directory
from models.cloud.point_mae import (
    PointMAEEncoder,
)
from util.config import recursive_replace_map
from env.env.wrap.normalize_env import NormalizeEnv
from icecream import ic
from train.losses import GaussianKLDivLoss
from train.metrics import pose_error
from models.modules import (PredictorHead, HistoryEncoder, PosePredictor,
                                PointEncoder, TokenEncoder, TokenDecoder, Aggregator, CLLoss)

from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

class StudentAgentRMA(nn.Module):
    '''
    model list:

    Transformer: 
        self.tokenizer: Linear layer for abs_goal, hand_state, robot_state, previous_action
        self.pos_embed: learnable positional embedding
        Vision:
            self.group: group point 
            self.patch_encoder: vision encoder, currently is pointnet
        self.encoder: self attention encoder, input various dimensions of information

    MLP_layers: 
        self.aggregator: Aggregate the current and historical memory information
        self.project: The last linear layer

    Predict
        self.pre_pose
    '''
    @dataclass
    class StudentAgentRMAConfig(FeatureBase.Config):

        horizon: int = 1
        p_drop: float = 0.0
        num_layer: int = 2                    ### useless
        pos_embed_type: Optional[str] = 'mlp'
        patch_type: str = 'mlp'               ### useless, mlp/knn/cnn
        batch_size: int = 4096                ### useless, 自动设为env数
        action_size: int = 20
        # reset delay for student
        # student is reset after t ~ U(0, T) step
        max_delay_steps: int = 7              ### useless
        estimate_level: str = 'state'         ### or action but action is not implemented yet
        without_teacher: bool = True
        
        use_gpcd: bool = True               ### Always be True
        use_interim_goal: bool = True
        ckpt: Optional[str] = None
        state_keys: Optional[List[str]] = None


        #### training parameters        
        learning_rate: float = 3e-4
        loss_type: str = "KLDiv" ## or MSE
        use_triplet_loss: bool = False ## useless
        use_amp: bool = False

        #########################
        ####      Model      ####
        #########################


        norm:str ='bn'  ### used in final mlp and patch encoder
        embed_size: int = 128 ### self attention embed size 

        #### state tokenizer
        shapes: Optional[Dict[str, int]] = None
        state_tokenizer_activate:bool=False
        state_tokenizer_hiddens:Optional[List[int]]=None

        #### Vision Encoder
        point_tokenizer: PointEncoder.Config = PointEncoder.Config()

        #### Encoder
        encoder: TokenEncoder.Config = TokenEncoder.Config()

        ###  Decoder
        decoder: TokenDecoder.Config = TokenDecoder.Config()

        ### Pose Predictor
        vision_pose_predictor: PosePredictor.PosePredictorConfig = PosePredictor.PosePredictorConfig(
            input='vision'
        )
        merge_pose_pred: bool = False

        ### final mlp
        aggregator: Aggregator.Config = Aggregator.Config()

        ## History
        use_history: bool = False
        history_tokenizer: HistoryEncoder.Config = HistoryEncoder.Config()
        constraint: CLLoss.Config = CLLoss.Config()
        

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            p_drop = self.p_drop
            self.encoder.self_atten = recursive_replace_map(self.encoder.self_atten, {
                'layer.hidden_size': self.embed_size,
                'layer.attention.self_attn.attention_probs_dropout_prob': p_drop,
                'layer.attention.output.hidden_dropout_prob': p_drop,
                'layer.output.hidden_dropout_prob': p_drop
            })

    def __init__(self, cfg: StudentAgentRMAConfig,
                 writer, device):
        super().__init__()

        self.cfg = cfg
        self.writer = writer
        self.device = device
        self.input_keys = cfg.state_keys
        self.action_size = cfg.action_size

        ### amp
        self.scaler = GradScaler() if cfg.use_amp else None

        ## vision encoder
        self.point_tokenizer = PointEncoder(cfg.point_tokenizer, cfg.embed_size, cfg.norm)
        num_pcd_tokens = cfg.point_tokenizer.num_tokens
        num_vision_tokens = num_pcd_tokens * 2 if cfg.use_gpcd else num_pcd_tokens

        ## state tokenizer
        state_hiddens = () if cfg.state_tokenizer_hiddens is None else tuple(cfg.state_tokenizer_hiddens)
        self.tokenizer = nn.ModuleDict({
            k: MLP((cfg.shapes[k], ) + state_hiddens + (cfg.embed_size, ), 
                    use_ln=True, use_bn=False, activate_output=cfg.state_tokenizer_activate)
            for k in self.input_keys
        })

        ## encoder
        self.encoder = TokenEncoder(cfg.encoder, cfg.embed_size, num_tokens= num_vision_tokens, norm = cfg.norm)

        ## history encoder
        num_history_tokens = 0
        if cfg.use_history:
            history_tokenizer = replace(cfg.history_tokenizer, num_envs=cfg.batch_size)
            self.history_tokenizer = HistoryEncoder(history_tokenizer, embed_size=cfg.embed_size, 
                                                    num_tokens=num_pcd_tokens + len(self.input_keys), norm=cfg.norm)

            ### contrastive loss
            num_history_tokens = self.history_tokenizer.decoder.num_query_tokens
            self.constraint = CLLoss(cfg.constraint, anchor_dim= num_history_tokens * cfg.embed_size, 
                                        negative_dim= None, positive_dim= cfg.embed_size)
            
        ## decoder
        num_decoder_tokens = num_vision_tokens + len(self.input_keys)
        self.decoder = TokenDecoder(cfg.decoder, cfg.embed_size, num_tokens= num_decoder_tokens, norm= cfg.norm,
                                    cond_dim= num_history_tokens * cfg.embed_size if cfg.use_history else 0)

        ### final network
        num_agg_tokens = self.decoder.num_query_tokens
        self.aggregator = Aggregator(cfg.aggregator, num_agg_tokens, cfg.embed_size, cfg.action_size, cfg.norm, batch_size=cfg.batch_size)
       
        ## pose predictor
        self.vision_pose_predictor = PosePredictor(
            self.cfg.vision_pose_predictor,
            num_vision_tokens * cfg.embed_size,
            cfg.norm
        )

        ### loss
        if cfg.estimate_level == 'action':
            if self.cfg.loss_type == "KLDiv":
                self.loss = GaussianKLDivLoss()
            elif self.cfg.loss_type == "MSE":
                self.loss = nn.MSELoss()
            else:
                raise NotImplementedError
            self.action_pose_error = pose_error
        else:
            if cfg.use_triplet_loss:
                self.loss = nn.TripletMarginLoss(margin=0.0)
            else:
                ### need env.determistic action = True
                self.loss = nn.MSELoss()


        self.losses = 0 ###总体loss
        ## Useless
        self.pose_losses = {'state': 0,
                            "embed": 0,
                            "token": 0,
                            "vision": 0} 
        self.aux_losses = {'phys_params': 0}

        self.optimizer = th.optim.Adam(
            # OK?
            self.parameters(),
            self.cfg.learning_rate)
        
        self.need_goal = None

        self.vision_tokens = None

    def _forward_impl(self, obs):
        cfg = self.cfg

        aux = dict()
        ## input tokens
        ctx_tokens = [
            self.tokenizer[k](obs[k].detach().clone())[:, None]
            for k in self.input_keys if k in obs] 
        self.vision_tokens = pcd_tokens = self.point_tokenizer(obs['partial_cloud'])

        ## goal tokens
        if self.cfg.use_gpcd:
            gpcd_tokens = self.point_tokenizer(obs['goal_cloud'])
            self.vision_tokens = th.concat([pcd_tokens, gpcd_tokens], dim=-2) 

        ## encoder
        embed_tokens = self.encoder(self.vision_tokens)

        ## pose prediction
        self.pose_losses['vision'], pose_pred = self.vision_pose_predictor(self.vision_tokens, obs)
        if self.cfg.merge_pose_pred:
            pose_pred_tokens = self.tokenizer['pose_pred'](pose_pred)[:, None]
            embed_tokens = th.cat([pose_pred_tokens, embed_tokens], dim=-2)

        ### history condition
        cond = self.history_tokenizer(th.concat([pcd_tokens] + ctx_tokens, dim=-2)) if cfg.use_history else None
        aux['cond'] = cond

        ## decoder
        decoded_tokens = self.decoder(th.cat(ctx_tokens + [embed_tokens], dim=-2), cond)
        decoded_tokens = decoded_tokens.reshape(*decoded_tokens.shape[:-2], -1) ### B * 512
       
        if not cfg.use_interim_goal:
            self.need_goal.fill_(0)

        output = self.aggregator(decoded_tokens) 
        output = output.reshape(*output.shape[:-1], 2, -1)

        return output, aux

    def reset(self, obs):
        cfg = self.cfg
        device = obs['partial_cloud'].device

        if not cfg.without_teacher:
            teacher_state = obs.get('teacher_state', None)
            if teacher_state is not None:
                teacher_state = teacher_state.detach().clone()

            teacher_action = obs.get('teacher_action', None)
            if teacher_action is not None:
                teacher_action = teacher_action.detach().clone()

        if cfg.max_delay_steps > 0:
            self.delay_count = -th.randint(
                high=cfg.max_delay_steps,
                size=(cfg.batch_size,),
                device=device
            )

        self.aggregator.reset()

        with autocast() if cfg.use_amp else nullcontext():
            output, _ = self._forward_impl(obs)

            if not cfg.without_teacher and cfg.max_delay_steps > 0:
                output = th.where(
                    self.delay_count[..., None, None] >= 0,
                    output,
                    teacher_action
                )

        if not cfg.use_interim_goal:
            self.need_goal = th.zeros(cfg.batch_size,
                                      dtype=bool,
                                      device=obs['partial_cloud'].device)

       
        return output.detach().clone()


    def reset_state(self, done: th.Tensor):
        # reset memory
        cfg = self.cfg
        keep = (~done)[..., None]

        self.aggregator.reset(keep)

        # reset counts
        if cfg.max_delay_steps > 0:
            num_reset: int = done.sum()
            self.delay_count[done] = -th.randint(
                high=cfg.max_delay_steps,
                size=(num_reset,),
                device=self.delay_count.device
            )
        if not cfg.use_interim_goal:
            self.need_goal |= done
        
        if cfg.use_history:
            self.history_tokenizer.reset_history(done)

    def get_output(self, obs):
        """
        Generate action and predicted state without backpropagation
        
        Args:
            obs: Environment observations
            
        Returns:
            tuple: (output, aux) where output contains action/state predictions
                  and aux contains auxiliary information like pose predictions
        """
        cfg = self.cfg
        
        with autocast() if cfg.use_amp else nullcontext():
            output, aux = self._forward_impl(obs)

        # Handle delay logic (maintain original logic)
        if not cfg.without_teacher and cfg.max_delay_steps > 0:
            if cfg.estimate_level == 'state':
                teacher_state = obs.get('teacher_state', None)
                if teacher_state is not None:
                    teacher_state = teacher_state.detach().clone()
                output = th.where(
                    self.delay_count[..., None] >= 0,
                    output, teacher_state
                )
            elif cfg.estimate_level == 'action':
                teacher_action = obs.get('teacher_action', None)
                if teacher_action is not None:
                    teacher_action = teacher_action.detach().clone()
                output = th.where(
                    self.delay_count[..., None, None] >= 0,
                    output,
                    teacher_action
                )
        
        return output, aux
    
    def update_policy(self, obs, step, output, aux):
        """
        Execute backpropagation and parameter updates
        
        Args:
            obs: Environment observations
            step: Current training step
            done: Environment done flags
        """
        cfg = self.cfg
        if not cfg.without_teacher:
            teacher_state = obs.get('teacher_state', None)
            if teacher_state is not None:
                teacher_state = teacher_state.detach().clone()

            teacher_action = obs.get('teacher_action', None)
            if teacher_action is not None:
                teacher_action = teacher_action.detach().clone()

            if 'neg_teacher_state' in obs:
                neg_teacher_state = obs.get(
                    'neg_teacher_state').detach().clone()
                
        # compute pose losses
        self.pose_losses['vision'], _ = self.vision_pose_predictor(self.vision_tokens, obs)

        # Update only for the case where current timestep excced the delay
        if cfg.max_delay_steps > 0:
            step_indices = (self.delay_count >= 0).nonzero().flatten()
        else:
            step_indices = Ellipsis

        assert (cfg.max_delay_steps <= 0)

        if cfg.use_history:
            cond = aux['cond']
            self.aux_losses['cond_pos'] = self.constraint(cond, positive = teacher_state.detach())

        if cfg.estimate_level == 'state':
            if cfg.use_triplet_loss:
                self.losses = self.losses + self.loss(
                    output[step_indices],
                    teacher_state[step_indices],
                    neg_teacher_state[step_indices])
            else: 
                self.losses = self.losses + self.loss(
                    output[step_indices], teacher_state[step_indices])
        else: ### current : estimate action
            if self.cfg.loss_type == "KLDiv":
                self.losses = self.losses + self.loss(
                    # mu
                    output[step_indices][..., 0, :],
                    # ls
                    output[step_indices][..., 1, :],
                    # mu
                    teacher_action[step_indices][..., 0, :],
                    # ls
                    teacher_action[step_indices][..., 1, :],
                )
            else:
                self.losses = self.losses + self.loss(
                    # mu
                    output[step_indices][..., 0, :],
                    # mu
                    teacher_action[step_indices][..., 0, :],
                )
            pos_err, rot_err = self.action_pose_error(output[step_indices][..., 0, :6], teacher_action[step_indices][..., 0, :6]) ### teacher 和 student的pose error

        if (step + 1) % self.cfg.horizon == 0: ###当前horizon=1, 进入这个branch
            
            pose_loss = sum([v for k, v in self.pose_losses.items()]) 
            aux_loss = sum([v for k, v in self.aux_losses.items()])
            loss = self.losses + pose_loss + aux_loss
            
            if self.training:
                grad_step(loss, self.optimizer, scaler=self.scaler)
            else:
                self.optimizer.zero_grad()
                
            if self.writer is not None:
                with th.no_grad():
                    self.writer.add_scalar('loss/action',
                                           self.losses / cfg.horizon,
                                           global_step=step)
                    self.writer.add_scalar('loss/pose',
                                           pose_loss / cfg.horizon,
                                           global_step=step)
                    for k, v in self.pose_losses.items():
                        self.writer.add_scalar('loss/pose'+k,
                                           v / cfg.horizon,
                                           global_step=step)
                    for k, v in self.aux_losses.items():
                        self.writer.add_scalar('loss/aux_'+k,
                                           v / cfg.horizon,
                                           global_step=step)
        
                    self.writer.add_scalar('log/learning_rate',
                                           self.optimizer.param_groups[0]['lr'],
                                           global_step=step)
                    try:
                        pos_err, rot_err = dcn(pos_err).mean(), dcn(rot_err).mean()
                        self.writer.add_scalar('error/pos',
                                           pos_err,
                                           global_step=step)
                        self.writer.add_scalar('error/rot',
                                            rot_err,
                                            global_step=step)
                    except:
                        pass
                
            # if self.training: 
            self.losses = 0.
            for k in self.pose_losses:
                self.pose_losses[k] = 0.
            for k in self.aux_losses:
                self.aux_losses[k] = 0.

            self.aggregator.memory_detach_()

    def forward(self, obs, step, done, aux=None):
        """
        Maintain backward compatibility, internally calls decoupled functions
        
        Args:
            obs: Environment observations
            step: Current training step
            done: Environment done flags
            aux: Auxiliary data (for compatibility)
            
        Returns:
            output: Action or state predictions
        """
        cfg = self.cfg
        
        # 1. Generate output (action/state)
        output, aux = self.get_output(obs)
        
        # 2. Execute backpropagation
        self.update_policy(obs, step, output, aux)

        ### useless
        if cfg.max_delay_steps > 0: 
            self.delay_count += 1
        # if aux is not None: 
        #     aux['pose'] = pose.clone()

        return output.detach().clone()

    def save(self, path: str):
        ensure_directory(Path(path).parent)
        save_ckpt(dict(self=self),
                  ckpt_file=path)

    def load(self, path: str, strict: bool = True):
        ckpt_path = last_ckpt(path)
        load_ckpt(dict(self=self),
                  ckpt_file=ckpt_path,
                  strict=strict,
                  exclude_keys=['history_tokenizer.history', 'aggregator.aggregator.memory'])

    def reset_optimizer(self):
        self.optimizer = th.optim.Adam(
            # OK?
            filter(lambda p: p.requires_grad, self.parameters()),
            self.cfg.learning_rate)


def test_1():
    batch_size = 5
    cfg = StudentAgentRMA.StudentAgentRMAConfig(
        shapes={
            'goal': 7,
            'hand_state': 7,
            'robot_state': 14,
            'previous_action': 20
        },
        batch_size=batch_size,
        max_delay_steps=0,
        without_teacher=False,
        pose_dim=7
    )
    ic(cfg)
    student = StudentAgentRMA(cfg, None, None).to("cuda")
    ic(student)
    obs1 = {
        'goal': th.rand(batch_size, 7, device="cuda"),
        'hand_state': th.rand(batch_size, 7, device="cuda"),
        'robot_state': th.rand(batch_size, 14, device="cuda"),
        'previous_action': th.rand(batch_size, 20, device="cuda"),
        'teacher_state': th.rand(batch_size, 128, device="cuda"),
        'partial_cloud': th.rand(batch_size, 84, 3, device="cuda")

    }
    state = student.reset(obs1)
    print(state.shape)
    obs2 = {
        'goal': th.rand(batch_size, 7, device="cuda"),
        'hand_state': th.rand(batch_size, 7, device="cuda"),
        'robot_state': th.rand(batch_size, 14, device="cuda"),
        'previous_action': th.rand(batch_size, 20, device="cuda"),
        'teacher_state': th.rand(batch_size, 128, device="cuda"),
        'partial_cloud': th.rand(batch_size, 310, 3, device="cuda")
    }
    done = th.zeros(batch_size, dtype=th.bool, device="cuda")
    state2 = student(obs2, 1, done)
    print(state2.shape)


def test_deep_gru():
    B: int = 1
    D_X: int = 4
    D_S: int = 8
    N_L: int = 2

    gru_1 = DeepGRU(D_X, D_S, N_L)
    gru_2 = SingleGRU(D_X, D_S)

    x = th.zeros((B, D_X))
    h_1 = th.zeros((N_L, B, D_S))
    h_2 = th.zeros((B, D_S))

    y_1, h_1 = gru_1(x, h_1)
    y_2, h_2 = gru_2(x, h_2)
    print(h_1.shape)
    print(h_2.shape)


def main():
    test_1()
    # test_deep_gru()


if __name__ == "__main__":
    main()