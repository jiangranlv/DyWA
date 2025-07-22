import torch as th  
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Union, Tuple
from models.rl.net.base import FeatureBase
from util.math_util import matrix_to_pose9d, pose7d_to_matrix, pose9d_to_matrix
from util.rot_loss import transform_rot9D_to_matrix
from util.contra_loss import contrastive_loss, triplet_loss

from functools import partial
from util.config import ConfigBase
from models.cloud.point_mae import (
    get_pos_enc_module,
    PointMAEEncoder,
    get_group_module_v2,
    MultiLayer_ConvPatchEncoder,
    MultiLayer_Pointnet2PatchEncoder,
    GroupAndMLPPatchEncoder, MLPPatchEncoder
)
from models.pointnet2 import PointNet2Module, pc_normalize_torch
from models.common import CrossAttensionDecoder, SingleGRU, SingleLSTM, DeepGRU


class Conv_block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, norm:str, kernel_size:int = 3, stride:int = 2, padding:int = 0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_channels)
        else:
            raise NotImplementedError
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv(x)
        if isinstance(self.norm, nn.LayerNorm):
            out = out.permute(0, 2, 1)
        out = self.norm(out)
        if isinstance(self.norm, nn.LayerNorm):
            out = out.permute(0, 2, 1)
        out = self.act(out)
        return out
    
class MLP_layer(nn.Module):
    def __init__(self, input_size: int,
                 state_size: int,
                 multiple_state_size: Union[List[int], None] = None,
                 norm='bn',
                 activate_last: bool = True):
        super().__init__()

        norm_type = None
        if norm == 'bn':
            norm_type = nn.BatchNorm1d
        elif norm == 'ln':
            norm_type = nn.LayerNorm
        else:
            raise NotImplementedError

        if multiple_state_size is not None:
            # assert len(multiple_state_size) >= 1
            self.layer = nn.Sequential()
            last_s = input_size
            for i, s in enumerate(multiple_state_size):
                self.layer.append(nn.Linear(last_s, s))
                self.layer.append(norm_type(s))
                self.layer.append(nn.ReLU(inplace=True))
                last_s = s
            
            # Add final layer with optional activation
            self.layer.append(nn.Linear(last_s, state_size))
            self.layer.append(norm_type(state_size))
            if activate_last:
                self.layer.append(nn.ReLU(inplace=True))

        else:
            self.layer = nn.Sequential(
                nn.Linear(input_size, state_size), 
                norm_type(state_size),
                nn.ReLU(inplace=True),
                nn.Linear(state_size, state_size),
                norm_type(state_size))
            if activate_last:
                self.layer.append(nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.layer(x)
        return out
    
class PredictorHead(nn.Module):

    @dataclass
    class PredictorHeadConfig(FeatureBase.Config):

        loss_coef:float=0
        loss_type:str = "MSE" ###
        mlp_states:Optional[List[int]]=None 
        out_dim:int = 4

    def __init__(self, cfg:PredictorHeadConfig, input_size, norm):
        super().__init__()
        self.cfg = cfg

        if self.cfg.loss_coef == 0: ### not use
            return 
        
        self.layer = MLP_layer(input_size, 
                                cfg.out_dim,
                                cfg.mlp_states,
                                norm=norm,
                                activate_last= False)
        
        if cfg.loss_type == "MSE":
            self.loss_func = nn.MSELoss()
        elif cfg.loss_type == "l1":
            self.loss_func = nn.L1Loss()
        else:
            raise NotImplementedError

    def forward(self, x, gt):
        if self.cfg.loss_coef == 0: ### not use
            return 0, None
        
        out = self.layer(x)
        loss = self.cfg.loss_coef * self.loss_func(out, gt)
        return loss, out


class FilmBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, cond_dim:int, norm:str,
                  cond_predict_scale:bool = False):
        super().__init__()

        self.blocks = nn.ModuleList([
            MLP_layer(in_channels, out_channels, norm= norm),
            MLP_layer(out_channels, out_channels, norm= norm),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels)
        )

        # make sure dimensions compatible
        self.residual_conv = MLP_layer(in_channels, out_channels, norm = norm, activate_last= False) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out
    
class PosePredictor(nn.Module):

    @dataclass
    class PosePredictorConfig(FeatureBase.Config):

        xyz_loss_coef:float=0
        rot_loss_coef:float=0
        xyz_loss_type:str = "MSE" ###
        rot_loss_type:str = "l1"
        xyz_dim:int=3
        rot_dim:int=9

        input:str="state" ### state or embed or pc_tokens
        pose_mlp_states:Optional[List[int]]=None ### xyz和rotation相同mlp
        xyz_mlp_states:Optional[List[int]]=None ### 
        rot_mlp_states:Optional[List[int]]=None

        state_size:int = 256

    def __init__(self, cfg:PosePredictorConfig, input_size, norm):
        super().__init__()

        self.cfg = cfg

        if self.cfg.rot_loss_coef == 0 and self.cfg.xyz_loss_coef == 0: ### not use
            return 

        self.pred_pose = MLP_layer(input_size, 
                                cfg.state_size,
                                cfg.pose_mlp_states,
                                norm=norm)
        
        if cfg.xyz_mlp_states is not None and cfg.rot_mlp_states is not None:
            self.pred_pose_xyz = MLP_layer(cfg.state_size, 
                                cfg.xyz_dim,
                                cfg.xyz_mlp_states,
                                norm=norm,
                                activate_last= False)
            # assert cfg.xyz_mlp_states[-1] == cfg.xyz_dim
            self.pred_pose_rot = MLP_layer(cfg.state_size, 
                                cfg.rot_dim,
                                cfg.rot_mlp_states,
                                norm=norm,
                                activate_last= False)
            # assert cfg.rot_mlp_states[-1] == cfg.rot_dim

        elif cfg.xyz_mlp_states is None and cfg.rot_mlp_states is None:
            ### xyz 和rotation用一个mlp predict                
            assert cfg.pose_mlp_states[-1] == cfg.xyz_dim + cfg.rot_dim
        
        else:
            ### xyz和rotation mlp states必须同时使用
            raise NotImplementedError
            
        if cfg.xyz_loss_type == "MSE":
            self.xyz_loss_func = nn.MSELoss()
        else:
            raise NotImplementedError
        
        if cfg.rot_loss_type == "l1":
            self.rot_loss_func = nn.L1Loss()
        else:
            raise NotImplementedError
        
        assert cfg.input in ['state', 'token', 'embed', 'vision']


    def forward(self, x:th.Tensor, obs:Dict[str, th.Tensor]) -> th.Tensor:

        if self.cfg.rot_loss_coef == 0 and self.cfg.xyz_loss_coef == 0:
            return 0, None
        
        if x.dim() == 3:
            x = x.flatten(1, 2)

        hidden = self.pred_pose(x)
        
        if self.cfg.xyz_mlp_states is not None and self.cfg.rot_mlp_states is not None:
            
            rot_pred = self.pred_pose_rot(hidden)
            xyz_pred = self.pred_pose_xyz(hidden)

        else:
            rot_pred = hidden[..., 3:]
            xyz_pred = hidden[..., :3]
        pose_pred_approx = th.cat([xyz_pred, rot_pred], dim = -1)
        
        pose_gt = pose9d_to_matrix(obs['rel_goal_gt'])
        xyz = pose_gt[:, :3, 3]
        rot_matrix = pose_gt[:, :3, :3]

        rot_pred = transform_rot9D_to_matrix(rot_pred)
        rot_loss = self.rot_loss_func(rot_pred, rot_matrix)
        xyz_loss = self.xyz_loss_func(xyz_pred, xyz)
        rot_loss = rot_loss * self.cfg.rot_loss_coef
        xyz_loss = xyz_loss * self.cfg.xyz_loss_coef


        return rot_loss + xyz_loss, pose_pred_approx

class PointEncoder(nn.Module):
    @dataclass
    class Config(ConfigBase):
        # patch encoder
        encode_patch: bool = False
        patch_encoder_type: str = 'mlp'
        patch_encoder_mlps: Optional[List[List[int]]] = None
        patch_nums: Optional[List[int]] = None
        K_nn: int = 32

        # global encoder
        pn_cfg : Optional[PointNet2Module.Config] = PointNet2Module.Config()
        num_tokens: int = 1
        token_pe: bool = False

        pos_embed_type: Optional[str] = 'mlp'

    def __init__(self, cfg:Config, embed_size:int, norm:str):
        super().__init__()
        self.cfg = cfg
        self.embed_size = embed_size
        if cfg.pos_embed_type is not None:
            self.pos_embed = get_pos_enc_module(cfg.pos_embed_type, embed_size, 
                                                in_channels= 3 if cfg.encode_patch else 4) # pos, scale
        else:
            self.pos_embed = None

        if cfg.encode_patch:   
            assert self.cfg.num_tokens == self.cfg.patch_nums[-1]
            if self.cfg.patch_encoder_type =='pointnet':
                self.patch_encoder = MultiLayer_ConvPatchEncoder(
                    mlps=cfg.patch_encoder_mlps,
                    patch_nums=cfg.patch_nums,
                    K = cfg.K_nn,
                    norm=norm
                )
            elif self.cfg.patch_encoder_type == "double_pointnet":
                raise NotImplementedError

            elif self.cfg.patch_encoder_type =='mlp':
                self.patch_encoder = GroupAndMLPPatchEncoder(
                    MLPPatchEncoder.Config(
                        pre_ln_bias=True
                    ),
                    patch_size=cfg.patch_nums[0],
                    encoder_channel=embed_size
                )
            elif self.cfg.patch_encoder_type == "pointnet++":
                self.patch_encoder = MultiLayer_Pointnet2PatchEncoder(
                    mlps=cfg.patch_encoder_mlps,
                    patch_nums=cfg.patch_nums,
                    K = cfg.K_nn,
                    norm=norm
                )
            elif self.cfg.patch_encoder_type == "pointTransformer":
                raise NotImplementedError()
            else: raise NotImplementedError()

        else:
            # self.point_encoder = My_PointNetEncoder(encode_dim=cfg.encode_PointNet_encode_dim)
            output_dim = self.embed_size * (cfg.num_tokens -1) if cfg.token_pe else self.embed_size * cfg.num_tokens
            self.point_encoder = PointNet2Module(cfg=cfg.pn_cfg, output_dim= output_dim, norm=norm)
    
    def forward(self, x):
        if self.cfg.encode_patch:
            xyz, z = self.patch_encoder(x)
        else:
            x, center, scale = pc_normalize_torch(x)
            z = self.point_encoder(x)
            z = z.reshape(z.shape[0], -1, self.embed_size)
            xyz = th.cat([center, scale], dim=-1)

        pe = self.pos_embed(xyz)
        if self.cfg.token_pe:
            z = th.concat([z, pe.unsqueeze(1)], dim=-2)
        else:
            z = z + pe
        return z
    
class TokenEncoder(nn.Module):
    @dataclass
    class Config(ConfigBase):
        self_atten: PointMAEEncoder.Config = PointMAEEncoder.Config(
            num_hidden_layers=2
        )
        encoder_type: str = "SelfAttn" 
        encoder_mlp: Optional[List[int]] = None
        res_link: bool = False

    def __init__(self, cfg:Config, embed_size:int, num_tokens:int, norm:str):
        super().__init__()
        self.cfg = cfg

        if "SelfAttn" in self.cfg.encoder_type:
            self.encoder = PointMAEEncoder(cfg.self_atten)
        elif "mlp" in self.cfg.encoder_type:
            self.encoder = MLP_layer(num_tokens * embed_size,
                                    num_tokens * embed_size,
                                    multiple_state_size=cfg.encoder_mlp,
                                        norm=norm)
        elif self.cfg.encoder_type == "Identity":
            self.encoder = nn.Identity()
        else: 
            raise NotImplementedError
        

    def forward(self, input_tokens):
        if "SelfAttn" in self.cfg.encoder_type:
            out,_,_ = self.encoder(input_tokens)
        elif "mlp" in self.cfg.encoder_type:
            cur_shape = input_tokens.shape
            input_tokens = input_tokens.reshape(*cur_shape[:-2], -1)
            out = self.encoder(input_tokens)
            if self.cfg.res_link:
                out = out + input_tokens
            out = out.reshape(*cur_shape[:-1], -1)
        elif self.cfg.encoder_type == "Identity":
            out = input_tokens
        else:
            raise NotImplementedError
        return out

class TokenDecoder(nn.Module):
    @dataclass
    class Config(ConfigBase):
        decoder_type: str = "mlp"
        cross_atten: CrossAttensionDecoder.Config = CrossAttensionDecoder.Config()
        decoder_mlp: Optional[List[int]] = None
        res_link: bool = False

        learnable_query: Optional[str] = None
        num_query_tokens: Optional[int] = None

        film_pred_scale: bool = False
        film_mlp:bool = False

        kernel_stride_list: Optional[List[List[int]]] = None

    def __init__(self, cfg:Config, embed_size:int, num_tokens:int, norm:str, cond_dim:int = 0):
        super().__init__()
        self.cfg = cfg
        self.num_query_tokens = cfg.num_query_tokens
        self.embed_size = embed_size

        if self.cfg.decoder_type == "CrossAttn":
            self.decoder =  CrossAttensionDecoder(cfg.cross_atten)
        elif self.cfg.decoder_type == "mlp":
            self.decoder = MLP_layer(num_tokens * embed_size + cond_dim,
                                    cfg.num_query_tokens * embed_size,
                                    multiple_state_size=cfg.decoder_mlp,
                                        norm=norm)
        elif self.cfg.decoder_type == 'conv':
            self.decoder = nn.Sequential(
                *([Conv_block(embed_size, embed_size, norm= norm, kernel_size=kernel, stride=stride, padding=0)
                 for kernel, stride in cfg.kernel_stride_list] +[nn.AdaptiveMaxPool1d(output_size= cfg.num_query_tokens)])
            )
        elif self.cfg.decoder_type == "film":
            if cfg.film_mlp:
                first_layer = MLP_layer(cond_dim, cond_dim, multiple_state_size=[], norm=norm)
            else:
                first_layer = nn.Identity()
            channels = [num_tokens * embed_size] + cfg.decoder_mlp + [cfg.num_query_tokens * embed_size]
            self.decoder = nn.Sequential(
                first_layer,
                *[FilmBlock(channels[i], channels[i+1], cond_dim, norm, cond_predict_scale= cfg.film_pred_scale
                            ) for i in range(len(channels)-1)])

        elif self.cfg.decoder_type == "Identity":
            self.decoder = nn.Identity()
            self.num_query_tokens = num_tokens
        else: 
            raise NotImplementedError
        
        if cfg.learnable_query is not None:
            self.register_parameter(
                cfg.learnable_query,
                nn.Parameter(
                    th.zeros(cfg.num_query_tokens, embed_size),
                    requires_grad=True
                )
            )     

    def forward(self, value_tokens, query_tokens = None):
        if self.cfg.decoder_type == "CrossAttn":
            if query_tokens is None and self.cfg.learnable_query is not None:
                query_tokens = getattr(self, self.cfg.learnable_query)
                query_tokens = query_tokens.to(value_tokens.device).unsqueeze(0).expand(value_tokens.shape[0], -1, -1)
            out = self.decoder(query_tokens, value_tokens, value_tokens)
        elif self.cfg.decoder_type == "mlp":
            if query_tokens is not None:
                value_tokens = th.cat([value_tokens, query_tokens], dim=-2)
            value_tokens = value_tokens.reshape(*value_tokens.shape[:-2], -1)
            out = self.decoder(value_tokens)
            if self.cfg.res_link:
                out = out + value_tokens
            out = out.reshape(-1, self.num_query_tokens, self.embed_size)
        elif self.cfg.decoder_type == "conv":
            out = self.decoder(value_tokens.permute(0, 2, 1))
            out = out.permute(0, 2, 1)
        elif self.cfg.decoder_type == "film":
            cond = query_tokens.reshape(query_tokens.shape[0], -1)
            cond = self.decoder[0](cond)
            x = value_tokens.reshape(*value_tokens.shape[:-2], -1)
            for layer in self.decoder[1:]:
                x = layer(x, cond)
            out = x.reshape(-1, self.num_query_tokens, self.embed_size)

        elif self.cfg.decoder_type == "Identity":
            if query_tokens is not None:
                value_tokens = th.cat([value_tokens, query_tokens], dim=-2)
            out = value_tokens
        else:
            raise NotImplementedError
        return out

class Aggregator(nn.Module):
    @dataclass
    class Config(ConfigBase):
        aggregator_type: str = "mlp"
        state_size: int = 256
        aggregator_mlp: Optional[List[int]] = None
        num_gru_layer: Optional[int] = None

    def __init__(self, cfg:Config, num_tokens: int, embed_size:int, action_size:int, norm:str, batch_size:Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        if cfg.aggregator_type == 'deep_gru':
            agg_cls = partial(DeepGRU, num_layer=cfg.num_gru_layer, batch_shape=(batch_size,))
        elif cfg.aggregator_type == 'gru':
            agg_cls = SingleGRU
        elif cfg.aggregator_type == 'lstm':
            agg_cls = SingleLSTM
        elif cfg.aggregator_type == 'mlp':
            agg_cls = MLP_layer
        else:
            raise ValueError(F'Unknown rnn_arch = {cfg.aggregator_type}')
        self.aggregator = agg_cls(num_tokens * embed_size,
                                  cfg.state_size,
                                  multiple_state_size = cfg.aggregator_mlp,
                                  norm=norm)

        self.project = nn.Linear(cfg.state_size, 2 * action_size)
     
    def forward(self, input_tokens):
        if self.cfg.aggregator_type in ['gru', 'deep_gru', 'lstm']:
            state, self.aggregator.memory = self.aggregator(input_tokens, self.aggregator.memory)
        else:
            state = self.aggregator(input_tokens)
        action = self.project(state)
        return action
    
    def reset(self, keep:th.Tensor = None):
        if self.cfg.aggregator_type == 'mlp':
            return
        if keep is None:
            keep = th.zeros(1).to(self.aggregator.memory.device)
        if self.cfg.aggregator_type == 'deep_gru':
            self.aggregator.memory = self.aggregator.memory * keep[None]
        elif self.cfg.aggregator_type == 'gru':
            self.aggregator.memory = self.aggregator.memory * keep
        else:
            self.aggregator.memory = (self.aggregator.memory[0] * keep, self.aggregator.memory[1] * keep)
    
    def memory_detach_(self):
        if self.cfg.aggregator_type in ['gru', 'deep_gru']:
            return self.aggregator.memory.detach_()
        elif self.cfg.aggregator_type == 'lstm':
            return (self.aggregator.memory[0].detach_(), self.aggregator.memory[1].detach_())
        else:
            return None

class HistoryEncoder(nn.Module):
    @dataclass
    class Config:
        encoder: TokenEncoder.Config = TokenEncoder.Config()
        decoder: TokenDecoder.Config = TokenDecoder.Config()

        history_len: int = 10
        num_envs: int = 512

    def __init__(self, cfg: Config, embed_size: int, num_tokens: int, norm: str):
        super(HistoryEncoder, self).__init__()
        self.history_len = cfg.history_len  
        self.num_envs = cfg.num_envs       
        self.cfg = cfg
        self.num_tokens = num_tokens

        self.register_buffer(f'history', th.zeros((self.num_envs, self.history_len,
                 num_tokens, embed_size), requires_grad=False))
        
        self.time_embedding = nn.Parameter(th.zeros(self.history_len, embed_size), requires_grad=True)

        self.encoder = TokenEncoder(cfg.encoder, embed_size, num_tokens=num_tokens * self.history_len, norm=norm)
        self.decoder = TokenDecoder(cfg.decoder, embed_size, num_tokens=num_tokens * self.history_len, norm=norm)

    def put_history(self, tensor: th.Tensor):
        assert tensor.shape[1] == self.num_tokens

        buffer = getattr(self, 'history')
        buffer = th.concat([buffer[:, 1:], tensor.detach().unsqueeze(1)], dim= 1)
        return buffer
        # buffer[:, :-1] = buffer[:, 1:].clone()  
        # buffer[:, -1] = tensor.detach()  
    
    def forward(self, cur_obs):

        history = self.put_history(cur_obs)
        batch_size, history_len, num_tokens, embed_size = history.shape

        # add embedding
        te = self.time_embedding.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_tokens, -1)
        input = (history + te).reshape(batch_size, -1, embed_size)

        embed = self.encoder(input)
        out = self.decoder(embed)

        return out

    def reset_history(self, done: th.Tensor):
        buffer = getattr(self, 'history')
        buffer[done] = th.zeros_like(buffer[done], requires_grad= False).to(buffer.device)

class CLLoss(nn.Module):
    @dataclass
    class Config:
        loss_coef: float = 0
        margin: float = 1.0
        loss_type: str = 'contrastive'

        dim: int = 256

    def __init__(self, cfg: Config, anchor_dim: int, positive_dim: int, negative_dim: int):
        super(CLLoss, self).__init__()
        self.cfg = cfg
        if cfg.loss_coef == 0:
            return 

        self.anchor_proj = self.create_linear(anchor_dim, cfg.dim)
        self.positive_proj = self.create_linear(positive_dim, cfg.dim)
        self.negative_proj = self.create_linear(negative_dim, cfg.dim)

    def create_linear(self, in_dim, out_dim):
        if in_dim is None: 
            return None
        return nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def map_dim(self, tensor, proj):
        if tensor is None:
            return None
        tensor = tensor.reshape(tensor.shape[0], -1) if tensor.dim() == 3 else tensor
        return proj(tensor)
    
    def forward(self, anchor:th.Tensor, positive =None, negative = None):
        if self.cfg.loss_coef == 0:
            return 0

        assert positive is not None or negative is not None
        anchor = self.map_dim(anchor, self.anchor_proj)
        positive = self.map_dim(positive, self.positive_proj)
        negative = self.map_dim(negative, self.negative_proj)

        loss = 0
        if self.cfg.loss_type == 'contrastive':
            loss += contrastive_loss(anchor, positive, y = 1, margin=self.cfg.margin) if positive is not None else 0
            loss += contrastive_loss(anchor, negative, y = 0, margin=self.cfg.margin) if negative is not None else 0
        elif self.cfg.loss_type == 'triplet':
            loss = triplet_loss(anchor, positive, negative, margin=self.cfg.margin)
        else:
            raise NotImplementedError

        return loss * self.cfg.loss_coef