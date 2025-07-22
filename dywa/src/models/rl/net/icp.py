#!/usr/bin/env python3

import torch as th
import torch.nn as nn
import einops
from pathlib import Path

from dataclasses import dataclass, fields

from typing import Tuple, Dict, Optional

from models.rl.net.base import FeatureBase
from models.cloud.point_mae import (
    PointMAEEncoder,
    get_group_module,
    get_patch_module,
    get_pos_enc_module,
    get_group_module_v2
)
from models.common import (transfer, MultiHeadLinear, MLP)
from train.ckpt import load_ckpt, last_ckpt
from util.config import recursive_replace_map, ConfigBase

from icecream import ic


class ICPNet(nn.Module):
    """
    Patchwise point features jointly processed with
    hand state and (goal state) inputs ;
    It can trained with various headers :
    collision, affordance, nearst point and etc.
    """
    @dataclass
    class Config(FeatureBase.Config):
        dim_in: Tuple[int, ...] = (512, 3)  # cloud size
        dim_out: int = 256

        # ctx_dim: int = 0
        keys: Optional[Dict[str, int]] = None
        # = field(default_factory=dict)
        headers: Tuple[str, ...] = ()
        num_query: int = 1

        patch_size: int = 32
        encoder_channel: int = 128
        encoder: PointMAEEncoder.Config = PointMAEEncoder.Config()

        # Type of positional embedding
        pos_embed_type: Optional[str] = 'mlp'
        group_type: str = 'fps'  # fps/hilbert
        patch_type: str = 'mlp'  # mlp/knn/cnn
        # some points might be included in multiple patches
        patch_overlap: float = 1.0
        # Dropout probability
        p_drop: float = 0.0

        ckpt: Optional[str] = None
        freeze_encoder: bool = False
        use_adapter: bool = False
        adapter_dim: int = 64
        tune_last_layer: bool = False
        late_late_fusion: bool = False
        output_attn: bool = False
        output_hidden: bool = False
        activate_header: bool = False
        pre_ln_bias: bool = True
        ignore_zero: bool = False  # False?

        use_vq: bool = False

        train_last_ln: bool = True
        header_inputs: Optional[Dict[str, int]] = None
        # mask_ratio: float = 0.0
        use_v2_module: bool = False

        def __init__(self, **kwds):
            names = set([f.name for f in fields(self)])
            for k, v in kwds.items():
                if k in names:
                    setattr(self, k, v)
            self.__post_init__()

        def __post_init__(self):
            p_drop = self.p_drop
            self.encoder = recursive_replace_map(self.encoder, {
                'layer.hidden_size': self.encoder_channel,
                'layer.attention.self_attn.attention_probs_dropout_prob': p_drop,
                'layer.attention.output.hidden_dropout_prob': p_drop,
                'layer.output.hidden_dropout_prob': p_drop,
                'use_adapter': self.use_adapter,
            })

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.keys = (dict() if self.cfg.keys is None
                     else self.cfg.keys)

        self.__header_keys = []
        if cfg.header_inputs is not None:
            self.__header_keys = sorted(list(cfg.header_inputs.keys()))

        if cfg.use_v2_module:
            assert (cfg.patch_overlap == 1.0)
            self.group = get_group_module_v2(cfg.group_type,
                                             cfg.patch_size,
                                             recenter=True)
            self.true_patch_size = cfg.patch_size
        else:
            (self.sort, self.group,
             self.true_patch_size, self.patch_stride) = get_group_module(
                cfg.group_type, cfg.patch_size, cfg.patch_overlap
            )
        self.patch_encoder = get_patch_module(cfg.patch_type,
                                              cfg.encoder_channel,
                                              self.true_patch_size,
                                              pre_ln_bias=cfg.pre_ln_bias)

        if cfg.pos_embed_type is not None:
            self.pos_embed = get_pos_enc_module(cfg.pos_embed_type,
                                                cfg.encoder_channel)
        else:
            self.pos_embed = None

        self.encoder = PointMAEEncoder(cfg.encoder)

        tokens = {}
        if cfg.keys:
            tokens = dict([(k, MultiHeadLinear(v,
                            cfg.num_query,
                            cfg.encoder_channel,
                            unbind=False))
                            for k, v in cfg.keys.items()])
        self.tokenizes = nn.ModuleDict(tokens)

        headers = {}
        for k in cfg.headers:
            # output_dim = cfg.dim_in[0]// self.true_patch_size if k == 'affordance' else 1
            # output_dim = 1
            output_dim = 1 + 3 + 3 if k == 'affordance' else 1

            in_dim = cfg.encoder.layer.hidden_size
            if cfg.late_late_fusion:
                in_dim = in_dim * (len(self.keys) + 1)
                headers[k] = MLP((in_dim, cfg.encoder.layer.hidden_size,
                                  cfg.encoder.layer.hidden_size,
                                  output_dim),
                                 # act_cls='frelu6'
                                 )
            else:
                extra_dim: int = 0
                if cfg.header_inputs is not None:
                    extra_dim: int = sum(cfg.header_inputs.values())
                headers[k] = MLP((cfg.encoder.layer.hidden_size + extra_dim,
                                  cfg.encoder.layer.hidden_size,
                                  cfg.encoder.layer.hidden_size,
                                  output_dim),
                                 # act_cls='frelu6',
                                 use_bn=False,
                                 use_ln=True,
                                 pre_ln_bias=cfg.pre_ln_bias)
            if k == 'affordance':
                if cfg.activate_header:
                    ic('activate')
                    headers[k].model.append(nn.Softmax())
            elif 'collision' in k:
                if cfg.activate_header:
                    ic('activate')
                    headers[k].model.append(nn.Sigmoid())
        self.headers = nn.ModuleDict(headers)
        if cfg.use_vq:
            self.output_vq_loss = nn.Identity()  # for hooks

        self.layernorm = nn.LayerNorm(cfg.encoder.layer.hidden_size,
                                      eps=cfg.encoder.layer.layer_norm_eps,
                                      elementwise_affine=cfg.train_last_ln)
        self.use_adapter = False
        if cfg.use_adapter:
            self.use_adapter = True
            self.adapter = nn.Sequential(
                nn.Linear(cfg.encoder_channel, cfg.adapter_dim),
                nn.ELU(),
                nn.Linear(cfg.adapter_dim, cfg.encoder_channel)
            )
            for k, v in self.adapter.named_parameters():
                nn.init.normal_(v, std=1e-3)

        if cfg.ckpt is not None:
            self.load(filename=cfg.ckpt, verbose=True)
        if cfg.freeze_encoder:
            for k, v in self.patch_encoder.named_parameters():
                v.requires_grad_(False)
            for k, v in self.pos_embed.named_parameters():
                v.requires_grad_(False)
            for k, v in self.encoder.named_parameters():
                if 'adapter' in k:
                    print(f"find adpater : {k}")
                    v.requires_grad_(True)
                    continue
                v.requires_grad_(False)
            for k, v in self.tokenizes.named_parameters():
                v.requires_grad_(False)
        if cfg.tune_last_layer:
            self.encoder.layer[-1].output.requires_grad_(True)

    def _embed(self, p: th.Tensor, c: th.Tensor,
               noise: Optional[th.Tensor] = None) -> th.Tensor:
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random
        noise.

        Args:
            sequence (`th.LongTensor` of shape `(batch_size, sequence_length, dim)`)
            noise (`th.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) which is
                mainly used for testing purposes to control randomness and maintain the reproducibility
        """
        z = self.patch_encoder(p - c[..., :, None, :])
        z_pre_pe = z
        if self.pos_embed is not None:
            pe = self.pos_embed(c)
            z = z + pe
        else:
            pe = None
        return z, pe, z_pre_pe

    def _group(self, x: th.Tensor,
               center: Optional[th.Tensor] = None,
               aux: Optional[Dict[str, th.Tensor]] = None):
        # [1] Parameter-free grouping of points
        cfg = self.cfg
        with th.no_grad():
            if cfg.use_v2_module:
                assert (center is None)
                p, c = self.group(x, aux=aux)
            else:
                if cfg.group_type == 'hilbert':
                    # PATCH BY HILBERT MAPPING
                    # First, sort by hilbert code
                    x = self.sort(x, aux=aux)
                    # Group into normalized patches and
                    # patch centers.
                    p, c = self.group(x, sub=False)
                elif cfg.group_type == 'fps':
                    # PATCH BY FPS
                    sort = True if cfg.patch_type == 'mlp' else False
                    p, c = self.group(x, sub=False, aux=aux, sort=sort)
                else:
                    raise ValueError(F'Unknown group_type={cfg.group_type}')

        if aux is not None:
            aux['patch'] = p

        return (p, c)

    def forward(self, x: th.Tensor,
                ctx: Dict[str, th.Tensor],
                aux: Optional[Dict[str, th.Tensor]] = None) -> th.Tensor:
        cfg = self.cfg
        s = x.shape
        x = x.reshape(-1, *s[-2:])
        p, c = self._group(x, aux=aux)
        if aux is not None:
            aux['centers'] = c
            aux['patches'] = p

        # Embed each patch
        z, pe, z_pre_pe = self._embed(p, c)

        # if cfg.mask_ratio > 0:
        #    z_prev, m, i = mask(z, cfg.mask_ratio)
        #    z_prev, _, _ = self.encoder.encoder(z_prev)
        #    z_prev = self.encoder.layernorm(z_prev)
        #    aux['mask'] = m
        #    aux['ids_restore'] = i

        # Concatenate context token.
        tokens = []
        if cfg.keys:
            tokens = [self.tokenizes[k](ctx[k].reshape(x.shape[0], -1))
                    for k in cfg.keys]

        # print("tokens:", tokens[0].shape)
        # ctx = th.cat([ctx[k] for k in cfg.keys], dim=-1)
        # ctx = ctx.reshape(x.shape[0], -1)
        # tok = self.tokenize_ctx(ctx)
        # print("z:", z.shape)
        if self.use_adapter:
            adapter_feature = self.adapter(z)
            z = z + adapter_feature
        seq = th.cat(tokens + [z], dim=-2)

        # NOTE: needed for pseudo-student training
        if aux is not None:
            # aux['z_pre_pe'] = z_pre_pe
            aux['z'] = seq

        if cfg.ignore_zero:
            key_padding_mask = einops.rearrange(
                p == 0,
                '... p d -> ... (p d)'
            ).all(dim=-1)

            pad_prefix = th.zeros(*key_padding_mask.shape[:-1], len(tokens),
                                  dtype=key_padding_mask.dtype,
                                  device=key_padding_mask.device)
            key_padding_mask = th.cat([pad_prefix, key_padding_mask],
                                      dim=-1)
            z, zs, a = self.encoder(
                seq,
                key_padding_mask=key_padding_mask,
                output_hidden_states=cfg.output_hidden,
                output_attentions=cfg.output_attn
            )
            # When exporting `key_padding_mask`, we ensure
            # to restore the original leading batch dimensions.
            if aux is not None:
                aux['key_padding_mask'] = key_padding_mask.reshape(*s[:-2], -1)
        else:
            z, zs, a = self.encoder(
                seq,
                output_hidden_states=cfg.output_hidden,
                output_attentions=cfg.output_attn
            )

        if aux is not None:
            aux['zs'] = zs

        if aux is not None:
            aux['icp_attn'] = a
        # print("z:", z.shape)
        # needed, or not?
        # print('pre-ln', z)
        # if aux is not None:
        #     aux['z-pre-ln']=z
        z = self.layernorm(z)
        # print(z[0, ..., 1:, :].shape)
        # print('icp-z', z[0, ..., 1:, :])

        if cfg.use_vq:
            raise ValueError('VQ deprecated')

        if len(cfg.headers) == 0:
            z = z.reshape(*s[:-2], -1,
                          # self.cfg.dim_out
                          z.shape[-1])
            return None, z
        # emb = z[..., :tok.shape[-2], :]
        # emb = emb.reshape(*s[:-2], -1)
        # out = emb
        # # out = self.out_proj(emb)
        # out = out.reshape(*s[:-2], -1)
        else:
            num_patches = cfg.dim_in[0] // self.true_patch_size
            patch_embeddings = z[..., -num_patches:, :]
            if len(self.__header_keys) > 0:
                header_values = th.cat([aux.get(k)
                                       for k in self.__header_keys], dim=-1)
                header_values = einops.repeat(
                    header_values, '... d -> ... s d',
                    s=patch_embeddings.shape[-2])
                header_inputs = th.cat([patch_embeddings,
                                        header_values], dim=-1)
            else:
                header_inputs = patch_embeddings

            emb = z.clone()
            out = {}
            for k in cfg.headers:
                # print("num_patches: ", num_patches)
                if 'collision' in k:
                    out[k] = self.headers[k](header_inputs)
                    # print("input: ", (z[..., -num_patches:, :]).shape)
                    # print("output: ",  out[k].shape)
                elif k == 'affordance':
                    out[k] = self.headers[k](header_inputs)
                else:
                    raise NotImplementedError()
            return out, emb

    def save(self, directory=None):
        params = {}
        params['patch_encoder'] = self.patch_encoder.state_dict()
        params['pos_embed'] = self.pos_embed.state_dict()
        params['tokenize'] = self.tokenizes.state_dict()
        params['encoder'] = self.encoder.state_dict()
        params['header'] = self.headers.state_dict()

        if directory is not None:
            th.save(params, directory)
            return
        else:
            return params

    def load(self, filename=None, params=None, verbose=False):
        if filename is not None:
            if not Path(filename).exists():
                filename = last_ckpt(filename)
            params = th.load(filename, map_location='cpu')
            if "model_state_dict" in params:
                params = params["model_state_dict"]
            else:
                params = params['model']
                output = transfer(self, params,
                                  prefix_map={
                                      'tokenizes.hand': 'tokenizes.hand_state',
                                      'encoder.': '',
                                  },
                                  verbose=True
                                  )
                ic(output)
                return
        else:
            assert (params is not None)

        new_tokenize = self.tokenizes.state_dict()
        pretrained_tokenize = {k: v for k, v in params['tokenize'].items()
                               if k in new_tokenize}
        new_tokenize.update(pretrained_tokenize)
        self.tokenizes.load_state_dict(new_tokenize, strict=True)
        if verbose:
            print(f"loading parameter for : {pretrained_tokenize.keys()}")

        encoder_param = params['encoder']
        is_strict = False if self.use_adapter else True
        self.encoder.load_state_dict(encoder_param, strict=is_strict)
        self.patch_encoder.load_state_dict(
            params['patch_encoder'], strict=True)
        self.pos_embed.load_state_dict(params['pos_embed'], strict=True)

        new_header = self.headers.state_dict()
        pretrained_header = {k: v for k, v in params['header'].items()
                             if k in new_header}
        new_header.update(pretrained_header)
        self.headers.load_state_dict(new_header)
        if verbose:
            print(f"loading parameter for : {pretrained_header.keys()}")


def test_model():
    from omegaconf import OmegaConf
    batch_size: int = 7
    x = th.randn((batch_size, 512, 3),
                 dtype=th.float32,
                 device='cuda')
    keys = {'a': 32, 'b': 16}
    headers = ['collision']
    ctx = {'a': th.randn((batch_size, 32),
                         device='cuda'), 'b': th.randn((batch_size, 16),
                                                       device='cuda')}
    num_query: int = 1
    cfg = ICPNet.Config(
        dim_in=(512, 3),
        dim_out=256,
        keys=keys,
        headers=headers
    )
    OmegaConf.save(OmegaConf.structured(cfg),
                   '/tmp/docker/icp.yaml')
    model = ICPNet(cfg).to('cuda')
    ic(model)
    # model.verbose()
    y, emb = model(x, ctx)
    print(y['collision'].shape, emb.shape)
    model.save('/tmp/docker/icp.pth')
    del model
    keys['c'] = 7
    ctx['c'] = th.randn((batch_size, 7),
                        device='cuda')

    cfg = ICPNet.Config(
        dim_in=(512, 3),
        dim_out=256,
        keys=keys,
        headers=[]
    )
    new_model = ICPNet(cfg).to('cuda')
    new_model.load('/tmp/docker/icp.pth', verbose=True)
    ic(new_model)
    y, emb = new_model(x, ctx)
    print(y, emb.shape)


if __name__ == '__main__':
    main()
