#!/usr/bin/env python3

import numpy as np
import torch as th
import torch.nn as nn

from typing import (
    Tuple, Dict, Optional, Union, Mapping, List, Any)
from dataclasses import dataclass, replace, InitVar
from util.config import ConfigBase
from functools import partial
from torch.utils._pytree import (tree_map, tree_flatten, tree_unflatten,
                                 _register_pytree_node)

from util.config import recursive_replace_map
from models.common import (
    merge_shapes, transfer, map_struct,
    tree_map_n)
from models.rl.ppo_config import DomainConfig
from models.rl.net.base import (
    AggregatorBase, FeatureBase, FuserBase)

from models.rl.net.mlp import (MLPFeatNet, MLPAggNet, MLPFuser)
from models.rl.net.cross import CrossFeatNet
from models.rl.net.sd_cross import StateDependentCrossFeatNet
from models.rl.net.pw_attn import PatchWiseAttentionFeatNet
from models.rl.net.point_patch import PointPatchFeatNet
from models.rl.net.point_patch_v2 import PointPatchV2FeatNet
from models.rl.net.point_patch_v3 import PointPatchV3FeatNet
from models.rl.net.point_patch_v4 import PointPatchV4FeatNet
from models.rl.net.point_patch_v5 import PointPatchV5FeatNet
from models.rl.net.point_patch_v7 import PointPatchV7FeatNet
from models.rl.net.point_patch_v8 import PointPatchV8FeatNet
from models.rl.net.flatten import FlattenFeatNet
from models.rl.net.flat_mlp import FlatMLPFeatNet
from models.rl.net.gru import (GRUAggNet,)
from models.rl.net.no_op import (NoOpFeatNet, NoOpAggNet, CatFuser)

from icecream import ic
import nvtx

S = Union[int, Tuple[int, ...]]
T = th.Tensor
rrm = recursive_replace_map

FEAT_KEY: str = '__feat__'
STATE_KEY: str = '__state__'


def _moduledict_flatten(d: nn.ModuleDict) -> Tuple[List[Any], Any]:
    return list(d.values()), list(d.keys())


def _moduledict_unflatten(
        values: List[nn.Module], context: Any) -> Dict[Any, Any]:
    return {key: value for key, value in zip(context, values)}


_register_pytree_node(nn.ModuleDict, _moduledict_flatten,
                      _moduledict_unflatten)


def _log_structured_tensor(k, x):
    print(k)
    if isinstance(x, dict):
        for kk, vv in x.items():
            _log_structured_tensor(kk, vv)
        return
    xx = x.reshape(-1, x.shape[-1])
    m = xx.mean(dim=0)
    s = xx.std(dim=0).mean()
    print('m', m, 's', s)


def _wrap_default(
        instance,
        default_fn,
        sentinel: str = '_from_default'):
    # <default branch>
    if hasattr(instance, sentinel) or instance is None:
        out = default_fn()
        object.__setattr__(out, sentinel, True)
        return out

    # <non-default branch>
    # return instance
    return None


class GenericStateEncoder(nn.Module):
    def __init__(self,
                 feature_encoders: Dict[str, nn.Module],
                 feature_aggregators: Dict[str, nn.Module],
                 feature_fuser: nn.Module,
                 state_aggregator: nn.Module):
        super().__init__()
        if isinstance(feature_encoders, Mapping):
            self.feature_encoders = nn.ModuleDict(
                feature_encoders)
        else:
            self.feature_encoders = feature_encoders

        if isinstance(feature_aggregators, Mapping):
            self.feature_aggregators = nn.ModuleDict(
                feature_aggregators)
        else:
            self.feature_aggregators = feature_aggregators

        self.feature_fuser = feature_fuser
        self.state_aggregator = state_aggregator

    def init_hidden(self, batch_shape: S,
                    *args, **kwds):
        # FIXME: introspection into cfg.dim_out
        # probably better to define @abstractproperty(dim_out)->S
        shapes = {}
        shapes[FEAT_KEY] = map_struct(
            self.feature_aggregators,
            lambda src, _: merge_shapes(batch_shape, src.cfg.dim_out),
            base_cls=AggregatorBase
        )
        # shapes = {k: merge_shapes(batch_shape,
        #                           self.feature_aggregators[k].cfg.dim_out)
        #           for k in self.feature_aggregators}
        shapes[STATE_KEY] = merge_shapes(batch_shape,
                                         self.state_aggregator.cfg.dim_out
                                         )
        return map_struct(shapes,
                          lambda src, _: th.zeros(src, *args, **kwds),
                          # FIXME: doesn't work for np.int* I guess
                          base_cls=(int, Tuple, List)
                          )

    def feature(self,
                observation: Dict[str, th.Tensor],
                ) -> Dict[str, th.Tensor]:
        out = {}
        with nvtx.annotate("feature"):
            # FIXME: assumes `observation` is a Dictionary now
            # which breaks compatibility with some envs.
            for k, v in observation.items():
                with nvtx.annotate(k):
                    if k not in self.feature_encoders:
                        continue
                    encoder = self.feature_encoders[k]
                    if isinstance(
                        encoder,
                        (StateDependentCrossFeatNet,
                         PatchWiseAttentionFeatNet,
                         PointPatchFeatNet,
                         PointPatchV2FeatNet,
                         PointPatchV3FeatNet,
                         PointPatchV4FeatNet,
                         PointPatchV5FeatNet,
                         PointPatchV7FeatNet,
                         PointPatchV8FeatNet
                         )):
                        # WARN: specialization !!
                        # requires full observation as extra context
                        out[k] = encoder(v, ctx=observation)
                    else:
                        # requires just the current key
                        out[k] = encoder(v)
        return out
        # return map_struct(self.feature_encoders,
        #                   lambda f, x: f(x),
        #                   observation,
        #                   base_cls=FeatureBase)

    def state(self,
              hidden: Dict[str, th.Tensor],
              action: th.Tensor,
              feature: Dict[str, th.Tensor]):
        """ Compute state autoregressively. """
        # FIXME:
        # Was a good try but does not currently work
        # def _map_hidden(f, h, x):
        #     return f(h, None, x)
        # new_feat = tree_map_n(_map_hidden,
        #                       self.feature_aggregators,
        #                       hidden[FEAT_KEY],
        #                       feature)

        # zip h, x
        hx = map_struct(
            hidden[FEAT_KEY],
            lambda h, x: (h, x),
            feature,
            base_cls=th.Tensor
        )
        new_feat = map_struct(
            self.feature_aggregators,
            lambda f, hx: f(hx[0], None, hx[1]),
            hx,
            base_cls=AggregatorBase
        )
        fused_feature = self.feature_fuser(new_feat)
        new_state = self.state_aggregator(
            hidden[STATE_KEY],
            action,
            fused_feature
        )
        new_hidden = {
            FEAT_KEY: new_feat,
            STATE_KEY: new_state
        }
        return (new_state, new_hidden)

    def state_seq(self,
                  feature: Dict[str, th.Tensor],
                  done: th.Tensor,
                  loop: bool = False
                  ):
        """
        Compute state directly from sequence.
        Requires that there's no feature-level aggregation.
        """
        new_feat = map_struct(
            self.feature_aggregators,
            lambda f, x: f(None, None, x),
            feature,
            base_cls=AggregatorBase
        )
        fused_feature = self.feature_fuser(new_feat)
        new_state = self.state_aggregator(
            None,
            None,
            fused_feature
        )
        return new_state

    def forward(self,
                hidden: Dict[str, th.Tensor],
                action: th.Tensor,
                observation: Dict[str, th.Tensor]):
        feature = self.feature(observation)

        # def _print(src, dst):
        #    print(src.shape, src.dtype, src.mean(), src.std())
        # map_struct(feature, _print,
        #           base_cls=th.Tensor)
        # for k, v in feature.items():
        #    print(k)
        #    print(v.shape, v.dtype, v.mean(), v.std())

        out = self.state(hidden, action, feature)
        # s, h = out
        # _log_structured_tensor('state', s)
        # _log_structured_tensor('hidden', h)
        return out

    # def to_parallel(self, device_ids):
    #     self.feature_encoders = nn.DataParallel(self.featur, device_ids)
    #     self.state = nn.DataParallel(self.feature, device_ids)


class MLPStateEncoder(GenericStateEncoder):
    """
    MLP state encoder with dictionary observations.
    """
    @dataclass
    class Config(ConfigBase):
        # domain: DomainConfig = DomainConfig()
        feature: Optional[Dict[str, FeatureBase.Config]] = None
        aggregator: Optional[Dict[str, AggregatorBase.Config]] = None
        fuser: Optional[FuserBase.Config] = None
        state: Optional[AggregatorBase.Config] = None
        obs_space: InitVar[Union[int, Dict[str, int], None]] = None
        act_space: InitVar[Optional[int]] = None

        def __post_init__(self, obs_space, act_space):
            # Populate missing fields from default,
            # and enforce invariants like dim_in/dim_obs/dim_act.
            # assert(isinstance(self.domain.obs_space, Mapping))
            # def _default_mlp(src):
            #     if self.feat_cls == 'noop':
            #         base = NoOpFeatNet.Config(dim_in=src)
            #     elif self.feat_cls == 'mlp':
            #         base = self.default_mlp
            #     out = replace(base, dim_in=src)
            #     return out

            # apply `dim_in` from `obs_space`
            if obs_space is not None:
                def _map_feature(src, dst):
                    # default-constructed config args take precedence
                    # to be overridden again from the (potentially updated)
                    # defaults.
                    # default = _wrap_default(dst,
                    #                         partial(_default_mlp, src))
                    # if default is not None:
                    #     return default
                    if src is not None:
                        dim_in = merge_shapes(src)
                    else:
                        dim_in = src
                    if isinstance(dst, FeatureBase.Config):
                        return replace(dst, dim_in=dim_in)
                    raise ValueError(F'Invalid cfg = {src}, {dst}')

                self.feature = map_struct(
                    obs_space,  # src
                    _map_feature,
                    self.feature,
                    # NOTE: `List,Tuple \in base_cls`
                    # indicates that two separate observations
                    # e.g. (state, goal) cannot be passed in as an iterable.
                    # Instead, use a dictionary instead: like
                    # {"state:(shape), "goal":(shape)}.
                    base_cls=(int, List, Tuple)
                )

            # def _map_mlp_v2(src,dst):
            # self.mlp = map_struct(
            #         self.mlp,
            #         lambda src, _: replace(src, dim_in=
            #         base_cls=FeatureBase.Config)

            # def _default_gru(agg_cls: str, dim_obs: S, dim_act: int = 0):
            #     if agg_cls == 'gru':
            #         base = self.default_gru
            #     elif agg_cls == 'mlp':
            #         base = MLPAggNet.Config(dim_obs=dim_obs,
            #                                 dim_out=self.default_gru.dim_out
            #                                 )
            #     else:
            #         base = NoOpAggNet.Config(dim_obs=dim_obs)
            #     out = replace(base,
            #                   dim_obs=dim_obs,
            #                   dim_act=dim_act)
            #     return out

            def _map_gru(src, dst):
                # First figure out `dim_obs`
                if not isinstance(src, FeatureBase.Config):
                    # raise KeyError(
                    #     F'Invalid feature config = {src}'
                    # )
                    return dst
                dim_obs = merge_shapes(src.dim_out)

                if not isinstance(dst, AggregatorBase.Config):
                    return dst

                # default-constructed config args take precedence
                # to be overridden again from the (potentially updated)
                # defaults.
                # default = _wrap_default(dst, partial(
                #     _default_gru,
                #     self.feat_agg_cls, dim_obs
                # ))
                # if default is not None:
                #     return default

                # NOTE: `replace` means that
                # dst itself not overwritten
                return replace(dst,
                               dim_obs=dim_obs,
                               # NOTE: we don't incorporate
                               # action inputs during feature aggreation!
                               dim_act=0)

            self.aggregator = map_struct(
                self.feature,
                _map_gru,
                self.aggregator,
                base_cls=FeatureBase.Config
            )

            # Configure fuser.
            # try:
            if isinstance(self.aggregator, Mapping):
                def _get_input_dims(src, dst):
                    if src is None:
                        return merge_shapes(0)
                    else:
                        return merge_shapes(src.dim_out)
                input_dims = map_struct(
                    self.aggregator,
                    _get_input_dims,
                    base_cls=AggregatorBase.Config
                )
                fuser = replace(
                    self.fuser,
                    input_dims=input_dims
                )
                self.fuser = fuser
            # except AttributeError:
            #     pass

            # Configure state aggregator.
            # default_state = _wrap_default(
            #     self.state,
            #     partial(
            #         _default_gru,
            #         self.state_agg_cls,
            #         dim_obs=self.fuser.mlp.dim_out,
            #         dim_act=self.domain.num_act))
            # if default_state is not None:
            #     self.state = default_state
            if isinstance(self.state, AggregatorBase.Config):
                dim_act: int = (0 if act_space is None else act_space)
                print(F'got dim_act={dim_act}')
                self.state = replace(
                    self.state,
                    dim_obs=merge_shapes(self.fuser.dim_out),
                    dim_act=dim_act)
            # else:
            #     raise ValueError('should not reach here')

    @classmethod
    def from_domain(cls, domain_cfg: DomainConfig, **kwds):
        cfg = recursive_replace_map(cls.Config(domain=domain_cfg),
                                    kwds)
        return cls.from_config(cfg)

    @classmethod
    def from_config(cls, cfg: Config):
        def feat_cls_fn(c: FeatureBase.Config):
            # FIXME: needs to be placed before
            # MLPFeatNet...
            # FIXME: can we somehow avoid this type of
            # brute enumerations...??
            if isinstance(c, FlatMLPFeatNet.Config):
                return FlatMLPFeatNet(c)
            if isinstance(c, MLPFeatNet.Config):
                return MLPFeatNet(c)
            if isinstance(c, CrossFeatNet.Config):
                return CrossFeatNet(c)
            if isinstance(c, StateDependentCrossFeatNet.Config):
                return StateDependentCrossFeatNet(c)
            if isinstance(c, PatchWiseAttentionFeatNet.Config):
                return PatchWiseAttentionFeatNet(c)
            if isinstance(c, PointPatchFeatNet.Config):
                return PointPatchFeatNet(c)
            if isinstance(c, PointPatchV2FeatNet.Config):
                return PointPatchV2FeatNet(c)
            if isinstance(c, PointPatchV3FeatNet.Config):
                return PointPatchV3FeatNet(c)
            if isinstance(c, PointPatchV4FeatNet.Config):
                return PointPatchV4FeatNet(c)
            if isinstance(c, PointPatchV5FeatNet.Config):
                return PointPatchV5FeatNet(c)
            if isinstance(c, PointPatchV7FeatNet.Config):
                return PointPatchV7FeatNet(c)
            if isinstance(c, PointPatchV8FeatNet.Config):
                return PointPatchV8FeatNet(c)
            if isinstance(c, FlattenFeatNet.Config):
                return FlattenFeatNet(c)
            elif isinstance(c, NoOpFeatNet.Config):
                return NoOpFeatNet(c)
            else:
                raise KeyError(F'Unknown feat_cls={c}')

        def agg_fn(c: AggregatorBase.Config):
            if isinstance(c, MLPAggNet.Config):
                return MLPAggNet(c)
            elif isinstance(c, GRUAggNet.Config):
                return GRUAggNet(c)
            elif isinstance(c, NoOpAggNet.Config):
                return NoOpAggNet(c)
            else:
                raise KeyError(F'Unknown cfg={c}')

        # feat_agg_cls = (GRUAggNet if cfg.feat_agg_cls == 'gru'
        #                 else MLPAggNet)
        # state_agg_cls = (GRUAggNet if cfg.state_agg_cls == 'gru'
        #                  else MLPAggNet)

        feature_encoders = map_struct(
            cfg.feature,
            lambda src, dst: feat_cls_fn(src),
            # base_cls=MLPFeatNet.Config
            base_cls=FeatureBase.Config
        )

        feature_aggregators = map_struct(
            cfg.aggregator,
            lambda src, dst: agg_fn(src),
            # base_cls=GRUAggNet.Config
            base_cls=AggregatorBase.Config
        )

        def _get_input_dims(src, dst):
            return merge_shapes(src.dim_out)
        input_dims = map_struct(
            cfg.aggregator,
            _get_input_dims,
            base_cls=AggregatorBase.Config
        )
        if isinstance(cfg.fuser, MLPFuser.Config):
            feature_fuser = MLPFuser(cfg.fuser,
                                     input_dims=input_dims)
        elif isinstance(cfg.fuser, CatFuser.Config):
            feature_fuser = CatFuser(cfg.fuser,
                                     input_dims=input_dims)
        else:
            raise KeyError(F'Unknown cfg.fuser={cfg.fuser}')

        state_aggregator = agg_fn(cfg.state)

        return cls(
            feature_encoders,
            feature_aggregators,
            feature_fuser,
            state_aggregator
        )


def test_inference():
    from env.random_env import RandomEnv
    domain_cfg = DomainConfig(
        num_env=8,
        obs_space={'a': 64, 'b': 32},
        num_act=6
    )
    state_encoder = MLPStateEncoder.from_domain(
        domain_cfg)
    ic(state_encoder)
    env = RandomEnv(domain_cfg)

    prev_hidden = state_encoder.init_hidden(
        batch_shape=domain_cfg.num_env)
    action = th.zeros((domain_cfg.num_env, domain_cfg.num_act),
                      device=env.device)
    obs = env.reset()
    curr_state, hidden = state_encoder(
        prev_hidden, action, obs)


def test_create_with_noop():
    domain_cfg = DomainConfig(
        num_env=8,
        obs_space={
            'object_state': 7,
            'cube_state': 7,
            'goal': 3
        },
        num_act=6
    )
    state_cfg = MLPStateEncoder.Config(
        domain=domain_cfg)
    state_cfg.mlp['goal'] = NoOpFeatNet.Config(
        dim_in=state_cfg.mlp['goal'].dim_in,
        dim_out=state_cfg.mlp['goal'].dim_out,
    )
    state_cfg.gru['goal'] = NoOpAggNet.Config(
        dim_obs=state_cfg.mlp['goal'].dim_out,
    )
    state_cfg.__post_init__()
    ic(state_cfg)

    state_net = MLPStateEncoder.from_config(state_cfg)
    ic(state_net)


def test_linear_agg():
    domain_cfg = DomainConfig(
        num_env=8,
        obs_space={
            'object_state': 7,
            'cube_state': 7,
            'goal': 3
        },
        num_act=6
    )

    ic('default')
    state_cfg = MLPStateEncoder.Config(
        domain=domain_cfg)
    ic(state_cfg)

    # state_cfg = recursive_replace_map(state_cfg, {
    #     'feat_agg_cls': 'mlp'}
    # )
    state_cfg = replace(state_cfg, feat_agg_cls='mlp')
    ic(state_cfg)


def test_override_defaults():
    domain_cfg = DomainConfig(
        num_env=8,
        obs_space={
            'object_state': 7,
            'cube_state': 7,
            'goal': 3
        },
        num_act=6
    )

    ic('default')
    state_cfg = MLPStateEncoder.Config(
        domain=domain_cfg)
    ic(state_cfg)

    ic('update default act_cls')
    state_cfg = recursive_replace_map(state_cfg, {
        'default_mlp.act_cls': 'tanh'}
    )
    ic(state_cfg)

    ic('manually update goal mlp')
    updated_mlp_cfg = dict(state_cfg.mlp)
    updated_mlp_cfg['goal'] = MLPFeatNet.Config(
        dim_in=3,
        dim_out=18,
        act_cls='relu'
    )
    state_cfg = replace(state_cfg, mlp=updated_mlp_cfg)
    ic(state_cfg)

    ic('update default act_cls again')
    state_cfg = recursive_replace_map(state_cfg, {
        'default_mlp.act_cls': 'relu'}
    )
    ic(state_cfg)

    ic('update default act_cls again')
    state_cfg = recursive_replace_map(state_cfg, {
        'default_mlp.act_cls': 'tanh'}
    )
    ic(state_cfg)


def test_transfer():
    a_cfg = DomainConfig(
        num_env=8,
        obs_space={'goal': 3, 'object_state': 7},
        num_act=6)
    a_encoder = MLPStateEncoder.from_domain(a_cfg)

    ab_cfg = DomainConfig(
        num_env=16,
        obs_space={'goal': 3, 'object_state': 7,
                   'robot_state': 7},
        num_act=6)
    ab_encoder = MLPStateEncoder.from_domain(ab_cfg)

    if True:
        source_dict = {k: v for (k, v) in a_encoder.state_dict().items()
                       if (
            'feature_encoders.goal' in k or
            'feature_encoders.object_state' in k
        )}
        ab_encoder.load_state_dict(
            source_dict,
            strict=False)

    if True:
        keys = transfer(
            ab_encoder,
            a_encoder.state_dict(),
            substrs=[
                'feature_encoders.goal',
                'feature_encoders.object_state',
                'feature_aggregators.goal',
                'feature_aggregators.object_state',
            ])
        print('== TRANSFER ==')
        print('missing keys', keys.missing_keys)
        print('unexpected keys', keys.unexpected_keys)
        print('==============')


def test_with_imagelike_inputs():
    from env.random_env import RandomEnv
    domain_cfg = DomainConfig(
        num_env=8,
        obs_space={
            'image': (2, 96, 96),
            'cube_state': 7,
            'goal': 3
        },
        num_act=6
    )
    cfg = MLPStateEncoder.Config(domain=domain_cfg)
    # TODO: rename `mlp` -> `feat`?
    channels = [16, 64, 128, 256]
    cfg.mlp['image'] = CNNFeatNet.Config(
        dim_in=2,
        dim_out=128,
        block_args=[dict(
            channels=c,
            kernel_size=2,
            stride=2,
            padding=0) for c in channels]
    )
    cfg.__post_init__()
    state_encoder = MLPStateEncoder.from_config(cfg)

    ic(state_encoder)
    env = RandomEnv(domain_cfg)

    prev_hidden = state_encoder.init_hidden(
        batch_shape=domain_cfg.num_env)
    action = th.zeros((domain_cfg.num_env, domain_cfg.num_act),
                      device=env.device)
    obs = env.reset()
    curr_state, hidden = state_encoder(
        prev_hidden, action, obs)

    def _print(src, dst):
        print(src.shape, src.dtype)

    print('state')
    _print(curr_state, None)

    print('hiddens')
    map_struct(hidden, _print,
               base_cls=th.Tensor)


def main():
    test_with_imagelike_inputs()


if __name__ == '__main__':
    main()
