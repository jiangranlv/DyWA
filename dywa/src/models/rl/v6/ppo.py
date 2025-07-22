##!/usr/bin/env python3
"""
Simplified version of PPO.
"""

from env.env.base import EnvIface

from pathlib import Path
from typing import (
    Dict, Optional, Tuple, Union, Iterable, Any, Mapping)
from dataclasses import dataclass
from util.config import ConfigBase
from collections import defaultdict
from functools import partial
from contextlib import ExitStack

import numpy as np
import torch as th
import torch.nn as nn
import einops

from torch.utils.tensorboard import SummaryWriter

from models.rl.padded_buffer import DictBuffer
from models.rl.kl_adaptive_lr_scheduler import KLAdaptiveLRScheduler
from models.rl.ppo_util import (
    gae_ax1,
    get_bound_loss,
    BoundLoss,
    PPOLossAndLogs
)
from models.rl.ppo_config import (DomainConfig, TrainConfig)
from util.torch_util import module_train_mode
from util.path import RunPath, ensure_directory
from models.common import grad_step

import nvtx
from tqdm.auto import tqdm

from models.rl.net.rnn_loop import loop_rnn
from models.rl.policy_util import get_action_distribution
from models.rl.util import mixed_reset

from train.ckpt import save_ckpt, load_ckpt
from train.util import add_scalar

from models.rl.generic_state_encoder import (
    STATE_KEY, FEAT_KEY)
from models.common import map_struct, map_tensor

from icecream import ic

S = Union[int, Tuple[int, ...]]
T = th.Tensor


class PPO(nn.Module):

    @dataclass
    class Config(ConfigBase):
        # Training related
        train: TrainConfig = TrainConfig()
        # Neural network device
        device: str = 'cuda:0'
        # Optionally bootstrap rewards on timeout.
        bootstrap_timeout: bool = True
        rebuild_batch: bool = True
        # Reset state across episode boundaries.
        # Some papers actually recommend skipping this reset.
        reset_state: bool = True

        # bptt_* options is a complete specification of
        # recurrent state aggregation, fully determined
        # by three parameters: horizon, burn_in, and stride.
        # How long to propagate gradients for BPTT
        bptt_horizon: int = 2
        # How many "burn-in" samples to use prior to grad. computation
        # for refreshing the hidden states
        bptt_burn_in: int = 6
        # Configuring stride < bptt_seq_len
        # allows for each state in the overlapping region
        # to incorporate more frames (obs+act) than
        # would otherwise be possible.
        bptt_stride: int = 4

        cut_actor_grad: bool = False
        accumulate_gradient: int = 1
        tanh_xfm: bool = False
        adam_eps: Optional[float] = None
        stable_normal: bool = True
        max_grad_norm: float = 1.0

        # Let's try to track the PPO config versions
        # so that we can load old configs seamlessly.
        version: int = 4

        @property
        def bptt_seq_len(self):
            """
            Total length of the sequence to use for
            temporal aggregation.
            """
            return self.bptt_horizon + self.bptt_burn_in

    def __init__(self,
                 cfg: Config,
                 env: EnvIface,
                 state_net: nn.Module,
                 actor_net: nn.Module,
                 value_net: nn.Module,
                 path: Optional[RunPath] = None,
                 writer: Optional[SummaryWriter] = None,
                 extra_nets: Optional[nn.Module] = None
                 ):
        super().__init__()
        self.stable_normal = (cfg.stable_normal or cfg.train.use_amp)

        self.cfg = cfg
        self.domain_cfg = DomainConfig.from_env(env)
        self.env = env
        self.device = th.device(cfg.device)
        self.state_net = state_net
        self.actor_net = actor_net
        self.value_net = value_net
        self.path = path
        self.writer = writer

        # Framework to deal with auxiliary modules & losses
        self.extra_nets = None
        if extra_nets is not None:
            if isinstance(extra_nets, nn.Module):
                # probably passed in as a module
                self.extra_nets = extra_nets
            else:
                if isinstance(extra_nets, Mapping):
                    # probably passed in as a dict
                    self.extra_nets = nn.ModuleDict(extra_nets)
                elif isinstance(extra_nets, Iterable):
                    # probably passed in as a list
                    self.extra_nets = nn.ModuleList(extra_nets)
                else:
                    raise ValueError(F'Invalid extra_nets = {extra_nets}')

        learn: bool = (cfg.train.train_steps > 0)

        if learn:
            self.ppo_loss: Optional[PPOLossAndLogs] = PPOLossAndLogs(
                cfg.train.loss.normalize_adv,
                cfg.train.loss.adv_eps,
                cfg.train.loss.clip,
                (cfg.train.loss.max_dv if cfg.train.loss.clip_value else None)
            )
            self.bound_loss: Optional[BoundLoss] = (
                get_bound_loss(
                    env.action_space,
                    self.domain_cfg,
                    cfg.train.loss.k_bound)
            )

            self.data = DictBuffer(
                cfg.train.rollout_size + cfg.bptt_seq_len + 1,
                self.device)

            # policy / critic optimizer
            eps = cfg.adam_eps
            if eps is None:
                eps = 1e-8
            self.optimizer = th.optim.Adam(
                self.parameters(),
                lr=cfg.train.lr,
                eps=eps
            )
            self.scheduler = KLAdaptiveLRScheduler(
                cfg.train.alr,
                self.optimizer)

            # Optionally enable mixed-precision training.
            if cfg.train.use_amp:
                self.scaler = th.cuda.amp.GradScaler()
            else:
                self.scaler = None
            self.optim_step: int = 0

            # Framework to deal with auxiliary losses.
            self._aux_losses = {}
            for k, m in self.named_modules():
                if hasattr(m, 'hook_aux_loss'):
                    m.hook_aux_loss(partial(self._add_aux_loss, k))

    def _add_aux_loss(self,
                      key: str,
                      module: nn.Module,
                      inputs: Iterable[th.Tensor],
                      loss: th.Tensor):
        self._aux_losses[key] = loss

    @nvtx.annotate("compute_state")
    def _compute_state(self,
                       D: Dict[str, th.Tensor],
                       dt: int = 1,
                       **kwds):
        """
        Compute "current state" based on
        forward aggregation for `dt` steps.
        This routine assumes self.state_net is `MLPStateEncoder`.
        """
        cfg = self.cfg
        stride: int = cfg.bptt_stride
        L = len(D['done'])
        S: int = L - dt

        def _S(x: th.Tensor, i: int, d: int = 1):
            """ slice of size `S`, starting from offset `i`. """
            return x[i:i + S:d]

        with nvtx.annotate("feature"):
            feature = self.state_net.feature(D['obsn'])

        with nvtx.annotate("init_hidden"):
            # [1] Allocate hiddens.
            def _init_hidden(src, dst):
                # [1] Allocate hiddens.
                if dst is None:
                    dst = th.empty_like(src[0:S + dt])
                # [2] Initialize values from data buffer.
                dst[0:S + dt:stride] = src[0:S + dt:stride]
                return dst
            hidden = map_tensor(D['hidden'], _init_hidden)

        with nvtx.annotate("state"):
            hidden = loop_rnn(self.state_net.state,
                              hidden,
                              D['actn'],
                              feature,
                              D['done'],
                              cfg.bptt_seq_len,
                              cfg.bptt_burn_in,
                              cfg.bptt_stride,
                              # NOTE: alpha=1.0 is desirable
                              # primarily because we assume no aggregation for
                              # now.
                              alpha=1.0)
            states = hidden[STATE_KEY]
            states = states[dt:S + dt]

        # Alternatively, at some point
        # (e.g. with Transformer):
        # states = loop_parallel(...)

        return states

    @nvtx.annotate("compute_targets")
    def _compute_targets(self, states, D: Dict[str, th.Tensor], dt: int):
        cfg = self.cfg
        N: int = cfg.train.chunk_size
        T: int = cfg.train.td_horizon
        S: int = (N * T)

        def _S(x: th.Tensor, i: int, d: int = 1):
            return x[i:i + S:d]

        def _R(x: th.Tensor):
            return x.reshape(N, T, *x.shape[1:])

        with th.no_grad():
            # We assume `states` index into D[dt:].
            values = self.value_net(states)
            prev_value = _R(_S(values, 0))
            curr_value = _R(_S(values, 1))
            prev_done = _R(_S(D['done'], dt))
            curr_done = _R(_S(D['done'], dt + 1))
            prev_mask = ~prev_done
            curr_rewd = _R(_S(D['rewd'], dt + 1))
            sigma = 0.0  # FIXME: unused legacy parameter
            advn = gae_ax1(curr_rewd, prev_value, curr_value,
                           prev_done, curr_done,
                           cfg.train.loss.gamma,
                           cfg.train.loss.lmbda,
                           sigma).detach_()
            retn = (prev_value + advn)

        # NOTE: outputting "prev_states"
        # i.e. right before the `action`.
        states = _R(_S(states, 0))

        return {
            'state': states,
            'value': values,
            'advantage': advn,
            'return': retn,
            'prev_mask': prev_mask,
            'reward': curr_rewd,
            'sigma': sigma
        }

    @nvtx.annotate("build_batch")
    def _build_batch(self, D: Dict[str, th.Tensor]):
        """
        Batch for training data.
        Note that the gradients for `state` must be preserved.
        """
        cfg = self.cfg

        # OUTPUT = (chunk_size X td_horizon X num_env X feature_dims)
        N: int = cfg.train.chunk_size
        T: int = cfg.train.td_horizon
        S: int = (N * T)
        dt: int = cfg.bptt_seq_len

        def _S(x: th.Tensor, i: int, d: int = 1):
            return x[i:i + S:d]

        def _R(x: th.Tensor):
            return x.reshape(N, T, *x.shape[1:])

        # Compute current (and previous) states
        # based on n-steps of aggregation.
        # NOTE: `states` now refer to a slice for
        # the timesteps correspondong to D[dt:].
        states = self._compute_state(D, dt)
        outputs = self._compute_targets(states, D, dt)

        # TODO: validate correctness of index offset
        outputs['action'] = _R(_S(D['actn'], dt + 1))
        outputs['old_logp'] = _R(_S(D['logp'], dt + 1))

        # FIXME: generalize
        if (self.extra_nets is not None) and ('trans_net' in self.extra_nets):
            self.extra_nets['trans_net'](
                outputs['state'], outputs['action'])

        return outputs, {}

    @nvtx.annotate("train_batch")
    def _train_batch(self,
                     batch: Dict[str, th.Tensor],
                     log_s: Dict[str, th.Tensor],
                     update_kl: bool,
                     enable_es: bool,

                     step_grad: bool = True,
                     loss_scale: Optional[float] = None,
                     **kwds) -> bool:
        """
        Returns:
            Flag to determine whether to continue training.
        """
        cfg = self.cfg

        # Fetch data.
        old_logp = batch['old_logp']
        old_val = batch['value']
        mask = batch['prev_mask']
        action = batch['action']
        state = batch['state']
        returns = batch['return']
        advantages = batch['advantage']

        # == model inference ==
        aux = {}
        dist = get_action_distribution(
            (state.detach() if cfg.cut_actor_grad else state),
            self.actor_net,
            self.domain_cfg.discrete,
            cfg.tanh_xfm,
            aux=aux,
            stable_normal=self.stable_normal)
        val = self.value_net(state)
        if (not self.domain_cfg.discrete) and (cfg.tanh_xfm):
            new_logp = dist.log_prob(action.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        else:
            new_logp = dist.log_prob(action)

        # Losses -- PPO
        losses, extra_logs = self.ppo_loss(
            action, advantages,
            old_logp, new_logp,
            returns, val,
            mask,
            old_val)

        # Losses -- Bound
        if ('mu' in aux) and (self.bound_loss is not None):
            mu = aux['mu']
            losses['bound'] = self.bound_loss(mu, mask)
        else:
            losses['bound'] = 0.0

        # Accumulate losses.
        with nvtx.annotate("sum_losses"):
            # policy + value + entropy + bounds-regularization
            loss = (
                (cfg.train.loss.k_pi * losses['policy']) +
                (cfg.train.loss.k_val * losses['value']) +
                (0.0 if (cfg.train.loss.k_ent <= 0)
                    else (cfg.train.loss.k_ent * losses['ent'])) +
                (cfg.train.loss.k_bound * losses['bound'])
            )

            # Deal with auxiliary losses
            for key, aux_loss in self._aux_losses.items():
                loss += cfg.train.loss.k_aux.get(key) * aux_loss
                log_s[F'loss/aux/{key}'].append(aux_loss)  # .detach_())
            # Clear aux loss cache
            self._aux_losses = {}

            loss = loss.mean()
            if loss_scale is not None:
                loss = loss * loss_scale

        def add_logs():

            # Add log_s
            with nvtx.annotate("log"):
                if extra_logs is not None:
                    for k, v in extra_logs.items():
                        log_s[F'log/{k}'].append(v)
                log_s['loss/policy'].append(losses['policy'].detach_())
                log_s['loss/value'].append(losses['value'].detach_())
                log_s['loss/entropy'].append(losses['ent'].detach_())

                if 'bound' in losses:
                    if isinstance(losses['bound'], th.Tensor):
                        log_s['loss/bound'].append(losses['bound'].detach_())
                    else:
                        log_s['loss/bound'].append(losses['bound'])

                if not self.domain_cfg.discrete:
                    log_s['log/logvar'].append(aux['ls'])

            with th.no_grad():
                log_s['log/avg_val'].append(val.mean().detach_())
                log_s['log/avg_ret'].append(returns.mean().detach_())
                log_s['log/std_val'].append(val.std().detach_())
                log_s['log/std_ret'].append(returns.std().detach_())
                log_s['log/std_state'].append(state.reshape(-1,
                                                            state.shape[-1]).std(dim=0).mean().detach_())

            log_s['loss/total'].append(loss.detach_())

            # NOTE: learning rate adjustment
            # probably has to be before the
            # early-stopping measures.
            with nvtx.annotate("ALR"):
                log_s['log/learning_rate'].append(
                    self.cfg.train.lr * self.scheduler.scale)

        # Early stopping logic
        approx_kl = extra_logs['approx_kl']
        if step_grad:
            if enable_es:
                should_es = (approx_kl.item() >
                             cfg.train.alr.factor * cfg.train.alr.kl_target)
                if should_es:
                    if cfg.train.alr.use_alr and update_kl:
                        self.scheduler.update_kl(approx_kl)
                    add_logs()
                    return False

        # Optimization (sgd steps)
        with nvtx.annotate("OPT"):
            grad_step(loss,
                      self.optimizer, self.scaler,
                      None,
                      cfg.max_grad_norm,
                      step_grad=step_grad,
                      retain_graph=False,
                      zero_grad=False)
            if step_grad:
                self.optimizer.zero_grad(set_to_none=True)
                if cfg.train.alr.use_alr:
                    if update_kl:
                        self.scheduler.update_kl(approx_kl)
                    self.scheduler.step()
                self.optim_step += 1

        add_logs()

        return True

    @nvtx.annotate("train_epoch")
    def _train_epoch(self):
        cfg = self.cfg

        E: int = self.domain_cfg.num_env
        P: int = cfg.accumulate_gradient
        assert ((E % P) == 0)
        mini_batch_size = E // P

        D = self.data.get()

        if P > 1:
            def _split_env(src, dst):
                out = einops.rearrange(src, 't (p k) ... -> t p k ...',
                                       p=P,
                                       k=mini_batch_size)
                out = th.unbind(out, dim=1)  # p x (t k ...)
                return out
            D = map_tensor(D, _split_env)

        opt0: int = self.optim_step
        log_s = defaultdict(list)  # scalar logs

        if not cfg.rebuild_batch:
            raise ValueError('rebuild_batch=False: temporarily not supported')
            with th.cuda.amp.autocast(enabled=cfg.train.use_amp):
                batches, bb_logs = self._build_batch(self.data.get())
                if bb_logs is not None:
                    for k, v in bb_logs.items():
                        log_s[k].append(v)
            log_s['log/sigma'].append(batches.pop('sigma', 0.0))

        updated_policy: bool = False
        with module_train_mode(self, True):
            for i in range(cfg.train.epoch):
                for j in range(P):

                    if P > 1:
                        def _index(src, _):
                            return src[j]

                        def _is_leaf(src, _):
                            return (isinstance(src, th.Tensor) or (isinstance(
                                src, tuple) and isinstance(src[0], th.Tensor)))
                        D_j = map_struct(D, _index,
                                         base_fn=_is_leaf,
                                         base_cls=th.Tensor
                                         )
                    else:
                        D_j = D

                    if cfg.rebuild_batch:
                        # Recompute states and TD targets.
                        with th.cuda.amp.autocast(enabled=cfg.train.use_amp):
                            batches, bb_logs = self._build_batch(D_j)
                        if bb_logs is not None:
                            for k, v in bb_logs.items():
                                log_s[k].append(v)
                        log_s['log/sigma'].append(batches.pop('sigma', 0.0))

                    # FIXME: what if the returned batch does not match
                    # chunk_size??
                    batch_indices = np.arange(cfg.train.chunk_size)
                    if cfg.train.shuffle:
                        np.random.shuffle(batch_indices)

                    for (ii, bi) in enumerate(batch_indices):
                        batch = {k: v[bi] for (k, v) in batches.items()}

                        with nvtx.annotate("LOOP"):
                            with th.cuda.amp.autocast(enabled=cfg.train.use_amp):
                                # NOTE: the _first_ kl divergence
                                # is pretty much guaranteed to have
                                # zero kl div, which will increase
                                # the learning rate. I think this is
                                # dangerous, so disabling KL-adaptive
                                # LR scheduler updates on the very first update
                                step_grad: bool = (j == (P - 1))
                                continue_training = self._train_batch(
                                    batch, log_s, update_kl=updated_policy, enable_es=(
                                        cfg.train.use_early_stopping and updated_policy),
                                    step_grad=step_grad,
                                    loss_scale=(1.0 / P)
                                )
                                if not continue_training:
                                    break
                                updated_policy = (updated_policy or step_grad)
                    if not continue_training:
                        break

        # (3) add some logs
        opt1: int = self.optim_step
        log_s['log/num_sgd_step'].append(opt1 - opt0)
        log_s['log/optim_step'].append(self.optim_step)

        # NOTE: must be called __after__
        # using the data buffers.
        self.data.clear()
        assert (len(self.data) == 0)

        return dict(log_s)

    def init_buf(self, obs: th.Tensor, hidden: th.Tensor, done: th.Tensor):
        cfg = self.cfg
        if self.domain_cfg.discrete:
            actn = th.zeros((self.domain_cfg.num_env,),
                            dtype=th.int32, device=self.device)
        else:
            actn = th.zeros((self.domain_cfg.num_env, self.domain_cfg.num_act),
                            dtype=th.float, device=self.device)
        logp = th.zeros((self.domain_cfg.num_env),
                        dtype=th.float, device=self.device)
        rewd = th.zeros((self.domain_cfg.num_env,),
                        dtype=th.float, device=self.device)

        with nvtx.annotate("init_buf()"):
            self.data.add_field_from_tensor('actn', actn)
            self.data.add_field_from_tensor('logp', logp)
            self.data.add_field_from_tensor('rewd', rewd)
            self.data.add_field_from_tensor('obsn', obs)
            self.data.add_field_from_tensor('done', done)
            self.data.add_field_from_tensor('hidden', hidden)

            self.data.append({
                'obsn': obs,
                'done': done,
                'rewd': rewd,
                'hidden': hidden
            })

    @nvtx.annotate("RPPO.interact()")
    def interact(self,
                 step: int,
                 hidden0: th.Tensor,
                 done0: th.Tensor,
                 store: bool = True,
                 extra: Optional[Dict[str, Any]] = None
                 ):
        """
        Interact with the environment.
        """
        cfg = self.cfg
        aux = {}
        with module_train_mode(self, False):
            with th.no_grad():
                # reset state0 (in-place),
                # in case done0 was True.
                with nvtx.annotate("reset"):
                    if cfg.reset_state:
                        reset_tensor = (~done0)[..., None]
                        map_tensor(
                            hidden0, lambda src, _: src.mul_(reset_tensor))

                with nvtx.annotate("get_action"):
                    with th.cuda.amp.autocast(enabled=cfg.train.use_amp):
                        # [1] Get and evaluate action
                        dist = get_action_distribution(
                            hidden0[STATE_KEY],
                            self.actor_net, self.domain_cfg.discrete, cfg.tanh_xfm,
                            aux=aux, stable_normal=self.stable_normal)
                        actn = dist.sample()

                with nvtx.annotate("logp"):
                    if (not self.domain_cfg.discrete) and (cfg.tanh_xfm):
                        logp = dist.log_prob(
                            actn.clamp(-1.0 + 1e-6, 1.0 - 1e-6))
                    else:
                        logp = dist.log_prob(actn)

                with nvtx.annotate("env-step"):
                    # [2] step env
                    with th.cuda.amp.autocast(False):
                        obsn, rewd, done, info = self.env.step(actn)

                with nvtx.annotate("update-state"):
                    # obsn = deepcopy(obsn)
                    state1, hidden1 = self.state_net(hidden0, actn, obsn)

                with nvtx.annotate("extra"):
                    if extra is not None:
                        assert (isinstance(extra, Mapping))
                        extra['actn'] = actn
                        extra['obsn'] = obsn
                        extra['info'] = info
                        extra['rewd'] = rewd

                with nvtx.annotate("boot-time"):
                    # [3] bootstrap value on timeout
                    if ('timeout' in info) and (cfg.bootstrap_timeout):
                        timeout = info['timeout']
                        val1 = self.value_net(state1)
                        rewd = th.where(
                            timeout,
                            rewd + cfg.train.loss.gamma * val1,
                            rewd)

                with nvtx.annotate("store"):
                    # [5] append data.
                    if store:
                        self.data.append({
                            # from prev timestep
                            'actn': actn,
                            'logp': logp,
                            # from curr timestep
                            'obsn': obsn,
                            'rewd': rewd,
                            'done': done,
                            'hidden': hidden1
                        })

                # [4] reset state.
                # NOTE: in-place modification to `state1` is ok
                # since self.data stores the state to a copied buffer.
                # state1 *= (~done)[..., None]
        return hidden1, done

    def load(self, path: str, strict: bool = False):
        load_ckpt(dict(self=self),
                  ckpt_file=path,
                  strict=strict)

    def save(self, path: str):
        ensure_directory(Path(path).parent)
        save_ckpt(dict(self=self),
                  ckpt_file=path)

    def init(self, obs):
        """
        Compute initial buffers.
        """
        with module_train_mode(self, False):
            with th.no_grad():
                init_hidden = self.state_net.init_hidden
                hidden = init_hidden(
                    batch_shape=self.domain_cfg.num_env,
                    dtype=th.float,
                    device=self.device)
                prev_action = th.zeros(
                    (self.domain_cfg.num_env, self.domain_cfg.num_act),
                    dtype=th.float, device=self.device)

                state, hidden = self.state_net(hidden, prev_action, obs)
                done = th.zeros((self.domain_cfg.num_env,),
                                dtype=bool,
                                device=self.device)
        return (prev_action, state, hidden, done)

    @nvtx.annotate("rppo.test")
    def test(self, sample: bool = True, steps: int = 8192):
        cfg = self.cfg
        if False:  # cfg.train.mixed_reset:
            with th.cuda.amp.autocast(False):
                obs = mixed_reset(self.env,
                                  self.domain_cfg.num_env,
                                  self.device,
                                  timeout=self.domain_cfg.timeout,
                                  num_steps=self.domain_cfg.timeout)
        else:
            with th.cuda.amp.autocast(False):
                obs = self.env.reset()
        prev_action, state, hidden, done = self.init(obs)

        # ic(obs.std(dim=0), obs.min(dim=0), obs.max(dim=0))
        yield (None, obs, None, None, None)

        with ExitStack() as stack:
            stack.enter_context(module_train_mode(self, False))
            stack.enter_context(nvtx.annotate("RPPO.test.step"))
            stack.enter_context(th.no_grad())
            pbar = stack.enter_context(tqdm(range(steps), desc='rppo.test'))
            for step in pbar:
                extra = {}
                hidden, done = self.interact(
                    step, hidden, done,
                    store=False, extra=extra)
                yield extra['actn'], extra['obsn'], extra['rewd'], done, extra['info']

    def _maybe_train(self, step: int):
        """
        Train if data is full, otherwise return (no-op).
        Thin wrapper around _train_epoch().
        """
        if not self.data.full:
            return

        with module_train_mode(self, True):
            with th.enable_grad():
                log_s = self._train_epoch()

        # [6] log.
        # NOTE: we use run_step - 1
        # since `maybe_train()` is invoked
        # after the step increment in interact().
        with nvtx.annotate("log"):
            writer = self.writer
            add_scalar(writer, 'log/run_step', step,
                       global_step=step)
            for k, v in log_s.items():
                add_scalar(writer, k, v, global_step=step)

    def _maybe_save(self, step: int):
        """
        Periodically save.
        Thin wrapper around save().
        """
        cfg = self.cfg
        if step % cfg.train.save_period == 0:
            ckpt_file = self.path.ckpt / F'step-{step:05d}.ckpt'
            self.save(ckpt_file)

    def learn(self, **kwds):
        """
        This is the main training entrypoint.
        """
        cfg = self.cfg

        # TODO: `mixed_reset` should probably
        # live outside of RPPO/PPO...
        if cfg.train.mixed_reset:
            with th.cuda.amp.autocast(False):
                obs = mixed_reset(self.env,
                                  self.domain_cfg.num_env,
                                  self.device,
                                  timeout=self.domain_cfg.timeout,
                                  num_steps=self.domain_cfg.timeout)
        else:
            with th.cuda.amp.autocast(False):
                obs = self.env.reset()
        prev_action, state, hidden, done = self.init(obs)
        self.init_buf(obs, hidden, done)

        name = kwds.pop('name', None)
        if name is not None:
            desc = name
        else:
            desc = 'rppo.learn'
        with tqdm(
                range(cfg.train.train_steps),
                disable=(not cfg.train.use_tqdm),
                desc=desc) as pbar:
            for step in pbar:
                hidden, done = self.interact(step, hidden, done)
                self._maybe_train(step)
                self._maybe_save(step)
                yield step
