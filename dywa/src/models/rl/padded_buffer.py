#!/usr/bin/env python3

"""
data format:


# reset()
obs

>action
obs, rew, done, info

>action
obs, rew, done, info // done=True

>action (ignored)
obs, rew, done, info // ignore rew; obs= init frame

STORAGE (naive?)
need? = (end0,

<obsn, actn, rewd, done, logp> == o,a,r,d,l
[_,_,_,d,l] reset() frame
[o,_,_,_,_] initial frame
[o,a,r,d,l] normal  frame
[   ...   ] ...interim...
[o,a,r,d,l] final   frame
[o,_<_,_,_] initial frame

"""

from typing import Tuple, Optional, Dict, Mapping, Iterable, List
from collections import defaultdict
import torch as th
from models.common import merge_shapes, map_struct

T = th.Tensor


class KeyMap(Mapping, Iterable):
    """
    Recursively convert attr/items to string expressions.
    """

    def __init__(self, key: str):
        self.key = key
        # self.children = set()
        self.children = {}

    def __iter__(self):
        return iter(self.children)

    def __len__(self):
        return len(self.children)

    def __getitem__(self, k: str):
        out = KeyMap(F'{self.key}[{k}]')
        self.children[k] = out
        return out

    def __getattr__(self, k: str):
        out = KeyMap(F'{self.key}.{k}')
        self.children[k] = out
        return out

    def __str__(self) -> str:
        return self.key

    def __repr__(self) -> str:
        return self.key


class DictBuffer:
    """
    SoA-style cyclic buffer.
    """

    def __init__(self, size: int, device: th.device):
        self.size: int = size
        self.full: bool = False
        self.index: int = 0
        self.device = device
        self.bufs = {}
        self.keys = {}

    def __len__(self):
        return (self.size if self.full else self.index)

    def add_field(self,
                  key: str,
                  shape: Tuple[int, ...],
                  dtype=th.float,
                  is_root: bool = True):
        """
        key: name of the field.
        shape: unbatched shape.
        """
        if is_root:
            # self.keys.append(KeyMap(key))
            self.keys[key] = KeyMap(key)
        b_shape = merge_shapes(self.size, shape)
        self.bufs[key] = th.empty(
            b_shape,
            dtype=dtype,
            device=self.device,
            requires_grad=False)

    def add_field_from_tensor(self, key: str, value: th.Tensor):
        # TODO: rename from `_tensor` -> `_sample`?
        key_map = KeyMap(key)
        map_struct(src=value,
                   op=lambda src, dst: self.add_field(
                       str(dst), src.shape, src.dtype,
                       is_root=False),
                   dst=key_map,
                   base_cls=th.Tensor)
        # self.keys.append(key_map)
        self.keys[key] = key_map

    def extend(self,
               bufs: Dict[str, th.Tensor],
               n: Optional[int] = None):
        # raise ValueError('extend() not currently fully supported')

        if n is None:
            n = next(iter(bufs.values())).shape[0]
        if n <= 0:
            return

        with th.no_grad():
            p2 = self.index + n
            if p2 >= self.size:
                # Branch 1 : overflow
                if n >= self.size:
                    # Just copy everything directly.
                    # NOTE: this specialization _might_
                    # break some assumptions if the user
                    # relies on the index-offset behavior
                    # of the buffer in case of overflow.
                    for k in bufs.keys():
                        self.bufs[k][...] = bufs[k][-self.size:]
                    self.index = 0
                    self.full = True
                else:
                    # In this case we need to handle segment logic.
                    m1 = self.size - self.index  # length of first segment
                    m2 = n - m1  # length of second segment
                    for k in bufs.keys():
                        self.bufs[k][self.index:] = bufs[k][:m1]
                        self.bufs[k][:m2] = bufs[k][m1:]
                    self.index = p2 % self.size
                    self.full = True
            else:
                # Branch 2 : typical case
                for k in bufs.keys():
                    self.bufs[k][self.index:p2] = bufs[k]
                self.index = p2

    def append(self, bufs: Dict[str, th.Tensor]):
        with th.no_grad():
            def _append(key: KeyMap, buf: th.Tensor):
                # Skip inputs with `None` values.
                if buf is None:
                    return

                # leaf
                if len(key) <= 0:
                    # Skip inputs with keys not present in the buffer.
                    if not str(key) in self.bufs:
                        raise KeyError(F'not has {key}')
                    self.bufs[str(key)][self.index] = buf
                    return

                # not leaf
                for k, v in key.children.items():
                    _append(v, buf[k])

                # if isinstance(buf, Mapping):
                #     for k, v in buf.items():
                #         subkey = F'{key}.{k}' if len(key) > 0 else k
                #         _append(subkey, v)
                #     return

                # if not hasattr(self.bufs, key):
                #     print(F'not has {key}')
                #     return

                # set value.
                # print(F'set = {key}')
                # self.bufs[key][self.index] = buf

            for k, v in bufs.items():
                _append(self.keys[k], v)

            # _append('', bufs)

            # for k in bufs.keys():
            #     if bufs[k] is None:
            #         continue
            #     if isinstance(bufs[k], Mapping):
            #         for k2 in bufs[k].keys():
            #             self.bufs[F'{k}.{k2}'][self.index] = bufs[k][k2]
            #     else:
            #         if k not in self.bufs:
            #             continue
            #         self.bufs[k][self.index] = bufs[k]

        # Increment data pointer.
        pos = self.index + 1
        if pos >= self.size:
            self.full = True
            # cycle back to the first index.
            pos = pos % self.size
        self.index = pos

    def clear(self, keep_last: bool = False):
        if keep_last:
            raise ValueError(
                F'Unsupported option keep_last={keep_last}')
        self.full = False
        self.index = 0

    def sample_transitions(
            self,
            batch_size: int,
            key0: Tuple[str, ...] = ('obsn', 'done'),
            key1: Tuple[str, ...] = ('obsn', 'actn', 'rewd', 'done'),

            naive: bool = False):

        if naive:
            buf = self.get()
            n = next(iter(buf.values())).shape[0]
            i0 = th.randint(0, n - 1, size=(batch_size,),
                            device=self.device)
            i1 = i0 + 1

            prv = {}
            for k in key0:
                prv[k] = buf[k][i0]
            nxt = {}
            for k in key1:
                nxt[k] = buf[k][i1]
            return (prv, nxt)

        raise ValueError(
            'Non-naive (slightly more efficient) sampling ' +
            'not implemented yet.'
        )
        if self.full:
            i0 = th.randint(0, self.size - 1,
                            size=batch_size,
                            device=self.device)
            i1 = i0 + 1
        else:
            pass

    def get(self):
        # NOTE: using `len(self)`
        # to catch the case (self.index==0 ^ full)
        n: int = len(self)

        def _get(key: KeyMap):
            if len(key.children) <= 0:
                # Account for the fact that this is a
                # cyclic buffer.
                if self.full and (self.index > 0):
                    return th.roll(self.bufs[str(key)], -self.index, 0)
                else:
                    return self.bufs[str(key)][:n]
            else:
                return {k: _get(v) for (k, v) in key.children.items()}

        out = {}
        for k, v in self.keys.items():
            out[k] = _get(v)
        return out


class PaddedBuffer:
    KEYS: Tuple[str, ...] = ('obsn', 'actn', 'rewd', 'done',
                             'logp', 'mu', 'ls', 'repr')

    def __init__(self, size: int,
                 device: th.device):
        self.size = size
        self.pos = 0
        self.full = False
        self.device = device
        self.bufs = {k: None for k in self.KEYS}

    def __len__(self):
        return (self.size if self.full else self.pos)

    def initialize(self, obs: T, act_dim: int,
                   discrete: bool,
                   repr_dim: int
                   ):
        """
        obs: shape (N, D...) where N = num env, D = obs shape
        """
        self.pos = 0
        self.full = False

        # first, insert the pre-reset frame
        N: int = obs.shape[0]
        A: int = act_dim

        act_shape = (N,) if discrete else (N, A)
        act_dtype = th.int32 if discrete else th.float

        repr_shape = (N, repr_dim)

        # [1] pre-init frame.
        pre_init_frame = {
            'obsn': th.zeros_like(obs),
            'repr': th.zeros(repr_shape, dtype=th.float, device=self.device),
            'actn': th.zeros(act_shape, dtype=act_dtype, device=self.device),
            'rewd': th.zeros((N,), dtype=th.float, device=self.device),
            'done': th.ones((N,), dtype=bool, device=self.device),
            'logp': th.zeros((N,), dtype=th.float, device=self.device),
            # TODO: `distribution` or `dist-params` instead,
            # to store distribution-relevant
            # parameters (e.g. logits) rather than specifically `mu,ls`...
            'mu': th.zeros((N, A), dtype=th.float, device=self.device),
            'ls': th.zeros((N, A), dtype=th.float, device=self.device),
        }
        # [1.a] allocate.
        self.allocate(pre_init_frame)
        self.append(pre_init_frame)

        # [2] init frame.
        init_frame = {
            'obsn': obs,
            'repr': th.zeros(repr_shape, dtype=th.float, device=self.device),
            'actn': th.zeros(act_shape, dtype=act_dtype, device=self.device),
            'rewd': th.zeros((N,), dtype=th.float, device=self.device),
            # TODO: `done` not guaranteed to be false.
            'done': th.zeros((N,), dtype=bool, device=self.device),
            'logp': th.zeros((N,), dtype=th.float, device=self.device),
            'mu': th.zeros((N, A), dtype=th.float, device=self.device),
            'ls': th.zeros((N, A), dtype=th.float, device=self.device),
        }
        self.append(init_frame)

    def allocate(self, bufs: Dict[str, T]):
        """
        Reset buffer from exemplar.
        """
        for k in self.KEYS:
            src = bufs[k]
            self.bufs[k] = th.empty((self.size,) + src.shape,
                                    dtype=src.dtype,
                                    device=self.device,
                                    requires_grad=False)

    def append(self, bufs: Dict[str, T]):
        # TODO: if `strict`,
        # assert (set(bufs.keys()) == set(self.KEYS))
        # self.KEYS:
        for k in bufs.keys():
            v = bufs[k]
            if v is not None:
                v = v.detach()
                self.bufs[k][self.pos] = v
        pos = (self.pos + 1)
        if pos >= self.size:
            self.full = True
            pos = pos % self.size
        self.pos = pos

    def clear(self, keep_last: bool = True):
        """ reset the flags but preserve the data. """
        self.full = False
        if keep_last:
            assert (self.size >= 2)
            for k in self.KEYS:
                # FIXME: ONLY works if running clear() right after full
                # otherwise this is invalid !
                self.bufs[k][:2] = self.bufs[k][-2:]
            self.pos = 2
        else:
            self.pos = 0

    def get(self) -> Dict[str, T]:
        # return dict(self.bufs)
        n = len(self)
        return {k: v[:n] for (k, v) in self.bufs.items()}


def test_dict_buffer():
    device = th.device('cpu')
    d = DictBuffer(size=8, device=device)
    d.add_field('obs', (1,))
    d.add_field('act', (1,))
    for obs in th.arange(16)[..., None]:
        d.append({'obs': obs})
    print(d.get())
    t0, t1 = d.sample_transitions(4, key0=['obs'], key1=['obs'],
                                  naive=True)
    print(t0['obs'])
    print(t1['obs'])


def test_dict_buffer_with_dict():
    device = th.device('cpu')
    d = DictBuffer(size=9, device=device)
    d.add_field_from_tensor('obs', {
        'img': th.zeros((8, 2, 4, 4)),
        'state': th.zeros((8, 7))
    })

    for i in range(11):
        d.append({'obs': {
            'img': th.zeros((8, 2, 4, 4)),
            'state': th.zeros((8, 7))
        }})
    D = d.get()
    # print(D['obs'].shape)
    print(list(D.keys()))
    print(list(D['obs'].keys()))
    print(D['obs']['img'].shape)
    print(D['obs']['state'].shape)

    # for obs in th.arange(16)[..., None]:
    #     d.append({'obs': obs})
    # print(d.get())
    # t0, t1 = d.sample_transitions(4, key0=['obs'], key1=['obs'],
    #                               naive=True)
    # print(t0['obs'])
    # print(t1['obs'])


def validate_dict_buffer():
    buf_len: int = 16 + 2
    device = th.device('cpu')
    buf = DictBuffer(buf_len, device=device)

    custom_buf = {}

    def _print_keys(x, prefix=''):
        out = []
        for k, v in x.items():
            key = F'{prefix}[{k}]'
            if isinstance(v, Mapping):
                _print_keys(v, key)
            else:
                print(key)

    def _append(d, k, x):
        if isinstance(x, th.Tensor):
            if k not in d:
                d[k] = []
            d[k].append(x)
            d[k] = d[k][-buf_len:]
            return

        if isinstance(x, Mapping):
            if k not in d:
                d[k] = {}
            for k2, v2 in x.items():
                _append(d[k], F'{k2}', v2)
            return

    for i in range(14):
        # create obs
        obs = map_struct(
            {'a': {'b': (1, 3), 'c': (4, 4), 'f': {'g': 1}},
             'd': (5,), 'e': (8, 9)},
            lambda src, _: th.randn(merge_shapes(src)),
            base_cls=(List, Tuple, int))
        # create field
        if i == 0:
            buf.add_field_from_tensor('obs', obs)
        # append to both bufs
        buf.append({'obs': obs})
        _append(custom_buf, 'obs', obs)

    custom_buf = map_struct(custom_buf,
                            lambda src, _: th.stack(src, dim=0),
                            base_cls=(Tuple, List))
    actual_buf = buf.get()

    print('=keys(custom)=')
    _print_keys(custom_buf)
    print('=keys(actual)=')
    _print_keys(actual_buf)

    delta = map_struct(custom_buf,
                       op=lambda src, dst: th.mean(th.abs(src - dst)),
                       dst=actual_buf,
                       base_cls=th.Tensor)
    print('tensor value differences')
    print(delta)


def main():
    # test_dict_buffer_with_dict()
    validate_dict_buffer()
    return

    device = th.device('cpu')
    act_dim: int = 8
    num_env: int = 4
    obs_dim: int = 4
    buf_len: int = 16 + 2
    buf = PaddedBuffer(buf_len, device=device)
    obs = th.randn((num_env, obs_dim),
                   dtype=th.float32,
                   device=device)
    print('obs', obs)
    buf.initialize(obs, act_dim,
                   discrete=False)
    print(len(buf))

    d = (buf.get())
    for k, v in d.items():
        if k == 'done':
            print(v[0])
            print(v[len(buf) - 1])
    buf.clear()
    print(len(buf))

    d = (buf.get())
    for k, v in d.items():
        if k == 'done':
            print(v[0])


if __name__ == '__main__':
    main()
