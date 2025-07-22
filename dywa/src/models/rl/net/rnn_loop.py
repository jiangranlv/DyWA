#!/usr/bin/env python3

from typing import Optional, Dict, Callable, Tuple
from functools import partial

import torch as th
import torch.nn as nn

from models.common import map_struct

import nvtx
from icecream import ic

map_tensor = partial(map_struct,
                     base_cls=th.Tensor)


def loop_rnn(
        state_fn: Callable[[th.Tensor, th.Tensor, th.Tensor], th.Tensor],

        hidden: Dict[str, th.Tensor],
        action: th.Tensor,
        feature: Dict[str, th.Tensor],

        done: Optional[th.Tensor],

        bptt_len: int,
        burn_in: int = 0,
        stride: int = 1,
        alpha: float = 0.5
):
    reset_state: bool = (done is not None)

    L = len(done)
    S: int = L - bptt_len

    for di in range(1, bptt_len + 1):
        # NOTE: burn-in threshold is checked with
        # > (not >=) since di starts from `1`.
        prev_index = slice(di - 1, S + di - 1, stride)
        curr_index = slice(di, S + di, stride)

        with th.set_grad_enabled(di > burn_in):
            with nvtx.annotate("get_prev"):
                prev_done = done[prev_index]
                have_prev = (~prev_done[..., None])
                # FIXME: this code assumes that
                # "zero_state" is the hidden state
                # when we start from scratch.
                prev_hidden = map_tensor(
                    hidden, (
                        lambda src, dst:
                        (src[prev_index] * have_prev
                            if reset_state
                            else src[prev_index])
                    ))
                # FIXME: this is `prev_action` but uses
                # `curr_index` since it assumes that the
                # data is arranged with an offset of -1.
                prev_action = action[curr_index]

            # Slice "current" feature vector (=observation).
            with nvtx.annotate("slice_curr_feat"):
                curr_feature = map_tensor(
                    feature, lambda src, dst: src[curr_index])

            # Apply one-step gating + aggregation mechanism.
            with nvtx.annotate("state_fn"):
                _, curr_hidden = state_fn(
                    prev_hidden,
                    prev_action,
                    curr_feature)

            # Update `hidden` from curr_hidden.
            with nvtx.annotate("slice_curr_hidden"):
                hidden = map_tensor(hidden,
                                    lambda src, dst: src.clone())

                def _update_hidden(src, dst):
                    # dst[curr_index] = src
                    # NOT sure if this is a good idea...
                    dst[curr_index] = th.lerp(dst[curr_index], src, alpha)
                    return dst
                map_tensor(curr_hidden,
                           _update_hidden,
                           hidden)
    return hidden


def main():
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # T: int = 11
    # cfg.train.rollout_size + cfg.bptt_seq_len + 1,
    B: int = 1
    H: int = 1
    A: int = 1
    O: int = 1

    seq_len = 4
    burn_in = 0
    stride = 1
    rollout_size = 8
    T: int = rollout_size + burn_in + seq_len + 1

    def _state_fn(h0, a0, o1):
        # print(h0)
        # h1 = [h0[i] + a0[i] + o1[i] for i in range(len(h0))]
        # return None, h1
        return None, 0.1 * h0 + a0 + o1

    hidden = th.zeros(
        (T, B, H), dtype=th.float) + 0.001 * 0 * th.arange(T)[..., None, None] + 1
    hidden.requires_grad_(True)
    action = th.zeros((T, B, A),
                      dtype=th.float) + 0.0 * th.arange(T)[..., None, None]
    feature = th.zeros((T, B, O),
                       dtype=th.float) + 0.0 * th.arange(T)[..., None, None]
    done = th.zeros((T, B), dtype=bool)
    # done[4] = True
    output = loop_rnn(_state_fn,
                      hidden, action, feature,
                      done,
                      seq_len,
                      burn_in,
                      stride)

    class Model(nn.Module):
        def forward(self, hidden, action, feature,
                    done,
                    seq_len,
                    burn_in,
                    stride):
            hidden = th.stack(hidden, dim=0)
            return loop_rnn(_state_fn,
                            hidden, action, feature,
                            done,
                            seq_len,
                            burn_in,
                            stride)
    # print(output, output.shape)
    print(th.autograd.grad(output, hidden,
                           grad_outputs=th.ones_like(output)
                           # grad_outputs=1 + th.arange(
                           #     output.shape[0])[..., None, None]
                           # )
                           )
          )

    # from torchview import draw_graph

    # model_graph = draw_graph(Model(),
    #                          input_data=(hidden.unbind(dim=0), action, feature,
    #                                      done,
    #                                      seq_len,
    #                                      burn_in,
    #                                      stride),
    #                          # input_size=x.shape,
    #                          device='cpu')
    # model_graph.visual_graph.render('/tmp/docker/rnn-loop.svg')


if __name__ == '__main__':
    main()
