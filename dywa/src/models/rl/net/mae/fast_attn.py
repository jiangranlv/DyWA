#!/usr/bin/env python3

from typing import Optional, Union, Tuple

import math
import torch as th
import torch.nn as nn
from einops import rearrange

from flash_attn.flash_attention import FlashAttention

from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEModel,
    ViTMAESelfAttention,
    ViTMAEConfig
)


def transfer_weights(
        src: ViTMAESelfAttention,
        dst: 'ViTMAESelfFlashAttention'):
    # weight = shape (out, in)
    with th.no_grad():
        wq, wk, wv = th.tensor_split(dst.qkv.weight, 3)
        wq.copy_(src.query.weight)
        wk.copy_(src.key.weight)
        wv.copy_(src.value.weight)

        bq, bk, bv = th.tensor_split(dst.qkv.bias, 3)
        bq.copy_(src.query.bias)
        bk.copy_(src.key.bias)
        bv.copy_(src.value.bias)


def replace_self_attention(model: nn.Module, config=None):
    if config is None:
        config = model.config

    def get_all_parent_layers(net, type):
        layers = []

        for name, l in net.named_modules():
            if isinstance(l, type):
                tokens = name.strip().split('.')

                layer = net
                for t in tokens[:-1]:
                    if not t.isnumeric():
                        layer = getattr(layer, t)
                    else:
                        layer = layer[int(t)]

                layers.append([layer, tokens[-1]])

        return layers

    for parent_layer, last_token in get_all_parent_layers(
            model, ViTMAESelfAttention):
        source = getattr(parent_layer, last_token)
        weight = source.query.weight
        attn = ViTMAESelfFlashAttention(config,
                                        device=weight.device,
                                        dtype=weight.dtype)
        transfer_weights(source, attn)
        # [1] copy train/eval flag.
        attn.train(source.training)
        # [2] copy requires_grad flag.
        attn.requires_grad_(weight.requires_grad)
        # attn.load_state_dict(getattr(parent_layer, last_token).state_dict())
        setattr(parent_layer, last_token, attn)


class ViTMAESelfFlashAttention(nn.Module):
    def __init__(self, config: ViTMAEConfig,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}.")

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv = nn.Linear(config.hidden_size, 3 * self.all_head_size,
                             bias=config.qkv_bias, device=device, dtype=dtype)
        self.attention = FlashAttention()
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def split_heads(self, x: th.Tensor) -> th.Tensor:
        return rearrange(x, '... (h d) -> ... h d', h=self.num_attention_heads)

    def transpose_for_scores(self, x: th.Tensor) -> th.Tensor:
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_slow(
        self, hidden_states, head_mask: Optional[th.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]]:
        qkv = rearrange(self.qkv(hidden_states),
                        '... s (three h d) -> ... three h s d',
                        three=3, h=self.num_attention_heads)
        query_layer, key_layer, value_layer = th.unbind(qkv, dim=-4)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = th.matmul(
            query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = th.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            context_layer,
            attention_probs) if output_attentions else (
            context_layer,
        )

        return outputs

    def forward(
        self, hidden_states, head_mask: Optional[th.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[th.Tensor, th.Tensor], Tuple[th.Tensor]]:

        # slow path
        if output_attentions:
            return self.forward_slow(
                hidden_states, head_mask, output_attentions)

        with th.cuda.amp.autocast(True):
            # fast path
            qkv = rearrange(self.qkv(hidden_states),
                            '... (three h d) -> ... three h d',
                            three=3, h=self.num_attention_heads)

            # input: (B, S, 3, H, D)
            # output: (B, S, H, D)
            context_layer, attention_probs = self.attention(
                qkv, need_weights=output_attentions)
        context_layer = rearrange(context_layer, '... h d -> ... (h d)',
                                  h=self.num_attention_heads)

        outputs = (
            context_layer,
            attention_probs) if output_attentions else (
            context_layer,
        )

        return outputs


def compare():
    device = 'cuda:0'
    cfg = ViTMAEConfig()
    attn1 = ViTMAESelfAttention(cfg).to(device)
    attn2 = ViTMAESelfFlashAttention(cfg).to(device)
    # attn2.load_state_dict(attn1.state_dict())
    transfer_weights(attn1, attn2)

    B: int = 1
    S: int = 4
    x = th.randn((B, S, cfg.hidden_size),
                 dtype=th.float,
                 device=device)

    # <slow track>
    with th.cuda.amp.autocast(True):
        out1 = attn1(x, output_attentions=True)[0]
        out2 = attn2(x, output_attentions=True)[0]
        delta = (out1 - out2)
        print(delta.min())
        print(delta.max())

    # <fast track>
    with th.cuda.amp.autocast(True):
        out1 = attn1(x)[0]
        out2 = attn2(x)[0]
        delta = (out1 - out2)
        print(delta.min())
        print(delta.max())


def main():
    device = 'cuda:0'
    cfg = ViTMAEConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act='gelu',
        layernorm_eps=1e-6,
        image_size=64,
        patch_size=8,
        num_channels=1,
        decoder_num_attention_heads=4,
        decoder_hidden_size=256,
        decoder_num_hidden_layers=2,
        decoder_intermediate_size=128,
        mask_ratio=0.25)
    model = ViTMAEModel(cfg)
    replace_self_attention(model)
    print(model)


if __name__ == '__main__':
    main()
