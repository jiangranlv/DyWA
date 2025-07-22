#!/usr/bin/env python3

from typing import Optional
from pathlib import Path
import cv2
import numpy as np

import torch as th
import torch.nn.functional as F
from einops import rearrange
from transformers.models.vit.modeling_vit import ViTModel

from matplotlib import pyplot as plt

from util.path import ensure_directory
from util.torch_util import dcn


def rollout(attentions: th.Tensor,
            discard_ratio: float = 0.9,
            head_fusion: str = 'max'):
    result = th.eye(attentions[0].size(-1),
                    dtype=attentions[0].dtype,
                    device=attentions[0].device)
    with th.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=-2)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=-2)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=-2)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(
                attention_heads_fused.size(0), -1)
            _, indices = flat.topk(
                int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = th.eye(attention_heads_fused.size(-1),
                       dtype=attention_heads_fused.dtype,
                       device=attention_heads_fused.device)
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            result = th.matmul(a, result)
    return result

    # Look at the total attention between the class token,
    # and the image patches
    # mask = result[:, 0, 1:]
    # mask = result[:, 0, 1:]

    # In case of 224x224 image, this brings us from 196 to 14
    # width = int(mask.size(-1)**0.5)
    # width = mask.size(-1)
    # mask = dcn(mask)
    # mask = mask / np.max(mask, axis=-1, keepdims=True)
    # return mask


def draw_patch_attentions(a: th.Tensor):
    """
    x: (..., S, P, 3)
        point cloud, grouped in terms of patches.
    a: (L, ..., S):
        attention score for each patch.
    """
    # a = dcn(last_att)

    # Attention from last layer
    # There are alternatives to this, like rollout()
    # last_attn = a[-1]  # 64,4,17,17
    if True:
        # == last-layer path ==
        last_attn = a[-1]
        output_patch_attn = last_attn[..., 0, 1:]
        output_patch_attn = output_patch_attn.max(dim=-2).values
    else:
        # == rollout path ==
        if False:
            # only leave 1 head
            aa = [x[..., 0:1, :, :] for x in a]
            last_attn = rollout(aa, discard_ratio=0.75)
        else:
            last_attn = rollout(a, discard_ratio=0.75,
                                head_fusion='max')

            # Attention corresponding to output token @ index=0
            output_patch_attn = last_attn[..., 0, 1:]  # 64, 4, 16

    # For the visualization,
    # We take the "maximum" attention for each patch among all heads.
    # Alternatives are (1) selecting per each head
    # or (2) taking the mean weight.
    # 64, 16
    # output_patch_attn = dcn(output_patch_attn.max(dim=-2).values)
    output_patch_attn = dcn(output_patch_attn)
    output_patch_attn /= output_patch_attn.max(
        axis=-1, keepdims=True)
    output_patch_attn[output_patch_attn < 0.5] = 0.0
    # output_patch_attn *= 2.0
    patch_colors = cv2.applyColorMap(
        (output_patch_attn * 255).astype(np.uint8),
        colormap=cv2.COLORMAP_VIRIDIS)[..., ::-1] / 255.0
    return patch_colors


def to_hwc(x: th.Tensor):
    return rearrange(x, '... c h w -> ... h w c')


def get_attention_maps(
        out_dir: str,
        pixel_values: th.Tensor,
        attentions: th.Tensor,
        num_heads: int,
        patch_size: int,
        threshold: float = 0.6):
    """
    # TODO: figure out what the `threshold parameter here is for.
    # Is this just pruning for "sufficiently high" attention values?
    """
    out_dir = ensure_directory(out_dir)

    # Figure out the spatial dimensions of the
    # attention map...
    print('pix', pixel_values.shape)
    h_featmap = pixel_values.shape[-2] // patch_size
    w_featmap = pixel_values.shape[-1] // patch_size
    print('featmap dims', h_featmap, w_featmap)
    print('attns', attentions.shape)

    # FIXME: unused code block ???
    # We keep only a certain percentage of the mass.
    if False:
        val, idx = th.sort(attentions)
        val /= th.sum(val, dim=1, keepdim=True)
        cumval = th.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = th.argsort(idx)
        for head in range(num_heads):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(num_heads,
                                  h_featmap, w_featmap).float()

        # Interpolate & upscale to match image dimensions.
        th_attn = th.nn.functional.interpolate(
            th_attn.unsqueeze(0),
            size=pixel_values.shape[-2:],
            # scale_factor=PATCH_SIZE,
            mode="nearest")[0].cpu().numpy()
        th_attn = dcn(th_attn)

    attentions = attentions.reshape(num_heads,
                                    h_featmap, w_featmap)
    attentions = (th.nn.functional.interpolate(
        attentions.unsqueeze(0),
        size=pixel_values.shape[-2:],
        # scale_factor=PATCH_SIZE,
        mode="nearest")[0])
    attentions = dcn(attentions)

    # Save attention heatmaps and return list of filenames.
    attention_maps = []
    for j in range(num_heads):
        fname = str(out_dir / F'head-{j:02d}.png')

        # Save the attention map.
        # FIXME:
        # alternatively, directly return the attn map ...
        plt.imsave(fname=fname,
                   arr=attentions[j],
                   format='png')

        # Append filename.
        attention_maps.append(fname)

    return attention_maps


def visualize_attention(
        out_dir: str,
        inputs,
        outputs,
        patch_size: int):
    pixel_values = inputs  # .pixel_values
    # forward pass
    # outputs = model(
    #     pixel_values,
    #     output_attentions=True,
    #     interpolate_pos_encoding=True # << what the fuck is this?
    #     )

    # get attentions of last layer
    attentions = outputs.attentions[-1]
    num_heads = attentions.shape[1]  # number of heads

    # We keep only the output patch attention.
    print('attentions', attentions.shape)
    attentions = attentions[0, :, 0, 1:].reshape(num_heads, -1)

    attention_maps = get_attention_maps(
        out_dir, pixel_values, attentions, num_heads,
        patch_size)

    return attention_maps


class SaveAttentionMaps:
    def __init__(self, out_dir: Optional[str] = None):
        self.count: int = 0
        self.out_dir = ensure_directory(out_dir)
        self.patch_size: int = -1

    def hook(self, vit: ViTModel):
        vit.register_forward_hook(self)
        self.patch_size = vit.config.patch_size

    def hook_img(self, model):
        model.register_forward_hook(self._save_img)

    def _save_img(self, module, inputs, outputs):
        prefix: str = F'{self.count:04d}'
        out_dir = ensure_directory(self.out_dir / prefix)
        # Export input image.
        imgs = inputs[0]  # first input to model
        img = imgs[0]  # first element of batch
        img = to_hwc(img)  # Convert convention: CHW -> HWC

        if img.shape[-1] == 1:
            # depth or binary mask
            plt.imsave(fname=F'{out_dir}/img.png',
                       arr=dcn(img.squeeze(-1)),
                       format='png')
        elif img.shape[-1] == 3:
            # RGB/BGR, three-channel image
            plt.imsave(fname=F'{out_dir}/img.png',
                       arr=dcn(img),
                       format='png')
        else:
            # In other cases, save each channel separately.
            for c in range(img.shape[-1]):
                plt.imsave(fname=F'{out_dir}/img-{c}.png',
                           arr=dcn(img[..., c]),
                           format='png')
        self.count += 1

    def __call__(self, module, inputs, outputs):
        prefix: str = F'{self.count:04d}'
        out_dir = ensure_directory(self.out_dir / prefix)

        # Export attention masks.
        maps = visualize_attention(out_dir,
                                   inputs[0],
                                   outputs,
                                   self.patch_size)

        if False:
            # Export input image.
            imgs = inputs[0]  # first input to model
            img = imgs[0]  # first element of batch
            img = to_hwc(img)  # Convert convention: CHW -> HWC
            print('img', img.shape)

            if img.shape[-1] == 1:
                # depth or binary mask
                plt.imsave(fname=F'{out_dir}/img.png',
                           arr=dcn(img.squeeze(-1)),
                           format='png')
            elif img.shape[-1] == 3:
                # RGB/BGR, three-channel image
                plt.imsave(fname=F'{out_dir}/img.png',
                           arr=dcn(img),
                           format='png')
            else:
                # In other cases, save each channel separately.
                for c in range(img.shape[-1]):
                    plt.imsave(fname=F'{out_dir}/img-{c}.png',
                               arr=dcn(img[..., c]),
                               format='png')


def plot_attn_path(path: str, num_cols: Optional[int] = None):
    path = Path(path)
    imgs = sorted(path.glob('img.png'))
    heads = sorted(path.glob('head-*.png'))

    if num_cols is None:
        num_cols = len(imgs)

    mosaic = []
    for group in (imgs, heads):
        num_rows = (len(group) + num_cols - 1) // num_cols
        rows = [['-' for _ in range(num_cols)] for _ in range(num_rows)]

        for i in range(len(group)):
            r = i // num_cols
            c = i % num_cols
            rows[r][c] = str(group[i])
        mosaic.extend(rows)
    print(mosaic)
    fig, ax = plt.subplot_mosaic(mosaic)

    for group in (imgs, heads):
        for filename in group:
            img = plt.imread(str(filename))
            ax[str(filename)].imshow(img)
    plt.show()


def main():
    plot_attn_path('/tmp/docker/attn2/0000/')


if __name__ == '__main__':
    main()
