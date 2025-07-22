#!/usr/bin/env python3

from typing import Iterable, Callable, Dict, Any, Union, Tuple
from dataclasses import dataclass

import numpy as np
import torch as th
import torch.nn as nn


def weights_to_splits(
        total: int, weights: Tuple[float, ...]) -> Tuple[int, ...]:
    """ Convert a set of weights into a set of lengths which sum to `total`. """
    ratios = np.divide(weights, np.sum(weights))
    lengths = np.round(np.multiply(total, ratios)).astype(np.int32)
    # Naive solution for adjusting `lengths`
    lengths[-1] += total - sum(lengths)
    return lengths


class TrainValidSplit(th.utils.data.Dataset):
    """
    Custom train-valid-... split with numpy backend.
    """

    @dataclass(frozen=True)
    class Config:
        splits: Tuple[float, ...] = (0.8, 0.2)
        names: Tuple[str, ...] = ('train', 'valid')
        seed: int = 0
        shuffle: str = 'numpy'  # or `torch` or `none`

    def __init__(self,
                 cfg: Config,
                 split: str,
                 dataset: th.utils.data.Dataset,
                 transform=None):
        # Save various properties...
        self.cfg = cfg
        self.dataset = dataset

        # Perform the split, optionally with shuffle.
        n: int = len(dataset)
        indices = np.arange(n)
        if cfg.shuffle == 'numpy':
            rng = np.random.default_rng(cfg.seed)
            rng.shuffle(indices)
        elif cfg.shuffle == 'torch':
            indices = th.randperm(
                n, generator=th.Generator().manual_seed(
                    cfg.seed), dtype=th.int32,
                device='cpu').detach().cpu().numpy()
        lengths = weights_to_splits(len(dataset), cfg.splits)
        offsets = np.cumsum(lengths)[:-1]
        self.indices = np.array_split(indices, offsets)[
            cfg.names.index(split)]
        self.transform = transform

    def __getitem__(self, index: int):
        offset_index = self.indices[index]
        out = self.dataset.__getitem__(offset_index)
        if self.transform is not None:
            out = self.transform(out)
        return out


def split_dataset(
        dataset: th.utils.data.Dataset,
        split: str,
        splits: Tuple[float, ...] = (0.8, 0.2),
        names: Tuple[str, ...] = ('train', 'valid'),
        seed: int = 0):
    """
    Alternative to `TrainValidSplit`, using
    torch.utils.data.random_split backend.
    """
    lengths = weights_to_splits(len(dataset), splits)
    index = names.index(split)
    return th.utils.data.random_split(dataset, lengths,
                                      th.Generator().manual_seed(seed))[index]


class WrapDict:
    def __init__(self, xfm: Callable,
                 key_in: Union[str, Tuple[str, ...]],
                 key_out: Union[str, Tuple[str, ...]]):
        self.xfm = xfm

        self.scalar_in = False
        if isinstance(key_in, str):
            key_in = (key_in,)
            self.scalar_in = True

        self.scalar_out = False
        if isinstance(key_out, str):
            key_out = (key_out,)
            self.scalar_out = True
        self.key_in = key_in
        self.key_out = key_out

    def __call__(self, x: Dict[str, Any]) -> Dict[str, Any]:
        y = dict(x)
        val_out = self.xfm(*list(x[k] for k in self.key_in))
        if self.scalar_out:
            val_out = [val_out]
        y.update({k: v for k, v in zip(self.key_out, val_out)})
        return y


class SelectKeys:
    """Select subset of dictionary-valued inputs.
    """

    def __init__(self, keys: Iterable[str]):
        self.keys = keys

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {k: inputs[k] for k in self.keys}
        return outputs


class NormalizeBounds:
    """
    Normalize a given entry of the dictionary to (-1,1).
    Also transforms output to torch Tensor.
    """

    def __init__(self, min_bound: np.ndarray, max_bound: np.ndarray, key: str):
        # Sanitize inputs.
        self.min_bound = np.asanyarray(min_bound, dtype=np.float32)
        self.max_bound = np.asanyarray(max_bound, dtype=np.float32)
        # Convert to center/quotient form.
        # the quotient is the reciprocal of the radius.
        self.center = 0.5 * (self.max_bound + self.min_bound)
        self.quotient = 2.0 * np.reciprocal(self.max_bound - self.min_bound)
        self.key = key

    def __call__(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if isinstance(inputs, dict):
            outputs = dict(inputs)
            outputs[self.key] = (
                #th.as_tensor(
                (inputs[self.key] - self.center) * self.quotient
                # dtype=th.float32
            )
            return outputs
        else:
            outputs = (
                #th.as_tensor(
                (inputs - self.center) * self.quotient
                # dtype=th.float32
            )
            return outputs


def unnormalize(x: th.Tensor,
                center: th.Tensor,
                radius: th.Tensor,
                in_place: bool) -> th.Tensor:
    """ Undo normalization. """
    if not in_place:
        x = x.clone()
    return x.mul_(radius).add_(center)


class UnNormalizeBounds(nn.Module):
    """ Undo NormalizeBounds. """

    def __init__(self, min_bound: np.ndarray, max_bound: np.ndarray,
                 in_place: bool = False):
        super().__init__()

        self.min_bound = np.asanyarray(min_bound, dtype=np.float32)
        self.max_bound = np.asanyarray(max_bound, dtype=np.float32)
        self.in_place = in_place
        self.register_buffer('center', th.as_tensor(
            0.5 * (self.max_bound + self.min_bound)))
        self.register_buffer('radius', th.as_tensor(
            0.5 * (self.max_bound - self.min_bound)))

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return unnormalize(x, self.center, self.radius, self.in_place)


def main():
    class DummyDataset(th.utils.data.Dataset):
        def __init__(self, n: int, transform=None):
            self.data = th.arange(n)
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index: int):
            out = self.data[index]
            if self.transform is not None:
                out = self.transform(out)
            return out
    dataset = DummyDataset(17, transform=None)
    train_dataset = TrainValidSplit(TrainValidSplit.Config(shuffle='torch'),
                                    'train', dataset)
    print('..')
    for data in train_dataset:
        print(data)

    print('..')

    alt_train_dataset = th.utils.data.random_split(
        dataset, weights_to_splits(len(dataset), [8, 2]),
        th.Generator().manual_seed(0))[0]
    for data in alt_train_dataset:
        print(data)

    alt_train_dataset_3 = split_dataset(dataset,
                                        'train',
                                        train_dataset.cfg.splits,
                                        train_dataset.cfg.names,
                                        train_dataset.cfg.seed)
    print('..')
    for data in alt_train_dataset_3:
        print(data)


if __name__ == '__main__':
    main()
