#!/usr/bin/env python3

from typing import Tuple, Union, Optional
import torch as th
import torch.nn as nn

from util.torch_util import masked_var_mean

import nvtx


def update_from_moments_legacy(
        mean, var, count,
        batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def update_from_moments(
        m0: th.Tensor, v0: th.Tensor, w0: th.Tensor,
        m1: th.Tensor, v1: th.Tensor, w1: th.Tensor,
        in_place: bool = True) -> Tuple[th.Tensor, th.Tensor, float]:
    # v1 = 0
    # m1 != 0

    # NOTE:
    # w0 cannot be updated in-place
    # since it's a scalar.
    if not in_place:
        m0 = m0.copy()
        v0 = v0.copy()
        w0 = w0.copy()

    d = m1 - m0
    w = w0 + w1

    w0_ = w0 / w
    w1_ = w1 / w  # 1 / (w0+1)

    m0 += d * w1_
    v0[...] = v0 * w0_ + v1 * w1_ + th.square(d) * (w0_ * w1_)
    #         v0 * w0_ + d^2 * (w0_ * w1_)
    #         v0 * w0_ + d^2 * w0/(w0+1)^2
    w0[...] = w

    # NOTE: without the in-place __setitem__,
    # the buffer variable may get overwritten,
    # and then it could result in inadvertent
    # gradient flow!
    return (m0, v0, w0)


class ConstantMeanStd(nn.Module):
    """
    Drop-in replacement for RunningMeanStd,
    without adaptive stats.
    """

    def __init__(self,
                 device: th.device,
                 shape: Tuple[int, ...] = (),
                 epsilon: float = 1e-4,
                 mean: Optional[th.Tensor] = None,
                 var: Optional[th.Tensor] = None
                 ):
        super().__init__()
        if mean is None:
            mean = th.zeros(shape, dtype=th.float, device=device,
                            requires_grad=False)
        else:
            mean = th.as_tensor(mean, dtype=th.float, device=device)
        if var is None:
            var = th.ones(shape, dtype=th.float, device=device,
                          requires_grad=False)
        else:
            var = th.as_tensor(var, dtype=th.float, device=device)

        self.register_buffer('mean', mean)
        self.register_buffer('var', var)
        # print(self.mean)
        # print(self.var)

    def copy(self) -> "ConstantMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = ConstantMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        raise ValueError('combine() not supported for `ConstantMeanStd`')

    def update(self, arr: th.Tensor,
               mask: Optional[th.Tensor] = None) -> None:
        pass

    def update_from_moments(self,
                            batch_mean: th.Tensor,
                            batch_var: th.Tensor,
                            batch_count: Union[int, float]):
        pass


class RunningMeanStd(nn.Module):
    def __init__(self,
                 device: th.device,
                 shape: Tuple[int, ...] = (),
                 epsilon: float = 1e-4,
                 legacy: bool = True):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__()
        self.legacy = legacy
        self.shape = shape
        self.register_buffer('mean',
                             th.zeros(shape, dtype=th.float, device=device,
                                      requires_grad=False))
        self.register_buffer('var',
                             th.ones(shape, dtype=th.float, device=device,
                                     requires_grad=False))
        self.register_buffer('count',
                             th.full((), epsilon, dtype=th.float,
                                     device=device, requires_grad=False))

    def extra_repr(self):
        return super().extra_repr() + F'shape={self.shape}'

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = self.count.copy()
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    @nvtx.annotate("RMS.update()")
    def update(self, arr: th.Tensor,
               mask: Optional[th.Tensor] = None) -> None:
        if arr.shape[0] <= 1:
            return

        # mask = mask.expand_as(arr)
        arr  = arr.reshape(-1, *self.mean.shape)

        skip: bool = False
        if mask is not None:
            mask = mask.reshape(-1)

            # FIXME: masked_var_mean does not support `dim` arg.
            # assert (th.numel(mask) == arr.shape[0])

            # NOTE: either branch on batch_count (requires
            # device synchronization) or filter out NaNs
            # Cannot update from just 1 sample
            # if batch_count <= 1:
            #     return
            with nvtx.annotate("A"):
                batch_count = mask.sum()
            with nvtx.annotate("B"):
                batch_var, batch_mean = masked_var_mean(arr, mask, dim=0)

            # Overwrite to non-nans, since it will be skipped.
            with nvtx.annotate("C"):
                skip = (batch_count <= 1).item()
            with nvtx.annotate("D"):
                # batch_var = th.where(skip, 1.0, batch_var)
                # batch_mean = th.where(skip, 0.0, batch_mean)
                batch_var = th.nan_to_num(batch_var)
                batch_mean = th.nan_to_num(batch_mean)
                # batch_var[skip] = 1
                # batch_mean[skip] = 0
        else:
            batch_var, batch_mean = th.var_mean(arr, dim=0)
            batch_count = arr.shape[0]

        if not skip:
            with nvtx.annotate("E"):
                self.update_from_moments(batch_mean, batch_var, batch_count)
        return (batch_var, batch_mean, batch_count)

    def update_from_moments(self,
                            batch_mean: th.Tensor,
                            batch_var: th.Tensor,
                            batch_count: Union[int, float]) -> None:
        if self.legacy:
            new_stats = update_from_moments_legacy(
                self.mean, self.var, self.count,
                batch_mean, batch_var, batch_count
            )
            self.mean[...], self.var[...], self.count[...] = new_stats

        else:
            update_from_moments(
                self.mean, self.var, self.count,
                batch_mean, batch_var, batch_count,
                in_place=True)


class RollingMeanStd(nn.Module):
    def __init__(self,
                 device: th.device,
                 shape: Tuple[int, ...] = (),
                 epsilon: float = 1e-4,
                 alpha: float = 0.001):
        """
        Calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        """
        super().__init__()
        self.alpha = alpha
        self.register_buffer('mean',
                             th.zeros(shape, dtype=th.float, device=device,
                                      requires_grad=False))
        self.register_buffer('var',
                             th.ones(shape, dtype=th.float, device=device,
                                     requires_grad=False))
        self.register_buffer('count',
                             th.full((), epsilon, dtype=th.float,
                                     device=device, requires_grad=False))

    def copy(self) -> "RunningMeanStd":
        """
        :return: Return a copy of the current object.
        """
        new_object = RunningMeanStd(shape=self.mean.shape)
        new_object.mean = self.mean.copy()
        new_object.var = self.var.copy()
        new_object.count = self.count.copy()
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        """
        Combine stats from another ``RunningMeanStd`` object.

        :param other: The other object to combine with.
        """
        self.update_from_moments(other.mean, other.var, other.count)

    @nvtx.annotate("RMS.update()")
    def update(self, arr: th.Tensor,
               mask: Optional[th.Tensor] = None) -> None:
        if arr.shape[0] <= 1:
            return
        skip: bool = False
        if mask is not None:
            # FIXME: masked_var_mean does not support `dim` arg.
            # assert (th.numel(mask) == arr.shape[0])

            # NOTE: either branch on batch_count (requires
            # device synchronization) or filter out NaNs
            # Cannot update from just 1 sample
            # if batch_count <= 1:
            #     return
            with nvtx.annotate("A"):
                batch_count = mask.sum()
            with nvtx.annotate("B"):
                batch_var, batch_mean = masked_var_mean(arr, mask, dim=0)

            # Overwrite to non-nans, since it will be skipped.
            with nvtx.annotate("C"):
                skip = (batch_count <= 1).item()
            with nvtx.annotate("D"):
                # batch_var = th.where(skip, 1.0, batch_var)
                # batch_mean = th.where(skip, 0.0, batch_mean)
                batch_var = th.nan_to_num(batch_var)
                batch_mean = th.nan_to_num(batch_mean)
                # batch_var[skip] = 1
                # batch_mean[skip] = 0
        else:
            batch_var, batch_mean = th.var_mean(arr, dim=0)
            batch_count = arr.shape[0]

        if not skip:
            with nvtx.annotate("E"):
                self.update_from_moments(batch_mean, batch_var, batch_count)
        return (batch_var, batch_mean, batch_count)

    def update_from_moments(self,
                            batch_mean: th.Tensor,
                            batch_var: th.Tensor,
                            batch_count: Union[int, float]) -> None:
        th.lerp(self.mean, batch_mean, self.alpha,
                out=self.mean)
        th.lerp(self.var, batch_var, self.alpha,
                out=self.var)


def test_zero_element():
    N: int = 4
    D: int = 8
    rms = RunningMeanStd('cuda:0', shape=(D,))

    print('rand-update')
    eps: float = 0.5
    x = th.randn((N, D), dtype=th.float, device='cuda:0')
    m = (th.rand((N, 1), dtype=th.float, device='cuda:0') < eps)
    rms.update(x, mask=m)
    print(rms.mean)
    print(rms.var)
    print(rms.count)

    print('zero-update')
    eps: float = 0.0
    x = th.randn((N, D), dtype=th.float, device='cuda:0')
    m = (th.rand((N, 1), dtype=th.float, device='cuda:0') < eps)
    rms.update(x, mask=m)
    print(rms.mean)
    print(rms.var)
    print(rms.count)

    print('one-update')
    x = th.randn((N, D), dtype=th.float, device='cuda:0')
    m = (th.rand((N, 1), dtype=th.float, device='cuda:0'))
    m = (m < m.min())
    rms.update(x, mask=m)
    print(rms.mean)
    print(rms.var)
    print(rms.count)


def main():
    N: int = 4
    D: int = 8

    rms = RunningMeanStd('cuda:0', shape=(D,))
    # rms.update()
    print(rms.mean.requires_grad)  # false

    eps: float = 0.5

    x = th.randn((N, D), dtype=th.float, device='cuda:0')
    m = (th.rand((N, 1), dtype=th.float, device='cuda:0') < eps)
    rms.update(x, mask=m)
    print(rms.mean)
    print(rms.var)

    print(m.sum())
    print('m', m)
    print(x.shape)
    print(m.shape)
    print(x[m.squeeze(dim=-1)].mean(dim=0))
    print(x[m.squeeze(dim=-1)].var(dim=0))
    print('x', x)
    x.masked_fill_(m, 0)
    print('x', x)


if __name__ == '__main__':
    test_zero_element()
