#!/usr/bin/env python3

from typing import Optional
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss


class CyclicWeight:
    def __init__(self, period: int, warmup: Optional[int] = 0):
        self.period = period
        self.warmup = warmup

    def __call__(self, step: int) -> float:
        if step <= self.warmup:
            beta = 0.0
        else:
            dt = (step - self.warmup) % (2 * self.period)
            if dt > self.period:
                # Hold at 1 for `period`
                beta = 1.0
            else:
                # Linearly ramp up to 1 for `period`
                beta = dt / self.period
        return beta


class UnitNormalKLDivLoss(nn.Module):
    """ KL divergence against centered unit-normal distribution """

    def __init__(self, reduction: Optional[str] = 'mean',
                 max_log_var: Optional[float] = 14.0):
        super().__init__()
        self.reduction = reduction
        if max_log_var is not None:
            assert (max_log_var >= 0)
        self.max_log_var = max_log_var

    def forward(self, mu: th.Tensor, log_std: th.Tensor) -> th.Tensor:
        log_var = (2.0 * log_std)
        if self.max_log_var is not None:
            with th.no_grad():
                # Approx. variance from this
                # would turn out to be
                # about 1e-6>
                log_var.clamp_(min=-self.max_log_var,
                               max=self.max_log_var)

        kl = -0.5 * th.sum(1 + log_var - th.square(mu)
                           - log_var.exp(), dim=-1)
        if self.reduction == 'none':
            return kl
        elif self.reduction == 'mean':
            kl = th.mean(kl) / mu.shape[-1]
        elif self.reduction == 'batchmean':
            kl = th.mean(kl, dim=0)
        else:
            raise KeyError(F'Unknown reduction = {self.reduction}')
        return kl


class SigmoidFocalLoss(nn.Module):
    """ Focal loss with binary cross entropy taking logits as inputs. """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets) -> th.Tensor:
        # FIXME: temporary scaling coefficient.
        return sigmoid_focal_loss(inputs, targets,
                                  alpha=self.alpha,
                                  gamma=self.gamma,
                                  reduction=self.reduction)


class GaussianKLDivLoss(nn.Module):
    """
    KL divergence between two gaussians.
    """

    def __init__(self, reduction: Optional[str] = 'mean',
                 max_log_var: Optional[float] = 14.0):
        super().__init__()
        self.reduction = reduction
        if max_log_var is not None:
            assert (max_log_var >= 0)
        self.max_log_var = max_log_var

    def forward(self, mu1, ls1, mu2, ls2):
        kl = gaussian_kl_div(mu1, ls1, mu2, ls2)  # .sum(dim=-1)
        if self.reduction == 'none':
            pass
        elif self.reduction == 'mean':
            kl = kl.mean()
        elif self.reduction == 'batchmean':
            kl = th.mean(kl.sum(dim=-1), dim=0)
        else:
            raise KeyError(F'Unknown reduction = {self.reduction}')
        return kl


def logistic_log_likelihood(x: th.Tensor, m: th.Tensor, log_s: th.Tensor,
                            log_eps: float = -7.0):
    """
    Log-likelihood of logistic distribution.

    Args:
        x:     the point at which to evaluate the log-likelihood.
        m:     the mode of the logistic distribution.
        log_s: the logarithm of the scale of the logistic distribution.
        log_eps: the minimum value at which to clamp log_s.

    Returns:
        $\\log{p_{logistic}(x|m,\\exp{log_s})}
    """
    # clamping for numerical stability
    log_s = log_s.clone()
    with th.no_grad():
        # Approx. variance from this
        # would turn out to be
        # about 1e-6>
        log_s.clamp_(min=log_eps)
    # Alternatively:
    # log_s = th.clamp_min(log_s, -7.0)
    z = -th.abs((x - m) / th.exp(log_s))
    loss = z - (2 * th.log1p(th.exp(z))) - log_s
    return th.mean(loss)


def kl_div_gaussian(q_mu, q_logvar, p_mu=None, p_logvar=None):
    '''Batched KL divergence D(q||p) computation.'''
    if p_mu is None or p_logvar is None:
        zero = q_mu.new_zeros(1)
        p_mu = p_mu or zero
        p_logvar = p_logvar or zero
    logvar_diff = q_logvar - p_logvar
    kl_div = -0.5 * (1.0 + logvar_diff - logvar_diff.exp() -
                     ((q_mu - p_mu) ** 2 / p_logvar.exp()))
    return kl_div.sum(dim=-1)


def gaussian_kl_div(mu1: th.Tensor, ls1: th.Tensor,
                    mu2: th.Tensor, ls2: th.Tensor,
                    min_log_var: float = -14.0
                    ) -> th.Tensor:
    """
    KL Divergence between two independent (diagonal) multivariate gaussians.

    Args:
        mu1: mean of the posterior.
        ls1: log-std of the posterior.
        mu2: mean of the prior.
        ls2: log-std of the prior.
        min_log_var: Minimum log-var, to clip for numerical stability.

    Returns:
        Elementwise KL divergence.

    Note:
        KL divergence is a measure of the information gained by
        revising one's beliefs from the prior probability P2 to the
        posterior probability P1.
        # so posterior should be complete, prior should be incomplete.
        # P1 is the "true" distribution;
        # P2 is the "approximate" distribution.
        Computes KL(P1||P2), where
        P1=N(mu1,exp(ls1)), P2=N(mu2,exp(ls2))
        so P2 is prior, P1 is posterior.
    """

    lv1 = (2 * ls1).clone()
    with th.no_grad():
        lv1.clamp_(min=min_log_var)
    lv2 = (2 * ls2).clone()
    with th.no_grad():
        lv2.clamp_(min=min_log_var)
    v1 = lv1.exp()
    v2 = lv2.exp()
    out1 = (ls2 - ls1 +
            (v1 + th.square(mu1 - mu2)) / (2 * v2)
            - 0.5)
    # out2 = kl_div_gaussian(mu1, lv1, mu2, lv2)

    # Q = th.distributions.Independent(
    #     th.distributions.Normal(mu1, ls1.exp()),
    #     1)
    # P = th.distributions.Independent(
    #     th.distributions.Normal(mu2, ls2.exp()),
    #     1)
    # out3 = th.distributions.kl_divergence(Q, P)
    return out1


class GenerativeLoss(nn.Module):
    """
    Auto-balancing VAE loss
    between MSE (reconstruction) and KL divergence
    terms.

    @see https://arxiv.org/pdf/2002.07514.pdf.
    """

    def __init__(self, alpha: float = 0.01, kappa: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.kappa = kappa
        self.register_buffer('gamma_rec',
                             th.ones((), dtype=th.float,
                                     requires_grad=False))
        self.register_buffer('gamma_reg',
                             th.ones((), dtype=th.float,
                                     requires_grad=False))

    def forward(self,
                loss_rec: th.Tensor,
                loss_reg: th.Tensor) -> th.Tensor:
        with th.no_grad():
            gamma_rec = th.sqrt(loss_rec)
            th.lerp(self.gamma_rec, gamma_rec, self.alpha,
                    out=self.gamma_rec)

            # NOTE:
            # For now, disable gamma updates
            # gamma_reg = th.sqrt(loss_reg)
            # th.lerp(self.gamma_reg, gamma_reg, self.alpha,
            #         out=self.gamma_reg)
        rec = th.where(self.gamma_rec <= 0, th.zeros_like(self.gamma_rec),
                       loss_rec / (self.gamma_rec ** 2))
        reg = th.where(self.gamma_reg <= 0, th.zeros_like(self.gamma_rec),
                       loss_reg / (self.gamma_reg ** 2))
        return (rec + self.kappa * reg)
