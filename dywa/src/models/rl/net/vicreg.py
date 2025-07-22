#!/usr/bin/env python3

"""
Adopted from VICRegLoss in pytorch-metric-learning.
See:
    https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#vicregloss
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class VICRegLoss(nn.Module):
    def __init__(
            self, invariance_lambda: float = 25,
            variance_mu: float = 25,
            covariance_v: float = 1,
            eps: float = 1e-3,
            **kwargs
    ):
        super().__init__()
        """
        The overall loss function is a weighted average of the invariance, variance and covariance terms:
            L(Z, Z') = λs(Z, Z') + µ[v(Z) + v(Z')] + ν[c(Z) + c(Z')],
        where λ, µ and ν are hyper-parameters controlling the importance of each term in the loss.
        """
        self.invariance_lambda = invariance_lambda
        self.variance_mu = variance_mu
        self.covariance_v = covariance_v
        self.eps = eps

    def compute_loss(self, z1, z2):
        if self.invariance_lambda != 0:
            invariance_loss = self.invariance_lambda * self.invariance_loss(
                z1, z2
            )
        else:
            invariance_loss = th.as_tensor(0.0,
                                           dtype=z1.dtype,
                                           device=z1.device)
        variance_loss1, variance_loss2 = self.variance_loss(z1, z2)
        covariance_loss = self.covariance_v * self.covariance_loss(z1, z2)
        return {
            'varloss': (self.variance_mu * (variance_loss1 + variance_loss2)),#.mean(),
            'invloss': (invariance_loss),#.mean(),
            'covloss': (covariance_loss)#.mean()
        }

    def invariance_loss(self, emb, ref_emb):
        return th.mean((emb - ref_emb) ** 2, dim=1)

    def variance_loss(self, emb, ref_emb):
        std_emb = th.sqrt(emb.var(dim=0) + self.eps)
        std_ref_emb = th.sqrt(ref_emb.var(dim=0) + self.eps)
        # / 2 for averaging
        return F.relu(1 - std_emb) / 2, F.relu(1 - std_ref_emb) / 2

    def covariance_loss(self, emb, ref_emb):
        N, D = emb.size()
        emb = emb - emb.mean(dim=0)
        ref_emb = ref_emb - ref_emb.mean(dim=0)
        cov_emb = (emb.T @ emb) / (N - 1)
        cov_ref_emb = (ref_emb.T @ ref_emb) / (N - 1)

        diag = th.eye(D, device=cov_emb.device)
        cov_loss = (
            cov_emb[~diag.bool()].pow_(2).sum() / D
            + cov_ref_emb[~diag.bool()].pow_(2).sum() / D
        )
        return cov_loss

    def forward(self, z1, z2):
        return self.compute_loss(z1, z2)
