#!/usr/bin/env python3

from typing import Optional
import numpy as np


class SubsampleQueries:
    """ Subsample `query_points` from dataset."""

    def __init__(self, n_query: int = 256,
                 seed: int = 0,
                 pos_fraction: Optional[float] = None):
        """
        Args:
            n_query: number of query points to sample.
            seed: RNG seed for np.random.default_rng.
            pos_fraction: Optional. If supplied, the positive
                and negative query points will be balanced
                based on the supplied fraction.
        """
        self.n_query: int = n_query
        self.rng = np.random.default_rng(seed)

        # Balance positive and negative samples.
        self.pos_fraction = pos_fraction
        self.balance = (self.pos_fraction is not None)

    def __call__(self, data):
        n_points: int = len(data['occ_labels'])

        if self.balance:
            p = float(self.pos_fraction)
            # Balanced draw of positive/negative queries.
            n_pos = np.count_nonzero(data['occ_labels'])
            n_neg = n_points - n_pos
            p_pos = (p) / n_pos
            p_neg = (1.0 - p) / n_neg
            pvals = np.where(data['occ_labels'], p_pos, p_neg)
            indices = self.rng.choice(
                n_points, size=self.n_query, p=pvals, replace=(
                    self.n_query > np.count_nonzero(p)))
        else:
            # Randomized draw of positive/negative queries.
            indices = self.rng.integers(n_points, size=self.n_query)

        out = dict(data)
        out['occ_labels'] = data['occ_labels'][indices]
        out['query_points'] = data['query_points'][indices]
        return out
