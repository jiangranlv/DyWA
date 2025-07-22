#!/usr/bin/env python3

import numpy as np


def ecdf(x: np.ndarray, num: int = 100):
    """
    Empirical CDF.

    Returns:
        p: percentiles
        v: values at those percentiles.
    """
    p = np.linspace(0, 100, num=num)
    v = np.percentile(x, p)
    return (p, v)
