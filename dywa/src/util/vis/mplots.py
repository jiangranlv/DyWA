#!/usr/bin/env python3

from typing import Optional
from matplotlib import pyplot as plt
import numpy as np


class SimpleStreamingPlot:
    def __init__(self, ax: Optional[plt.Axes] = None,
                 lims=None):
        # ??
        plt.ion()

        if ax is None:
            ax = plt.gca()
        self.ax = ax
        ax.axhline()
        self.ln, = ax.plot([], [])
        self.lims = lims

    def update(self, *args, **kwds):
        ax = self.ax
        ln = self.ln

        ln.set_data(*args, **kwds)
        xy = ln.get_xydata()
        x = xy[..., 0]
        y = xy[..., 1]
        xmin = np.quantile(x, 0.05)
        xmax = np.quantile(x, 0.95)
        ymin = np.quantile(y, 0.05)
        ymax = np.quantile(y, 0.95)
        ax.set_xlim(xmin, xmax)

        if self.lims is not None:
            ax.set_ylim(self.lims[0], self.lims[1])
        else:
            ax.set_ylim(ymin, ymax)

        ln.figure.canvas.flush_events()
        plt.pause(0.001)
