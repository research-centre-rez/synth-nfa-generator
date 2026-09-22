import matplotlib.pyplot as plt
import numpy as np


def plot_bow(curve_fa=None, curves_rod=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(4, 5))

    if curve_fa:
        draw_curve(curve_fa, ax=ax, label="FA", linewidth=3, c="red")

    if curves_rod:
        for i, div_curve in enumerate(curves_rod):
            draw_curve(div_curve, ax=ax, label=f"Rod {i+1}", alpha=0.5)

    plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.ylabel("FA Height[mm]")
    plt.xlabel("Bow Divergence[mm]")
    plt.subplots_adjust(right=0.6, top=0.9)


def draw_curve(curve, samples=101, ax=None, **kwargs):
    if ax is None:
        ax = plt

    zs = np.linspace(0, 1, samples + 1)
    points = curve.evaluate_multi(zs)
    xx = points[0]
    zz = points[2]
    ax.plot(xx, zz, **kwargs)
