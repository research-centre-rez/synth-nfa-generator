import numpy as np

import synthnf.geometry.curves as curves


def random_bow(
    total_rod_count,
    max_bow_mm,
    max_divergence_mm,
    spans,
    span_margin_mm,
    rnd=None,
    seed=None,
):
    rnd = rnd or np.random.RandomState(seed)
    bow_mm = (rnd.rand(2) * 2 - 1) * max_bow_mm
    divergences_xy_mm = np.array(
        [
            rnd_divergence_adjusted(spans, max_divergence_mm, rnd)
            for _ in range(total_rod_count)
        ]
    )

    return create_nfa_curves(bow_mm, divergences_xy_mm, span_margin_mm, spans)


def create_nfa_curves(bow_xy_mm, rods_divergences_xy_mm, span_margin_mm, spans):
    fa_height_mm = np.max(spans)
    bow_x, bow_y = bow_xy_mm
    # LIMITED to a single bent
    bezier_nodes = np.array(
        [
            [0, 0, 0],  # bottom
            [0, 0, fa_height_mm * 0.05],  # top
            [bow_x, bow_y, fa_height_mm * 1 / 3],
            [0, 0, fa_height_mm * 2 / 3],
            [0, 0, fa_height_mm * 3 / 4],  # until 1 third there is nothing happening
            [0, 0, fa_height_mm],  # top
        ]
    )
    curve_fa = curves.bezier_from_nodes(bezier_nodes)
    # This steps is here to overcome issue with beziers.
    # In bezier, if you sample it by linear t (e.g. np.linspace(0,1,11),
    # you get x,y,z where z does not grow lineary
    # this is a huge problem if you want to append them together with other functions
    # The bezier module does NOT implement curve-plane intersection therefore
    # this workaroud is required.
    curve_fa = curves.interpolate_z(curve_fa, np.linspace(0, 1, 1001))

    curves_rod = []
    for rod_div_xy in rods_divergences_xy_mm:
        piecewise_curve = curves.construct_piecewise_curve(
            spans, rod_div_xy, span_margin_mm
        )
        curves_rod.append(curves.merge_curves(curve_fa, piecewise_curve))

    return curve_fa, curves_rod


def rnd_divergence_adjusted(pieces, max_divergence, rnd=None):
    if rnd is None:
        rnd = np.random.RandomState()
    heights = np.diff(pieces)
    max_div = np.max(heights)
    # ensure divergence are smaller for smaller spans
    divergence_factor = heights / max_div

    divergences = (
        (rnd.rand(len(heights), 2) * 2 - 1) * max_divergence * divergence_factor[None].T
    )
    return np.nan_to_num(divergences)
