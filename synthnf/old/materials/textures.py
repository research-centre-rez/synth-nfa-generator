import numpy as np
import perlin_numpy

import synthnf.utils as ut


def rod_noise(
    *,
    rod_height_mm=None,
    rod_width_mm=None,
    resolution_factor=1,
    rnd=None,
    seed=None,
):
    if rod_height_mm and rod_width_mm:
        circumference_mm = rod_width_mm * np.pi
        vertical_frequency = 2 ** int(np.log2(rod_height_mm / circumference_mm))
    else:
        vertical_frequency = 128

    # HACK
    # vertical_frequency = vertical_frequency//2

    if seed is not None:
        rnd = np.random.RandomState(seed=seed)

    base_res = 8 * resolution_factor

    return perlin_numpy.generate_fractal_noise_2d(
        np.array([base_res * vertical_frequency, base_res]),
        [1 * vertical_frequency, 1],
        2,
        persistence=0.5,
        lacunarity=2,
        tileable=(False, True),
        rnd=rnd,
    )


def grid_noise(
    *,
    resolution_factor=1,
    horizontal_fequency=24,
    rnd=None,
    seed=None,
):
    if seed is not None:
        rnd = np.random.RandomState(seed=seed)

    # Based on measurements
    base_res = 32 * resolution_factor
    frequency = np.array([1, horizontal_fequency])
    shape = frequency * base_res
    return perlin_numpy.generate_fractal_noise_2d(
        shape,
        frequency,
        2,
        persistence=0.5,
        lacunarity=2,
        tileable=(True, True),
        rnd=rnd,
    )


def grid_bumpmap(*, resolution_factor=1, horizontal_fequency=4, rnd=None, seed=None):
    if seed is not None:
        rnd = np.random.RandomState(seed=seed)

    base_res = 32 * resolution_factor
    frequency = np.array([1, horizontal_fequency])
    shape = frequency * base_res
    return perlin_numpy.generate_fractal_noise_2d(
        shape,
        frequency,
        2,
        persistence=0.5,
        lacunarity=2,
        tileable=(True, True),
        rnd=rnd,
    )


def blend(tex, tex_weight, blend_ratio, clip=True):
    r = (ut.normalize(tex) - 0.5) * tex_weight

    offset = (1 - blend_ratio) * (-tex_weight) + (1 + tex_weight) * blend_ratio
    r = r + offset
    if clip:
        r = np.clip(r, 0, 1)

    return r
