import skimage.draw as skdraw

import scipy.ndimage as ndi
import cv2

import numpy as np
from scipy.interpolate import splprep, splev


def get_spots_texture(
    height,
    width,
    n_spots,
    spot_radius_mean_px=10,
    spot_radius_var_px=2,
    blur_sigma=2,
    seed=None,
):
    rnd = np.random.default_rng(seed=seed)
    # TODO custom distribution
    canvas = np.zeros((height, width))
    ys = np.int32(rnd.beta(8, 3, size=n_spots) * height)
    xs = rnd.integers(0, width, size=n_spots)
    radii = np.abs(rnd.normal(spot_radius_mean_px, spot_radius_var_px, n_spots)).astype(
        int
    )
    alpha = np.minimum(np.random.normal(0.8, 0.2, n_spots), 1)
    for i, (y, x, r, a) in enumerate(zip(ys, xs, radii, alpha)):
        spot = get_spot(r * 2, r * 2, rnd=rnd) * a
        canvas = merge_spot(canvas, spot, y, x, x_circular=True)

    canvas = ndi.gaussian_filter(canvas, blur_sigma, mode="wrap")
    return canvas


def merge_spot(canvas, spot, y, x, x_circular=True, y_circular=False):
    # max_y = canvas.shape[0] - 1
    canvas_shape = np.array(canvas.shape)
    spot_shape = np.array(spot.shape)
    offsets = spot_shape // 2

    # corner_tl = np.array([0, 0])
    # corner_br = canvas_shape

    origin = np.array([y, x])
    # top_left = origin - offsets
    # bottom_right = top_left + spot_shape

    crop_tl = -np.minimum(0, origin - offsets)
    crop_br = np.maximum((origin + offsets) - canvas_shape, 0)

    if not y_circular:
        spot = spot[crop_tl[0] : -crop_br[0] or None]
        offsets = offsets - [0, crop_tl[0]]

    if not x_circular:
        spot = spot[:, crop_tl[1] : -crop_br[1] or None]
        offsets = offsets - [0, crop_tl[1]]

    spot_shape = np.array(spot.shape)

    for yy in range(spot_shape[0]):
        for xx in range(spot_shape[1]):
            idx = np.array([yy, xx])
            pos = origin - offsets + idx
            canvas_idx = np.mod(pos, canvas_shape)

            canvas[canvas_idx[0], canvas_idx[1]] = np.maximum(
                spot[yy, xx], canvas[canvas_idx[0], canvas_idx[1]]
            )
    return canvas


def _ensure_channels(a):
    if len(a.shape) == 2:
        a = np.dstack([a, a, a, np.ones_like(a)])
    return a


def mode_overlay_bw(a, b, b_alpha=None):
    a = _ensure_channels(a)
    b = _ensure_channels(b)

    if b_alpha is None:
        b_alpha = np.ones(b.shape[0], b.shape[1])
    b_alpha = b_alpha[:, :, None]

    assert a.shape == b.shape, f"{a.shape=} {b.shape=}"

    a_orig = a.copy()
    ab = np.zeros_like(a)

    mask = a >= 0.5
    ab[~mask] = (2 * a * b)[~mask]
    ab[mask] = (1 - 2 * (1 - a) * (1 - b))[mask]

    res = ab * b_alpha + a_orig * (1 - b_alpha)
    return np.squeeze(res)


def interpolate_with_curve(xs, ys, n=100):
    points = np.array([xs, ys])
    ggx, ggy = splprep(points, u=None, s=0.0, per=1)

    gg_xspline = np.linspace(ggy.min(), ggy.max(), n)
    ggxnew, ggynew = splev(gg_xspline, ggx, der=0)
    return np.array([ggxnew, ggynew])


def fill_blob(width, height, points):
    canvas = np.zeros((height, width), dtype=np.uint8)
    contours = (points * np.array([[width, height]]).T).astype(int).T
    cv2.fillPoly(canvas, pts=[contours], color=(255, 255, 255))
    return canvas


def get_blob(width, height, n=4, rnd=None, var=0.05):
    if rnd is None:
        rnd = np.random.default_rng()
    x = np.linspace(0, 2 * np.pi, n, endpoint=True)
    x = x + rnd.random()
    shifts = rnd.normal(1, var, size=(n,))
    xs = np.sin(x) * shifts
    ys = np.cos(x) * shifts

    interpolated_points = interpolate_with_curve(xs, ys)
    p_min = np.min(interpolated_points)
    p_max = np.max(interpolated_points)

    interpolated_points = (interpolated_points - p_min) / (p_max - p_min)
    blob = fill_blob(width, height, interpolated_points)
    blob = np.float32(blob) / 255
    return blob, interpolated_points


def get_circle(radius):
    circle = np.zeros((radius * 4, radius * 4), dtype=float)
    rr, cc = skdraw.disk((radius * 2, radius * 2), radius)
    circle[rr, cc] = 1
    return circle


def get_spot(width, height, sigma=None, rnd=None):
    if rnd is None:
        rnd = np.random.default_rng()

    # radius = min(width, height) // 2

    if width <= 2 or height <= 2:
        return np.zeros((width, height))

    spot, _ = get_blob(width - 2, height - 2, rnd=rnd)
    spot = np.pad(spot, 1)

    return spot
