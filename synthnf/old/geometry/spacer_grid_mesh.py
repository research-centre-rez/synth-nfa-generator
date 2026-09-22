import numpy as np
import synthnf.io as io
from synthnf.geometry import compose_mesh


def create_grid_mesh_from_tooth(
    tooth_mesh_data, rod_count, rod_width_mm, rod_gap_width_mm
):
    p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z = io.decompose_ply(tooth_mesh_data)

    p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z = mirror_down(
        p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z
    )

    p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z = array_left(
        p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, n=rod_count - 2
    )

    grid_info = p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z

    distance = -grid_dist_from_center(rod_count, rod_width_mm, rod_gap_width_mm)

    # UV mapping
    u = 1 - _norm(p_x)  # It was arraye'd left so uv mapping is reversed
    v = _norm(p_z)
    p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u, v = array_hexagon(
        distance, p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u, v
    )
    px, py, pz, *_ = grid_info

    min_px_idc = np.squeeze(np.argwhere(px == np.min(px)))
    left_pz = p_z[min_px_idc]
    min_px_idc = min_px_idc[np.argsort(left_pz)]

    max_px_idc = np.squeeze(np.argwhere(px == np.max(px)))
    right_pz = p_z[max_px_idc]
    max_px_idc = max_px_idc[np.argsort(right_pz)]

    left_indices = min_px_idc[None] + np.arange(6)[:, None] * len(px)
    right_indices = max_px_idc[None] + np.arange(6)[:, None] * len(px)
    right_indices = np.roll(right_indices, -1, axis=0)

    p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u, v = fill_borders(
        p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u, v, left_indices, right_indices
    )

    return compose_mesh(p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u, v)


def fill_borders(
    p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u, v, left_indices, right_indices
):
    # add faces

    stich_v1 = []
    stich_v2 = []
    stich_v3 = []
    for l, r in zip(left_indices, right_indices):  # noqa: E741
        for a, b, c in zip(l, l[1:], r):
            stich_v1.append(a)
            stich_v2.append(b)
            stich_v3.append(c)

        for a, b, c in zip(l[1:], r[1:], r):
            stich_v1.append(a)
            stich_v2.append(b)
            stich_v3.append(c)

    v_1 = np.concatenate([v_1, stich_v1])
    v_2 = np.concatenate([v_2, stich_v2])
    v_3 = np.concatenate([v_3, stich_v3])

    n_x = n_x.copy()
    n_y = n_y.copy()

    # fix normals with averaged normals
    for l, r in zip(left_indices, right_indices):  # noqa: E741
        n_x[l] = (n_x[l] + n_x[r]) / 2
        n_x[r] = n_x[l]

        n_y[l] = (n_y[l] + n_y[r]) / 2
        n_y[r] = n_y[l]

    return p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u, v


def mirror_down(p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z):
    n_points = len(p_x)
    p_x = np.concatenate([p_x, p_x])
    p_y = np.concatenate([p_y, p_y])
    p_z = np.concatenate([p_z, -p_z + 2 * np.min(p_z)])

    v_1_m = v_1 + n_points
    v_2_m = v_3 + n_points
    v_3_m = v_2 + n_points

    v_1 = np.concatenate([v_1, v_1_m])
    v_2 = np.concatenate([v_2, v_2_m])
    v_3 = np.concatenate([v_3, v_3_m])

    n_x = np.concatenate([n_x, n_x])
    n_y = np.concatenate([n_y, n_y])
    n_z = np.concatenate([n_z, -n_z])

    return p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z


def array_left(p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, n=1, width=None):
    if n == 0:
        return p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z

    n_points = len(p_x)

    if width is None:
        x_min = np.min(p_x)
        x_max = np.max(p_x)
        width = x_max - x_min
    width_shifts = -np.arange(0, n + 1) * width

    p_x = np.concatenate(np.stack([p_x] * (n + 1)) + width_shifts[:, None])

    p_y = np.concatenate([p_y] * (n + 1))
    p_z = np.concatenate([p_z] * (n + 1))

    v_1 = np.concatenate([v_1 + n_points * i for i in range(0, n + 1)])
    v_2 = np.concatenate([v_2 + n_points * i for i in range(0, n + 1)])
    v_3 = np.concatenate([v_3 + n_points * i for i in range(0, n + 1)])

    n_x = np.concatenate([n_x] * (n + 1))
    n_y = np.concatenate([n_y] * (n + 1))
    n_z = np.concatenate([n_z] * (n + 1))

    return p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z


def rotate_2d(deg, points):
    assert points.shape[0] == 2, "array should have 2 rows and n columns"
    angle = np.deg2rad(deg)
    s = np.sin(angle)
    c = np.cos(angle)

    rot = np.array([[c, s], [-s, c]])
    return np.dot(rot, points)


def array_hexagon(dist, p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u, v):
    deg = 60
    n_points = len(p_x)
    p_x = p_x - np.mean(p_x)
    p_y = p_y - np.mean(p_y) + dist
    p_z = p_z - np.mean(p_z)

    min_p_x_arg = np.argmin(p_x)
    max_p_x_arg = np.argmax(p_x)

    for i in range(1, 6):
        r_p_x, r_p_y = rotate_2d(deg * i, np.stack([p_x[:n_points], p_y[:n_points]]))

        p_x = np.concatenate([p_x, r_p_x])
        p_y = np.concatenate([p_y, r_p_y])

        r_n_x, r_n_y = rotate_2d(deg * i, np.stack([n_x[:n_points], n_y[:n_points]]))

        n_x = np.concatenate([n_x, r_n_x])
        n_y = np.concatenate([n_y, r_n_y])
        n_z = np.concatenate([n_z, n_z[:n_points]])

    p_z = np.concatenate([p_z] * 6)

    v_1 = np.concatenate([v_1 + n_points * i for i in range(0, 6)])
    v_2 = np.concatenate([v_2 + n_points * i for i in range(0, 6)])
    v_3 = np.concatenate([v_3 + n_points * i for i in range(0, 6)])

    # UV recalc
    v = np.tile(v, 6)
    # Need to add a bit of padding to right to make space for filling

    # remember a position of top left point from 1st grid

    one_grid_n = len(p_x) // 6
    face_width = np.abs(p_x[max_p_x_arg] - p_x[min_p_x_arg])
    filling_width = np.abs(p_x[min_p_x_arg] - p_x[max_p_x_arg + one_grid_n])

    # this is the width of a single face +1 filling on UV coords
    u_part_single = 1 / (face_width + filling_width)
    # this is all of them
    u_part_hex = u_part_single / 6

    u = np.concatenate([u * u_part_hex + 1 / 6 for i in range(6)])

    return p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u, v


def grid_dist_from_center(visible_rod_count, rod_width_mm, gap_between_rods_width_mm):
    rod_w_gap_mm = rod_width_mm + gap_between_rods_width_mm
    total_mm = rod_w_gap_mm * visible_rod_count - rod_width_mm

    return np.sqrt(total_mm**2 - (total_mm / 2) ** 2) + 3


def _norm(arr):
    mi = np.min(arr)
    ma = np.max(arr)
    return (arr - mi) / (ma - mi)
