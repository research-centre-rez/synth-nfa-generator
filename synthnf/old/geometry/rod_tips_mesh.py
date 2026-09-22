import synthnf.geometry
import numpy as np
import synthnf.io as io


def create_mesh(ply, scale, rod_centers, z_displacement=None):
    z_displacement = (
        z_displacement if z_displacement is not None else [0] * len(rod_centers)
    )

    p_xs = []
    p_ys = []
    p_zs = []
    v_1s = []
    v_2s = []
    v_3s = []
    n_xs = []
    n_ys = []
    n_zs = []

    p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z = io.decompose_ply(ply)
    for i, ((rx, ry), rz) in enumerate(zip(rod_centers, z_displacement)):
        n = i * len(p_x)

        mp_x = (scale * p_x) + rx
        mp_y = (scale * p_y) + ry
        mp_z = p_z * scale + rz

        p_xs.append(mp_x)
        p_ys.append(mp_y)
        p_zs.append(mp_z)
        v_1s.append(v_1 + n)
        v_2s.append(v_2 + n)
        v_3s.append(v_3 + n)
        n_xs.append(n_x)
        n_ys.append(n_y)
        n_zs.append(n_z)

    return synthnf.geometry.compose_mesh(
        np.concatenate(p_xs),
        np.concatenate(p_ys),
        np.concatenate(p_zs),
        np.concatenate(v_1s),
        np.concatenate(v_2s),
        np.concatenate(v_3s),
        np.concatenate(n_xs),
        np.concatenate(n_ys),
        np.concatenate(n_zs),
        mesh_name="tips",
    )
