import mitsuba as mi
import numpy as np
import drjit as dr


def compose_mesh(
    p_x, p_y, p_z, v_1, v_2, v_3, n_x, n_y, n_z, u=None, v=None, mesh_name=None
):
    vertex_pos = mi.Point3f(np.float32(p_x), np.float32(p_y), np.float32(p_z))
    vertex_norm = mi.Vector3f(np.float32(n_x), np.float32(n_y), np.float32(n_z))

    face_indices = mi.Vector3u([np.float32(v_1), np.float32(v_2), np.float32(v_3)])

    has_vertex_texcoords = u is not None and v is not None

    mesh = mi.Mesh(
        mesh_name or "grid_teeth",
        vertex_count=len(vertex_pos[0]),
        face_count=len(face_indices[0]),
        has_vertex_normals=True,
        has_vertex_texcoords=False,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = dr.ravel(vertex_pos)
    mesh_params["faces"] = dr.ravel(face_indices)
    mesh_params["vertex_normals"] = dr.ravel(vertex_norm)

    if has_vertex_texcoords:
        uv = np.vstack([u, v]).T
        mesh_params["vertex_texcoords"] = np.ravel(uv)

    return mesh
