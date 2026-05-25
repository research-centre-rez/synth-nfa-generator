import numpy as np
import mitsuba as mi
import synthnf.config.defaults as defaults
import numpy.typing as npt
from dataclasses import dataclass

def create_fa_mesh(
    rod_curves,
    rod_centers,
    rod_width_mm,
    rod_height_mm=None,
    z_displacement=None,
    num_textures=None,
    cylinder_segments = 32,
):
    if z_displacement is None:
        z_displacement = 0
    else:
        z_displacement = np.array(z_displacement)
        if len(z_displacement.shape) == 1:
            z_displacement = np.expand_dims(z_displacement, axis=1)

    if num_textures is None:
        num_textures = len(rod_curves)

    rod_vertices_faces = [
        get_rod_geometry(100, cylinder_segments, rod_width_mm / 2, curve, height=rod_height_mm)
        for curve in rod_curves
    ]

    fa_vertices = np.array([r[0] for r in rod_vertices_faces])

    # shift in z-axis
    fa_vertices[:, :, 2] += z_displacement
    # move rod vertices according to rod centers
    fa_vertices[:, :, :2] += np.expand_dims(rod_centers, 1)

    fa_faces = [r[1] for r in rod_vertices_faces]
    v, f = bake_rod_geometry(fa_vertices, fa_faces)

    # process textures
    fa_uvs = [r[2] for r in rod_vertices_faces]

    uvs = np.concatenate(
        [(uv + [i % num_textures, 0]) for i, uv in enumerate(fa_uvs)], axis=0
    )
    uvs[:, 0] = uvs[:, 0] / num_textures

    return get_mesh(v, f, uvs)


def get_mesh(vertex_pos, face_indices, uv, mesh_name=None):
    mesh = mi.Mesh(
        mesh_name or "mesh",
        vertex_count=len(vertex_pos),
        face_count=len(face_indices),
        has_vertex_normals=False,
        has_vertex_texcoords=True,
    )

    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = np.ravel(vertex_pos)
    mesh_params["faces"] = np.ravel(face_indices)
    mesh_params["vertex_texcoords"] = np.ravel(uv)

    return mesh


def get_rod_geometry(rod_segments, cylinder_faces, radius=1, curve=None, height=None):
    height = height or 1
    actual_segments = rod_segments + 1
    curve_eval_points = np.linspace(0, 1, actual_segments)

    if curve is None:
        c_x = np.zeros((actual_segments,))
        c_y = np.copy(c_x)
        c_z = curve_eval_points * (height)
    else:
        c_x, c_y, c_z = curve.evaluate_multi(curve_eval_points)

    x = c_x
    y = c_y
    z = c_z

    spacing = np.linspace(0, 2 * np.pi, cylinder_faces)
    x_circle = np.sin(spacing) * radius
    y_circle = np.cos(spacing) * radius
    # z = -z

    cylinder_coor_shape = (actual_segments, len(x_circle))
    cylinder_x = np.broadcast_to(x_circle, cylinder_coor_shape) + np.broadcast_to(
        x[:, np.newaxis], cylinder_coor_shape
    )
    cylinder_y = np.broadcast_to(y_circle, cylinder_coor_shape) + np.broadcast_to(
        y[:, np.newaxis], cylinder_coor_shape
    )
    cylinder_z = np.broadcast_to(z[:, np.newaxis], cylinder_coor_shape)
    grid_mesh = np.concatenate(
        [
            cylinder_x[:, :, np.newaxis],
            cylinder_y[:, :, np.newaxis],
            cylinder_z[:, :, np.newaxis],
        ],
        axis=2,
    )
    vertex_pos = grid_mesh.reshape((-1, 3))
    face_indices = []
    for i in range(actual_segments - 1):
        for j in range(cylinder_faces - 1):
            # a -- b
            # | -- |
            # c -- d
            idxs = np.array([(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]).T
            a, b, c, d = np.ravel_multi_index(idxs, (actual_segments, cylinder_faces))

            # face_indices.append([a,b,c])
            # face_indices.append([c,b,d])
            # inward faces
            face_indices.append([a, c, b])
            face_indices.append([c, d, b])

    # Texture coors
    y_coors = np.linspace(0, 1, actual_segments)
    uv_y = np.multiply(
        np.expand_dims(y_coors, axis=1), np.ones(cylinder_faces)[None]
    ).flatten()
    uv_x = np.concatenate([np.linspace(0, 1, cylinder_faces)] * (actual_segments))
    uv = np.vstack([uv_x, uv_y]).T

    vertices = np.array(vertex_pos)
    faces = np.array(face_indices, dtype=np.int32)
    return vertices, faces, uv


@dataclass
class RodCenters:
    centers:npt.NDArray
    rods_per_face:int

    def face_rods_idcs(self,face_num:int):
        if not (0 < face_num <= 6):
            raise ValueError("Face Number is not in 1,2 ... 6")

        start_rod = (self.rods_per_face - 1)* (face_num-1)
        rod_ids = np.arange(start_rod,start_rod+self.rods_per_face)
        if face_num==6:
            rod_ids[-1] = 0
        
        return rod_ids
    

def generate_rod_centers_hexagon(
    rods_per_face=None, rod_width_mm=None, rod_gap_mm=None, outern_layers=None
):
    rod_width_mm = rod_width_mm or defaults.blueprints.fuel_rod.width_mm
    rod_gap_mm = rod_gap_mm or defaults.blueprints.fuel_rod.gap_mm
    rods_per_face = defaults.blueprints.fa.rods_per_face

    from_layer = -1
    if outern_layers:
        from_layer = rods_per_face - 1 - outern_layers

    rod_center_distance_mm = rod_width_mm + rod_gap_mm

    pythagorean_factor = np.sqrt(3) / 2
    rod_centers = np.zeros((0, 2))

    for layer in range(rods_per_face - 1, from_layer, -1):
        if layer == 0:
            rod_centers = np.vstack([rod_centers, [[0, 0]]])
            break

        side_width = rod_center_distance_mm * layer
        corners = [
            [-side_width / 2, side_width * pythagorean_factor],  # 11 oclock,
            [side_width / 2, side_width * pythagorean_factor],  # 1 oclock
            [side_width, 0],  # 3 oclock
            [side_width / 2, -side_width * pythagorean_factor],  # 5 oclock
            [-side_width / 2, -side_width * pythagorean_factor],  # 7 oclock,
            [-side_width, 0],  # 9 oclock
        ]

        rc = []
        for c1, c2 in zip(corners, corners[1:] + [corners[0]]):
            c1_x, c1_y = c1
            c2_x, c2_y = c2

            xx = np.linspace(c1_x, c2_x, layer, endpoint=False)
            yy = np.linspace(c1_y, c2_y, layer, endpoint=False)

            rc.append(np.stack([xx, yy]).T)

        rc = np.vstack(rc)
        rod_centers = np.vstack([rod_centers, rc])
    return RodCenters(rod_centers,rods_per_face)


def generate_rod_centers(*args, **kwargs):
    shape = "hexagon"
    if "shape" in kwargs:
        shape = kwargs["shape"]
        del kwargs["shape"]

    match shape:
        case "hexagon":
            return generate_rod_centers_hexagon(*args, **kwargs)
        case _:
            raise Exception("Not supported shape")


def bake_rod_geometry(vertices_list, faces_list):
    vertices = np.concatenate(vertices_list, axis=0)
    offsets = np.cumsum(np.concatenate([[0], [len(v) for v in vertices_list]]))

    faces = np.concatenate(
        [np.add(f, offset) for f, offset in zip(faces_list, offsets)]
    )

    return vertices, faces
