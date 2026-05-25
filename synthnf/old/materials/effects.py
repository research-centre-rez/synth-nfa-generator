import mitsuba as mi
import numpy as np


def color_model_faces(model_dict):
    model_dict["material"] = {
        "type": "diffuse",
        "reflectance": {
            "type": "mesh_attribute",
            "name": "face_color",
        },
    }
    mesh = mi.load_dict(model_dict)

    attribute_size = mesh.vertex_count() * 3
    mesh.add_attribute("face_color", 3, [0] * attribute_size)

    N = mesh.face_count()

    face_colors = np.repeat(np.random.rand(N) * 0.8, 3)

    mesh_params = mi.traverse(mesh)
    mesh_params["face_color"] = face_colors
    mesh_params.update()

    return mesh


def debug_pink():
    return {"type": "diffuse", "reflectance": {"type": "rgb", "value": [1, 0, 1]}}
