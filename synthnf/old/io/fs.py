import pathlib
import tempfile
import imageio
import numpy as np
import json


def create_temp_folder():
    temp_dir = pathlib.Path(tempfile.TemporaryDirectory().name)
    temp_dir = temp_dir.parent / "synthnfa" / temp_dir.name

    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def save_mesh_to_temp(mesh):
    temp_dir = create_temp_folder()
    mesh_path = str(temp_dir / "grid.ply")
    mesh.write_ply(mesh_path)
    return mesh_path


def save_float_image_to_temp(img, filename="image.png", temp_dir=None):
    if temp_dir is None:
        temp_dir = create_temp_folder()

    path = temp_dir / "image.png"
    imwrite(path, img)
    return path


def imwrite(filepath, img):
    ensure_dir_exists(filepath)
    img_c = np.uint8(np.clip(0, 1, img) * 255)
    imageio.imwrite(filepath, img_c)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def ensure_dir_exists(filepath):
    dir_path = pathlib.Path(filepath).parent
    dir_path.mkdir(exist_ok=True, parents=True)


def save_json(path, obj):
    ensure_dir_exists(path)
    with open(path, "w") as f:
        return json.dump(obj, f)
