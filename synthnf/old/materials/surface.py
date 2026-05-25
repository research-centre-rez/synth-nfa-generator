import numpy as np
import pandas as pd
import synthnf.io as io
import synthnf.config.assets as assets
from synthnf.config.defaults import materials as defmat


def create_gray_diffuse(intensity):
    return {
        "type": "diffuse",
        "reflectance": {
            "type": "spectrum",
            "value": intensity,
        },
    }


def inconel_blend(
    cloudiness=None, gray_diffuse=None, blend_texture=None, alpha_u=None, alpha_v=None
):
    inconel = create_inconel(alpha_u, alpha_v)

    cloudiness = cloudiness or 0.5
    if blend_texture is None:
        blend_texture = np.ones((2, 2)) * cloudiness

    gray_diffuse = gray_diffuse or 1
    bsdf = create_gray_diffuse(gray_diffuse)

    return blend_materials(inconel, bsdf, blend_texture)


def zirconium_blend(
    cloudiness=None, gray_diffuse=None, blend_texture=None, alpha_u=None, alpha_v=None
):
    gray_diffuse = gray_diffuse or 1
    cloudiness = cloudiness or 0.5
    if blend_texture is None:
        blend_texture = np.ones((2, 2)) * cloudiness

    bsdf = create_gray_diffuse(gray_diffuse)
    zircon_conductor = create_zirconium(alpha_u=alpha_u, alpha_v=alpha_v)

    return blend_materials(zircon_conductor, bsdf, blend_texture)


def create_zirconium(alpha_u=None, alpha_v=None):
    alpha_u = alpha_u or defmat.zirconium.alpha_u
    alpha_v = alpha_v or defmat.zirconium.alpha_v

    df_zirconium = pd.read_csv(assets.get_asset_path("spectrum_zirconium.csv"))
    wv = df_zirconium["wavelength"] * 1000
    n = df_zirconium["n"]
    eta = df_zirconium["k"]
    eta_spectrum = list(zip(wv, n))
    k_spectrum = list(zip(wv, eta))
    return create_roughconductor(eta_spectrum, k_spectrum, alpha_u, alpha_v)


def create_inconel(alpha_u=None, alpha_v=None):
    alpha_u = alpha_u or defmat.zirconium.alpha_u
    alpha_v = alpha_v or defmat.zirconium.alpha_v

    df_zirconium = pd.read_csv(assets.get_asset_path("spectrum_inconel.csv"))
    wv = df_zirconium["wavelength"] * 1000
    n = df_zirconium["n"]
    eta = df_zirconium["k"]
    eta_spectrum = list(zip(wv, n))
    k_spectrum = list(zip(wv, eta))
    return create_roughconductor(eta_spectrum, k_spectrum, alpha_u, alpha_v)


def blend_materials(mat_1, mat_2, texture=None, float_weight=None):
    blend_dict = {"type": "blendbsdf", "bsdf_0": mat_1, "bsdf_1": mat_2}
    if texture is not None:
        blend_path = io.save_float_image_to_temp(texture)
        blend_dict["weight"] = {"type": "bitmap", "filename": str(blend_path)}
        return blend_dict

    if float_weight is None:
        float_weight = 0.5

    blend_dict["weight"] = {"type": "spectrum", "value": float_weight}

    return blend_dict


def create_roughconductor(eta_spectrum, k_spectrum, alpha_u, alpha_v):
    return {
        "type": "roughconductor",
        "eta": {
            "type": "spectrum",
            "value": eta_spectrum,
        },
        "k": {
            "type": "spectrum",
            "value": k_spectrum,
        },
        "distribution": "ggx",
        "alpha_u": alpha_u,
        "alpha_v": alpha_v,
        "sample_visible": False,
    }
