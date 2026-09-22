import numpy as np


import json
import synthnf.geometry.fuel_rods_mesh as frm
from . import artificial as simu
import synthnf.materials.textures as textures
import synthnf.config.defaults as defaults
import synthnf.materials as mat
import synthnf.utils as utils


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class RandomParameters:
    def __init__(
        self,
        *,
        rod_centers=None,
        rod_height_mm=None,
        rod_width_mm=None,
        rod_gap_mm=None,
        rods_per_face=None,
        grid_tops_mm=None,
        span_margin_mm=None,
        max_bow_mm=None,
        max_divergence_mm=None,
        max_z_displacement_mm=None,
        n_textures=None,
        cloudiness=None,
        swing_max_angle_deg=None,
        swing_periods_per_inspection=None,
        swing_deg_xy=0,
        cam_res_x = None,
        cam_res_y = None,
        seed=None,
    ):
        self.n_textures = n_textures or 1
        self.seed = (
            seed if seed is not None else np.random.randint(np.iinfo(np.int32).max)
        )
        self.rnd = np.random.RandomState(seed=self.seed)

        self.rod_height_mm = rod_height_mm or defaults.blueprints.fuel_rod.height_mm
        self.rod_width_mm = rod_width_mm or defaults.blueprints.fuel_rod.width_mm
        self.rod_gap_mm = rod_gap_mm or defaults.blueprints.fuel_rod.gap_mm
        self.grid_tops_mm = grid_tops_mm or np.array(
            defaults.blueprints.fa.grid_tops_mm
        )
        self.rods_per_face = rods_per_face or defaults.blueprints.fa.rods_per_face

        self.rod_centers = (
            rod_centers
            if rod_centers is not None
            else frm.generate_rod_centers(
                self.rods_per_face, self.rod_width_mm, self.rod_gap_mm
            )
        )

        avg_span_margin_mm = np.mean(np.diff(self.grid_tops_mm))
        self.span_margin_mm = span_margin_mm or avg_span_margin_mm / 4

        self.max_bow_mm = max_bow_mm or 0
        self.max_divergence_mm = max_divergence_mm or 0
        self.max_z_displacement_mm = max_z_displacement_mm or 0

        self.swing_max_angle_deg = swing_max_angle_deg
        self.swing_periods_per_inspection = swing_periods_per_inspection
        self.swing_deg_xy = swing_deg_xy

        self.cloudiness = cloudiness if cloudiness is not None else self.rnd.rand()
        self.cam_res_x = cam_res_x or defaults.scene_params.camera.resolution_width
        self.cam_res_y = cam_res_y or defaults.scene_params.camera.resolution_height

    def pendulum_swing_in_time(self, t):
        x = 0
        if self.swing_periods_per_inspection is not None:
            x = 2 * np.pi * t * self.swing_periods_per_inspection

        relative_position = np.sin(x)

        if self.swing_max_angle_deg is None:
            return 0, 0
        else:
            return relative_position * self.swing_max_angle_deg, self.swing_deg_xy

    def rods_noise(self):
        noises = []
        for _ in range(self.n_textures):
            noise = textures.rod_noise(
                rod_height_mm=self.rod_height_mm,
                rod_width_mm=self.rod_width_mm,
                resolution_factor=4,
                rnd=self.rnd,
            )
            noises.append(noise)

        return np.hstack(noises)

    def rod_material(self):
        noise = self.rods_noise()
        # TODO: upper third is affected less than the rest
        blend_texture = textures.blend(noise, 0.2, self.cloudiness)
        return mat.zirconium_blend(blend_texture=blend_texture)

    def grid_material(self):
        noise = textures.grid_noise(resolution_factor=4, rnd=self.rnd)
        blend_texture = textures.blend(noise, 0.2, self.cloudiness)
        material = mat.inconel_blend(blend_texture=blend_texture)

        bumpmap = textures.grid_bumpmap(resolution_factor=4, rnd=self.rnd)

        return {
            "type": "bumpmap",
            "arbitrary": {
                "type": "bitmap",
                "raw": True,
                "data": np.expand_dims(utils.normalize(bumpmap) * 2, axis=2),
            },
            "material": material,
        }

    def bow_divergence(self):
        return simu.random_bow(
            len(self.rod_centers),
            self.max_bow_mm,
            self.max_divergence_mm,
            self.grid_tops_mm,
            self.span_margin_mm,
            rnd=self.rnd,
        )

    def z_displacement(self):
        return self.rnd.rand(len(self.rod_centers)) * self.max_z_displacement_mm

    def serialize_to_dict(
        self,
    ):
        d = self.__dict__.copy()

        to_del = ["rnd"]
        for k in to_del:
            del d[k]

        for k in list(d.keys()):
            v = d[k]
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()

        return d
