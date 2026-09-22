import pandas as pd
import mitsuba as mi
import synthnf.geometry.curves as curves
import synthnf.simulation as simu
import synthnf.models as models
import synthnf.scene as scene
from synthnf.scene.base import FACE_TO_ROTATION_DEG

import matplotlib.pyplot as plt

import synthnf.report as rep

import numpy as np
import logging

logger = logging.getLogger("synthnf")


class FAInspection:
    def __init__(self, simulation: simu.RandomParameters):
        self.simulation = simulation

        self.grid_tops_mm = self.simulation.grid_tops_mm
        self.fa_height_mm = self.simulation.grid_tops_mm[-1]
        self.rod_centers = self.simulation.rod_centers

        self.curve_fa, self.curves_rod = self.simulation.bow_divergence()

        grid_smpls = self.grid_tops_mm[1:-1] / np.max(self.grid_tops_mm)
        self.grid_x, self.grid_y, self.grid_z = self.curve_fa.evaluate_multi(grid_smpls)

        self.center_of_gravity_height = 15000  # TODO remove constant

        rod_mat = self.simulation.rod_material()

        self.z_displacement = self.simulation.z_displacement()
        self.model_tips = models.rod_tips(
            rod_centers=self.rod_centers,
            material=rod_mat,
            z_displacement=self.z_displacement,
        )
        self.model_butts = models.rod_butts(
            rod_centers=self.rod_centers,
            material=rod_mat,
            z_displacement=self.z_displacement,
        )

        if "to_world" in self.model_tips:
            self.model_tips["to_world"] = self.model_tips["to_world"].translate(
                [0, 0, self.fa_height_mm]
            )
        else:
            self.model_tips["to_world"] = mi.ScalarTransform4f().translate(
                [0, 0, self.fa_height_mm]
            )

        self.model_fa = models.fuel_rods(
            curves_rod=self.curves_rod,
            material=rod_mat,
            rod_height_mm=self.fa_height_mm,
            rod_width_mm=self.simulation.rod_width_mm,
            gap_width_mm=self.simulation.rod_gap_mm,
            n_textures=self.simulation.n_textures,
            rod_centers=self.rod_centers,
            z_displacement=self.z_displacement,
        )

        grid_material = self.simulation.grid_material()
        self.model_grids = add_grids(
            self.grid_x, self.grid_y, self.grid_z, grid_material
        )

    def _prepare_scene_dict(
        self, 
        cam_z, 
        face_num,
        object_masks=False,
        ):
        scene_dict = scene.inspection_dict(
            cam_z=cam_z, 
            face_num=face_num,
            cam_res_x=self.simulation.cam_res_x,
            cam_res_y=self.simulation.cam_res_y,
        )

        scene_dict["tips"] = self.model_tips.copy()
        scene_dict["butts"] = self.model_butts.copy()
        scene_dict["model"] = self.model_fa.copy()

        keys = [f"model_grid_{int(i)}" for i in range(len(self.model_grids))]
        for key, g in zip(keys, self.model_grids):
            scene_dict[key] = g.copy()

        model_keys = ["tips", "butts", "model", *keys]

        if object_masks:
            integrator = scene_dict['integrator']
            aovs = {
                'type': 'aov',
                #'aovs':'xx:shape_index',
                'aovs':'xx:prim_index',
                'my_image': {
                    'type': 'path',
                },
                'integrator': integrator,
            }
            scene_dict['integrator'] = aovs
            scene_dict['sensor']['film']['rfilter']['type'] = 'box'
            
        return scene_dict, model_keys

    def _swing_fa(self, scene_dict, model_keys, swing_deg_xz, swing_deg_xy):
        swing = swing_transformation(
            swing_deg_xz, self.center_of_gravity_height, swing_deg_xy
        )

        for model_key in model_keys:
            model = scene_dict[model_key]
            scene.append_transform(model, swing)

    def render_frame(self, cam_z, face_num=1, spp=128):
        scene_dict, model_keys = self._prepare_scene_dict(cam_z, face_num)
        t = cam_z / self.fa_height_mm
        swing_deg_xz, swing_deg_xy = self.simulation.pendulum_swing_in_time(t)
        self._swing_fa(scene_dict, model_keys, swing_deg_xz, swing_deg_xy)

        return scene.render_scene(scene_dict, spp=spp, alpha=False, denoise=True)

    def render_mask(self,cam_z,face_num=1):
        spp = 1
        scene_dict, model_keys = self._prepare_scene_dict(cam_z, face_num, object_masks = True)
        t = cam_z / self.fa_height_mm
        swing_deg_xz, swing_deg_xy = self.simulation.pendulum_swing_in_time(t)
        self._swing_fa(scene_dict, model_keys, swing_deg_xz, swing_deg_xy)

        
        return scene.render_scene(scene_dict, spp=1, raw=True)
        

    def render_video(
        self, 
        n_frames=None, 
        cam_speed=None, 
        top_down=True,
        face_num=1
    ):
        # XOR
        assert n_frames is not None or cam_speed is not None
        assert not (n_frames is not None and cam_speed is not None)


        n_frames = n_frames or int(self.fa_height_mm / cam_speed)
        cam_zs = np.linspace(0, self.fa_height_mm, n_frames)
        if top_down:
            cam_zs = np.flip(cam_zs)

        for cam_z in cam_zs:
            frame = self.render_frame(cam_z,face_num=face_num)
            yield cam_z, frame

    def render_shrunk(
        self,
        res_y=2048 * 2,
        res_x=512 * 2,
        orto_x=200,
        orto_y=500,
        alpha=True,
        grids=True,
        face_num=1,
    ):
        scene_dict = scene.shrunk_dict(
            self.curve_fa,
            res_y=res_y,
            res_x=res_x,
            orto_y=orto_y,
            orto_x=orto_x,
            face_num=face_num,
        )

        scene_dict["tips"] = self.model_tips
        scene_dict["butts"] = self.model_butts
        scene_dict["model"] = self.model_fa


        if grids:
            for i, g in enumerate(self.model_grids):
                scene_dict[f"model_grid_{int(i)}"] = g

        return scene.render_scene(scene_dict, alpha=alpha, denoise=True)

    def plot_bow(self, bow=True, divergence=True, limit=None, ax=None, face_num=1):
        if ax is None:
            _, ax = plt.subplots(1, 1)

        angle_deg = FACE_TO_ROTATION_DEG[face_num]
        curve = None
        if bow:
            curve = curves.CurveRotatedZAxis(self.curve_fa, angle_deg)
        divs = None
        if divergence:
            divs = [
                curves.CurveRotatedZAxis(d, angle_deg) for d in self.curves_rod[:limit]
            ]

        rep.plot_bow(curve, divs, ax=ax)

        for s in zip(self.grid_tops_mm):
            ax.axhline(s, c="gray", alpha=0.3)

    def sample_shifts(self, n, top_down=True, face_num=1):
        ts = np.linspace(0, 1, n)
        #if top_down:
        #    ts = np.flip(ts)
    
        if face_num not in FACE_TO_ROTATION_DEG:
            raise ValueError(f"Face number {face_num} not supported")

        angle_deg = FACE_TO_ROTATION_DEG[face_num]
        c = curves.CurveRotatedZAxis(self.curve_fa, angle_deg)
        x, y, z = c.evaluate_multi(ts)

        return pd.DataFrame(
            np.vstack([x, y, z]).T, columns=["abs_shift_x_mm", "abs_shift_y_mm", "z_mm"]
        )

    def collect_metadata(self, n_frames, top_down=True, face_num=1):
        shrunk = self.render_shrunk(face_num=face_num)

        fig, ax = plt.subplots(1, 1, figsize=(4, 8))
        self.plot_bow(limit=11, ax=ax, face_num=face_num)

        shifts_df = self.sample_shifts(n_frames, top_down=top_down, face_num=face_num)

        cam_speed = self.fa_height_mm / n_frames

        fa_ply_filepath = self.model_fa["filename"]
        params_json = self.simulation.serialize_to_dict()

        return {
            "shifts": shifts_df,
            "shrunk": shrunk,
            "bow_fig": fig,
            "cam_speed": cam_speed,
            "displacement": list(self.z_displacement),
            "model_filepath": str(fa_ply_filepath),
            "simulation_parameters": params_json,
        }


def add_grids(grid_x, grid_y, grid_z, grid_material):
    grids = []
    for x, y, z in zip(grid_x, grid_y, grid_z):
        grid = models.spacer_grid(material=grid_material)
        grid["to_world"] = mi.ScalarTransform4f().translate([x, y, z])
        grids.append(grid)
    return grids


def swing_transformation(
    swing_deg,
    center_of_gravity_height,
    xy_plane_rotation_deg,
):
    to_center = mi.ScalarTransform4f().translate([0, 0, -center_of_gravity_height])
    rot_xz = mi.ScalarTransform4f().rotate([0, 1, 0], swing_deg)
    from_center = mi.ScalarTransform4f().translate([0, 0, center_of_gravity_height])
    rot_xy = mi.ScalarTransform4f().rotate([0, 1, 0], xy_plane_rotation_deg)

    return rot_xy @ from_center @ rot_xz @ to_center
