import numpy as np
import mitsuba as mi
import numpy.typing as npt
import synthnf.scene as scene
from synthnf.inspection import FAInspection


class SwingSupressingInspection(FAInspection):
    def __init__(self, params, swing_camera_relative_mm):
        super(self.__class__, self).__init__(params)
        self.swing_camera_relative_mm = swing_camera_relative_mm

    def render_frame(self, cam_z, face_num=1, spp=128):
        scene_dict, model_keys = self._prepare_scene_dict(cam_z, face_num)
        t = cam_z / self.fa_height_mm
        swing_deg_xz, swing_deg_xy = self.simulation.pendulum_swing_in_time(t)
        self._swing_fa(scene_dict, model_keys, swing_deg_xz, swing_deg_xy)

        frame_bottom = scene.render_scene(
            scene_dict, spp=spp, alpha=False, denoise=True
        )

        scene_dict, model_keys = self._prepare_scene_dict(
            cam_z + self.swing_camera_relative_mm, face_num
        )
        # same swing but does not depend on camera
        self._swing_fa(scene_dict, model_keys, swing_deg_xz, swing_deg_xy)
        frame_top = scene.render_scene(scene_dict, spp=spp, alpha=False, denoise=True)
        return FramePackage(frame_bottom, swing=frame_top)

    def collect_metadata(self, n_frames, top_down=True, face_num=1):
        meta = super(self.__class__, self).collect_metadata(
            n_frames,
            top_down=top_down,
            face_num=face_num,
        )

        ts = np.linspace(0, 1, n_frames)

        swing_deg_xz, swing_deg_xy = self.simulation.pendulum_swing_in_time(ts)
        meta["swing_xz"] = swing_deg_xz
        meta["swing_xy"] = swing_deg_xy
        return meta


class SwingSupressionVerticalCamInspection(FAInspection):
    def __init__(
        self,
        params,
        swing_cam_height_mm,
        swing_cam_dist_mm,
        fov=None,
        cam_roll_deg=0,
    ):
        super(self.__class__, self).__init__(params)
        self.params = params
        self.swing_cam_height_mm = swing_cam_height_mm
        self.swing_cam_dist_mm = swing_cam_dist_mm
        self.fov = fov
        self.cam_roll_deg = cam_roll_deg

    def render_frame(self, cam_z, face_num=1, spp=128):
        scene_dict, model_keys = self._prepare_scene_dict(cam_z, face_num)

        # this is the roll of the camera because the original camera 
        # looks down
        roll_axis = [0, 0, 1]
        roll_transform = mi.ScalarTransform4f().rotate(roll_axis, self.cam_roll_deg)
        sensor = scene_dict["sensor"]
        sensor["to_world"] =  sensor["to_world"] @ roll_transform

        t = cam_z / self.fa_height_mm
        swing_deg_xz, swing_deg_xy = self.simulation.pendulum_swing_in_time(t)
        self._swing_fa(scene_dict, model_keys, swing_deg_xz, swing_deg_xy)

        frame_bottom = scene.render_scene(
            scene_dict, spp=spp, alpha=False, denoise=True
        )

        # I should create the scene again
        scene_dict, model_keys = self._prepare_scene_dict(
            self.params.rod_height_mm, face_num
        )

        self._swing_fa(scene_dict, model_keys, swing_deg_xz, swing_deg_xy)
        for obj_key in ["sensor", "light_1", "light"]:
            del scene_dict[obj_key]

        scene_dict["sensor"] = scene.cam_perspective_lookat(
            [self.swing_cam_dist_mm, 0, cam_z + self.swing_cam_height_mm],
            target=[0, 0, 0],
            up=[0.5, 1, 0],  # this cam is vertical
            res_x=1920,
            res_y=1080,
            fov=self.fov,
        )

        scene_dict["light"] = {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": 0.2,
            },
        }

        frame_top = scene.render_scene(scene_dict, spp=spp, alpha=True, denoise=True)

        return FramePackage(frame_bottom, swing=frame_top)

    def collect_metadata(self, n_frames, top_down=True, face_num=1):
        meta = super(self.__class__, self).collect_metadata(
            n_frames,
            top_down=top_down,
            face_num=face_num,
        )
        ts = np.linspace(0, 1, n_frames)

        swing_deg_xz, swing_deg_xy = self.simulation.pendulum_swing_in_time(ts)
        meta["swing_xz"] = swing_deg_xz
        meta["swing_xy"] = swing_deg_xy
        return meta


class FramePackage(np.ndarray):
    def __new__(
        cls,
        main_frame: npt.NDArray,
        **kwargs,
    ) -> "FramePackage":
        return super().__new__(
            cls,
            main_frame.shape,
            dtype=main_frame.dtype,
            buffer=main_frame.flatten(),
        )

    def __init__(self, main_frame, swing=None):
        self.main_frame = main_frame
        self.swing = swing

    def to_dict(self):
        return {
            "": self.main_frame,
            "swing": self.swing,
        }
