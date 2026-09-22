import mitsuba as mi


def create_mount_emmiter(
    location, cam_light_intensity, light_height_mm=10, radius_mm=1
):
    x, y, z = location
    return {
        "type": "cylinder",
        "radius": radius_mm,
        "p0": [x, y, z + light_height_mm / 2],
        "p1": [x, y, z - light_height_mm / 2],
        "emitter": {
            "type": "area",
            "radiance": {
                "type": "rgb",
                "value": cam_light_intensity,
            },
        },
    }


def cam_perspective_lookat(
    origin, 
    target=None, 
    up=None, 
    fov=None, 
    res_x=None, 
    res_y=None,
):
    if target is None:
        target = [0, 0, 0]
    if up is None:
        up = [0, 0, 1]
    if res_x is None:
        res_x = 600
    if res_y is None:
        res_y = 800
    fov = fov or 28

    return {
        "type": "perspective",
        "fov": fov,
        "to_world": mi.ScalarTransform4f().look_at(origin=origin, target=target, up=up),
        "far_clip": 100_000,
        "film": {
            "type": "hdrfilm",
            "pixel_format": "rgba",
            "width": res_x,
            "height": res_y,
            "rfilter": {"type": "box"},
        },
    }


def cam_orto_lookat(
    origin, target=None, res_x=None, res_y=None, orto_scale_xy=None, up=None
):
    if target is None:
        target = [0, 0, 0]
    if up is None:
        up = [0, 0, 1]
    if res_x is None:
        res_x = 600
    if res_y is None:
        res_y = 800
    if orto_scale_xy is None:
        orto_scale_xy = [1, 1]

    return {
        "type": "orthographic",
        "film": {
            "type": "hdrfilm",
            "pixel_format": "rgba",
            "width": res_x,
            "height": res_y,
            "rfilter": {"type": "box"},
        },
        "to_world": mi.ScalarTransform4f().look_at(
            origin=origin, target=target, up=up
        ).scale([*orto_scale_xy, 1]),
    }
