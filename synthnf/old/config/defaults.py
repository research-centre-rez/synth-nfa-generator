from types import SimpleNamespace


class NestedNamespace(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.__setattr__(key, NestedNamespace(value))
            else:
                self.__setattr__(key, value)


materials = NestedNamespace(
    {
        "inconel": {
            "alpha_u": 0.1,
            "alpha_v": 0.4,
        },
        "zirconium": {
            "alpha_u": 0.03,
            "alpha_v": 0.05,
        },
    }
)

blueprints = NestedNamespace(
    {
        "camera_ring": {
            "diameter_mm": 860,
            "light_height_mm": 500,
            "light_diameter_mm": 10,
            "light_offset_mm": 228,
        },
        "fuel_rod": {"width_mm": 9.1, "gap_mm": 3.65, "height_mm": 3800},
        "fa": {
            "rods_per_face": 11,
            "grid_tops_mm": [
                0,
                200,
                455,
                795,
                1135,
                1475,
                1815,
                2155,
                2495,
                2835,
                3175,
                3515,
                3755,
                3800,
            ],
            "grid_heights_mm": [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 0],
        },
    }
)

scene_params = NestedNamespace(
    {
        "illumination": {"cam_light_intensity": 50},
        "textures_per_fuel_rods": 20,
        "camera": {
            "resolution_width": 750,
            "resolution_height": 600,
        },
    }
)
