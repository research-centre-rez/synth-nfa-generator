import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
# mi.set_variant("llvm_ad_rgb")

scene = mi.load_dict(
    {
        "type": "scene",
        "integrator": {"type": "path"},
        "sensor": {
            "type": "perspective",
            "film": {
                "type": "hdrfilm",
                "width": 64,
                "height": 64,
            },
        },
        "sphere": {"type": "sphere"},
        "light": {"type": "constant"},
    }
)

image = mi.render(scene, spp=4)
print(image.shape)
