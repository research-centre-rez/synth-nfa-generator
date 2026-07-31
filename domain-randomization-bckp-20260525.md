---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: .venv (3.12.3)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import synthnf.assets as assets

import warnings

from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
import drjit as dr
import synthnf.mitsuba_layer as ml
import cv2

import mitsuba as mi
mi.set_variant('cuda_ad_rgb') # , 'llvm_ad_rgb'
mi.set_log_level(mi.LogLevel.Error)
```

```python
import os
import tempfile

# Force Python's tempfile module to use a directory in your home workspace
os.environ["TMPDIR"] = os.path.expanduser("~/tmp")
os.makedirs(os.path.expanduser("~/tmp"), exist_ok=True)
tempfile.tempdir = os.path.expanduser("~/tmp")
```

# Define Scene 

```python
import synthnf.scene_spec as ss

rod_count = 17
inspection_scene = ss.InspectionScene(
    nfa_spec= ss.NfaSpec.from_shape(
        ss.RodsSquareSpec(rod_height=150,column_number=rod_count),
        rods_material_spec=ss.MaterialOxidizedConductorSpec(
            conductor_spec = ss.MaterialNamedAnyConductorSpec(
                conductor_name = "custom_Zircon",
                rough_conductor_spec=ss.MaterialRoughConductorSpec(alpha_u=.025,alpha_v=.05)
            ),
            oxidation_amount = .5,
            # commented out to speed up rendering
            # oxide_spots_spec = ss.TextureOxideSpotsSpec(
            #     oxide_spots_coverage = .0,
            #     poisson_disk_radius=.09,
            #     opacity=.25,
            #     blur_sigma = 0,
            # ),
        ),
        grids_material_spec=ss.MaterialOxidizedConductorSpec(
            conductor_spec = ss.MaterialNamedAnyConductorSpec(
                conductor_name = "custom_Inconel",
                rough_conductor_spec=ss.MaterialRoughConductorSpec(alpha_u=.025,alpha_v=.05)
            ),
            oxidation_amount = 0,
        ),
        grids = [
            ss.SpacerGridSpec(
                z_location=0,
            )
        ]
        
    ),
    cam_ring_spec= ss.AhlbergCameraRingSpec.single_cam_four_x_lights(field_of_view=40, light_intensity=200,light_height_offset=0), # , resolution_x=800, resolution_y=450
    env_map_spec=ss.EnvironmentMapSpec(
        intensity_scale=1.05
    ),

    medium_spec = ss.HeterogenousMediumSpec(
        albedo=ss.MaterialBSDFSpec(r=0.3,g=0.5,b=0.5) ,
        hg_phase_g=0.6,
        scale = .001,
        volume_spec=ss.MediumRandomGridVolumeSpec(
            resolution= 64,
            cube_width=1000,
            heterogenity_noise_max=.1
        ),
    )
)
```

```python
import synthnf.scene_variation as sv

krng = sv.KeyedRNG(123)
keys=[2]

allowed_rod_grid_materials = ['custom_Zircon', 'custom_Inconel', 'a-C', 'Na_palik', 'Ag', 'Nb', 'Rh', 'Se', 'SiC', 'Be', 'Cr', 'Ta', 'CsI', 'Te', 'ThF4', 'Cu2O', 'TiC', 'CuO', 'd-C', 'TiO2', 'Hg', 'VC', 'HgTe', 'V_palik', 'VN', 'K', 'W', 'MgO', 'Mo']

inspection_variation = sv.InspectionVariationParams(
    nfa_variation = sv.NfaVariationParams(
        rods_variations= [
            sv.FuelRodVariationParams(
                radius=sv.MultiplicativeScalarUniformVariationSpecs(low=.95,high=1.05),
                height=sv.MultiplicativeScalarUniformVariationSpecs(low=.9,high=1.1),
                xyz = sv.XyzUniformVariationSpecs(
                    x=sv.AdditiveScalarUniformVariationSpecs(-1.2,1.2),
                    #y=sv.AdditiveScalarUniformVariationSpecs(-1.5,1.5),
                )
            )
        ]*100,
        grids_variations=[
            sv.SpacerGridVariation(
                z_location=sv.AdditiveScalarUniformVariationSpecs(-50,50),
                enabled=sv.ChanceBoolVariation(success_probability=.2),
                tooth_ply_filename=sv.ChoiceVariationParams([
                    "g4face.ply",
                    "g4face-spiky-flat-holed.ply",
                    "g4face-spiky-flat-slits.ply",
                    "g4face-dull-flatish-holed.ply"
                ])
            )
        ],
        rods_material_variation=sv.MaterialOxidizedConductorVariation(
            conductor_variation= sv.MaterialConductorVariation(
                conductor_name=sv.ChoiceVariationParams(allowed_rod_grid_materials),
                rough_conductor_variation=sv.MaterialRoughConductorVariation(
                    alpha_u=sv.MultiplicativeScalarUniformVariationSpecs(.8,1.2),
                    alpha_v=sv.MultiplicativeScalarUniformVariationSpecs(.8,1.2),
                )
            ),
            oxidation_amount=sv.AdditiveScalarUniformVariationSpecs(0,.2),
            oxide_spots_spec= sv.TextureOxideSpotsVariation(
                oxide_spots_coverage=sv.AdditiveScalarUniformVariationSpecs(low = -.25,high=.25 ),
                poisson_disk_radius=sv.AdditiveScalarUniformVariationSpecs(low = -.01,high = .02),
                opacity=sv.AdditiveScalarUniformVariationSpecs(low = -.1,high = .05),
                blur_sigma = sv.AdditiveScalarUniformVariationSpecs(low = 1,high = 4),
            ),
        ),
        grids_material_variation=sv.MaterialOxidizedConductorVariation(
            conductor_variation= sv.MaterialConductorVariation(
                conductor_name=sv.ChoiceVariationParams(allowed_rod_grid_materials),
                rough_conductor_variation=sv.MaterialRoughConductorVariation(
                    alpha_u=sv.MultiplicativeScalarUniformVariationSpecs(.8,1.2),
                    alpha_v=sv.MultiplicativeScalarUniformVariationSpecs(.8,1.2),
                )
            ),
            oxidation_amount=sv.AdditiveScalarUniformVariationSpecs(0,.1)
        )
    ),
    cam_ring_variation = sv.CamRingVariationParams(
        emitter_variations = [
            sv.EmitterVariation(
                intensity=sv.MultiplicativeScalarUniformVariationSpecs(low=.8,high=1.2),
                lookat_target_xyz = sv.XyzUniformVariationSpecs(
                    x = sv.AdditiveScalarUniformVariationSpecs(-15,15)
                ),
                lookat_origin_xyz = sv.XyzUniformVariationSpecs(
                    x = sv.AdditiveScalarUniformVariationSpecs(-15,15),
                    z = sv.AdditiveScalarUniformVariationSpecs(-15,15),
                )
            )
        ]*2,
        sensor_variation =[
            sv.PerspectiveSensorVariation(
                field_of_view=sv.MultiplicativeScalarUniformVariationSpecs(low=.95,high=1.05),
                lookat_target_xyz = sv.XyzUniformVariationSpecs(
                    x = sv.AdditiveScalarUniformVariationSpecs(-20,20)
                ),
                lookat_origin_xyz = sv.XyzUniformVariationSpecs(
                    x = sv.AdditiveScalarUniformVariationSpecs(-15,15),
                    y = sv.AdditiveScalarUniformVariationSpecs(-15,15),
                    z = sv.AdditiveScalarUniformVariationSpecs(-15,15),
                )
            )
        ]
    ),
    env_map_variation=sv.EnvironmentMapVariationParams(
        intensity_scale=sv.MultiplicativeScalarUniformVariationSpecs(.85,1.1),
        env_map_file=sv.ChoiceVariationParams(assets.env_map_names),
        rot_left_right_angle = sv.AdditiveScalarUniformVariationSpecs(0,360)
    ),
    medium_variation = sv.HeterogenousMediumVariation(
        # I could use scalar operations here as long as the base value is np.array
        # albedo=sv.XyzUniformVariationSpecs(
        #     x = sv.MultiplicativeScalarUniformVariationSpecs(0.9,1.2),
        #     y = sv.MultiplicativeScalarUniformVariationSpecs(0.1,1.2),
        #     z = sv.MultiplicativeScalarUniformVariationSpecs(0.1,1.2),
        # ) ,
        albedo = sv.MultiplicativeScalarUniformVariationSpecs(.1,1.2),
        hg_phase_g=sv.MultiplicativeScalarUniformVariationSpecs(.1,1),
        scale =sv.MultiplicativeScalarUniformVariationSpecs(1,10),
        enabled = sv.ChanceBoolVariation(success_probability=.60)
    )
)
```

# Example rendering

This is the example how to render a NFA scene with the least amount of interaction

```python

# uncomment only if you want to see a variation of the original scene
# inspection_scene_varied = inspection_variation.sample(inspection_scene,krng,'inspection_var',*keys)
# inspection_scene = inspection_scene_varied

# mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene,krng)
# render_res = mitsuba_scene.render(denoise=True,rodgrid_labels=True,oxide_labels=True,spp=64)


# post_process_rendering_variation = ml.PostProcessRenderingVariation(
#     noise_denoise_blend=sv.MultiplicativeScalarUniformVariationSpecs(.6,.95)
# )
# post_processed_image = post_process_rendering_variation.sample(render_res,krng,"post_process")

# rods_instance_mask, grid_mask = ml.unsafe_read_labels(render_res.label_instances)
# plt.imshow(rods_instance_mask)
# plt.show()
# plt.imshow(grid_mask)
# plt.show()
# plt.imshow(post_processed_image)
# plt.show()
# plt.imshow(render_res.label_oxide)
# plt.show()
```

# Denoiser

```python
denoiser = mi.OptixDenoiser(
    input_size=[1600, 900], 
    albedo=False, 
    normals=False, 
    temporal=False
)
```

# Hooking Into Mitsuba Dictionary

This is a piece of code for adding custom content into the scene:
## 1. Manual addition of debris
### 1. Bolt

```python
# helper functions
def plot_results(raw_scene, albedo_mask):
    img_albedo = ml.srgb_bitmap(albedo_mask)
    img_scene = ml.srgb_bitmap(denoiser(raw_scene[:, :, :3]))

    alpha_channel = dr.ones(mi.TensorXf, (albedo_mask.shape[0], albedo_mask.shape[1], 1))
    raw_mask = dr.concat([albedo_mask, alpha_channel], axis=2)
    multiplier = dr.scalar.TensorXf([1.0, 0.3, 1.0, 1.0]) 
    pink_mask = raw_mask * multiplier
    img_pink = ml.srgb_bitmap(pink_mask[:, :, :3])

    fig, axes = plt.subplot_mosaic(
        [["top_left", "top_right"],
        ["bottom",   "bottom"]],
        figsize=(12, 10),
        layout='constrained'
    )

    axes["top_left"].imshow(img_albedo)
    axes["top_left"].set_title('Debris mask', fontsize=12, fontweight='bold')
    axes["top_left"].axis('off')

    axes["top_right"].imshow(img_scene)
    axes["top_right"].set_title('Scene with debris', fontsize=12, fontweight='bold')
    axes["top_right"].axis('off')

    axes["bottom"].imshow(img_scene)
    axes["bottom"].imshow(img_pink, alpha=0.5)
    axes["bottom"].set_title('Masked debris', fontsize=14, fontweight='bold')
    axes["bottom"].axis('off')
    plt.show()

def store_image_and_mask(raw_scene, albedo_mask, image_path, mask_path):
    img_srgb = ml.srgb_bitmap(raw_scene[:, :, :3])
    img_np = np.clip(np.array(img_srgb), 0.0, 1.0)
    img_uint8 = (img_np * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, img_bgr)

    mask_np = albedo_mask.numpy().astype("uint8") # * 255
    print(max(mask_np.flatten()), min(mask_np.flatten()))
    cv2.imwrite(mask_path, mask_np)
```

```python
# add debris to scene manually
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene,krng)
scene_dict = mitsuba_scene.to_scene_dict()

# add custom bolt
scene_dict['bolt'] = {
    'type': 'ply',
    'filename': 'assets/bolt.ply',
    'to_world': mi.ScalarAffineTransform4f().translate([-32, -103.5, 23]).rotate([0, 1, 0], angle=80).scale([0.8, 0.8, 0.8]), # .translate([-20, -100, -20]).scale([4, 4, 4])
    'bsdf': {
        'type': 'conductor',
        'material': 'Cr'
    }
}

scene = mi.load_dict(scene_dict)
raw_scene = mi.render(scene,spp=32)
```

```python
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene, krng)
scene_dict_labels = mitsuba_scene.to_scene_dict()

# add debris objects with unique albedo materials for labeling
scene_dict_labels['bolt'] = {
    'type': 'ply',
    'filename': 'assets/bolt.ply',
    'to_world': mi.ScalarAffineTransform4f().translate([-32, -104.5, 23]).rotate([0, 1, 0], angle=80).scale([0.8, 0.8, 0.8]),
    'bsdf': {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}
    }
}

# filter out rods and grid
scene_dict_labels = {
    k: v for k, v in scene_dict_labels.items() 
    if not (isinstance(k, str) and (k.startswith('rod') or k.startswith('grid')))
}

# attach aov integrator requesting the `albedo` pass
scene_dict_labels['integrator'] = {
    'type': 'aov',
    'aovs': 'al:albedo',
    'beauty': {
        'type': 'path'
    }
}

scene_labels = mi.load_dict(scene_dict_labels)
image = mi.render(scene_labels, spp=1)

# extract the Albedo channels (channels 4..6 after RGBA beauty pass)
albedo_mask = image[:, :, 4:7]

# plot and store results
plot_results(raw_scene, albedo_mask)
store_image_and_mask(raw_scene, albedo_mask, "data/raw/images/sample_0.png", "data/raw/masks/sample_0.png")

```

### 2. Curved wire

```python
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene,krng)
scene_dict = mitsuba_scene.to_scene_dict()

# curved metal - not labellable yet
scene_dict['curved_wire'] = {
    'type': 'bsplinecurve',
    'to_world': mi.ScalarAffineTransform4f().translate([0, -120, 5]).rotate([1,0,0], angle=90).scale([20,20,20]),
    'filename': 'assets/curves.txt',
    'bsdf': {
        'type': 'conductor',
        'material': 'Cr'
    }
}

scene = mi.load_dict(scene_dict)
raw_scene = mi.render(scene,spp=32)

```

```python
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene, krng)
scene_dict_labels = mitsuba_scene.to_scene_dict()

# add debris objects with unique albedo materials for labeling
scene_dict_labels['curved_wire'] = {
    'type': 'bsplinecurve',
    'to_world': mi.ScalarAffineTransform4f().translate([0, -120, 5]).rotate([1,0,0], angle=90).scale([20,20,20]),
    'filename': 'assets/curves.txt',
    'bsdf': {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]} 
    }
}

# filter out rods and grid
scene_dict_labels = {
    k: v for k, v in scene_dict_labels.items() 
    if not (isinstance(k, str) and (k.startswith('rod') or k.startswith('grid')))
}

# attach aov integrator requesting the `albedo` pass
scene_dict_labels['integrator'] = {
    'type': 'aov',
    'aovs': 'al:albedo',
    'beauty': {
        'type': 'path'
    }
}

scene_labels = mi.load_dict(scene_dict_labels)
image = mi.render(scene_labels, spp=1)

# extract the Albedo channels (channels 4..6 after RGBA beauty pass)
albedo_mask = image[:, :, 4:7]

# plot and store results
plot_results(raw_scene, albedo_mask)
store_image_and_mask(raw_scene, albedo_mask, "data/raw/images/sample_1.png", "data/raw/masks/sample_1.png")
```

### 3. Wire

```python
# add debris to scene manually
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene,krng)
scene_dict = mitsuba_scene.to_scene_dict()

# metal cylinder wire
scene_dict['wire'] = {
    'type': 'cylinder',
    'radius': 0.05,
    'to_world': mi.ScalarAffineTransform4f().translate([17, -104.5, 23]).rotate([0,1,0], angle=90).scale([15,15,15]),
    'material': {
        'type': 'conductor',
        'material': 'Cr'
    }
}

scene = mi.load_dict(scene_dict)
raw_scene = mi.render(scene,spp=32)
```

```python
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene, krng)
scene_dict_labels = mitsuba_scene.to_scene_dict()

scene_dict_labels['wire'] = {
    'type': 'cylinder',
    'radius': 0.05,
    'to_world': mi.ScalarAffineTransform4f().translate([17, -110, 23]).rotate([0,1,0], angle=90).scale([15,15,15]),
    'bsdf': {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [1.0, 1.0, 1.0]}  
    }
}

# filter out rods and grid
scene_dict_labels = {
    k: v for k, v in scene_dict_labels.items() 
    if not (isinstance(k, str) and (k.startswith('rod') or k.startswith('grid')))
}

# attach aov integrator requesting the `albedo` pass
scene_dict_labels['integrator'] = {
    'type': 'aov',
    'aovs': 'al:albedo',
    'beauty': {
        'type': 'path'
    }
}

scene_labels = mi.load_dict(scene_dict_labels)
image = mi.render(scene_labels, spp=1)

# extract the Albedo channels (channels 4..6 after RGBA beauty pass)
albedo_mask = image[:, :, 4:7]

# plot and store results
plot_results(raw_scene, albedo_mask)
store_image_and_mask(raw_scene, albedo_mask, "data/raw/images/sample_2.png", "data/raw/masks/sample_2.png")

```

### 2. Manual addition of fretting/rod scratches


```python
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene, krng)
scene_dict_rod = mitsuba_scene.to_scene_dict()

# filter scene - uncomment for normal map testing
# allowed_keys = {
#     'shared_rod_material', 'shared_grid_material', 'main_camera',
#     'emitter_00000', 'emitter_00001', 'emitter_00002', 'emitter_00003',
#     'medium', 'medium_boundaries', 'env_map', 'type', 'integrator', 'rod_00000'
# }

# scene_dict_rod = {k: v for k, v in scene_dict_rod.items() if k in allowed_keys}

# extract the original rod material and remove it
rod_material = scene_dict_rod['rod_00000']['material']['rod_material']
scene_dict_rod['rod_00000'].pop('material', None)  

# adjust normal map placement
uv_transform = mi.ScalarAffineTransform4f().translate([0, -0.4, 0])

# add normal map from file
normal_map = {
    'type': 'normalmap',
    'normalmap': {
        'type': 'bitmap',
        'raw': True,
        'filename': 'assets/normal_map.jpg',
        'to_uv': uv_transform
    },
    # original material
    'bsdf': rod_material
}

scene_dict_rod['rod_00000']['blend_bsdf'] = {
    'type': 'blendbsdf',
    'weight': {
        'type': 'bitmap',
        'filename': 'assets/bitmap_mask.jpg',
        'to_uv': uv_transform
    },
    'bsdf_0': normal_map,
    'bsdf_1': {
        'type': 'conductor',
        'material': 'Cr'
    }
}

# load scenes
scene_rod = mi.load_dict(scene_dict_rod)

# # region of interest definition
# roi_x = 10       # left right corner of ROI (px)
# roi_y = 800       
# roi_width = 700   
# roi_height = 700  

# # scene adjustment -crop to roi
# tt_r = mi.traverse(scene_rod)
# tt_r['main_camera.film.crop_size'] = (roi_width, roi_height)
# tt_r['main_camera.film.crop_offset'] = (roi_x, roi_y)
# tt_r['env_map.scale'] = 0
# tt_r.update()

# render the original and modified scenes
raw_mod = mi.render(scene_rod,spp=32)
```

```python
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene, krng)
scene_dict_rod = mitsuba_scene.to_scene_dict()

# # filter scene
# allowed_keys = {
#     'shared_rod_material', 'shared_grid_material', 'main_camera',
#     'emitter_00000', 'emitter_00001', 'emitter_00002', 'emitter_00003',
#     'medium', 'medium_boundaries', 'env_map', 'type', 'integrator', 'rod_00000'
# }
# scene_dict_rod = {k: v for k, v in scene_dict_rod.items() if k in allowed_keys}

# remove old material and bsdf from the rod
scene_dict_rod['rod_00000'].pop('material', None)
scene_dict_rod['rod_00000'].pop('bsdf', None)

# attach a matte-black base BSDF so the shape is valid but reflects nothing
scene_dict_rod['rod_00000']['bsdf'] = {
    'type': 'diffuse',
    'reflectance': {
        'type': 'rgb',
        'value': [0.0, 0.0, 0.0]
    }
}

# attach emitter bitmapped by label mask
scene_dict_rod['rod_00000']['emitter'] = {
    'type': 'area',
    'radiance': {
        'type': 'bitmap',
        'filename': 'assets/bitmap_mask.jpg',
        'to_uv': uv_transform
    }
}

scene_rod = mi.load_dict(scene_dict_rod)

# crop to roi
# roi_x = 10       
# roi_y = 800       
# roi_width = 700   
# roi_height = 700  

tt_r = mi.traverse(scene_rod)

# tt_r['main_camera.film.crop_size'] = (roi_width, roi_height)
# tt_r['main_camera.film.crop_offset'] = (roi_x, roi_y)
# turn off all other emitters in scene

tt_r['env_map.scale'] = 0
for e in mitsuba_scene.emitters:
    tt_r[f"{e.element_key}.emitter.radiance.value"] = mi.Color3f(0,0,0)
tt_r.update()

# render the label mask
raw_label = mi.render(scene_rod, spp=1)

plot_results(raw_mod, raw_label[:, :, :3])
store_image_and_mask(raw_mod, raw_label[:, :,:3], "data/raw/images/sample_3.png", "data/raw/masks/sample_3.png")
```

## 3. Generate varied images without debris

```python
output_dir = "./data/raw/"

# define post-processing configuration outside the loop
post_process_rendering_variation = ml.PostProcessRenderingVariation(
    noise_denoise_blend=sv.MultiplicativeScalarUniformVariationSpecs(0.6, 0.95)
)

num_variations = 7

for i in range(num_variations):
    # 1. Update RNG key per iteration to get unique variations
    # Assumes key generation/splitting logic based on your library (e.g., JAX/custom PRNG)
    iteration_key = krng.split() if hasattr(krng, "split") else krng

    # 2. Sample variation of the original scene
    inspection_scene_varied = inspection_variation.sample(
        inspection_scene, iteration_key, f"inspection_var_{i}", *keys
    )

    # 3. Convert to Mitsuba scene & render
    mitsuba_scene = ml.MitsubaScene.from_inspection_scene(
        inspection_scene_varied, iteration_key
    )
    render_res = mitsuba_scene.render(
        denoise=True, rodgrid_labels=True, oxide_labels=True, spp=64
    )
    
    post_processed_image = post_process_rendering_variation.sample(
        render_res, iteration_key, f"post_process_{i}"
    )

    # 5. Extract dimensions to generate a matching black image
    height, width = post_processed_image.shape[:2]
    black_image = np.zeros((height, width), dtype=np.uint8)

    # 6. Save images directly to PNG format
    render_path = os.path.join(output_dir, f"images/sample_{i+5}.png")
    black_path = os.path.join(output_dir, f"masks/sample_{i+5}.png")

    plt.imsave(render_path, post_processed_image)
    plt.imsave(black_path, black_image, cmap="gray")

    print(f"Iteration {i+1}/{num_variations} saved:\n - {render_path}\n - {black_path}")
```

```python
black_image = np.zeros((900, 1600), dtype=np.uint8)
plt.imsave('./data/raw/masks/black_image.png', black_image, cmap='gray')
```

## 4. Adding debris based on camera position

```python
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene, krng)
scene_dict = mitsuba_scene.to_scene_dict()

# locate the sensor (camera) key inside the generated dict
sensor_key = next(
    (
        k for k, v in scene_dict.items() 
        if isinstance(v, dict) and v.get('type') in ['perspective', 'thinlens', 'orthographic']
    ), 
    None
)
if sensor_key is None:
    raise ValueError("Could not find a valid sensor in the scene dict to calculate placement.")

sensor_config = scene_dict[sensor_key]

# extract the camera's to_world transformation matrix
cam_to_world_raw = sensor_config.get('to_world', mi.ScalarAffineTransform4f())
cam_to_world = mi.ScalarAffineTransform4f(cam_to_world_raw)
print(cam_to_world)

# extract camera clipping bounds to avoid placing the object too close/far
near_clip = sensor_config.get('near_clip', 1.0) # default fallback
far_clip = sensor_config.get('far_clip', 10000.0)

# distance from camera
placement_distance = 330.0 

if placement_distance <= near_clip or placement_distance >= far_clip:
    # clamp distance dynamically to guarantee visibility
    placement_distance = max(near_clip * 2.0, min(placement_distance, far_clip * 0.5))


local_target_point = mi.ScalarPoint3f(0.0, 0.0, placement_distance)

world_target_point = cam_to_world @ local_target_point
print(world_target_point)

bolt_transform = mi.ScalarAffineTransform4f() \
    .translate(world_target_point) \
    .scale([1, 1, 1])  

scene_dict['my_bolt'] = {
    'type': 'ply',
    'filename': 'assets/bolt.ply',
    'to_world': bolt_transform,
    'bsdf': {
        'type': 'conductor',
        'material': 'Ag'
    }
}

scene = mi.load_dict(scene_dict)
raw = mi.render(scene,spp=32)

# plt.imshow(ml.srgb_bitmap(raw[:,:,:3]))
# plt.title('raw, noisy image')
# plt.show()
```
