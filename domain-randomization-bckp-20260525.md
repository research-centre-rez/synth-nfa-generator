---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: synth-nfa
    language: python
    name: synth-nfa
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

import matplotlib.pyplot as plt

import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
mi.set_log_level(mi.LogLevel.Error)
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
    cam_ring_spec= ss.AhlbergCameraRingSpec.single_cam_four_x_lights(field_of_view=40, light_intensity=200,light_height_offset=0),
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
import synthnf.mitsuba_layer as ml

# uncomment only if you want to see a variation of the original scene
# inspection_scene_varied = inspection_variation.sample(inspection_scene,krng,'inspection_var',*keys)
# inspection_scene = inspection_scene_varied

mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene,krng)
render_res = mitsuba_scene.render(denoise=True,rodgrid_labels=True,oxide_labels=True,spp=64)


post_process_rendering_variation = ml.PostProcessRenderingVariation(
    noise_denoise_blend=sv.MultiplicativeScalarUniformVariationSpecs(.6,.95)
)
post_processed_image = post_process_rendering_variation.sample(render_res,krng,"post_process")

rods_instance_mask, grid_mask = ml.unsafe_read_labels(render_res.label_instances)
plt.imshow(rods_instance_mask)
plt.show()
plt.imshow(grid_mask)
plt.show()
plt.imshow(post_processed_image)
plt.show()
plt.imshow(render_res.label_oxide)
plt.show()
```

# Hooking Into Mitsuba Dictionary

This is a piece of code for adding custom content into the scene:

```python
mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene,krng)
scene_dict = mitsuba_scene.to_scene_dict()


## pick up from here

scene_dict['big_ball'] = {
    'type': 'sphere',
    'to_world': mi.ScalarAffineTransform4f().scale([150, 150, 150]),#.translate([0, 0, 0]),
    'bsdf': {
        'type': 'diffuse'
    }
}

scene = mi.load_dict(scene_dict)
raw = mi.render(scene,spp=32)

plt.imshow(raw[:,:,:3])
plt.title('raw, noisy image')
plt.show()
```

# SGRB Conversion

Don't forget to apply intensity conversion. It won't be that dark.

```python
plt.imshow(ml.srgb_bitmap(raw))
```

# Denoiser

if you deneoise super low spp image, you will get essentialy garbage, add `spp = 32` to avoid this

```python
denoiser = mi.OptixDenoiser(
    input_size=[raw.shape[1], raw.shape[0]], 
    albedo=False, 
    normals=False, 
    temporal=False
)
plt.title('Low quality denoised image')
lowq = mi.render(scene,spp=1)
plt.imshow(ml.srgb_bitmap((denoiser(lowq)))
plt.show()

hiqw = mi.render(mi.load_dict(scene_dict),spp=32)
plt.imshow(ml.srgb_bitmap((denoiser(hiqw)))
plt.title('higher quality denoised image')
plt.show()
```

```python

```

```python

```

```python

```
