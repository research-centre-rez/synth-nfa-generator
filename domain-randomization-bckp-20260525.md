---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.3
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
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
mi.set_log_level(mi.LogLevel.Error)
```

```python
import synthnf.assets as assets

import warnings

from dataclasses import dataclass
import numpy.typing as npt
import numpy as np

import matplotlib.pyplot as plt
```

# Define Scene 

```python
import synthnf.scene_spec as ss


rod_inspection_scene = ss.InspectionScene(
    nfa_spec= ss.NfaSpec.single_rod(
        4.75,
        150 ,
        rods_material_spec=ss.MaterialOxidizedConductor(
            conductor_spec= ss.MaterialNamedAnyConductorSpec(
                conductor_name = "Au",
                rough_conductor_spec=ss.RoughConductorSpec(alpha_u=.1,alpha_v=1)
            ),
            oxidation_amount=0
        )
    ),
    cam_ring_spec = ss.AhlbergCameraRingSpec.single_cam_four_x_lights(field_of_view=10, light_intensity=1000,light_height_offset=5),
    env_map_spec=ss.EnvironmentMapSpec(
        env_map_file="machine_shop_02_4k.hdr",
        intensity_scale=.3
    )
)

inspection_scene = ss.InspectionScene(
    nfa_spec= ss.NfaSpec.from_shape(
        ss.RodsSquareSpec(rod_height=150),
        rods_material_spec=ss.MaterialOxidizedConductor(
            conductor_spec = ss.MaterialNamedAnyConductorSpec(
                conductor_name = "custom_Zircon",
                rough_conductor_spec=ss.RoughConductorSpec(alpha_u=.025,alpha_v=.05)
            ),
            oxidation_amount = 0
        )
    ),
    #cam_ring_spec = GenericCameraRingSpec.single_cam_two_panel_light(field_of_view=35,light_intensity = 4000,light_offset=400),
    cam_ring_spec= ss.AhlbergCameraRingSpec.single_cam_four_x_lights(field_of_view=40, light_intensity=200,light_height_offset=0),
    env_map_spec=ss.EnvironmentMapSpec(
        #env_map_file='20060121-06_hd.hdr',
        intensity_scale=.8
    ),
    #global_illumination= GlobalIlluminationSpec(intensity=1)
)
```

```python
import synthnf.scene_variation as sv

krng = sv.KeyedRNG(123)
keys=[2]

inspection_variation = sv.InspectionVariationParams(
    nfa_variation = sv.NfaVariationParams(
        rods_variations= [
            sv.FuelRodVariationParams(
                radius=sv.MultiplicativeScalarUniformVariationSpecs(low=.95,high=1.05),
                height=sv.MultiplicativeScalarUniformVariationSpecs(low=.9,high=1.1),
                xyz = sv.AdditiveXyzUniformVariationSpecs(
                    x=sv.AdditiveScalarUniformVariationSpecs(-1,1),
                    y=sv.AdditiveScalarUniformVariationSpecs(-1,1),
                )
            )
        ]*len(inspection_scene.nfa_spec.rods_specs),
        rods_material_variation=sv.MaterialOxidizedConductorVariation(
            conductor_variation= sv.MaterialConductorVariation(
                conductor_name=sv.ChoiceVariationParams(assets.mitsuba_materials)
            ),
            oxidation_amount=sv.MultiplicativeScalarUniformVariationSpecs(.8,1)
        )
    ),
    cam_ring_variation = sv.CamRingVariationParams(
        emitter_variations = [
            sv.EmitterVariation(
                intensity=sv.MultiplicativeScalarUniformVariationSpecs(low=.8,high=1.2),
                lookat_target_xyz = sv.AdditiveXyzUniformVariationSpecs(
                    x = sv.AdditiveScalarUniformVariationSpecs(-5,5)
                )
            )
        ]*2
    ),
    env_map_variation=sv.EnvironmentMapVariationParams(
        intensity_scale=sv.MultiplicativeScalarUniformVariationSpecs(.85,1),
        env_map_file=sv.ChoiceVariationParams(assets.env_map_names),
        rot_left_right_angle = sv.AdditiveScalarUniformVariationSpecs(0,360)
    ),
)

inspection_scene_varied = inspection_variation.sample(inspection_scene,krng,'inspection_var',*keys)
```

```python
import synthnf.mitsuba_layer as ml

@dataclass
class PostProcessRenderingVariation:
    noise_denoise_blend:sv.MultiplicativeScalarUniformVariationSpecs = sv.MultiplicativeScalarUniformVariationSpecs()

    def sample(self, render_result:ml.RenderResult,krng,*keys):
        alpha = self.noise_denoise_blend.sample(1,krng,*keys)

        return alpha*render_result.denoised_rgba + (1-alpha)*render_result.raw_rgba
        
post_process_rendering_variation = PostProcessRenderingVariation(
    noise_denoise_blend=sv.MultiplicativeScalarUniformVariationSpecs(.6,.95)
)
```

```python
inspection_scene_varied = inspection_variation.sample(inspection_scene,krng,'inspection_var',*keys)
mitsuba_scene_varied = ml.MitsubaScene.from_inspection_scene(inspection_scene_varied)
render_res_varied = mitsuba_scene_varied.render(denoise=True,include_labels=True,spp=64)
post_processed_image = post_process_rendering_variation.sample(render_res_varied,krng,"post_process")


# plt.imshow(render_res_varied.label_instances)
# plt.show()
plt.imshow(post_processed_image)
```
