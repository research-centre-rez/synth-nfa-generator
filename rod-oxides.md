---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
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
import synthnf.oxides as ox
```

```python
import synthnf.assets as assets
import matplotlib.pyplot as plt
import warnings

from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from functools import wraps
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
mi.set_log_level(mi.LogLevel.Error)
```

# Define Scene 

```python
import matplotlib.pyplot as plt
import imageio
```

```python
ox_poisson = ox.poisson_disk(seed = 123,radius = 0.04)
plt.scatter(ox_poisson.T[0],ox_poisson.T[1])
```

```python
import numpy as np
import matplotlib.pyplot as plt

def perlin_noise_2d(
    height,
    width,
    seed,
    scale = 0.02,
    noise_frequency = .02,
    noise_octaves = 5,
    noise_lacunarity = 2,
    noise_gain = .5
):
    noise = ox.perlin_noise_gen(
        seed,
        scale = scale,
        noise_frequency = noise_frequency,
        noise_octaves = noise_octaves,
        noise_lacunarity = noise_lacunarity,
        noise_gain = noise_gain,
    )
    
    y, x = np.mgrid[:height, :width]

    coords = np.array(
        [
            x.ravel() * scale,
            y.ravel() * scale,
        ],
        dtype=np.float32,
    )
    
    return noise.gen_from_coords(coords).reshape(height, width)
    
image = perlin_noise_2d(
    756,
    256,
    123,
)

_,ax = plt.subplots(1,1)
ax.imshow(image, cmap="gray")
#axs[1].imshow(low_octave_image, cmap="gray")
plt.show()
```

```python
import synthnf.scene_variation as sv

krng = sv.KeyedRNG(123)
keys = []
```

```python
rod_height = 150
rod_radius = 4.75
res_x = 256
rod_center_x = 10
rod_center_y = 57


xy_ratio =np.ceil(rod_height/(2*np.pi*rod_radius ))
res_y = res_x*int(xy_ratio) 


# this should be somehow translated to user 
# something like spot_size_inversed maybe

poisson_seed = krng.uint32('poisson_seed',*keys)
noise_seed = krng.uint32('noise_seed',*keys)

spot_texture_generator=ox.OxideSpotTextureGenerator(
    cylinder_noise= ox.CylinderNoise(
        ox.perlin_noise_gen(noise_seed)
    ),
)

rn = 10 # rod_number for plotting
_,axs  =plt.subplots(1,rn,figsize = (20,10))
for i,ax in enumerate(axs):
    j = i
    i = 0
    rcx = rod_center_x + (rod_radius*i)
    rcy = rod_center_y + (rod_radius*i)
    
    texture_result = spot_texture_generator.generate(
        rcx,
        rcy,
        rod_radius,
        rod_height,
        krng,
        f'rod_{i}',
        *keys,
        noise_texture_zoom=10,
        min_oxide_size_px = 16,
        max_oxide_size_px = 22,
        oxide_spots_coverage_threshold =  (j/10 *2)-1,
        poisson_disk_radius = .04,
    )
    spts = texture_result.spot_centers_all[texture_result.points_filter]
    
    ax.imshow(texture_result.oxide_mask)
    ax.scatter(spts.T[0]*res_x,spts.T[1]*res_y,alpha =.3,marker='x',color = 'red')
    ax.set_ylim(0,res_y)
    ax.set_xlim(0,res_x)
plt.show()

```

# RENDERING

```python
import synthnf.scene_spec as ss
import synthnf.scene_variation as sv
import synthnf.mitsuba_layer as ml

krng = sv.KeyedRNG(123)

rod_height = 150
rod_radius = 4.75
rod_count = 17

inspection_scene = ss.InspectionScene(
    nfa_spec= ss.NfaSpec.from_shape(
        ss.RodsSquareSpec(rod_height=rod_height,column_number=rod_count,rod_radius=rod_radius),
        rods_material_spec=ss.MaterialOxidizedConductorSpec(
            conductor_spec = ss.MaterialNamedAnyConductorSpec(
                conductor_name = "custom_Zircon",
                rough_conductor_spec=ss.MaterialRoughConductorSpec(alpha_u=.025,alpha_v=.05)
            ),
            oxidation_amount = .5,
            oxide_spots_specs = [ ss.TextureOxideSpotsSpec(oxide_spots_coverage = 0,poisson_disk_radius=.05 ) for i in range(rod_count)]
        ),
        grids_material_spec=ss.MaterialOxidizedConductorSpec(
            conductor_spec = ss.MaterialNamedAnyConductorSpec(
                conductor_name = "custom_Inconel",
                rough_conductor_spec=ss.MaterialRoughConductorSpec(alpha_u=.025,alpha_v=.05)
            ),
            oxidation_amount = .5,
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

mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene,krng,['test'])
```

```python
render_res = mitsuba_scene.render(spp=32,rodgrid_labels=True,oxide_labels = True)

rod_labels,grid_mask = ml.unsafe_read_labels(render_res.label_instances)
_,axs = plt.subplots(1,4,figsize = (20,10))
axs[0].imshow(render_res.raw_rgba)
axs[1].imshow(render_res.label_oxide)
axs[2].imshow(rod_labels)
axs[3].imshow(grid_mask)
plt.show()
render_res.label_scene
```
