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
import matplotlib.pyplot as plt
import warnings

from dataclasses import dataclass
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from functools import wraps

import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
mi.set_log_level(mi.LogLevel.Error)
```

```python
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def make_potato_mask(
    height: int = 512,
    width: int = 512,
    seed: int = 42,
    center: tuple[float, float] | None = None,
    base_radius: float = 150.0,
    stretch_x: float = 1.25,
    stretch_y: float = 0.95,
    irregularity: float = 0.18,
    harmonics: int = 5,
) -> np.ndarray:
    """
    Create a binary mask containing a smooth potato-shaped object.

    Returns
    -------
    mask:
        Boolean array of shape (height, width).
    """

    rng = np.random.default_rng(seed)

    if center is None:
        cx = width / 2
        cy = height / 2
    else:
        cx, cy = center

    # Random low-frequency Fourier components define the outline.
    amplitudes = rng.uniform(-1.0, 1.0, size=harmonics)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=harmonics)

    y, x = np.mgrid[:height, :width]

    # Elliptical coordinates.
    dx = (x - cx) / stretch_x
    dy = (y - cy) / stretch_y

    angle = np.arctan2(dy, dx)
    distance = np.sqrt(dx**2 + dy**2)

    radius = np.ones_like(angle) * base_radius

    for harmonic_idx in range(1, harmonics + 1):
        radius += (
            base_radius
            * irregularity
            * amplitudes[harmonic_idx - 1]
            / harmonic_idx
            * np.cos(harmonic_idx * angle + phases[harmonic_idx - 1])
        )

    mask = distance <= radius
    return mask



mask = make_potato_mask(
    height=512,
    width=512,
    seed=7,
    base_radius=150,
    stretch_x=1,
    stretch_y=0.95,
    irregularity=0.22,
    harmonics=4,
)

plt.imshow(mask, cmap="gray")
plt.show()
```

# Define Scene 

```python
import matplotlib.pyplot as plt
import imageio
```

```python
height2width_ratio = 5
```

```python
from scipy.stats import qmc
import numpy as np

pts_lists = []
for i in range(height2width_ratio):
    sampler = qmc.PoissonDisk(
        d=2,            # 2D space
        #radius=0.03,
        radius=0.04,
        seed=42 + i,
    )

    p1 = sampler.fill_space()
    pts_lists.append(p1 + [0,i])

pts = np.concatenate(pts_lists)/[1,height2width_ratio]
pts = np.float32(pts)
plt.scatter(pts.T[0],pts.T[1])
```

```python
import numpy as np
import matplotlib.pyplot as plt
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite,FractalType


height = 256 * height2width_ratio
width = 256
scale = 0.02
#scale = 1

y, x = np.mgrid[:height, :width]

coords = np.array(
    [
        x.ravel() * scale,
        y.ravel() * scale,
    ],
    dtype=np.float32,
)

noise = FastNoiseLite(seed=42)
noise.frequency = 0.02
noise.fractal_type = FractalType.FractalType_FBm
noise.fractal_octaves = 5
noise.fractal_lacunarity = 1.8
noise.fractal_gain = 0.8

image = noise.gen_from_coords(coords).reshape(height, width)

# noise = FastNoiseLite(seed=42)
# noise.frequency = 0.02
# noise.fractal_type = FractalType.FractalType_FBm
# noise.fractal_octaves = 2
# noise.fractal_lacunarity = 1.8
# noise.fractal_gain = 0.8
# low_octave_image = noise.gen_from_coords(coords).reshape(height, width)

_,axs = plt.subplots(1,2)
axs[0].imshow(image, cmap="gray")
#axs[1].imshow(low_octave_image, cmap="gray")
plt.show()
```

```python
rng = np.random.default_rng(seed = 123)
rnd_vals = rng.uniform(size = len(pts))
```

```python
max_val = np.max(coords)
points_probs = noise.gen_from_coords((pts*[1,height2width_ratio]).astype(np.float32).T * max_val)
points_probs -= np.min(points_probs)
points_probs /= np.max(points_probs)

points_recal = pts[rnd_vals < points_probs ]

_,axs = plt.subplots(1,1,figsize = (10,10))
axs.scatter(points_recal.T[0], points_recal.T[1], s=10)


# max_val = np.max(coords)
# unipoints_probs = noise.gen_from_coords(unipoints.T * max_val)
# unipoints_probs -= np.min(unipoints_probs)
# unipoints_probs /= np.max(unipoints_probs)

# unipoints_recal = unipoints[rnd_vals < unipoints_probs ]

# axs[1].scatter(unipoints_recal.T[0], unipoints_recal.T[1], s=10)

# for ax in axs:
#     ax.set_aspect("equal")
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)


```

```python

```

```python
import cv2
import scipy.ndimage as ndi


def draw_circles(canvas,centers,radii):
    if np.isscalar(radii):
        radii = np.repeat(radii,len(xy))

    centers = np.int32(centers)
    radii = np.int32(radii)
    for (x,y),radius in zip(centers,radii):
        cv2.circle(canvas,(x,y),radius,1,thickness = -1)
        

class RodOxideStopsGenerator:

    def __init__(self, shape):
        self.shape = shape
    

    def get_oxide_mask(self,seed):
        
        oxide_mask = np.uint8(np.zeros(self.shape))
        
        draw_circles(
            oxide_mask,
            points_recal*oxide_mask.T.shape,
            radii
        )
        
        return oxide_mask
        

g = RodOxideStopsGenerator()
        


points_filter = rnd_vals < points_probs
points_recal = pts[points_filter ]
radii = ((points_probs[points_filter] +1)/2 )*10 +10




oxide_mask = np.uint8(np.zeros((height,width)))

draw_circles(
    oxide_mask,
    points_recal*oxide_mask.T.shape,
    radii
)

oxide_mask = np.float32(oxide_mask)
oxide_mask = ndi.gaussian_filter(oxide_mask,1.2)
#plt.imshow( np.maximum(oxide_mask ,(low_octave_image + .2)/1.2 ))
plt.imshow( oxide_mask)
```

# RENDERING

```python
import synthnf.scene_spec as ss
import synthnf.scene_variation as sv
import synthnf.mitsuba_layer as ml

krng = sv.KeyedRNG(123)

inspection_scene = ss.InspectionScene(
    nfa_spec= ss.NfaSpec.from_shape(
        ss.RodsSquareSpec(rod_height=150,column_number=17),
        rods_material_spec=ss.MaterialOxidizedConductor(
            conductor_spec = ss.MaterialNamedAnyConductorSpec(
                conductor_name = "custom_Zircon",
                rough_conductor_spec=ss.MaterialRoughConductorSpec(alpha_u=.025,alpha_v=.05)
            ),
            oxidation_amount = .5
        ),
    ),
    cam_ring_spec= ss.AhlbergCameraRingSpec.single_cam_four_x_lights(field_of_view=40, light_intensity=200,light_height_offset=0),
    env_map_spec=ss.EnvironmentMapSpec(
        intensity_scale=1.05
    ),
)

mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene)

scene_dict = mitsuba_scene.to_scene_dict()



porcelain = {
    "type": "roughplastic",
    "distribution": "ggx",
    "int_ior": 1.5,
    "ext_ior": 1.0,
    "alpha": 0.125,
    "diffuse_reflectance": {
        "type": "rgb",
        "value": [0.62, 0.60, 0.76],
    },
}

oxide_generator = 

for i in range(17):
    element_key = f'rod_{i:05}'

    oxide_mask = oxide_generator.get_oxide_mask(seed = 123)
    rod_material_weight = (oxide_mask[:,:,None] + 0 )/4
    
    new_material = {
        "type": "blendbsdf",
        "weight": {
            "type": "bitmap",
            "data": mi.TensorXf(rod_material_weight),
            "raw": True,
            "filter_type": "nearest",
            "wrap_mode": "repeat",
            "to_uv": mi.ScalarTransform4f.scale([1, 1.0, 1.0])
        },
        "bsdf_0": {'type':'ref','id':'shared_rod_material'},
        "bsdf_1": porcelain,
    }
    
    scene_dict[element_key]['material'] = new_material

rend = mi.render(mi.load_dict(scene_dict),spp = 32)
rendered = ml.srgb_bitmap(rend)
plt.imshow(rendered)
```

```python
ox=imageio.imread('/home/knotek/ox-gradient-red.png')
plt.imshow(ox)
```

```python
render_res= mitsuba_scene.render(denoise=True,include_labels=True,spp=64)


post_process_rendering_variation = ml.PostProcessRenderingVariation(
    noise_denoise_blend=sv.MultiplicativeScalarUniformVariationSpecs(.6,.95)
)
post_processed_image = post_process_rendering_variation.sample(render_res,krng,"post_process")
plt.imshow(post_processed_image)

```
