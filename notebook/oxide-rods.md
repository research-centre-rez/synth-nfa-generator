---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: synth
    language: python
    name: synth
---

```python
%load_ext autoreload 
%autoreload 2
```

```python
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
```

```python
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from tqdm.cli import tqdm
import imageio
```

```python
import synthoxides.simulation as simu
import synthoxides.visualization as visu
```

```python

def simulate_growth(steps=None,seed = 123,mm_per_pixel = 10,pbar=None):
    if steps and steps <= 0:
        return []
        
    # Parameters
    width = 9.1*np.pi
    height = 100
    min_distance = .015
    
    points = simu.poisson_disk_sampling(width, height, min_distance)
    ng = simu.SpotNeighborGraph.from_spots(points,domain_width=width,domain_height = height,spot_size_mm = .75)

    textures = []

    not_filled = True
    tex = np.array([0])
    iteration = 0
    steps = steps if steps else np.inf
    while not_filled and iteration < steps:
        ng = ng.grow()
        tex = visu.to_texture(ng,mm_per_pixel)
        textures.append(tex)
        not_filled = np.any(tex == 0)
        iteration +=1    
        if pbar is not None:
            pbar.update(1)
    return textures

#pbar = tqdm(desc='Simulating oxide growth')
pbar = None
all_textures = [simulate_growth(steps =5,seed = i,pbar=pbar) for i in tqdm(range(11))]
```

```python
plt.figure(figsize=(20,10))
plt.imshow(np.sum([ t[0] for t in all_textures],axis=0))
```

```python
picked = [ t[0] for t in all_textures]

_,axs = plt.subplots(len(picked),1,figsize = (8,4*len(picked)))
for ax,tex in zip(axs,picked):
    ax.imshow(tex.T)
```

<!-- #region -->
# Scene


Nuclear Fuel Assembly (FA) has a hexagonal footprint with rods 9.1mm wide with 3.65 gap between. There are 11 rods per face.

NOTE: The plot show only one camera even thoug in reality there are three - each recording every other face. 

<!-- #endregion -->

```python
from synthnf.config.defaults import blueprints as bl
import synthnf.geometry.fuel_rods_mesh as frm

RODS_PER_FACE = 11
ROD_DIAMETER_MM = 9.1
ROD_GAP_MM = 3.65

def plot_lights(ax, light_radius,ring_diameter,light_offset):
    
    cam_light_left = plt.Circle(
        [-light_offset/2,-ring_diameter/2],
        radius = light_radius *3, # *3 to make it more visible
        color = 'orange',
        label = "Camera Light"
    )
    
    cam_light_right = plt.Circle(
        [light_offset/2,-ring_diameter/2],
        radius = light_radius*3,
        color = 'orange',
    )
    ax.add_patch(cam_light_left)
    ax.add_patch(cam_light_right)

def plot_camera(ax,ring_diameter):

    cam_ring = plt.Circle(
        (0, 0), 
        ring_diameter/2, 
        color='black', 
        fill=False,
        alpha = .2,
        label='Camera Ring Mount'
    )

    move_down = [0, ring_diameter/2]
    cam_poly = np.array([[-60,60],[60,60],[30,-50],[-30,-50]]) - move_down
    
    cam_box = plt.Polygon(
        cam_poly,
        edgecolor ='black',
        facecolor = 'black',
        label='Camera'
    )

    ax.add_patch(cam_ring)
    ax.add_patch(cam_box)
    

def plot_rods(ax,n_rods,rods_width_mm,rods_gaps_mm):
    rcs = frm.generate_rod_centers(n_rods,rods_width_mm,rods_gaps_mm)

    for x,y in rcs:
        rod = plt.Circle(
            (x,y),
            radius = rods_width_mm/2,
            color = 'blue',
        )
        ax.add_patch(rod)
    

ring_diameter = bl.camera_ring.diameter_mm
light_offset = bl.camera_ring.light_offset_mm
diameter_padded =  ring_diameter/2 * 1.2
camera_position = [0,ring_diameter/2,0]

light_radius = bl.camera_ring.light_diameter_mm/2
```

```python
import synthnf.materials.textures as textures
import synthnf.scene as scene
import synthnf.materials as mat
import synthnf.config.defaults as defaults
import synthnf.config.assets as assets



cam_distance = ring_diameter/2
camera = scene.cam_perspective_lookat(
        [0, cam_distance, 20],
        target = [0,0,0],
        res_y=600,
        res_x=750
    )


integrator = {"type": "path"}

default_high_contrast_material = {
    'type': 'diffuse',
    'reflectance': {
        'type': 'rgb',
        'value': [1, 0, 1]
    }
}

default_scene = {
    "type": "scene",
    "integrator": integrator,
    "sensor":camera,
    "light": {
        "type": "constant",
        'radiance': {
            'type': 'rgb',
            'value': 1
        }
     },
}

def compose_scene_rod_piece_real_param(
    cam_distance,
    cam_ligth_intensity = 20,
    light_height = 100,
    cloudiness = .5,
    uv = .1
):
    scene_dict = default_scene.copy()
    material = mat.zirconium_blend(cloudiness=cloudiness,alpha_u=uv,alpha_v=uv)    

    rcs = frm.generate_rod_centers(RODS_PER_FACE, ROD_DIAMETER_MM, ROD_GAP_MM)
    
    first_row = np.arange(0,14)
    second_row = np.arange(57,73)
    third_row = np.arange(111,126) 
    third_res = np.arange(159,163)
    idcs = np.concatenate([first_row,second_row, third_row,third_res])

    
    for i,(x,y) in enumerate(rcs[idcs]):
        fr = {        
            'type': 'cylinder',
            'radius': ROD_DIAMETER_MM/2,
            'p0':[x,y,50],
            'p1':[x,y,-50],
            'material': material
        }
        scene_dict[f'model_{i}']= fr  
    
    scene_dict['sensor'] = scene.cam_perspective_lookat(
        [0, cam_distance, 0],
        target = [0,0,0],
        res_y=600,
        res_x=750,
        fov=25.88
    )
    scene_dict['light'] = scene.create_mount_emmiter(
        [-light_offset/2,cam_distance,0],
        cam_ligth_intensity,
        light_height_mm=light_height
    )
    scene_dict['light_1'] = scene.create_mount_emmiter(
        [light_offset/2,cam_distance,0],
        cam_ligth_intensity,
        light_height_mm=light_height
    )
    return scene_dict
```

```python
import synthnf.scene as scene
```

```python
rod_imgs = []
cloudiness = [.1,.3,.45]
uvs = [0.058,0.12,0.28]
mi.set_log_level(mi.LogLevel.Error)
for i,uv in tqdm(zip(cloudiness,uvs), desc= "Generating different oxidation levels",total = len(uvs)):
    img_scene = compose_scene_rod_piece_real_param(
        cam_distance,
        cloudiness=i,
        light_height=500,
        cam_ligth_intensity=100,
        uv=uv,
    )
    # replace textures
    for i,texture_img in enumerate(picked):

        oxide_texture_intensity = .3
        oxide_texture = np.dstack([texture_img]*3)*oxide_texture_intensity
        # this is how full oxide looks like

        oxide_intensity = 1
        oxide_texture_element = mi.load_dict({
            'type' : 'bitmap',
            'data' : oxide_texture*oxide_intensity,
            'raw' : True
        })
        
        # values between 0-1 wont' work. Use 1
        alpha = 1
        alpha_texture_element = mi.load_dict({
            'type' : 'bitmap',
            'data' : np.ones_like(oxide_texture) * alpha, # np.float32(oxide_texture > 0), 
            'raw' : True
        })

        
        weight_texture_element = mi.load_dict({
            'type' : 'bitmap',
            'data' : oxide_texture,  #dr.ones(mi.TensorXf, shape = [30, 30, 3]),
            'raw' : True
        })
        
        rod = img_scene[f'model_{i}']   

        texture = {
            'type': 'mask',
            'material': {
                'type': 'diffuse',
                'reflectance':oxide_texture_element,
            },
            # having alpha is important otherwise the colors are crap
            'opacity':alpha_texture_element,
        }
    
        material = {
            'type':'blendbsdf',
            'weight':weight_texture_element,
            'bsdf_0': rod['material'],
            'bsdf_1': texture
        }
        
        rod['material'] = material
    img = scene.render_scene(img_scene,seed = 123,denoise=True)
    rod_imgs.append(img)
```

```python
import scipy.ndimage as ndi
    
for img in rod_imgs:
    fig,ax = plt.subplots(1,1,figsize = (20,10))
    ax.imshow(img)
    plt.show()
```

```python
assert False
```

```python
import imageio


def imwrite(uri,img_f):
    img = np.uint8(img_f*255)
    if len(img.shape) ==2:
        img = np.dstack([img]*3)
    imageio.imwrite(uri,img)

base_path = Path('/home/knotek/data/synth/triplet')
base_path.mkdir(exist_ok=True,parents=True)

names = ['fresh','used','gone']
for name,(img,_),lbl in zip(names, rod_imgs,clean_labels):
    img_path = base_path/f"img_{name}.png"
    imwrite(img_path,img)
    mask_path = base_path/f"label_{name}.png"
    imwrite(mask_path,lbl)
    
```

```python

```

# TODO
- pokrouceni
- distorze obrazu
- jiny textury oxidu
- jinej material

```python

```

```python
fresh_crop = fresh_img[:,30:290][-100:]

start = 244
width = fresh_crop.shape[1]
render_crop = rod_imgs[0][:,start:start+width][200:300,:,0]
render_crop = render_crop
joined = np.vstack([fresh_crop,render_crop])

_,(ax_real,ax_render,ax_joined) = plt.subplots(1,3,figsize= (16,4))
plt.suptitle("Fresh")

ax_real.imshow(fresh_crop,cmap='gray',vmin=0,vmax=1)
ax_real.set_title("Real")

ax_render.imshow(render_crop,cmap='gray',vmin=0,vmax=1)
ax_render.set_title("Render")

ax_joined.imshow(joined,cmap='gray',vmin=0,vmax=1)
ax_joined.set_title("Comparison")
plt.show()
```

```python
used_crop = resize(used_img[:100,28:267],(100,width))

width = used_crop.shape[1]
render_crop = rod_imgs[1][:,start:start+width][200:300,:,0]
render_crop = render_crop

used_joined = np.vstack([used_crop,render_crop])

_,(ax_real,ax_render,ax_joined) = plt.subplots(1,3,figsize= (16,4))
plt.suptitle("Used v1")

ax_real.imshow(used_crop,cmap='gray',vmin=0,vmax=1)
ax_real.set_title("Real")

ax_render.imshow(render_crop,cmap='gray',vmin=0,vmax=1)
ax_render.set_title("Render")

ax_joined.imshow(used_joined,cmap='gray',vmin=0,vmax=1)
ax_joined.set_title("Comparison")
plt.show()
```

```python
gone_crop = resize(gone_img[:100,33:-32],(100,width))

width = gone_crop.shape[1]
render_crop = rod_imgs[2][:,start:start+width][200:300,:,0]
render_crop = render_crop

joined_gone = np.vstack([gone_crop,render_crop])

_,(ax_real,ax_render,ax_joined) = plt.subplots(1,3,figsize= (16,4))
plt.suptitle("Used v2")

ax_real.imshow(gone_crop,cmap='gray',vmin=0,vmax=1)
ax_real.set_title("Real")

ax_render.imshow(render_crop,cmap='gray',vmin=0,vmax=1)
ax_render.set_title("Render")

ax_joined.imshow(joined_gone,cmap='gray',vmin=0,vmax=1)
ax_joined.set_title("Comparison")
plt.show()
```

```python
_,axs = plt.subplots(3,1,figsize=(8,12))
imgs_joined = [joined,used_joined,joined_gone]

for ax,img in zip(axs,imgs_joined):
    ax.imshow(img,cmap='gray',vmin=0,vmax=1)
```


