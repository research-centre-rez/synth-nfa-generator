from dataclasses import dataclass
import numpy.typing as npt
import synthnf.scene_variation as sv
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite,FractalType
from scipy.stats import qmc
import numpy as np
import cv2
import scipy.ndimage as ndi



class CylinderNoise():
    
    def __init__(self,noise):
        self.noise = noise
    
    def full_cylinder_surface(
        self,
        resolution_x,
        resolution_y,
        center_x,
        center_y,
        radius,
        height,
        texture_zoom=1
    ):
        cylinder_scales = np.array([radius,radius,height])/texture_zoom
        uc = unit_stacked_cylinder(resolution_x,resolution_y)
        
        cc = (uc + [center_x,center_y,0])*cylinder_scales
        return self.noise.gen_from_coords(np.float32(cc).T).reshape(resolution_y,resolution_x)
    
    def sample_3d_points(
        self,
        points,
        radius,
        height,
        texture_zoom = 1,
    ):
        cylinder_scales = np.array([radius,radius,height])/texture_zoom
        cc = points*cylinder_scales
    
        return self.noise.gen_from_coords(np.float32(cc).T)


@dataclass
class OxideSpotTextureGeneratedResult:
    oxide_mask:npt.NDArray
    spot_centers_all:npt.NDArray
    sufrace_probabilities_all:npt.NDArray
    points_filter:npt.NDArray
    # I'd like to have parameters of the noise as well so I can recreate it
    
        
class OxideSpotTextureGenerator():
    
    def __init__(
        self,
        cylinder_noise:CylinderNoise, 
        texture_resolution_x = 256, 
    ):
        self.cylinder_noise = cylinder_noise
        self.texture_resolution_x= texture_resolution_x
        pass
    
    def generate(self,rod_center_x,rod_center_y, rod_radius, rod_height ,poisson_disk_radius, krng,noise_texture_zoom=1,*keys):
        poisson_seed = krng.uint32('poisson_seed',*keys)
        spot_centers = poisson_disk(
            poisson_seed, 
            poisson_disk_radius,
            hw_ratio = int(rod_height/(2*np.pi*rod_radius)),
        )

        
        p_cyl = cylinder_surface_to_3d(spot_centers)
        surface_probabilities = self.cylinder_noise.sample_3d_points(
            p_cyl + [rod_center_x*noise_texture_zoom,rod_center_y*noise_texture_zoom,0],
            rod_radius,
            rod_height,
            texture_zoom=noise_texture_zoom,
        )
        
        keep_probability = normalize(surface_probabilities)
    
        points_filter = keep_probability>krng.uniform_list(len(keep_probability))
        spts = spot_centers[points_filter]

        radii = ((surface_probabilities[points_filter] +1)/2 )*15 +10

        xy_ratio =np.ceil(rod_height/(2*np.pi*rod_radius ))
        res_y = self.texture_resolution_x*int(xy_ratio) 
        oxide_mask = np.zeros(
            (res_y ,self.texture_resolution_x),
            dtype = np.uint8
        )

        draw_circles(
            oxide_mask,
            spts*oxide_mask.T.shape,
            radii
        )
        
        #oxide_mask = np.roll(oxide_mask,int(krng.uniform(0,oxide_mask.shape[1])),axis=1)
        return OxideSpotTextureGeneratedResult(
            oxide_mask=oxide_mask,
            spot_centers_all=spot_centers,
            sufrace_probabilities_all=surface_probabilities,
            points_filter = points_filter,
        )



def poisson_disk(
    seed,
    radius,
    hw_ratio = 1,
    squash_y = True,
    ):
    
    pts_lists = []
    for i in range(hw_ratio):
        sampler = qmc.PoissonDisk(
            d=2,
            radius=radius,
            seed=seed + i*1234,
        )
        p1 = sampler.fill_space()
        pts_lists.append(p1 + [0,i])
    
    pts = np.concatenate(pts_lists)
    if squash_y:
        pts = pts/[1,hw_ratio]

    return np.float32(pts)


def perlin_noise_gen(
    seed,
    scale = 0.02,
    noise_frequency = .02,
    noise_octaves = 3,
    noise_lacunarity = 2,
    noise_gain = .5
):
    noise = FastNoiseLite(seed=seed)
    noise.frequency = noise_frequency
    noise.fractal_type = FractalType.FractalType_FBm
    noise.fractal_octaves = noise_octaves
    noise.fractal_lacunarity = noise_lacunarity
    noise.fractal_gain = noise_gain
    
    return noise


def cylinder_surface_to_3d(surf_01):
    
    t = surf_01[:,0] * 2*np.pi
    z_unit = surf_01[:,1]
    
    x_unit = np.cos(t)
    y_unit = np.sin(t)
    
    return np.stack([x_unit,y_unit,z_unit]).T


def draw_circles(canvas,centers,radii):

    if np.isscalar(radii):
        radii = np.repeat(radii,len(xy))

    centers = np.int32(centers)
    radii = np.int32(radii)
    for (x,y),radius in zip(centers,radii):
        cv2.circle(canvas,(x,y),radius,1,thickness = -1)
        
        
def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min)/(arr_max - arr_min)
    
def unit_stacked_cylinder(res_x,res_y):
    t = np.linspace(0,2*np.pi,res_x)
    
    xs = np.cos(t)/2 +.5
    ys = np.sin(t)/2 +.5
    zs = np.arange(res_y)/res_y
    
    xy_plane = np.vstack([xs,ys,np.zeros((res_x))]).T
    
    return np.concatenate([xy_plane+[0,0,z]for z in zs],axis= 0)

