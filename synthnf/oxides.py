from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from pyfastnoiselite.pyfastnoiselite import FastNoiseLite, FractalType
from scipy.stats import qmc


def normalize_values(arr):
    if len(arr) == 0:
        return np.array([])
    mmin = np.nanmin(arr)
    mmax = np.nanmax(arr)
    if mmin ==mmax:
        return np.zeros_like(arr)

    return (arr-mmin)/(mmax-mmin)
    

class ScaleIndependentNoise:
    """
    the goal of this is to bind parameters to the resolution. The frequency paramter is bound 
    to the resolution this way.

    If you create noises with
    noise_a) frequency = 1 and frequency_resolution = 256 
    noise_b) frequency = 1 and frequency_resolution = 512

    then noise_a(256,256) ~= noise_b(512,512)

    Now you can upscale, downscale the noise with a sensible setting
    """
    def __init__(
        self,
        seed,
        octaves = 2,
        lacunarity = 2,
        frequency = 1,
        frequency_resolution = 256,
        
    ):
        self.perlin = perlin_noise_gen(
            seed,
            noise_frequency=frequency,
            noise_octaves=octaves,
            noise_lacunarity=lacunarity,
        )
        self.frequency_resolution = frequency_resolution
        
    def generate_2d(
        self,
        width,
        height,
        normalize=True,
        shift_x=0,
        shift_y=0,
    ): 
        
        xs = shift_x + np.arange(width)
        ys = shift_y + np.arange(height)
        
        grid_x,grid_y = np.meshgrid(xs,ys)        
        xyz = np.float32([
            [x,y,0] for x,y in zip(grid_x.flatten(),grid_y.flatten())
        ])

        points = self.generate_from_coords(
            xyz,
            normalize=normalize
        )
        
        return points.reshape(height,width)

    def generate_from_coords(
        self,
        points_2d_or_3d,
        normalize = True,
    ):
        """
        Generate from coords shaped (n,2 or 3) 
        If 2d points are provided, the z=0 is added
        Note that a point (x,y) won't produce the same results as x,y,z with z = 0
        """
        shape = points_2d_or_3d.shape
        if shape[1] == 2:
            x,y = points_2d_or_3d.T
            z = np.zeros_like(x)
        elif shape[1] == 3:
            x,y,z = points_2d_or_3d.T
        else:
            raise ValueError(f"Invalid shape of the input array:{shape}. Expected (n,2) or (n,3)")


        xyz = np.stack([
            x/self.frequency_resolution,
            y/self.frequency_resolution,
            z/self.frequency_resolution,
        ]).astype(np.float32)
            
        noise_val = self.perlin.gen_from_coords(xyz)
        if normalize:
            return normalize_values(noise_val)
        return noise_val



class CylinderNoise:
    def __init__(self, noise):
        self.noise = noise

    def full_cylinder_surface(
        self,
        resolution_x,
        resolution_y,
        center_x,
        center_y,
        radius,
        height,
        texture_zoom=1,
    ):
        cylinder_scales = np.array([radius, radius, height]) / texture_zoom
        uc = unit_stacked_cylinder(resolution_x, resolution_y)

        cc = (uc + [center_x, center_y, 0]) * cylinder_scales
        return self.noise.gen_from_coords(np.float32(cc).T).reshape(
            resolution_y, resolution_x
        )

    def sample_3d_points(
        self,
        points,
        radius,
        height,
        texture_zoom=1,
    ):
        cylinder_scales = np.array([radius, radius, height]) / texture_zoom
        cc = points * cylinder_scales

        return self.noise.gen_from_coords(np.float32(cc).T)


@dataclass
class OxideSpotTextureGeneratedResult:
    oxide_mask: npt.NDArray
    spot_centers_all: npt.NDArray
    spot_probabilities_all: npt.NDArray
    spot_filters: npt.NDArray
    spot_radii_all:npt.NDArray
    surface_probability_texture: npt.NDArray|None = None
    # I'd like to have parameters of the noise as well so I can recreate it
    
    def plot(self, ax, include_oxides = True, include_probabilities = True):
        included = self.spot_centers_all[self.spot_filters]
        not_included = self.spot_centers_all[~self.spot_filters]
        

        if include_oxides:
            ax.imshow(self.oxide_mask,cmap='gray')
        
        if include_probabilities and self.surface_probability_texture is not None:
            ax.imshow(self.surface_probability_texture,cmap='gray',alpha = .5)
        ax.scatter(*included.T,marker='x',c = 'r')
        ax.scatter(*not_included.T,marker='x',c = 'orange',alpha = .3)

class OxideSpotTextureGenerator:
    def __init__(
        self,
        cylinder_noise: CylinderNoise,
        texture_resolution_x=256,
    ):
        self.cylinder_noise = cylinder_noise
        self.texture_resolution_x = texture_resolution_x

    def generate(
        self,
        rod_center_x,
        rod_center_y,
        rod_radius,
        rod_height,
        krng,
        *keys,
        noise_texture_zoom=1,
        min_oxide_size_px=10,
        max_oxide_size_px=25,
        oxide_spots_coverage_threshold=0.5,
        poisson_disk_radius=0.06,
    ):
        poisson_seed = krng.uint32("poisson_seed", *keys)
        spot_centers = poisson_disk(
            poisson_seed,
            poisson_disk_radius,
            hw_ratio=int(rod_height / (2 * np.pi * rod_radius)),
        )

        p_cyl = cylinder_surface_to_3d(spot_centers)
        surface_probabilities = self.cylinder_noise.sample_3d_points(
            p_cyl
            + [rod_center_x * noise_texture_zoom, rod_center_y * noise_texture_zoom, 0],
            rod_radius,
            rod_height,
            texture_zoom=noise_texture_zoom,
        )

        keep_probability = normalize(surface_probabilities)

        # this is controllable, yet pretty bad looking way to filter the spots
        # n = len(spot_centers)
        # desc_sort_idcs = np.argsort(keep_probability)[::-1]
        # top_k_idcs = desc_sort_idcs[:int(n*oxide_spots_coverage)]
        # points_filter = np.zeros(n, dtype=bool)
        # points_filter[top_k_idcs] = True

        random_vals = (
            krng.uniform_list(len(keep_probability)) - oxide_spots_coverage_threshold
        )
        points_filter = keep_probability > random_vals
        spts = spot_centers[points_filter]

        ox_size_diff = max_oxide_size_px - min_oxide_size_px
        radii = surface_probabilities * ox_size_diff + min_oxide_size_px

        xy_ratio = np.ceil(rod_height / (2 * np.pi * rod_radius))
        res_y = self.texture_resolution_x * int(xy_ratio)
        oxide_mask = np.zeros((res_y, self.texture_resolution_x), dtype=np.uint8)
        draw_circles(oxide_mask, spts * oxide_mask.T.shape, radii[points_filter])

        # oxide_mask = np.roll(oxide_mask,int(krng.uniform(0,oxide_mask.shape[1])),axis=1)
        return OxideSpotTextureGeneratedResult(
            oxide_mask=oxide_mask,
            spot_centers_all=spot_centers,
            surface_probabilities_all=surface_probabilities,
            points_filter=points_filter,
            spot_radii_all = radii,
        )


def poisson_disk(
    seed,
    radius,
    hw_ratio=1,
):
    tall = True
    #make sure that height weight ratio is more than 1
    if hw_ratio<1:
        hw_ratio = 1/hw_ratio
        tall = False
    
    hw_ratio = int(np.ceil(hw_ratio))

    pts_lists = []
    for i in range(hw_ratio):
        sampler = qmc.PoissonDisk(
            d=2,
            radius=radius,
            seed=seed + i,
        )
        p1 = sampler.fill_space()
        pts_lists.append(p1 + [0, i])

    pts = np.concatenate(pts_lists)
    if tall:
        pts = pts / [1, hw_ratio]
    else:
        pts = pts / [hw_ratio, 1]

    return np.float32(pts)

def disk_size_from_expected(
    number_of_expected,
    hw_ratio,
    scaling_factor = 0.7783203125,
):
    """
    Estimates disk size based on the expected numper of points. Estimate is based on points
    per square area. To take care about the fact it's a circle the scaling factor has been
    empirically found out. 
    """
    if hw_ratio < 1:
        hw_ratio = 1/hw_ratio
    h = hw_ratio
    w = 1
    
    estimate = np.sqrt(w*h/number_of_expected)

    return estimate * scaling_factor

def perlin_noise_gen(
    seed,
    noise_frequency=0.02,
    noise_octaves=3,
    noise_lacunarity=2,
    noise_gain=0.5,
):
    noise = FastNoiseLite(seed=seed)
    noise.frequency = noise_frequency
    noise.fractal_type = FractalType.FractalType_FBm
    noise.fractal_octaves = noise_octaves
    noise.fractal_lacunarity = noise_lacunarity
    noise.fractal_gain = noise_gain

    return noise


def cylinder_surface_to_3d(surf_01):

    t = surf_01[:, 0] * 2 * np.pi
    z_unit = surf_01[:, 1]

    x_unit = np.cos(t)
    y_unit = np.sin(t)

    return np.stack([x_unit, y_unit, z_unit]).T


def draw_circles(canvas, centers, radii):
    centers = np.int32(centers)
    radii = np.int32(radii)
    for (x, y), radius in zip(centers, radii):
        cv2.circle(canvas, (x, y), radius, 1, thickness=-1)


def normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def unit_stacked_cylinder(res_x, res_y):
    t = np.linspace(0, 2 * np.pi, res_x)

    xs = np.cos(t) / 2 + 0.5
    ys = np.sin(t) / 2 + 0.5
    zs = np.arange(res_y) / res_y

    xy_plane = np.vstack([xs, ys, np.zeros(res_x)]).T

    return np.concatenate([xy_plane + [0, 0, z] for z in zs], axis=0)
