import pandas as pd
import warnings
import cv2
import scipy.ndimage as ndi
from typing import Any
import synthnf.assets as assets
import synthnf.scene_spec as ss
import synthnf.scene_variation as sv
import synthnf.oxides as ox
import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, asdict
import json
from datetime import datetime

InstanceMask =npt.NDArray
Image = npt.NDArray

@dataclass
class RenderResult:
    raw_rgba:Image
    raw_gray:Image
    denoised_rgba:Image
    denoised_gray:Image
    render_scene:mi.SceneParameters
    label_rgba:Image
    label_instances:InstanceMask
    label_scene:mi.SceneParameters # todo make abstract later
    label_oxide:Image

@dataclass
class MitsubaElement:
    element_key:str
    element_dict:dict[str,Any]

    def to_dict(self):
        inner_out_dict = {
        }
        for k,v in self.element_dict.items():
            if isinstance(v,MitsubaElement):
                inner_out_dict[k]=v.to_dict()
            else:
                inner_out_dict[k]=v
            
        
        return {
            self.element_key:inner_out_dict
        }

@dataclass
class MitsubaRodElement(MitsubaElement):
    oxide_texture:Image | None = None
    


@dataclass
class MitsubaElementCollection:
    elements:tuple[MitsubaElement,...]

    def to_dict(self):
        res = {}
        for e in self.elements:
            res = res|e.to_dict()
        return res

@dataclass(frozen=True)
class MitsubaScene:
    rods:list[MitsubaElement]
    rods_shared_material: MitsubaElement
    grids:list[MitsubaElement]
    grids_shared_material: MitsubaElement
    main_camera:MitsubaElement
    emitters:list[MitsubaElement]
    env_map:MitsubaElement|None=None
    global_illumination:MitsubaElement|None=None
    medium:MitsubaElement|None=None
    
    def to_scene_dict(self):
        scene_dict = {
            "type":"scene",
        }

        if self.medium is not None:
            scene_dict["integrator"]={
                "type": "volpathmis",
                "max_depth": 8,
                "rr_depth": 5,
            }
        else:
            scene_dict["integrator"]={"type":"path"}

            

        rods_list = [r.to_dict() for r in self.rods]
        merged_rods = {k: v for d in rods_list for k, v in d.items()}
        scene_dict = scene_dict | merged_rods
        scene_dict = scene_dict| self.rods_shared_material.to_dict()
        scene_dict = scene_dict| self.grids_shared_material.to_dict()
        cam_dict = self.main_camera.to_dict()
        if self.medium is not None:
            cam_dict[self.main_camera.element_key]["medium_ref"] = {
                "type": "ref",
                # TODO this is a huge problem in the future
                # this assumes that the density is the first argument
                # I add this reference in the .from_scene... factory method
                "id": self.medium.elements[0].element_key,
            }

        scene_dict = scene_dict | cam_dict    
        
        rods_emitters = [e.to_dict() for e in self.emitters]
        merged_emitters = {k: v for d in rods_emitters for k, v in d.items()}
        scene_dict = scene_dict | merged_emitters

        grids = [g.to_dict() for g in self.grids]
        merged_grids = {k: v for d in grids for k, v in d.items()}
        scene_dict = scene_dict | merged_grids

        if self.medium:
            scene_dict = scene_dict | self.medium.to_dict()
        if self.env_map:
            scene_dict = scene_dict | self.env_map.to_dict()
        if self.global_illumination:
            scene_dict = scene_dict | self.global_illumination.to_dict()

        return scene_dict


    def render(self, spp = 256,denoise=False,rodgrid_labels=False,oxide_labels = False)->RenderResult:
        scene_dict = self.to_scene_dict()
        scene = mi.load_dict(scene_dict)
        
        label_sensor_dict = self.main_camera.to_dict()
        cam_key = list(label_sensor_dict.keys())[0]
        label_sensor_dict[cam_key]['film']['rfilter']: {'type': 'box'}
        # HACK (IN A SCENE WHERE CAMERA IS SUBMERGED) removing the reference to the medium from camera will
        # result in medium not being included in rendering rods. 
        if self.medium is not None and 'medium_ref' in label_sensor_dict[cam_key]:
            del label_sensor_dict[cam_key]['medium_ref']
        label_sensor = mi.load_dict(label_sensor_dict).sensors()[0]
        
        beauty_sensor_dict = self.main_camera.to_dict()
        cam_key = list(beauty_sensor_dict.keys())[0]
        beauty_sensor_dict[cam_key]['film']['rfilter']:{'type': 'tent'}
        beauty_sensor = mi.load_dict(beauty_sensor_dict).sensors()[0]
        
        rendered_rgba = mi.render(scene,spp=spp,sensor = beauty_sensor)    
        render_scene = mi.traverse(scene).copy()
        raw_rgba = dr.clip(rendered_rgba,0,1)
        raw_gray = dr.mean(raw_rgba[:,:,:3],axis=2)

        label_rgba = None
        label_instances = None
        
        if rodgrid_labels:
            aovintegrator = mi.load_dict({
                'type': 'aov',
                'aovs': 'si:shape_index',
                'beauty': {'type': 'path'}
            })
            rendered_label = mi.render(scene,spp=1,sensor = label_sensor,integrator = aovintegrator)
            label_rgba=rendered_label[:,:,:3]
            label_instances = rendered_label[:,:,4]
            label_scene = mi.traverse(scene).copy()
        if oxide_labels:
            pink = [1,0,1]
            traverse_oxide_label_scene(scene,self,label_color = pink)
            path_integrator = mi.load_dict({"type":"path","max_depth":1})
            lrend = mi.render(scene,spp = 8,integrator=path_integrator,sensor = label_sensor)
            oxide_mask = np.all(np.clip(lrend[:,:,:3],0,1) == np.array(pink)[None,None],axis=2)


        
        sensor = self.main_camera.element_dict
        res_x = sensor['film']['width']
        res_y = sensor['film']['height']
        denoised_rgba=None
        denoised_gray=None
        if denoise:
            denoiser = mi.OptixDenoiser(
                input_size=[res_x, res_y], 
                albedo=False, 
                normals=False, 
                temporal=False
            )
            denoised_rgba = dr.clip(denoiser(rendered_rgba),0,1)
            denoised_gray = dr.mean(denoised_rgba[:,:,:3],axis=2)

        return RenderResult(
            raw_rgba=srgb_bitmap(raw_rgba),
            raw_gray=srgb_bitmap(raw_gray),
            denoised_rgba=srgb_bitmap(denoised_rgba),
            denoised_gray=srgb_bitmap(denoised_gray),
            render_scene = render_scene,
            label_rgba=srgb_bitmap(label_rgba),
            label_instances = label_instances,
            label_scene = label_scene,
            label_oxide = oxide_mask
        )
    
    @staticmethod
    def from_inspection_scene(inspection_scene:ss.InspectionScene,krng,*keys):
        
        nfa_spec = inspection_scene.nfa_spec

        material_id =  "shared_rod_material"
        grids_material_id = "shared_grid_material"
        rods_material =  ss.MaterialBSDFSpec()
        grids_material = ss.MaterialBSDFSpec()
        rods = []
        grids = []
        if nfa_spec is not None:
            pure_black = {    
                "type": "diffuse",
                "reflectance": {
                    "type": "rgb",
                    "value": [0.0, 0.0, 0.0],
                },
            }
            if nfa_spec.rods_material_spec is not None:
                rods_material = nfa_spec.rods_material_spec
            if nfa_spec.grids_material_spec is not None:
                grids_material = nfa_spec.grids_material_spec
            rods =[
                MitsubaRodElement(f"rod_{i:05}",{
                    'type': 'cylinder',
                    'radius': rod.radius,
                    'p0':np.array(rod.xyz) - [0,0,rod.height/2],
                    'p1':np.array(rod.xyz) + [0,0,rod.height/2],
                    'material':{
                        "type": "mask",
                        "rod_material":{'type':'ref','id':material_id},
                        'opacity': {
                            "type": "rgb",
                            "value": 1,
                        }
                    }
                }) for i,rod in enumerate(nfa_spec.rods_specs)
            ]
            
            grids = [
                MitsubaElement(
                    f'grid_{i:05}',
                    mitsuba_grid(grid_spec,nfa_spec.rods_shape_spec, material_id = grids_material_id)
                ) 
                for i,grid_spec in enumerate(nfa_spec.grids_specs) if grid_spec.enabled
            ]
            

        rods_material_mitsuba = MitsubaElement(
            material_id, 
            convert_to_mitsuba_material(rods_material,material_id=material_id)
        )
        if isinstance(rods_material,ss.MaterialOxidizedConductorSpec) and rods_material.oxide_spots_spec is not None:
            oxide_spots_spec = rods_material.oxide_spots_spec
            
            poisson_seed = krng.uint32('poisson_seed',*keys)//2
            noise_seed = krng.uint32('noise_seed',*keys)//2

            spot_texture_generator=ox.OxideSpotTextureGenerator(
                cylinder_noise= ox.CylinderNoise(
                    ox.perlin_noise_gen(noise_seed)
                ),
            )
            rod_count = nfa_spec.rods_shape_spec.row_number

            oxide_bank = {}
            for i,rod_dict in enumerate(rods[:rod_count]):
                rcx,rcy = rod_dict.element_dict['p0'][:2]
                rod_radius = rod_dict.element_dict['radius']
                rod_height = np.abs(rod_dict.element_dict['p0'][2]-rod_dict.element_dict['p1'][2])
                
                texture_result = spot_texture_generator.generate(
                    rcx,
                    rcy,
                    rod_radius,
                    rod_height,
                    krng,
                    f'rod_{i}',
                    'texture',
                    *keys,
                    noise_texture_zoom=oxide_spots_spec.noise_texture_zoom,
                    min_oxide_size_px = oxide_spots_spec.min_oxide_size_px,
                    max_oxide_size_px = oxide_spots_spec.max_oxide_size_px,
                    oxide_spots_coverage_threshold = oxide_spots_spec.oxide_spots_coverage,
                    poisson_disk_radius = oxide_spots_spec.poisson_disk_radius,
                )
                oxide_bank[rod_dict.element_key] = texture_result

            new_rods = []
            for r,rod_texture_res in zip(rods,oxide_bank.values()):
                new_dict = dict(r.element_dict)
                oxide_mask = rod_texture_res.oxide_mask
                rod_material_weight = (oxide_mask[:,:,None])*oxide_spots_spec.opacity
                
                
                rod_material_dict = {
                    "type": "blendbsdf",
                    "weight": {
                        "type": "bitmap",
                        "data": mi.TensorXf(rod_material_weight),
                        "raw": True,
                        "filter_type": "nearest",
                        "wrap_mode": "repeat",
                    },
                    "bsdf_0": {'type':'ref','id':material_id},
                    "bsdf_1": porcelain_plastic(),
                }
                
                new_dict['material']['rod_material'] = rod_material_dict
                
                new_dict['emmiter'] = {
                    "type": "area",
                    "radiance": {
                        "type": "bitmap",
                        "data": mi.TensorXf(np.zeros_like(rod_material_weight)),
                        "raw": True,
                        "filter_type": "nearest",
                        "wrap_mode": "repeat",
                    }
                }
                
                new_rod = MitsubaRodElement(
                    r.element_key,
                    new_dict,
                    oxide_texture = rod_material_weight
                )
                new_rods.append(new_rod)
            
            rods = new_rods + rods[len(new_rods):]            
        

        if grids_material.oxide_spots_spec is not None:
            warnings.warn("Applying oxide spots to grids material is not supported yet")
        grids_material = MitsubaElement(
            grids_material_id, 
            convert_to_mitsuba_material(grids_material,material_id = grids_material_id)
        )
        
        sens = inspection_scene.cam_ring_spec.sensor_specs[0]
        if len(inspection_scene.cam_ring_spec.sensor_specs) >1:
            warnings.warn("More than 1 camera found. Only the first was used")
        
        main_cam= MitsubaElement('main_camera',{
            "type": "perspective",
            "fov": sens.field_of_view,
            "to_world": mi.ScalarTransform4f().look_at(
                origin=sens.lookat_origin_xyz, 
                target=sens.lookat_target_xyz, 
                up=sens.lookat_up_xyz
            ),
            "far_clip": sens.far_clip,
            "film": {
                "type": "hdrfilm",
                "pixel_format": "rgba",
                "width": sens.resolution_x,
                "height": sens.resolution_y,
                "rfilter": {"type": "box"}, # <- TODO: this is not a default filter but I don't know why it was chosen.
            },
        })

        emitters = [ MitsubaElement(f'emitter_{i:05}', {
                'type': 'rectangle',
                'to_world': (
                    mi.ScalarAffineTransform4f().look_at(
                        origin=es.lookat_origin_xyz,
                        target=es.lookat_target_xyz,
                        up=es.lookat_up_xyz
                    )
                    # divided by to because recangle original 
                    # size is 2x2
                    .scale([
                        es.panel_width/2, 
                        es.panel_height/2, 
                        1
                    ])
                ),
                "emitter": {
                    "type": "area",
                    "radiance": {
                        "type": "rgb",
                        "value": es.intensity,
                    },
                },
            })
            for i,es in enumerate(inspection_scene.cam_ring_spec.emitter_specs)
        ]

        env_map = None
        if inspection_scene.env_map_spec:
            filepath = assets.get_asset_path(inspection_scene.env_map_spec.env_map_file)
            lr_angle = inspection_scene.env_map_spec.rot_left_right_angle
            env_map = MitsubaElement('env_map', {
                'type': 'envmap',
                'filename':filepath,
                'scale':inspection_scene.env_map_spec.intensity_scale,
                'to_world':(
                    mi.ScalarAffineTransform4f()\
                    .rotate((0,0,1),angle=lr_angle)\
                    .rotate((1,0,0),angle = 90)\
                    .scale(.5)
                )
            })
        global_illumination = None
        if inspection_scene.global_illumination_spec:
            global_illumination = MitsubaElement(
                'global_illumination',
                {
                    'type': 'constant',
                    'radiance': {
                        'type': 'rgb',
                        'value': inspection_scene.global_illumination.intensity,
                    }
                }
            )

        medium = None
        if inspection_scene.medium_spec is not None and inspection_scene.medium_spec.enabled:
            medium_content_dict,medium_boundaries_dict = create_medium(
                inspection_scene.medium_spec,
                element_name='medium',
            )
            
            medium_content = MitsubaElement(
                'medium',
                medium_content_dict
            )
            medium_boundaries = MitsubaElement(
                'medium_boundaries',
                medium_boundaries_dict
            )

            medium = MitsubaElementCollection((medium_content,medium_boundaries))

        return MitsubaScene(
            rods = rods,
            rods_shared_material = rods_material_mitsuba,
            grids = grids,
            grids_shared_material = grids_material,
            main_camera = main_cam ,
            emitters=emitters,
            env_map=env_map,
            global_illumination = global_illumination,
            medium = medium
        )

@dataclass
class PostProcessRenderingVariation:
    noise_denoise_blend:sv.MultiplicativeScalarUniformVariationSpecs = sv.MultiplicativeScalarUniformVariationSpecs()

    def sample(self, render_result:RenderResult,krng,*keys):
        alpha = self.noise_denoise_blend.sample(1,krng,*keys)

        return alpha*render_result.denoised_rgba + (1-alpha)*render_result.raw_rgba


def create_manual_conductor(eta_spectrum, k_spectrum, alpha_u=None, alpha_v=None):
    mat = {
        
        "eta": {
            "type": "spectrum",
            "value": eta_spectrum,
        },
        "k": {
            "type": "spectrum",
            "value": k_spectrum,
        },
        #"distribution": "ggx",
        #"sample_visible": False,
    }
    if alpha_u is not None and alpha_v is not None:
        mat["type"] = "roughconductor"
        mat["alpha_u"]= alpha_u
        mat["alpha_v"]= alpha_v
    else:
        mat["type"] = "conductor"
        
    return mat



def inconel_conductor(alpha_u=None, alpha_v=None):
    df_zirconium = pd.read_csv(assets.get_asset_path("spectrum_inconel.csv"))
    wv = df_zirconium["wavelength"] * 1000
    n = df_zirconium["n"]
    eta = df_zirconium["k"]
    eta_spectrum = list(zip(wv, n))
    k_spectrum = list(zip(wv, eta))
    return create_manual_conductor(eta_spectrum, k_spectrum, alpha_u=alpha_u, alpha_v=alpha_v)
    
def zirconium_conductor(alpha_u=None, alpha_v=None):
    df_zirconium = pd.read_csv(assets.get_asset_path("spectrum_zirconium.csv"))
    wv = df_zirconium["wavelength"] * 1000
    n = df_zirconium["n"]
    eta = df_zirconium["k"]
    eta_spectrum = list(zip(wv, n))
    k_spectrum = list(zip(wv, eta))
    return create_manual_conductor(eta_spectrum, k_spectrum, alpha_u=alpha_u, alpha_v=alpha_v)

def porcelain_plastic():
    return {
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

def traverse_oxide_label_scene(
    scene:mi.SceneParameters,
    mitsuba_scene:MitsubaScene, # used to disable emmiters and non-visible-rods,
    label_color,
    label_emitter_intensity = 10
):
    tt = mi.traverse(scene)
    label_color_3ch = np.array(label_color)[None,None]
    for r in mitsuba_scene.rods:    
        rod_bsdf_opacity = r.element_key+'.bsdf.opacity.value'
        if r.oxide_texture is not None:
            # turn oxides to emmiter so that it can be masked out
            oxide_texture = r.oxide_texture
            texture_tensor = mi.TensorXf(np.dstack([oxide_texture]*3) * label_color_3ch*label_emitter_intensity )
            tt[r.element_key + '.emitter.radiance.data'] = texture_tensor
        if rod_bsdf_opacity in tt:
            tt[rod_bsdf_opacity] = 0

    # disable any other emitters
    tt['env_map.scale'] = 0
    for e in mitsuba_scene.emitters:
        tt[f"{e.element_key}.emitter.radiance.value"] = mi.Color3f(0,0,0)
    tt.update();

# At this moment, this is recursive
def convert_to_mitsuba_material(material_definition,material_id=None):

    
    definition = {}

    if isinstance(material_definition,ss.MaterialOxidizedConductorSpec):
        blend_val = float(np.clip(material_definition.oxidation_amount,0,1))
        
        definition = {
            'type': 'blendbsdf',
            'weight': {"type": "spectrum", "value": blend_val},
            'rods':  convert_to_mitsuba_material(material_definition.conductor_spec,material_id=None),
            'oxidation': convert_to_mitsuba_material(material_definition.oxidation_spec,material_id=None),
        }
    elif isinstance(material_definition,ss.MaterialNamedAnyConductorSpec):
        if material_definition.conductor_name == 'custom_Zircon':
            if material_definition.rough_conductor_spec is None:
                definition = zirconium_conductor()
            else:
                rcs = material_definition.rough_conductor_spec
                definition = zirconium_conductor(rcs.alpha_u,rcs.alpha_v)
        elif material_definition.conductor_name == 'custom_Inconel':
            if material_definition.rough_conductor_spec is None:
                definition = inconel_conductor()
            else:
                rcs = material_definition.rough_conductor_spec
                definition = inconel_conductor(rcs.alpha_u,rcs.alpha_v)
        elif material_definition.conductor_name == 'custom_Porcelain':
            definition = porcelain_plastic()
        else:
            definition = {
                'material': material_definition.conductor_name
            }
            if material_definition.rough_conductor_spec is None:
                definition['type'] = 'conductor'
            else:
                rcs = material_definition.rough_conductor_spec
                definition['type'] = 'roughconductor'
                definition['alpha_u'] = rcs.alpha_u
                definition['alpha_v'] = rcs.alpha_v
    elif isinstance(material_definition,ss.MaterialOxideLayer):
        definition = {
            'type': 'roughplastic',
            'distribution': 'ggx',
            'alpha':.5,
            'int_ior': 1.0,#1.61,
            'ext_ior':1.325, #water
            'diffuse_reflectance': {
                'type': 'rgb',
                'value': material_definition.gray_scale
            }
        }        
    elif isinstance(material_definition,ss.MaterialBSDFSpec):
        rgb = (material_definition.r, material_definition.g, material_definition.b)

        definition =  {
            "type": "diffuse", 
            "reflectance": {
                "type": "rgb", "value": rgb
            }
        }
    else:
        raise RuntimeError("Material is not known type")

    if material_id is not None:
        definition['id'] = material_id
    return definition

def create_density_grid(
    grid_resolution:int,
    smooth_sigma:float,
    medium_density:float,
    heterogenity_noise_max:float,
    grid_seed = 42
):
    # TODO this shoul not be finxed
    rng = np.random.default_rng(grid_seed)
    
    xyz_res = [grid_resolution]*3
    noise = rng.random(xyz_res).astype(np.float32)
    noise = ndi.gaussian_filter(noise, sigma=smooth_sigma)
    
    # Normalize to 0..1
    noise -= noise.min()
    noise /= noise.max()
    
    # Add base water density plus turbulent variation
    density = medium_density + heterogenity_noise_max * noise
    
    # Mitsuba expects a 4D grid: X, Y, Z, channels
    density_grid = density[..., None]
    return mi.VolumeGrid(mi.TensorXf(density_grid))


def create_medium(
    heterogenous_medium_spec:ss.HeterogenousMediumSpec,
    element_name = 'medium',
):
    volume_spec = heterogenous_medium_spec.volume_spec
    density_grid = create_density_grid(
        volume_spec.resolution,
        volume_spec.smooth_sigma,
        volume_spec.density,
        volume_spec.heterogenity_noise_max
    )

    grid_to_world = mi.ScalarTransform4f.scale([volume_spec.cube_width]*3)
    
    if volume_spec.centered_to_origin:
        grid_to_world = mi.ScalarTransform4f.translate([-volume_spec.cube_width/2]*3) @ grid_to_world
    
    medium_content = {
    "type": "heterogeneous",
        "sigma_t": {
                "type": "gridvolume",
                "grid": density_grid,
                "to_world": grid_to_world
            },
    

        "albedo": {
            "type": "rgb",
            "value": heterogenous_medium_spec.albedo.to_rgb_tuple()
        },
    
        "phase": {
            "type": "hg",
            "g": heterogenous_medium_spec.hg_phase_g
        },
        "scale":heterogenous_medium_spec.scale
    }
    medium_boundaries = {
        "type": "cube", # it will be cylinder in the future
        "to_world": grid_to_world,
    
        # Index-matched boundary: not visible as a surface.
        "bsdf": {
            "type": "null"
        },            
        "interior": {
            "type": "ref",
            "id": "medium" # this will be problem if it changes name
        }
    }

    return medium_content,medium_boundaries
    
    

def srgb_bitmap(arr):
    if arr is None:
        return None
        
    bmp = mi.Bitmap(arr).convert(
        pixel_format=mi.Bitmap.PixelFormat.RGBA,
        component_format=mi.Struct.Type.Float32,
        srgb_gamma=True
    )
    return np.array(bmp)



def clean_rod_mask_jittering(mask):
    m = ndi.binary_opening(mask,np.ones((mask.shape[0]//2,1)))
    return mask * m


    
def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return (ymin, ymax + 1, xmin, xmax + 1)
    
    
def unsafe_read_labels(label_instances,clean_jitter = False):
    # FILTER RODS COARSE - shape-index-based instances from mitsuba
    unique_vals = np.unique(label_instances)
    
    rods_instance_mask = np.zeros_like(label_instances)
    h,w = rods_instance_mask.shape
    masks = np.array([
        (label_instances==val)*(i+1) for i,val in enumerate(unique_vals)
        if val!=0
    ])
    widths = np.array([ (b[3]-b[2]) for b in [bbox(m) for m in masks]])
    heights = np.array([ (b[1]-b[0]) for b in [bbox(m) for m in masks]])
    good_widths = (widths > 35) & (widths < 200) # & (heights > h*.8)
    filtered_masks = masks[good_widths][:17]

    for i,mask in enumerate(filtered_masks):
        mask = ndi.binary_opening(mask,np.ones((21,21)))
        rods_instance_mask += mask*(i+1)

    
    # FILTER MASK HERE
    grid_mask = np.zeros_like(label_instances)
    
    possible_grids = np.argwhere((widths > label_instances.shape[1] *.5) & (widths< w))
    if len(possible_grids) == 1:
        grid_mask_idx = np.squeeze(possible_grids)
        grid_mask = masks[grid_mask_idx]
        grid_mask = ndi.binary_fill_holes(grid_mask)


        n,arr,stats,centroids = cv2.connectedComponentsWithStats(np.uint8(grid_mask))
        
        # x = stats[i, cv2.CC_STAT_LEFT]
        # y = stats[i, cv2.CC_STAT_TOP]
        ws = stats[:, cv2.CC_STAT_WIDTH]
        hs = stats[:, cv2.CC_STAT_HEIGHT]
        areas = stats[:, cv2.CC_STAT_AREA]
        # skip the largest component as it is the background
        # this is pretty bad assumption
        grid_mask =  arr == np.argmax((hs*ws)[1:])+1

        # IF MASK IS PRESENT get rid of disconected components
        # that results from grid spliting the reods
        
        n,arr,stats,centroids = cv2.connectedComponentsWithStats(np.uint8(rods_instance_mask>0))
        rod_cmp_masks = np.array([ (arr == mask_idx) for mask_idx in range(1,n)])
        ws = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
        # rods is anything above scaled average
        ws[(ws == arr.shape[1])] = np.nan
        rod_cmp_filter = (ws > np.nanmean(ws)*.8)
        

        base_mask = np.zeros_like(arr)
        for i in np.argwhere(rod_cmp_filter):
            base_mask += (arr == i)
        rods_instance_mask = rods_instance_mask * base_mask
        
    else:
        if clean_jitter:
            rods_instance_mask = clean_rod_mask_jittering(rods_instance_mask)
        

    return rods_instance_mask, grid_mask
    
    

def dump_scene_params(inspection_scene,filename):
    dict_serialized_scene = asdict(inspection_scene)
    scene_parameters = {
        "scene":dict_serialized_scene,
        "timestamp":datetime.now().strftime("%Y%m%d-%H:%M:%S"),
        "note":""
    }
    
    with open(filename,'w') as f:
        json.dump(scene_parameters,f)


import synthnf.mesh_geometry as mg
from pathlib import Path

def mitsuba_grid(grid_spec,rods_shape_spec,material_id):
    tooth_ply_path = assets.get_asset_path(grid_spec.tooth_ply_filename)
    fourface_tooth_square_scene_el = mg.spacer_grid(
        tooth_ply_path,
        rods_per_face=rods_shape_spec.row_number,
        rod_width_mm = rods_shape_spec.rod_radius*2,
        gap_width_mm= rods_shape_spec.offset - rods_shape_spec.rod_radius*2,
        material={},
        is_hexagon = False
    )

    return {
        "type": "ply",
        "filename": fourface_tooth_square_scene_el['filename'],
        # "face_normals": False,
        # "flip_normals":False,
        "to_world": mi.ScalarTransform4f().translate((0,0,grid_spec.z_location)),#.scale((1,1,1)),
        "material":{"type":"ref","id":material_id},
        # "bsdf": {
        #     "type": "conductor",
        #     "material": "Al",
        # },
    }