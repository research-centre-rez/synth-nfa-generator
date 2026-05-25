import pandas as pd
from typing import Any
import synthnf.assets as assets
import synthnf.scene_spec as ss
import drjit as dr
import mitsuba as mi
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

InstanceMask =npt.NDArray
Image = npt.NDArray

@dataclass
class RenderResult:
    raw_rgba:Image
    raw_gray:Image
    denoised_rgba:Image
    denoised_gray:Image
    label_rgba:Image
    label_instances:InstanceMask
    label_scene:mi.Scene # todo make abstract later


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
class MitsubaScene:
    rods:list[MitsubaElement]
    rods_shared_material: MitsubaElement
    main_camera:MitsubaElement
    emitters:list[MitsubaElement]
    env_map:MitsubaElement|None=None
    global_illumination:MitsubaElement|None=None
    
    def to_scene_dict(self):
        scene_dict = {
            "type":"scene",
            "integrator":{"type":"path"}
        }

        rods_list = [r.to_dict() for r in self.rods]
        merged_rods = {k: v for d in rods_list for k, v in d.items()}
        scene_dict = scene_dict | merged_rods
        scene_dict = scene_dict | self.rods_shared_material.to_dict()
        scene_dict = scene_dict | self.main_camera.to_dict()
        rods_emitters = [e.to_dict() for e in self.emitters]
        merged_emitters = {k: v for d in rods_emitters for k, v in d.items()}
        scene_dict = scene_dict | merged_emitters
        
        if self.env_map:
            scene_dict = scene_dict | self.env_map.to_dict()
        if self.global_illumination:
            scene_dict = scene_dict | self.global_illumination.to_dict()
        return scene_dict


    def render(self, spp = 256,denoise=False,include_labels=False)->RenderResult:
        scene_dict = self.to_scene_dict()
        scene = mi.load_dict(scene_dict)
        rendered_rgba = mi.render(scene,spp=spp)    
        raw_rgba = dr.clip(rendered_rgba,0,1)
        raw_gray = dr.mean(raw_rgba[:,:,:3],axis=2)

        label_rgba = None
        label_instances = None
        label_scene = None
        if include_labels:
            scene_label_dict = to_label_scene(
                scene_dict,
                sensor_id=self.main_camera.element_key,
                integrator_id='integrator',
            )
            label_scene = mi.load_dict(scene_label_dict)
            rendered_label = mi.render(label_scene,spp=1)
            label_rgba=rendered_label[:,:,:3]
            label_instances = rendered_label[:,:,4]

        
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
            label_rgba=srgb_bitmap(label_rgba),
            label_instances = label_instances,
            label_scene = label_scene,
        )
    
    @staticmethod
    def from_inspection_scene(inspection_scene:ss.InspectionScene):
        if not inspection_scene.nfa_spec.rods_material_spec:
            material =  ss.MaterialBSDFSpec()
        else:
            material = inspection_scene.nfa_spec.rods_material_spec

        material_id =  "shared_rod_material"
        material = MitsubaElement('shared_material', convert_to_mitsuba_material(material,material_id=material_id))
        rods =[MitsubaElement(f"rod_{i:05}",{
                'type': 'cylinder',
                'radius': rod.radius,
                'p0':np.array(rod.xyz) - [0,0,rod.height/2],
                'p1':np.array(rod.xyz) + [0,0,rod.height/2],
                'material':{'type':'ref','id':material_id},
            }) for i,rod in enumerate(inspection_scene.nfa_spec.rods_specs)
        ]

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
        if inspection_scene.global_illumination:
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

            
        return MitsubaScene(
            rods = rods,
            rods_shared_material = material,
            main_camera = main_cam ,
            emitters=emitters,
            env_map=env_map,
            global_illumination = global_illumination
        )


def to_label_scene(
    scene_dict,
    sensor_id = "sensor",
    integrator_id="integrator"
):    
    return {
        **scene_dict,
        # this overwrites previous sensor
        sensor_id: {
            **scene_dict[sensor_id],
            'film': {
                **scene_dict[sensor_id]['film'],
                'rfilter': {'type': 'box'}
            }
        },
        integrator_id: {
            'type': 'aov',
            'aovs': 'si:shape_index',
            'beauty': {'type': 'path'}
        }
    }


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



def zirconium_conductor(alpha_u=None, alpha_v=None):
    df_zirconium = pd.read_csv(assets.get_asset_path("spectrum_zirconium.csv"))
    wv = df_zirconium["wavelength"] * 1000
    n = df_zirconium["n"]
    eta = df_zirconium["k"]
    eta_spectrum = list(zip(wv, n))
    k_spectrum = list(zip(wv, eta))
    return create_manual_conductor(eta_spectrum, k_spectrum, alpha_u=alpha_u, alpha_v=alpha_v)


# At this moment, this is recursive
def convert_to_mitsuba_material(material_definition,material_id=None):

    
    definition = {}

    if isinstance(material_definition,ss.MaterialOxidizedConductor):
        blend_val = float(np.clip(material_definition.oxidation_amount,0,1))
        
        definition = {
            'type': 'blendbsdf',
            'weight': {"type": "spectrum", "value": blend_val},
            #'weight': {"type": "bitmap", "raw":True, "bitmap":mi.Bitmap(np.ones((100,100,3))*float(material_definition.oxidation_amount))},
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


def srgb_bitmap(arr):
    if arr is None:
        return None
        
    bmp = mi.Bitmap(arr).convert(
        pixel_format=mi.Bitmap.PixelFormat.RGBA,
        component_format=mi.Struct.Type.Float32,
        srgb_gamma=True
    )
    return np.array(bmp)