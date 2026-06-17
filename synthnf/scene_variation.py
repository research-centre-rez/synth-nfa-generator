from typing import Any
import hashlib
from dataclasses import replace
import itertools
import synthnf.scene_spec as ss
from dataclasses import dataclass, replace
import numpy as np

class KeyedRNG:
    def __init__(self, root_seed: int,digest_size = 16):
        self.root_seed = root_seed 
        self._digest_size=digest_size

    def unit(self, *keys) -> float:
        x = self.pythonint(*keys)
        return x / (1 << (8 * self._digest_size))
    
    def pythonint(self,*keys)->int:
        return stable_hash(
            self.root_seed, *keys,
            digest_size=self._digest_size
        )
    
    def uint32(self,*keys)->int:
        return stable_hash(
            self.root_seed, *keys,
            digest_size=8
        ) % (2**32)

    def uniform(self, low=0,high=1, *keys) -> float:
        u = self.unit(*keys)
        return low + u * (high - low)

    def choice(self, options,krng,*keys):
        idx = int(krng.uniform(0,len(options),*keys))
        return options[idx]

    def uniform_list(self,size,low=0,high=1,*keys) -> list[float]:
        return np.array([
            self.uniform(low,high, i,*keys) 
            for i in np.arange(size)
        ])

    # def normal(self, spec, *keys) -> float:
    #     u1 = self.unit(*keys, 0)
    #     u2 = self.unit(*keys, 1)
    #     z = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
    #     x = spec.mean + spec.std * z
    #     return clamp(x, spec.min_value, spec.max_value)


def stable_hash(*parts: Any, digest_size: int = 16,byteorder = 'little') -> int:
    h = hashlib.blake2b(digest_size=digest_size)

    for part in parts:
        data = repr(part).encode("utf-8")
        h.update(len(data).to_bytes(4, byteorder))
        h.update(data)

    return int.from_bytes(h.digest(), byteorder)

@dataclass(frozen=True)
class ChanceBoolVariation:
    success_probability:float = 1

    def sample(self,_,krng,*keys):
        u = krng.uniform(0,1,'chance',*keys)
        return u<self.success_probability

@dataclass(frozen=True)
class AdditiveScalarUniformVariationSpecs:
    low:float = 0
    high:float = 0

    def sample(self,base,krng,*keys):
        v = krng.uniform(self.low,self.high,*keys)     
        return base+v
        
@dataclass(frozen=True)
class MultiplicativeScalarUniformVariationSpecs:
    low:float = 1
    high:float = 1

    def sample(self,base,krng,*keys):
        v = krng.uniform(self.low,self.high,*keys)     
        return base*v
    
@dataclass(frozen=True)
class XyzUniformVariationSpecs:
    x:AdditiveScalarUniformVariationSpecs=AdditiveScalarUniformVariationSpecs()
    y:AdditiveScalarUniformVariationSpecs=AdditiveScalarUniformVariationSpecs()
    z:AdditiveScalarUniformVariationSpecs=AdditiveScalarUniformVariationSpecs()

    def sample(self,base,krng,*keys):
        return np.array([
            var.sample(b,krng,k,*keys) 
            for b,var,k in zip(base,[self.x,self.y,self.z],"xyz")
        ])   
        
@dataclass(frozen=True)
class ChoiceVariationParams:
    options:tuple[str,...] = ()
    # todo weights

    def sample(self,base,krng,*keys):
        if self.options == 0:
            return base
        return krng.choice(self.options,krng,*keys)
        
@dataclass(frozen=True)
class FuelRodVariationParams:
    xyz:XyzUniformVariationSpecs=XyzUniformVariationSpecs()
    radius:MultiplicativeScalarUniformVariationSpecs=MultiplicativeScalarUniformVariationSpecs()
    height:MultiplicativeScalarUniformVariationSpecs=MultiplicativeScalarUniformVariationSpecs()

    def sample(self,fuel_rod_spec:ss.FuelRodSpec,krng,*keys)->ss.FuelRodSpec:
        new_xyz = tuple(self.xyz.sample(fuel_rod_spec.xyz ,krng,'xyz_var',*keys))
        return replace(
            fuel_rod_spec,
            radius=self.radius.sample(fuel_rod_spec.radius, krng,'radius',*keys),
            height=self.height.sample(fuel_rod_spec.height, krng,'height',*keys),
            xyz = new_xyz
        )

@dataclass(frozen=True)
class SpacerGridVariation:
    #tooth_type - plain and complex
    z_location:AdditiveScalarUniformVariationSpecs=AdditiveScalarUniformVariationSpecs()
    enabled:ChanceBoolVariation = ChanceBoolVariation(1)
    tooth_ply_filename:ChoiceVariationParams = ChoiceVariationParams()
    def sample(self, grid_spec:ss.SpacerGridSpec, krng,*keys):
        return replace(
            grid_spec,
            z_location = self.z_location.sample(grid_spec.z_location,krng,"z_loc",*keys),
            enabled = self.enabled.sample(None, krng,"enabled",*keys),
            tooth_ply_filename = self.tooth_ply_filename.sample(grid_spec.tooth_ply_filename,krng,"tooth_ply",*keys),
        )

@dataclass(frozen=True)
class MaterialRoughConductorVariation:
    alpha_u:MultiplicativeScalarUniformVariationSpecs=MultiplicativeScalarUniformVariationSpecs()
    alpha_v:MultiplicativeScalarUniformVariationSpecs=MultiplicativeScalarUniformVariationSpecs()

    def sample(self,rough_conductor_spec:ss.MaterialRoughConductorSpec|None,krng,*keys):
        if rough_conductor_spec is None:
            return None
        return replace(
            rough_conductor_spec,
            alpha_u = self.alpha_u.sample(rough_conductor_spec.alpha_u,krng,'u',*keys),
            alpha_v = self.alpha_v.sample(rough_conductor_spec.alpha_v,krng,'v',*keys)
        )

@dataclass(frozen=True)
class MaterialConductorVariation:
    conductor_name:ChoiceVariationParams = ChoiceVariationParams()
    rough_conductor_variation:MaterialRoughConductorVariation=MaterialRoughConductorVariation()

    def sample(self,rods_material_spec:ss.MaterialSpec,krng,*keys)->ss.MaterialSpec:
        if isinstance(rods_material_spec ,ss.MaterialNamedAnyConductorSpec):
            
            return replace(
                rods_material_spec,
                conductor_name = self.conductor_name.sample(rods_material_spec,krng,'name',*keys),
                rough_conductor_spec = self.rough_conductor_variation.sample(rods_material_spec.rough_conductor_spec,krng,'rough',*keys)
            )
        else:
            warnings.warn("Base variation cannot vary non-conductor material")
            return rods_material_spec

@dataclass(frozen=True)
class TextureOxideSpotsVariation:
    noise_texture_zoom:MultiplicativeScalarUniformVariationSpecs=MultiplicativeScalarUniformVariationSpecs()
    min_oxide_size_px:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    max_oxide_size_px:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    oxide_spots_coverage:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    opacity:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    poisson_disk_radius:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    blur_sigma:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    
    def sample(self, oxide_spots_spec:ss.TextureOxideSpotsSpec,krng,*keys):
        if oxide_spots_spec is None:
            return None
        return replace(
            oxide_spots_spec,
            noise_texture_zoom = self.noise_texture_zoom.sample(oxide_spots_spec.noise_texture_zoom,krng , 'noise_texture_zoom',*keys),
            min_oxide_size_px = self.min_oxide_size_px.sample(oxide_spots_spec.min_oxide_size_px,krng,'min_oxide_size_px',*keys),
            max_oxide_size_px = self.max_oxide_size_px.sample(oxide_spots_spec.max_oxide_size_px,krng,'max_oxspx',*keys),
            oxide_spots_coverage = self.oxide_spots_coverage.sample(oxide_spots_spec.oxide_spots_coverage,krng,'oxsc',*keys),
            opacity = self.opacity.sample(oxide_spots_spec.opacity,krng,'opacity',*keys),
            poisson_disk_radius = self.poisson_disk_radius.sample(oxide_spots_spec.poisson_disk_radius,krng,'pdr',*keys),
            blur_sigma = self.blur_sigma.sample(oxide_spots_spec.blur_sigma,krng,'blur_s',*keys),
        )
        
@dataclass(frozen=True)
class MaterialOxidizedConductorVariation:
    conductor_variation:MaterialConductorVariation = MaterialConductorVariation()
    oxidation_amount:MultiplicativeScalarUniformVariationSpecs=MultiplicativeScalarUniformVariationSpecs(1,1)
    oxide_spots_spec:TextureOxideSpotsVariation=TextureOxideSpotsVariation()

    def sample(self,oxidized_conductor:ss.MaterialOxidizedConductorSpec, krng,*keys):
        ox = self.oxidation_amount.sample(oxidized_conductor.oxidation_amount,krng,'oxide_amount',*keys)
        ox = np.clip(ox,0,1)

        return replace(
            oxidized_conductor,
            conductor_spec = self.conductor_variation.sample(oxidized_conductor.conductor_spec, krng,'cond_spec',*keys),
            oxidation_amount = ox,
            oxide_spots_spec = self.oxide_spots_spec.sample(oxidized_conductor.oxide_spots_spec,krng,f"spot_oxidation",*keys)
        )

class DummyVar:
    def sample(self,any_spec:Any, krng,*keys):
        return any_spec
        
def zip_specifications_w_variations(specs,variations):
    # you shouldn't have more variations than specifiations
    s_n = len(specs)
    v_n = len(variations)
    if s_n > v_n:
        dummy = DummyVar()
        variations = variations  + [dummy]*(s_n - v_n)
    elif s_n < v_n:
        # throw warning maybe?
        variations = variations[:s_n]
        
    return  enumerate(zip(specs,variations))


    
@dataclass(frozen=True)
class NfaVariationParams:
    rods_variations:tuple[FuelRodVariationParams,...] = () # Variation per rod, not ideal at the moment
    grids_variations:tuple[SpacerGridVariation,...] = ()
    rods_material_variation:MaterialOxidizedConductorVariation = MaterialOxidizedConductorVariation()
    grids_material_variation:MaterialOxidizedConductorVariation = MaterialOxidizedConductorVariation()
    # Do I want to vary gloal offsets and stuff, those params are 
    # not translated to scene so no I don't want to do it now

    def sample(self,nfa_spec:ss.NfaSpec,krng,*keys):
        if nfa_spec is None:
            return nfa_spec
        return replace(
            nfa_spec,
            rods_specs = [ 
                var.sample(spec, krng,f'rod_{i}','rod_spec',*keys)
                for i,(spec,var) in zip_specifications_w_variations(nfa_spec.rods_specs, self.rods_variations)
            ],
            grids_specs = [ 
                var.sample(spec, krng,f'grid_{i}','grids_spec',*keys)
                for i,(spec,var) in zip_specifications_w_variations(nfa_spec.grids_specs, self.grids_variations)
            ],
            rods_material_spec = self.rods_material_variation.sample(
                nfa_spec.rods_material_spec,
                krng,
                'mat',
                *keys,
            ),
            grids_material_spec = self.grids_material_variation.sample(
                nfa_spec.grids_material_spec,
                krng,
                'grid_mat',
                *keys,
            )
        )

@dataclass
class EmitterVariation:
    lookat_origin_xyz:XyzUniformVariationSpecs = XyzUniformVariationSpecs()
    intensity:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    lookat_up_xyz:XyzUniformVariationSpecs = XyzUniformVariationSpecs()
    lookat_target_xyz:XyzUniformVariationSpecs=XyzUniformVariationSpecs()
    panel_width:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    panel_height:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()

    def sample(self,emitter_spec:ss.PanelEmitterSpec,krng,*keys)->ss.PanelEmitterSpec:
        new_lookat_origin_xyz = tuple(self.lookat_origin_xyz.sample(emitter_spec.lookat_origin_xyz ,krng,'lookat_origin_xyz',*keys))
        new_lookat_up_xyz = tuple(self.lookat_up_xyz.sample(emitter_spec.lookat_up_xyz ,krng,'lookat_up_xyz',*keys))
        new_lookat_target_xyz = tuple(self.lookat_target_xyz.sample(emitter_spec.lookat_target_xyz ,krng,'lookat_target_xyz',*keys))
        return replace(
            emitter_spec,
            lookat_origin_xyz = new_lookat_origin_xyz,
            intensity = self.intensity.sample(emitter_spec.intensity,krng,'intensity',*keys),
            lookat_up_xyz = new_lookat_up_xyz,
            lookat_target_xyz = new_lookat_target_xyz,
            panel_width=self.panel_width.sample(emitter_spec.panel_width, krng,'panel_width',*keys),
            panel_height=self.panel_height.sample(emitter_spec.panel_height, krng,'panel_height',*keys),
        )

@dataclass
class PerspectiveSensorVariation:
    lookat_origin_xyz:XyzUniformVariationSpecs = XyzUniformVariationSpecs()
    lookat_up_xyz:XyzUniformVariationSpecs = XyzUniformVariationSpecs()
    lookat_target_xyz:XyzUniformVariationSpecs=XyzUniformVariationSpecs()

    field_of_view:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()

    def sample(self,sensor_spec:ss.PerspectiveSensorSpec,krng,*keys)->ss.PerspectiveSensorSpec:
        new_lookat_origin_xyz = tuple(self.lookat_origin_xyz.sample(sensor_spec.lookat_origin_xyz ,krng,'lookat_origin_xyz',*keys))
        new_lookat_up_xyz = tuple(self.lookat_up_xyz.sample(sensor_spec.lookat_up_xyz ,krng,'lookat_up_xyz',*keys))
        new_lookat_target_xyz = tuple(self.lookat_target_xyz.sample(sensor_spec.lookat_target_xyz ,krng,'lookat_target_xyz',*keys))
        return replace(
            sensor_spec,
            lookat_origin_xyz = new_lookat_origin_xyz,
            lookat_up_xyz = new_lookat_up_xyz,
            lookat_target_xyz = new_lookat_target_xyz,
            field_of_view = self.field_of_view.sample(sensor_spec.field_of_view,krng,'fov',*keys),
        )
            

@dataclass
class CamRingVariationParams:
    emitter_variations:tuple[EmitterVariation,...]
    sensor_variation:tuple[PerspectiveSensorVariation,...]

    def sample(self,cam_ring_spec:ss.GenericCameraRingSpec,krng,*keys):
        return replace(
            cam_ring_spec,
            emitter_specs = [ 
                var.sample(spec, krng,f'emitter_{i}',*keys)
                for i,(spec,var) in zip_specifications_w_variations(cam_ring_spec.emitter_specs,self.emitter_variations)
            ],
            sensor_specs =  [
                var.sample(spec, krng,f'sensor_{i}',*keys)
                for i,(spec,var) in zip_specifications_w_variations(cam_ring_spec.sensor_specs,self.sensor_variation)
            ],
        )

@dataclass(frozen=True)
class EnvironmentMapVariationParams:
    intensity_scale:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    env_map_file:ChoiceVariationParams = ChoiceVariationParams()
    rot_left_right_angle:AdditiveScalarUniformVariationSpecs=AdditiveScalarUniformVariationSpecs(0,0)

    def sample(self,env_map_spec:ss.EnvironmentMapSpec, krng,*keys):
        intensity = self.intensity_scale.sample(env_map_spec.intensity_scale,krng,"intensity",*keys)
        intensity = np.maximum(intensity,0)
        return replace(
            env_map_spec,
            intensity_scale = intensity,
            env_map_file = self.env_map_file.sample("DUMMY NON USED VALUE", krng,"env_map",*keys),
            rot_left_right_angle = self.rot_left_right_angle.sample(env_map_spec.rot_left_right_angle,krng,"lr_rot",*keys),
        )

@dataclass(frozen=True)
class GlobalIlluminationVariation:
    intensity:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs(1,1)
    
    def sample(self,global_illumination_spec:ss.GlobalIlluminationSpec, krng,*keys):
        if global_illumination_spec is None:
            return None
        return replace(
            global_illumination_spec,
            intensity = self.intensity.sample(global_illumination_spec.intensity,krng,"gl_intensity",*keys),
        )

@dataclass(frozen=True)
class HeterogenousMediumVariation:
    albedo:XyzUniformVariationSpecs=XyzUniformVariationSpecs()
    hg_phase_g:MultiplicativeScalarUniformVariationSpecs=MultiplicativeScalarUniformVariationSpecs()
    scale:MultiplicativeScalarUniformVariationSpecs=MultiplicativeScalarUniformVariationSpecs()
    enabled:ChanceBoolVariation=ChanceBoolVariation()

    def sample(self,spec: ss.HeterogenousMediumSpec,krng,*keys):
        if spec is None:
            return None
        if not isinstance(spec, ss.HeterogenousMediumSpec):
            raise RuntimeError(f"Heterogenous Medium variaion canot vary specification of type {type(spec)}")

        xyz = np.array([spec.albedo.r,spec.albedo.g,spec.albedo.b])
        albedo_xyz = self.albedo.sample(xyz, krng, "albedo",*keys)
        albedo_bsdf = ss.MaterialBSDFSpec(
            r=  albedo_xyz[0],
            g=albedo_xyz[1],
            b=albedo_xyz[2],
        )
        
        return replace(
            spec,
            albedo = albedo_bsdf,
            hg_phase_g = self.hg_phase_g.sample(spec.hg_phase_g, krng ,"hg-phase_g", *keys),
            scale = self.scale.sample(spec.scale,krng,"scale",*keys),
            enabled = self.enabled.sample(None, krng,"med_enabled",*keys)
        )

@dataclass(frozen=True)
class InspectionVariationParams:
    nfa_variation:NfaVariationParams
    cam_ring_variation:CamRingVariationParams
    env_map_variation:EnvironmentMapVariationParams = EnvironmentMapVariationParams()
    global_illumination_variation:GlobalIlluminationVariation = GlobalIlluminationVariation()
    medium_variation:HeterogenousMediumVariation = HeterogenousMediumVariation()
    
    def sample(self,inspection_scene:ss.InspectionScene,krng,*keys):
        return replace(
            inspection_scene,
            nfa_spec = self.nfa_variation.sample(
                inspection_scene.nfa_spec,
                krng,
                'nfa_var',
                *keys,
            ),
            cam_ring_spec = self.cam_ring_variation.sample(
                inspection_scene.cam_ring_spec,
                krng,
                'cr_var',
                *keys,
            ),
            env_map_spec = self.env_map_variation.sample(
                inspection_scene.env_map_spec,
                krng,
                'env_map',
                *keys,
            ),
            global_illumination_spec = self.global_illumination_variation.sample(
                inspection_scene.global_illumination_spec,
                krng,
                'gl',
                *keys
            ),
            medium_spec = self.medium_variation.sample(
                inspection_scene.medium_spec,
                krng,
                'medium',
                *keys
            )
        )