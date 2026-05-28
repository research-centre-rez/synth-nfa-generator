from typing import Any
import hashlib
from dataclasses import replace
import itertools
import synthnf.scene_spec as ss
from dataclasses import dataclass, replace
import numpy as np

class KeyedRNG:
    def __init__(self, root_seed: int):
        self.root_seed = root_seed

    def unit(self, *keys) -> float:
        digest_size = 16
        x = stable_hash(
            self.root_seed, *keys,
            digest_size=digest_size
        )
        return x / (1 << (8 * digest_size))

    def uniform(self, low,high, *keys) -> float:
        u = self.unit(*keys)
        return low + u * (high - low)

    def choice(self, options,krng,*keys):
        idx = int(krng.uniform(0,len(options),*keys))
        return options[idx]

    # def uniform_list(self,specs,*keys) -> list[float]:
    #     return [self.uniform(spec,i,*keys) for i,spec in enumerate(specs)]

    # def normal(self, spec, *keys) -> float:
    #     u1 = self.unit(*keys, 0)
    #     u2 = self.unit(*keys, 1)
    #     z = math.sqrt(-2.0 * math.log(max(u1, 1e-12))) * math.cos(2 * math.pi * u2)
    #     x = spec.mean + spec.std * z
    #     return clamp(x, spec.min_value, spec.max_value)


def stable_hash(*parts: Any, digest_size: int = 16) -> int:
    h = hashlib.blake2b(digest_size=digest_size)

    for part in parts:
        data = repr(part).encode("utf-8")
        h.update(len(data).to_bytes(4, "little"))
        h.update(data)

    return int.from_bytes(h.digest(), "little")


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
class AdditiveXyzUniformVariationSpecs:
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
    xyz:AdditiveXyzUniformVariationSpecs=AdditiveXyzUniformVariationSpecs()
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
class MaterialOxidizedConductorVariation:
    conductor_variation:MaterialConductorVariation = MaterialConductorVariation()
    oxidation_amount:MultiplicativeScalarUniformVariationSpecs=MultiplicativeScalarUniformVariationSpecs(1,1)
    #TODO oxidation_variation MaterialOxidizedConductor().oxidation_spec 
    #TODO heterogenous

    def sample(self,oxidized_conductor:ss.MaterialOxidizedConductor, krng,*keys):
        ox = self.oxidation_amount.sample(oxidized_conductor.oxidation_amount,krng,'oxide_amount',*keys)
        ox = np.clip(ox,0,1)
        
        return replace(
            oxidized_conductor,
            conductor_spec = self.conductor_variation.sample(oxidized_conductor.conductor_spec, krng,'cond_spec',*keys),
            oxidation_amount = ox,
            #oxidation_spec=
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
    rods_material_variation:MaterialOxidizedConductorVariation = MaterialOxidizedConductorVariation()
    # Do I want to vary gloal offsets and stuff, those params are 
    # not translated to scene so no I don't want to do it now

    def sample(self,nfa_spec:ss.NfaSpec,krng,*keys):
        return replace(
            nfa_spec,
            rods_specs = [ 
                var.sample(spec, krng,f'rod_{i}','rod_spec',*keys)
                for i,(spec,var) in zip_specifications_w_variations(nfa_spec.rods_specs, self.rods_variations)
            ],
            rods_material_spec = self.rods_material_variation.sample(nfa_spec.rods_material_spec,krng,'mat',*keys)
        )

@dataclass
class EmitterVariation:
    lookat_origin_xyz:AdditiveXyzUniformVariationSpecs = AdditiveXyzUniformVariationSpecs()
    intensity:MultiplicativeScalarUniformVariationSpecs = MultiplicativeScalarUniformVariationSpecs()
    lookat_up_xyz:AdditiveXyzUniformVariationSpecs = AdditiveXyzUniformVariationSpecs()
    lookat_target_xyz:AdditiveXyzUniformVariationSpecs=AdditiveXyzUniformVariationSpecs()
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
    lookat_origin_xyz:AdditiveXyzUniformVariationSpecs = AdditiveXyzUniformVariationSpecs()
    lookat_up_xyz:AdditiveXyzUniformVariationSpecs = AdditiveXyzUniformVariationSpecs()
    lookat_target_xyz:AdditiveXyzUniformVariationSpecs=AdditiveXyzUniformVariationSpecs()

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
class InspectionVariationParams:
    nfa_variation:NfaVariationParams
    cam_ring_variation:CamRingVariationParams
    env_map_variation:EnvironmentMapVariationParams = EnvironmentMapVariationParams()
    global_illumination_variation:GlobalIlluminationVariation = GlobalIlluminationVariation()
    
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
            )
        )