from dataclasses import dataclass, astuple
from typing import Any
import numpy as np

MaterialSpec = object

@dataclass(frozen=True)
class MaterialRoughConductorSpec:
    alpha_u:float = .1
    alpha_v:float = .1
        
@dataclass(frozen=True, slots=True)
class MaterialBSDFSpec:
    r:float = 1
    g:float = 0
    b:float = 1

    def to_rgb_tuple(self):
        return self.r,self.g,self.b

@dataclass(frozen=True)
class MaterialOxideLayer:
    gray_scale:float = .3
    
@dataclass
class TextureOxideSpotsSpec:
    noise_texture_zoom:float=10
    min_oxide_size_px:int = 16
    max_oxide_size_px:int = 22
    oxide_spots_coverage:float = 0
    poisson_disk_radius:float = .04


@dataclass(frozen=True)
class MaterialNamedAnyConductorSpec:
    conductor_name:str = "custom_Zircon"
    rough_conductor_spec:MaterialRoughConductorSpec|None = None
    
@dataclass(frozen=True)
class MaterialOxidizedConductorSpec:
    conductor_spec:MaterialNamedAnyConductorSpec = MaterialNamedAnyConductorSpec()
    oxidation_spec:MaterialOxideLayer = MaterialOxideLayer()
    oxidation_amount:float = .1
    oxide_spots_specs:tuple[TextureOxideSpotsSpec,...] = ()


@dataclass(frozen=True)
class MediumRandomGridVolumeSpec:
    resolution:int = 64
    density:float = 0.03 #water
    heterogenity_noise_max:float = 0.1
    smooth_sigma:float = 5 # in respect to resolution
    # at this point it's just a cube
    cube_width:int = 1
    centered_to_origin:bool=True
    
@dataclass(frozen=True)
class HeterogenousMediumSpec:
    # params represent murky water
    albedo:MaterialBSDFSpec=MaterialBSDFSpec(r=.8,g=.85,b=.95) 
    hg_phase_g:float=.75
    scale:float = 1
    volume_spec:MediumRandomGridVolumeSpec=MediumRandomGridVolumeSpec()
    enabled:bool = True
    

@dataclass(frozen=True)
class FuelRodSpec:
    xyz:tuple[float,float,float]
    radius:float
    height:float


@dataclass(frozen=True)
class RodsHexagonSpec:
    row_number:int = 11
    offset:int = 12.75
    rod_height:int = 3800
    rod_radius:int = 4.55
    center_z:int = 0

    def collect_rods(self)->tuple[FuelRodSpec]:
        if self.edge_count == 0:
            return ()
            
        rod_centers = array_hexagon_grid(self.row_number, self.offset)
        xyzs = [ (x,y,self.center_z) for x,y in rod_centers]
        return tuple([ 
            FuelRodSpec(
                xyz = np.array(xyz),
                radius=self.rod_radius,
                height=self.rod_height
            ) for xyz in xyzs
        ])
    
@dataclass
class RodsSquareSpec:
    column_number:int = 17
    row_number:int = 17
    offset:int = 12.54
    rod_height:int = 3800
    rod_radius:int = 4.75
    center_z:int = 0

    def collect_rods(self)->tuple[FuelRodSpec]:
        if self.column_number == 0 or self.row_number == 0:
            return ()
        rod_centers = array_rectangle_grid(
            column_number= self.column_number,
            row_number=self.row_number,
            offset_rows=self.offset,
            offset_columns=self.offset
        )
        xyzs = [ (x,y,self.center_z) for x,y in rod_centers]
        return tuple([ 
            FuelRodSpec(
                xyz = xyz,
                radius=self.rod_radius,
                height=self.rod_height
            ) for xyz in xyzs
        ])

@dataclass(frozen=True)
class SpacerGridSpec:
    #tooth_type - plain and complex
    z_location:float
    enabled:bool = True
    tooth_ply_filename:str = "g4face.ply"
    
    
@dataclass
class NfaSpec:
    rods_shape_spec: RodsHexagonSpec|RodsSquareSpec
    rods_specs:tuple[FuelRodSpec,...] = ()
    grids_specs:tuple[SpacerGridSpec,...] = ()
    rods_material_spec:MaterialSpec|None= None
    grids_material_spec:MaterialSpec|None= None
    
    @staticmethod
    def from_shape(rods_shape_spec, rods_material_spec,grids:list[SpacerGridSpec] = [], grids_material_spec:MaterialSpec|None = None):
        rods_specs= rods_shape_spec.collect_rods()
        return NfaSpec(
            rods_shape_spec,
            rods_specs = rods_specs,
            rods_material_spec= rods_material_spec,
            grids_material_spec = grids_material_spec,
            grids_specs = grids
        )

    def single_rod(rod_radius,rod_height,rods_material_spec):
        dummy_shape = RodsSquareSpec(1,1,0, rod_height,rod_radius)
        return NfaSpec(
            dummy_shape,
            rods_specs = dummy_shape.collect_rods(),
            rods_material_spec = rods_material_spec
        )
        
    

@dataclass
class PerspectiveSensorSpec:
    lookat_origin_xyz:tuple[float,float,float]
    lookat_up_xyz:tuple[float,float,float] = (0,0,1)
    lookat_target_xyz:tuple[float,float,float] = (0,0,0)
    field_of_view:float = 28
    resolution_x:int = 800
    resolution_y:int = 600
    far_clip:int=100_000

@dataclass
class PanelEmitterSpec:
    lookat_origin_xyz:tuple[float,float,float]
    intensity:float = 200
    lookat_up_xyz:tuple[float,float,float] = (0,0,1)
    lookat_target_xyz:tuple[float,float,float] = (0,0,0)
    panel_width:int = 20
    panel_height:int = 50


@dataclass
class GenericCameraRingSpec():
    sensor_specs:list[PerspectiveSensorSpec]
    emitter_specs:list[PanelEmitterSpec]

    @staticmethod
    def single_cam_two_panel_light(
        ring_radius:float = 430, 
        light_offset = 228,
        ring_z:float = 0,
        field_of_view:float = 25.88, 
        light_intensity = 1000,
    ):
        x = 0
        y = -ring_radius
        z = ring_z
        
        emitter_left = PanelEmitterSpec(lookat_origin_xyz = (x-light_offset/2,y,z),intensity=light_intensity)
        emitter_right = PanelEmitterSpec(lookat_origin_xyz = (x+light_offset/2,y,z),intensity=light_intensity)
        main_cam = PerspectiveSensorSpec(lookat_origin_xyz= (x,y,z),field_of_view= field_of_view)
        return GenericCameraRingSpec(
            sensor_specs = [main_cam],
            emitter_specs = [emitter_left,emitter_right]
        )

class AhlbergCameraRingSpec():
    sensor_specs:list[PerspectiveSensorSpec]
    emitter_specs:list[PanelEmitterSpec]

    @staticmethod
    def single_cam_four_x_lights(
        square_width:float = 900, 
        ring_z:float = 0,
        field_of_view:float = 23, 
        light_intensity = 1000,
        light_height_offset = 50,
        resolution_x = 1600,
        resolution_y= 900
    ):
        hf = square_width/2

        # make lights closer
        light_shrink_factor = .6
        hfs = hf*light_shrink_factor
        
        emmiter_origins = [
            (-hfs,-hfs),
            (-hfs,hfs),
            (hfs,-hfs),
            (hfs,hfs),
        ]
        emmiters = [
            PanelEmitterSpec(
                lookat_origin_xyz = (x,y,ring_z+light_height_offset),
                intensity=light_intensity,
                panel_height=150
            )
            for x,y in emmiter_origins
        ]
        
        main_cam = PerspectiveSensorSpec(
            lookat_origin_xyz= (0,-hf,ring_z),
            field_of_view= field_of_view,
            resolution_x = resolution_x,
            resolution_y=resolution_y,
            
        )
        return GenericCameraRingSpec(
            sensor_specs = [main_cam],
            emitter_specs = emmiters
        )
    
@dataclass
class GlobalIlluminationSpec:
    intensity:float

@dataclass
class EnvironmentMapSpec:
    intensity_scale:float = 1
    env_map_file:str = 'wells_gallery.hdr'
    rot_left_right_angle:float = 0
    
@dataclass
class InspectionScene:
    cam_ring_spec:GenericCameraRingSpec
    nfa_spec:NfaSpec|None = None
    env_map_spec:EnvironmentMapSpec|None = None 
    global_illumination_spec:GlobalIlluminationSpec|None = None
    medium_spec:HeterogenousMediumSpec|None = None

def array_rectangle_grid(
    column_number,
    row_number,
    offset_rows=1,
    offset_columns=1,
    center = True
):
    x = np.repeat(np.arange(column_number)[None],row_number,axis=0)
    y = np.repeat(np.arange(row_number)[None],column_number,axis=0)
    # H,W,2
    mesh = np.dstack([x,y.T]).astype(np.float32)
    # H*W,2 rowwise
    mesh_2d = np.vstack(mesh)
    if center:
        mesh_2d -= np.mean(mesh_2d,axis=0)
    return  mesh_2d * [offset_rows,offset_columns]

def array_hexagon_grid(edge_count, offset = 1,center=True):
    pythagorean_factor = np.sqrt(3) / 2
    rod_centers = np.zeros((0, 2))

    from_layer = -1
    for layer in range(edge_count - 1, from_layer, -1):
        if layer == 0:
            rod_centers = np.vstack([rod_centers, [[0, 0]]])
            break

        side_width = offset * layer
        corners = [
            [-side_width / 2, side_width * pythagorean_factor],  # 11 oclock,
            [side_width / 2, side_width * pythagorean_factor],  # 1 oclock
            [side_width, 0],  # 3 oclock
            [side_width / 2, -side_width * pythagorean_factor],  # 5 oclock
            [-side_width / 2, -side_width * pythagorean_factor],  # 7 oclock,
            [-side_width, 0],  # 9 oclock
        ]

        rc = []
        for c1, c2 in zip(corners, corners[1:] + [corners[0]]):
            c1_x, c1_y = c1
            c2_x, c2_y = c2

            xx = np.linspace(c1_x, c2_x, layer, endpoint=False)
            yy = np.linspace(c1_y, c2_y, layer, endpoint=False)

            rc.append(np.stack([xx, yy]).T)

        rc = np.vstack(rc)
        rod_centers = np.vstack([rod_centers, rc])
    if center:
        rod_centers -= np.mean(rod_centers,axis=0)
    return rod_centers 