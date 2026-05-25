import synthnf.config.assets as assets
import synthnf.config.defaults as defaults
import synthnf.materials as mat
import synthnf.geometry.spacer_grid_mesh as sgm
import synthnf.geometry.fuel_rods_mesh as frm
import synthnf.geometry.rod_tips_mesh as rtm
import synthnf.geometry.curves as curves
import synthnf.io as io


def spacer_grid(
    tooth_ply=None,
    rods_per_face=None,
    rod_width_mm=None,
    gap_width_mm=None,
    material=None,
):
    rods_per_face = rods_per_face or defaults.blueprints.fa.rods_per_face
    rod_width_mm = rod_width_mm or defaults.blueprints.fuel_rod.width_mm
    gap_width_mm = gap_width_mm or defaults.blueprints.fuel_rod.gap_mm
    material = material or mat.inconel_blend()

    if tooth_ply is None:
        tooth_path = str(assets.get_asset_path("default_tooth.ply"))
        tooth_ply = io.read_ply(tooth_path)

        # assuming the default tooth has size of 1
        # then rescale to fit rod+gap
        scale = rod_width_mm + gap_width_mm
        tooth_ply["points"]["x"] = tooth_ply["points"]["x"] * scale
        tooth_ply["points"]["y"] = tooth_ply["points"]["y"] * scale
        tooth_ply["points"]["z"] = tooth_ply["points"]["z"] * scale

    grid_mesh = sgm.create_grid_mesh_from_tooth(
        tooth_ply, rods_per_face, rod_width_mm, gap_width_mm
    )

    grid_mesh_path = io.save_mesh_to_temp(grid_mesh)

    return {"type": "ply", "filename": grid_mesh_path, "material": material}


def fuel_rods(
    rod_centers=None,
    rod_width_mm=None,
    rod_height_mm=None,
    gap_width_mm=None,
    curves_rod=None,
    material=None,
    n_textures=None,
    z_displacement=None,
):
    rod_width_mm = rod_width_mm or defaults.blueprints.fuel_rod.width_mm
    rod_height_mm = rod_height_mm or defaults.blueprints.fuel_rod.height_mm
    gap_width_mm = gap_width_mm or defaults.blueprints.fuel_rod.gap_mm
    rod_centers = (
        rod_centers
        if rod_centers is not None
        else frm.generate_rod_centers(
            rod_width_mm=rod_width_mm, rod_gap_mm=gap_width_mm
        )
    )

    if curves_rod is None:
        c = curves.straight_curve(to_z=rod_height_mm)
        curves_rod = [c] * len(rod_centers)

    mesh = frm.create_fa_mesh(
        curves_rod,
        rod_centers,
        rod_width_mm,
        rod_height_mm,
        num_textures=n_textures,
        z_displacement=z_displacement,
    )

    mesh_path = io.save_mesh_to_temp(mesh)
    material = material or mat.zirconium_blend()

    return {"type": "ply", "filename": mesh_path, "material": material}

# def fuel_rods_ind(
#     rod_centers=None,
#     rod_width_mm=None,
#     rod_height_mm=None,
#     gap_width_mm=None,
#     curves_rod=None,
#     material=None,
#     n_textures=None,
#     z_displacement=None,
# ):
#     rod_width_mm = rod_width_mm or defaults.blueprints.fuel_rod.width_mm
#     rod_height_mm = rod_height_mm or defaults.blueprints.fuel_rod.height_mm
#     gap_width_mm = gap_width_mm or defaults.blueprints.fuel_rod.gap_mm
#     rod_centers = (
#         rod_centers
#         if rod_centers is not None
#         else frm.generate_rod_centers(
#             rod_width_mm=rod_width_mm, rod_gap_mm=gap_width_mm
#         )
#     )

#     if curves_rod is None:
#         c = curves.straight_curve(to_z=rod_height_mm)
#         curves_rod = [c] * len(rod_centers)

#     mesh = frm.create_fa_mesh(
#         curves_rod,
#         rod_centers,
#         rod_width_mm,
#         rod_height_mm,
#         num_textures=n_textures,
#         z_displacement=z_displacement,
#     )

#     mesh_path = io.save_mesh_to_temp(mesh)
#     material = material or mat.zirconium_blend()

#     return {"type": "ply", "filename": mesh_path, "material": material}


def rod_tips(
    rod_width_mm=None,
    rod_centers=None,
    tip_path=None,
    material=None,
    z_displacement=None,
):
    tip_path = tip_path or assets.get_asset_path("tip_wta.ply")
    rod_centers = rod_centers if rod_centers is not None else frm.generate_rod_centers()
    rod_width_mm = rod_width_mm or defaults.blueprints.fuel_rod.width_mm
    material = material or mat.zirconium_blend()

    ply = io.read_ply(tip_path)
    scale = rod_width_mm
    mesh = rtm.create_mesh(ply, scale, rod_centers, z_displacement=z_displacement)

    grid_mesh_path = io.save_mesh_to_temp(mesh)

    return {"type": "ply", "filename": str(grid_mesh_path), "material": material}


def rod_butts(
    rod_width_mm=None,
    rod_centers=None,
    butt_path=None,
    material=None,
    z_displacement=None,
):
    butt_path = butt_path or assets.get_asset_path("butt_wta.ply")
    rod_centers = rod_centers if rod_centers is not None else frm.generate_rod_centers()
    rod_width_mm = rod_width_mm or defaults.blueprints.fuel_rod.width_mm
    material = material or mat.zirconium_blend()

    ply = io.read_ply(butt_path)
    scale = rod_width_mm
    mesh = rtm.create_mesh(ply, scale, rod_centers, z_displacement=z_displacement)

    grid_mesh_path = io.save_mesh_to_temp(mesh)

    return {"type": "ply", "filename": str(grid_mesh_path), "material": material}
