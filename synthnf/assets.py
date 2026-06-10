from pathlib import Path

assets_path = (
    Path(
        __file__,
    ).parent
    / "../../assets"
)

mitsuba_materials_w_description = {
    "custom_Zircon":"this is my custom zirconium",
    "custom_Porcelain":"custom porcelain (some oxides look like that)",
    "a-C": "Amorphous carbon",
    "Na_palik": "Sodium",
    "Ag": "Silver",
    "Nb": "Niobium",
    "Al": "Aluminium",
    "Ni_palik": "Nickel",
    "AlAs": "Cubic aluminium arsenide",
    "Rh": "Rhodium",
    "AlSb": "Cubic aluminium antimonide",
    "Se": "Selenium",
    "Au": "Gold",
    "SiC": "Hexagonal silicon carbide",
    "Be": "Polycrystalline beryllium",
    "SnTe": "Tin telluride",
    "Cr": "Chromium",
    "Ta": "Tantalum",
    "CsI": "Cubic caesium iodide",
    "Te": "Trigonal tellurium",
    "Cu": "Copper",
    "ThF4": "Polycryst. thorium (IV) fluoride",
    "Cu2O": "Copper (I) oxide",
    "TiC": "Polycrystalline titanium carbide",
    "CuO": "Copper (II) oxide",
    "TiN": "Titanium nitride",
    "d-C": "Cubic diamond",
    "TiO2": "Tetragonal titan. dioxide",
    "Hg": "Mercury",
    "VC": "Vanadium carbide",
    "HgTe": "Mercury telluride",
    "V_palik": "Vanadium",
    "Ir": "Iridium",
    "VN": "Vanadium nitride",
    "K": "Polycrystalline potassium",
    "W": "Tungsten",
    "Li": "Lithium",
    "MgO": "Magnesium oxide",
    "Mo": "Molybdenum",
}
mitsuba_materials = tuple(mitsuba_materials_w_description.keys())

env_map_names = (
    "20100905-21_hdr.hdr",
    "future_restaurant.hdr",
    "garage4_hd.hdr",
    "metro_the_pit.hdr",
    #"tabbaco_plant.hdr", # too dark in default
    "techgate_overcast.hdr", # this one is good
    "tungsten_evening.hdr",
    "tunnel_machine.hdr",
    "uprat5_hd.hdr",
    "vienna_metro.hdr",
    "vienna_station_interior.hdr",
    "vienna_station_overcast.hdr",
    "wells_gallery.hdr",
    "machine_shop_02_4k.hdr",
)


def get_asset_path(filepath):
    p = Path('..')/'assets'/filepath
    return str(p.absolute())
