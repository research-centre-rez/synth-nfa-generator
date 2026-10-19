import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo

    return Path, mo


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell
def _():
    import synthnf.assets as assets

    assert assets.assets_path.exists()
    return


@app.cell
def _():
    return


@app.cell
def _():
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm.auto import tqdm

    return imageio, np, plt, tqdm


@app.cell
def _():
    import mitsuba as mi

    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
    mi.set_log_level(mi.LogLevel.Error)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define Scene
    """)
    return


@app.cell
def _(Path):
    rod_count = 17
    rod_height = 4500

    rod_height = 500
    rod_count = 7
    output_dir = Path("/tmp/tmpframes_5")
    n_frames = 5
    return n_frames, output_dir, rod_count, rod_height


@app.cell
def _(np, rod_count, rod_height):
    import synthnf.scene_spec as ss

    grid_zs = np.linspace(0, rod_height, 10)[1:-1] - rod_height / 2
    inspection_scene = ss.InspectionScene(
        nfa_spec=ss.NfaSpec.from_shape(
            ss.RodsSquareSpec(rod_height=rod_height, column_number=rod_count),
            rods_material_spec=ss.MaterialOxidizedConductorSpec(
                conductor_spec=ss.MaterialNamedAnyConductorSpec(
                    conductor_name="custom_Zircon",
                    rough_conductor_spec=ss.MaterialRoughConductorSpec(
                        alpha_u=0.025, alpha_v=0.05
                    ),
                ),
                oxidation_amount=0.5,
                oxide_spots_spec=ss.TextureOxideSpotsSpec(
                    oxide_spots_coverage=0.0,
                    poisson_disk_radius=0.09,
                    opacity=0.075,
                    blur_sigma=0,
                ),
            ),
            grids_material_spec=ss.MaterialOxidizedConductorSpec(
                conductor_spec=ss.MaterialNamedAnyConductorSpec(
                    conductor_name="custom_Inconel",
                    rough_conductor_spec=ss.MaterialRoughConductorSpec(
                        alpha_u=0.025, alpha_v=0.05
                    ),
                ),
                oxidation_amount=0.5,
            ),
            grids=[
                ss.SpacerGridSpec(
                    z_location=z, tooth_ply_filename="g4face-spiky-flat-holed.ply"
                )
                for z in grid_zs
            ],
        ),
        # cam_ring_spec = GenericCameraRingSpec.single_cam_two_panel_light(field_of_view=35,light_intensity = 4000,light_offset=400),
        cam_ring_spec=ss.AhlbergCameraRingSpec.single_cam_four_x_lights(
            field_of_view=40, light_intensity=200, light_height_offset=0
        ),
        env_map_spec=ss.EnvironmentMapSpec(
            # env_map_file='20060121-06_hd.hdr',
            env_map_file="machine_shop_02_4k.hdr",
            intensity_scale=1.05,
        ),
        medium_spec=ss.HeterogenousMediumSpec(
            albedo=ss.MaterialBSDFSpec(r=0.3, g=0.5, b=0.5),
            hg_phase_g=0.6,
            scale=0.001,
            volume_spec=ss.MediumRandomGridVolumeSpec(
                resolution=64, cube_width=1000, heterogenity_noise_max=0.1
            ),
        ),
        # global_illumination= GlobalIlluminationSpec(intensity=1)
    )
    return (inspection_scene,)


@app.cell
def _():
    import synthnf.scene_variation as sv

    krng = sv.KeyedRNG(123)
    return (krng,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - [x] render built scene
    - [x] abstract scene representation
    - [x] aov
    - [x] variation
    - [x] env map
    - [x] extreme materials - rod is a rod even though it's made of different material
    - [x] other cam rings
    - [x] homogenous oxides (roughplastic appearance)
    - [x] heterogenous oxides
    - [ ] add clean render result?
    - [ ] thinlens sensor for depth of field
    - [x] water medium + variation
    - [ ] bend rods
    - [x] vary env map xy-plane rotation
    - [x] dropout on fuel grids - give them "enabled" flag on spec and add def bool(0<=thr< 1) that will return true if rnd<thr or something
    - [ ] there is no way to apply same variation to two different things (like calculate shift and pass it to panel width) - WORKAROUND SOLUTION:you can define your own ariation that doesn't calculate separate variations for each stule
    - [ ] dropout on fuel rods - that may not make sense, since there is another rod behind, it would just make everything harder, maybe it would work if there was only a single row
    - [x] vary grids
    - [ ] missing safe label rendering, now it's even worse
    - [ ] 2d scene plotting
    **Nope**
    - [ ] vary rendering parameters (spp, denoising etc) I will take care of this in normal training augumentation
    """)
    return


@app.cell
def _():
    import synthnf.mitsuba_layer as ml

    return (ml,)


@app.cell
def _(inspection_scene, krng, ml, n_frames, output_dir):
    mitsuba_scene = ml.MitsubaScene.from_inspection_scene(inspection_scene, krng)
    mitsuba_scene.render_frames(
        n_frames=n_frames, output_dir=output_dir, krng=krng, replace_existing=False
    )
    return


@app.cell
def _(Path, output_dir, plt):
    from synthnf.utils import load_frames_as_numpy

    frame_names = list(sorted((output_dir / "frames").glob("*.png")))

    frames = load_frames_as_numpy(Path(output_dir / "frames/"))
    print(frame_names)
    plt.title("0")
    plt.imshow(frames[0])
    plt.show()
    plt.title("2")
    plt.imshow(frames[2])
    plt.show()
    return (frames,)


@app.cell
def _():
    assert False
    return


@app.cell
def _(frames, np, plt):
    # Reversed since the cam goes bottom up
    joined = np.hstack(
        [
            frames[-1, :, 500:600],
            frames[-2, :, 500:600],
        ]
    )

    teeth_1 = 478
    teeth_2 = 449
    plt.imshow(joined)
    plt.axhline(teeth_1, alpha=0.4)
    plt.axhline(teeth_2, alpha=0.4)
    plt.ylim((400, 600))
    plt.show()

    teeth_shift = np.abs(teeth_2 - teeth_1)
    teeth_shift
    return (teeth_shift,)


@app.cell
def _(frames, imageio, np, plt, teeth_shift):
    from synthnf.utils import frames_to_oio

    oio = frames_to_oio(frames, teeth_shift)
    # plt.plot(base[0])
    # plt.show()
    plt.figure(figsize=(15, 20))
    # plt.ylim((1000,5000))
    plt.imshow(oio)
    plt.ylim((3500, 4000))
    plt.show()
    base = np.clip(oio, 0, 1)
    imageio.imwrite("/tmp/synth_oio.png", np.uint8(base * 255))
    return (base,)


@app.cell
def _(base, plt):
    plt.figure(figsize=(15, 20))
    # plt.ylim((1000,5000))
    plt.imshow(base)
    plt.ylim((3500, 4000))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Render Video from Saved Frames
    """)
    return


@app.cell
def _(Path, imageio, tqdm):
    from synthnf.utils import frames_to_video

    video_path = "/tmp/synth_video.mp4"
    frames_path = Path(
        "/home/knotek/data/kobka-fuel/jk-workspace/20261015-synth-oio-w-video/frames/"
    )
    _frs = tqdm(imageio.imread(p) for p in sorted(frames_path.glob("*.png")))
    frames_to_video(video_path, _frs)
    print(f"Saved to {video_path}")
    return


if __name__ == "__main__":
    app.run()
