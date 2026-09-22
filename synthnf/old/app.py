import mitsuba as mi

# this must be the first thing to run
mi.set_variant("cuda_ad_rgb")  # noqa: E402

import pathlib  # noqa: E402
import argparse  # noqa: E402
import logging  # noqa: E402
import synthnf.inspection.video as vid  # noqa: E402
import synthnf.io as io  # noqa: E402
import synthnf.simulation as simu # noqa: E402
import synthnf.inspection as ins # noqa: E402

# setup logging
mi.set_log_level(mi.LogLevel.Error)
logging.basicConfig()
logger = logging.getLogger("synthnf")
logger_re = logging.getLogger("synthnf.renderer")


def pop_with_def(dict_,key,default):
    if key in dict_:
        return dict_.pop(key)
    return default


def create_inspection(out_folder, params_dict, n_frames, force):
    if not force and out_folder.exists():
        raise Exception(
            f"Output folder already exists. Use '--force' flag to override behavior. Folder:{out_folder.absolute()}"
        )

    swing_cam_above_mm = pop_with_def(params_dict,"swing_cam_above_mm", 0)
    cam_roll_deg = pop_with_def(params_dict,"cam_roll_deg", 0)
    simulation_model = simu.RandomParameters(**params_dict)
    logger.info("Generating inspection scene with seed %d", simulation_model.seed)
    # inspection = ins.SwingSupressingInspection(simulation_model, swing_cam_above_mm)
    swing_cam_height_mm = 10_000
    swing_cam_dist_mm = 0
    # inspection = ins.SwingSupressionVerticalCamInspection(
    #     simulation_model,
    #     swing_cam_height_mm,
    #     swing_cam_dist_mm,
    #     fov=10,
    #     cam_roll_deg=cam_roll_deg,
    # )
    inspection = ins.FAInspection(simulation_model)
    vid.run_inspection_all_sides(out_folder, inspection, n_frames)


def setup_argparse():
    parser = argparse.ArgumentParser(description="Generate FA inspection videos")

    parser.add_argument(
        "destination",
        help="Folder to which videos will be genereated.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--config",
        help="Folder to which videos will be genereated.",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--frames_number",
        help="The number of frames each video should have",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Generation proceeds even if the destination folder already exists",
    )
    return parser


if __name__ == "__main__":
    parser = setup_argparse()
    args = parser.parse_args()

    # todo log level through argparse
    logger.setLevel(logging.INFO)
    mi.set_log_level(mi.LogLevel.Error)

    params_dict = io.load_json(args.config)

    create_inspection(args.destination, params_dict, args.frames_number, args.force)
