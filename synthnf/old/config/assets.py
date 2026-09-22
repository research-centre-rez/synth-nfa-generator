import pathlib

assets_path = (
    pathlib.Path(
        __file__,
    ).parent
    / "../../assets"
)


def get_asset_path(filename):
    return assets_path / filename
