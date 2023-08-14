from pathlib import Path
import shutil

import xarray as xr

from models import (
    hasegawa_wakatani_pspectral_1d, hasegawa_wakatani_findiff_1d
)
from plots import (pcolor_compare_1d, plot_profiles_compare_1d)


def compare_1d(
    grid_size=682,
    domain=50,
    C=1,
    κ=1,
    Dx=1e-4,
    Dy=1e-4,
    Dz=1e-5,
    νx=1e-2,
    νy=1e-4,
    νz=1e-5,
    tf=300,
    seed=42,
    solver="Dopri8",
    atol=1e-10,
    rtol=1e-10,
    video_length=10,
    video_fps=100,
):
    """Run and plot the single poloidal model

    2 simulations are runned with the same parameters for comparison:
    pseudo-spectral method and finite-difference method
    The script outputs workspace/compare_1d/compare_1d.zarr and 2 plots
    """

    parameters = locals()
    path = Path(__file__).parents[1] / "workspace" / "compare_1d"
    path.mkdir(parents=True, exist_ok=True)
    pspectral_filename = path / "pspectral_1d.zarr"
    findiff_filename = path / "findiff_1d.zarr"
    hasegawa_wakatani_pspectral_1d(**parameters, filename=pspectral_filename)
    hasegawa_wakatani_findiff_1d(
        **parameters, acc=10, boundary="periodic", filename=findiff_filename
    )
    shutil.rmtree(pspectral_filename, ignore_errors=True)
    pspectral_filename = pspectral_filename.with_stem(
        pspectral_filename.stem + "_decomposed"
    )
    with xr.open_dataarray(pspectral_filename, engine="zarr") as da_spectral:
        da_spectral = da_spectral.expand_dims(dim={"method": ["pspectral"]})
    with xr.open_dataarray(findiff_filename, engine="zarr") as da_findiff:
        da_findiff = da_findiff.expand_dims(dim={"method": ["findiff"]})
    da = xr.concat([da_spectral, da_findiff], dim="method")
    da.to_zarr(path / "compare_1d.zarr", mode="w")
    shutil.rmtree(pspectral_filename, ignore_errors=True)
    shutil.rmtree(findiff_filename, ignore_errors=True)

    pcolor_compare_1d(da, path)
    plot_profiles_compare_1d(da, path)
