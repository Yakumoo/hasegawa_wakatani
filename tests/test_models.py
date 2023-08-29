from pathlib import Path

from hasegawa_wakatani.models import (
    hasegawa_mima_pspectral_2d,
    hasegawa_wakatani_pspectral_3d,
    hasegawa_wakatani_pspectral_2d,
    hasegawa_wakatani_pspectral_1d,
    hasegawa_wakatani_findiff_1d,
    hasegawa_wakatani_findiff_2d,
)

from hasegawa_wakatani.plots import (
    visualization_2d,
    plot_components_1d,
    plot_pspectral_1d,
)

common_kwargs = {"grid_size": 64, "tf": 0.5, "video_length": 1, "video_fps": 5}

class TestMimaPspectral2D:
    def run(self, tmp_path, **kwargs):
        filename = tmp_path / "hasegawa_mima_pspectral_2d.zarr"
        hasegawa_mima_pspectral_2d(**(common_kwargs | kwargs), filename=filename)

    def test_base(self, tmp_path):
        self.run(tmp_path)
        visualization_2d(tmp_path / "hasegawa_mima_pspectral_2d.zarr")
        self.run(tmp_path, tf=2)

    def test_grids_size(self, tmp_path):
        self.run(tmp_path, grid_size=(64, 32), domain=(10, 20))

    def test_solver(self, tmp_path):
        self.run(tmp_path, solver="CrankNicolsonRK4")

class TestPspectral3D:
    def run(self, tmp_path, **kwargs):
        filename = tmp_path / "hasegawa_wakatani_pspectral_2d.zarr"
        hasegawa_wakatani_pspectral_3d(**(common_kwargs | kwargs), filename=filename)

    def test_base(self, tmp_path):
        self.run(tmp_path)
        self.run(tmp_path, tf=2)

class TestPspectral2D:
    def run(self, tmp_path, **kwargs):
        filename = tmp_path / "hasegawa_wakatani_pspectral_2d.zarr"
        hasegawa_wakatani_pspectral_2d(**(common_kwargs | kwargs), filename=filename)

    def test_base(self, tmp_path):
        self.run(tmp_path)
        visualization_2d(tmp_path / "hasegawa_wakatani_pspectral_2d.zarr")
        self.run(tmp_path, tf=2)
        
class TestPspectral1D:
    def test_base(self, tmp_path):
        filename = tmp_path / "hasegawa_wakatani_pspectral_1d.zarr"
        hasegawa_wakatani_pspectral_1d(**common_kwargs, filename=filename)
        plot_pspectral_1d(filename)
        hasegawa_wakatani_pspectral_1d(**(common_kwargs|{"tf":2}), filename=filename)

class TestFindiff1D:
    def run(self, tmp_path, **kwargs):
        filename = tmp_path / "hasegawa_wakatani_findiff_1d.zarr"
        hasegawa_wakatani_findiff_1d(**(common_kwargs|kwargs), filename=filename)

    def test_base(self, tmp_path):
        self.run(tmp_path)
        plot_components_1d(tmp_path / "hasegawa_wakatani_findiff_1d.zarr")
        self.run(tmp_path, tf=2)

    def test_periodic(self, tmp_path):
        self.run(tmp_path, boundary="periodic", acc=6)

    def test_dirichlet(self, tmp_path):
        self.run(tmp_path, boundary="dirichlet", acc=6)
   
    def test_neumann(self, tmp_path):
        self.run(tmp_path, boundary="neumann", acc=6)

    def test_force(self, tmp_path):
        self.run(tmp_path, boundary="force 0.001", acc=6)

class TestFindiff2D:
    def run(self, tmp_path, **kwargs):
        filename = tmp_path / "hasegawa_wakatani_findiff_1d.zarr"
        hasegawa_wakatani_findiff_2d(**(common_kwargs|kwargs), filename=filename)

    def test_base(self, tmp_path):
        self.run(tmp_path)
        visualization_2d(tmp_path / "hasegawa_wakatani_findiff_1d.zarr")
        self.run(tmp_path, tf=2)

    def test_arakawa_false(self, tmp_path):
        self.run(tmp_path, arakawa=False)

    def test_periodic(self, tmp_path):
        self.run(tmp_path, boundary="periodic", acc=6)

    def test_dirichlet(self, tmp_path):
        self.run(tmp_path, boundary="dirichlet", acc=6)
   



