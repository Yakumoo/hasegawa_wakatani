from pathlib import Path

from hasegawa_wakatani.models import (
    hasegawa_mima_pspectral_2d,
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
    
    def test_base(self, tmp_path):
        filename = tmp_path / "hasegawa_mima_pspectral_2d.zarr"
        hasegawa_mima_pspectral_2d(**common_kwargs, filename=filename, modified=False)
        hasegawa_mima_pspectral_2d(**common_kwargs, filename=filename, modified=True)
        visualization_2d(filename)

class TestPspectral2D:
    def test_base(self, tmp_path):
        filename = tmp_path / "hasegawa_wakatani_pspectral_2d.zarr"
        hasegawa_wakatani_pspectral_2d(**common_kwargs, filename=filename, modified=False)
        hasegawa_wakatani_pspectral_2d(**common_kwargs, filename=filename, modified=True)
        visualization_2d(filename)
        
class TestPspectral1D:
    def test_base(self, tmp_path):
        filename = tmp_path / "hasegawa_wakatani_pspectral_1d.zarr"
        hasegawa_wakatani_pspectral_1d(**common_kwargs, filename=filename)
        plot_pspectral_1d(filename)

class TestFindiff1D:
    def run(self, tmp_path, boundary, acc=2):
        filename = tmp_path / "hasegawa_wakatani_findiff_1d.zarr"
        hasegawa_wakatani_findiff_1d(**common_kwargs, filename=filename, boundary=boundary, acc=acc)

    def test_periodic(self, tmp_path):
        self.run(tmp_path, "periodic")
        self.run(tmp_path, "periodic", acc=6)
        plot_components_1d(tmp_path / "hasegawa_wakatani_findiff_1d.zarr")

    def test_dirichlet(self, tmp_path):
        self.run(tmp_path, "dirichlet")
        self.run(tmp_path, "dirichlet", acc=6)
   
    def test_neumann(self, tmp_path):
        self.run(tmp_path, "neumann")

    def test_force(self, tmp_path):
        self.run(tmp_path, "force 0.001")

class TestFindiff2D:
    def run(self, tmp_path, **kwargs):
        filename = tmp_path / "hasegawa_wakatani_findiff_1d.zarr"
        hasegawa_wakatani_findiff_2d(**common_kwargs, filename=filename, **kwargs)

    def test_periodic(self, tmp_path):
        self.run(tmp_path, boundary="periodic")
        self.run(tmp_path, boundary="periodic", acc=6, arakawa=False)
        self.run(tmp_path, boundary="periodic", acc=6, arakawa=True)
        visualization_2d(tmp_path / "hasegawa_wakatani_findiff_1d.zarr")

    def test_dirichlet(self, tmp_path):
        self.run(tmp_path, boundary="dirichlet")
        self.run(tmp_path, boundary="dirichlet", acc=6, arakawa=False)
        self.run(tmp_path, boundary="dirichlet", acc=6, arakawa=True)
   
    def test_neumann(self, tmp_path):
        self.run(tmp_path, boundary="neumann")


