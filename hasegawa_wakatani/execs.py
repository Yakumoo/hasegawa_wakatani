from models import (
    hasegawa_wakatani_spectral_1D, hasegawa_wakatani_finite_difference_1D
)


def compare_1d(
    grid_size=1024,
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
    common_parameters = locals()
