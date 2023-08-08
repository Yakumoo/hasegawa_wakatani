from typing import Callable, Optional, Tuple, Union, Any, Sequence
import dataclasses
from pathlib import Path

import xarray as xr

import jax
import jax.numpy as jnp
from jax.experimental import sparse

import jax_cfd.base.grids as grids
from jax_cfd.spectral import time_stepping
from diffrax import ODETerm

from utils import (
    brick_wall_filter_2d,
    make_hermitian,
    init_fields_fourier_2d,
    process_params_2d,
    process_boundary,
    fourier_to_real,
    real_to_fourier,
    get_terms,
    rfft_mesh,
    find_ky,
    simulation_base,
    diff_matrix,
    open_with_vorticity,
    get_padded_shape,
)


@dataclasses.dataclass
class HasegawaWakataniSpectral2D(time_stepping.ImplicitExplicitODE):
    """Breaks the Hasegawa Wakatani equation into implicit and explicit parts.
    Implicit parts are the linear terms and explicit parts are the non-linear
    terms.
    Attributes:
        grid: underlying grid of the process
        κ: gradient of the density
        ν: kinematic viscosity, strength of the diffusion term
        D: diffusion coefficient
        C: adiabatic coefficient
    """

    grid: grids.Grid
    C: float = 1
    Dx: float = 1e-4
    Dy: float = 1e-4
    Dz: float = 1e-4
    νx: float = 1e-4
    νy: float = 1e-4
    νz: float = 1e-4
    κ: float = 1

    def __post_init__(self):
        padded_grid = grids.Grid(
            shape=get_padded_shape(self.grid.shape), domain=self.grid.domain
        )
        filter_ = brick_wall_filter_2d(*self.grid.shape, nyquist=True)
        ks = rfft_mesh(padded_grid)
        self.kx, self.ky = ks * filter_
        kx2, ky2 = jnp.square(self.kx), jnp.square(self.ky)
        self.k2 = kx2 + ky2
        # use this one for division
        self.k2_div = 1 / jnp.square(ks).sum(0).at[0, 0].set(1)
        self.k2_div = self.k2_div.at[0, 0].set(0)

        self.linear_term = jnp.empty((*self.k2.shape, 2, 2), dtype=complex)
        self.linear_term = (
            self.linear_term.at[:, :, 0, 0].set(
                -self.C * self.k2_div - self.νx * kx2 - self.νy * ky2
            )
            .at[:, :, 0, 1].set(self.C * self.k2_div)
            .at[:, :, 1, 0].set(-1j * self.ky * self.κ + self.C)
            .at[:, :, 1, 1].set(-self.C - self.Dx * kx2 - self.Dy * ky2)
            .at[0, 0].set(0)
            # zonal flows
            .at[:, 0, 0, 0].set(-self.νz)
            .at[:, 0, 0, 1].set(0)
            .at[:, 0, 1, 0].set(0)
            .at[:, 0, 1, 1].set(-self.Dz)
        ) * filter_[..., None, None] # yapf: disable

        self.linear_term = make_hermitian(self.linear_term)

    def explicit_terms(self, ŷ):
        φh, nh = jnp.moveaxis(make_hermitian(ŷ.view(dtype=complex)), -1, 0)

        dφdx, dφdy, dndx, dndy, dωdx, dωdy = jnp.fft.irfft2(
            1j * jnp.array([
                self.kx * φh,
                self.ky * φh,
                self.kx * nh,
                self.ky * nh,
                -self.kx * self.k2 * φh,
                -self.ky * self.k2 * φh,
            ]),
            axes=(1, 2), norm="forward"
        )

        dnh, dφh = jnp.fft.rfft2(
            jnp.array([dφdx * dndy - dφdy * dndx, dφdx * dωdy - dφdy * dωdx]),
            axes=(1, 2), norm="forward"
        )
        term = make_hermitian(jnp.stack((dφh * self.k2_div, -dnh), axis=-1))

        return term.view(dtype=float)

    def implicit_terms(self, ŷ):
        term = self.linear_term @ ŷ.view(dtype=complex)[..., None]
        return make_hermitian(term.squeeze()).view(dtype=float)

    def implicit_solve(self, ŷ, time_step):
        inv = jnp.linalg.inv(jnp.eye(2) - time_step * self.linear_term)
        term = inv @ ŷ.view(dtype=complex)[..., None]
        return term.squeeze().view(dtype=float)


@dataclasses.dataclass
class HasegawaMimaSpectral2D(time_stepping.ImplicitExplicitODE):
    """Breaks the Hasegawa Mima equation into implicit and explicit parts.
    Implicit parts are the linear terms and explicit parts are the non-linear
    terms.
    Attributes:
        grid: underlying grid of the process
        ν: kinematic viscosity, strength of the diffusion term
    """

    grid: grids.Grid
    ν: float
    νz: float
    κ: float
    force_amplitude: float
    force_ky: float
    force_σ: float
    seed: int

    def __post_init__(self):
        padded_grid = grids.Grid(
            shape=get_padded_shape(self.grid.shape), domain=self.grid.domain
        )
        filter_ = brick_wall_filter_2d(*self.grid.shape, nyquist=True)
        self.kx, self.ky = rfft_mesh(padded_grid) * filter_
        kx2, ky2 = jnp.square(self.kx), jnp.square(self.ky)
        self.k2 = kx2 + ky2
        νk = -(self.ν * self.k2).at[:, 0].set(self.νz)
        self.linear = νk - 1j * self.ky * self.κ / (1 + self.k2)
        self.forcing = jnp.exp(
            -(kx2 + jnp.square(self.ky - self.force_ky)) / 2
            / jnp.square(self.force_σ)
        ) * self.force_amplitude * filter_

        self.key = jax.random.PRNGKey(seed=self.seed)

    def explicit_terms(self, ŷ):
        φh = ŷ.view(dtype=complex)

        dφdx, dφdy, dωdx, dωdy = jnp.fft.irfft2(
            1j * jnp.array([
                self.kx * φh,
                self.ky * φh,
                -self.kx * self.k2 * φh,
                -self.ky * self.k2 * φh,
            ]),
            axes=(1, 2),
            norm="forward"
        )
        # Poisson bracket
        dφh = jnp.fft.rfft2(
            dφdx * (dφdy-dωdy) - dφdy * (dφdx-dωdx), norm="forward"
        )

        self.key, subkey = jax.random.split(self.key)
        phase = jax.random.uniform(subkey, shape=φh.shape)
        phase = jnp.exp(2j * jnp.pi * phase)
        term = make_hermitian(-dφh / (1 + self.k2) + self.forcing * phase)

        return term.view(dtype=float)

    def implicit_terms(self, ŷ):
        term = self.linear * ŷ.view(dtype=complex)
        return make_hermitian(term).view(dtype=float)

    def implicit_solve(self, ŷ, time_step):
        term = ŷ.view(dtype=complex) / (1 - time_step * self.linear)
        return term.view(dtype=float)


def hasegawa_mima_pspectral_2d(
    tf: float = 10,
    grid_size: Union[int, tuple[int, int]] = 1024,
    domain: Union[float, tuple[float, float]] = 16 * jnp.pi,
    video_length: float = 10.0,
    video_fps: float = 20,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    ν: float = 1e-4,
    νz: float = 1e-5,
    κ: float = 1,
    force_amplitude: float = 5e-3,
    force_ky: float = 1,
    filename: Union[str, Path] = None,
    seed: int = 42,
    solver: Union[str, type] = "Dopri8",
):
    nx, ny, lx, ly, grid = process_params_2d(grid_size, domain)

    κ, ν, νz = [x.item() if hasattr(x, "item") else x for x in (κ, ν, νz)]

    m = HasegawaMimaSpectral2D(
        grid,
        ν=ν,
        νz=νz,
        κ=κ,
        force_amplitude=force_amplitude,
        force_ky=force_ky,
        force_σ=2 * jnp.pi / max(lx, ly) * 5,
        seed=seed,
    )

    yh0 = init_fields_fourier_2d(
        grid=grid, key=jax.random.PRNGKey(seed=seed), n=1
    )

    # avoid tracer leak
    def split_callback():
        m.key = jax.random.PRNGKey(m.seed)

    return simulation_base(
        terms=get_terms(m, solver),
        tf=tf,
        coords={
            "x": jnp.linspace(0, lx, nx),
            "y": jnp.linspace(0, ly, ny),
            "field": ["φ"],
        },
        attrs={
            "model": "hasegawa_mima_pspectral_2d",
            "grid_size": (nx, ny),
            "domain": (lx, ly),
            "ν": ν,
            "νz": νz,
            "κ": κ,
            "force_amplitude": force_amplitude,
            "force_ky": force_ky,
            "seed": seed,
        },
        y0=yh0.view(dtype=float),
        solver=solver,
        atol=atol,
        rtol=rtol,
        video_length=video_length,
        video_fps=video_fps,
        filename=filename,
        apply=(real_to_fourier, fourier_to_real),
        split_callback=split_callback,
    )


def hasegawa_wakatani_pspectral_2d(
    tf: float = 10,
    grid_size: Union[int, tuple[int, int]] = 1024,
    domain: Union[float, tuple[float, float]] = 16 * jnp.pi,
    video_length: float = 10.0,
    video_fps: float = 20,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    C: float = 1,
    κ: float = 1,
    Dx: float = 1e-4,
    Dy: float = None,
    Dz: float = 1e-5,
    νx: float = 1e-4,
    νy: float = None,
    νz: float = 1e-5,
    filename: Union[str, Path] = None,
    seed: int = 42,
    solver: Union[str, type] = "Dopri8",
    save_diff: bool = False
):
    """Hasegawa-Wakatani pseudo-spectral 2D simulation

    2/3 zero padding rule is used for computing the non-linear term to avoid
    aliasing effects.

    Args:
        tf: final time simulation
        grid_size: shape without the padding
        domain: box size (cartesian space)
        video_length: length of the video (seconds)
        video_fps: frames per second of the video
        atol, rtol: absolute and relative tolerance
        C: adiabatic parameter
        κ: density gradient
        Dx, Dy, Dz: diffusion coefficients (x, y and zonal)
        νx, νy, νz: viscosity coefficients (x, y and zonal)
        filename: path and name of the .zarr file
        seed: seed for generating pseudo-random number key and for reproducibility
        solver: the solver name passed to diffeqsolve
    """
    nx, ny, lx, ly, grid = process_params_2d(grid_size, domain)

    Dy = Dy or Dx
    νy = νy or νx
    C, κ, Dx, Dy, Dz, νx, νy, νz = [
        x.item() if hasattr(x, "item") else x for x in (C, κ, Dx, Dy, Dz, νx, νy, νz)
    ]

    m = HasegawaWakataniSpectral2D(
        grid, C=C, Dx=Dx, Dy=Dy, Dz=Dz, νx=νx, νy=νy, νz=νz, κ=κ
    )

    yh0 = init_fields_fourier_2d(
        grid=grid, key=jax.random.PRNGKey(seed=seed), n=2
    )

    return simulation_base(
        terms=get_terms(m, solver),
        tf=tf,
        coords={
            "x": jnp.linspace(0, lx, nx),
            "y": jnp.linspace(0, ly, ny),
            "field": ["φ", "n"],
        },
        attrs={
            "model": f"hasegawa_wakatani_pspectral_{'1' if ny == 4 else '2'}d",
            "grid_size": (nx, ny),
            "domain": (lx, ly),
            "C": C,
            "κ": κ,
            "Dx": Dx,
            "Dy": Dy,
            "Dz": Dz,
            "νx": νx,
            "νy": νy,
            "νz": νz,
            "seed": seed,
        },
        y0=yh0.view(dtype=float),
        solver=solver,
        atol=atol,
        rtol=rtol,
        video_length=video_length,
        video_fps=video_fps,
        filename=filename,
        apply=(real_to_fourier, fourier_to_real),
    )


def hasegawa_wakatani_pspectral_1d(
    tf: float = 10,
    grid_size: int = 1024,
    domain: float = 16 * jnp.pi,
    video_length: float = 10.0,
    video_fps: float = 20,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    C: float = 1,
    κ: float = 1,
    Dx: float = 1e-4,
    Dy: float = None,
    Dz: float = 1e-5,
    νx: float = 1e-4,
    νy: float = None,
    νz: float = 1e-5,
    filename: Union[str, Path] = None,
    seed: int = 42,
    solver: Union[str, type] = "Dopri8",
    ky: Optional[float] = None,
):
    """Hasegawa-Wakatani pseudo-spectral 1D

    Simple wrapper around `hasegawa_wakatani_pspectral_2d`
    This is the reduced model: with a unique single poloidal mode

    """

    # take the first element if sized
    npx = grid_size[0] if hasattr(grid_size, "__len__") else grid_size
    if hasattr(domain, "__len__"):
        lx = domain[0]
        if ky is None:
            ky = 2 * jnp.pi / domain[1]
    else:
        lx = domain

    ky = ky or find_ky(C=C, D=Dy, κ=κ, ν=νy)

    da_2d = hasegawa_wakatani_pspectral_2d(
        tf=tf,
        grid_size=(npx, 4),
        domain=(lx, 2 * jnp.pi / ky),
        video_length=video_length,
        video_fps=video_fps,
        atol=atol,
        rtol=rtol,
        C=C,
        κ=κ,
        Dx=Dx,
        Dy=Dy,
        Dz=Dz,
        νx=νx,
        νy=νy,
        νz=νz,
        filename=filename,
        seed=seed,
        solver=solver,
        save_diff=True,
    )

    # decompose and plot
    file_path = Path(filename)
    da = open_with_vorticity(file_path)
    y = jnp.fft.rfft(jnp.array(da), axis=2, norm="forward")
    y = [
        x for i in range(len(da.coords["field"])) for x in (
            jnp.real(y[..., 1, i]),
            jnp.imag(y[..., 1, i]),
            jnp.real(y[..., 0, i])
        )
    ]
    field_names = [
        x for f in da.coords["field"].values
        for x in (f"{f}k_real", f"{f}k_imag", f"{f}b")
    ]

    da.attrs.update({
        "domain": da.attrs["domain"][0],
        "filename": str(file_path),
    })

    da = xr.DataArray(
        data=y,
        dims=["field", "time", "x"],
        coords={
            "time": da.time, "x": da.x, "field": field_names
        },
        attrs=da.attrs
    )
    da.to_zarr(
        file_path.with_name(f"{file_path.stem}_decomposed.zarr"), mode="w"
    )

    return da_2d


def hasegawa_wakatani_findiff_1d(
    C: float = 1,
    κ: float = 1,
    Dx: float = 1e-4,
    Dy: float = 1e-4,
    Dz: float = 0,
    νx: float = 1e-2,
    νy: float = 1e-4,
    νz: float = 0,
    ky: float = None,
    boundary: Union[str, tuple[str, float]] = "periodic",
    tf: float = 300,
    domain: float = 16 * jnp.pi,
    grid_size: int = 682,
    acc: int = 2,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    seed: int = 42,
    solver: Union[str, type] = "Dopri8",
    video_length: float = 10,
    video_fps: float = 20,
    filename: Union[str, Path] = None,
):
    """Hasegawa-Wakatani finite-difference 1D

    Reduced model with a single poloidal mode using finite-difference method,
    meaning that other than periodic boundary conditions are allowed.

    Args:
        tf: final time simulation
        grid_size: number of divisions in x direction, the resolution
        domain: box size in x direction
        video_length: length of the video (seconds)
        video_fps: frames per second of the video
        atol, rtol: absolute and relative tolerance
        C: adiabatic parameter
        κ: density gradient
        Dx, Dy, Dz: diffusion coefficients (x, y and zonal)
        νx, νy, νz: viscosity coefficients (x, y and zonal)
        filename: path and name of the .zarr file
        seed: seed for generating pseudo-random number key and for reproducibility
        solver: the solver name passed to diffeqsolve
        ky: value of the wave number of the poloidal mode
            If not provided, the most unstable one is computed
        boundary: boundary conditions of the left and right hand side of the domain
            [periodic, dirichlet, neumann, force]
            if force, neumann (left) and dirichlet (right) are used, κ is set to κ=0,
            a particle flux source (left) is added to nb and consider to increase tf
    """
    if ky is None:
        ky = find_ky(C=C, D=Dy, κ=κ, ν=νy)

    ky2 = jnp.square(ky)
    Dy = Dy or Dx
    νy = νy or νx

    # make sure these paramaters are not np.ndarray otherwise it will freeze
    C, κ, Dx, Dy, Dz, νx, νy, νz, ky = [
        χ.item() if hasattr(χ, "item") else χ
        for χ in (C, κ, Dx, Dy, Dz, νx, νy, νz, ky)
    ]

    shape = (grid_size, )
    grid = grids.Grid((grid_size, ), domain=[(0, domain)])

    bc_name, bc_value = process_boundary(boundary)

    # initial conditions
    key = jax.random.PRNGKey(seed=seed)
    x = jnp.linspace(0, domain, grid_size)

    def init_fields(key, A=1e-4, σ=0.5):
        """Initialize in the fourier space."""
        grid_ = grids.Grid((grid_size, 4),
                           domain=[(0, domain), (0, 2 * jnp.pi / ky)])
        y = init_fields_fourier_2d(
            grid_, key, n=2, A=A, σ=σ, padding=False, laplacian=True
        )
        y = jnp.fft.ifft(y, axis=0, norm="forward")

        Ωk = y[:, 1, 0]
        nk = y[..., 1, 1]
        Ωb = jnp.real(y[..., 0, 0])
        nb = jnp.real(y[..., 0, 1])

        return jnp.concatenate([
            Ωk.view(dtype=float), nk.view(dtype=float), Ωb, nb
        ])

    def combine_state(dΩk, dnk, dΩb, dnb):
        return jnp.concatenate([
            dΩk.view(dtype=float), dnk.view(dtype=float), dΩb, dnb
        ])

    def split_state(y):
        Ωk = y[..., :2 * grid_size].view(dtype=complex)
        nk = y[..., 2 * grid_size:4 * grid_size].view(dtype=complex)
        Ωb = y[..., 4 * grid_size:5 * grid_size]
        nb = y[..., 5 * grid_size:]
        return Ωk, nk, Ωb, nb

    k1, k2 = jax.random.split(key)
    if bc_name == "force":
        dx, nz, rhs = diff_matrix(grid, axis=0, order=1, acc=acc, bc_name="dirichlet")
        ddx, nz, rhs = diff_matrix(grid, axis=0, order=2, acc=acc, bc_name="dirichlet")
        dx_force, nz_force, rhs_force = diff_matrix(
            grid, axis=0, order=1, acc=acc, bc_name="force"
        )

        κ = 0
        forcing = jnp.exp(
            -jnp.square((x - 0.1*domain) / (domain/10))
        ) * bc_value
        well = forcing[::-1].at[-1].set(0)
        # buffering
        pad_size = int(grid_size * 0.05)
        pad = jnp.linspace(1, 100, pad_size)
        padded = jnp.concatenate(
            (pad[::-1], jnp.ones(grid_size - 2*pad_size), pad)
        )
        νx_, νy_, νz_, Dx_, Dy_ = [padded * χ for χ in (νx, νy, νz, Dx, Dy)]
        y0 = init_fields(key)
        y0 = combine_state(*[χ.at[nz].set(rhs) for χ in split_state(y0)])
    else:
        dx, nz, rhs = diff_matrix(
            grid, axis=0, order=1, acc=acc, bc_name=bc_name, bc_value=bc_value
        )
        ddx, nz, rhs = diff_matrix(
            grid, axis=0, order=2, acc=acc, bc_name=bc_name, bc_value=bc_value
        )
        dx_force, nz_force, rhs_force = dx.todense(), nz, rhs
        forcing, well = 0, 0
        νx_, νy_, νz_, Dx_, Dy_ = νx, νy, νz, Dx, Dy
        y0 = init_fields(key)

    ddxk_inv = jnp.linalg.pinv(
        ddx.todense() - jnp.eye(grid_size) * ky2, hermitian=True
    )
    ddx_inv = jnp.linalg.pinv(ddx.todense(), hermitian=True)

    def term(t, y, args):
        Ωk, nk, Ωb, nb = split_state(y)
        φk, φb = ddxk_inv @ Ωk, ddx_inv @ Ωb
        if bc_name != "periodic":
            Ωk, nk, φk = [χ.at[nz].set(rhs) for χ in (Ωk, nk, φk)]
            Ωb, nb, φb = [χ.at[nz_force].set(rhs_force) for χ in (Ωb, nb, φb)]

        φc = jnp.conjugate(φk)
        dφbdx = dx_force @ φb
        c_term = C * (φk-nk)

        return combine_state(
            dΩk=c_term + νx*ddx@Ωk - νy*ky2*Ωk - 1j * ky * (dφbdx*Ωk - (dx_force@Ωb) * φk),
            dnk=c_term + Dx*ddx@nk - Dy*ky2*nk - 1j * ky * (dφbdx*nk + (κ - dx_force@nb) * φk),
            dΩb=2 * ky * dx @ jnp.imag(φc * Ωk) - νz*Ωb,
            dnb=2 * ky * dx @ jnp.imag(φc * nk) - (Dz+well) * nb + forcing,
        ) # yapf: disable


    def decompose(y):
        """decompose real and imag parts"""
        Ωk, nk, Ωb, nb = split_state(y[..., :6 * grid_size])
        return jnp.stack(
            arrays=[Ωk.real, Ωk.imag, nk.real, nk.imag, Ωb, nb], axis=1
        )

    def recompose(y):
        Ωk_real, Ωk_imag, nk_real, nk_imag, Ωb, nb = y
        return combine_state(
            Ωk_real + 1j*Ωk_imag, nk_real + 1j*nk_imag, Ωb, nb
        )

    return simulation_base(
        terms=ODETerm(term),
        tf=tf,
        coords={
            "field": [
                "Ωk_real",
                "Ωk_imag",
                "nk_real",
                "nk_imag",
                "Ωb",
                "nb",
            ],
            "x": x,
        },
        attrs={
            "model": "hasegawa_wakatani_findiff_1d",
            "grid_size": grid_size,
            "domain": domain,
            "C": C,
            "κ": κ,
            "Dx": Dx,
            "Dy": Dy,
            "Dz": Dz,
            "νx": νx,
            "νy": νy,
            "νz": νz,
            "ky": ky,
            "boundary": bc_name + (f" {bc_value}" if bc_value != 0 else ""),
            "seed": seed,
            "acc": acc,
        },
        y0=y0,
        solver=solver,
        atol=atol,
        rtol=rtol,
        video_length=video_length,
        video_fps=video_fps,
        filename=filename,
        apply=(recompose, decompose),
    )


def hasegawa_wakatani_findiff_2d(
    tf: float = 10,
    grid_size: Union[int, tuple[int, int]] = 128,
    domain: Union[float, tuple[float, float]] = 16 * jnp.pi,
    video_length: float = 10.0,
    video_fps: float = 20,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    C: float = 1,
    κ: float = 1,
    D: float = 1e-1,
    Dz: float = 1e-2,
    ν: float = 1e-1,
    νz: float = 1e-2,
    boundary: Union[str, tuple[str, float]] = "periodic",
    acc: int = 2,
    seed: int = 42,
    solver: Union[str, type] = "Dopri8",
    filename: Union[str, Path] = None,
):
    C, κ, D, Dz, ν, νz = [
        χ.item() if hasattr(χ, "item") else χ for χ in (C, κ, D, Dz, ν, νz)
    ]

    bc_name, bc_value = process_boundary(boundary)
    nx, ny, lx, ly, grid = process_params_2d(grid_size, domain)

    diff_matrix_kwargs = {
        "grid": grid,
        "acc": acc,
        "bc_name": bc_name,
        "bc_value": bc_value,
    }
    dx_bcoo, nz, rhs = diff_matrix(axis=0, order=1, **diff_matrix_kwargs)
    dy_bcoo, nz, rhs = diff_matrix(axis=1, order=1, **diff_matrix_kwargs)
    Δ_bcoo, nz, rhs = diff_matrix(axis=[0, 1], order=2, **diff_matrix_kwargs)
    Δ_matmul = jax.tree_util.Partial(sparse.sparsify(jnp.matmul), Δ_bcoo)

    if bc_name == "periodic":
        y0 = init_fields_fourier_2d(
            grid,
            key=jax.random.PRNGKey(seed=seed),
            n=2,
            padding=False,
            laplacian=True
        )
        y0 = jnp.fft.irfft2(y0, axes=(0, 1), norm="forward")
    else:
        rhs = rhs.reshape(-1, 1)
        xv, yv = jnp.meshgrid(jnp.linspace(0, lx, nx), jnp.linspace(0, ly, ny))
        σx, σy = lx / 100, ly / 100
        n = jnp.exp(
            -(jnp.square((xv - lx/2) / σx) + jnp.square((yv - ly/2) / σy)) / 2
        ) * 1e-2
        Ω = jnp.zeros(grid.shape)
        y0 = jnp.stack([Ω, n], axis=-1)
        y0 = y0.reshape(-1, 2).at[nz].set(rhs).reshape(nx, ny, 2)

    def dx(y):
        return (dx_bcoo @ y.reshape(*y.shape[:-2], -1, 1)).reshape(*y.shape)

    def dy(y):
        return (dy_bcoo @ y.reshape(*y.shape[:-2], -1, 1)).reshape(*y.shape)

    def laplacian(y):
        return (Δ_bcoo @ y.reshape(*y.shape[:-2], -1, 1)).reshape(*y.shape)

    def term(t, y, args):
        if bc_name != "periodic":
            y = y.reshape(-1, 2).at[nz].set(rhs).reshape(nx, ny, 2)

        Ω, n = y[..., 0], y[..., 1]

        φ = jax.scipy.sparse.linalg.cg(
            Δ_matmul, Ω.ravel(), tol=rtol
        )[0].reshape(nx, ny)

        Ωnφ = jnp.array([Ω, n, φ])

        Ωb, nb, φb = Ωnφ.mean(-1, keepdims=True)
        Ωt, nt, φt = Ω - Ωb, n - nb, φ - φb
        c_term = C * (φt-nt)
        dΩdx, dndx, dφdx = dx(Ωnφ)
        dΩdy, dndy, dφdy = dy(Ωnφ)

        term = jnp.stack([
            c_term + ν * laplacian(Ωt) - νz*Ωb - dφdx*dΩdy + dφdy*dΩdx,
            c_term + D * laplacian(nt) - Dz*nb - dφdx*dndy + dφdy*(dndx - κ)
        ], axis=-1)  # yapf: disable

        if bc_name == "dirichlet":
            # make sure the values at the boundaries don't change
            term = term.reshape(-1, 2).at[nz].set(0).reshape(nx, ny, 2)

        return term

    return simulation_base(
        terms=ODETerm(term),
        tf=tf,
        coords={
            "x": jnp.linspace(0, lx, nx),
            "y": jnp.linspace(0, ly, ny),
            "field": ["Ω", "n"],
        },
        attrs={
            "model": "hasegawa_wakatani_findiff_2d",
            "grid_size": (nx, ny),
            "domain": (lx, ly),
            "C": C,
            "κ": κ,
            "D": D,
            "Dz": Dz,
            "ν": ν,
            "νz": νz,
            "boundary": bc_name + (f" {bc_value}" if bc_value != 0 else ""),
            "acc": acc,
            "seed": seed,
            "acc": acc,
        },
        y0=y0,
        solver=solver,
        atol=atol,
        rtol=rtol,
        video_length=video_length,
        video_fps=video_fps,
        filename=filename,
    )
