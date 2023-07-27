#!/usr/bin/env python3

# yapf main.py -i --style='{DEDENT_CLOSING_BRACKETS = true, COALESCE_BRACKETS = true, ARITHMETIC_PRECEDENCE_INDICATION = true, SPLIT_ALL_TOP_LEVEL_COMMA_SEPARATED_VALUES = true, SPLIT_BEFORE_ARITHMETIC_OPERATOR = true, SPLIT_BEFORE_DOT = true, SPLIT_BEFORE_EXPRESSION_AFTER_OPENING_PAREN = true, SPLIT_BEFORE_FIRST_ARGUMENT = true}'

import os
from inspect import signature
from pathlib import Path
from datetime import timedelta
from timeit import default_timer as timer
from argparse import ArgumentParser
import dataclasses
from functools import partial
from typing import Callable, Optional, Tuple, Union, Any, Sequence

import jax
import jax.numpy as jnp
from jax.experimental import host_callback, sparse
from jax.experimental.ode import odeint
from jax import lax, random

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

Array = Any
KeyArray = Union[Array, jax._src.prng.PRNGKeyArray]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import xarray as xr
import tqdm.auto
import yaml
import scipy
from findiff import FinDiff, BoundaryConditions
import zarr

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
from jax_cfd.spectral import time_stepping

import diffrax
from diffrax import (
    diffeqsolve,
    Dopri8,
    Kvaerno3,
    Kvaerno4,
    Kvaerno5,
    MultiTerm,
    ODETerm,
    SaveAt,
    PIDController,
    AbstractWrappedSolver,
    NewtonNonlinearSolver,
    AbstractNonlinearSolver,
    AbstractAdaptiveSolver,
    AbstractTerm,
    KenCarp5,
)


def brick_wall_filter_2d(grid: grids.Grid) -> Array:
    """Implements the 2/3 rule."""
    npx, npy = grid.shape
    nx, ny = npx // 3, npy//3 + 1
    return (
        jnp.zeros((npx, npy//2 + 1))
        .at[:nx, :ny].set(1)
        .at[-nx:, :ny].set(1)
    ) # yapf: disable


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
        filter_ = brick_wall_filter_2d(self.grid)
        ks = rfft_mesh(self.grid)
        self.kx, self.ky = ks * filter_
        kx2, ky2 = jnp.square(self.kx), jnp.square(self.ky)
        self.k2 = kx2 + ky2
        # use this one for division
        self.k2_div = jnp.square(ks).sum(0).at[0, 0].set(1)

        self.linear_term = jnp.empty((*self.k2.shape, 2, 2), dtype=complex)
        self.linear_term = (
            self.linear_term.at[:, :, 0, 0].set(
                -self.C / self.k2_div - self.νx * kx2 - self.νy * ky2
            )
            .at[:, :, 0, 1].set(self.C / self.k2_div)
            .at[:, :, 1, 0].set(-1j * self.ky * self.κ + self.C)
            .at[:, :, 1, 1].set(-self.C - self.Dx * kx2 - self.Dy * ky2)
            # zonal flows
            .at[:, 0, 0, 0].set(-self.νz)
            .at[:, 0, 0, 1].set(0)
            .at[:, 0, 1, 0].set(0)
            .at[:, 0, 1, 1].set(-self.Dz)
        ) # yapf: disable
        self.linear_term = make_hermitian(self.linear_term)

    def explicit_terms(self, ŷ):
        φh, nh = jnp.moveaxis(ŷ.view(dtype=complex), -1, 0)

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
        term = make_hermitian(jnp.stack((dφh / self.k2_div, -dnh), axis=-1))

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

        filter_ = brick_wall_filter_2d(self.grid)
        self.kx, self.ky = rfft_mesh(self.grid) * filter_
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


def make_hermitian(a):
    """Make the 2D Fourier space hermitian
    
    Symmetrize (conjugate) along kx in the Fourier space
    and set the zero and the Nyquist frequencies to zero
    
    Args:
        a: complex array of shape (kx, ky, ...)
    """
    x, y = a.shape[:2]
    b = a.at[:x // 2:-1, 0].set(jnp.conj(a[1:x // 2, 0]))
    b = b.at[x//2 + 1, :].set(0)
    b = b.at[:, -1].set(0)
    b = b.at[0, 0].set(0)
    return b


class SolverWrapTqdm(AbstractWrappedSolver):
    """Solver wrapper for progress bar

    It shows the tqdm progress bar while calling diffeqsolve
    at each `dt` interval of the simulation
    """
    tqdm_bar: tqdm.auto.tqdm
    nonlinear_solver: AbstractNonlinearSolver = None  # for implicit solvers
    # controls the simulation time interval for updating tqdm_bar
    dt: Union[int, float] = 1

    @property
    def term_structure(self):
        return self.solver.term_structure

    def order(self, terms):  # used for ODEs
        return self.solver.order(terms)

    def strong_order(self, terms):  # used for SDEs
        return self.solver.strong_order(terms)

    def error_order(self, terms):  # used for adaptive stepping
        return self.solver.error_order(terms)

    def func(self, *args, **kwargs):  # used for ODEs
        return self.solver.func(*args, **kwargs)

    def interpolation_cls(self, *args, **kwargs):  # used for SDEs
        return self.solver.interpolation_cls(*args, **kwargs)

    def init(self, terms, t0, t1, y0, args):
        solver_state = self.solver.init(terms, t0, t1, y0, args)

        last_t = jnp.array(t1)
        return (solver_state, last_t)

    def step(self, terms, t0, t1, y0, args, state, made_jump):
        solver_state, last_t = state
        y1, y_error, dense_info, solver_state, result = self.solver.step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )
        last_t = jax.lax.cond(
            t1 > last_t + self.dt,
            lambda: host_callback.id_tap(
                tap_func=(lambda t, transform: self.tqdm_bar.update(t)),
                arg=t1 - last_t,
                result=t1,
            ),
            lambda: last_t,
        )
        state = (solver_state, last_t)
        return y1, y_error, dense_info, state, result


class CrankNicolsonRK4(AbstractAdaptiveSolver):
    """
    Low storage scheme Carpenter-Kennedy 4th order https://doi.org/10.1007/978-3-540-30728-0 (Appendix D, p.535)
    A tuple of 3 ODETerm must be provided in diffeqsolve in this order:
    - explicit term (non-linear term)
    - implicit term (linear term)
    - implicit solve, it solves "y_{n+1} = y_n + dt * implicit_terms(y_{n+1})" : Callable[[t, y_n, dt], y_{n+1}]
    """

    term_structure = (AbstractTerm, AbstractTerm, AbstractTerm)
    interpolation_cls = diffrax.LocalLinearInterpolation
    # yapf: disable
    alphas = jnp.array([0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1])
    betas = jnp.array([0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257])
    gammas = jnp.array([0.1496590219993, 0.3792103129999, 0.8229550293869, 0.6994504559488, 0.1530572479681])
    # yapf: enable

    def init(self, terms, t0, t1, y0, args):
        self.μdt = jnp.diff(self.alphas) / 2  # precompute the diff
        return None

    def order(self, terms):
        return 4

    def step(
        self,
        terms: Tuple[AbstractTerm, AbstractTerm, AbstractTerm],
        t0,
        t1,
        y0,
        args,
        solver_state,
        made_jump,
    ):
        del solver_state, made_jump

        term_ex, term_im, term_solve = terms
        α, μdt, β, γ = self.alphas, self.μdt, self.betas, self.gammas  # short aliases
        dt = t1 - t0
        control_ex = term_ex.contr(t0, t1)
        control_im = term_im.contr(t0, t1)
        control_solve = term_solve.contr(t0, t1)

        # first iteration out of the loop so we can save ex0 and im0 for euler_y1
        ex0 = term_ex.vf(t0, y0, args)
        im0 = term_im.vf(t0, y0, args)
        h = ex0
        μ = dt * μdt[0]  # (α[1] - α[0]) / 2
        y1 = term_solve.vf(
            t0 + α[1] * dt, y0 + γ[0] * dt * h + μ*im0, μ
        )  # parse time_step as args

        # loop from 1
        for k in range(1, len(β)):
            tk = t0 + α[k] * dt
            h = term_ex.vf(tk, y1, args) + β[k] * h
            μ = dt * μdt[k]  # (α[k+1] - α[k]) / 2
            y1 = term_solve.vf(
                t0 + α[k + 1] * dt,
                y1 + γ[k] * dt * h + μ * term_im.vf(tk, y1, args),
                μ
            )

        euler_y1 = y0 + dt * (ex0+im0)
        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, diffrax.RESULTS.successful

    def func(self, terms, t0, y0, args):
        term_ex, term_im, term_solve = terms
        f1 = term_ex.vf(t0, y0, args)
        f2 = term_im.vf(t0, y0, args)
        return f1 + f2


def unpad(
    y: Array, grid: grids.Grid, axes: tuple[int, int] = (-2, -1)
) -> Array:
    """Unpad the Fourier space
    
    Remove the zero padding (2/3 rule). Thus, the final shape is smaller

    Args:
        y: complex array with shape (..., kx, ky, ...)
        grid: the grid object with the padded shape
        axes: axis position of kx, ky, must be consecutive

    Return:
        the unpadded array of `y`
    """
    npx, npy = grid.shape
    nx = int(npx / 3) * 2
    ny = npy//3 + 1
    mask = brick_wall_filter_2d(grid).astype(bool)
    new_shape = list(y.shape)
    new_shape[axes[0]] = nx
    new_shape[axes[1]] = ny
    index = [slice(None)] * (y.ndim - 1)
    pos = y.ndim + axes[0] if axes[0] < 0 else axes[0]
    index[pos] = mask
    return y[tuple(index)].reshape(*new_shape)


def init_hw_spectral_2d(
    grid: grids.Grid,
    key: KeyArray,
    n: int = 1,
    A: float = 1e-4,
    σ: float = 0.5,
    padding=True,
    vorticity=False,
) -> Array:
    """Create the initial fields in the fourier space
    
    Args:
        grid: Grid object with the padding shape
        key: for creating the random fields
        n: the number of fields
        A: amplitude of the gaussian
        σ: standard deviation of the gaussian

    Return:
        array of shape (grid_x, grid_y // 2 + 1) + ((n,) if n>1 else tuple())
    """

    if padding:
        # we use the unpadded shape
        unpadded_shape = (jnp.array(grid.shape) / 3).astype(int) * 2
        unpadded_grid = grids.Grid(unpadded_shape, domain=grid.domain)
    else:
        unpadded_grid = grid

    kx, ky = rfft_mesh(unpadded_grid)
    k2 = jnp.square(kx) + jnp.square(ky)
    ŷ0 = A * jnp.exp(
        -k2[..., None] / 2 / jnp.square(σ)
        + 2j * jnp.pi * jax.random.uniform(key, kx.shape + (n, ))
    )

    if vorticity:
        ŷ0 = ŷ0.at[..., 0].set(-k2 * ŷ0[..., 0])

    if padding:
        # now we pad
        filter_ = brick_wall_filter_2d(grid)
        empty = jnp.tile(filter_[..., None], (1, 1, n)).astype(complex)
        mask = filter_.astype(bool)
        ŷ0 = empty.at[mask].set(ŷ0.reshape(-1, n))

    ŷ0 = make_hermitian(ŷ0).squeeze()

    return ŷ0


def process_params_2D(grid_size, domain):
    if jnp.isscalar(domain):
        lx = jnp.array(domain).item()
        ly = lx
    else:
        assert len(domain) == 2
        lx, ly = tuple(domain)

    if jnp.isscalar(grid_size):
        nx, ny = grid_size, grid_size
    else:
        assert len(grid_size) == 2
        nx, ny = grid_size
    assert jnp.issubdtype(
        jnp.array([nx, ny]).dtype, jnp.integer
    ), "grid_size must be 2 integers."

    grid = grids.Grid((nx, ny), domain=((0, lx), (0, ly)))

    return nx, ny, lx, ly, grid


def to_direct_space(grid: grids.Grid) -> Callable[[Array], Array]:
    """Convert the simulation data to direct space"""

    def to_direct_space_(yh: Array):
        # move to cpu to handle large data
        y = jax.device_put(
            yh, device=jax.devices("cpu")[0]
        ).view(dtype=complex)
        y = unpad(y, grid, axes=(1, 2))
        y = jnp.fft.irfft2(y, axes=(1, 2), norm="forward")
        nt, nx, ny = y.shape[:3]
        return y.reshape(nt, nx, ny, -1)

    return to_direct_space_


def to_fourier_space(grid: grids.Grid) -> Callable[[Array], Array]:
    """Convert the initial state to Fourier space"""

    def to_fourier_space_(y: Array):
        npx, npy = grid.shape
        y0 = y.squeeze()
        y = jnp.fft.rfft2(y0, axes=(0, 1), norm="forward")
        return (  # padding
            jnp.zeros((npx, npy // 2 + 1, 2), dtype=complex)
            .at[brick_wall_filter_2d(grid).astype(bool)]
            .set(y.reshape(-1, *y0.shape[2:]))
            .view(dtype=float)
        )

    return to_fourier_space_


def get_terms(m, solver):

    def step(t, y, args=None):
        return m.explicit_terms(y) + m.implicit_terms(y)

    terms = ((
        ODETerm(lambda t, y, args: m.explicit_terms(y)),
        ODETerm(lambda t, y, args: m.implicit_terms(y)),
        ODETerm(lambda t, y, dt: m.implicit_solve(y, dt)),
    ) if solver == "CrankNicolsonRK4" else ODETerm(step))

    return terms


def hasegawa_mima_spectral_2D(
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
    npx, npy, lx, ly, grid = process_params_2D(grid_size, domain)

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

    yh0 = init_hw_spectral_2d(
        grid=grid, key=jax.random.PRNGKey(seed=seed), n=1
    )

    # avoid tracer leak
    def split_callback():
        m.key = jax.random.PRNGKey(m.seed)

    return simulation_base(
        terms=get_terms(m, solver),
        tf=tf,
        coords={
            "x": jnp.linspace(0, lx, int(npx / 3) * 2),
            "y": jnp.linspace(0, ly, int(npy / 3) * 2),
            "field": ["φ"],
        },
        attrs={
            "model": "hasegawa_mima_spectral_2D",
            "grid_size": (npx, npy),
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
        apply=(to_fourier_space(grid), to_direct_space(grid)),
        split_callback=split_callback,
    )


def hasegawa_wakatani_spectral_2D(
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
):
    """Hasegawa-Wakatani pseudo-spectral 2D simulation

    2/3 zero padding rule is used for computing the non-linear term to avoid
    aliasing effects.

    Args:
        tf: final time simulation
        grid_size: padded shape (Fourier space)
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
    npx, npy, lx, ly, grid = process_params_2D(grid_size, domain)

    Dy = Dy or Dx
    νy = νy or νx
    C, κ, Dx, Dy, Dz, νx, νy, νz = [
        x.item() if hasattr(x, "item") else x for x in (C, κ, Dx, Dy, Dz, νx, νy, νz)
    ]

    m = HasegawaWakataniSpectral2D(
        grid, C=C, Dx=Dx, Dy=Dy, Dz=Dz, νx=νx, νy=νy, νz=νz, κ=κ
    )

    yh0 = init_hw_spectral_2d(
        grid=grid, key=jax.random.PRNGKey(seed=seed), n=2
    )

    return simulation_base(
        terms=get_terms(m, solver),
        tf=tf,
        coords={
            "x": jnp.linspace(0, lx, int(npx / 3) * 2),
            "y": jnp.linspace(0, ly, int(npy / 3) * 2),
            "field": ["φ", "n"],
        },
        attrs={
            "model": f"hasegawa_wakatani_spectral_{'1' if npy == 6 else '2'}D",
            "grid_size": (npx, npy),
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
        apply=(to_fourier_space(grid), to_direct_space(grid)),
    )


def rfft_mesh(grid):
    return 2 * jnp.pi * jnp.array(grid.rfft_mesh())


def gridmesh_from_da(da):
    lx, ly = da.attrs["domain"]
    nx, ny = da.coords["x"].size, da.coords["y"].size
    grid = grids.Grid((nx, ny), domain=((0, lx), (0, ly)))
    kx, ky = rfft_mesh(grid)
    return grid, kx, ky


def visualization_2D(filename):
    plot_time_frames_2D(filename)
    plot_spectrum(filename)
    return animation_2D(filename)


def animation_2D(x, fields=["Ω", "n"]):
    """Create a .mp4 video of the 2D simulation"""

    assert set(fields).issubset({"n", "φ", "Ω"})
    da = open_with_vorticity(x)
    if "n" not in da.coords["field"].values:
        fields = ["Ω", "φ"]

    video_nframes = len(da)
    video_fps = da.attrs["video_fps"]
    tf = da.coords["time"].values[-1]

    fig, axes = plt.subplots(1, len(fields), figsize=(10, 5))
    images = [
        ax.imshow(da.sel(time=0, field=field), rasterized=True) for field,
        ax in zip(fields, axes)
    ]
    for image, field, ax in zip(images, fields, axes):
        image.set(cmap="seismic")
        ax.set(title=field, xticks=[], yticks=[])
    text_time = fig.text(0.5, 0.96, "")
    colorbar = fig.colorbar(
        images[1],
        ax=axes,
        location="right",
        pad=0,
        cax=ax.inset_axes([1.01, 0, 0.02, 1]),
    )
    colorbar.formatter.set_powerlimits((0.1, 10))
    fig.tight_layout()
    args = {
        "vmax_last": jnp.abs(jnp.array(da.sel(time=0, field=fields))).max()
    }

    def init():
        return images

    def update(frame, args):
        y = da.isel(time=frame)
        vmax = jnp.abs(jnp.array(y.sel(field=fields))).max()
        vmax = 0.2*vmax + 0.8 * args["vmax_last"]
        args["vmax_last"] = vmax
        for image, field in zip(images, fields):
            image.set(data=y.sel(field=field).T, clim=(-vmax, vmax))
        colorbar.update_normal(image.colorbar.mappable)
        pbar.update(1)
        text_time.set_text(f"Time={frame / video_nframes * tf:.3f}")
        return images

    ani = FuncAnimation(
        fig,
        partial(update, args=args),
        frames=video_nframes,
        interval=1000 / video_fps,
        init_func=init,
        blit=True,
    )

    tqdm_kwargs = {
        "iterable": range(video_nframes),
        "desc": "Animation",
    }
    if isinstance(x, xr.DataArray):
        plt.close()
        with tqdm.auto.tqdm(**tqdm_kwargs) as pbar:
            a = ani.to_html5_video(embed_limit=100)
        return HTML(a)
    else:

        with tqdm.auto.tqdm(**tqdm_kwargs) as pbar:
            ani.save(
                Path(x).with_suffix(".mp4"),
                writer=FFMpegWriter(fps=video_fps)
            )
        fig.savefig(Path(x).with_name("last_frame.pdf"), dpi=100)


def plot_time_frames_2D(filename, frames=[0, 0.25, 0.5, 1]):
    """Plot 2D fields at differents time frames"""

    da = open_with_vorticity(filename)
    second_field = "n" if "n" in da.coords["field"].values else "φ"
    lx, ly = da.attrs["domain"]
    fig, ax = plt.subplots(2, len(frames), figsize=(8, 5))

    for i, frame in enumerate(frames):
        t = frame * da.attrs["tf"]
        y = da.sel(time=t, method="nearest")
        vmax = jnp.abs(jnp.array(y)).max()
        imshow_kwargs = {
            "rasterized": True,
            "cmap": "seismic",
            "vmin": -vmax,
            "vmax": vmax,
            "extent": (0, lx, 0, ly),
        }
        img = ax[0, i].imshow(y.sel(field="Ω").T, **imshow_kwargs)
        ax[0, i].set(title=f"$\Omega \ t={t:.1f}$", xticks=[], yticks=[])
        ax[1, i].imshow(y.sel(field=second_field).T, **imshow_kwargs)
        ax[1, i].set(title=second_field, xticks=[], yticks=[])
        fig.colorbar(
            img,
            ax=ax[:, i],
            format="%.2g",
            orientation="horizontal",
            cax=ax[1, i].inset_axes([0, -0.1, 1, 0.02]),
        )

    fig.tight_layout()
    file_path = Path(filename)
    fig.savefig(
        file_path.with_name(f"{file_path.stem}_time_frames.pdf"),
        dpi=100,
        bbox_inches="tight",
        pad_inches=0,
    )


def plot_spectrum(filenames, n_last_frames=100):
    """Plot the energy-k spectrum in log-log scale
    
    The energy is averaged on the `n_last_frames` last frames of the simulaton
    """

    filenames_ = filenames if isinstance(filenames,
                                         (list, tuple)) else [filenames]
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, filename in enumerate(filenames_):
        da = open_with_vorticity(filename)
        grid, kx, ky = gridmesh_from_da(da)
        k2 = jnp.square(kx) + jnp.square(ky)
        k = jnp.sqrt(k2)
        dk = kx[1, 0] - kx[0, 0]
        kn = jnp.arange(dk, kx.max(), 2 * dk)

        y = da.isel(time=slice(-n_last_frames, None)).sel(field="Ω")
        # convert to φk
        y = -jnp.fft.rfft2(jnp.array(y), norm="forward") / k2.at[0, 0].set(1)
        y = k2 * jnp.square(jnp.abs(y)).mean(0) / 2
        y = jax.vmap(
            fun=lambda j: jnp.where(
                condition=(kn[j] - dk < k) &(k < kn[j] + dk), x=y, y=0
            ).sum()
        )(jnp.arange(kn.size)) # yapf: disable
        # En = jnp.array([Ek[(kn[j] - dk < k) & (k < kn[j] + dk)].sum() for j in range(kn.size)])
        ax.loglog(kn, y, label=f"{Path(filename).stem}")
    ax.set(
        # title="E(k)",
        xlabel="k",
        ylabel="E",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        Path(filename).with_name("spectrum.pdf"),
        bbox_inches="tight",
        pad_inches=0
    )


def hasegawa_wakatani_spectral_1D(
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

    Simple wrapper around `hasegawa_wakatani_spectral_2D`
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

    return hasegawa_wakatani_spectral_2D(
        tf=tf,
        grid_size=(npx, 6),
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
    )


def open_with_vorticity(filename) -> xr.DataArray:
    """Open the .zarr file

    It adds the vorticity field Ω if needed
    """
    with xr.open_dataarray(filename, engine="zarr") as da:
        npx, npy = da.attrs["grid_size"]
        lx, ly = da.attrs["domain"]
        da = da.load()

    da.attrs.update(zarr.open(filename).attrs)

    if "Ω" in da.coords["field"].values:
        return da

    # add vorticity field
    grid, kx, ky = gridmesh_from_da(da)
    vorticity = jnp.fft.irfft2(
        -(np.square(kx) + jnp.square(ky))
        * jnp.fft.rfft2(jnp.array(da.sel(field="φ")), norm="forward"),
        norm="forward",
    )
    vorticity = xr.DataArray(
        vorticity[..., None],
        dims=da.dims,
        coords={
            "time": da.time,
            "x": da.x,
            "y": da.y,
            "field": ["Ω"],
        },
    )
    return xr.concat((da, vorticity), dim="field")


def last_1D_to_2D(filename):
    """Project the single mode vorticity at last time step to 2D"""
    file_path = Path(filename)
    da = open_with_vorticity(filename)
    nx = len(da.coords["x"].values)
    Lx, Ly = da.attrs["domain"]
    Ωk = jnp.fft.rfft2(jnp.array(da.isel(time=-1).sel(field="Ω")))
    grid = grids.Grid((nx, nx), domain=((0, Lx), (0, Lx)))
    kx_squared, ky_squared = rfft_mesh(grid)
    ky_id = jnp.argmin(jnp.abs(ky_squared[0] - 2 * jnp.pi / Ly))
    mask = jnp.zeros_like(kx_squared).at[:, 0].set(1).at[:, ky_id].set(1)
    Ωk_squared = jnp.zeros_like(
        kx_squared, dtype=complex
    ).at[:, 0].set(Ωk[:, 0]).at[:, ky_id].set(Ωk[:, 1])
    Ω = jnp.fft.irfft2(Ωk_squared)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(Ω.T, rasterized=True, cmap="seismic")
    ax.set(
        xticks=[],
        yticks=[],
        xlabel="x",
        ylabel="y",
        title=f"Ω, t={da.attrs['tf']}"
    )
    fig.tight_layout()
    fig.savefig(
        file_path.with_name(f"{file_path.stem}_1D_to_2D.pdf"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )


def plot_spectral_1D(filename):
    """Plot the result of the single poloidal mode simulation

    time ↑
          -> x
    """
    da = open_with_vorticity(filename)

    # plot history
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    x = (da.coords["x"].values, )
    t = da.coords["time"].values
    for i, field_name in enumerate(("Ω", "n")):
        v = da.sel(field=field_name, y=0)
        vmax = abs(v).max()
        img = ax[i].pcolormesh(
            x, t, v, rasterized=True, cmap="seismic", vmin=-vmax, vmax=vmax
        )
        ax[i].set(title=field_name, ylabel="time", xlabel="x")
        fig.colorbar(img, ax=ax[i])

    file_path = Path(filename)
    fig.savefig(
        file_path.with_suffix(".pdf"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    # decompose and plot
    y = jnp.fft.rfft(jnp.array(da), axis=2)
    Ωk = y[..., 1, 2]
    nk = y[..., 1, 1]
    Ωb = jnp.real(y[..., 0, 2])
    nb = jnp.real(y[..., 0, 1])

    da.attrs.update({
        "domain": da.attrs["domain"][0],
        "filename": file_path,
    })

    da = xr.DataArray(
        data=[jnp.real(Ωk), jnp.imag(Ωk), jnp.real(nk), jnp.imag(nk), Ωb, nb],
        dims=["field", "time", "x"],
        coords={
            "time": da.time,
            "x": da.x,
            "field": ["Ωk_real", "Ωk_imag", "nk_real", "nk_imag", "Ωb", "nb"],
        },
        attrs=da.attrs
    )
    plot_components_1D(da)
    last_1D_to_2D(filename)


def hw_growth_rate(ky, C, D, κ, ν):
    ksq = np.square(ky)
    a = (D*ksq + C + C/ksq + ν*ksq) / 2
    b = (D*ksq + C - C/ksq - ν*ksq) / 2
    g = np.square(b) + np.square(C / ky)
    h = np.sqrt(np.square(g) + np.square(C * κ / ky))
    j = np.sqrt((h+g) / 2)
    return j - a


def find_ky(C: float, D: float, κ: float, ν: float) -> float:
    return scipy.optimize.minimize_scalar(
        fun=(lambda ky: -hw_growth_rate(ky, C, D, κ, ν)),
        bounds=(1e-4, 10),
    ).x.item()


def process_boundary(boundary):
    if boundary is not None:
        if isinstance(boundary, (tuple, list)):
            assert len(boundary) == 2, "boundary must be [bc_name, bc_value]"
            bc_name, bc_value = boundary
        else:
            splitted = boundary.split()
            assert len(splitted) <= 2, 'boundary must be "bc_name bc_value"'
            bc_name, bc_value = splitted if len(splitted) == 2 else (splitted[0], 0)
            bc_value = float(bc_value)
    else:
        bc_name, bc_value = "periodic", None

    return bc_name, bc_value


def hasegawa_wakatani_finite_difference_1D(
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
            The relations with the pseudo-spectral method (zero padding):
            grid_size_spectral = ceil(grid_size_findiff * 3 / 4) * 2
            grid_size_findiff = int(grid_size_spectral / 3) * 2
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
    z = jnp.zeros(shape)

    def random_field(key, A=1e-4, n=10):
        """Create a bunch of sinus with different frequencies, amplitudes and phases"""
        y0 = z
        offset = jnp.floor(domain / 2 * ky / (2 * jnp.pi)).astype(int)
        for i in range(offset, n + offset):
            key, k1, k2, k3 = jax.random.split(key, num=4)
            y0 += (
                # sinus phase
                jnp.sin(
                    i * x / (domain/2) * jnp.pi
                    + jax.random.uniform(k2, maxval=2 * jnp.pi)
                )
                # amplitude
                * jax.random.uniform(k1, minval=-A, maxval=A)
                # complex phase
                * jnp.exp(2j * jnp.pi * jax.random.uniform(k2))
            )
        if bc_name != "periodic":
            σ = domain / 2 / 2
            # apply gaussian window
            y0 *= jnp.exp(-jnp.square((x - domain/2) / σ))
            y0 = y0.at[jnp.array([0, -1])].set(0j)
        return y0.view(dtype=float)

    def random_field_(key, A=1e-4, σ=0.5):
        """Initialize in the fourier space."""
        grid_ = grids.Grid((grid_size, 4),
                           domain=[(0, domain), (0, 2 * jnp.pi / ky)])
        y = init_hw_spectral_2d(
            grid_, key, n=2, A=A, σ=σ, padding=False, vorticity=True
        )
        y = jnp.fft.rfft(
            jnp.fft.irfft2(y, axes=(0, 1), norm="forward"), axis=1
        )

        Ωk = y[:, 1, 0]
        nk = y[..., 1, 1]
        Ωb = jnp.real(y[..., 0, 0])
        nb = jnp.real(y[..., 0, 1])

        return jnp.concatenate([
            Ωk.view(dtype=float), nk.view(dtype=float), Ωb, nb
        ])

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
        well = jnp.diag(forcing[::-1].at[-1].set(0))
        # buffering
        pad_size = int(grid_size * 0.05)
        pad = jnp.linspace(1, 100, pad_size)
        padded = jnp.concatenate(
            (pad[::-1], jnp.ones(grid_size - 2*pad_size), pad)
        )
        νx_, νy_, νz_, Dx_, Dy_ = [padded * χ for χ in (νx, νy, νz, Dx, Dy)]
        y0 = jnp.concatenate((random_field(k1), random_field(k2), z, z))
        # y0 = jnp.zeros((6, grid_size))
    else:
        dx, nz, rhs = diff_matrix(
            grid, axis=0, order=1, acc=acc, bc_name=bc_name, bc_value=bc_value
        )
        ddx, nz, rhs = diff_matrix(
            grid, axis=0, order=2, acc=acc, bc_name=bc_name, bc_value=bc_value
        )
        dx_force, nz_force, rhs_force = dx, nz, rhs
        forcing, well = 0, 0
        νx_, νy_, νz_, Dx_, Dy_ = νx, νy, νz, Dx, Dy
        # y0 = jnp.concatenate((random_field(k1), random_field(k2), z, z))
        y0 = random_field_(key)

    ddxk_inv = jnp.linalg.pinv(
        ddx.todense() - jnp.eye(grid_size) * ky2, hermitian=True
    )
    ddx_inv = jnp.linalg.pinv(ddx.todense(), hermitian=True)

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

    def step(t, y, args):
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
        Ωk, nk, Ωb, nb = split_state(y)
        return jnp.stack(
            arrays=[
                jnp.real(Ωk),
                jnp.imag(Ωk),
                jnp.real(nk),
                jnp.imag(nk),
                Ωb,
                nb,
            ],
            axis=1
        )

    def recompose(y):
        Ωk_real, Ωk_imag, nk_real, nk_imag, Ωb, nb = y
        return combine_state(
            Ωk_real + 1j*Ωk_imag, nk_real + 1j*nk_imag, Ωb, nb
        )

    return simulation_base(
        terms=ODETerm(step),
        tf=tf,
        coords={
            "field": ["Ωk_real", "Ωk_imag", "nk_real", "nk_imag", "Ωb", "nb"],
            "x": x,
        },
        attrs={
            "model": "hasegawa_wakatani_finite_difference_1D",
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


def plot_components_1D(filename):
    """Plot the components 

    Plot Real(Xk) and Xb where X = Xb + Xk*exp(1j*ky*y) + conjugate(Xk)*exp(-1j*ky*y)
    is the single poloidal field
    """
    if not isinstance(filename, xr.DataArray):
        with xr.open_dataarray(filename, engine="zarr") as da:
            da = da.load()
    else:
        da = filename
        filename = da.attrs["filename"]

    Ωk, Ωk_imag, nk, nk_imag, Ωb, nb = jnp.array(da.transpose("field", ...))

    domain = da.attrs["domain"]
    grid_size = da.attrs["grid_size"]
    tf = da.attrs["tf"]
    κ = da.attrs["κ"]
    boundary = da.attrs.get("boundary", "periodic").split()
    x = da.coords["x"].values

    nrows = 2
    ncolumns = 3
    fig, axes = plt.subplots(nrows, ncolumns, figsize=(7, 4))  #(2, 15))

    for i, (k, v) in enumerate({
            "Real($\Omega_k$)": Ωk,
            "$\overline{\Omega}$": Ωb,
            "$\Omega$": Ωb + 2*Ωk,
            "Real($n_k$)": nk,
            "$\overline{n}$": nb,
            "$n$": nb + 2*nk,
    }.items()):
        ax = axes.ravel()[i]
        vmax = jnp.maximum(jnp.abs(v.min()), v.max())
        im = ax.pcolormesh(
            x,
            da.coords["time"].values,
            v,
            cmap="seismic",
            vmin=-vmax,
            vmax=vmax,
            rasterized=True,
        )
        fig.colorbar(
            im,
            ax=ax,
            location="right",
            cax=ax.inset_axes([1.04, 0, 0.02, 1]),
        ).formatter.set_powerlimits((0, 0))

        ax_settings = {"title": k}

        if i // (ncolumns) == nrows - 1:
            ax_settings["xlabel"] = "x"
        else:
            ax_settings["xticks"] = []

        if i % ncolumns == 0:
            ax_settings["ylabel"] = "time"
        else:
            ax_settings["yticks"] = []

        ax.set(**ax_settings)

    if boundary[0] == "force":
        forcing = (
            jnp.exp(-jnp.square((x - 0.1*domain) / (domain/10)))
            * float(boundary[1])
        )
        forcing -= forcing[::-1]  # show the well
        ax = axes.ravel()[4].twinx()
        ax.plot(x, forcing, label="source")
        ax.set(yticks=[])
        ax.legend(fontsize="small")

    ax = axes.ravel()[5].twinx()
    n_profil = da.isel(time=-1)
    n_profil = (
        n_profil.sel(field="nb") + 2 * n_profil.sel(field="nk_real") + κ *
        (domain-x)
    )
    ax.plot(x, n_profil, label="profile")
    ax.set(yticks=[])
    ax.legend(fontsize="small")

    fig.tight_layout()
    file_path = Path(filename)
    fig.savefig(
        file_path.with_name(f"{file_path.stem}_decompose.pdf"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    return Ωk, nk, Ωb, nb


def get_available_memory():
    """Return the available memory in GB"""
    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        import subprocess
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_info = subprocess.check_output(
            command.split()
        ).decode('ascii').split('\n')[:-1][1:]
        memory_values = [int(x.split()[0]) for i, x in enumerate(memory_info)]
        memory_value = memory_values[0]  # take the first gpu
        return memory_value * float(
            os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.9)
        ) / 1000
    else:
        import psutil
        return psutil.virtual_memory()[3] / 1000000000


def solve(diffeqsolve_kwargs, max_memory_will_use, save, split_callback=None):
    """Split the simulation if the memory usage is too large."""

    available_memory = get_available_memory()
    ts = diffeqsolve_kwargs["saveat"].subs.ts
    pbar = diffeqsolve_kwargs["solver"].tqdm_bar

    if max_memory_will_use > available_memory:
        n_iters = int(max_memory_will_use / available_memory) + 1
        tss = jnp.array_split(ts, n_iters)
        for i in range(n_iters):
            t1 = tss[i][-1].item()
            diffeqsolve_kwargs.update({
                "saveat":
                SaveAt(ts=tss[i], controller_state=True, solver_state=True),
                "t1":
                t1,
            })
            sol = diffeqsolve(**diffeqsolve_kwargs)
            pbar.update(t1 - pbar.n)
            diffeqsolve_kwargs.update({
                "y0": sol.ys[-1],
                "controller_state": sol.controller_state,
                "solver_state": sol.solver_state,
                "t0": t1,
            })
            da = save(sol.ys, tss[i])
            if split_callback is not None:
                split_callback()
    else:
        da = save(diffeqsolve(**diffeqsolve_kwargs).ys, ts)

    return da


def simulation_base(
    terms: Union[ODETerm, MultiTerm, Sequence[ODETerm]],
    tf: float,
    coords: dict[str, Any],
    attrs: dict[str, Any],
    y0: Array,
    solver: Union[str, type] = "Dopri8",
    atol: float = 1e-6,
    rtol: float = 1e-6,
    video_length: float = 1,
    video_fps: Union[int, float] = 20,
    apply: Optional[tuple[Callable[[Array], Array], Callable[[Array],
                                                             Array]]] = None,
    filename: Optional[Union[str, Path]] = None,
    split_callback: Optional[Callable[[], None]] = None,
):
    """General purpose simulation and saving

    diffeqsolve might be called multiple times (splitting) in order to fit the
    data into the memory. If splitted, the output is saved incrementally.
    The output is saved as an xr.DataArray in a .zarr file

    Args:
        terms: terms to be passed to diffeqsolve
        tf: final time, the initial time is t0=0
        coords: coordinates of the DataArray,
            a 'time' coordinate will be added
        attrs: attributes of the DataArray, useful for saving paramaters
        y0: initial condition/state, complex not supported, use `.view(dtype=float)`
        solver: the diffrax solver, default to Dopri8 (Runge Kutta order 8)
        atol, rtol: absolute and relative tolerance
        video_length, video_fps: length (in seconds) and fps of the output saved.
            The total number of frames saved is video_length * video_fps
        apply: (to_diffeqsolve, to_dataarray). 
            When resuming a simulation: y0 = to_diffeqsolve(last_y)
            For converting to DataArray: to_dataarray(ys)
            where ys has an additional leading time dimension
        filename: the output file name, expected to be a .zarr file
        split_callback: callback when the simulation is splitted

    
    """
    file_path = Path(filename)
    resume = False

    nonlinear_solver = NewtonNonlinearSolver(rtol=rtol, atol=atol)
    solvers = {
        "Dopri8": Dopri8(),
        "CrankNicolsonRK4": CrankNicolsonRK4(),
        "Kvaerno3": Kvaerno3(nonlinear_solver=nonlinear_solver),
        "Kvaerno4": Kvaerno4(nonlinear_solver=nonlinear_solver),
        "Kvaerno5": Kvaerno5(nonlinear_solver=nonlinear_solver),
        "KenCarp5": KenCarp5(nonlinear_solver=nonlinear_solver),
    }

    if isinstance(solver, str):
        assert solver in solvers

    attrs_ = attrs | {
        "video_length": video_length,
        "video_fps": video_fps,
        "solver": solver,
        "atol": atol,
        "rtol": rtol,
    }

    if file_path.exists():
        with xr.open_dataarray(file_path, engine="zarr") as da:
            # load all attrs
            da.attrs.update(zarr.open(file_path).attrs)
            # if all attrs are the same and tf is greater than the previous tf, it means we want to continue
            same = jnp.array([(
                np.array(v).shape == np.array(da.attrs[k]).shape
                and v == type(v)(da.attrs[k])
            ) for (k, v) in attrs_.items() if (
                isinstance(v, (int, float, str, tuple, list)) and k in da.attrs
            )]).all()

            if tf > da.attrs.get("tf", float("inf")) and same:
                t0 = da.attrs["tf"]
                y0 = jnp.array(da.isel(time=-1))
                runtime_old = da.attrs.get("runtime", 0)
                resume = True
                print(f"The simulation is resumed from {filename}.")

    attrs_.update({"tf": tf})
    dims = ["time"] + list(coords.keys())
    video_nframes = int(video_length * video_fps)

    if resume:
        time_linspace = jnp.linspace(
            t0 + 1/video_fps, tf, int((tf-t0) / tf * video_nframes)
        )
        if apply:
            y0 = apply[0](y0)
    else:
        # overwrite with an empty file
        y_dummy = apply[1](y0[None]) if apply is not None else y0[None]
        xr.DataArray(
            data=jnp.array([]).reshape(-1, *y_dummy.shape[1:]),
            dims=dims,
            coords={"time": []} | coords,
            attrs=attrs_,
        ).to_zarr(file_path, mode="w") # yapf: disable

        runtime_old = 0
        t0 = 0
        # the number of frames saved is video_nframes, it is not proportional to tf
        time_linspace = jnp.linspace(t0, tf, video_nframes + 1)

    for k, v in attrs_.items():
        # print only short parameters
        if not isinstance(v, (np.ndarray, )):
            print(f"{k:<20}: {v}")

    max_memory_will_use = (
        video_nframes * y0.size * {
            jnp.dtype("float64"): 8, jnp.dtype("float32"): 4
        }[y0.dtype]  # number of bytes
        * 2  # it seems it will be buffered twice ...
        / 1000000000  # convert to GB
    )

    def save(ys, ts):
        da = xr.DataArray(
            apply[1](ys) if apply else ys,
            dims=dims,
            coords={"time": ts} | coords,
            attrs=attrs_,
        )

        da.to_zarr(file_path, append_dim="time")

        return da

    t1 = timer()
    with tqdm.auto.tqdm(
            total=tf,
            desc="Simulation",
            bar_format=
            "{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            initial=t0,
    ) as tqdm_bar:

        diffeqsolve_kwargs = {
            "terms": terms,
            "solver":
            SolverWrapTqdm(solvers[solver], tqdm_bar, nonlinear_solver),
            "t0": t0,
            "t1": tf,
            "dt0": 1e-3,
            "y0": y0,
            "saveat": SaveAt(ts=time_linspace),
            "stepsize_controller": PIDController(atol=atol, rtol=rtol),
            "max_steps": None,
        }
        da = solve(
            diffeqsolve_kwargs, max_memory_will_use, save, split_callback
        )

    # add the post-simulation attributes
    z = zarr.open(file_path)
    attrs_new = {
        "runtime": z.attrs.get("runtime", 0) + timer() - t1,
    }
    if resume:
        attrs_new["tf"] = tf
    z.attrs.update(attrs_new)

    return da


def diff_matrix(
    grid: grids.Grid,
    axis: Union[int, list[int]],
    order: int,
    acc: int = 2,
    bc_name: str = "periodic",
    bc_value: float = 0,
    csr: bool = False
) -> tuple[Union[sparse.BCOO, sparse.BCSR], Optional[Array], Optional[Array]]:
    """Compute the differential matrix operator

    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Usage:
    `dy = (op @ y.at[nz].set(rhs).ravel()).reshape(y.shape)`

    Args:
        grid: the grid object
        axis: axis coordinate for derivative. For example:
            0 => ∂x, 1 => ∂y, 3 => ∂z, [0, 1] => ∂x + ∂y
        order: derivative order, ∂^order
        acc: order of accuracy, must be a positive even integer 
        bc_name: boundary condition name, 'periodic', 'dirichlet', 'neumann', 'force'
        bc_value: value of the boundary condition
        csr: use BCSR or BCOO sparse matrix 

    Return:
        op: differential sparse 2D matrix operator, BCOO or BCSR with shape
            (prod(grid.shape), prod(grid.shape))
        nz: indices where the constraint must be applied on the state, None if periodic
        rhs: values of the constraint of the state, None if periodic
    """
    assert acc < min(grid.shape) / 2, f"acc={acc} is too big. The grid.shape is {grid.shape}."

    if isinstance(axis, int):
        op = FinDiff(axis, grid.step[axis], order, acc=acc)
    else:
        op = sum([FinDiff(i, grid.step[i], order, acc=acc) for i in axis])

    square_size = jnp.prod(jnp.array(grid.shape))
    op_shape = (square_size, square_size)
    if bc_name == "periodic":
        pos = jnp.prod(jnp.array(grid.shape[:-1], dtype=int) + 1) * (acc-1)
        op = op.matrix(grid.shape)[pos].toarray()
        row = jnp.roll(jnp.array(op), -pos).squeeze()
        indices = jnp.nonzero(row)[0]
        data = row[indices]
        data = jnp.tile(data, square_size)
        if csr:
            indptr = jnp.arange(0, len(data) + 1, len(indices))
            indices = jnp.concatenate(
                jax.vmap(fun=lambda i: (indices+i) % square_size
                         )(jnp.arange(square_size)),
                axis=0,
            )
            op = sparse.CSR((data, indices, indptr), shape=op_shape)

        else:
            ones = jnp.ones_like(indices)
            indices = jnp.concatenate(
                jax.vmap(
                    fun=lambda i: jnp.stack(
                        arrays=(ones * i, (indices+i) % square_size), axis=-1
                    )
                )(jnp.arange(square_size)),
                axis=0,
            )
            op = sparse.BCOO((data, indices), shape=op_shape)
        nz, rhs = None, None
    else:
        bc = BoundaryConditions(shape=grid.shape)
        for i in range(len(grid.shape)):
            find = FinDiff(i, grid.step[i], 1, acc=acc)
            slices = tuple([slice(None)] * i)
            bc[slices + (0,)], bc[slices + (-1,)] = {
                "dirichlet": (bc_value, bc_value),
                "neumann": ((find, bc_value), (find, bc_value)),
                "force": ((find, 0), 1e-1),  # neumann ad dirichlet
            }[bc_name]
        nz = jnp.array(bc.row_inds())
        rhs = jnp.array(bc.rhs[nz].toarray()).ravel()

        op = op.matrix(shape=grid.shape)  # convert to scipy sparse
        op[nz, :] = bc.lhs[nz, :]  # apply boundary conditions
        # convert to jax sparse
        op = (sparse.BCSR if csr else sparse.BCOO).from_scipy_sparse(op)
    return op, nz, rhs


def hasegawa_wakatani_finite_difference_2D(
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
    nx, ny, lx, ly, grid = process_params_2D(grid_size, domain)

    diff_matrix_kwargs = {
        "grid": grid,
        "acc": acc,
        "bc_name": bc_name,
        "bc_value": bc_value,
    }
    dx_bcoo, nz, rhs = diff_matrix(axis=0, order=1, **diff_matrix_kwargs)
    dy_bcoo, nz, rhs = diff_matrix(axis=1, order=1, **diff_matrix_kwargs)
    laplacian_bcoo, nz, rhs = diff_matrix(axis=[0, 1], order=2, **diff_matrix_kwargs)

    if bc_name == "periodic":
        y0 = init_hw_spectral_2d(
            grid,
            key=jax.random.PRNGKey(seed=seed),
            n=2,
            padding=False,
            vorticity=True
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
        return (laplacian_bcoo
                @ y.reshape(*y.shape[:-2], -1, 1)).reshape(*y.shape)

    def term(t, y, args):
        if bc_name != "periodic":
            y = y.reshape(-1, 2).at[nz].set(rhs).reshape(nx, ny, 2)

        Ω, n = y[..., 0], y[..., 1]

        φ = jax.scipy.sparse.linalg.cg(
            jax.tree_util.Partial(sparse.sparsify(jnp.matmul), laplacian_bcoo),
            Ω.ravel(),
            tol=atol
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
            "model": "hasegawa_wakatani_finite_difference_2D",
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


def hasegawa_wakatani_spectral_single_poloidal_mode(
    C: float = 1,
    κ: float = 1,
    Dx: float = 1e-4,
    Dy: float = 1e-4,
    Dz: float = 0,
    νx: float = 1e-2,
    νy: float = 1e-4,
    νz: float = 0,
    ky: float = None,
    tf: float = 300,
    domain: float = 16 * jnp.pi,
    grid_size: int = 682,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    seed: int = 42,
    solver: Union[str, type] = "Dopri8",
    video_length: float = 10,
    video_fps: float = 20,
    filename: Union[str, Path] = None,
):
    ky = ky or find_ky(C=C, D=Dy, κ=κ, ν=νy)

    grid = grids.Grid((grid_size, 6),
                      domain=[(0, domain), (0, 2 * jnp.pi / ky)])
    y0 = init_hw_spectral_2d(
        grid, key=jax.random.PRNGKey(seed=seed), n=2, vorticity=True
    )
    y0 = jnp.fft.rfft(
        jnp.fft.irfft2(y0, axes=(0, 1), norm="forward"),
        axis=1,
        norm="backward"
    )
    y0 = jnp.fft.fft(y0, axis=0, norm="forward")
    y0 = jnp.stack([y0[:, 1, 0], y0[:, 1, 1], y0[:, 0, 0], y0[:, 0, 1]],
                   axis=-1)

    filter_ = brick_wall_filter_2d(grid)[:, 0]
    kx = rfft_mesh(grid)[0, :, 0]
    kx2 = jnp.square(kx)
    ky2 = jnp.square(ky)
    k2 = kx2 + ky2
    kx *= filter_
    ones = jnp.ones_like(kx)

    linear = jnp.array([
        [-C / k2 - νx*kx2 - νy*ky2, -C * ones, 0 * ones, 0 * ones],
        [-C / k2 - 1j*ky*κ/k2, -C - Dx*kx2 - Dy*ky2, 0 * ones, 0 * ones],
        [0 * ones, 0 * ones, -νz * ones, 0 * ones],
        [0 * ones, 0 * ones, 0 * ones, -Dz * ones]
    ]) * filter_[None, None] # yapf: disable
    linear = jnp.moveaxis(linear, -1, 0)

    def term(t, y, args):
        y = y.view(dtype=complex)

        Ωkh, nkh, Ωbh, nbh = jnp.moveaxis(y, -1, 0)
        dφbdx, dΩbdx, dnbdx, φk, Ωk, nk = jnp.fft.ifft(jnp.array([
            - 1j * kx * Ωbh / kx2.at[0].set(1),
            1j * kx * Ωbh,
            1j * kx * nbh,
            - Ωkh / k2,
            Ωkh,
            nkh
        ]), norm="forward")

        dφbΩk, dΩbφk, dφbnk, dnbφκ, φcΩk, φcnk = jnp.fft.fft(jnp.array([
            dφbdx * Ωk,
            dΩbdx * φk,
            dφbdx * nk,
            dnbdx * φk,
            jnp.conjugate(φk) * Ωk,
            jnp.conjugate(φk) * nk
        ]), norm="forward")

        term = 1j * ky * jnp.stack(
            arrays=[
                dΩbφk - dφbΩk,
                dnbφκ - dφbnk,
                2 * kx * jnp.imag(φcΩk),
                2 * kx * jnp.imag(φcnk),
            ],
            axis=-1
        )

        term += (linear @ y[..., None]).reshape(*y.shape)
        term *= filter_[:, None]

        return term.view(dtype=float)

    def to_direct_space(y):
        y = y.view(dtype=complex)
        mask = brick_wall_filter_2d(grid)[:, 0].astype(bool)
        unpadded_size = int(grid_size / 3) * 2
        y = y[:, mask].reshape(-1, unpadded_size, 4)
        y = jnp.fft.ifft(
            y, axis=-2, norm="forward"
        ) / grid_size * unpadded_size
        Ωk, nk, Ωb, nb = jnp.moveaxis(y, -1, 0)
        return jnp.stack(
            arrays=[
                jnp.real(Ωk),
                jnp.imag(Ωk),
                jnp.real(nk),
                jnp.imag(nk),
                jnp.real(Ωb),
                jnp.real(nb)
            ],
            axis=2
        )

    def to_fourier_space(y):
        raise NotImplementedError("delete the zarr file.")

    return simulation_base(
        terms=ODETerm(term),
        tf=tf,
        coords={
            "x": jnp.linspace(0, domain, int(grid_size / 3) * 2),
            "field": ["Ωk_real", "Ωk_imag", "nk_real", "nk_imag", "Ωb", "nb"],
        },
        attrs={
            "model": "hasegawa_wakatani_spectral_single_poloidal_mode",
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
            "seed": seed,
        },
        y0=y0.view(dtype=float),
        solver=solver,
        atol=atol,
        rtol=rtol,
        video_length=video_length,
        video_fps=video_fps,
        filename=filename,
        apply=(to_fourier_space, to_direct_space)
    )


def main():
    schemes = {
        "hasegawa_mima_spectral_2D":
        (hasegawa_mima_spectral_2D, visualization_2D),
        "hasegawa_wakatani_spectral_2D":
        (hasegawa_wakatani_spectral_2D, visualization_2D),
        "hasegawa_wakatani_spectral_1D":
        (hasegawa_wakatani_spectral_1D, plot_spectral_1D),
        "hasegawa_wakatani_finite_difference_1D":
        (hasegawa_wakatani_finite_difference_1D, plot_components_1D),
        "hasegawa_wakatani_finite_difference_2D":
        (hasegawa_wakatani_finite_difference_2D, visualization_2D),
        "hasegawa_wakatani_spectral_single_poloidal_mode":
        (hasegawa_wakatani_spectral_single_poloidal_mode, plot_components_1D),
    }

    parser = ArgumentParser(
        prog="hasegawa_wakatani_simulation",
        description=
        f"Script for running Hasegawa-Wakatani simulations, saving and/or plotting. The implemented models are {list(schemes.keys())}."
    )
    parser.add_argument("filename")
    parser.add_argument("--tf", type=float, help="Final simulation time")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help=
        "Force using CPU when jax is configured with GPU. TODO: create a seperate file and import after jax.config.update()",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Run in eager mode / disable jit for debugging",
    )
    # parser.add_argument("--double", action="store_true", help="Enable 64 double precision")
    args = parser.parse_args()

    if args.cpu:
        jax.config.update("jax_platform_name", "cpu")

    if args.eager:
        jax.config.update("jax_disable_jit", False)

    p = Path(args.filename)
    if p.suffix == ".yaml":
        with open(p, "r") as f:
            try:
                yaml_data = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(exc)
        assert (
            len(set(yaml_data.keys()).intersection(schemes.keys())) == 1
        ), f"Choose only one scheme among {schemes.keys()}"

        model = [k for k in schemes if k in yaml_data][0]
        simulation, post_process = schemes[model]
        simulation_kwargs = {
            "filename": p.with_suffix(".zarr")
        } | yaml_data[model]

        simulation(**simulation_kwargs)
        post_process(simulation_kwargs["filename"])

    elif p.suffix == ".zarr":
        with xr.open_dataarray(p, engine="zarr") as da:
            simulation_kwargs = da.load().attrs
            model = simulation_kwargs.pop("model")

        simulation, post_process = schemes[model]
        # resume
        if args.tf is not None:
            simulation_kwargs.update({
                "tf": args.tf,
                "filename": p,
            })
            simulation_kwargs = {
                k: simulation_kwargs[k]
                for k in signature(simulation).parameters
                if k in simulation_kwargs
            }
            simulation(**simulation_kwargs)
            post_process(p)
        else:  # plotting only
            post_process(p)
    elif p.is_dir():
        filename = p / "HasegawaWakataniSpectral2D.zarr"
        hasegawa_wakatani_spectral_2D(filename=filename)
        visualization_2D(filename)


if __name__ == "__main__":

    main()
