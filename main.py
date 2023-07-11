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
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import host_callback, sparse
from jax.experimental.ode import odeint
from jax import lax, random

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import xarray as xr
import tqdm.auto
import yaml
import scipy
from findiff import FinDiff, BoundaryConditions

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
    AbstractImplicitSolver,
    AbstractAdaptiveSolver,
    AbstractTerm,
    KenCarp5,
)


def brick_wall_filter_2d(grid):
    """Implements the 2/3 rule."""
    npx, npy = grid.shape
    nx, ny = npx // 3, npy//3 + 1
    filter = jnp.zeros((npx, npy//2 + 1))
    filter = filter.at[:nx, :ny].set(1)
    filter = filter.at[-nx:, :ny].set(1)

    return filter


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
        self.kx, self.ky = rfft_mesh(self.grid)
        kx2, ky2 = jnp.square(self.kx), jnp.square(self.ky)
        self.ksq = kx2 + ky2
        self.ksq_div = self.ksq.at[0, 0].set(1)  # use this one for division

        npx, npy = self.grid.shape
        self.nx, self.ny = int(npx / 3) * 2, npy//3 + 1
        self.filter_ = brick_wall_filter_2d(self.grid)
        self.mask = self.filter_.astype(bool)

        self.kx *= self.filter_
        self.ky *= self.filter_

        self.linear_term = jnp.empty((*self.ksq.shape, 2, 2), dtype=complex)
        self.linear_term = (
            self.linear_term.at[:, :, 0, 0].set(
                -self.C / self.ksq_div - self.νx * kx2 - self.νy * ky2
            )
            .at[:, :, 0, 1].set(self.C / self.ksq_div)
            .at[:, :, 1, 0].set(-1j * self.ky * self.κ + self.C)
            .at[:, :, 1, 1].set(-self.C - self.Dx * kx2 - self.Dy * ky2)
            # zonal flows
            .at[:, 0, 0, 0].set(-self.νz * self.ksq[:, 0])
            .at[:, 0, 0, 1].set(0)
            .at[:, 0, 1, 0].set(0)
            .at[:, 0, 1, 1].set(-self.Dz * self.ksq[:, 0])
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
                -self.kx * self.ksq * φh,
                -self.ky * self.ksq * φh,
            ]),
            axes=(1, 2),
        )

        dnh, dφh = jnp.fft.rfft2(
            jnp.array([dφdx * dndy - dφdy * dndx, dφdx * dωdy - dφdy * dωdx]),
            axes=(1, 2),
        )
        term = make_hermitian(jnp.stack((dφh / self.ksq_div, -dnh), axis=-1))

        return term.view(dtype=float)

    def implicit_terms(self, ŷ):
        term = self.linear_term @ ŷ.view(dtype=complex)[..., None]
        return make_hermitian(term.squeeze()).view(dtype=float)

    def implicit_solve(self, ŷ, time_step):
        inv = jnp.linalg.inv(jnp.eye(2) - time_step * self.linear_term)
        term = inv @ ŷ.view(dtype=complex)[..., None]
        return term.squeeze().view(dtype=float)


def make_hermitian(a):
    """
    Symmetrize (conjugate) along kx in the Fourier space
    and set the Nyquist frequencies to zero
    arg: a: complex array of shape (..., kx, ky)
    """
    x, y = a.shape[:2]
    b = a.at[:x // 2:-1, 0].set(jnp.conj(a[1:x // 2, 0]))
    b = b.at[x//2 + 1, :].set(0)
    b = b.at[:, -1].set(0)
    b = b.at[0, 0].set(0)
    return b


class SolverWrapTqdm(AbstractWrappedSolver):
    tqdm_bar: tqdm.auto.tqdm
    # controls the simulation time interval for updating tqdm_bar
    dt: Union[int, float] = 1
    nonlinear_solver: AbstractNonlinearSolver = None  # for implicit solvers

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
    alphas = jnp.array([
        0,
        0.1496590219993,
        0.3704009573644,
        0.6222557631345,
        0.9582821306748,
        1
    ])
    betas = jnp.array([
        0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257
    ])
    gammas = jnp.array([
        0.1496590219993,
        0.3792103129999,
        0.8229550293869,
        0.6994504559488,
        0.1530572479681,
    ])

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


def unpad(y, grid, axes=(-2, -1)):
    npx, npy = grid.shape
    nx = int(npx / 3) * 2
    ny = npy//3 + 1
    mask = brick_wall_filter_2d(grid).astype(bool)
    new_shape = list(y.shape)
    new_shape[axes[0]] = nx
    new_shape[axes[1]] = ny
    index = [slice(None)] * (len(new_shape) - 1)
    index[axes[0]] = mask
    return y[tuple(index)].reshape(*new_shape)


def init_hw_spectral_2d(grid, key, n=1, A=1e-4, σ=0.5):
    """
    Create the 2 initial fields in the fourier space:
    electric potential and density
    n: nb of fields
    if unpad=True, remove the padding (smaller shape)

    return: array of shape (grid_x, grid_y // 2 + 1, 2)
    """
    kx, ky = rfft_mesh(grid)
    ŷ0 = A * jnp.exp(
        -(jnp.square(kx) + jnp.square(ky)) / 2 / jnp.square(σ)
        + 1j * jax.random.uniform(key, kx.shape, maxval=2 * jnp.pi)
    )
    ŷ0 = make_hermitian(ŷ0)

    if n > 1:
        ŷ0 = jnp.tile(ŷ0[..., None], reps=(1, n))

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


def hasegawa_wakatani_spectral_2D(
    tf=10,
    grid_size=512,
    domain=16 * jnp.pi,
    video_length=10.0,  # seconds
    video_fps=20,  # fps
    atol=1e-6,
    rtol=1e-6,
    C=1,
    κ=1,
    Dx=1e-3,
    Dy=None,
    Dz=1e-5,
    νx=1e-3,
    νy=None,
    νz=1e-5,
    filename=None,
    seed=42,
    solver="Dopri8",
):
    npx, npy, lx, ly, grid = process_params_2D(grid_size, domain)

    Dy = Dy or Dx
    νy = νy or νx
    C, κ, Dx, Dy, Dz, νx, νy, νz = [
        x.item() if hasattr(x, "item") else x for x in (C, κ, Dx, Dy, Dz, νx, νy, νz)
    ]

    m = HasegawaWakataniSpectral2D(
        grid, C=C, Dx=Dx, Dy=Dy, Dz=Dz, νx=νx, νy=νy, νz=νz, κ=κ
    )

    yh0 = (
        init_hw_spectral_2d(grid=grid, key=jax.random.PRNGKey(seed=seed), n=2)
        * brick_wall_filter_2d(grid)[..., None]
    )

    def step(t, y, args=None):
        return m.explicit_terms(y) + m.implicit_terms(y)

    terms = ((
        ODETerm(lambda t, y, args: m.explicit_terms(y)),
        ODETerm(lambda t, y, args: m.implicit_terms(y)),
        ODETerm(lambda t, y, dt: m.implicit_solve(y, dt)),
    ) if solver == "CrankNicolsonRK4" else ODETerm(step))

    def to_direct_space(yh):
        # move to cpu to handle large data
        y = jax.device_put(
            yh, device=jax.devices("cpu")[0]
        ).view(dtype=complex)
        y = unpad(y, grid, axes=(1, 2))
        # y = y[:, m.mask].reshape(-1, m.nx, m.ny, 2)  # unpad
        y = jnp.fft.irfft2(y, axes=(1, 2))
        return y

    def to_fourier_space(y):
        y = jnp.fft.rfft2(y, axes=(0, 1))
        return (  # padding
            jnp.zeros((npx, npy // 2 + 1, 2), dtype=complex)
            .at[m.mask]
            .set(y.reshape(-1, 2))
            .view(dtype=float)
        )

    return simulation_base(
        terms=terms,
        tf=tf,
        coords={
            "x": jnp.linspace(0, lx, int(npx / 3) * 2),
            "y": jnp.linspace(0, ly, int(npy / 3) * 2),
            "field": ["φ", "n"],
        },
        attrs={
            "model": "hasegawa_wakatani_spectral_2D",
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
        apply=(to_fourier_space, to_direct_space),
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
    assert set(fields).issubset({"n", "φ", "Ω"})
    da = open_spectral_with_vorticity(x)
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
        vmax = 0.3*vmax + 0.7 * args["vmax_last"]
        args["vmax_last"] = vmax
        for image, field, ax in zip(images, fields, axes):
            image.set(data=y.sel(field=field).transpose(), clim=(-vmax, vmax))
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
    da = open_spectral_with_vorticity(filename)
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
        ax[1, i].imshow(y.sel(field="n").T, **imshow_kwargs)
        ax[1, i].set(title="$n$", xticks=[], yticks=[])
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
    filenames_ = filenames if isinstance(filenames,
                                         (list, tuple)) else [filenames]
    fig, ax = plt.subplots(figsize=(4, 3))
    for i, filename in enumerate(filenames_):
        da = open_spectral_with_vorticity(filename)
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
    tf=300,
    grid_size=1024,
    domain=16 * jnp.pi,
    video_length=10.0,
    video_fps=20,
    atol=1e-10,
    rtol=1e-10,
    C=1,
    κ=1,
    Dx=1e-4,
    Dy=1e-4,
    Dz=1e-5,
    νx=1e-2,
    νy=1e-4,
    νz=1e-5,
    ky=None,
    filename=None,
    seed=42,
    solver="Dopri8",
):

    ky = ky or find_ky(C=C, D=Dy, κ=κ, ν=νy)
    # take the first element if sized
    npx = grid_size[0] if hasattr(grid_size, "__len__") else grid_size
    lx = domain[0] if hasattr(domain, "__len__") else domain

    da = hasegawa_wakatani_spectral_2D(
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

    # overwrite the model name
    da.attrs.update({
        "model": "hasegawa_wakatani_spectral_1D",
        "ky": ky,
    })
    da.to_zarr(filename, mode="w")


def open_spectral_with_vorticity(filename):
    with xr.open_dataarray(filename, engine="zarr") as da:
        npx, npy = da.attrs["grid_size"]
        lx, ly = da.attrs["domain"]
        da = da.load()

    if "Ω" in da.coords["field"].values:
        return da

    # add vorticity field
    grid, kx, ky = gridmesh_from_da(da)
    vorticity = jnp.fft.irfft2(
        -(np.square(kx) + jnp.square(ky))
        * jnp.fft.rfft2(jnp.array(da.sel(field="φ"))),
        norm="forward",  # no normalization
    ) / npx / npy  # we normalize with the padding size

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


def plot_spectral_1D(filename):
    da = open_spectral_with_vorticity(filename)

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


def hasegawa_wakatani_growth_rate(ky, C, D, κ, ν):
    ksq = np.square(ky)
    a = (D*ksq + C + C/ksq + ν*ksq) / 2
    b = (D*ksq + C - C/ksq - ν*ksq) / 2
    g = np.square(b) + np.square(C / ky)
    h = np.sqrt(np.square(g) + np.square(C * κ / ky))
    j = np.sqrt((h+g) / 2)
    return j - a


def find_ky(C, D, κ, ν):
    return scipy.optimize.minimize_scalar(
        fun=(
            lambda ky,
            C,
            D,
            κ,
            ν: -hasegawa_wakatani_growth_rate(ky, C, D, κ, ν)
        ),
        bounds=(1e-4, 10),
        args=(C, D, κ, ν)
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
    C=1,
    κ=1,
    Dx=1e-4,
    Dy=1e-4,
    Dz=0,
    νx=1e-2,
    νy=1e-4,
    νz=0,
    ky=None,
    boundary="periodic",
    tf=300,
    domain=16 * jnp.pi,
    grid_size=682,
    acc=2,
    atol=1e-3,
    rtol=1e-3,
    seed=42,
    solver="Dopri8",
    video_length=10,
    video_fps=20,
    filename=None,
):
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
        grid_ = grids.Grid(
            shape=(int(jnp.ceil(grid_size * 3 / 4) * 2), 6),
            domain=[(0, domain), (0, 2 * jnp.pi / ky)]
        )
        kxs, kys = rfft_mesh(grid_)
        y = init_hw_spectral_2d(grid_, key, n=2, A=A, σ=σ)
        #convert to Ω
        y = y.at[..., 0].set(-y[..., 0] * (jnp.square(kxs) + jnp.square(kys)))
        y = unpad(y, grid_, axes=(0, 1))
        y = jnp.fft.rfft(jnp.fft.irfft2(y, axes=(0, 1)), axis=1)

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

    def explicit(t, y, args):  # convection, non-linear
        Ωk, Ωb, nk, nb = split_state(y)
        φk, φb = ddxk_inv @ Ωk, ddx_inv @ Ωb
        φc = jnp.conjugate(φk)
        dφbdx = dx_force @ φb

        return ky * combine_state(
            dΩk=1j * (dx_force@Ωb*φk - dφbdx*Ωk),
            dnk=1j * (dx_force@nb*φk - dφbdx*nk),
            dΩb=2 * dx @ jnp.imag(φc * Ωk),
            dnb=2 * dx @ jnp.imag(φc * nk) + forcing/ky,
        )

    def implicit(t, y, args):  # diffusion
        Ωk, Ωb, nk, nb = split_state(y)
        φk = ddxk_inv @ Ωk
        c_term = C * (φk-nk)

        return combine_state(
            dΩk=c_term + νx*ddx@Ωk - νy*ky2*Ωk,
            dnk=c_term + Dx*ddx@nk - Dy*ky2*nk - 1j*ky*κ*φk,
            dΩb=-νz * Ωb,
            dnb=(Dz-well) * nb,
        )

    def step(t, y, args):
        Ωk, nk, Ωb, nb = split_state(y)
        φk, φb = ddxk_inv @ Ωk, ddxk_inv @ Ωb

        if bc_name != "periodic":
            Ωk, nk, φk = [χ.at[nz].set(rhs) for χ in (Ωk, nk, φk)]
            Ωb, nb, φb = [χ.at[nz_force].set(rhs_force) for χ in (Ωb, nb, φb)]

        φc = jnp.conjugate(φk)
        dφbdx = dx_force @ φb
        c_term = C * (φk-nk)

        return combine_state(
            dΩk=-1j * ky * (dφbdx*Ωk - (dx_force@Ωb) * φk) + c_term + νx*ddx@Ωk - νy*ky2*Ωk,
            dnk=-1j * ky * (dφbdx*nk + (κ - dx_force@nb) * φk) + c_term + Dx*ddx@nk - Dy*ky2*nk,
            dΩb=2 * ky * dx @ jnp.imag(φc * Ωk) - νz*Ωb,
            dnb=2 * ky * dx @ jnp.imag(φc * nk) + (Dz-well) * nb + forcing,
        ) # yapf: disable

    def step_(t, y, args):
        return explicit(t, y, args) + implicit(t, y, args)

    imex_solvers = ["KenCarp5"]

    def decompose(y):
        """decompose real and imag parts"""
        Ωk, nk, Ωb, nb = split_state(y)
        return jnp.stack(
            arrays=[
                jnp.real(Ωk), jnp.imag(Ωk), jnp.real(nk), jnp.imag(nk), Ωb, nb
            ],
            axis=1
        )

    def recompose(y):
        Ωk_real, Ωk_imag, nk_real, nk_imag, Ωb, nb = y
        return combine_state(
            Ωk_real + 1j*Ωk_imag, nk_real + 1j*nk_imag, Ωb, nb
        )

    return simulation_base(
        terms=MultiTerm(ODETerm(explicit), ODETerm(implicit))
        if solver in imex_solvers else ODETerm(step),
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


def plot_components_1D(filename, all=True):
    if not isinstance(filename, xr.DataArray):
        with xr.open_dataarray(filename, engine="zarr") as da:
            da = da.load()
    else:
        da = filename
        filename = da.attrs["filename"]

    Ωk_real, Ωk_imag, nk_real, nk_imag, Ωb, nb = jnp.array(da.transpose("field", ...))
    Ωk = Ωk_real  #jnp.abs(Ωk_real + 1j * Ωk_imag)
    nk = nk_real  #jnp.abs(nk_real + 1j * nk_imag)
    # Ω2φ = da.attrs["Ω2φ"]
    domain = da.attrs["domain"]
    grid_size = da.attrs["grid_size"]
    tf = da.attrs["tf"]
    κ = da.attrs["κ"]
    boundary = da.attrs.get("boundary", "periodic").split()
    x = da.coords["x"].values

    # φk = jnp.real(Ω2φ @ Ωk[..., None]).squeeze()
    # φb = jnp.real(Ω2φ @ Ωb[..., None]).squeeze()

    nrows = 2
    ncolumns = 3
    fig, axes = plt.subplots(nrows, ncolumns, figsize=(7, 4))  #(2, 15))

    for i, (k, v) in enumerate({
            "$\Omega_k$": Ωk,
            "$\overline{\Omega}$": Ωb,
            "$\Omega$": Ωb + 2*Ωk,
            "$n_k$": nk,
            "$\overline{n}$": nb,
            "$n$": nb + 2*nk,
            # "$\phi_k$": φk,
            # "$\overline{\phi}$": φb,
            # "$\phi$": φb + 2 * φk
    }.items()):
        ax = axes.ravel()[i]
        vmax = jnp.maximum(jnp.abs(v.min()), v.max())
        im = ax.pcolormesh(
            x,
            jnp.linspace(0, tf, v.shape[0]),
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


def solve(diffeqsolve_kwargs, max_memory_will_use, save):
    """Split the simulation if the memory usage is too large."""

    def solve_(kwargs):
        return diffeqsolve(**kwargs)

    available_memory = get_available_memory()
    ts = diffeqsolve_kwargs["saveat"].subs.ts

    if max_memory_will_use > available_memory:
        n_iters = int(max_memory_will_use / available_memory) + 1
        tss = jnp.array_split(ts, n_iters)
        for i in range(n_iters):
            diffeqsolve_kwargs.update({
                "saveat":
                SaveAt(ts=tss[i], controller_state=True, solver_state=True),
                "t0":
                tss[i][0],
                "tf":
                tss[i][-1],
            })
            sol = solve_(**diffeqsolve_kwargs)
            diffeqsolve_kwargs.update({
                "y0": sol.ys[-1],
                "controller_state": sol.controller_state,
                "solver_state": sol.solver_state,
            })
            da = save(sol.ys)
    else:
        da = save(solve_(diffeqsolve_kwargs).ys, ts)

    return da


def simulation_base(
    terms,
    tf,
    coords,
    attrs,
    y0,
    solver="Dopri8",
    atol=1e-6,
    rtol=1e-6,
    video_length=1,
    video_fps=20,
    apply=None,
    filename=None,
):
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

    if file_path.exists():
        with xr.open_dataarray(file_path, engine="zarr") as da:
            # da_old = da.load()
            # if all attrs are the same and tf is greater than the previous tf, it means we want to continue
            same = jnp.array([(
                np.array(v).shape == np.array(da.attrs[k]).shape
                and v == type(v)(da.attrs[k])
            ) for (k, v) in attrs.items() if (
                isinstance(v, (int, float, str, tuple, list)) and k in da.attrs
            )]).all()

            if tf > da.attrs.get("tf", float("inf")) and same:
                t0 = da.attrs["tf"]
                y0 = jnp.array(da.isel(time=-1))
                runtime_old = da.attrs.get("runtime", 0)
                resume = True
                print(f"The simulation is resumed from {filename}.")

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
            coords={
                "time": []
            } | coords,
        ).to_zarr(
            file_path, mode="w"
        )

        runtime_old = 0
        t0 = 0
        # the number of frames saved is video_nframes, it is not proportional to tf
        time_linspace = jnp.linspace(t0, tf, video_nframes)

    attrs_ = attrs | {
        "tf": tf,
        "video_length": video_length,
        "video_fps": video_fps,
        "solver": solver,
        "atol": atol,
        "rtol": rtol,
    }

    for k, v in attrs_.items():
        # print only short parameters
        if not isinstance(v, (np.ndarray, )):
            print(f"{k:<20}: {v}")

    max_memory_will_use = (
        video_nframes * jnp.prod(jnp.array(y0.shape)) * {
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

        if filename is not None:
            da.to_zarr(
                file_path, append_dim="time" if file_path.exists() else None
            )

        return da

    t1 = timer()
    with tqdm.auto.tqdm(
            total=tf,
            desc="Simulation",
            bar_format=
            "{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            initial=t0,
    ) as tqdm_bar:

        diffeqsolve_kwargs = dict(
            terms=terms,
            solver=SolverWrapTqdm(
                solvers[solver],
                tqdm_bar=tqdm_bar,
                nonlinear_solver=nonlinear_solver
            ),
            t0=t0,
            t1=tf,
            dt0=1e-3,
            y0=y0,
            saveat=SaveAt(ts=time_linspace),
            stepsize_controller=PIDController(atol=atol, rtol=rtol),
            max_steps=None,
        )
        da = solve(diffeqsolve_kwargs, max_memory_will_use, save)

    # add the runtime in the attributes
    da.attrs["runtime"] = runtime_old + timer() - t1
    if filename is not None:
        da.to_zarr(file_path, mode="w")

    return da


def diff_matrix(
    grid, axis, order, acc=2, bc_name="periodic", bc_value=0, csr=False
):
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
            nones = tuple([None] * i)
            bc[nones + (0,)], bc[nones + (-1,)] = {
                "dirichlet": (bc_value, bc_value),
                "neumann": ((find, bc_value), (find, bc_value)),
                "force": ((find, 0), 1e-1),  # neumann ad dirichlet
            }[bc_name]
        nz = jnp.array(bc.row_inds())
        rhs = jnp.array(bc.rhs[nz].toarray()).ravel()

        op = op.matrix(shape=grid.shape)  # convert to scipy sparse
        op[nz, :] = bc.lhs[nz, :]  # apply boundary conditions
        # convert to jax sparse
        if csr:
            op = op.tocsr()
            op = sparse.BCSR((op.data, op.indices, op.indptr), shape=op_shape)
        else:
            op = op.tocoo()
            op = sparse.BCOO((op.data, jnp.stack([op.row, op.col], axis=-1)),
                             shape=op_shape)
    return op, nz, rhs


def hasegawa_wakatani_finite_difference_2D(
    tf=10,
    grid_size=128,
    domain=16 * jnp.pi,
    video_length=10.0,
    video_fps=20,
    atol=1e-9,
    rtol=1e-6,
    C=1,
    κ=1,
    D=1e-1,
    Dz=1e-2,
    ν=1e-1,
    νz=1e-2,
    boundary="periodic",
    acc=2,
    filename=None,
    seed=42,
    solver="Dopri8",
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
    laplacian_csr, nz, rhs = diff_matrix(
        axis=[0, 1], order=2, **diff_matrix_kwargs, csr=True
    )

    # print("computing the pseudo-inverse: very slow !")
    # laplacian_inv = jnp.linalg.pinv(laplacian_bcoo.todense(), hermitian=True)

    if bc_name == "periodic":
        y0 = init_hw_spectral_2d(grid, key=jax.random.PRNGKey(seed=seed), n=2)
        y0 = jnp.fft.irfft2(y0, axes=(0, 1))
    else:
        xv, yv = jnp.meshgrid(jnp.linspace(0, lx, nx), jnp.linspace(0, ly, ny))
        σx, σy = lx / 10, ly / 10
        n = jnp.exp(
            -(jnp.square((xv - lx/2) / σx) + jnp.square((yv - ly/2) / σy))
        ) * 1e-2
        Ω = jnp.zeros(grid.shape)
        Ω, n = [χ.at[nz].set(rhs) for χ in (Ω, n)]
        y0 = jnp.stack([Ω, n], axis=-1)

    def dx(y):
        return (dx_bcoo @ y.ravel()).reshape(nx, ny)

    def dy(y):
        return (dy_bcoo @ y.ravel()).reshape(nx, ny)

    def laplacian(y):
        return (laplacian_bcoo @ y.ravel()).reshape(nx, ny)

    def step(t, y, args):
        Ω, n = y[..., 0], y[..., 1]
        # φ = (laplacian_inv @ Ω.ravel()).reshape(nx, ny)
        φ = jax.scipy.sparse.linalg.cg(
            lambda x: laplacian_bcoo @ x, Ω.ravel(), tol=rtol
        )[0].reshape(nx, ny)
        # φ = sparse.linalg.spsolve(laplacian_csr.data, laplacian_csr.indices, laplacian_csr.indptr, Ω.ravel(), tol=atol).reshape(nx, ny)
        Ωb, nb, φb = [jnp.mean(χ, axis=1) for χ in (Ω, n, φ)]
        Ωt, nt, φt = Ω - Ωb, n - nb, φ - φb
        c_term = C * (φt-nt)

        dφdx = dx(φ)
        dφdy = dy(φ)

        dΩ = dφdy * dx(Ω) - dφdx * dy(Ω) + c_term + ν * laplacian(Ωt) - νz*Ωb
        dn = dφdy * dx(n) - dφdx * dy(n) + c_term + D * laplacian(nt) - Dz*nb
        dn -= κ * dφdy
        return jnp.stack(arrays=[dΩ, dn], axis=-1)

    return simulation_base(
        terms=ODETerm(step),
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


def main():
    parser = ArgumentParser("Hasegawa Wakatani simulation")
    parser.add_argument("filename")
    parser.add_argument("--tf", type=float)
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

    schemes = {
        "hasegawa_wakatani_spectral_2D":
        (hasegawa_wakatani_spectral_2D, visualization_2D),
        "hasegawa_wakatani_spectral_1D":
        (hasegawa_wakatani_spectral_1D, plot_spectral_1D),
        "hasegawa_wakatani_finite_difference_1D":
        (hasegawa_wakatani_finite_difference_1D, plot_components_1D),
        "hasegawa_wakatani_finite_difference_2D":
        (hasegawa_wakatani_finite_difference_2D, visualization_2D),
    }
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
