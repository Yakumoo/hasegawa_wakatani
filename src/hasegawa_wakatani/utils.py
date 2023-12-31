import os
from typing import Callable, Optional, Tuple, Union, Any, Sequence
from timeit import default_timer as timer
from pathlib import Path

import numpy as np
import scipy
from findiff import FinDiff
import zarr
import xarray as xr
import tqdm.auto

import jax
import jax.numpy as jnp
from jax.experimental import host_callback, sparse

Array = Any
KeyArray = Union[Array, jax._src.prng.PRNGKeyArray]

import diffrax
from diffrax import (
    diffeqsolve,
    MultiTerm,
    ODETerm,
    SaveAt,
    PIDController,
    AbstractWrappedSolver,
    AbstractAdaptiveSolver,
    AbstractTerm,
)
import jax_cfd.base.grids as grids


def get_padded_shape(shape: Sequence[int]):
    """Get the padded shape using the 2/3 zero padding rule

    We use np instead of jnp because a shape should be considered as a static parameter

    Args:
        shape: the (full) space shape without the symmetry, (not the half shape)
    Return:
        the padded shape (full, not the half)
    """
    padded_shape = np.array(shape) * 3 / 2
    return np.where(
        padded_shape % 2 == 0, padded_shape, np.ceil(padded_shape / 2) * 2
    ).astype(int)


def make_hermitian(a, axes=(0, 1)):
    """Make the 2D Fourier space hermitian

    Symmetrize (conjugate) along kx in the Fourier space
    and set the zero and the Nyquist frequencies to zero

    Args:
        a: complex array of shape (kx, ky, ...)
    """
    b = jnp.moveaxis(a, axes, range(len(axes)))
    if len(axes) == 2:
        x, y = b.shape[:2]
        b = b.at[-1 : x // 2 : -1, 0].set(jnp.conj(b[1 : x // 2, 0]))
        b = b.at[x // 2, :].set(0)
        b = b.at[:, -1].set(0)
        b = b.at[0, 0].set(jnp.real(b[0, 0]))
    elif len(axes) == 3:
        x, y, z = b.shape[:3]
        b = (
            b.at[-1 : x // 2 : -1, -1 : y // 2 : -1, 0]
            .set(jnp.conj(b[1 : x // 2, 1 : y // 2, 0]))
            .at[-1 : x // 2 : -1, 1 : y // 2, 0]
            .set(jnp.conj(b[1 : x // 2, -1 : y // 2 : -1, 0]))
        )
        b = b.at[x // 2].set(0)
        b = b.at[:, y // 2].set(0)
        b = b.at[:, :, -1].set(0)
        b = b.at[0, 0, 0].set(jnp.real(b[0, 0, 0]))
    else:
        raise NotImplementedError("Only 2D and 3D are implemented.")

    return jnp.moveaxis(b, range(len(axes)), axes)


class SolverWrapTqdm(AbstractWrappedSolver):
    """Solver wrapper for progress bar

    It shows the tqdm progress bar while calling diffeqsolve
    at each `dt` interval of the simulation
    """

    pbar: tqdm.auto.tqdm
    # controls the simulation time interval for updating pbar
    dt: Union[int, float] = 1e-1

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
                tap_func=(lambda t, tf: self.pbar.update(t - self.pbar.n)),
                arg=t1,
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
    alphas = jnp.array(
        [0, 0.1496590219993, 0.3704009573644, 0.6222557631345, 0.9582821306748, 1]
    )
    betas = jnp.array(
        [0, -0.4178904745, -1.192151694643, -1.697784692471, -1.514183444257]
    )
    gammas = jnp.array(
        [
            0.1496590219993,
            0.3792103129999,
            0.8229550293869,
            0.6994504559488,
            0.1530572479681,
        ]
    )
    μdt = jnp.diff(alphas) / 2  # precompute the diff

    def init(self, terms, t0, t1, y0, args):
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

        # first iteration out of the loop so we can save ex0 and im0 for euler_y1
        ex0 = term_ex.vf(t0, y0, args)
        im0 = term_im.vf(t0, y0, args)
        h = ex0
        μ = dt * μdt[0]  # (α[1] - α[0]) / 2
        y1 = term_solve.vf(
            t0 + α[1] * dt, y0 + γ[0] * dt * h + μ * im0, μ
        )  # parse time_step as args

        # loop from 1
        for k in range(1, len(β)):
            tk = t0 + α[k] * dt
            h = term_ex.vf(tk, y1, args) + β[k] * h
            μ = dt * μdt[k]  # (α[k+1] - α[k]) / 2
            y1 = term_solve.vf(
                t0 + α[k + 1] * dt, y1 + γ[k] * dt * h + μ * term_im.vf(tk, y1, args), μ
            )

        euler_y1 = y0 + dt * (ex0 + im0)
        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)
        return y1, y_error, dense_info, None, diffrax.RESULTS.successful

    def func(self, terms, t0, y0, args):
        term_ex, term_im, term_solve = terms
        f1 = term_ex.vf(t0, y0, args)
        f2 = term_im.vf(t0, y0, args)
        return f1 + f2


def fourier_pad(y, axes=(-2, -1)):
    space_shape = [y.shape[x] for x in axes]
    space_shape[-1] = (space_shape[-1] - 1) * 2 + 1
    space_shape_padded = get_padded_shape(space_shape)
    pad_width = [[0, 0] for _ in y.shape]
    for i, axis in enumerate(axes):
        pad_width[axis] = [(space_shape_padded[i] - space_shape[i]) // 2] * 2
    pad_width[axes[-1]][0] = 0

    y_padded = jnp.fft.fftshift(y, axes=axes[:-1])
    y_padded = jnp.pad(array=y_padded, pad_width=pad_width)
    y_padded = jnp.fft.ifftshift(y_padded, axes=axes[:-1])

    return y_padded


def fourier_unpad(y: Array, axes: tuple[int, int] = (-2, -1)) -> Array:
    """Unpad the Fourier space

    Remove the zero padding (2/3 rule). Thus, the final shape is smaller

    Args:
        y: complex array with shape (..., kx, ky, ...)
        grid: the grid object with the padded shape
        axes: axis position of kx, ky, must be consecutive

    Return:
        the unpadded array of `y`
    """
    shape = list(y.shape)
    shape[axes[-1]] = (shape[axes[-1]] - 1) * 2 + 1
    select = [slice(None)] * y.ndim
    for axis in axes:
        pad_width = (shape[axis] - int(shape[axis] / 3) * 2) // 2
        select[axis] = slice(np.where(axis == axes[-1], 0, pad_width), -pad_width)

    y_unpadded = jnp.fft.fftshift(y, axes=axes[:-1])[tuple(select)]
    y_unpadded = jnp.fft.ifftshift(y_unpadded, axes=axes[:-1])

    return y_unpadded


def init_fields_fourier_2d(
    grid: grids.Grid,
    key: KeyArray,
    n: int = 1,
    A: float = 1e-4,
    σ: float = 0.5,
    laplacian=False,
) -> Array:
    """Create the initial fields in the fourier space

    Args:
        grid: Grid object without the padding shape
        key: for creating the random fields
        n: the number of fields
        A: amplitude of the gaussian
        σ: standard deviation of the gaussian
        laplacian: apply laplacian tot he first field or not

    Return:
        array of shape (grid_x, grid_y // 2 + 1) + ((n,) if n>1 else tuple())
    """

    ks = rfft_mesh(grid)
    k2 = jnp.square(ks).sum(0)
    ŷ0 = A * jnp.exp(
        -k2[..., None] / 2 / jnp.square(σ)
        + 2j * jnp.pi * jax.random.uniform(key, k2.shape + (n,))
    )

    if laplacian:
        ŷ0 = ŷ0.at[..., 0].set(-k2 * ŷ0[..., 0])

    ŷ0 = make_hermitian(ŷ0).squeeze()

    return ŷ0


def process_space_params(
    grid_size, domain, ndim=2
) -> tuple[Sequence[int], Sequence[float], grids.Grid]:
    if jnp.isscalar(domain):
        ls = [jnp.array(domain).item() for _ in range(ndim)]
    else:
        assert len(domain) == ndim
        ls = tuple(domain)

    if jnp.isscalar(grid_size):
        ns = [grid_size for i in range(ndim)]
    else:
        assert len(grid_size) == ndim
        ns = grid_size

    assert jnp.issubdtype(
        jnp.array(ns).dtype, jnp.integer
    ), "grid_size must be integers."

    grid = grids.Grid(ns, domain=[(0, x) for x in ls])

    return ns, ls, grid


def fourier_to_real(t: float, ŷ: Array, args=None, ndim=2) -> Array:
    """Convert the simulation data to direct space"""
    axes = tuple(range(ndim))
    # move to cpu to handle large data
    y = jax.device_put(ŷ, device=jax.devices("cpu")[0]).view(dtype=complex)
    y = make_hermitian(y, axes=axes)
    y = jnp.fft.irfftn(y, axes=axes, norm="forward")
    return y.reshape(*y.shape[:ndim], -1)


def real_to_fourier(y: Array) -> Array:
    """Convert the initial state to Fourier space"""
    return jnp.fft.rfftn(y.squeeze(), axes=range(y.ndim - 1), norm="forward").view(
        dtype=float
    )


def get_terms(m, solver):
    def term(t, y, args=None):
        return m.explicit_terms(y) + m.implicit_terms(y)

    return (
        (
            ODETerm(lambda t, y, args: m.explicit_terms(y)),
            ODETerm(lambda t, y, args: m.implicit_terms(y)),
            ODETerm(lambda t, y, dt: m.implicit_solve(y, dt)),
        )
        if solver == "CrankNicolsonRK4"
        else ODETerm(term)
    )


def rfft_mesh(grid):
    return 2 * jnp.pi * jnp.array(grid.rfft_mesh())


def gridmesh_from_da(da: xr.DataArray) -> tuple[grids.Grid, Array]:
    grid = grids.Grid(
        da.attrs["grid_size"], domain=[(0, x) for x in da.attrs["domain"]]
    )
    return grid, rfft_mesh(grid)


def open_with_vorticity(filename) -> xr.DataArray:
    """Open the .zarr file

    It adds the vorticity field Ω if needed
    """
    with xr.open_dataarray(filename, engine="zarr") as da:
        da = da.load()

    da.attrs.update(zarr.open(filename).attrs)

    if "Ω" in da.coords["field"].values:
        return da

    # add vorticity field
    grid, ks = gridmesh_from_da(da)
    φs = [c for c in da.coords["field"].values if "φ" in c]
    axes = (1, 2, 3) if "z" in da.coords else (1, 2)
    vorticity = jnp.fft.irfftn(
        -jnp.square(ks).sum(0)[..., None]
        * jnp.fft.rfftn(jnp.array(da.sel(field=φs)), axes=axes, norm="forward"),
        axes=axes,
        norm="forward",
    )
    vorticity = xr.DataArray(
        vorticity,
        dims=da.dims,
        coords={
            "time": da.time,
            "x": da.x,
            "y": da.y,
            "field": [field_name.replace("φ", "Ω") for field_name in φs],
        },
    )
    return xr.concat((da, vorticity), dim="field")


def hw_growth_rate(ky, C, D, κ, ν):
    ksq = jnp.square(ky)
    a = (D * ksq + C + C / ksq + ν * ksq) / 2
    b = (D * ksq + C - C / ksq - ν * ksq) / 2
    g = jnp.square(b) + jnp.square(C / ky)
    h = jnp.sqrt(jnp.square(g) + jnp.square(C * κ / ky))
    j = jnp.sqrt((h + g) / 2)
    return j - a


def find_ky(C: float, D: float, κ: float, ν: float) -> float:
    return scipy.optimize.minimize_scalar(
        fun=(lambda ky: -hw_growth_rate(ky, C, D, κ, ν)),
        bounds=(1e-4, 10),
    ).x.item()


def get_available_memory() -> float:
    """Return the available memory in GB"""
    if jax.lib.xla_bridge.get_backend().platform == "gpu":
        import subprocess

        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        memory_info = (
            subprocess.check_output(command.split())
            .decode("ascii")
            .split("\n")[:-1][1:]
        )
        memory_values = [int(x.split()[0]) for i, x in enumerate(memory_info)]
        memory_value = memory_values[0]  # take the first gpu
        return (
            memory_value
            * float(os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.8))
            / 1000
        )
    else:
        import psutil

        return psutil.virtual_memory()[3] / 1000000000


def solve_save(
    diffeqsolve_kwargs,
    max_memory_will_use,
    save,
    split_callback=None,
):
    """Split the simulation if the memory usage is too large."""

    available_memory = get_available_memory()
    ts = diffeqsolve_kwargs["saveat"].subs.ts
    pbar = diffeqsolve_kwargs["solver"].pbar

    if max_memory_will_use > available_memory:
        y0 = diffeqsolve_kwargs["y0"]
        fn = diffeqsolve_kwargs["saveat"].subs.fn
        n_iters = int(max_memory_will_use / available_memory) + 1
        tss = jnp.array_split(ts, n_iters)
        for i in range(n_iters):
            t1 = tss[i][-1].item()
            diffeqsolve_kwargs.update(
                {
                    "saveat": SaveAt(
                        ts=tss[i], controller_state=True, solver_state=True, fn=fn
                    ),
                    "t1": t1,
                }
            )
            sol = diffeqsolve(**diffeqsolve_kwargs)
            pbar.update(t1 - pbar.n)
            # get the last state from sol.solver_state
            _y0, _ = jax.tree_util.tree_flatten(sol.solver_state)
            _y0 = _y0[np.where([x.shape == y0.shape for x in _y0])[0][0]]

            diffeqsolve_kwargs.update(
                {
                    "y0": _y0,
                    "controller_state": sol.controller_state,
                    "solver_state": sol.solver_state,
                    "t0": t1,
                }
            )
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
    apply: Optional[
        tuple[Callable[[Array], Array], Callable[[float, Array, Any], Array]]
    ] = None,
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
            When resuming a simulation: y0 = to_diffeqsolve(y)
            For converting to DataArray: to_dataarray(t, y, args=None)
        filename: the output file name, expected to be a .zarr file
        split_callback: callback when the simulation is splitted


    """
    file_path = Path(filename)
    resume = False

    solvers = {
        "Tsit5": diffrax.Tsit5(),
        "Dopri5": diffrax.Dopri5(),
        "Dopri8": diffrax.Dopri8(),
        "CrankNicolsonRK4": CrankNicolsonRK4(),
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
            same = jnp.array(
                [
                    np.array_equal(v, da.attrs[k])
                    for (k, v) in attrs_.items()
                    if k in da.attrs
                ]
            ).all()

            if (
                tf > da.attrs.get("tf", float("inf"))
                and len(da.coords["time"]) > 0
                and same
            ):
                t0 = da.attrs["tf"]
                y0 = jnp.array(da.isel(time=-1))
                resume = True
                print(f"The simulation is resumed from {filename}.")

    attrs_.update({"tf": tf})
    dims = ["time"] + list(coords.keys())
    video_nframes = int(video_length * video_fps)

    if resume:
        time_linspace = jnp.linspace(
            t0 + 1 / video_fps, tf, int((tf - t0) / tf * video_nframes)
        )
        if apply:
            y0 = apply[0](y0)
    else:
        # overwrite with an empty file
        y_dummy = apply[1](0, y0, None) if apply is not None else y0
        xr.DataArray(
            data=jnp.array([]).reshape(-1, *y_dummy.shape),
            dims=dims,
            coords={"time": []} | coords,
            attrs=attrs_,
        ).to_zarr(file_path, mode="w")

        t0 = 0
        # the number of frames saved is video_nframes, it is not proportional to tf
        time_linspace = jnp.linspace(t0, tf, video_nframes + 1)

    for k, v in attrs_.items():
        # print only short aparameters
        if isinstance(v, str) or not (hasattr(v, "__len__") and len(v) > 10):
            print(f"{k:<20}: {v}")

    max_memory_will_use = (
        y0.size
        * video_nframes
        * {jnp.dtype("float64"): 8, jnp.dtype("float32"): 4}[y0.dtype]  # bytes
        * 4  # it seems it will be buffered twice + 2 for safety
        / 1000000000  # convert to GB
    )

    def save(ys, ts):
        da = xr.DataArray(
            ys,
            dims=dims,
            coords={"time": ts} | coords,
            attrs=attrs_,
        )

        da.to_zarr(file_path, append_dim="time")

        return da

    if apply is not None:
        save_fn = apply[1]
    else:
        # by default we put in the CPU because it might not fit in the GPU
        def save_fn(t, y, args=None):
            return jax.device_put(y, device=jax.devices("cpu")[0])

    t1 = timer()
    with tqdm.auto.tqdm(
        total=tf,
        desc="Simulation",
        bar_format="{l_bar}{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        initial=t0,
    ) as pbar:
        diffeqsolve_kwargs = {
            "terms": terms,
            "solver": SolverWrapTqdm(solvers[solver], pbar),
            "t0": t0,
            "t1": tf,
            "dt0": 1e-3,
            "y0": y0,
            "saveat": SaveAt(ts=time_linspace, fn=save_fn),
            "stepsize_controller": PIDController(atol=atol, rtol=rtol),
            "max_steps": None,
        }
        da = solve_save(
            diffeqsolve_kwargs,
            max_memory_will_use,
            save,
            split_callback,
        )
        pbar.update(tf - pbar.n)

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
    padding: bool = False,
) -> sparse.BCOO:
    """Compute the differential matrix operator

    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Example usage for dirichlet boundary conditions (we set 0 at the edges):
    `dy = (op @ y.at[nz].set(0).ravel()).reshape(y.shape)`

    Args:
        grid: the grid object
        axis: axis coordinate for derivative. For example:
            0 => ∂x, 1 => ∂y, 3 => ∂z, [0, 1] => ∂x + ∂y
        order: derivative order, ∂^order
        acc: order of accuracy, must be a positive even integer
        padding: if True, there is no padding and we suppose boudaries are periodic
            otherwise, a padding is added to the matrix

    Return:
        op: differential sparse 2D matrix (jax or scipy) operator with shape (x, x) where
            x = np.prod(grid.shape) + (acc if padding else 0)
    """
    params = locals()
    assert (
        acc < min(grid.shape) / 2
    ), f"acc={acc} is too big. The grid.shape is {grid.shape}."

    if isinstance(axis, int):
        op = FinDiff(axis, grid.step[axis], order, acc=acc)
    else:
        op = sum([FinDiff(i, grid.step[i], order, acc=acc) for i in axis])

    square_size = jnp.prod(jnp.array(grid.shape)).item()
    op_shape = (square_size, square_size)

    if padding:
        nz = jnp.arange(square_size).reshape(grid.shape)
        mask = (
            jnp.ones(grid.shape)
            .at[tuple([slice(acc // 2, -acc // 2)] * grid.ndim)]
            .set(0)
        )
        nz = nz[mask.astype(bool)]

        op = op.matrix(shape=np.array(grid.shape) + acc)  # convert to scipy sparse
        # set identity to padding
        op[nz, :] = scipy.sparse.eye(op.shape[0], format="lil")[nz, :]
        op = sparse.BCOO.from_scipy_sparse(op)  # convert to jax sparse
    else:  # periodic boundaries
        pos = jnp.prod(jnp.array(grid.shape[:-1], dtype=int) + 1) * (acc - 1)
        op = op.matrix(grid.shape)[pos].toarray()
        row = jnp.roll(jnp.array(op), -pos).squeeze()
        indices = jnp.nonzero(row)[0]
        data = jnp.tile(row[indices], square_size)
        ones = jnp.ones_like(indices)
        indices = jnp.concatenate(
            jax.vmap(
                fun=lambda i: jnp.stack(
                    arrays=(ones * i, (indices + i) % square_size), axis=-1
                )
            )(jnp.arange(square_size)),
            axis=0,
        )
        op = sparse.BCOO((data, indices), shape=op_shape)

    if hasattr(op, "eliminate_zeros"):
        op.eliminate_zeros()

    return op


def diff_fn(
    grid: grids.Grid,
    axis: Union[int, list[int]],
    order: int,
    acc: int = 2,
    boundary: Union[
        str, Callable[[Array, tuple[int, int], int, dict], None]
    ] = "periodic",
):
    assert (
        acc < min(grid.shape) / 2
    ), f"acc={acc} is too big. The grid.shape is {grid.shape}."

    if callable(boundary):  # custom padding
        mode = boundary
    else:
        assert boundary in {
            "periodic",
            "dirichlet",
            "neumann",
        }, f"Invalid boundary={boundary}, accepted boundaries are: periodic, dirichlet, neumann"
        mode = {
            "periodic": "wrap",
            "dirichlet": "constant",
            "neumann": "edge",
        }[boundary]

    if isinstance(axis, int):
        op = FinDiff(axis, grid.step[axis], order, acc=acc)
    else:
        op = sum([FinDiff(i, grid.step[i], order, acc=acc) for i in axis])

    stencil = op.stencil(grid.shape).data[tuple("C" for _ in range(grid.ndim))]
    stencil = [
        (
            tuple(
                slice(acc // 2 + offset, acc // 2 + offset + grid.shape[i])
                for i, offset in enumerate(k)
            ),
            v,
        )
        for k, v in stencil.items()
    ]

    def fn(y):
        pad_width = [(0, 0)] * y.ndim
        for i in range(grid.ndim):
            pad_width[-i - 1] = (acc // 2, acc // 2)
        y_padded = jnp.pad(y, pad_width=pad_width, mode=mode)
        pre = tuple([slice(None)] * (y.ndim - grid.ndim))
        return sum([y_padded[pre + k] * v for k, v in stencil])

    return fn


def arakawa_fn(grid, boundary):
    mode = {
        "periodic": "wrap",
        "dirichlet": "constant",
        "neumann": "edge",
    }[boundary]
    arakawa_norm = 12 * jnp.prod(jnp.array(grid.step))
    c = slice(1, -1)
    p = slice(2, None)
    m = slice(0, -2)

    def _arakawa(ζ, ψ):
        ζψ = jnp.pad(jnp.array([ζ, ψ]), pad_width=[(0, 0), (1, 1), (1, 1)], mode=mode)
        ζ_east, ψ_east = ζψ[:, p, c]
        ζ_west, ψ_west = ζψ[:, m, c]
        ζ_north, ψ_north = ζψ[:, c, p]
        ζ_south, ψ_south = ζψ[:, c, m]
        ζ_ne, ψ_ne = ζψ[:, p, p]
        ζ_se, ψ_se = ζψ[:, p, m]
        ζ_sw, ψ_sw = ζψ[:, m, m]
        ζ_nw, ψ_nw = ζψ[:, m, p]

        return (
            ζ_east * (ψ_north - ψ_south + ψ_ne - ψ_se)
            - ζ_west * (ψ_north - ψ_south + ψ_nw - ψ_sw)
            - ζ_north * (ψ_east - ψ_west + ψ_ne - ψ_nw)
            + ζ_south * (ψ_east - ψ_west + ψ_se - ψ_sw)
            + ζ_se * (ψ_east - ψ_south)
            + ζ_ne * (ψ_north - ψ_east)
            - ζ_nw * (ψ_north - ψ_west)
            - ζ_sw * (ψ_west - ψ_south)
        ) / arakawa_norm

    return _arakawa


def append_total_1d(da):
    das = [da] + [
        da.sel(field=fields)
        .weighted(xr.DataArray([2, 1], coords={"field": fields}))
        .sum(dim="field")
        .expand_dims(dim={"field": [fields[0][0]]})
        for fields in [["Ωk_real", "Ωb"], ["nk_real", "nb"]]
    ]
    return (
        xr.concat(das, dim="field")
        .sel(field=["Ωk_real", "Ωb", "Ω", "nk_real", "nb", "n"])
        .assign_coords(
            {
                "field": [
                    "Real($\Omega_k$)",
                    "$\overline{\Omega}$",
                    "Ω",
                    "Real($n_k$)",
                    "$\overline{n}$",
                    "n",
                ]
            }
        )
    )
