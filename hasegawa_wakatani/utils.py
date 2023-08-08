import os
from typing import Callable, Optional, Tuple, Union, Any, Sequence
from timeit import default_timer as timer
from pathlib import Path

import numpy as np
import scipy
from findiff import FinDiff, BoundaryConditions
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
    Dopri5,
    Dopri8,
    MultiTerm,
    ODETerm,
    SaveAt,
    PIDController,
    AbstractWrappedSolver,
    NewtonNonlinearSolver,
    AbstractNonlinearSolver,
    AbstractAdaptiveSolver,
    AbstractTerm,
)
import jax_cfd.base.grids as grids


def get_padded_shape(shape):
    padded_shape = jnp.array(shape) * 3 / 2
    return jnp.where(
        padded_shape % 2 == 0, padded_shape, jnp.ceil(padded_shape / 2) * 2
    ).astype(int).tolist()


def brick_wall_filter_2d(nx, ny, nyquist=False):
    """Implements the 2/3 rule."""
    npx, npy = get_padded_shape([nx, ny])
    npx3 = npx // 3
    filter_ = jnp.zeros((npx, npy//2 + 1))
    if nyquist:
        npy3 = npy // 3
        filter_ = filter_.at[-npx3 + 1:, :npy3].set(1)
    else:
        npy3 = npy//3 + 1
        filter_ = filter_.at[-npx3:, :npy3].set(1)
    filter_ = filter_.at[:npx3, :npy3].set(1)

    return filter_


def make_hermitian(a):
    """Make the 2D Fourier space hermitian
    
    Symmetrize (conjugate) along kx in the Fourier space
    and set the zero and the Nyquist frequencies to zero
    
    Args:
        a: complex array of shape (kx, ky, ...)
    """
    x, y = a.shape[:2]
    return (a
        .at[-1:x // 2:-1, 0].set(jnp.conj(a[1:x // 2, 0]))
        .at[x // 2, :].set(0)
        .at[:, -1].set(0)
        .at[0, 0].set(jnp.real(a[0, 0]))
    ) # yapf: disable


class SolverWrapTqdm(AbstractWrappedSolver):
    """Solver wrapper for progress bar

    It shows the tqdm progress bar while calling diffeqsolve
    at each `dt` interval of the simulation
    """
    tqdm_bar: tqdm.auto.tqdm
    # controls the simulation time interval for updating tqdm_bar
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


def unpad(y: Array, axes: tuple[int, int] = (-2, -1)) -> Array:
    """Unpad the Fourier space
    
    Remove the zero padding (2/3 rule). Thus, the final shape is smaller

    Args:
        y: complex array with shape (..., kx, ky, ...)
        grid: the grid object with the padded shape
        axes: axis position of kx, ky, must be consecutive

    Return:
        the unpadded array of `y`
    """
    npx, npy = y.shape[axes[0]], y.shape[axes[1]]
    nx = int(npx / 3) * 2
    ny = int(jnp.ceil(npy / 3 * 2))
    mask = brick_wall_filter_2d(nx, ny).astype(bool)
    new_shape = list(y.shape)
    new_shape[axes[0]] = nx
    new_shape[axes[1]] = ny
    index = [slice(None)] * (y.ndim - 1)
    pos = y.ndim + axes[0] if axes[0] < 0 else axes[0]
    index[pos] = mask
    return y[tuple(index)].reshape(*new_shape)


def init_fields_fourier_2d(
    grid: grids.Grid,
    key: KeyArray,
    n: int = 1,
    A: float = 1e-4,
    σ: float = 0.5,
    padding=True,
    laplacian=False,
) -> Array:
    """Create the initial fields in the fourier space
    
    Args:
        grid: Grid object without the padding shape
        key: for creating the random fields
        n: the number of fields
        A: amplitude of the gaussian
        σ: standard deviation of the gaussian

    Return:
        array of shape (grid_x, grid_y // 2 + 1) + ((n,) if n>1 else tuple())
    """

    kx, ky = rfft_mesh(grid)
    k2 = jnp.square(kx) + jnp.square(ky)
    ŷ0 = A * jnp.exp(
        -k2[..., None] / 2 / jnp.square(σ)
        + 2j * jnp.pi * jax.random.uniform(key, kx.shape + (n, ))
    )

    if laplacian:
        ŷ0 = ŷ0.at[..., 0].set(-k2 * ŷ0[..., 0])

    if padding:
        # now we pad
        mask = brick_wall_filter_2d(*grid.shape)
        empty = jnp.tile(mask[..., None], (1, 1, n)).astype(complex)
        ŷ0 = empty.at[mask.astype(bool)].set(ŷ0.reshape(-1, n))

    ŷ0 = make_hermitian(ŷ0).squeeze()

    return ŷ0


def process_params_2d(grid_size, domain):
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


def fourier_to_real(ŷ: Array) -> Array:
    """Convert the simulation data to direct space"""

    # move to cpu to handle large data
    y = jax.device_put(ŷ, device=jax.devices("cpu")[0]).view(dtype=complex)
    y = unpad(y, axes=(1, 2))
    y = jnp.fft.irfft2(y, axes=(1, 2), norm="forward")
    nt, nx, ny = y.shape[:3]
    return y.reshape(nt, nx, ny, -1)


def real_to_fourier(y: Array) -> Array:
    """Convert the initial state to Fourier space"""
    nx, ny = y.shape[:2]
    npx, npy = get_padded_shape([nx, ny])
    y0 = y.squeeze()
    y = jnp.fft.rfft2(y0, axes=(0, 1), norm="forward")
    return (  # padding
        jnp.zeros((npx, npy // 2 + 1, 2), dtype=complex)
        .at[brick_wall_filter_2d(nx, ny).astype(bool)]
        .set(y.reshape(-1, *y0.shape[2:]))
        .view(dtype=float)
    )


def get_terms(m, solver):

    def term(t, y, args=None):
        return m.explicit_terms(y) + m.implicit_terms(y)

    terms = ((
        ODETerm(lambda t, y, args: m.explicit_terms(y)),
        ODETerm(lambda t, y, args: m.implicit_terms(y)),
        ODETerm(lambda t, y, dt: m.implicit_solve(y, dt)),
    ) if solver == "CrankNicolsonRK4" else ODETerm(term))

    return terms


def rfft_mesh(grid):
    return 2 * jnp.pi * jnp.array(grid.rfft_mesh())


def gridmesh_from_da(da):
    lx, ly = da.attrs["domain"]
    nx, ny = da.coords["x"].size, da.coords["y"].size
    grid = grids.Grid((nx, ny), domain=((0, lx), (0, ly)))
    kx, ky = rfft_mesh(grid)
    return grid, kx, ky


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
    φs = [c for c in da.coords["field"].values if "φ" in c]
    vorticity = jnp.fft.irfft2(
        -(jnp.square(kx) + jnp.square(ky))[..., None] * jnp.fft
        .rfft2(jnp.array(da.sel(field=φs)), axes=(1, 2), norm="forward"),
        axes=(1, 2),
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
    a = (D*ksq + C + C/ksq + ν*ksq) / 2
    b = (D*ksq + C - C/ksq - ν*ksq) / 2
    g = jnp.square(b) + jnp.square(C / ky)
    h = jnp.sqrt(jnp.square(g) + jnp.square(C * κ / ky))
    j = jnp.sqrt((h+g) / 2)
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
            os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.8)
        ) / 1000
    else:
        import psutil
        return psutil.virtual_memory()[3] / 1000000000


def solve_save(
    diffeqsolve_kwargs, max_memory_will_use, save, split_callback=None
):
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
    saveat_fn: Callable[[float, Array, Any], Array] = (lambda t, y, args: y),
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
            When resuming a simulation: y0 = to_diffeqsolve(ys[-1])
            For converting to DataArray: to_dataarray(ys)
            where ys has an additional leading time dimension
        filename: the output file name, expected to be a .zarr file
        split_callback: callback when the simulation is splitted

    
    """
    file_path = Path(filename)
    resume = False

    nonlinear_solver = NewtonNonlinearSolver(rtol=rtol, atol=atol)
    solvers = {
        "Dopri5": Dopri5(),
        "Dopri8": Dopri8(),
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
            same = jnp.array([
                np.array_equal(v, da.attrs[k]) for (k, v) in attrs_.items()
                if k in da.attrs
            ]).all()

            if tf > da.attrs.get("tf", float("inf")) and len(da.coords["time"]
                                                             ) > 0 and same:
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
        y_dummy = saveat_fn(0, y0, None)[None]
        y_dummy = apply[1](y_dummy) if apply is not None else y_dummy
        xr.DataArray(
            data=jnp.array([]).reshape(-1, *y_dummy.shape[1:]),
            dims=dims,
            coords={"time": []} | coords,
            attrs=attrs_,
        ).to_zarr(file_path, mode="w") # yapf: disable

        t0 = 0
        # the number of frames saved is video_nframes, it is not proportional to tf
        time_linspace = jnp.linspace(t0, tf, video_nframes + 1)

    for k, v in attrs_.items():
        # print only short aparameters
        if isinstance(v, str) or not (hasattr(v, "__len__") and len(v) > 10):
            print(f"{k:<20}: {v}")

    max_memory_will_use = (
        y0.size * video_nframes * {
            jnp.dtype("float64"): 8, jnp.dtype("float32"): 4
        }[y0.dtype]  # number of bytes
        * 3  # it seems it will be buffered twice + 1 for safety
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
            "solver": SolverWrapTqdm(solvers[solver], tqdm_bar),
            "t0": t0,
            "t1": tf,
            "dt0": 1e-3,
            "y0": y0,
            "saveat": SaveAt(ts=time_linspace, fn=saveat_fn),
            "stepsize_controller": PIDController(atol=atol, rtol=rtol),
            "max_steps": None,
        }
        da = solve_save(
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
) -> tuple[sparse.BCOO, Optional[Array], Optional[Array]]:
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

    Return:
        op: differential BCOO sparse 2D matrix operator with shape
            (prod(grid.shape), prod(grid.shape))
        nz: indices where the constraint must be applied on the state, None if periodic
        rhs: values of the constraint of the state, None if periodic
    """
    params = locals()
    assert acc < min(grid.shape) / 2, f"acc={acc} is too big. The grid.shape is {grid.shape}."

    if isinstance(axis, int):
        op = FinDiff(axis, grid.step[axis], order, acc=acc)
    else:
        op = sum([FinDiff(i, grid.step[i], order, acc=acc) for i in axis])

    square_size = jnp.prod(jnp.array(grid.shape)).item()
    op_shape = (square_size, square_size)
    if bc_name == "periodic":
        pos = jnp.prod(jnp.array(grid.shape[:-1], dtype=int) + 1) * (acc-1)
        op = op.matrix(grid.shape)[pos].toarray()
        row = jnp.roll(jnp.array(op), -pos).squeeze()
        indices = jnp.nonzero(row)[0]
        data = jnp.tile(row[indices], square_size)
        ones = jnp.ones_like(indices)
        indices = jnp.concatenate(
            jax.vmap(
                fun=lambda i: jnp.
                stack(arrays=(ones * i, (indices+i) % square_size), axis=-1)
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

        # when using acc > 2 and a boundary condition, the matrix is strange at the edges
        # we fix it using acc=2 at the edges only
        if acc > 2:
            params["acc"] = 2
            op2, _, _ = diff_matrix(**params)
            op2 = scipy.sparse.coo_matrix(
                (op2.data, (op2.indices[:, 0], op2.indices[:, 1])),
                shape=op2.shape
            ).tolil()
            pos = jnp.arange(square_size).reshape(grid.shape)
            pos = pos[tuple([slice(acc // 2, -acc // 2)] * len(grid.shape))]
            op2[pos, :] = op[pos, :]
            op = op2

        if hasattr(op, "eliminate_zeros"):
            op.eliminate_zeros()

        op = sparse.BCOO.from_scipy_sparse(op)  # convert to jax sparse
    return op, nz, rhs


def append_total_1d(da):
    weights = {
        "Ωk_real": 2,
        "nk_real": 2,
        "Ωb": 1,
        "nb": 1,
    }
    das = [da] + [
        da.sel(field=fields).weighted(
            xr.DataArray([weights[f] for f in fields], coords={"field": fields})
        )
        .sum(dim="field")
        .expand_dims(dim={"field": [fields[0][0]]})
        for fields in [["Ωk_real", "Ωb"], ["nk_real", "nb"]]
    ] # yapf: disable
    return (
        xr.concat(das, dim="field").sel(
            field=["Ωk_real", "Ωb", "Ω", "nk_real", "nb", "n"]
        ).assign_coords({
            "field": [
                "Real($\Omega_k$)",
                "$\overline{\Omega}$",
                "Ω",
                "Real($n_k$)",
                "$\overline{n}$",
                "n"
            ]
        })
    )