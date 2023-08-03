from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import xarray as xr
import tqdm.auto

import jax
import jax.numpy as jnp
import jax_cfd.base.grids as grids

from utils import open_with_vorticity, gridmesh_from_da, rfft_mesh


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


def plot_history_1D(filename):
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


def plot_spectral_1D(filename):
    file_path = Path(filename)
    plot_history_1D(file_path)
    plot_components_1D(file_path.with_stem(f"{file_path.stem}_decomposed"))
    last_1D_to_2D(file_path)


def plot_components_1D(filename, tlim=None, xlim=None):
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

    if tlim is not None:
        da = da.sel(time=slice(*tlim))
    if xlim is not None:
        da = da.sel(x=slice(*xlim))

    da = da.sel(field=["Ωk_real", "Ωb", "nk_real", "nb"]
                ).transpose("field", ...)
    Ωk, Ωb, nk, nb = jnp.array(da)

    domain = da.attrs["domain"]
    grid_size = da.attrs["grid_size"]
    tf = da.attrs["tf"]
    κ = da.attrs["κ"]
    boundary = da.attrs.get("boundary", "periodic").split()
    x = da.coords["x"].values

    nrows = 2
    ncolumns = 3
    fig, axes = plt.subplots(nrows, ncolumns, figsize=(7, 4))

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
        file_path.with_name(f"{file_path.stem}.pdf"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    return Ωk, nk, Ωb, nb
