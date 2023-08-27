from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.ticker
import xarray as xr
import tqdm.auto
import numpy as np

import jax
import jax.numpy as jnp
import jax_cfd.base.grids as grids

from hasegawa_wakatani.utils import open_with_vorticity, gridmesh_from_da, rfft_mesh, append_total_1d


def visualization_2d(filename):
    plot_metrics(filename)
    plot_time_frames_2d(filename)
    plot_spectrum(filename)
    return animation_2d(filename)


def animation_2d(x, fields=["Ω", "n"]):
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


def plot_time_frames_2d(filename, frames=[0, 0.25, 0.5, 1]):
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
        grid, (kx, ky) = gridmesh_from_da(da)
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


def last_1d_to_2d(filename):
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
        file_path.with_name(f"{file_path.stem}_1d_to_2d.pdf"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )


def plot_history_1d(filename):
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


def plot_pspectral_1d(filename):
    file_path = Path(filename)
    plot_history_1d(file_path)
    plot_components_1d(file_path.with_stem(f"{file_path.stem}_decomposed"))
    last_1d_to_2d(file_path)


def plot_components_1d(filename, tlim=None, xlim=None):
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

    file_path = Path(filename)

    if tlim is not None:
        da = da.sel(time=slice(*tlim))
    if xlim is not None:
        da = da.sel(x=slice(*xlim))

    tf = da.coords["time"].values[-1]
    domain = da.attrs["domain"]
    boundary = da.attrs.get("boundary", "periodic").split()
    x = da.coords["x"].values

    g = append_total_1d(da).plot.pcolormesh(
        x="x",
        y="time",
        col="field",
        col_wrap=3,
        rasterized=True,
        vmax=abs(da.sel(field="Ωb")).max(),
        cmap="seismic"
    )
    if boundary[0] == "force":
        ax = g.axs.flat[-2]
        forcing = (
            jnp.exp(-jnp.square((x - 0.1*domain) / (domain/10)))
            * float(boundary[1])
        )
        forcing -= forcing[::-1]  # show the well
        forcing -= forcing.min()
        forcing *= tf / forcing.max()
        ax.plot(x, forcing, label="source")
        ax.legend(fontsize="small")

    ax = g.axs.flat[-1]
    n_profil = (
        g.data.isel(time=-1).sel(field="n") + da.attrs["κ"] * (domain-x)
    )
    n_profil -= n_profil.min()
    n_profil *= tf / n_profil.max()
    ax.plot(x, n_profil, label="profile")
    ax.legend(fontsize="small", loc="lower left")

    g.fig.savefig(
        file_path.with_suffix(".jpg"),
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )
    return g


def pcolor_compare_1d(da, directory):
    g = append_total_1d(da).plot.pcolormesh(
        x="x",
        y="time",
        row="method",
        col="field",
        rasterized=True,
        vmax=abs(da.sel(field="Ωb")).max(),
        cmap="seismic"
    )
    g.fig.savefig(
        Path(directory) / "compare_1d_pcolor.pdf",
        dpi=100,
        bbox_inches="tight",
        pad_inches=0
    )


def plot_profiles_compare_1d(da, directory, ts=None):
    if ts is None:
        ts = jnp.linspace(0, da.attrs["tf"], 5)[1:]
    g = append_total_1d(da).sel(
        time=ts, method="nearest"
    ).drop_sel(field=["Ω", "n"]).plot.line(
        x="x",
        hue="method",
        row="field",
        col="time",
        sharey=False,
        linestyle=(0, (5, 1)),
    )
    g.fig.savefig(
        Path(directory) / "compare_1d_profiles.pdf",
        bbox_inches="tight",
        pad_inches=0
    )


def pcolor_compare_1d_params(da, directory):
    melted = append_total_1d(da).drop_sel(field=["Ω", "n"])
    das = [ # melt method into field
        melted.sel({"method": m}).assign_coords({
            "field": [f+f" {m}" for f in melted.coords["field"].values]
        }) for m in ("pspectral", "findiff")
    ]
    melted = xr.concat(das, dim="field")

    fmt = matplotlib.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    g = xr.plot.FacetGrid(melted, col="field", row="param", size=2)

    for i, ax in enumerate(g.axs):
        melted_param = melted.isel(param=i)
        vmax = abs(melted_param.isel(field=1)).max()
        for j in range(len(melted.coords["field"])):
            img = melted_param.isel(field=j).plot.pcolormesh(
                x="x",
                y="time",
                rasterized=True,
                vmax=vmax,
                cmap="seismic",
                ax=ax[j],
                add_colorbar=False
            )
            if j == 1:
                cbar = g.fig.colorbar(
                    img,
                    ax=ax,
                    cax=ax[-1].inset_axes([1.2, 0, 0.05, 1]),
                    format=fmt,
                )
            ax[j].set(title=None, xlabel=None, ylabel=None)

    g.set_titles(template='{value}')
    g.set_xlabels(label="x")
    g.set_ylabels(label="time")
    g.fig.tight_layout()
    g.fig.savefig(
        Path(directory) / "compare_1d_params_pcolor.pdf",
        dpi=100,
        bbox_inches="tight",
        pad_inches=0
    )

def plot_metrics(filename):
    da = open_with_vorticity(filename)
    dim = {"x", "y", "z"}.intersection(set(da.dims))

    grid, ks = gridmesh_from_da(da)
    k2 = jnp.square(ks).sum(0).at[0,0].set(1)
    axes = np.array(range(1, len(dim)+1))
    fft_kwargs = dict(axes=axes, norm="forward")
    φk = - jnp.fft.rfft2(jnp.array(da.sel(field="Ω")), axes=axes, norm="forward") / k2
    v2 = jnp.square(jnp.fft.irfft2(1j*ks[:, None]*φk, axes=axes+1, norm="forward")).sum(0)


    if "n" in da.coords["field"]:
        enstrophy = (da.sel(field="n") - da.sel(field="Ω"))**2
        enstrophy = enstrophy.mean(dim=dim) / 2
        
        energy = jnp.square(jnp.array(da.sel(field="n"))) + v2
        energy = energy.mean(axes) / 2

        data = [energy, enstrophy]
        metric_names = ["energy", "enstrophy"]
    else: # hasegawa-mima
        data = [v2.mean(axes)]
        metric_names = ["energy"]

    
    xr.DataArray(
        data=data,
        dims=["metric", "time"],
        coords={"metric": metric_names, "time": da.coords["time"]}
    ).plot(x="time", hue="metric")
    
    plt.savefig(
        Path(filename).with_name("metrics.pdf"),
        bbox_inches="tight",
        pad_inches=0
    )

