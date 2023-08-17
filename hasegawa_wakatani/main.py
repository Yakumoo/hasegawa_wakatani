#!/usr/bin/env python3

# yapf main.py -i --style='{DEDENT_CLOSING_BRACKETS = true, COALESCE_BRACKETS = true, ARITHMETIC_PRECEDENCE_INDICATION = true, SPLIT_ALL_TOP_LEVEL_COMMA_SEPARATED_VALUES = true, SPLIT_BEFORE_ARITHMETIC_OPERATOR = true, SPLIT_BEFORE_DOT = true, SPLIT_BEFORE_EXPRESSION_AFTER_OPENING_PAREN = true, SPLIT_BEFORE_FIRST_ARGUMENT = true}'

from inspect import signature
from pathlib import Path
from argparse import ArgumentParser

import yaml
import xarray as xr

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


def main():

    parser = ArgumentParser(
        prog="hasegawa_wakatani_simulation",
        description=
        """Script for running Hasegawa-Wakatani simulations, saving and/or plotting. The implemented models are :
        - hasegawa_mima_pspectral_2d  
        - hasegawa_wakatani_pspectral_{1,2}d
        - hasegawa_wakatani_findiff_{1,2}d
        """
    )
    parser.add_argument(
        "filename_or_command",
        type=str,
        help="""
        The input filename (.yaml or .zarr) or the command to execute. Possible commands are:
        - compare_1d: executes the single poloidal mode simulations with pseudo-spectral and finite difference methods with the same parameters
        """
    )
    parser.add_argument("--tf", type=float, help="Final simulation time")
    parser.add_argument("--grid_size", type=int, help="Simulation resolution")
    parser.add_argument("--domain", type=float, help="Simulation domain")
    parser.add_argument("--C", type=float, help="Adiabatic parameter")
    parser.add_argument("--κ", type=float, help="Grandient density")
    parser.add_argument(
        "--tol",
        type=float,
        help="Tolerance for the time stepping, atol=rtol=tol"
    )
    parser.add_argument(
        "--video_length", type=float, help="Video length of the output"
    )
    parser.add_argument(
        "--video_fps", type=float, help="Video fps of the output"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed of the simulation (for the initial state)"
    )
    parser.add_argument(
        "--acc", type=int, help="Accuracy of the finite difference method"
    )
    parser.add_argument(
        "--boundary",
        type=str,
        help="Boundary conditions (finite difference method only)"
    )
    parser.add_argument("--νx", type=float, help="Viscosity x direction")
    parser.add_argument("--νy", type=float, help="Viscosity y direction")
    parser.add_argument("--νz", type=float, help="Viscosity zonal flow")
    parser.add_argument("--ν", type=float, help="Viscosity x and y direction")
    parser.add_argument("--Dx", type=float, help="Diffusion x direction")
    parser.add_argument("--Dy", type=float, help="Diffusion x direction")
    parser.add_argument("--Dz", type=float, help="Diffusion zonal flow")
    parser.add_argument("--D", type=float, help="Diffusion x and y direction")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force using CPU when jax is configured with GPU",
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

    from models import (
        hasegawa_mima_pspectral_2d,
        hasegawa_wakatani_pspectral_2d,
        hasegawa_wakatani_pspectral_1d,
        hasegawa_wakatani_findiff_1d,
        hasegawa_wakatani_findiff_2d
    )
    from plots import (visualization_2d, plot_pspectral_1d, plot_components_1d)
    from commands import (compare_1d, compare_1d_params)

    schemes = {
        "hasegawa_mima_pspectral_2d":
        (hasegawa_mima_pspectral_2d, visualization_2d),
        "hasegawa_wakatani_pspectral_2d":
        (hasegawa_wakatani_pspectral_2d, visualization_2d),
        "hasegawa_wakatani_pspectral_1d":
        (hasegawa_wakatani_pspectral_1d, plot_pspectral_1d),
        "hasegawa_wakatani_findiff_1d":
        (hasegawa_wakatani_findiff_1d, plot_components_1d),
        "hasegawa_wakatani_findiff_2d":
        (hasegawa_wakatani_findiff_2d, visualization_2d),
    }
    commands = {
        "compare_1d": compare_1d, "compare_1d_params": compare_1d_params
    }

    simulation_kwargs = {}
    for k in set(vars(args).keys()) - {'cpu', 'eager', 'filename_or_command'}:
        v = getattr(args, k, None)
        if v is not None:
            simulation_kwargs[k] = v

    # commands
    if args.filename_or_command in commands:
        return commands[args.filename_or_command](**simulation_kwargs)

    # filename
    p = Path(args.filename_or_command)
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
        } | yaml_data[model] | simulation_kwargs

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
        simulation_kwargs = {
            "filename": p / "HasegawaWakataniSpectral2D.zarr",
        } | simulation_kwargs
        hasegawa_wakatani_pspectral_2d(**simulation_kwargs)
        visualization_2d(simulation_kwargs["filename"])
    else:
        print(
            f"filename_or_command={args.filename_or_command} argument is invalid, see --help for more details."
        )


if __name__ == "__main__":
    main()
