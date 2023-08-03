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
        - hasegawa_mima_spectral_2D
        - hasegawa_wakatani_spectral_{1,2}D
        - hasegawa_wakatani_finite_difference_{1,2}D
        """
    )
    parser.add_argument("filename")
    parser.add_argument("--tf", type=float, help="Final simulation time")
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
        hasegawa_mima_spectral_2D,
        hasegawa_wakatani_spectral_2D,
        hasegawa_wakatani_spectral_1D,
        hasegawa_wakatani_finite_difference_1D,
        hasegawa_wakatani_finite_difference_2D
    )
    from plot import (visualization_2D, plot_spectral_1D, plot_components_1D)

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
