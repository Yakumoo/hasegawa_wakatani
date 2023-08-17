# Hasegawa-Wakatani turbulence simulations with [Jax](https://jax.readthedocs.io)
## For more details, check out the [wiki](https://github.com/Yakumoo/hasegawa_wakatani/wiki)
https://github.com/Yakumoo/hasegawa_wakatani/assets/26108275/dfd353bb-03be-4cfd-a015-f0cbbfbf9cbd

# Basic usage
1. Create your `.yaml` file with the model and the parameters. For example:
```yaml
hasegawa_wakatani_pspectral_2d:
  grid_size: 682
  tf: 100
  video_length: 20
```
2. Run the python script:
```python
python main.py path_to_yaml_file.yaml
```
It will create a [`.zarr`](https://zarr.readthedocs.io) file and plot the simulation in the same folder as the yaml file

## To launch a command
For example, this command will create a `workspace/compare_1d` directory where the simulation and the plots are saved:
```python
python main.py compare_1d
```

# Available models
- `hasegawa_mima_pspectral_2d`: Hasegawa-Mima equation with pseudo-spectral method in 2D with forcing/injection
- `hasegawa_wakatani_pspectral_2d`: Pseudo-spectral method in 2D (periodic)
- `hasegawa_wakatani_pspectral_1d`: Pseudo-spectral method with a single poloïdal mode, it uses `hasegawa_wakatani_pspectral_2d`
- `hasegawa_wakatani_findiff_2d`: Finite-difference method in 2D (periodic, Dirichlet, Neumann)
- `hasegawa_wakatani_findiff_1d`: Finite-difference method with a single poloïdal mode (and forcing)

# Available commands
- `compare_1d`: Run the pseudo-spectral and the finite-difference simulations using the same parameters then plot
- `compare_1d_params`: Run `compare_1d` with differents values of `C` and `κ`
