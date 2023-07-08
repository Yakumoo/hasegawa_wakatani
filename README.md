# Hasegawa-Wakatani turbulence simulations with [Jax](https://jax.readthedocs.io)
https://github.com/Yakumoo/hasegawa_wakatani/assets/26108275/dfd353bb-03be-4cfd-a015-f0cbbfbf9cbd



# Usage
1. Create your `.yaml` file with the model and the parameters. For example:
```yaml
hasegawa_wakatani_spectral_2D:
  grid_size: 1024
  tf: 100
  video_length: 20
```
2. Run the python script:
```python
python main.py path_to_yaml_file.yaml
```
It will create a `.nc` file and plot the simulation in the same folder as the yaml file

## To plot from an existing `.nc` file
```python
python main.py path_to_nc_file.nc
```

## To resume a simulation
Increase the parameter `tf` in the yaml file and call again with the yaml file.
Otherwise use the command:
```python
python main.py path_to_nc_file.nc --tf 1000
```

# Available models
- `hasegawa_wakatani_spectral_2D`: Pseudo-spectral method in 2D (periodic)
- `hasegawa_wakatani_spectral_1D`: Pseudo-spectral method with a single poloïdal mode, it uses `hasegawa_wakatani_spectral_2D`
- `hasegawa_wakatani_finite_difference_2D`: Finite-difference method in 2D (periodic, Dirichlet, Neumann)
- `hasegawa_wakatani_finite_difference_1D`: Finite-difference method with a single poloïdal mode (and forcing)


# Equations
## Cartesian space
$$\frac{\partial\Delta^2{\phi}}{\partial t} = -[\phi, \Delta^2\phi] + C (\tilde{\phi} - \tilde n) + \nu\Delta^4 \tilde{\phi} - \nu_z\Delta^2 \overline{\phi} \qquad \frac{\partial n}{\partial t} = -[\phi, n] + C (\tilde{\phi} - \tilde n) + D\Delta \tilde{n} - D_z \overline{n} - \kappa \frac{\partial\phi}{\partial y}$$
## Fourier space
$$\frac{\partial \phi_k}{\partial t} = - \frac{C_k}{k^2} (\phi_k -n_k) - ν_k\phi_k + \frac{1}{k^2} [\phi, \Delta{\phi}]_k \qquad \frac{\partial n_k}{\partial t} = C_k (\phi_k -n_k) - i \kappa k_y \phi_k -D_kn_k - [\phi,n]_k$$
## Single poloïdal mode
$$\frac{\partial \Omega_k}{\partial t} = ik_y \left(\phi_k\frac{\partial \overline{\Omega}}{\partial x} - \Omega_k \frac{\partial \overline{\phi}}{\partial x} \right) + C\left(\phi_k - n_k\right) + ν_x \frac{\partial ^2\Omega_k}{\partial x^2} - ν_y k^2 \Omega_k$$

$$\frac{\partial n_k}{\partial t} = ik_y \left( \left(\frac{\partial \overline{n}}{\partial x} - \kappa\right)\phi_k - \Omega_k \frac{\partial \overline{\phi}}{\partial x}\right) + C\left(\phi_k - n_k\right) + D_x \frac{\partial ^2n_k}{\partial x^2} - D_y k^2 n_k$$

$$\frac{\partial \overline{\Omega}}{\partial t} = 2 k_y \frac{\partial }{\partial x} \mathrm{Im}\{\phi_k^* \Omega_k\} - \nu_z\overline{\Omega} \qquad \frac{\partial \overline{n}}{\partial t} = 2 k_y \frac{\partial }{\partial x} \mathrm{Im}\{\phi_k^* n_k\} + D_z\overline{n}$$

# Credits
- [Ozgur Gurcan](https://gurcani.github.io)'s [repo](https://github.com/gurcani/hwak_cuda)
- [Pierre Morel](https://www.lpp.polytechnique.fr/-Pierre-Morel)
