# Comparison METROPOLIS1 vs METROPOLIS2

This repository contains Python scripts to compare simulations from both METROPOLIS (first of the
name) and METROPOLIS2.
The repository is intended to compare the calibrated Île-de-France simulation from Saifuzzaman et
al. (2012) but it can be useful to compare any METROPOLIS1 simulation.
The results from these scripts are presented in Javaudin and de Palma (2024).

## References

Saifuzzaman, M., de Palma, A., & Motamedi, K. (2012). Calibration of METROPOLIS for Ile-de-France.
_Working Paper_.

de Palma, A., Marchal, F., & Nesterov, Y. (1997). METROPOLIS: Modular system for dynamic traffic
simulation. _Transportation Research Record, 1607_(1), 178-184.

Javaudin, L., & de Palma, A. (2024). METROPOLIS2: A Multi-Modal Agent-Based Transport Simulator.
_THEMA Working Paper_.

## Running METROPOLIS1

METROPOLIS1 can be run directly from the web interface.

The Île-de-France simulation used in Javaudin and de Palma (2024) is
[this one](https://metropolis.sauder.ubc.ca/614).
The simulation run corresponding to the results is
[this one](https://metropolis.sauder.ubc.ca/614/run/12155).
Note however that a modified version of METROPOLIS1 was used to run the simulator with additional
output.
This modified version is not available on the web interface.
Some script parts only work with this modified version.

## Running METROPOLIS2

The Python scripts are writing METROPOLIS2's input and reading METROPOLIS2's output but they are not
running the simulator.
METROPOLIS2 can be run from a command line interface with the input files as arguments.
Version 0.8.0 of METROPOLIS2 was used for the results presented in Javaudin and de Palma (2024).

Current scripts are compatible with METROPOLIS2 version 1.0.0.

## Python code

The Python packages required to run the code are listed in `requirements.txt`.

The Python code is located in the `python/` directory.
Configuration variables are defined as global variables at the beginning of the scripts.

- `mpl_utils.py`: Functions used to plot graphs with `matplotlib`.

- `metropolis1_to_metropolis2.py`: Convert a simulation from METROPOLIS1's web interface in
  METROPOLIS2's input files.
- `compare_metropolis1_and_2.py`: Various functions to print statistics and generate graphs
  comparing the output of METROPOLIS1 and METROPOLIS2. Note that this requires the modified version
  of METROPOLIS1.
- `create_network.py`: Create a GeoParquet of the METROPOLIS1's road network.
- `plot_speed_density_function.py`: Simple script to plot an example bottleneck speed-density
  function (used for the graph in Javaudin and de Palma, 2024).
