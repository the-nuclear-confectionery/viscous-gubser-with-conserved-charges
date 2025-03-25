# Viscous Gubser flow with conserved charges (VGCC)
A Python implementation of semi-analytical solutions for Gubser flow with conserved charges (baryon number $B$, strangeness $S$, and electric charge $Q$) for benchmarking relativistic fluid simulations.

[![License: GPL v3](https://img.shields.io/badge/License-University_of_Illinois/NCSA_Open_Source-blue.svg)](https://spdx.org/licenses/NCSA.html)
<p align="center">
<img src="utils/VGCC_evolution.png" alt="logo" width="500"/>
</p>

## Overview

This repository provides tools to generate semi-analytical solutions for Gubser flow with conserved charges. These solutions can serve as benchmarks for relativistic viscous hydrodynamics codes.

For theoretical details, see:
- K. Ingles, J. Salinas San Martín, W. Serenone, and J. Noronha-Hostler,
 _Viscous Gubser flow with conserved charges to benchmark fluid simulations_, [Phys. Rev. D **\<VOLUME NO.\>**, \<IDENTIFIER\> (2025)](), [arXiv:2502:XXXX [nucl-th]]().

## Features
- Semi-analytical solutions for Gubser flow with BSQ conserved charges
- Compatible with the [CCAKE](https://github.com/the-nuclear-confectionery/CCAKE) hydrodynamics code; see details [here](https://inspirehep.net/literature/2787415)
- Support for multiple equations of state
- Customizable parameters via YAML configuration

## Installation
1. Clone the repository:
```terminal
git clone https://github.com/<username>/viscous_gubser_with_conserved_charges.git
cd viscous_gubser_with_conserved_charges
```

2. Install dependencies:
```terminal
pip install numpy matplotlib scipy pandas tqdm
```

## Usage

To generate the solutions of Gubser flow with conserved charges, execute in the terminal the following command:
```terminal
python vgcc.py [-h] [--config <FILE>] [--mode <{initial_condition,evolution}>] [--eos <{EoS1,EoS2}>]  [--debug]
```

where the default arguments are:
```terminal
mode: initial_condition
eos: EoS2
config: config.yaml
```

### Available Modes

1. `initial_condition`: Generates initial conditions for hydrodynamic simulations.
2. `evolution`: Solves hydrodynamic evolution equations until $\tau_f$ (defined on [config.yaml](config.yaml)).


### Equation of State

The script provides the flexibility of changing and implementing different equations of state (EoS); please see [PRD **\<VOLUME NO.\>**, \<IDENTIFIER\> (2025)](), [arXiv:2502:XXXX [nucl-th]]() for details.

Current options are `EoS1` (massless QGP) and `EoS2` (conformal plasma).
The details of the equations of state included by default and instructions on how to implement a custom EoS are given in the [eos folder]().

### Configuration

The parameters used for solving the hydrodynamics equations of viscous Gubser flow with conserved charges are defined in [config.yaml](config.yaml).

## Plotting tools

Visualization scripts used to generate the figures of [PRD **\<VOLUME NO.>**, \<IDENTIFIER> (2025)](), [arXiv:2502:XXXX [nucl-th]]() are provided on the [plots](plots) directory.

## Citation

If you use this code, please cite:
```bibtex
@article{ingles2025viscous,
  title={Viscous Gubser flow with conserved charges to benchmark fluid simulations},
  author={Ingles, K. and {Salinas San Martín}, J. and Serenone, W. and Noronha-Hostler, J.},
  journal={Phys. Rev. D},
  volume={},
  pages={},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. An example on how to create your own script is included in [CONTRIBUTING](). A template EoS script is also included inside the [eos](eos) directory.

## Authorship

The content of this repository was developed by J. Salinas San Martín and K. Ingles at the University of Illinois.