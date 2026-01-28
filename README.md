# Viscous Gubser flow with conserved charges (VGCC)
A Python implementation of semi-analytical solutions for Gubser flow with conserved charges (baryon number $B$, strangeness $S$, and electric charge $Q$) for benchmarking relativistic fluid simulations.

[![License: GPL v3](https://img.shields.io/badge/License-University_of_Illinois/NCSA_Open_Source-blue.svg)](https://spdx.org/licenses/NCSA.html)
[![DOI](https://zenodo.org/badge/954478331.svg)](https://doi.org/10.5281/zenodo.18397955)

<p align="center">
<img src="utils/VGCC_evolution.png" alt="logo" width="700"/>
</p>


## Overview

This repository provides tools to generate semi-analytical solutions for Gubser flow with conserved charges. These solutions can serve as benchmarks for relativistic viscous hydrodynamics codes.

For theoretical details, see:
- K. Ingles, J. Salinas San Martín, W. Serenone, and J. Noronha-Hostler,
 _Viscous Gubser flow with conserved charges to benchmark fluid simulations_, [Phys. Rev. C **\<VOLUME NO.\>**, \<IDENTIFIER\> (2025)](https://doi.org/10.1103/v334-d32w), [arXiv:2503.20021 [nucl-th]](https://arxiv.org/abs/2503.20021).

## Features
- Semi-analytical solutions for Gubser flow with BSQ conserved charges
- Compatible with the [CCAKE](https://github.com/the-nuclear-confectionery/CCAKE) hydrodynamics code; see details [here](https://inspirehep.net/literature/2787415)
- Support for multiple equations of state
- Customizable parameters via YAML configuration

## Installation
1. Clone the repository:
```terminal
git clone git@github.com:the-nuclear-confectionery/viscous-gubser-with-conserved-charges.git
cd viscous-gubser-with-conserved-charges
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

The script provides the flexibility of changing and implementing different equations of state (EoS); please see [PRC **\<VOLUME NO.\>**, \<IDENTIFIER\> (2025)](https://doi.org/10.1103/v334-d32w), [arXiv:2503.20021 [nucl-th]](https://arxiv.org/abs/2503.20021) for details.

Current options are `EoS1` (massless QGP) and `EoS2` (conformal plasma).
The details of the equations of state included by default and instructions on how to implement a custom EoS are given in the [eos](eos) folder.

### Configuration

The parameters used for solving the hydrodynamics equations of viscous Gubser flow with conserved charges are defined in [config.yaml](config.yaml).

## Plotting tools

Visualization scripts used to generate the figures of [PRC **\<VOLUME NO.>**, \<IDENTIFIER> (2025)](https://doi.org/10.1103/v334-d32w), [arXiv:2503.20021 [nucl-th]](https://arxiv.org/abs/2503.20021) are provided on the [plots](plots) directory.

## Citation

If you use this code, please cite:
```bibtex
@article{Ingles:2025yrv,
    author = "Ingles, Kevin and Salinas San Mart{\'\i}n, Jordi and Serenone, Willian and Noronha-Hostler, Jacquelyn",
    title = "{Viscous Gubser flow with conserved charges to benchmark fluid simulations}",
    eprint = "2503.20021",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    month = "3",
    year = "2025"
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Additional equations of state can be included by creating the corresponding files inside the [eos](eos) and [eom](eom) directories.

## Authorship

The content of this repository was developed by J. Salinas San Martín and K. Ingles at the University of Illinois.