# Viscous Gubser flow with conserved charges (VGCC)

A Python implementation of semi-analytical solutions for Gubser flow with conserved charges (baryon number $B$, strangeness $S$, and electric charge $Q$) for benchmarking relativistic fluid simulations.

## Overview

This repository provides tools to generate semi-analytical solutions for Gubser flow with conserved charges. These solutions can serve as benchmarks for relativistic viscous hydrodynamics codes.

For theoretical details, see:
- K. Ingles, W. Serenone, J. Salinas San Martín, and J. Noronha-Hostler,
 _Viscous Gubser flow with conserved charges to benchmark fluid simulations_, [Phys. Rev. D **\<VOLUME NO.\>**, \<IDENTIFIER\> (2025)](), [arXiv:2502:XXXX [nucl-th]]().

## Features
- Semi-analytical solutions for Gubser flow with BSQ conserved charges
- Compatible with the [CCAKE](https://github.com/the-nuclear-confectionery/CCAKE) hydrodynamics code; see details [here](https://inspirehep.net/literature/2787415)
- Support for multiple equations of state
- Customizable parameters via YAML configuration

## Installation
1. Clone the repository:
```bash
git clone https://github.com/<username>/viscous_gubser_with_conserved_charges.git
cd viscous_gubser_with_conserved_charges
```

2. Install dependencies:
```bash
pip install numpy matplotlib scipy pandas tqdm
```

## Usage

To generate the solutions of Gubser flow with conserved charges, execute in the terminal the following command:
```bash
python vgcc.py --mode <mode> --eos <eos>
```

where the default arguments are:
```terminal
mode: initial_conditions
eos: conformal_plasma
```

### Available Modes

1. `semi-analytical`: Solves hydrodynamic evolution equations until $\tau_f$ (defined on [config.yaml](config.yaml)).
2. `initial_conditions`: Generates initial conditions for hydrodynamic simulations.


### Equation of State

The script provides the flexibility of changing and implementing different equations of state (EoS); please see [PRD **\<VOLUME NO.\>**, \<IDENTIFIER\> (2025)](), [arXiv:2502:XXXX [nucl-th]]() for details.

Current options are `EoS1` (conformal plasma) and `EoS2` (massless QGP).
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
  author={Ingles, K. and Serenone, W. and {Salinas San Martín}, J. and Noronha-Hostler, J.},
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