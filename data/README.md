# Numerical Data for Viscous Gubser Flow with Conserved Charges

This directory contains **numerical output data** used to benchmark relativistic viscous hydrodynamic simulations with conserved charges against semi-analytic Gubser-flow solutions.

The data provided here are intended for **reproducibility, validation, and independent analysis**.

## Directory structure

- `CCAKE_results/`
  - Numerical results obtained using the **CCAKE** hydrodynamic solver.
  - These files represent fully evolved viscous hydrodynamic fields and freeze-out hypersurfaces.
- `output/`
  - This directory contains **semi-analytic solutions** of viscous Gubser flow with conserved charges.
  - These outputs serve as **reference benchmarks** against which numerical hydrodynamic simulations (e.g. CCAKE results) can be validated.
  - The data here are generated independently of any specific hydrodynamic code.

## Physical context

The data correspond to solutions of **viscous Gubser flow with conserved charges**, as described in the associated publication. They are suitable for:
- Code-to-code benchmarking
- Validation of numerical hydrodynamic solvers
- Comparison with semi-analytic viscous solutions

## Citation

If you use this dataset, please cite the Zenodo record associated with this archive, as well as the corresponding journal article describing the physical model.
