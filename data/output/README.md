# Analytic Viscous Gubser Solutions for $μ_B/T = 0.3$

This directory contains **analytic reference solutions** for viscous Gubser flow with a conserved charge at fixed:

$$ \mu_B / T = 0.3 $$

These files are intended for direct comparison with numerical hydrodynamic simulations.

## Files

### `VGCC_initial_condition.dat`
Initial condition for the viscous Gubser flow with conserved charge.

Each row corresponds to a transverse grid point and contains:
- Transverse coordinates
- Initial energy density
- Charge density
- Flow velocity components
- Viscous corrections

### `VGCC_tau=*.dat`
Analytic solution evaluated at fixed proper times ($ \tau $).

- The filename encodes the proper time (e.g. `tau=1.30`)
- Each file uses the same grid and column ordering as the initial condition

These files allow:
- Time-resolved validation of hydrodynamic solvers
- Quantitative error analysis
- Visualization of analytic flow profiles

## Notes

- Grid definitions are consistent across all time slices.
