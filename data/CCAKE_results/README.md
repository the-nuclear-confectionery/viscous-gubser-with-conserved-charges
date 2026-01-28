# CCAKE Results for $μ_B/T = 0.3$

This directory contains **numerical hydrodynamic outputs** for viscous Gubser flow with a conserved charge, computed using CCAKE at a fixed ratio:

$$ \mu_B / T = 0.3 $$

## Files

### `freeze_out.dat`
Freeze-out hypersurface data extracted from the hydrodynamic evolution.

Each row corresponds to a freeze-out surface element and contains:
- Space-time coordinates
- Flow four-velocity components
- Thermodynamic quantities evaluated at freeze-out
- Local chemical potential and temperature information

These data can be used for:
- Particle production calculations
- Validation of freeze-out prescriptions
- Comparison with analytic freeze-out surfaces

### `system_state_*.dat`
Hydrodynamic field snapshots at successive proper times.

- `system_state_0.dat` corresponds to the earliest stored time
- Higher indices represent later evolution times

Each file contains:
- Grid indices and spatial coordinates
- Energy density
- Charge density
- Flow velocity components
- Viscous stress tensor components
- Equation-of-state metadata

## Notes

- Grid spacing and domain size are fixed across all files.
- The evolution assumes conformal symmetry consistent with Gubser flow.
