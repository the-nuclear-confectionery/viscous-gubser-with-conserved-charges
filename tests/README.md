# EoS1 validation

Run from the repository root in the project's Python environment:

```bash
python3 -m unittest discover -s tests -v
```

Uses the existing NumPy and SciPy dependencies; no additional test framework.
The four tests cover the reduced conservation equations, the zero-charge
limit, the exact ideal evolution, and baryon conservation along viscous
trajectories for initial mu_B/T = 0, 10, and 15, with -3 <= rho <= 3.
Integration uses rtol=1e-10 and atol=1e-12. The conserved baryon quantity
is checked with rtol=1e-8 and atol=1e-10.
