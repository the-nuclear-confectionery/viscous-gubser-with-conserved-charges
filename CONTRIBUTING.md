# Extensions

To write your own script to run the code, follow the following example:

```python
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from numpy import concatenate
from numpy import linspace
from numpy import array
from system_massless_qgp import MasslessQGP
from variable_conversions import milne_T
from variable_conversions import milne_mu

# Create an instance of the system you want to solve: MasslessQGP or ConformalPlasma
system = MasslessQGP()

# de Sitter time steps
rhos_backward = linspace(-10, 0, 1000)[::-1]
rhos_forward = linspace(0, 10, 1000)

# Initial conditions: (temperature, chemcical potential, shear pressure)
y0s = array([1.0, 1.0, 0.0])

# Obtain the numerical solutions in de Sitter space (that's how the equations are implemented)
# Here, I assume the initial condition is given at rho = 0
soln_1 = odeint(system.eom.for_scipy, y0s, rhos_forward)    # Forward in de Sitter time 
soln_2 = odeint(system.eom.for_scipy, y0s, rhos_backword)   # Backward in de Sitter time

# Glue forward and backward evolutions together
t_hat = concatenate((soln_1[:, 0][::-1], soln_2[:, 0]))       
mu_hat = concatenate((soln_1[:, 1][::-1], soln_2[:, 1]))
pi_bar_hat = concatenate((soln_1[:, 2][::-1], soln_2[:, 2]))
rhos = concatenate((rhos_1[::-1], rhos_2))

# Create interpolating functions from the solutions
t_interp = interp1d(rhos, t_hat)
mu_interp = interp1d(rhos, mu_hat)
pi_interp = interp1d(rhos, pi_bar_hat)

# Convert temperature and chemcial potential from de Sitter values to Milne values
t_evol = milne_T(tau, xs, 1, t_interp)
mu_evol = milne_mu(tau, xs, 1, mu_interp)

# Milne coordinates, for example, for plotting
tau = 1.0    # fm/c
xs = linspace(-6, 6, 200)

# Convert thermodynamic quantities from de Sitter values to Milne values
e_evol = system.milne_energy(tau, xs, 0.0, 1.0, t_interp, mu_interp)
n_evol = system.milne_number(tau, xs, 0.0, 1.0, t_interp, mu_interp)
s_evol = system.milne_entropy(tau, xs, 0.0, 1.0, t_interp, mu_interp)
pimunu_evol = system.milne_pi(
    tau, xs, 0.0, 1, t_interp, mu_interp, pi_interp, nonzero_xy=True,)
```